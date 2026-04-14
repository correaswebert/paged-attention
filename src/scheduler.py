import queue
import time

from cache_manager import KVCacheManager
from model import PagedAttentionModel
from processor import Processor


class InferenceRequest:
    """Tracks the state of a single request during continuous batching."""

    def __init__(
        self, prompt_tokens: list[int], out_queue: queue.Queue, max_new_tokens: int = 50
    ):
        self.prompt_tokens = prompt_tokens
        self.out_queue = out_queue
        self.max_new_tokens = max_new_tokens

        self.block_table = []
        self.generated_tokens = []
        self.pos = 0

    def get_token_to_feed(self) -> int:
        """Returns the prompt token if in prefill, or the last generated token if in decode."""
        if self.pos < len(self.prompt_tokens):
            return self.prompt_tokens[self.pos]
        return self.generated_tokens[-1]


class Scheduler:
    """Continuous Batching Scheduler for Paged Attention."""

    def __init__(
        self,
        model: PagedAttentionModel,
        kv_manager: KVCacheManager,
        processor: Processor,
        max_batch_size: int = 16,
    ):
        self.model = model
        self.kv_manager = kv_manager
        self.processor = processor

        self.req_queue = queue.Queue()
        self.active_requests: list[InferenceRequest] = []
        self.max_batch_size = max_batch_size

        # Used to dynamically stop generation
        self.eos_token_id = processor.tokenizer.eos_token_id

    def add_request(self, tokens: list[int], out_queue: queue.Queue):
        self.req_queue.put(InferenceRequest(tokens, out_queue))

    def process_loop(self):
        """Main engine loop. Every iteration executes exactly ONE batched forward pass."""
        while True:
            while (
                len(self.active_requests) < self.max_batch_size
                and not self.req_queue.empty()
            ):
                new_req = self.req_queue.get()
                try:
                    new_req.block_table.append(self.kv_manager.allocate())
                    self.active_requests.append(new_req)
                except MemoryError:
                    self.req_queue.queue.insert(0, new_req)
                    break

            if not self.active_requests:
                time.sleep(0.01)  # Idle, wait for incoming traffic
                continue

            for req in self.active_requests:
                # If crossing a block boundary, allocate a new physical block
                if req.pos > 0 and req.pos % self.kv_manager.block_size == 0:
                    try:
                        req.block_table.append(self.kv_manager.allocate())
                    except MemoryError:
                        print(
                            f"[Warning] Out of KV cache blocks for request at pos {req.pos}"
                        )

            batch_tokens = [req.get_token_to_feed() for req in self.active_requests]
            batch_positions = [req.pos for req in self.active_requests]
            batch_block_tables = [req.block_table for req in self.active_requests]

            next_tokens = self.model.forward_batch(
                batch_tokens, batch_positions, batch_block_tables, self.kv_manager
            )

            finished_requests = []

            for i, req in enumerate(self.active_requests):
                token = next_tokens[i]

                # If we are past the prompt (Decode phase), stream the generated token
                if req.pos >= len(req.prompt_tokens) - 1:
                    req.generated_tokens.append(token)
                    req.out_queue.put(self.processor.decode(token))

                req.pos += 1

                # Check for Completion (EOS token or reached max length)
                if (
                    token == self.eos_token_id
                    or len(req.generated_tokens) >= req.max_new_tokens
                ):
                    req.out_queue.put(None)  # Signal EOF to client

                    for block in req.block_table:
                        self.kv_manager.free(block)

                    finished_requests.append(req)

            for req in finished_requests:
                self.active_requests.remove(req)
