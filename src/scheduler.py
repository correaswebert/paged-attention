import threading
import time
from queue import Queue

from cache_manager import KVCacheManager
from model import PagedAttentionModel
from request import InferenceRequest


class Scheduler:
    """Continuous Batching Scheduler for Paged Attention."""

    def __init__(
        self,
        model: PagedAttentionModel,
        kv_manager: KVCacheManager,
        eos_token_id,
        max_batch_size: int = 16,
    ):
        self.model = model
        self.kv_manager = kv_manager

        self.pending_requests: Queue[InferenceRequest] = Queue()
        self.active_requests: list[InferenceRequest] = []
        self.max_batch_size = max_batch_size

        # Used to dynamically stop generation
        self.eos_token_id = eos_token_id

    def run(self):
        threading.Thread(target=self.process_loop, daemon=True).start()

    def stop(self):
        ...

    def add_request(self, request: InferenceRequest):
        self.pending_requests.put(request)

    def process_loop(self):
        """Main engine loop. Every iteration executes exactly ONE batched forward pass."""
        while True:
            while (
                len(self.active_requests) < self.max_batch_size
                and not self.pending_requests.empty()
            ):
                next_req = self.pending_requests.get()

                try:
                    phy_block_id = self.kv_manager.allocate()
                    next_req.block_table.append(phy_block_id)
                    self.active_requests.append(next_req)
                except MemoryError:
                    self.pending_requests.queue.insert(0, next_req)
                    break

            if not self.active_requests:
                time.sleep(0.01)  # Idle, wait for incoming traffic
                continue

            for req in self.active_requests:
                # If crossing a block boundary, allocate a new physical block
                if req.pos > 0 and req.pos % self.kv_manager.block_size == 0:
                    try:
                        phy_block_id = self.kv_manager.allocate()
                        req.block_table.append(phy_block_id)
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
                    req.out_queue.put(token)

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
