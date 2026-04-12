import queue

from cache_manager import KVCacheManager
from model import PagedAttentionModel
from processor import Processor


class Scheduler:
    """Queues requests, manages block lifecycle, and runs generation."""

    def __init__(
        self,
        model: PagedAttentionModel,
        kv_manager: KVCacheManager,
        processor: Processor,
    ):
        self.model = model
        self.kv_manager = kv_manager
        self.processor = processor
        self.req_queue = queue.Queue()

    def add_request(self, tokens: list[int], out_queue: queue.Queue):
        self.req_queue.put(({"tokens": tokens}, out_queue))

    def process_loop(self):
        """Runs in a background thread to sequentially process the queue."""
        while True:
            req, out_queue = self.req_queue.get()
            if req is None:
                break

            prompt_tokens = req["tokens"]
            block_table = [self.kv_manager.allocate()]
            pos = 0

            # Context Prefill + Generation loop (Generate up to 20 new tokens)
            total_steps = len(prompt_tokens) + 20
            last_token = prompt_tokens[0]

            for i in range(1, total_steps):
                # Dynamically allocate new blocks if we cross the block size boundary
                if pos > 0 and pos % self.kv_manager.block_size == 0:
                    block_table.append(self.kv_manager.allocate())

                token_to_feed = (
                    prompt_tokens[i] if i < len(prompt_tokens) else last_token
                )

                next_token = self.model.forward_step(
                    token_to_feed, pos, block_table, self.kv_manager
                )
                pos += 1
                last_token = next_token

                # If we are in the generation phase, stream the decoded string back
                if i >= len(prompt_tokens):
                    text_chunk = self.processor.decode(next_token)
                    out_queue.put(text_chunk)

            out_queue.put(None)  # EOF Signal

            for b in block_table:
                self.kv_manager.free(b)
