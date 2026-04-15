from queue import Queue
from typing import Optional

type BlockId = int
type BlockTable = list[BlockId]
type TokenId = int


class InferenceRequest:
    """Tracks the state of a single request during continuous batching."""

    def __init__(
        self,
        prompt_tokens: list[TokenId],
        out_queue: Queue[Optional[str]],
        max_new_tokens: int = 50,
    ):
        self.prompt_tokens = prompt_tokens
        self.out_queue = out_queue
        self.max_new_tokens = max_new_tokens

        self.block_table: list[BlockId] = []
        self.generated_tokens: list[TokenId] = []
        self.pos = 0

    def get_token_to_feed(self) -> int:
        """Returns the prompt token if in prefill, or the last generated token if in decode."""
        if self.pos < len(self.prompt_tokens):
            return self.prompt_tokens[self.pos]
        return self.generated_tokens[-1]
