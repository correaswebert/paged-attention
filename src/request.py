from dataclasses import dataclass

from torch import Tensor


@dataclass
class Request:
    kv_cache: tuple[Tensor, Tensor]

    prompt: str
    tokenized_prompt: list[int]

    response: str = ""
    tokenized_response: list[int] = []

    block_table: list[list[int]] = []
    

    @property
    def last_block_size(self):
        return self.block_table[-1][1]
