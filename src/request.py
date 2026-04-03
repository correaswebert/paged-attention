from dataclasses import dataclass

from torch import Tensor


from typing import List, Optional

class LogicalTokenBlock:
    def __init__(self, block_size: int, block_id: int):
        self.block_size = block_size
        self.block_id = block_id
        self.tokens: List[int] = []

    def is_full(self) -> bool:
        return len(self.tokens) == self.block_size

    def append(self, token: int):
        if self.is_full():
            raise ValueError("Logical token block is full")
        self.tokens.append(token)

@dataclass
class Request:
    prompt: str

    response: str = ""
    tokenized_prompt: Optional[list[int]] = None
    tokenized_response: Optional[list[int]] = None
    
    logical_blocks: Optional[list[LogicalTokenBlock]] = None
    block_table: Optional[list[int]] = None # Maps logical block index to physical block index

    def __post_init__(self):
        if self.tokenized_prompt is None:
            self.tokenized_prompt = []
        if self.tokenized_response is None:
            self.tokenized_response = []
        if self.logical_blocks is None:
            self.logical_blocks = []
        if self.block_table is None:
            self.block_table = []

    @property
    def last_logical_block(self) -> Optional[LogicalTokenBlock]:
        if self.logical_blocks is None or len(self.logical_blocks) == 0:
            return None
        return self.logical_blocks[-1]
