from dataclasses import dataclass


@dataclass
class Request:
    prompt: str
    tokenized_prompt: list[int]
    block_table: dict[int, int] = {}
