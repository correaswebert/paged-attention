import torch
from torch import Tensor

from request import Request


class FlashAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: Tensor,
        kv_cache: Tensor,
        block_table: Tensor,
        decode=True,
        current_pos=0,
        causal=True
    ):
        return x


class MLP(torch.nn.Module):
    def __init__(self):
        ...

    def forward(self):
        ...


class Model:
    def __init__(self):
        self.flash_attn = FlashAttention()
        self.mlp = MLP()

    def get_next_token(self, request: Request, block_table: Tensor):
        # mock token generation
        # x = self.flash_attn.forward(x, kv_cache, block_table)
        yield len(request.tokenized_response) + 100


def get_next_token():
    yield 0
