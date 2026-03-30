import torch
from torch import Tensor

from request import Request


class FlashAttention(torch.nn.Module):
    def __init__(self):
        ...

        def forward(
            self,
            x: Tensor,
            kv_cache: Tensor,
            decode=True,
            current_pos=0,
            causal=True
        ):
            ...


class MLP(torch.nn.Module):
    def __init__(self):
        ...

    def forward(self):
        ...


class Model:
    def __init__(self):
        self.flash_attn = FlashAttention()
        self.mlp = MLP()

    def get_next_token(self, request: Request):
        x = self.flash_attn.forward()
        yield 0


def get_next_token():
    yield 0
