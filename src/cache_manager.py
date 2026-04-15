import torch
from torch import Tensor


class KVCacheManager:
    """Manages the physical blocks of KV Cache for Paged Attention."""

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        device="cuda",
    ):
        self.block_size = block_size
        # Shape: [num_layers, num_blocks, block_size, num_heads, head_dim]
        # Each layer gets its own independent set of paged KV blocks.
        self.k_cache = torch.zeros(
            (num_layers, num_blocks, block_size, num_heads, head_dim),
            dtype=torch.float16,
            device=device,
        )
        self.v_cache = torch.zeros(
            (num_layers, num_blocks, block_size, num_heads, head_dim),
            dtype=torch.float16,
            device=device,
        )
        self.free_blocks = list(range(num_blocks))[::-1]

    def get_layer_cache(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        # Returns [num_blocks, block_size, num_heads, head_dim] for this layer
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def allocate(self) -> int:
        if not self.free_blocks:
            raise MemoryError("Out of KV Cache blocks!")
        return self.free_blocks.pop()

    def free(self, block_idx: int):
        self.free_blocks.append(block_idx)
