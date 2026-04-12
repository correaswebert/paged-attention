import torch


class KVCacheManager:
    """Manages the physical blocks of KV Cache for Paged Attention."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        device="cuda",
    ):
        self.block_size = block_size
        # flash_attn_with_kvcache expects caches of shape:
        # [num_blocks, block_size, num_heads, head_dim]
        self.k_cache = torch.zeros(
            (num_blocks, block_size, num_heads, head_dim),
            dtype=torch.float16,
            device=device,
        )
        self.v_cache = torch.zeros(
            (num_blocks, block_size, num_heads, head_dim),
            dtype=torch.float16,
            device=device,
        )
        self.free_blocks = list(range(num_blocks))[::-1]

    def allocate(self) -> int:
        if not self.free_blocks:
            raise MemoryError("Out of KV Cache blocks!")
        return self.free_blocks.pop()

    def free(self, block_idx: int):
        self.free_blocks.append(block_idx)
