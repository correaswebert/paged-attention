import torch


OOM_PREVENTION_BUFFER_SIZE = 300 * 1024 * 1024


class CacheEngine:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        head_dim: int,
        dtype=torch.bfloat16,
        device="cuda"
    ):
        shape = (num_blocks, block_size, head_dim)

        self.k_cache = torch.empty(shape, dtype=dtype, device=device)
        self.v_cache = torch.empty(shape, dtype=dtype, device=device)


class CacheManager:
    def __init__(
        self,
        block_size: int,
        head_dim: int,
        dtype=torch.float16,
        device="cuda",
    ) -> None:
        self.block_size = block_size
        self.head_dim = head_dim

        self.dtype = dtype
        self.device = device

        self.kv_cache_size = 1024 # TODO: calculate this instead of hardcoding

        # each block can have a maximum of 'block_size' number of tokens
        # every token consumes 'head_dim x sizeof(dtype)' number of bytes
        bytes_per_block = block_size * head_dim * dtype.itemsize

        # a pair of blocks for KV takes twice the bytes per block
        self.num_blocks = self.kv_cache_size // (2 * bytes_per_block)

        shape = (self.num_blocks, block_size, head_dim)

        self.k_cache = torch.empty(shape, dtype=dtype, device=device)
        self.v_cache = torch.empty(shape, dtype=dtype, device=device)

        # tracks which physical blocks are free
        self.free_list = [i for i in range(self.num_blocks)]
        
        # tracks the reference count of each physical block
        self.refcount = [0 for _ in range(self.num_blocks)]

    @property
    def kv_cache(self):
        return self.k_cache, self.v_cache

    def allocate(self) -> int:
        """Issue the next free physical block"""
        try:
            phy_block_id = self.free_list.pop()

        except IndexError:
            raise Exception("All physical blocks are allocated")
            # TODO: trigger cache offloading

        self.refcount[phy_block_id] = 1

        return phy_block_id

    def free(self, phy_block_id: int):
        self.refcount[phy_block_id] -= 1

        if self.refcount[phy_block_id] == 0:
            self.free_list.append(phy_block_id)

    def append(self):
        """Adds token at the end of block list
        
        The token is appended to the end of the physical (and correspondingly
        logical) block. If the refcount is not one, then a copy-on-write is
        triggered to diverge. In case the physical block is full, a new block
        is allocated.
        """
        ...

    def copy(self):
        ...

    def fork(self):
        ...
