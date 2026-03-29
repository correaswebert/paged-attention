import torch


OOM_PREVENTION_BUFFER_SIZE = 300 * 1024 * 1024


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

    def allocate(self):
        ...

    def free(self):
        ...
