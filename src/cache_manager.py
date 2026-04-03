import torch

from request import Request
from request import LogicalTokenBlock

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

        self.kv_cache_size = 10 * 1024 * 1024 # 10MB default for testing

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

    def append(self, request: Request, token_id: int):
        """Adds token at the end of block list
        
        The token is appended to the end of the logical block. In case the
        logical block is full, a new logical and physical block are allocated.
        """

        last_logical_block = request.last_logical_block

        if last_logical_block is None or last_logical_block.is_full():
            logical_block_id = len(request.logical_blocks) if request.logical_blocks else 0
            new_block = LogicalTokenBlock(self.block_size, logical_block_id)
            
            phy_block_id = self.allocate()
            
            request.logical_blocks.append(new_block)
            request.block_table.append(phy_block_id)
            
            last_logical_block = new_block

        last_logical_block.append(token_id)
        # In a real implementation we would also write the KV cache to physical memory

    def free_request(self, request: Request):
        if not request.block_table:
            return

        for phy_block_id in request.block_table:
            self.free(phy_block_id)

        request.block_table = []
        
    def get_block_table_tensor(self, request: Request) -> torch.Tensor:
        if not request.block_table:
            return torch.tensor([], dtype=torch.int32, device=self.device)
        return torch.tensor(request.block_table, dtype=torch.int32, device=self.device)

    def copy(self):
        ...

    def fork(self):
        ...
