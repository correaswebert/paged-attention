from cache import CacheManager
from request import Request
from model import Model

class Scheduler:
    def __init__(self): 
        self.cache_manager = CacheManager(16, 16)
        self.model = Model()
        self.waitq: list[Request] = []
        self.runq: list[Request] = []

    def add_task(self, request: Request):
        self.waitq.append(request)

    def execute(self, request: Request):
        """Runs the task now instead of scheduling it"""

        phy_block_id = self.cache_manager.allocate()
        request.block_table.append([phy_block_id, 0])

        # until the generation has completed
        for token in self.model.get_next_token(request):
            if request.last_block_size >= self.cache_manager.block_size:
                phy_block_id = self.cache_manager.allocate()
                request.block_table.append([phy_block_id, 0])

