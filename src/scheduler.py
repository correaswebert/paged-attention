from cache import CacheManager
from request import Request


def forward():
    ...


class Worker:
    ...


class Scheduler:
    def __init__(self): 
        self.cache_manager = CacheManager(16, 16)
        self.waitq: list[Request] = []
        self.runq: list[Request] = []
        self.NUM_WORKERS = 2
        self.workers = [Worker() for _ in range(self.NUM_WORKERS)]

    def add_task(self, request: Request):
        self.waitq.append(request)

    def run_task(self):
        self.waitq
