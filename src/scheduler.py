import torch
from torch import Tensor

from cache_manager import CacheManager
from request import Request
from model import Model

class Scheduler:
    def __init__(self): 
        # Using smaller sizes for testing / demonstration
        self.cache_manager = CacheManager(block_size=16, head_dim=64)
        self.model = Model()
        self.waitq: list[Request] = []
        self.runq: list[Request] = []

    def add_task(self, request: Request):
        self.waitq.append(request)

    def execute(self, request: Request):
        """Runs the task to completion"""
        self.add_task(request)
        
        while self.waitq or self.runq:
            self.schedule()
            self.step()

    def schedule(self):
        """Moves requests from wait queue to run queue depending on available blocks"""
        # A simple prefill schedule: just try to admit as many tasks as possible
        # Normally, we'd estimate if there's enough blocks for the prompt
        
        # For simplicity, if we have a request in waitq, try to admit it
        remaining_waitq = []
        for req in self.waitq:
            # check if we can allocate physical blocks for the initial prompt
            # here we might just lazily move it and let the first step allocate
            self.runq.append(req)
            
        self.waitq = remaining_waitq

    def step(self):
        """Executes one decoding step for all running requests"""
        completed = []
        for req in list(self.runq): # using list() to allow concurrent removal
            # Decode one token
            try:
                # Get block table as tensor
                block_table_tensor = self.cache_manager.get_block_table_tensor(req)
                
                # get_next_token is a mock in model.py
                token_gen = self.model.get_next_token(req, block_table_tensor)
                next_token = next(token_gen)
                req.tokenized_response.append(next_token)
                
                # Update KV cache
                self.cache_manager.append(req, next_token)
                
                print(f"[Scheduler] Req '{req.prompt}' generated {len(req.tokenized_response)} tokens. Block table len: {len(block_table_tensor)}")
                
                # Stop condition (mocked: 10 tokens max)
                if len(req.tokenized_response) >= 10:
                    completed.append(req)
                elif req.last_logical_block and req.last_logical_block.is_full():
                    # Round robin: block is full, yield to other requests in waitq
                    print(f"[Scheduler] Req '{req.prompt}' filled a block. Preempting.")
                    self.runq.remove(req)
                    self.waitq.append(req)
            except Exception as e:
                # OOM or other error triggers preemption/fail
                print(f"Error executing request: {e}")
                completed.append(req)

        # Remove completed requests and free their physical memory
        for req in completed:
            if req in self.runq:
                self.runq.remove(req)
            self.cache_manager.free_request(req)
            print(f"[Scheduler] Req '{req.prompt}' completed.")

