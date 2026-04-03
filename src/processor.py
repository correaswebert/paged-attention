from tokenizer import Tokenizer
from scheduler import Scheduler
from request import Request

class Processor:
    def __init__(self, scheduler: Scheduler):
        self.tokenizer = Tokenizer()
        self.scheduler = scheduler

    def process(self, prompt: str):
        # Create request and tokenize
        req = Request(prompt=prompt)
        req.tokenized_prompt = self.tokenizer.encode(prompt)
        
        # Enqueue the request in the scheduler
        self.scheduler.add_task(req)
