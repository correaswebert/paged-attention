from request import Request
from scheduler import Scheduler
from tokenizer import Tokenizer


def main():
    tokenizer = Tokenizer()
    scheduler = Scheduler()

    try:
        prompt = input(">>> ")
        tokenized_prompt = tokenizer.encode(prompt)

        request = Request(prompt=prompt, tokenized_prompt=tokenized_prompt)

        scheduler.execute(request)
        
    except KeyboardInterrupt:
        pass
