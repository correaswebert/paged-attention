from request import Request
from scheduler import Scheduler
from tokenizer import Tokenizer


def main():
    tokenizer = Tokenizer()
    scheduler = Scheduler()

    while True:
        prompt = input(">>> ")
        tokenized_prompt = tokenizer.encode(prompt)

        request = Request(prompt=prompt, tokenized_prompt=tokenized_prompt)

        scheduler.add_task(request)


if __name__ == "__main__":
    try:
        main()
    
    except KeyboardInterrupt:
        pass
