import typer

from tokenizer import Tokenizer
from scheduler import Scheduler


def main(prompt: str):
    tok = Tokenizer()
    sched = Scheduler()

    tokens = tok.encode(prompt)
    sched.execute(tokens)


if __name__ == "__main__":
    typer.run(main)
