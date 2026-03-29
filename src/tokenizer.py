import tiktoken


class Tokenizer:
    def __init__(self):
        self.enc = tiktoken.encoding_for_model("gpt-4o")
        self.special_tokens = {
            "<|endoftext|>"
        }

    def encode(self, prompt: str) -> list[int]:
        return self.enc.encode(prompt, allowed_special=self.special_tokens)

    def decode(self, tokens: list[int]) -> str:
        return self.enc.decode(tokens)
