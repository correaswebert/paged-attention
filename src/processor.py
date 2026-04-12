from transformers import AutoTokenizer


class Processor:
    """Handles tokenization and detokenization using HF AutoTokenizer."""

    def __init__(self, model_id: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, return_tensors="pt")[0].tolist()

    def decode(self, token_id: int) -> str:
        # skip_special_tokens avoids printing padding/EOS tokens explicitly
        return self.tokenizer.decode([token_id], skip_special_tokens=True)
