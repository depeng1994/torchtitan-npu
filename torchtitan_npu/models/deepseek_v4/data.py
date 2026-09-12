from dataclasses import dataclass

from torchtitan.components.tokenizer import BaseTokenizer


class SyntheticTokenizer(BaseTokenizer):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenizer.Config):
        vocab_size: int = 129280

    def __init__(self, config: Config, *, tokenizer_path: str | None = None):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.eos_id = 1

    def encode(self, text: str, **kwargs) -> list[int]:
        return [2 + (ord(char) % max(self.vocab_size - 2, 1)) for char in text]

    def decode(self, token_ids, **kwargs) -> str:
        return "".join(chr(int(token_id) % 128) for token_id in token_ids)

    def get_vocab_size(self) -> int:
        return self.vocab_size
