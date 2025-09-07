from typing import List

class SentenceGenerator:
    def __init__(self, mock: bool = True):
        self.mock = mock
        if not self.mock:
            # TODO: Load T5 small (quantized)
            pass

    def __call__(self, tokens: List[str]) -> str:
        if self.mock:
            return " ".join(tokens).strip()
        # TODO: real inference
        return " ".join(tokens).strip()
