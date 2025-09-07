from typing import List

class SentenceGenerator:
    def __init__(self, mock: bool = True):
        self.mock = mock
        if not self.mock:
            # TODO: load T5-small (quantized) or ONNX equivalent
            pass

    def __call__(self, tokens: List[str]) -> str:
        if self.mock:
            # naive join
            return " ".join(tokens).strip()
        # TODO: run T5 and return refined sentence
        return " ".join(tokens).strip()
