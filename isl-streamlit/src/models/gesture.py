from typing import Tuple, Any

class GestureRecognizer:
    def __init__(self, mock: bool = True):
        self.mock = mock
        if not self.mock:
            # TODO: load TF-Lite/ONNX gesture model
            pass

    def __call__(self, img_or_landmarks: Any) -> Tuple[str, float]:
        if self.mock:
            return "hello", 0.92
        # TODO: run inference and return (token/word, confidence)
        return "unknown", 0.0
