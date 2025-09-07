from typing import Tuple, Any

class GestureRecognizer:
    def __init__(self, mock: bool = True):
        self.mock = mock
        if not self.mock:
            # TODO: Load ONNX/TF-Lite model
            pass

    def __call__(self, img_or_landmarks: Any) -> Tuple[str, float]:
        if self.mock:
            return "hello", 0.92
        # TODO: real inference
        return "unknown", 0.0
