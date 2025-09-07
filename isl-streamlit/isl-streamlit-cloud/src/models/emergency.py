from typing import Tuple
import numpy as np, random

EMERGENCY_CLASSES = ["help", "fire", "pain", "thief"]

class EmergencyDetector:
    def __init__(self, mock: bool = True):
        self.mock = mock
        if not self.mock:
            # TODO: Load YOLO ONNX
            pass

    def __call__(self, bgr_img: np.ndarray) -> Tuple[str, float]:
        if self.mock:
            if random.random() < 0.02:
                return random.choice(EMERGENCY_CLASSES), 0.8 + 0.2 * random.random()
            return "none", 0.0
        # TODO: real inference
        return "none", 0.0
