from typing import Tuple
import numpy as np
import random

EMERGENCY_CLASSES = ["help", "fire", "pain", "thief"]

class EmergencyDetector:
    def __init__(self, mock: bool = True):
        self.mock = mock
        if not self.mock:
            # TODO: load YOLOv5n/s ONNX model
            pass

    def __call__(self, bgr_img: np.ndarray) -> Tuple[str, float]:
        if self.mock:
            # Randomly return none vs a fake class with tiny probability for demo
            if random.random() < 0.02:
                cls = random.choice(EMERGENCY_CLASSES)
                score = 0.8 + 0.2 * random.random()
                return cls, score
            return "none", 0.0
        # TODO: run YOLO and return (class, score)
        return "none", 0.0
