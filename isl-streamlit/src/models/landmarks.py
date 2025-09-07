from typing import Optional
import numpy as np

class LandmarkExtractor:
    def __init__(self, mock: bool = True):
        self.mock = mock
        if not self.mock:
            # TODO: initialize MediaPipe Hands
            pass

    def __call__(self, bgr_img: np.ndarray) -> Optional[np.ndarray]:
        if self.mock:
            return None  # No landmarks in mock mode
        # TODO: return 21x2 or 21x3 normalized keypoints
        return None
