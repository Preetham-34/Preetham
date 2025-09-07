from typing import Optional
import numpy as np

class LandmarkExtractor:
    def __init__(self, mock: bool = True):
        self.mock = mock
        if not self.mock:
            # TODO: Initialize MediaPipe Hands
            pass

    def __call__(self, bgr_img: np.ndarray) -> Optional[np.ndarray]:
        if self.mock:
            return None
        # TODO: return normalized 21xN landmarks
        return None
