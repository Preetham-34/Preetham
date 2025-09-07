from .landmarks import LandmarkExtractor
from .gesture import GestureRecognizer
from .sentence import SentenceGenerator
from .emergency import EmergencyDetector

class Models:
    def __init__(self, mock: bool = True):
        self.landmarks = LandmarkExtractor(mock=mock)
        self.gesture = GestureRecognizer(mock=mock)
        self.sentence = SentenceGenerator(mock=mock)
        self.emergency = EmergencyDetector(mock=mock)
