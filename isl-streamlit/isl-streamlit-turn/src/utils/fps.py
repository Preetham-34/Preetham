import time

class FpsTracker:
    def __init__(self):
        self.last = time.time()
        self.avg = 0.0

    def tick(self):
        now = time.time()
        dt = max(1e-6, now - self.last)
        fps = 1.0 / dt
        self.avg = 0.9 * self.avg + 0.1 * fps if self.avg > 0 else fps
        self.last = now
        return self.avg
