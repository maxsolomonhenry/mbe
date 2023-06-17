import numpy as np
from olabuffer import OlaBuffer
from .yin import Yin

import matplotlib.pyplot as plt

class Mbe(OlaBuffer):
    DEBUG = False

    def __init__(self, frame_size, num_overlap, sr):
        super().__init__(frame_size, num_overlap)

        window_size = frame_size // 2
        self._yin = Yin(window_size, sr)

        self._debug = []

    def _pre_processor(self, x):
        return x

    def _processor(self, frame):

        pitch_hz = self._yin.predict(frame)

        if self.DEBUG:
            self._debug.append(pitch_hz)

        return frame
    
    def _post_processor(self, x):
        return x
    
    def get_debug(self):
        return np.array(self._debug)