import numpy as np
from .util import rms
from .ola_buffer import OlaBuffer
from .oscillator import Oscillator
from .yin import Yin

class Mbe(OlaBuffer):
    DEBUG = False

    def __init__(self, frame_size, num_overlap, num_partials, sr):
        super().__init__(frame_size, num_overlap)

        yin_window_size = frame_size // 2
        self._yin = Yin(yin_window_size, sr)

        self._window = np.hamming(frame_size)

        self._num_partials = num_partials
        self._oscillators = [Oscillator(sr) for _ in range(num_partials)]

        self._hop_size = frame_size // num_overlap

        self._frame_pitch_hz = None
        self._frame_gains = np.full(self._num_partials, None)
        
        self._interp_pitch_hz = np.zeros(self._hop_size)
        self._interp_gains = np.zeros([self._num_partials, self._hop_size])
        self._interp_idx = 0

        self._debug = []

    def _pre_processor(self, x):
        return x

    def _processor(self, frame):

        new_pitch_hz = self._yin.predict(frame)

        # TODO: Temporary.
        new_gains = np.ones(self._num_partials) * rms(frame)

        self._update_interp_tracks(new_pitch_hz, new_gains)

        if self.DEBUG:
            self._debug.append(new_pitch_hz)

        X = np.fft.rfft(frame)
        phase = np.random.rand(X.shape[0]) * 2 * np.pi
        X = np.abs(X) * (np.exp(np.complex(0, 1) * phase))

        frame = np.fft.irfft(X)

        return frame
    
    def _post_processor(self, x):

        pitch_hz = self._interp_pitch_hz[self._interp_idx]
        gains = self._interp_gains[:, self._interp_idx]
        self._interp_idx += 1

        x = 0

        if pitch_hz == 0:
            return x

        for i in range(self._num_partials):
            partial_hz = (i + 1) * pitch_hz
            x += self._oscillators[i].tick(partial_hz) * gains[i]

        return x
    
    def _update_interp_tracks(self, new_pitch_hz, new_gains):

        assert len(new_gains) == self._num_partials, "Must have one gain per partial."

        self._interp_idx = 0

        if self._frame_pitch_hz is None:
            self._frame_pitch_hz = new_pitch_hz

        if self._frame_gains[0] is None:
            self._frame_gains = new_gains

        self._interp_pitch_hz = np.linspace(
            self._frame_pitch_hz, new_pitch_hz, self._hop_size, endpoint=False)
        
        for i in range(self._num_partials):
            self._interp_gains[i, :] = np.linspace(
                self._frame_gains[i], new_gains[i], self._hop_size, endpoint=False)
        
        self._frame_pitch_hz = new_pitch_hz
        self._frame_gains = new_gains
    
    def get_debug(self):
        return np.array(self._debug)
    