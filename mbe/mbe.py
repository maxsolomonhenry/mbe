import numpy as np
from .ola_buffer import OlaBuffer
from .oscillator import Oscillator
from .yin import Yin

class Mbe(OlaBuffer):
    DEBUG = False

    def __init__(self, frame_size, num_overlap, num_partials, sr):
        super().__init__(frame_size, num_overlap)

        window_size = frame_size // 2
        self._yin = Yin(window_size, sr)

        self._num_partials = num_partials
        self._oscillators = [Oscillator(sr) for _ in range(num_partials)]

        self._hop_size = frame_size // num_overlap
        self._frame_pitch_hz = None
        self._interp_pitch_hz = np.zeros(self._hop_size)
        self._pitch_idx = 0

        self._debug = []

    def _pre_processor(self, x):
        return x

    def _processor(self, frame):

        new_pitch_hz = self._yin.predict(frame)

        self._update_pitch_track(new_pitch_hz)

        if self.DEBUG:
            self._debug.append(new_pitch_hz)

        X = np.fft.rfft(frame)
        phase = np.random.rand(X.shape[0]) * 2 * np.pi
        X = np.abs(X) * (np.exp(np.complex(0, 1) * phase))

        frame = np.fft.irfft(X)

        return frame
    
    def _post_processor(self, x):

        pitch_hz = self._interp_pitch_hz[self._pitch_idx]
        self._pitch_idx += 1

        assert self._pitch_idx <= self._hop_size, f"Track did not reset properly."

        if pitch_hz == 0:
            return x

        for i, oscillator in enumerate(self._oscillators, start=1):
            x += oscillator.tick(i * pitch_hz) / i

        # Debug: normalize.
        x /= self._num_partials

        return x
    
    def _update_pitch_track(self, new_pitch_hz):
        self._pitch_idx = 0

        # If not initialized (i.e., first frame)...
        if self._frame_pitch_hz is None:
            self._frame_pitch_hz = new_pitch_hz

        self._interp_pitch_hz = np.linspace(
            self._frame_pitch_hz, new_pitch_hz, self._hop_size, endpoint=False)
        
        self._frame_pitch_hz = new_pitch_hz
    
    def get_debug(self):
        return np.array(self._debug)
    