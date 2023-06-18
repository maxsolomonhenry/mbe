import numpy as np
from .util import rms, interpolated_read, mag_to_db
from .ola_buffer import OlaBuffer
from .oscillator import Oscillator
from .yin import Yin

class Mbe(OlaBuffer):
    DEBUG = True
    VOICED_THRESHOLD = 2

    def __init__(self, frame_size, num_overlap, num_partials, sr):
        super().__init__(frame_size, num_overlap)

        self._sr = sr

        yin_window_size = frame_size // 2
        self._yin = Yin(yin_window_size, sr)

        self._window = self._make_energy_normalized_window(frame_size)
        self._nfft = frame_size * 4

        self._W = np.fft.fft(self._window, self._nfft)
        self._num_lobe_bins = 12 # TODO: Rename.
        self._nyquist = sr // 2 - (2 * self._num_lobe_bins * sr / self._nfft)

        self._num_partials = num_partials
        self._oscillators = [Oscillator(sr) for _ in range(num_partials)]

        self._hop_size = frame_size // num_overlap

        self._frame_pitch_hz = None
        self._frame_gains = np.full(self._num_partials, None)
        
        self._interp_pitch_hz = np.zeros(self._hop_size)
        self._interp_gains = np.zeros([self._num_partials, self._hop_size])
        self._interp_idx = 0

        self._debug = []

    def _make_energy_normalized_window(self, frame_size):
        window = np.hanning(frame_size)
        window /= np.sqrt(np.sum(window ** 2))
        return window
    
    def _calculate_partial_gains_and_errors(self, frame, f0):

        gains = np.zeros(self._num_partials)
        errors = np.zeros(self._num_partials)

        frame *= self._window
        X = np.fft.fft(frame, self._nfft)

        mX = np.abs(X)
        mW = np.abs(self._W)

        f0_bin = f0 / self._sr * self._nfft
        for i in range(self._num_partials):
            bin_idx = (i + 1) * f0_bin

            power = 0
            for j in range(self._num_lobe_bins):

                power += self._W[j] * np.conj(interpolated_read(X, bin_idx + j))
                power += self._W[j] * np.conj(interpolated_read(X, - bin_idx - j))

                if j == 0:
                    continue

                power += self._W[-j] * np.conj(interpolated_read(X, bin_idx - j))
                power += self._W[-j] * np.conj(interpolated_read(X, -bin_idx + j))

            power /= self._nfft
            power = np.real(power)

            error = 0
            norm = 0

            for j in range(self._num_lobe_bins):
                error += (interpolated_read(mX, bin_idx + j) - power * mW[j]) ** 2
                norm += interpolated_read(mX, bin_idx + j) ** 2
                
                if j == 0:
                    continue
                
                error += (interpolated_read(mX, bin_idx - j) - power * mW[-j]) ** 2
                norm += interpolated_read(mX, bin_idx - j) ** 2

            if norm != 0:
                error /= norm

            gains[i] = power
            errors[i] = error

        return gains, errors

    def _pre_processor(self, x):
        return x

    def _processor(self, frame):

        new_pitch_hz = self._yin.predict(frame)

        new_gains = np.zeros(self._num_partials)
        errors = np.zeros(self._num_partials)
        if new_pitch_hz != 0:
            new_gains, errors = self._calculate_partial_gains_and_errors(
                frame, new_pitch_hz)
        
        noise_gains = np.zeros(self._num_partials)
        for i in range(self._num_partials):
            if errors[i] > self.VOICED_THRESHOLD:
                noise_gains[i] = new_gains[i].copy()
                new_gains[i] = 0

        self._update_interp_tracks(new_pitch_hz, new_gains)

        if self.DEBUG:
            self._debug.append(errors)

        frame = self._make_unvoiced_frame(noise_gains, new_pitch_hz)
        return frame
    

    def _make_unvoiced_frame(self, noise_gains, f0):

        f0_bin = f0 / self._sr * self._nfft
        bin_idxs = [(i + 1) * f0_bin for i in range(self._num_partials)]

        hN = self._frame_size // 2 + 1
        tmp = np.arange(hN)
        mX = np.interp(tmp, bin_idxs, noise_gains)

        max_idx = int(np.ceil(np.max(bin_idxs)))

        if max_idx > hN:
            mX[max_idx:] = 0

        mX *= np.sqrt(self._frame_size)

        phase = np.random.rand(hN) * 2 * np.pi
        X = mX * (np.exp(np.complex(0, 1) * phase))

        frame = np.fft.irfft(X) / 4

        return frame

    def _post_processor(self, x):

        pitch_hz = self._interp_pitch_hz[self._interp_idx]
        gains = self._interp_gains[:, self._interp_idx]
        self._interp_idx += 1

        if pitch_hz == 0:
            return x

        for i in range(self._num_partials):
            partial_hz = (i + 1) * pitch_hz

            if partial_hz >= self._nyquist:
                continue

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
        return self._debug