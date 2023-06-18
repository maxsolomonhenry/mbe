from scipy.io import wavfile
import numpy as np

def audioread(fpath):
    sr, audio_data = wavfile.read(fpath)

    max_value = np.iinfo(audio_data.dtype).max
    audio_data = audio_data.astype(np.float32) / max_value
    return audio_data, sr

def db_to_mag(x):
    return 10.0 ** (x / 20.0)

def db_to_power(x):
    return 10.0 ** (x / 10.0)

def mag_to_db(x):
    return 20.0 * np.log10(x)

def power_to_db(x):
    return 10.0 * np.log10(x)

def rms(x):
    return np.sqrt(np.mean(x ** 2))

def synthesize_f0(f0, frame_rate, sr):
    f0 = upsample(f0, frame_rate, sr)
    phase = np.cumsum(2 * np.pi * f0 / sr)
    return np.cos(phase)

def upsample(x, frame_rate, sr):
    num_frames = len(x)
    num_samples = int(num_frames * (sr / frame_rate))

    idx_x = np.linspace(0, num_frames, num_frames)
    idx_upsample = np.linspace(0, num_frames, num_samples)

    return np.interp(idx_upsample, idx_x, x)

def interpolated_read(x, idx):
    EPS = 1e-10

    if np.abs(idx) >= (len(x) - 1):
        return 0

    residual = idx % 1

    is_whole_number = (residual < EPS)
    if is_whole_number:
        return x[int(idx)]
    
    below = x[np.floor(idx).astype(int)]
    above = x[np.ceil(idx).astype(int)]

    return below * (1 - residual) + above * (residual)