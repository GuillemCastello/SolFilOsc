"""CPU-only PSD and noise-model utilities (no TensorFlow).

These functions are deliberately kept out of ``cnn.py`` so that the per-pixel
Lomb-Scargle workers — which run in separate joblib/loky processes and only need
``compute_ls_psd_safe`` — never import TensorFlow and therefore never allocate a
GPU/CUDA context. Importing this module must stay free of any deep-learning
dependency.
"""

import numpy as np
from astropy.timeseries import LombScargle


def noise_model(x, a, alpha, b):
    return a * x ** (-alpha) + b


def compute_ls_psd_safe(tt, yy, freqs, *, apply_hann=True, min_var=1e-12, min_energy=1e-20):
    tt = np.asarray(tt, np.float64)
    yy = np.asarray(yy, np.float64)

    ok = np.isfinite(tt) & np.isfinite(yy)
    if ok.sum() < 10:
        return None

    tt = tt[ok]
    yy = yy[ok]

    if yy.size < 10:
        return None

    yy = yy - np.mean(yy)

    var = np.var(yy)
    if not np.isfinite(var) or var <= float(min_var):
        return None

    if apply_hann and yy.size >= 8:
        yy = yy * np.hanning(yy.size)

    energy = np.sum(yy * yy)
    if not np.isfinite(energy) or energy <= float(min_energy):
        return None

    pxx = LombScargle(tt, yy).power(freqs)
    pxx = np.asarray(pxx, np.float64)

    if pxx.shape != freqs.shape:
        return None
    if not np.all(np.isfinite(pxx)):
        return None
    if np.all(pxx <= 0):
        return None

    return pxx.astype(np.float32)
