"""Conformal-prediction calibration for daily solar image cubes."""

import math
import os

import h5py as h5
import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import binary_dilation
from tqdm import tqdm

from .cnn import CNN, _build_scaler, cnn_predict_noise_params, compute_ls_psd_safe, load_cnn, noise_model
from .constants import CNN_FREQUENCY_GRID, N_PIXEL_WORKERS

def compute_cp_calibration(
    images: np.ndarray,
    tdeltas: np.ndarray,
    masks: np.ndarray,
    cnn_model: CNN,
    scaler: MinMaxScaler,
    *,
    n_calib: int = 100_000,
    delta: float = 0.001,
    rng_seed: int = 42,
    n_jobs: int = N_PIXEL_WORKERS,
    batch_size_cnn: int = 4096,
):
    T, H, W = images.shape
    freqs = CNN_FREQUENCY_GRID.astype(np.float64)
    rng = np.random.default_rng(int(rng_seed))
    eps = 1e-30

    mask0 = (masks[0] > 0)
    exclusion_mask = binary_dilation(mask0, iterations=10)
    calib_candidate_mask = ~exclusion_mask

    calib_candidate_mask &= (images[0] > 0)

    finite_count = np.isfinite(images).sum(axis=0)
    valid_mask = calib_candidate_mask & (finite_count >= max(10, int(0.8 * T)))

    temporal_mean = np.nanmean(images.astype(np.float64), axis=0)
    mean_vals = temporal_mean[valid_mask & np.isfinite(temporal_mean)]
    if mean_vals.size == 0:
        raise RuntimeError("No valid candidate pixels found for CP calibration.")

    p5, p95 = np.percentile(mean_vals, [5.0, 95.0])
    valid_mask &= np.isfinite(temporal_mean) & (temporal_mean >= p5) & (temporal_mean <= p95)

    ys, xs = np.where(valid_mask)
    n_avail = len(ys)
    if n_avail == 0:
        raise RuntimeError("No valid calibration pixels found after filtering.")

    n_draw = min(int(n_calib), n_avail)
    if n_draw < n_avail:
        idx = rng.choice(n_avail, size=n_draw, replace=False)
        ys = ys[idx]
        xs = xs[idx]

    print(
        f"[CP calib] Candidate mask built from ~dilate(masks[0], 10). "
        f"Temporal-mean filter kept [{p5:.3e}, {p95:.3e}] percentile range."
    )
    print(
        f"[CP calib] Drawing {n_draw} pixels for calibration "
        f"(delta={delta}, -> {100*(1-delta):.5f}% upper quantile)."
    )

    t_ok = np.isfinite(tdeltas)
    tdeltas_clean = tdeltas[t_ok]

    def _psd_one(y_px, x_px):
        y = images[t_ok, y_px, x_px].astype(np.float64)
        ok = np.isfinite(y)
        if ok.sum() < 10:
            return None

        yy = y[ok]
        tt = tdeltas_clean[ok]

        return compute_ls_psd_safe(
            tt, yy, freqs,
            apply_hann=True,
            min_var=1e-12,
            min_energy=1e-20,
        )

    print("[CP calib] Computing calibration PSDs ...")
    psds_list = Parallel(n_jobs=n_jobs)(
        delayed(_psd_one)(int(ys[k]), int(xs[k]))
        for k in tqdm(range(n_draw), desc="Calib PSDs", leave=False)
    )
    psds_list = [p for p in psds_list if p is not None]

    if not psds_list:
        raise RuntimeError("All sampled calibration pixels were invalid for PSD computation.")

    calib_psds = np.vstack(psds_list).astype(np.float32)
    row_ok = np.all(np.isfinite(calib_psds), axis=1) & np.any(calib_psds > 0, axis=1)
    calib_psds = calib_psds[row_ok]

    n_eff = calib_psds.shape[0]
    if n_eff == 0:
        raise RuntimeError("No valid finite calibration PSDs remained after filtering.")

    print(f"[CP calib] Effective calibration PSDs after finite filtering: {n_eff}")

    print("[CP calib] Running CNN inference ...")
    n_batches = math.ceil(n_eff / batch_size_cnn)
    all_params = []
    for b in range(n_batches):
        sl = slice(b * batch_size_cnn, (b + 1) * batch_size_cnn)
        all_params.append(cnn_predict_noise_params(cnn_model, calib_psds[sl], scaler))
    noise_params = np.vstack(all_params)

    print("[CP calib] Building residual statistics ...")
    R = np.empty_like(calib_psds, dtype=np.float64)

    for i in range(n_eff):
        Shat = noise_model(freqs, *noise_params[i])
        psd_safe = np.maximum(calib_psds[i].astype(np.float64), eps)
        shat_safe = np.maximum(Shat, eps)
        R[i, :] = np.log(psd_safe) - np.log(shat_safe)

    row_ok = np.all(np.isfinite(R), axis=1)
    R = R[row_ok]
    if R.shape[0] == 0:
        raise RuntimeError("All calibration residual rows were non-finite.")

    m = np.median(R, axis=0)
    s = 1.4826 * np.median(np.abs(R - m[None, :]), axis=0)
    s = np.maximum(s, 1e-8)

    Z = (R - m[None, :]) / s[None, :]
    qk = np.quantile(Z, 1.0 - delta, axis=0, method="higher")

    if not (np.all(np.isfinite(m)) and np.all(np.isfinite(s)) and np.all(np.isfinite(qk))):
        raise RuntimeError("CP statistics contain non-finite values; aborting cache write.")

    factor = np.exp(m + qk * s)
    print(
        f"[CP calib] Done. multiplicative factor range: "
        f"[{factor.min():.3e}, {factor.max():.3e}]"
    )

    return m.astype(np.float64), s.astype(np.float64), qk.astype(np.float64)

def get_cp_cache_path(day_outroot, day, cp_delta):
    return os.path.join(
        os.path.abspath(day_outroot),
        f"cp_stats_{day}_delta{cp_delta:.6f}_dil10_p05p95.npz"
    )

def ensure_cp_cache_for_day(
    *,
    day,
    data_h5,
    masks_h5,
    outroot,
    cnn_weights_path,
    cp_n_calib=100_000,
    cp_delta=0.01,
    cp_batch_size_cnn=4096,
    n_jobs=N_PIXEL_WORKERS,
):
    os.makedirs(outroot, exist_ok=True)

    cp_cache_path = get_cp_cache_path(outroot, day, cp_delta)
    if os.path.exists(cp_cache_path):
        print(f"[CP precompute] Cache already exists: {cp_cache_path}")
        return cp_cache_path

    print(f"[CP precompute] Building daily CP cache for day={day}")

    with h5.File(data_h5, "r") as hf:
        images = np.array(hf["time_series"][:], dtype=np.float32)
        tdeltas = np.array(hf["tdeltas"][:], dtype=np.float64)

    with h5.File(masks_h5, "r") as hm:
        masks = np.array(hm["masks"][:], dtype=np.uint8)

    order = np.argsort(tdeltas)
    tdeltas = tdeltas[order]
    images = images[order]
    masks = masks[order]

    cnn_model = load_cnn(cnn_weights_path)
    scaler = _build_scaler()

    m_cp, s_cp, qk = compute_cp_calibration(
        images, tdeltas, masks, cnn_model, scaler,
        n_calib=int(cp_n_calib),
        delta=float(cp_delta),
        n_jobs=int(n_jobs),
        batch_size_cnn=int(cp_batch_size_cnn),
    )

    np.savez(
        cp_cache_path,
        m=m_cp,
        s=s_cp,
        qk=qk,
        delta=np.float64(cp_delta),
    )

    print(f"[CP precompute] Done -> {cp_cache_path}")
    return cp_cache_path
