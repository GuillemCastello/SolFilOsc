"""Image degradation and scale-selection helpers."""

import numpy as np

def _block_means_integral(stack3d, N, S):
    """Overlapping NxN block means with stride S over a (T, H, W) stack.

    Uses per-frame integral images (summed-area tables) so the cost is
    O(T*H*W) instead of one Python-level ``.mean()`` call per block.
    Blocks containing NaN yield NaN, matching ``np.ndarray.mean``.
    """
    T, H, W = stack3d.shape
    r0 = np.arange(0, H - N + 1, S, dtype=np.intp)
    c0 = np.arange(0, W - N + 1, S, dtype=np.intp)
    r1 = r0 + N
    c1 = c0 + N
    out = np.empty((T, len(r0), len(c0)), dtype=np.float32)

    # Chunk over time to bound the float64 integral-image temporaries.
    chunk = max(1, int(2**26 // max(1, (H + 1) * (W + 1))))
    ii = np.zeros((min(chunk, T), H + 1, W + 1), dtype=np.float64)

    for t0 in range(0, T, chunk):
        t1 = min(t0 + chunk, T)
        a = stack3d[t0:t1].astype(np.float64, copy=False)

        nan_mask = np.isnan(a)
        has_nan = bool(nan_mask.any())
        if has_nan:
            a = np.where(nan_mask, 0.0, a)

        buf = ii[: t1 - t0]
        buf[:, 1:, 1:] = a.cumsum(axis=1).cumsum(axis=2)
        sums = (
            buf[:, r1][:, :, c1] - buf[:, r0][:, :, c1]
            - buf[:, r1][:, :, c0] + buf[:, r0][:, :, c0]
        )
        block = sums / float(N * N)

        if has_nan:
            cnt = np.zeros_like(buf)
            cnt[:, 1:, 1:] = nan_mask.cumsum(axis=1).cumsum(axis=2)
            n_nan = (
                cnt[:, r1][:, :, c1] - cnt[:, r0][:, :, c1]
                - cnt[:, r1][:, :, c0] + cnt[:, r0][:, :, c0]
            )
            block[n_nan > 0] = np.nan

        out[t0:t1] = block.astype(np.float32)
    return out

def block_reduce_mean(frame2d, N, S):
    H, W = frame2d.shape

    if S == N:
        rh = (H // N) * N
        rw = (W // N) * N
        return frame2d[:rh, :rw].reshape(H // N, N, W // N, N).mean(axis=(1, 3))

    return _block_means_integral(np.asarray(frame2d)[None], N, S)[0]

def degrade_stack_unweighted(images, N, S):
    T, H, W = images.shape

    if S == N:
        oh = (H - N) // S + 1
        ow = (W - N) // S + 1
        rh = oh * N
        rw = ow * N
        return images[:, :rh, :rw].reshape(T, oh, N, ow, N).mean(axis=(2, 4)).astype(np.float32)

    return _block_means_integral(images, N, S)

def degrade_mask_coverage(mask2d, N, S):
    m = (mask2d > 0).astype(np.float32)
    return block_reduce_mean(m, N, S)

def choose_scales_distinct_quotients(bbox_H, start=50, stop=2):
    prev = None
    vals = []
    for N in range(int(start), int(stop), -1):
        q = int(bbox_H // N)
        if prev is None or q != prev:
            vals.append(int(N))
            prev = q
    return vals

def choose_scales_log_ladder(Nmax, Nmin, n=12):
    vals = np.unique(np.round(np.geomspace(int(Nmax), int(Nmin), int(n))).astype(int))[::-1]
    return [int(v) for v in vals if v >= int(Nmin)]

def apply_null_time_transform(stack, mode, rng):
    if mode == "none":
        return stack
    if mode == "circular_shift":
        return np.roll(stack, shift=int(rng.integers(1, stack.shape[0])), axis=0)
    if mode == "shuffle":
        return stack[rng.permutation(stack.shape[0])]
    raise ValueError("null mode must be: none | circular_shift | shuffle")
