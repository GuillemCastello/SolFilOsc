"""Image degradation and scale-selection helpers."""

import numpy as np

def block_reduce_mean(frame2d, N, S):
    H, W = frame2d.shape
    row_starts = np.arange(0, H - N + 1, S, dtype=np.intp)
    col_starts = np.arange(0, W - N + 1, S, dtype=np.intp)

    if S == N:
        rh = (H // N) * N
        rw = (W // N) * N
        return frame2d[:rh, :rw].reshape(H // N, N, W // N, N).mean(axis=(1, 3))

    out = np.empty((len(row_starts), len(col_starts)), dtype=np.float32)
    for ii, r0 in enumerate(row_starts):
        for jj, c0 in enumerate(col_starts):
            out[ii, jj] = frame2d[r0:r0+N, c0:c0+N].mean()
    return out

def degrade_stack_unweighted(images, N, S):
    T, H, W = images.shape
    row_starts = np.arange(0, H - N + 1, S, dtype=np.intp)
    col_starts = np.arange(0, W - N + 1, S, dtype=np.intp)
    oh = len(row_starts)
    ow = len(col_starts)

    if S == N:
        rh = oh * N
        rw = ow * N
        return images[:, :rh, :rw].reshape(T, oh, N, ow, N).mean(axis=(2, 4)).astype(np.float32)

    out = np.empty((T, oh, ow), dtype=np.float32)
    for k in range(T):
        f = images[k]
        for ii, r0 in enumerate(row_starts):
            for jj, c0 in enumerate(col_starts):
                out[k, ii, jj] = f[r0:r0+N, c0:c0+N].mean()
    return out

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
