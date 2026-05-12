import numpy as np
from decorators import benchmark
from typing import Tuple
from scipy.special import factorial

#@benchmark
def remove_limb_darkening(data, grid, u2=0.88, v2=-0.23, radius=900):
    #######################################
    # data : numpy array to remove limb darkening from
    # u2 & v2: obtained from Cox, A.N.: 2000, Allen’s astrophysical quantities, 355
    # Other methods can be applied but this is used broadly
    # radius: for our original images the radius is 900 (obtained from header)
    ########################################

    # Normalize to get values between 0 and 1
    # 0 -> center / 1 -> border / basically tells us the angle radially
    grid = grid/radius

    # values from outside the sun go to zero
    out = np.where(grid>1) 
    grid[out]=0

    # Angle mu for limb darkening / 1->center / 0->border
    mu = np.cos(grid) 
    
    # Correct the data
    limb_darkening = 1 - u2 - v2 + u2*mu + v2*mu**2
    corrected_data = data/limb_darkening
    return corrected_data


def correct_limb_darkening(
    img: np.ndarray,
    centre: Tuple[int, int] = (1023, 1023),
    radius: int = 899,
    degree: int = 5,
    nbins: int = 200,
) -> np.ndarray:
    """
    Flatten Hα limb darkening.

    Parameters
    ----------
    img     : 2-D ndarray
    centre  : (row, col) tuple in px.  If None, found automatically.
    radius  : float px.  If None, found automatically.
    degree  : polynomial order in μ = cos θ (default 5).
    nbins   : # radial bins for robust median profile.

    Returns
    -------
    corr : 2-D ndarray   (same shape as img)
    """
    # ---------------- radial profile -----------------
    y, x   = np.indices(img.shape)
    r_norm = np.sqrt((y-centre[0])**2 + (x-centre[1])**2) / radius
    mask   = r_norm <= 0.98                      # only inside the disk
    r_vals = r_norm[mask].ravel()
    i_vals = img[mask].ravel()

    # robust median profile
    bins    = np.linspace(0, 1, nbins+1)
    idx     = np.digitize(r_vals, bins) - 1
    mu_bins = np.sqrt(1 - ((bins[:-1]+bins[1:])*0.5)**2)    # μ for each bin
    med_I   = np.array([np.median(i_vals[idx==k])
                        if np.any(idx==k) else 0
                        for k in range(nbins)])

    # poly fit in μ
    good    = med_I > 0          # ignore empty bins
    coeff   = np.polyfit(mu_bins[good], med_I[good], degree)

    # build 2-D model
    mu_full = np.sqrt(1 - np.clip(r_norm, 0, 1)**2)
    model   = np.polyval(coeff, mu_full)

    # avoid divide-by-tiny near the limb
    floor   = 1e-6
    corr    = img / np.maximum(model, floor)

    return corr

# ---------- Zernike helpers ----------
def _zernike_radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """Radial Zernike polynomial R_n^m(rho), with n>=0, |m|<=n, (n-m) even."""
    m = abs(m)
    if (n - m) % 2 != 0:
        return np.zeros_like(rho)
    R = np.zeros_like(rho, dtype=np.float64)
    kmax = (n - m) // 2
    for k in range(kmax + 1):
        c = ((-1.0) ** k *
             factorial(n - k) /
             (factorial(k) *
              factorial((n + m)//2 - k) *
              factorial((n - m)//2 - k)))
        R = R + c * (rho ** (n - 2 * k))
    return R

def _zernike_basis_filtered(
    rho: np.ndarray,
    theta: np.ndarray,
    n_max: int,
    m_max: int | None = None,
    include_piston: bool = True,
):
    """
    Build Zernike basis up to n_max on unit disk.
    If m_max is not None, keep only |m| <= m_max.
    For m>0 include both cos and sin terms; for m=0 only the radial term.
    Returns list of modes and (n,m,kind) metadata.
    """
    basis, meta = [], []
    for n in range(n_max + 1):
        for m in range(n + 1):              # m >= 0
            if (n - m) % 2 != 0:
                continue
            if (m_max is not None) and (m > m_max):
                continue
            R = _zernike_radial(n, m, rho)
            if m == 0:
                if n == 0 and not include_piston:
                    continue
                basis.append(R)
                meta.append((n, 0, 'piston' if n == 0 else 'cos'))
            else:
                basis.append(R * np.cos(m * theta)); meta.append((n, +m, 'cos'))
                basis.append(R * np.sin(m * theta)); meta.append((n, -m, 'sin'))
    if not basis:
        basis = [np.ones_like(rho)]
        meta  = [(0, 0, 'piston')]
    return basis, meta

# ---------- Drop-in replacement tuned for post-limb-darkening frames ----------
def polynomial_background(
    img: np.ndarray,
    centre: Tuple[int, int],
    radius: float,
    order: int,
    sigma_clip: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Zernike-based robust background removal (for images AFTER limb-darkening correction).

    Parameters
    ----------
    img : 2D ndarray
        Input image (e.g., 2048x2048), integers 0..N; cast to float64 internally.
    centre : (row, col)
        Solar disk center in pixels.
    radius : float
        Solar disk radius in pixels.
    order : int
        Maximum Zernike radial order 'n_max'. For post-LD frames, 4–6 is usually enough.
    sigma_clip : float
        Sigma threshold for residual clipping (e.g., 3.0).

    Returns
    -------
    corrected : 2D float64
        Background-corrected image, normalized to on-disk median ≈ 1.
    background : 2D float64
        Fitted smooth background (NaN outside the disk).
    """
    # ---- tuned defaults for your pipeline ----
    rim_fraction = 0.97      # keep a small rim exclusion to avoid limb edge
    m_max = 2                # limit angular complexity to avoid fitting solar texture
    use_log = True           # multiplicative residuals benefit from log domain
    max_iter = 8             # robust residual clipping iterations
    tol = 1e-3               # stop when inlier mask changes <0.1%
    subsample = 2            # speed-up: fit on a decimated grid
    include_piston = True    # keep overall scale (we renormalize after)
    ridge = 1e-8             # tiny Tikhonov for stability
    # -----------------------------------------

    img = img.astype(np.float64, copy=False)

    # Normalize coords to the unit disk
    yy, xx = np.indices(img.shape)
    yc, xc = centre
    x = (xx - xc) / float(radius)
    y = (yy - yc) / float(radius)
    rho = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)

    # Disk mask with limb exclusion
    disk_mask = rho <= rim_fraction

    # Work in log domain for multiplicative background
    work = np.log(np.maximum(img, 1e-6)) if use_log else img.copy()

    # Subsample for fitting speed (evaluate full-res later)
    if subsample > 1:
        accel = ((xx % subsample == 0) & (yy % subsample == 0))
        fit_mask = disk_mask & accel
    else:
        fit_mask = disk_mask

    rho_s = rho[fit_mask]
    th_s  = theta[fit_mask]
    z_s   = work[fit_mask]

    # Build filtered Zernike basis at sample points (P x M design matrix)
    basis_list, _ = _zernike_basis_filtered(rho_s, th_s, n_max=int(order),
                                            m_max=m_max, include_piston=include_piston)
    A = np.vstack([b.ravel() for b in basis_list]).T  # shape (P, M)

    # Robust LSQ with residual sigma-clipping
    inliers = np.isfinite(z_s).copy()
    prev_inliers = None

    for _ in range(max_iter):
        if inliers.sum() < A.shape[1]:
            inliers[:] = True
        Ai, zi = A[inliers], z_s[inliers]
        # normal equations with tiny ridge
        ATA = Ai.T @ Ai
        ATz = Ai.T @ zi
        c = np.linalg.solve(ATA + ridge * np.eye(ATA.shape[0]), ATz)

        pred = A @ c
        resid = z_s - pred
        med  = np.median(resid[inliers]) if inliers.any() else np.median(resid)
        mad  = np.median(np.abs(resid - med)) + 1e-12
        rsig = 1.4826 * mad
        new_inliers = np.abs(resid - med) < sigma_clip * max(rsig, 1e-12)

        if prev_inliers is not None:
            changed = np.count_nonzero(new_inliers ^ prev_inliers) / max(np.count_nonzero(new_inliers), 1)
            if changed < tol:
                inliers = new_inliers
                break
        prev_inliers = inliers
        inliers = new_inliers

    # Final fit on last inliers
    if inliers.sum() < A.shape[1]:
        inliers[:] = True
    Ai, zi = A[inliers], z_s[inliers]
    ATA = Ai.T @ Ai
    ATz = Ai.T @ zi
    c = np.linalg.solve(ATA + ridge * np.eye(ATA.shape[0]), ATz)

    # -------- Evaluate background on full grid (inside disk) --------
    basis_full, _ = _zernike_basis_filtered(rho, theta, n_max=int(order),
                                            m_max=m_max, include_piston=include_piston)
    bg = np.zeros_like(work, dtype=np.float64)
    for ck, Zk in zip(c, basis_full):
        bg += ck * Zk

    background = np.full_like(bg, np.nan)
    background[disk_mask] = bg[disk_mask]

    # Undo log if needed and compute corrected image
    if use_log:
        bgl = np.exp(background)
        denom = np.where(np.isfinite(bgl), np.maximum(bgl, 1e-6), 1.0)
        corrected = img / denom
        background_lin = bgl
    else:
        denom = np.where(np.isfinite(background), np.maximum(background, 1e-6), 1.0)
        corrected = img / denom
        background_lin = background

    # Normalize corrected to on-disk median ~1 (handy after your LD correction)
    med_on_disk = np.nanmedian(corrected[disk_mask])
    if np.isfinite(med_on_disk) and med_on_disk > 0:
        corrected /= med_on_disk

    return corrected, background_lin


def correct_limb_darkening_and_background(
    img: np.ndarray,
    centre: Tuple[int, int] = (1023, 1023),
    radius: int = 900,
    order: int = 4,
    sigma_clip: float = 3,
) -> np.ndarray:
    """
    Flatten Hα limb darkening and remove background.

    Parameters
    ----------
    img     : 2-D ndarray
    centre  : (row, col) tuple in px.  If None, found automatically.
    radius  : float px.  If None, found automatically.
    order   : polynomial order for background fit (default 4).
    sigma_clip: number of standard deviations for sigma clipping (default 3).

    Returns
    -------
    corrected : 2-D ndarray   (same shape as img)
    background: 2-D ndarray   (same shape as img)
    """
    # Correct limb darkening
    img_corr = correct_limb_darkening(img, centre, radius)

    # Remove background
    corrected, background = polynomial_background(img_corr, centre, radius, order, sigma_clip)

    return corrected