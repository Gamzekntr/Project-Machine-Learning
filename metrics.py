"""Metrics between generated samples and reference samples.

JS distance and PMF error follow App. B.2 of the paper: 64 x 64 bins,
JS with one extra count per bin, PMF with 1e-6 added to every bin proportion.
"""
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

ANGLE_RANGE = ((-np.pi, np.pi), (-np.pi, np.pi))


def _np(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def histogram_2d(samples, bins=64, range=ANGLE_RANGE):
    s = _np(samples)
    counts, _, _ = np.histogram2d(s[:, 0], s[:, 1], bins=bins, range=range)
    return counts


# ---------------------------------------------------------------- on histograms
def js_distance_counts(counts_p, counts_q, pseudocount=1.0):
    p = np.ravel(counts_p).astype(np.float64) + pseudocount
    q = np.ravel(counts_q).astype(np.float64) + pseudocount
    return float(jensenshannon(p / p.sum(), q / q.sum()))


def kl_divergence_counts(counts_p, counts_q, eps=1e-10):
    """KL(p || q)."""
    p = np.ravel(counts_p).astype(np.float64)
    q = np.ravel(counts_q).astype(np.float64)
    p, q = p / p.sum() + eps, q / q.sum() + eps
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def pmf_error_counts(counts_p, counts_q, eps=1e-6):
    p = np.ravel(counts_p).astype(np.float64)
    q = np.ravel(counts_q).astype(np.float64)
    p = p / p.sum() + eps
    q = q / q.sum() + eps
    return float(np.mean((np.log(p) - np.log(q)) ** 2))


# ---------------------------------------------------------------- on samples
def js_distance(samples_p, samples_q, bins=64, range=ANGLE_RANGE, pseudocount=1.0):
    return js_distance_counts(histogram_2d(samples_p, bins, range),
                              histogram_2d(samples_q, bins, range), pseudocount)


def kl_divergence(samples_p, samples_q, bins=64, range=ANGLE_RANGE, eps=1e-10):
    return kl_divergence_counts(histogram_2d(samples_p, bins, range),
                                histogram_2d(samples_q, bins, range), eps)


def pmf_error(samples_p, samples_q, bins=64, range=ANGLE_RANGE, eps=1e-6):
    return pmf_error_counts(histogram_2d(samples_p, bins, range),
                            histogram_2d(samples_q, bins, range), eps)


# ---------------------------------------------------------------- Wasserstein
def wasserstein_1d(x, y):
    return float(wasserstein_distance(_np(x).ravel(), _np(y).ravel()))


def circular_wasserstein_1d(theta_x, theta_y, n_grid=3600):
    """W1 for angles in radians, where -pi and pi are the same point.

    W1 = min_c  integral |F - G - c|, the minimum is at c = median(F - G).
    """
    edges = np.linspace(-np.pi, np.pi, n_grid + 1)
    tx, ty = _np(theta_x).ravel(), _np(theta_y).ravel()
    fx = np.cumsum(np.histogram(tx, edges)[0]) / len(tx)
    fy = np.cumsum(np.histogram(ty, edges)[0]) / len(ty)
    d = fx - fy
    return float(np.sum(np.abs(d - np.median(d))) * (2 * np.pi / n_grid))


def marginal_wasserstein_angles(angles_p, angles_q):
    """(W1 of phi, W1 of psi) in radians."""
    a, b = _np(angles_p), _np(angles_q)
    return (circular_wasserstein_1d(a[:, 0], b[:, 0]),
            circular_wasserstein_1d(a[:, 1], b[:, 1]))


# ---------------------------------------------------------------- exact reference (Mueller-Brown)
def boltzmann_bin_probabilities(potential_fn, range, bins=64, kT=1.0, oversample=8):
    """Probability of each bin under p(x) ~ exp(-U(x) / kT), integrated on a finer grid."""
    (x0, x1), (y0, y1) = range
    n = bins * oversample
    xs = x0 + (np.arange(n) + 0.5) * (x1 - x0) / n
    ys = y0 + (np.arange(n) + 0.5) * (y1 - y0) / n
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], axis=-1)
    try:
        U = potential_fn(pts)
    except Exception:  # potentials written for torch tensors
        import torch
        U = potential_fn(torch.as_tensor(pts, dtype=torch.float64))
    U = _np(U).reshape(n, n).astype(np.float64)
    w = np.exp(-(U - U.min()) / kT)
    probs = w.reshape(bins, oversample, bins, oversample).sum(axis=(1, 3))
    return probs / probs.sum()
