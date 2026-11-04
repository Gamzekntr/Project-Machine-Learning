"""Evaluation metrics (App. B.2): PMF error and Jensen-Shannon divergence on 2D histograms,
plus helpers to turn an energy model into a density on a grid ('learned density', Fig. 1)."""
import numpy as np
import torch
from scipy.spatial.distance import jensenshannon


def _to_np(a):
    return a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)


def hist_range(ref, pad=0.0):
    ref = _to_np(ref)
    lo, hi = ref.min(0), ref.max(0)
    w = hi - lo
    return [[lo[0] - pad * w[0], hi[0] + pad * w[0]], [lo[1] - pad * w[1], hi[1] + pad * w[1]]]


def hist2d(samples, bins, rng):
    s = _to_np(samples)
    H, _, _ = np.histogram2d(s[:, 0], s[:, 1], bins=bins, range=rng)
    return H


def pmf_error(samples, ref, bins=64, rng=None, eps=1e-6):
    """Mean over bins of (log p - log q)^2 with p, q bin proportions (+1e-6). Durumeric et al. 2024."""
    rng = rng or hist_range(ref)
    p = hist2d(samples, bins, rng); p = p / p.sum() + eps
    q = hist2d(ref, bins, rng); q = q / q.sum() + eps
    return float(np.mean((np.log(p) - np.log(q)) ** 2))


def js_divergence(samples, ref, bins=64, rng=None):
    """JS divergence between flattened 2D histograms; +1 pseudo-count per bin (App. B.2).
    scipy returns the JS *distance* (sqrt); we square it to get the divergence."""
    rng = rng or hist_range(ref)
    p = hist2d(samples, bins, rng).ravel() + 1
    q = hist2d(ref, bins, rng).ravel() + 1
    return float(jensenshannon(p / p.sum(), q / q.sum()) ** 2)


@torch.no_grad()
def grid_log_density(log_p_fn, rng, n=200, t=1e-5):
    """Evaluate log p(x, t) on an n x n grid and normalise numerically.
    Returns (X, Y, log_density) with the density integrating to ~1 over the grid."""
    xs = torch.linspace(rng[0][0], rng[0][1], n)
    ys = torch.linspace(rng[1][0], rng[1][1], n)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    pts = torch.stack([X.ravel(), Y.ravel()], -1)
    tt = torch.full((pts.shape[0],), t)
    lp = torch.cat([log_p_fn(c, tt[: len(c)]) for c in pts.split(10_000)])
    cell = (xs[1] - xs[0]) * (ys[1] - ys[0])
    lp = lp - torch.logsumexp(lp, 0) - torch.log(cell)
    return X.numpy(), Y.numpy(), lp.reshape(n, n).numpy()


def samples_from_grid(X, Y, log_density, n, seed=0):
    """Draw approximate samples from a gridded density (to compare p0 with iid samples by PMF)."""
    rng = np.random.default_rng(seed)
    p = np.exp(log_density - log_density.max()).ravel()
    p /= p.sum()
    idx = rng.choice(p.size, size=n, p=p)
    dx = X[0, 1] - X[0, 0]; dy = Y[1, 0] - Y[0, 0]
    pts = np.stack([X.ravel()[idx], Y.ravel()[idx]], -1)
    return pts + rng.uniform(-0.5, 0.5, size=pts.shape) * np.array([dx, dy])
