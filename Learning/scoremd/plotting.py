"""Plot helpers: free-energy images like the paper's figures (-log density in units of kT)."""
import numpy as np
import matplotlib.pyplot as plt
from .metrics import hist2d, _to_np


def free_energy_hist(ax, samples, rng, bins=100, vmax=8, title=None, cmap="turbo"):
    """Plot F = -log p (shifted to min 0) of a 2D sample set; empty bins are left white."""
    H = hist2d(samples, bins, rng).T
    with np.errstate(divide="ignore"):
        F = -np.log(H / H.sum())
    F -= np.nanmin(F[np.isfinite(F)])
    F[~np.isfinite(F)] = np.nan
    im = ax.imshow(F, origin="lower", extent=[rng[0][0], rng[0][1], rng[1][0], rng[1][1]],
                   aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    return im


def density_image(ax, X, Y, log_density, title=None, cmap="viridis"):
    ax.imshow(np.exp(log_density), origin="lower", extent=[X.min(), X.max(), Y.min(), Y.max()],
              aspect="auto", cmap=cmap)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)


def scatter(ax, samples, title=None, s=1, alpha=0.3, **kw):
    s_ = _to_np(samples)
    ax.scatter(s_[:, 0], s_[:, 1], s=s, alpha=alpha, **kw)
    if title:
        ax.set_title(title, fontsize=10)
