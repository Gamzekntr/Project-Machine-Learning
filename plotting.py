"""Plots for the report: free-energy maps, Ramachandran plot, time series, molecules."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

import analysis

CMAP = "viridis"
PI = np.pi
TICKS = [-PI, -PI / 2, 0, PI / 2, PI]
TICKLABELS = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]
ANGLE_RANGE = [[-PI, PI], [-PI, PI]]
ELEMENT_COLORS = {"C": "#505050", "N": "#3050F8", "O": "#E8261D", "H": "#E8E8E8"}
ELEMENT_SIZES = {"C": 70, "N": 70, "O": 70, "H": 28}


def set_style():
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "figure.dpi": 150, "savefig.bbox": "tight", "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def free_energy_2d(x, y, bins=64, range=None):
    """-log p in units of kT with minimum 0; NaN for empty bins."""
    counts, xe, ye = np.histogram2d(np.ravel(x), np.ravel(y), bins=bins, range=range)
    with np.errstate(divide="ignore"):
        F = -np.log(counts / counts.sum())
    F[~np.isfinite(F)] = np.nan
    F -= np.nanmin(F)
    return F, xe, ye


def plot_free_energy_2d(ax, x, y, bins=64, range=None, vmax=None, cmap=CMAP,
                        colorbar=True, cax=None, xlabel="$x$", ylabel="$y$"):
    F, xe, ye = free_energy_2d(x, y, bins, range)
    im = ax.pcolormesh(xe, ye, F.T, cmap=cmap, vmin=0, vmax=vmax, shading="flat", rasterized=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if colorbar:
        cb = plt.colorbar(im, cax=cax) if cax is not None else plt.colorbar(im, ax=ax, pad=0.02)
        cb.set_label(r"Free energy / $k_\mathrm{B}T$")
    return im


def _angle_axis(ax, which="both"):
    if which in ("x", "both"):
        ax.set_xticks(TICKS)
        ax.set_xticklabels(TICKLABELS)
        ax.set_xlim(-PI, PI)
    if which in ("y", "both"):
        ax.set_yticks(TICKS)
        ax.set_yticklabels(TICKLABELS)
        ax.set_ylim(-PI, PI)


def draw_state_regions(ax, color="white", lw=0.8):
    kw = dict(color=color, lw=lw, ls="--", alpha=0.9)
    r = np.radians
    ax.axvline(0, **kw)
    for psi in analysis.PSI_ALPHA_R:
        ax.plot([-PI, 0], [r(psi)] * 2, **kw)
    for psi in analysis.PSI_ALPHA_L:
        ax.plot([0, PI], [r(psi)] * 2, **kw)


def label_states(ax, phi, psi, states, fontsize=8):
    """Label each state next to its most populated bin."""
    for k, name in enumerate(analysis.STATE_NAMES):
        mask = states == k
        if mask.sum() < 10:
            continue
        H, xe, ye = np.histogram2d(phi[mask], psi[mask], bins=36, range=ANGLE_RANGE)
        i, j = np.unravel_index(np.argmax(H), H.shape)
        cx, cy = 0.5 * (xe[i] + xe[i + 1]), 0.5 * (ye[j] + ye[j + 1])
        tx = cx + (0.5 if cx < 1.0 else -0.5)
        ty = cy + (-0.52 if cy > 1.0 else 0.52)
        ax.annotate(analysis.STATE_LABELS[name], (cx, cy), xytext=(tx, ty),
                    fontsize=fontsize, color="black", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc=to_rgba("white", 0.85), ec="none"),
                    arrowprops=dict(arrowstyle="-", color="white", lw=0.8))


def plot_ramachandran(ax, phi, psi, bins=72, vmax=None, show_states=True, colorbar=True, cax=None):
    phi, psi = np.asarray(phi), np.asarray(psi)
    im = plot_free_energy_2d(ax, phi, psi, bins=bins, range=ANGLE_RANGE, vmax=vmax,
                             colorbar=colorbar, cax=cax, xlabel=r"$\phi$", ylabel=r"$\psi$")
    _angle_axis(ax)
    ax.set_aspect("equal")
    if show_states:
        states = analysis.assign_states(phi, psi)
        draw_state_regions(ax)
        label_states(ax, phi, psi, states)
    return im


def plot_angle_timeseries(ax, t_ns, angle, label, color="C0", s=0.3):
    # dots instead of lines, since the angle jumps between -pi and pi
    ax.scatter(t_ns, angle, s=s, color=color, linewidths=0, rasterized=True)
    ax.set_ylabel(label)
    _angle_axis(ax, "y")


def plot_free_energy_1d(ax, angle, bins=72, label=None, color="C0", **kw):
    counts, edges = np.histogram(angle, bins=bins, range=(-PI, PI))
    with np.errstate(divide="ignore"):
        F = -np.log(counts / counts.sum())
    F -= F[np.isfinite(F)].min()
    ax.plot(0.5 * (edges[:-1] + edges[1:]), F, label=label, color=color, **kw)
    _angle_axis(ax, "x")
    ax.set_ylabel(r"Free energy / $k_\mathrm{B}T$")


def infer_bonds(xyz, elements, cutoff_heavy=0.17, cutoff_h=0.125):
    """Bonds from distances in nm."""
    bonds = []
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            cut = cutoff_h if "H" in (elements[i], elements[j]) else cutoff_heavy
            if np.linalg.norm(xyz[i] - xyz[j]) < cut:
                bonds.append((i, j))
    return np.array(bonds)


def plot_molecule(ax, xyz, elements, bonds=None, highlight=None, view=(20, -60)):
    """Ball-and-stick plot of one frame (nm) on a 3D axis; `highlight` circles some atoms."""
    xyz = np.asarray(xyz) * 10.0
    xyz = xyz - xyz.mean(axis=0)
    if bonds is None or len(bonds) == 0:
        bonds = infer_bonds(xyz / 10.0, elements)
    for i, j in bonds:
        ax.plot(*zip(xyz[i], xyz[j]), color="#9a9a9a", lw=1.6, zorder=1)
    for el in ("H", "C", "N", "O"):
        idx = [k for k, e in enumerate(elements) if e == el]
        if idx:
            ax.scatter(*xyz[idx].T, s=ELEMENT_SIZES[el], color=ELEMENT_COLORS[el],
                       edgecolors="#333333", linewidths=0.4, depthshade=False, zorder=2)
    if highlight is not None:
        ax.scatter(*xyz[list(highlight)].T, s=150, facecolors="none", edgecolors="#F2B701",
                   linewidths=1.2, depthshade=False, zorder=3)
    ax.view_init(*view)
    lim = np.abs(xyz).max()
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    try:
        ax.set_box_aspect((1, 1, 1), zoom=1.45)
    except TypeError:
        ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
