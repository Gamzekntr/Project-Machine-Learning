"""
Visualisation helpers for the ScoreMD function.
All functions accept numpy arrays or torch tensors.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- style
PI = np.pi


def _np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


TICKS = [-PI, -PI / 2, 0, PI / 2, PI]
TICKLABELS = [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"]
COLORS = ["#1b1b1b", "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
LINESTYLES = ["-", "--", "--", "--", "--", "--"]


def _style():
    mpl.rcParams.update({
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
        "legend.frameon": False,
    })


_style()

def _cmap():
    cm = plt.get_cmap("jet").copy()
    cm.set_bad("white")            #Empty bins are white like the paper
    return cm
#2D free energy
def free_energy_2d(a, b, bins=64, ranges=((-PI, PI), (-PI, PI)), min_count=1):
    H, xe, ye = np.histogram2d(_np(a), _np(b), bins=bins, range=ranges)
    p = H / H.sum()
    with np.errstate(divide="ignore"):
        F = -np.log(p)
    F[H < min_count] = np.nan
    F -= np.nanmin(F)
    extent = (xe[0], xe[-1], ye[0], ye[-1])
    return F.T, extent             # transpose: rows = y, cols = x

#One Ramachandran free-energy map.
def plot_density(phi, psi, ax=None, title=None, vmax=None, bins=64,
                 colorbar=True, contours=False):

    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 3.8))
    F, ext = free_energy_2d(phi, psi, bins=bins)
    im = ax.imshow(F, origin="lower", extent=ext, cmap=_cmap(),
                   vmin=0, vmax=vmax, aspect="equal", interpolation="nearest")
    if contours:
        ax.contour(F, levels=np.arange(2, np.nanmax(F), 2), extent=ext,
                   colors="k", linewidths=0.4, alpha=0.5)
    ax.set_xticks(TICKS); ax.set_xticklabels(TICKLABELS)
    ax.set_yticks(TICKS); ax.set_yticklabels(TICKLABELS)
    ax.set_xlabel(r"$\varphi$"); ax.set_ylabel(r"$\psi$")
    if title:
        ax.set_title(title)
    if colorbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(r"Energy / $k_BT$")
    return ax, im


def plot_density_grid(datasets, ncols=None, vmax=None, bins=64, figsize_per=(2.6, 2.6),
                      suptitle=None, save=None):
    names = list(datasets)
    n = len(names)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, sharex=True, sharey=True,
                             figsize=(figsize_per[0] * ncols + 0.9, figsize_per[1] * nrows),
                             constrained_layout=True)
    if vmax is None:   # one global scale -> panels are comparable
        vmax = max(np.nanmax(free_energy_2d(*datasets[k], bins=bins)[0]) for k in names)
        vmax = min(vmax, 12.0)
    im = None
    for i, name in enumerate(names):
        ax = axes[i // ncols, i % ncols]
        _, im = plot_density(*datasets[name], ax=ax, title=name, vmax=vmax,
                             bins=bins, colorbar=False)
        if i % ncols:
            ax.set_ylabel("")
        if i // ncols < nrows - 1:
            ax.set_xlabel("")
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label(r"Energy / $k_BT$")
    if suptitle:
        fig.suptitle(suptitle)
    if save:
        fig.savefig(save)
    return fig, axes


#1D projections
def plot_projection_1d(datasets, which="phi", bins=100, ax=None, ylim=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 3.3))
    edges = np.linspace(-PI, PI, bins + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])
    for i, (name, ang) in enumerate(datasets.items()):
        h, _ = np.histogram(_np(ang), bins=edges)
        with np.errstate(divide="ignore"):
            F = -np.log(h / h.sum())
        F[h == 0] = np.nan
        F -= np.nanmin(F)
        ax.plot(mid, F, ls=LINESTYLES[i % 6], color=COLORS[i % 6],
                lw=2.2 if i == 0 else 1.6, label=name)
    ax.set_xticks(TICKS); ax.set_xticklabels(TICKLABELS)
    ax.set_xlim(-PI, PI)
    ax.set_xlabel(r"$\varphi$" if which == "phi" else r"$\psi$")
    ax.set_ylabel(r"Energy / $k_BT$")
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=9)
    return ax


#Bonds
def bond_lengths(beads, i=0, j=1):
    beads = _np(beads)
    return np.linalg.norm(beads[:, i] - beads[:, j], axis=-1)


def plot_bond_density(datasets, ax=None, xlabel=r"Length ($\AA$)", bins=120):
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 3.3))
    allv = np.concatenate([_np(v) for v in datasets.values()])
    lo, hi = np.percentile(allv, [0.1, 99.9])
    pad = 0.15 * (hi - lo)
    edges = np.linspace(lo - pad, hi + pad, bins + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])
    for i, (name, v) in enumerate(datasets.items()):
        h, _ = np.histogram(_np(v), bins=edges, density=True)
        ax.plot(mid, h, ls=LINESTYLES[i % 6], color=COLORS[i % 6],
                lw=2.2 if i == 0 else 1.6, label=name)
    ax.set_xlabel(xlabel); ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    return ax


#Geometry
BEAD_NAMES = ["C", "N", "CA", "C", "N"]
BEAD_ELEMENTS = ["C", "N", "C", "C", "N"]
BONDS = [(0, 1), (1, 2), (2, 3), (3, 4)]

#Coarse-grained beads (N,5,3).Trajectory with bonds.
def cg_trajectory(beads, scale_nm=1.0):

    import mdtraj as md
    top = md.Topology()
    chain = top.add_chain()
    res = top.add_residue("ALA", chain)
    atoms = [top.add_atom(n, md.element.get_by_symbol(e), res)
             for n, e in zip(BEAD_NAMES, BEAD_ELEMENTS)]
    for i, j in BONDS:
        top.add_bond(atoms[i], atoms[j])
    xyz = _np(beads).astype(np.float32) * scale_nm
    return md.Trajectory(xyz, top)


def view_molecule(obj, scale_nm=1.0, index=None, representation="ball+stick",
                  width="520px", height="380px"):
    import nglview as nv
    import mdtraj as md
    traj = obj if isinstance(obj, md.Trajectory) else cg_trajectory(obj, scale_nm)
    if index is not None:
        traj = traj[index]
    view = nv.show_mdtraj(traj, default=False)
    view.clear_representations()
    if representation == "ball+stick":
        view.add_representation("ball+stick", radius=0.12)
    else:
        view.add_representation(representation)
    view.background = "white"
    view._set_size(width, height)
    return view


def _dihedral_np(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1n).sum(-1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdims=True) * b1n
    return np.arctan2((np.cross(b1n, v) * w).sum(-1), (v * w).sum(-1))


def pick_basin_structures(beads, basins=None, seed=0):
    beads = _np(beads)
    phi = _dihedral_np(beads[:, 0], beads[:, 1], beads[:, 2], beads[:, 3])
    psi = _dihedral_np(beads[:, 1], beads[:, 2], beads[:, 3], beads[:, 4])
    if basins is None:
        basins = {"C7eq (beta-like)": ((-2.5, -0.5), (1.0, 3.0)),
                  "alpha-R": ((-2.0, -0.5), (-1.2, 0.6)),
                  "alpha-L (rare)": ((0.5, 2.0), (-0.5, 1.5))}
    rng = np.random.default_rng(seed)
    out = {}
    for name, ((p0, p1), (s0, s1)) in basins.items():
        ids = np.where((phi > p0) & (phi < p1) & (psi > s0) & (psi < s1))[0]
        out[name] = int(rng.choice(ids)) if len(ids) else None
    return out


def plot_geometries_mpl(beads_list, titles=None, scale_nm=1.0, figsize_per=(2.6, 2.6),
                        save=None):
    colors = {"C": "#555555", "N": "#3050F8"}
    n = len(beads_list)
    fig = plt.figure(figsize=(figsize_per[0] * n, figsize_per[1]))
    for k, x in enumerate(beads_list):
        ax = fig.add_subplot(1, n, k + 1, projection="3d")
        x = _np(x) * scale_nm
        x = x - x.mean(0)
        for i, j in BONDS:
            ax.plot(*zip(x[i], x[j]), color="#999999", lw=3, solid_capstyle="round")
        for i in range(5):
            ax.scatter(*x[i], s=260, color=colors[BEAD_ELEMENTS[i]],
                       edgecolor="k", linewidth=0.6, depthshade=True)
            ax.text(*x[i], f" {BEAD_NAMES[i]}", fontsize=8)
        r = np.abs(x).max() * 1.1
        ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
        if titles:
            ax.set_title(titles[k], fontsize=10)
    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig

#2D coordinate densities
def plot_density_xy(samples, ax=None, extent=None, title=None, vmax=10.0, bins=64,
                    potential_fn=None, kT=1.0, contour_levels=None, colorbar=True,
                    xlabel="$x$", ylabel="$y$"):
    s = _np(samples)
    if extent is None:
        extent = (s[:, 0].min(), s[:, 0].max(), s[:, 1].min(), s[:, 1].max())
    if ax is None:
        _, ax = plt.subplots(figsize=(4.4, 3.8))
    F, ext = free_energy_2d(s[:, 0], s[:, 1], bins=bins,
                            ranges=((extent[0], extent[1]), (extent[2], extent[3])))
    im = ax.imshow(F, origin="lower", extent=ext, cmap=_cmap(), vmin=0, vmax=vmax,
                   aspect="equal", interpolation="nearest")
    if potential_fn is not None:
        _overlay_contours(ax, potential_fn, kT, extent, contour_levels)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if colorbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(r"Energy / $k_BT$")
    return ax, im

def _overlay_contours(ax, potential_fn, kT, extent, levels=None, n=200):
    gx = np.linspace(extent[0], extent[1], n)
    gy = np.linspace(extent[2], extent[3], n)
    grid = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)
    try:
        U = potential_fn(grid)
    except Exception:                       #Potential written for torch only
        import torch
        U = potential_fn(torch.as_tensor(grid, dtype=torch.float32))
    U = _np(U).reshape(n, n) / kT
    U = U - U.min()
    ax.contour(gx, gy, U, levels=levels if levels is not None else np.arange(2, 20, 2),
               colors="k", linewidths=0.4, alpha=0.5)

#Compare several 2D datasets
def plot_density_xy_grid(datasets, extent=None, ncols=None, vmax=10.0, bins=64,
                         potential_fn=None, kT=1.0, figsize_per=(3.0, 2.8),
                         suptitle=None, save=None):
    names = list(datasets)
    n = len(names)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))
    if extent is None:                      # common window covering all datasets
        allv = np.concatenate([_np(v) for v in datasets.values()])
        lo, hi = np.percentile(allv, 0.2, axis=0), np.percentile(allv, 99.8, axis=0)
        pad = 0.08 * (hi - lo)
        extent = (lo[0] - pad[0], hi[0] + pad[0], lo[1] - pad[1], hi[1] + pad[1])
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, sharex=True, sharey=True,
                             figsize=(figsize_per[0] * ncols + 0.9, figsize_per[1] * nrows),
                             constrained_layout=True)
    im = None
    for i, name in enumerate(names):
        ax = axes[i // ncols, i % ncols]
        _, im = plot_density_xy(datasets[name], ax=ax, extent=extent, title=name,
                                vmax=vmax, bins=bins, potential_fn=potential_fn,
                                kT=kT, colorbar=False)
        if i % ncols:
            ax.set_ylabel("")
        if i // ncols < nrows - 1:
            ax.set_xlabel("")
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label(r"Energy / $k_BT$")
    if suptitle:
        fig.suptitle(suptitle)
    if save:
        fig.savefig(save)
    return fig, axes


def energy_grid(energy_fn, extent, n=200, batch=20000):
    gx = np.linspace(extent[0], extent[1], n)
    gy = np.linspace(extent[2], extent[3], n)
    pts = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2).astype(np.float32)
    out = []
    for i in range(0, len(pts), batch):
        out.append(_np(energy_fn(pts[i:i + batch])).reshape(-1))
    E = np.concatenate(out).reshape(n, n)
    return E - E.min()


def plot_energy_map(E, extent, ax=None, title=None, vmax=10.0, potential_fn=None,
                    kT=1.0, colorbar=True, xlabel="$x$", ylabel="$y$"):
    if ax is None:
        _, ax = plt.subplots(figsize=(4.4, 3.8))
    im = ax.imshow(_np(E), origin="lower", extent=extent, cmap=_cmap(), vmin=0,
                   vmax=vmax, aspect="equal", interpolation="nearest")
    if potential_fn is not None:
        _overlay_contours(ax, potential_fn, kT, extent)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if colorbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(r"Energy / $k_BT$")
    return ax, im


#Dihedral time series
def plot_dihedral_timeseries(phi, psi, step=1, unit="frame", max_points=5000,
                             axes=None, title=None):
    phi, psi = _np(phi), _np(psi)
    if len(phi) > max_points:
        sl = slice(0, max_points)
        phi, psi = phi[sl], psi[sl]
    t = np.arange(len(phi)) * step
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(7, 3.8), sharex=True, constrained_layout=True)
    for ax, y, lab, c in zip(axes, (phi, psi), (r"$\varphi$", r"$\psi$"),
                             (COLORS[1], COLORS[2])):
        ax.plot(t, y, ".", ms=1.5, color=c, rasterized=True)
        ax.set_yticks(TICKS); ax.set_yticklabels(TICKLABELS)
        ax.set_ylim(-PI, PI); ax.set_ylabel(lab)
    axes[-1].set_xlabel(unit)
    if title:
        axes[0].set_title(title)
    return axes


#Full-atom geometry (static)
ALDP_ELEMENTS = ["H", "C", "H", "H", "C", "O", "N", "H", "C", "H", "C", "H", "H", "H",
                 "C", "O", "N", "H", "C", "H", "H", "H"]
_COV_RADIUS = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66}       # Angstrom
_CPK = {"H": "#f2f2f2", "C": "#4d4d4d", "N": "#3050f8", "O": "#e8231b"}
_DRAW_SIZE = {"H": 70, "C": 190, "N": 200, "O": 200}

#Bonds from interatomic distances (covalent radii). positions in Angstrom.
def guess_bonds(positions_A, elements, tol=1.25):
    n = len(elements)
    bonds = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions_A[i] - positions_A[j])
            if d < tol * (_COV_RADIUS[elements[i]] + _COV_RADIUS[elements[j]]):
                bonds.append((i, j))
    return bonds


def plot_molecule_mpl(positions, elements=None, bonds=None, units="nm", titles=None,
                      ncols=None, figsize_per=(2.8, 2.8), view=(20, 35), save=None):
    P = _np(positions).astype(float)
    if P.ndim == 2:
        P = P[None]
    P = P * (10.0 if units == "nm" else 1.0)            # -> Angstrom
    elements = elements or ALDP_ELEMENTS
    if len(elements) != P.shape[1]:
        raise ValueError(f"{len(elements)} elements but {P.shape[1]} atoms")
    n = P.shape[0]
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows))
    for k in range(n):
        ax = fig.add_subplot(nrows, ncols, k + 1, projection="3d")
        x = P[k] - P[k].mean(0)
        bl = bonds if bonds is not None else guess_bonds(x, elements)
        for i, j in bl:
            ax.plot(*zip(x[i], x[j]), color="#9a9a9a", lw=2.2, solid_capstyle="round")
        for el in sorted(set(elements)):
            idx = [i for i, e in enumerate(elements) if e == el]
            ax.scatter(x[idx, 0], x[idx, 1], x[idx, 2], s=_DRAW_SIZE[el], c=_CPK[el],
                       edgecolor="k", linewidth=0.5, depthshade=True)
        r = np.abs(x).max() * 1.05
        ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(*view)
        if titles:
            ax.set_title(titles[k], fontsize=10)
    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig
