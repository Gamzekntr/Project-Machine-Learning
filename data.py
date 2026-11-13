"""Loading the alanine dipeptide dataset (bgmol) and the Mueller-Brown samples.

    from data import load_data
    for x in load_data("aldp"):   # (1024, 22, 3), nm
        ...
    for x in load_data("toy"):    # (1024, 2), samples from run_md.py
        ...

Train/val/test are consecutive blocks of the trajectory (80/10/10 %),
because neighbouring MD frames are strongly correlated.
"""
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

try:
    from mueller_brown import mueller_brown_force as muller_brown_force
    from mueller_brown import mueller_brown_potential as muller_brown_potential
except ImportError:
    muller_brown_potential = muller_brown_force = None

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = Path(os.environ.get("PML_DATA", Path.home() / "pml-data"))
DEFAULT_MB_PATH = Path(os.environ.get("MB_SAMPLES", HERE / "data" / "mueller_brown_samples.npy"))
MB_NAMES = ("toy", "mb", "mueller_brown", "muller_brown")
DEFAULT_FRACTIONS = (0.8, 0.1, 0.1)
DEFAULT_GAP = 1000  # frames dropped between the splits (1 ns)
SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------- alanine dipeptide
def _cache_dir(root):
    return Path(root) / "AImplicitUnconstrained" / "cache"


def _bgmol_dataset(root, read):
    import bgmol_compat  # noqa: F401
    from bgmol.datasets import AImplicitUnconstrained

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    have_file = (root / "AImplicitUnconstrained" / "traj0.h5").is_file()
    return AImplicitUnconstrained(root=str(root), download=not have_file, read=read)


def _read_with_bgmol(root, with_forces):
    from openmm import unit

    ds = _bgmol_dataset(root, read=True)
    traj = ds.trajectory
    top = traj.topology
    uc = traj.unitcell_lengths
    integrator = ds.integrator
    out = dict(
        xyz=np.asarray(traj.xyz, dtype=np.float32),
        time=np.asarray(traj.time, dtype=np.float64),
        energies=np.asarray(ds.energies, dtype=np.float32).reshape(-1),
        atom_names=np.array([a.name for a in top.atoms]),
        residue_names=np.array([a.residue.name for a in top.atoms]),
        elements=np.array([a.element.symbol for a in top.atoms]),
        bonds=np.array([[b[0].index, b[1].index] for b in top.bonds], dtype=np.int64).reshape(-1, 2),
        temperature=np.array(ds.temperature, dtype=np.float64),
        box=np.asarray(uc[0] if uc is not None else [np.nan] * 3, dtype=np.float64),
        box_constant=np.array(uc is None or bool(np.allclose(uc, uc[0]))),
        uses_pbc=np.array(bool(ds.system.system.usesPeriodicBoundaryConditions())),
        timestep_fs=np.array(integrator.getStepSize().value_in_unit(unit.femtosecond)),
        friction_per_ps=np.array(integrator.getFriction().value_in_unit(unit.picosecond ** -1)),
    )
    if with_forces:
        out["forces"] = np.asarray(ds.forces, dtype=np.float32)
    return out


def _write_cache(arrays, cache):
    cache.mkdir(parents=True, exist_ok=True)
    for key in ("xyz", "time", "energies", "forces"):
        if key in arrays:
            np.save(cache / f"{key}.npy", arrays[key])
    meta = ("atom_names", "residue_names", "elements", "bonds", "temperature",
            "box", "box_constant", "uses_pbc", "timestep_fs", "friction_per_ps")
    np.savez(cache / "topology.npz", **{k: arrays[k] for k in meta if k in arrays})


@lru_cache(maxsize=2)
def load_aldp_arrays(root=DEFAULT_ROOT, with_forces=False, mmap=True):
    """Alanine dipeptide data as numpy arrays.

    The first call downloads/reads the data with bgmol and saves it as .npy files;
    later calls load these (memory-mapped).
    Keys: xyz (N, 22, 3) nm, time (N,) ps, energies (N,) kJ/mol,
    forces (N, 22, 3) kJ/(mol nm) if with_forces, plus topology and simulation info.
    """
    cache = _cache_dir(root)
    need = ["xyz.npy", "time.npy", "energies.npy", "topology.npz"]
    if with_forces:
        need.append("forces.npy")
    if not all((cache / f).is_file() for f in need):
        _write_cache(_read_with_bgmol(root, with_forces), cache)

    mode = "r" if mmap else None
    out = {k: np.load(cache / f"{k}.npy", mmap_mode=mode) for k in ("xyz", "time", "energies")}
    if with_forces:
        out["forces"] = np.load(cache / "forces.npy", mmap_mode=mode)
    with np.load(cache / "topology.npz") as top:
        for k in top.files:
            out[k] = top[k].item() if top[k].ndim == 0 else top[k]
    return out


def load_aldp_force_field(root=DEFAULT_ROOT):
    """bgmol system (Amber ff99SB-ILDN + OBC implicit solvent) and temperature in K.

    system.system is the openmm.System, system.topology the OpenMM topology.
    """
    ds = _bgmol_dataset(root, read=False)
    return ds.system, float(ds.temperature)


class AldpEnergy:
    """Energies (kJ/mol) and forces (kJ/(mol nm)) of the force field for positions in nm.

        U, F = AldpEnergy()(x)   # x: (N, 22, 3)
    """

    def __init__(self, root=DEFAULT_ROOT, platform="CPU"):
        import openmm as mm
        from openmm import unit

        system, self.temperature = load_aldp_force_field(root)
        self.unit = unit
        self.context = mm.Context(system.system, mm.VerletIntegrator(0.001),
                                  mm.Platform.getPlatformByName(platform))

    def __call__(self, positions):
        if torch.is_tensor(positions):
            positions = positions.detach().cpu().numpy()
        x = np.asarray(positions, dtype=np.float64)
        if x.shape[-1] != 3:
            x = x.reshape(len(x), -1, 3)
        x = x.reshape(-1, x.shape[-2], 3)

        u = self.unit
        energies = np.empty(len(x))
        forces = np.empty_like(x)
        for i, frame in enumerate(x):
            self.context.setPositions(frame)
            state = self.context.getState(getEnergy=True, getForces=True)
            energies[i] = state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
            forces[i] = state.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole / u.nanometer)
        return energies, forces


# ---------------------------------------------------------------- Mueller-Brown
def load_mb_arrays(path=DEFAULT_MB_PATH):
    """Mueller-Brown samples of shape (N, 2) from a .npy, .npz or .pt file.

    A trajectory of shape (steps, particles, 2) is flattened particle by particle.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no Mueller-Brown samples at {path} (run run_md.py or pass mb_path=...)")
    keys = ("x", "samples", "trajectory", "positions")
    if path.suffix == ".npy":
        x = np.load(path)
    elif path.suffix == ".npz":
        with np.load(path) as f:
            x = f[next((k for k in keys if k in f.files), f.files[0])]
    elif path.suffix in (".pt", ".pth"):
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            obj = next(obj[k] for k in keys if k in obj)
        x = obj.detach().cpu().numpy() if torch.is_tensor(obj) else np.asarray(obj)
    else:
        raise ValueError(f"unsupported file type: {path.suffix}")

    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3:
        x = np.transpose(x, (1, 0, 2)).reshape(-1, x.shape[-1])
    if x.ndim != 2 or x.shape[-1] != 2:
        raise ValueError(f"expected shape (N, 2), got {x.shape}")
    return x


# ---------------------------------------------------------------- splits
def split_indices(n, fractions=DEFAULT_FRACTIONS, gap=DEFAULT_GAP):
    """Indices of consecutive train/val/test blocks, with `gap` frames left out between them."""
    fractions = np.asarray(fractions, dtype=float) / np.sum(fractions)
    edges = np.concatenate([[0], np.round(np.cumsum(fractions) * n).astype(int)])
    out = {}
    for i, name in enumerate(SPLITS):
        start = edges[i] + (gap if i > 0 else 0)
        stop = edges[i + 1]
        if stop <= start:
            raise ValueError(f"split '{name}' is empty (n={n}, gap={gap})")
        out[name] = np.arange(start, stop)
    return out


def get_split(dataset="aldp", split="train", n_samples=None, seed=0, root=DEFAULT_ROOT,
              mb_path=DEFAULT_MB_PATH, fractions=DEFAULT_FRACTIONS, gap=None):
    """One split as a numpy array; n_samples draws a random subset of it."""
    if split not in SPLITS + ("all",):
        raise ValueError(f"split must be one of {SPLITS + ('all',)}")
    if dataset == "aldp":
        x = load_aldp_arrays(root)["xyz"]
        gap = DEFAULT_GAP if gap is None else gap
    elif dataset in MB_NAMES:
        x = load_mb_arrays(mb_path)
        gap = max(1, len(x) // 1000) if gap is None else gap
    else:
        raise ValueError(f"unknown dataset '{dataset}', use 'aldp' or 'toy'")

    idx = np.arange(len(x)) if split == "all" else split_indices(len(x), fractions, gap)[split]
    if n_samples is not None and n_samples < len(idx):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=n_samples, replace=False))
    return np.asarray(x[idx])


# ---------------------------------------------------------------- generator
def load_data(dataset="aldp", transformation=None, split="train", batch_size=1024,
              shuffle=True, n_epochs=1, n_samples=None, seed=0, device="cpu",
              dtype=torch.float32, root=DEFAULT_ROOT, mb_path=DEFAULT_MB_PATH,
              fractions=DEFAULT_FRACTIONS, gap=None, drop_last=False):
    """Generator over samples of a dataset.

    dataset:        'aldp' (positions (22, 3) in nm) or 'toy' (Mueller-Brown points (2,))
    transformation: function applied to every batch, None = raw data
    split:          'train', 'val', 'test' or 'all'
    batch_size:     None yields single samples
    n_epochs:       passes over the data, None = endless
    n_samples:      use a random subset of this size (the paper uses 50k)
    """
    data = get_split(dataset, split, n_samples=n_samples, seed=seed, root=root,
                     mb_path=mb_path, fractions=fractions, gap=gap)
    n = len(data)

    def generator():
        rng = np.random.default_rng(seed + 1)
        epoch = 0
        while n_epochs is None or epoch < n_epochs:
            order = rng.permutation(n) if shuffle else np.arange(n)
            if batch_size is None:
                for i in order:
                    x = torch.as_tensor(data[i:i + 1], dtype=dtype, device=device)
                    if transformation is not None:
                        x = transformation(x)
                    yield x[0]
            else:
                stop = n - n % batch_size if drop_last else n
                for start in range(0, stop, batch_size):
                    x = torch.as_tensor(data[order[start:start + batch_size]], dtype=dtype, device=device)
                    if transformation is not None:
                        x = transformation(x)
                    yield x
            epoch += 1

    return generator()


if __name__ == "__main__":
    arrays = load_aldp_arrays()
    print("positions:", arrays["xyz"].shape, arrays["xyz"].dtype)
    print("time step:", float(arrays["time"][1] - arrays["time"][0]), "ps")
    for name, idx in split_indices(len(arrays["xyz"])).items():
        print(f"{name:5s}: frames {idx[0]}..{idx[-1]} ({len(idx)})")
    print("first batch:", tuple(next(load_data("aldp", batch_size=8)).shape))
