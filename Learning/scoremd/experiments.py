"""Standard recipes shared by the notebooks: datasets, model configs, cached training, evaluation.

Every notebook calls `get_model(name)`: it loads `results/<name>_seed<k>.pt` if it exists and
otherwise trains the model with the config below (and caches it). NOTE: the cache key is the model name + overrides, not the full config:
after editing CONFIGS, delete the matching .pt files (or pass
retrain=True) to retrain. Change STEPS_SCALE to train longer/shorter everywhere.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import metrics, models, potentials, sampling, simulate
from .moe import TimeMoE
from .train import train

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

STEPS_SCALE = 1.0      # multiply all training steps (e.g. 5.0 to get closer to the paper's budget)
T_EVAL = 1e-5          # diffusion time used for simulation (App. B.3)
T_TWO_FOR_ONE = 0.02   # Two For One baseline (App. B.3)
MOE_BOUNDS = [0.1, 0.6]


# ----------------------------------------------------------------------------------------------
# Datasets (returned already normalised; `info` holds the normalisation)
# ----------------------------------------------------------------------------------------------
def load_system(system, n=100_000, seed=0):
    if system == "gmm":
        g = potentials.GaussianMixture(**potentials.FIG1_GMM)
        torch.manual_seed(seed)
        raw = g.sample(n)
        shift = raw.mean(0)
        scale = raw.std()          # one scalar -> isotropic, keeps the analytic density simple
        x = (raw - shift) / scale
        return x, dict(system="gmm", shift=shift, scale=scale, gmm=g)
    if system == "mb":
        path = RESULTS / "mueller_brown_data.pt"
        if path.exists():
            raw = torch.load(path).to(torch.get_default_dtype())
        else:
            print("Generating Mueller-Brown data with Langevin MD (~1 min, cached afterwards) ...")
            raw = potentials.generate_mueller_brown_data(n_samples=n, seed=seed)
            torch.save(raw, path)
        shift, scale = raw.mean(0), raw.std(0)
        return (raw - shift) / scale, dict(system="mb", shift=shift, scale=scale)
    raise ValueError(system)


def mb_true_force_normalised(info, kT=23.0):
    """Exact force for the *normalised* Mueller-Brown coordinates, in units of kT: grad log p."""
    shift, scale = info["shift"], info["scale"]
    return lambda xn: potentials.mueller_brown_force(xn * scale + shift) * scale / kT


# ----------------------------------------------------------------------------------------------
# Model configurations (Table 9 for the toy systems)
# ----------------------------------------------------------------------------------------------
CONFIGS = {
    # --- Figure 1 Gaussian mixture ---
    "gmm_diffusion": dict(system="gmm", kind="single", hidden=92, layers=3, steps=4000, alpha=0.0),
    "gmm_fp":        dict(system="gmm", kind="single", hidden=92, layers=3, steps=4000, alpha=0.01),
    # --- Mueller-Brown (App. C.4) ---
    "mb_diffusion":  dict(system="mb", kind="single", hidden=92, layers=3, steps=4000, alpha=0.0),
    "mb_fp":         dict(system="mb", kind="single", hidden=92, layers=3, steps=4000, alpha=0.5),
    "mb_mixture":    dict(system="mb", kind="moe", hidden=[64, 64, 54], layers=[3, 2, 2],
                          steps=[3000, 750, 750], alpha=[0.0, 0.0, 0.0]),
    "mb_both":       dict(system="mb", kind="moe", hidden=[64, 64, 54], layers=[3, 2, 2],
                          steps=[3000, 750, 750], alpha=[0.15, 0.0, 0.0]),
}


def build_model(cfg):
    if cfg["kind"] == "single":
        return models.EnergyNet(2, cfg["hidden"], cfg["layers"])
    # MoE: only the first (small-t) expert is energy-based; the others are plain score networks
    experts = [models.EnergyNet(2, cfg["hidden"][0], cfg["layers"][0])]
    experts += [models.ScoreNet(2, h, l) for h, l in zip(cfg["hidden"][1:], cfg["layers"][1:])]
    return TimeMoE(experts, MOE_BOUNDS)


def _train_cfg(model, cfg, x, seed, verbose):
    lr = cfg.get("lr", 1e-3)
    if cfg["kind"] == "single":
        return train(model, x, steps=int(cfg["steps"] * STEPS_SCALE), lr=lr, fp_alpha=cfg["alpha"],
                     seed=seed, verbose=verbose)
    bounds = [1e-5] + MOE_BOUNDS + [1.0]
    hist = []
    for i, expert in enumerate(model.experts):
        if verbose:
            print(f"-- expert {i}: t in [{bounds[i]}, {bounds[i + 1]})  alpha={cfg['alpha'][i]}")
        hist.append(train(expert, x, steps=int(cfg["steps"][i] * STEPS_SCALE), lr=lr,
                          t_range=(bounds[i], bounds[i + 1]), fp_alpha=cfg["alpha"][i],
                          seed=seed + 17 * i, verbose=verbose))
    return hist


def get_model(name, seed=0, retrain=False, verbose=True, **overrides):
    """Load a cached model or train it. `overrides` modify the config (and the cache file name)."""
    cfg = dict(CONFIGS[name], **overrides)
    tag = name + "".join(f"_{k}{v}" for k, v in sorted(overrides.items())).replace(" ", "")
    path = RESULTS / f"{tag}_seed{seed}.pt"
    x, info = load_system(cfg["system"])
    torch.manual_seed(seed)
    model = build_model(cfg)
    if path.exists() and not retrain:
        model.load_state_dict(torch.load(path))
        if verbose:
            print(f"loaded {path.name}")
    else:
        t0 = time.time()
        _train_cfg(model, cfg, x, seed, verbose)
        torch.save(model.state_dict(), path)
        if verbose:
            print(f"trained {tag} (seed {seed}) in {time.time() - t0:.0f}s -> {path.name}")
    model.eval()
    return model


# ----------------------------------------------------------------------------------------------
# Sampling / simulation / evaluation
# ----------------------------------------------------------------------------------------------
def iid_samples(model, n=20_000, steps=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    return sampling.sample_reverse_sde(model, n, 2, steps=steps, generator=g)


def simulate_model(model, x_data, t_eval=T_EVAL, n_chains=100, n_steps=10_000, dt=0.05, gamma=1.0,
                   save_every=50, burn_in_frac=0.1, seed=0, force=None):
    """Langevin MD (normalised coordinates, kT = 1, M = 1) with the learned force kT * s(x, t_eval).
    Chains start from random training configurations (as in the paper)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(x_data), (n_chains,), generator=g)
    F = force or simulate.model_force(model, t_eval=t_eval)
    traj = simulate.langevin(F, x_data[idx], n_steps=n_steps, dt=dt, gamma=gamma,
                             save_every=save_every, generator=g)
    keep = traj[int(burn_in_frac * len(traj)):]
    return keep.reshape(-1, 2), traj


def compare(samples, ref, n=None, bins=64, seed=0):
    """PMF error and JS divergence with *equal* sample sizes (subsample both to n)."""
    rng_np = np.random.default_rng(seed)
    n = n or min(len(samples), len(ref))
    s = samples[rng_np.choice(len(samples), n, replace=False)]
    r = ref[rng_np.choice(len(ref), n, replace=False)]
    rng = metrics.hist_range(ref)
    return dict(pmf=metrics.pmf_error(s, r, bins, rng), js=metrics.js_divergence(s, r, bins, rng))


def mb_small_basin_fraction(samples, info):
    """Fraction of samples in the right-hand (least populated) Mueller-Brown basin (x > 0.3)."""
    thr = (0.3 - info["shift"][0]) / info["scale"][0]
    s = samples if torch.is_tensor(samples) else torch.as_tensor(samples)
    return float((s[:, 0] > thr).double().mean())


def save_json(obj, name):
    with open(RESULTS / name, "w") as f:
        json.dump(obj, f, indent=2, default=float)
