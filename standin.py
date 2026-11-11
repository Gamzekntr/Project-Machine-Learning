"""
TEMPORARY STAND-IN for the teammates' modules (potential, force, Langevin,
load_data) so that model.py / viz.py can be developed and tested now.
Delete this file (and switch the import in run_toy.py) when the real ones exist.

Everything here is numpy; mueller_brown_* also accept torch tensors.
"""
import math
import numpy as np

# Mueller-Brown parameters (paper Eq. 34): U = sum_k A_k exp(g_k)
_A = [-200.0, -100.0, -170.0, 15.0]


def _terms(x, y):
    xp = _xp(x)
    g = [-(x - 1) ** 2 - 10 * y ** 2,
         -x ** 2 - 10 * (y - 0.5) ** 2,
         -6.5 * (x + 0.5) ** 2 + 11 * (x + 0.5) * (y - 1.5) - 6.5 * (y - 1.5) ** 2,
         0.7 * (x + 1) ** 2 + 0.6 * (x + 1) * (y - 1) + 0.7 * (y - 1) ** 2]
    dgx = [-2 * (x - 1), -2 * x, -13 * (x + 0.5) + 11 * (y - 1.5),
           1.4 * (x + 1) + 0.6 * (y - 1)]
    dgy = [-20 * y, -20 * (y - 0.5), 11 * (x + 0.5) - 13 * (y - 1.5),
           0.6 * (x + 1) + 1.4 * (y - 1)]
    return xp, g, dgx, dgy


def _xp(x):
    if type(x).__module__.startswith("torch"):
        import torch
        return torch
    return np


def mueller_brown_potential(x):
    """U(x), x: (N, 2) -> (N,)."""
    xp, g, _, _ = _terms(x[..., 0], x[..., 1])
    return sum(a * xp.exp(gk) for a, gk in zip(_A, g))


def mueller_brown_force(x):
    """ANALYTIC force F = -grad U, x: (N, 2) -> (N, 2)."""
    xp, g, dgx, dgy = _terms(x[..., 0], x[..., 1])
    e = [a * xp.exp(gk) for a, gk in zip(_A, g)]
    fx = -sum(ek * d for ek, d in zip(e, dgx))
    fy = -sum(ek * d for ek, d in zip(e, dgy))
    return xp.stack([fx, fy], -1)


def langevin_integrator(x0, v0, force_fn, dt, n_steps, kT=1.0, gamma=1.0, mass=1.0,
                        record_every=1, seed=0):
    """Underdamped Langevin (OpenMM-style 'velocity then position' update).
    Returns trajectory (n_records, N, dim)."""
    rng = np.random.default_rng(seed)
    x, v = np.array(x0, dtype=float), np.array(v0, dtype=float)
    a = math.exp(-gamma * dt)
    noise = math.sqrt(kT * (1 - a * a) / mass)
    traj = []
    for k in range(n_steps):
        v = a * v + (1 - a) / gamma * force_fn(x) / mass + noise * rng.normal(size=x.shape)
        x = x + dt * v
        if (k + 1) % record_every == 0:
            traj.append(x.copy())
    return np.stack(traj)


def load_data(dataset="toy", transformation=None, n_samples=100_000, batch_size=1024,
              kT=23.0, seed=0, **kw):
    """Generator yielding (batch, 2) torch tensors of Mueller-Brown samples."""
    import torch
    if dataset != "toy":
        raise NotImplementedError("stand-in only provides the toy system")
    n_chains, mass = 2000, 0.5
    rng = np.random.default_rng(seed)
    x0 = np.array([-1.5, -0.3]) + np.array([2.5, 2.3]) * rng.random((n_chains, 2))
    v0 = rng.normal(size=(n_chains, 2)) * math.sqrt(kT / mass)
    burn = langevin_integrator(x0, v0, mueller_brown_force, 0.005, 2000, kT, 1.0, mass,
                               record_every=2000, seed=seed)
    # continue from the burned-in state with fresh velocities
    traj = langevin_integrator(burn[-1], v0, mueller_brown_force, 0.005,
                               (n_samples // n_chains) * 50, kT, 1.0, mass,
                               record_every=50, seed=seed + 1)
    data = traj.reshape(-1, 2).astype(np.float32)
    data = data[rng.permutation(len(data))]
    if transformation is not None:
        data = transformation(data)
    data = torch.as_tensor(data)
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]
