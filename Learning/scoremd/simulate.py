"""Langevin dynamics (Eq. 5) with the OpenMM-style integrator of App. B.1 (Eq. 33).

    v <- a v + (1 - a)/gamma * F / M + sqrt(kT (1 - a^2) / M) R,   a = exp(-gamma dt)
    x <- x + dt v

With a learned model the force is F = kT * s_theta(x, t~0)  (score-force relation, Eq. 6).
"""
import math
import torch


def langevin(force_fn, x0, n_steps, dt, gamma=1.0, kT=1.0, mass=1.0, save_every=10,
             generator=None, max_abs=1e3):
    """Run parallel underdamped Langevin chains. x0: (n_chains, D).
    Returns trajectory tensor of shape (n_saved, n_chains, D). Chains whose coordinates blow up
    beyond `max_abs` are frozen (and reported) so one unstable chain cannot produce NaNs."""
    x = x0.clone()
    v = math.sqrt(kT / mass) * torch.randn(x.shape, generator=generator, dtype=x.dtype)
    a = math.exp(-gamma * dt)
    noise_scale = math.sqrt(kT * (1 - a ** 2) / mass)
    alive = torch.ones(x.shape[0], dtype=torch.bool)
    frames = []
    for step in range(n_steps):
        F = force_fn(x)
        R = torch.randn(x.shape, generator=generator, dtype=x.dtype)
        v_new = a * v + (1 - a) / gamma * F / mass + noise_scale * R
        x_new = x + dt * v_new
        ok = torch.isfinite(x_new).all(-1) & (x_new.abs() < max_abs).all(-1) & alive
        x = torch.where(ok[:, None], x_new, x)
        v = torch.where(ok[:, None], v_new, v)
        alive = ok
        if (step + 1) % save_every == 0:
            frames.append(x.clone())
    if (~alive).any():
        print(f"[langevin] warning: {int((~alive).sum())}/{len(alive)} chains became unstable and were frozen")
    return torch.stack(frames)


def model_force(model, t_eval=1e-5, kT=1.0):
    """Force from a diffusion model via the score-force relation: F(x) = kT * s_theta(x, t_eval)."""
    def F(x):
        t = torch.full((x.shape[0],), t_eval, dtype=x.dtype)
        return kT * model.score(x, t, create_graph=False)
    return F
