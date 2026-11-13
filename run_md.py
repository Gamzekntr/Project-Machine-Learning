"""
Issue #2 - Run a Müller-Brown simulation and save the samples as training data.

Run from the MS1 folder:
    .venv/bin/python run_md.py

Settings follow the course paper (Appendix C.4): kT = 23, dt = 0.005, mass = 0.5,
one frame kept every 50 steps. The paper does not give the friction; we use gamma = 1.
"""
from pathlib import Path

import numpy as np
import torch

from langevin import langevin_integrator, random_velocities
from mueller_brown import mueller_brown_force

# ---------------- Settings ----------------
KT = 23.0              # temperature (same energy units as the potential)
DT = 0.005             # time step
MASS = 0.5
GAMMA = 1.0            # friction
N_PARTICLES = 100      # independent particles simulated side by side
N_STEPS = 52_000       # steps per particle
SAVE_EVERY = 50        # keep 1 frame every 50 steps
BURN_IN_STEPS = 2_000  # first steps are thrown away (particles still "settling in")
SEED = 0

# All particles start in the deepest valley, so we can check if they find the others.
START_POINT = [-0.558, 1.442]

OUTPUT_FOLDER = Path(__file__).parent / "data"


def run_simulation():
    x0 = torch.tensor([START_POINT]).repeat(N_PARTICLES, 1)
    v0 = random_velocities(N_PARTICLES, 2, kT=KT, mass=MASS, seed=SEED)

    trajectory = langevin_integrator(
        x0, v0, mueller_brown_force,
        dt=DT, n_steps=N_STEPS,
        kT=KT, gamma=GAMMA, mass=MASS,
        save_every=SAVE_EVERY, seed=SEED,
    )                                            # shape (frames, particles, 2)

    frames_to_drop = BURN_IN_STEPS // SAVE_EVERY
    trajectory = trajectory[frames_to_drop:]     # remove burn-in
    return trajectory


def save_samples(trajectory):
    OUTPUT_FOLDER.mkdir(exist_ok=True)
    samples = trajectory.reshape(-1, 2)          # flatten to a simple list of (x, y) points

    settings = dict(kT=KT, dt=DT, mass=MASS, gamma=GAMMA, n_particles=N_PARTICLES,
                    n_steps=N_STEPS, save_every=SAVE_EVERY, burn_in_steps=BURN_IN_STEPS,
                    seed=SEED, start_point=START_POINT)

    torch.save({"samples": samples, "trajectory": trajectory, "settings": settings},
               OUTPUT_FOLDER / "mueller_brown_samples.pt")
    np.save(OUTPUT_FOLDER / "mueller_brown_samples.npy", samples.numpy())
    return samples


if __name__ == "__main__":
    trajectory = run_simulation()
    samples = save_samples(trajectory)
    print(f"Trajectory shape (frames, particles, 2): {tuple(trajectory.shape)}")
    print(f"Saved {len(samples):,} samples to {OUTPUT_FOLDER}")
