"""
Milestone 1 prototype on the Mueller-Brown toy system:
python run_toy.py --epochs 20       #Quick run
python run_toy.py                   #Default run
load_data('toy'),muller_brown_potential(x),muller_brown_force(x) are taken from teammates' code
until those exist, `standin.py` provides them.
"""
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

try:                                                    # teammates' implementation
    from data import load_data, muller_brown_potential, muller_brown_force
except ImportError:                                     # temporary stand-in
    from standin import load_data, muller_brown_potential, muller_brown_force
    print("note: using standin.py (teammates' `data` module not found)")

import viz
from model import (Model, Normalizer, collect, train, sample_iid, check_units,
                   pmf_error, T_EPS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--n-steps", type=int, default=1000, help="sampler steps")
    p.add_argument("--n-gen", type=int, default=50_000, help="number of generated samples")
    p.add_argument("--kT", type=float, default=23.0,
                   help="temperature used to generate the data (units of the potential)")
    p.add_argument("--out", default="toy_results.png")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    print("device:", dev)

    # ---- data (physical units) -> normalised
    x_phys = collect(load_data("toy"))
    norm = Normalizer(x_phys)
    data = norm.to_norm(x_phys).to(dev)
    print(f"data: {tuple(x_phys.shape)}, scale = {norm.scale:.3f}")

    # Training
    model = Model(2, a.hidden, a.layers).to(dev)
    losses = train(model, data, epochs=a.epochs, batch_size=a.batch)

    # Sampling
    gen = norm.to_phys(sample_iid(model, a.n_gen, a.n_steps, dev, a.seed)).cpu()

    # Evaluation
    ref_np, gen_np = x_phys.numpy(), gen.numpy()
    lo, hi = np.percentile(ref_np, 0.1, axis=0), np.percentile(ref_np, 99.9, axis=0)
    pad = 0.1 * (hi - lo)
    extent = (lo[0] - pad[0], hi[0] + pad[0], lo[1] - pad[1], hi[1] + pad[1])
    ranges = ((extent[0], extent[1]), (extent[2], extent[3]))
    half = len(ref_np) // 2
    print(f"PMF error generated vs reference : {pmf_error(ref_np, gen_np, ranges):.3f}")
    print(f"PMF error noise floor (half/half): {pmf_error(ref_np[:half], ref_np[half:], ranges):.3f}")
    chk = check_units(model, norm, x_phys[:5000].to(dev), a.kT,
                      muller_brown_potential, muller_brown_force)
    print("units check vs analytic potential/force:",
          {k: round(v, 3) for k, v in chk.items()})

    # Plots
    @torch.no_grad()
    def energy_fn(pts):
        x = torch.as_tensor(pts, dtype=torch.float32, device=dev)
        return model.energy(norm.to_norm(x), T_EPS)

    E = viz.energy_grid(energy_fn, extent, n=200)
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.7), constrained_layout=True)
    for ax, (name, s) in zip(axs[:2], {"Reference (Langevin)": ref_np,
                                       "Generated (reverse SDE)": gen_np}.items()):
        _, im = viz.plot_density_xy(s, ax=ax, extent=extent, title=name, colorbar=False,
                                    potential_fn=muller_brown_potential, kT=a.kT)
    viz.plot_energy_map(E, extent, ax=axs[2], colorbar=False,
                        title=r"Learned energy $E_\theta(x, t\to 0)$",
                        potential_fn=muller_brown_potential, kT=a.kT)
    fig.colorbar(im, ax=axs[:3], fraction=0.02, pad=0.02).set_label(r"Energy / $k_BT$")
    axs[3].plot(np.arange(1, len(losses) + 1), losses, color=viz.COLORS[1])
    axs[3].set_xlabel("epoch"); axs[3].set_ylabel("DSM loss")
    axs[3].set_yscale("log"); axs[3].set_title("Training")
    fig.savefig(a.out)
    print("saved", a.out)
    torch.save({"model": model.state_dict(), "mu": norm.mu, "scale": norm.scale,
                "args": vars(a)}, "toy_model.pt")


if __name__ == "__main__":
    main()
