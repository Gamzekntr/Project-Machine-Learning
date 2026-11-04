# Part 3 — The problem: diffusion sampling and simulation disagree

**Paper sections:** §1 (Fig. 1), §3 intro (Eq. 7), §5.1 "(In-)consistent sampling", §4 (Two For One)
**Notebook:** [03_inconsistency.ipynb](03_inconsistency.ipynb)

---

## The two densities of one model (Eq. 7)

$$ p(x) \sim \text{Denoising}[s_\theta](x,t) \qquad\text{vs.}\qquad p_0(x)\propto \exp\big(\log p_\theta(x,0)\big) $$

* `p(x)`: what you get by running the reverse SDE from noise (iid sampling).
* `p₀(x)`: the density *implied by the energy at t = 0* — this is what Langevin simulation with the
  learned force converges to.

For a perfect model they are identical. **In practice they are not** — even for a 2D mixture of two
Gaussians (Fig. 1): iid sampling reproduces both modes with the right weights (1/5 vs 4/5), but the
learned density at `t = 0` can show a spurious third mode and the wrong mass ratio.

## Why can iid sampling be fine while `t = 0` is wrong?

The reverse SDE spends almost all of its time at moderate/large `t`, where the score is learned well.
The last tiny stretch near `t = 0` only "polishes" the sample, so small score errors there barely move
it. A Langevin simulation, by contrast, lives **only** at `t ≈ 0`, for millions of steps: every error
in the force accumulates into a wrong stationary distribution.

## The existing fix: *Two For One* (Arts et al., 2023)

Evaluate the force at a larger diffusion time, e.g. `t = 0.02`, where the score is more reliable.
But `s_θ(x, 0.02)` is the score of the **noised** distribution `p_{0.02}` — the simulated structures
are blurred (bond lengths too broad: Fig. 3 & Fig. 12 in the paper). We reproduce this effect too.

## How we measure consistency

* **Learned-density image** (as in Fig. 1): evaluate `log p_θ(x, 1e-5)` on a grid, normalise, plot.
* **Mode masses**: fraction of probability in each mode for data / iid / `p₀`.
* **PMF error / JS divergence** between histograms (App. B.2) — see Part 7.

## What the notebook does

1. Train a plain energy-based diffusion model (DSM only) on the Fig. 1 Gaussian mixture.
2. Plot reference / iid samples / learned density at `t = 0` side by side (top row of Fig. 1).
3. Compare to the *exact* `log p_t` of the mixture (we know it in closed form!) as a function of `t` —
   showing that the error is concentrated at small `t`.
4. On Müller–Brown: iid vs Langevin with `t = 1e-5` vs *Two For One* (`t = 0.02`).

> Note: whether you see the *third mode* exactly as in Fig. 1 depends on the seed/training budget.
> The mismatch in mass and the error concentrated at small `t` should be visible regardless.
