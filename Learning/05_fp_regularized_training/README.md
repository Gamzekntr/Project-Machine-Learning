# Part 5 — The fix: training with Fokker–Planck regularisation

**Paper sections:** §3.1 "Fokker–Planck regularization" (Eq. 10), Fig. 1 bottom row, §5.1, Table 1
**Notebook:** [05_fp_regularized_training.ipynb](05_fp_regularized_training.ipynb)
**Code:** [scoremd/train.py](../scoremd/train.py)

---

## The objective (Eq. 10)

$$ \min_\theta\ \mathbb E_{t\sim\mathcal U(0,1)}\,\mathbb E_{x(0)}\,\mathbb E_{x(t)|x(0)}\Big[\mathcal L_{DSM}[\nabla_x\log p_\theta](x(t),t) + \alpha\cdot\mathcal L_{FP}[\log p_\theta](x(t),t)\Big] $$

* DSM still teaches the score at every `t`.
* The FP term ties `log p_θ` at different `t` together. Intuition from the paper: it **transfers
  accuracy from the stable large-`t` regime to the ill-conditioned small-`t` regime**, because
  `∂ₜ log p` at small `t` must agree with what the (well-learned) scores imply.
* `α` trades off iid-sample quality and consistency (Table 1: iid gets slightly worse, sim much better).

## Implementation choices (and where we deviate)

| item | paper | here |
|---|---|---|
| `λ(t)` for DSM | not fully specified | `s(t)²` (ε-prediction) |
| `λ_FP(t)` | "same as λ" | uniform `1` by default — with `λ = s(t)²` the FP term would vanish exactly at small `t` where it is needed; try `weight=lambda t: sde.marginal_std(t)**2` and compare! |
| residual estimator | product of two weak residuals | same (`unbiased=True`) |
| `σ`, `(h_s, h_d)` | `1e-4`, `(1e-3, 5e-4)` | same |
| `α` (Müller–Brown) | `5e-4` | see notebook; the right scale depends on `λ_FP` and the loss normalisation, so we sweep it |
| network | MLP, 3 layers × 92 | same |
| training | 180 epochs (~140k steps) | a few thousand steps (CPU budget) |

Cost: every FP step needs 2 × (2 score evals + 3 energy evals) extra forward/backward passes —
App. C.3 reports ~5× longer training. You'll see the same here.

## What the notebook does

1. Train an FP-regularised `EnergyNet` on the Fig. 1 mixture → reproduce the bottom row of Fig. 1.
2. Compare FP error vs. `t` with and without regularisation (Fig. 6).
3. On Müller–Brown: iid vs. Langevin simulation, mass in the small basin, PMF error — before and after.
4. α-sweep: the iid-vs-sim trade-off.
