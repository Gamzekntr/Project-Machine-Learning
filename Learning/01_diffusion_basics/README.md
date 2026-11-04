# Part 1 — Diffusion models in a nutshell

**Paper sections:** §2.1 (Generative score-based modeling), Appendix A.1 (VP SDE)
**Notebook:** [01_diffusion_basics.ipynb](01_diffusion_basics.ipynb)
**Code:** [scoremd/sde.py](../scoremd/sde.py), [scoremd/losses.py](../scoremd/losses.py) (`dsm_loss`), [scoremd/sampling.py](../scoremd/sampling.py)

---

## 1. The forward (noising) process

A diffusion model slowly destroys data with Gaussian noise. "Diffusion time" `t` runs from
`t = 0` (clean data) to `t = 1` (pure noise). This is an SDE (Eq. 1):

$$ dx = f(x,t)\,dt + g(t)\,dw $$

The paper uses the **variance-preserving (VP) SDE** of Song et al. (2021) (App. A.1):

$$ f(x,t) = -\tfrac12\beta(t)\,x,\qquad g(t)=\sqrt{\beta(t)},\qquad \beta(t)=\beta_{\min}+t(\beta_{\max}-\beta_{\min}),\ (\beta_{\min},\beta_{\max})=(0.1,20) $$

Because the drift is linear, the transition kernel is a Gaussian we can write down (Eq. 19–20):

$$ p_t(x(t)\mid x(0)) = \mathcal N\big(m(t)\,x(0),\ s(t)^2 I\big),\quad m(t)=e^{-\frac12\int_0^t\beta},\quad s(t)^2 = 1-e^{-\int_0^t\beta} $$

So we can jump to any `t` in one step: `x_t = m(t) x_0 + s(t) ε`. At `t = 1`, `m ≈ 0` and `s ≈ 1`:
the data has become `N(0, I)`. **This is why the data must be normalised to unit variance** (App. A.1).

> Notation note: the paper writes `σ(t)` for what we call `s(t)` (the *standard deviation*). Eq. 20
> writes `(1 - e^{...}) I` — that is the *variance*.

## 2. The reverse (generative) process

Anderson's theorem gives the reverse-time SDE (Eq. 2):

$$ dx = \big[f(x,t) - g(t)^2\,\nabla_x\log p_t(x)\big]dt + g(t)\,d\bar w $$

Everything is known except the **score** `∇ₓ log p_t(x)`. If a network `s_θ(x,t)` learns it, we can
start from noise and integrate backwards to `t ≈ 0` → an **iid sample** from the data distribution.
We use simple Euler–Maruyama:

```
x ← x + [½β(t)x + β(t) s_θ(x,t)] Δt + √(β(t)Δt) z
```

## 3. Denoising score matching (DSM)

We cannot regress on the unknown `∇ log p_t`, but we *can* regress on the score of the Gaussian kernel,
`∇ log p_{0t}(x_t | x_0) = -ε / s(t)` (Eq. 3 → Eq. 4). Vincent (2011) showed the minimiser is the
true marginal score. With weight `λ(t) = s(t)²` the loss becomes the familiar

$$ \mathcal L_{\text{DSM}} = \mathbb E_{t,x_0,\varepsilon}\,\big\| s(t)\,s_\theta(x_t,t) + \varepsilon \big\|^2 $$

## 4. Why small `t` is hard (the key motivation of the whole paper)

As `t → 0`, `s(t) → 0`, the regression target `-ε/s(t)` explodes and becomes extremely noisy.
Practitioners truncate training to `t ∈ (ε, 1)`. For **sampling** that's fine (small errors at the
very last step barely matter). But the paper wants to use the score **at `t = 0`** as a *force field*
(Part 2) — and that is exactly where the model is least accurate.

## What the notebook does

1. Visualise `m(t)`, `s(t)`, `β(t)` and the forward noising of a 2D dataset.
2. Train a small diffusion model (an `EnergyNet`, introduced properly in Part 2) with DSM.
3. Generate iid samples with the reverse SDE and compare to the data.
4. Plot how noisy the DSM target is as a function of `t` (motivation for Part 4).

## Check yourself
- Why must `m(t)² + s(t)² = 1` in the VP SDE? (Hint: variance of `x_t` if `x_0` has unit variance.)
- What happens to sample quality if you stop the reverse SDE at `t = 0.05` instead of `1e-5`?
