# Part 4 — The Fokker–Planck equation and how to compute its residual cheaply

**Paper sections:** §3.1 (Eq. 8–9, 11–14), App. A.2 (proof of the weak estimator), A.2.1 (relation to
Hutchinson), A.3 (finite differences), Fig. 6 / Fig. 11 (FP error vs. `t`)
**Notebook:** [04_fokker_planck.ipynb](04_fokker_planck.ipynb)
**Code:** [scoremd/losses.py](../scoremd/losses.py) (`fp_residual_exact`, `fp_residual_weak`, `fp_loss`)

---

## 1. The log-density Fokker–Planck equation (Eq. 8)

If `x` follows the forward SDE `dx = f dt + g dw`, its density obeys the Fokker–Planck PDE. Written
for `log p_t` (divide by `p_t` and use `∇p/p = ∇ log p`):

$$ \partial_t \log p_t = \underbrace{\tfrac12 g^2\big[\operatorname{div}(\nabla\log p_t) + \|\nabla\log p_t\|^2\big] - \langle f,\nabla\log p_t\rangle - \operatorname{div} f}_{\mathcal F_p(x,t)} $$

**Derivation sketch.** FP: `∂ₜp = -div(f p) + ½g² Δp`. Divide by `p`:
`-div(f p)/p = -div f - ⟨f, ∇log p⟩` and `Δp/p = Δ log p + ‖∇ log p‖²`. Done.

For VP: `f = -½βx`, `g² = β`, `div f = -½βD`.

This links the energy `log p_t` at *all* diffusion times. Any exact model satisfies it. A model that
only matches scores pointwise via DSM need not — and the paper (and Lai et al. 2023) find the
violation is **largest near `t = 0`** (Fig. 6).

## 2. The residual and the loss (Eq. 9)

$$ R(x,t) = \mathcal F_{p_\theta}(x,t) - \partial_t \log p_\theta(x,t),\qquad \mathcal L_{FP} = \lambda_{FP}(t)\,D^{-2}\,\|R\|^2 $$

Note this needs `log p_θ` itself (for `∂ₜ log p`), not just the score → **energy-based model required**.

## 3. Making it cheap

Computing `div(∇ log p)` = trace of the Hessian costs `D` backward passes. Two tricks:

**(a) Weak residual + antithetic divergence estimator (Eq. 11–12, App. A.2).**
Define `R̃(x,t) = E_v[R(x+v,t)]`, `v ~ N(0,σ²I)`. Integration by parts against the Gaussian gives

$$ \mathbb E_v[\operatorname{div} s(x+v)] = \mathbb E_v\Big[\big(\tfrac{v}{\sigma}\big)^{\!\top}\tfrac{s(x+v)-s(x-v)}{2\sigma}\Big] $$

→ only two score evaluations (first derivatives). A single `v` gives an **unbiased** but noisy
estimate. The paper uses `σ = 1e-4`. (We use float64 in the toys so this finite difference is accurate.)

**(b) Unbiased squared residual (Eq. 13).** `E[R̃(v)]² ≠ E[R̃(v)²]` (the variance adds bias), but for
two independent draws `E[R̃(v)·R̃(v')] = (E R̃)²`. So the loss uses the product of two residuals.

**(c) Finite differences in `t` (Eq. 14, App. A.3):** with `(h_s, h_d) = (1e-3, 5e-4)`,

$$ \partial_t\log p \approx \frac{h_s^2\log p(t+h_d) + (h_d^2-h_s^2)\log p(t) - h_d^2\log p(t-h_s)}{h_s h_d (h_s+h_d)} $$

## 4. Relation to Hutchinson (App. A.2.1)

`vᵀ J v / σ²` is Hutchinson's trace estimator. The difference: the paper perturbs **all** terms of
the residual with the same `v`, so if the strong residual is exactly 0, the weak one is too —
no truncation error, for any `σ`.

## What the notebook does

1. **Unit test:** the exact diffused Gaussian mixture has `R ≡ 0` (to machine precision) — this
   validates our implementation of Eq. 8.
2. Show the weak estimator is unbiased (average over many `v` → exact value) and plot its variance.
3. Check the finite-difference `∂ₜ log p` against autograd.
4. Measure the FP error `‖R(x,t)‖` of the DSM-only model from Part 3 as a function of `t`
   (reproduces the shape of Fig. 6: largest at small `t`).
