# Part 2 — The score is a force: Boltzmann distribution & Langevin dynamics

**Paper sections:** §2.2 (Boltzmann distribution, Eq. 5–6), §3.2 (conservative parameterisation),
App. B.1 (Langevin integrator), App. C.4 (Müller–Brown dataset), App. C.1 (conservative vs. score)
**Notebook:** [02_score_force_boltzmann.ipynb](02_score_force_boltzmann.ipynb)
**Code:** [scoremd/potentials.py](../scoremd/potentials.py), [scoremd/simulate.py](../scoremd/simulate.py), [scoremd/models.py](../scoremd/models.py)

---

## 1. Molecular dynamics with Langevin dynamics (Eq. 5)

$$ dx = v\,dt,\qquad M\,dv = -\nabla_x U(x)\,dt - \gamma M v\,dt + \sqrt{2\gamma k_BT}\,dw_t $$

Here `t` is **physical time** (not diffusion time!). Run long enough, positions are distributed
according to the **Boltzmann distribution**

$$ p(x) = \frac{1}{Z}\exp\!\Big(-\frac{U(x)}{k_BT}\Big). $$

Note the stationary distribution does **not** depend on mass `M` or friction `γ` — those only change
the *dynamics* (how fast you move between states). We use this freedom in the notebooks.

Integrator (App. B.1, OpenMM style):
```
a = exp(-γ Δt)
v ← a v + (1-a)/γ · F/M + sqrt(kT (1-a²)/M) · R
x ← x + Δt v
```

## 2. The score–force relation (Eq. 6)

Take the log of the Boltzmann density and differentiate:

$$ \nabla_x \log p(x) = -\frac{\nabla_x U(x)}{k_BT} - \underbrace{\nabla_x \log Z}_{=0} = \frac{F(x)}{k_BT}. $$

So **the score at diffusion time `t = 0` is the force (divided by kT)**. A diffusion model trained
only on *samples* from an MD simulation therefore implicitly learns a force field — no force labels
needed. This is huge for **coarse-graining (CG)**, where the true CG forces are unknown (App. A.5:
the CG density is a marginal of the all-atom Boltzmann distribution).

One model → two uses (Eq. 7):

| use | what we evaluate | gives |
|---|---|---|
| **iid sampling** | `s_θ(x, t)` for all `t ∈ (0,1)`, reverse SDE | `p(x)` — independent samples |
| **simulation** | `s_θ(x, t≈0)` as a force in Langevin dynamics | `p₀(x) ∝ exp(log p_θ(x,0))` — trajectories, kinetics |

In practice the paper evaluates at `t = 1e-5` (App. B.3).

## 3. Conservative parameterisation (§3.2, App. C.1)

A real force is the gradient of an energy (curl-free). Two choices:

* `ScoreNet`: `s_θ(x,t) = NN(x,t) ∈ R^D` — standard, but no energy, forces may be non-conservative.
* `EnergyNet`: `log p_θ(x,t) = NN(x,t) ∈ R`, `s_θ = ∇ₓ NN` via autograd — conservative, gives the energy.

The paper uses the energy version for the small-`t` model because (i) the Fokker–Planck loss of
Part 4 needs `log p_θ` itself, and (ii) simulations with non-conservative score models were unstable
(App. C.1, Fig. 7).

## 4. Müller–Brown toy system (App. C.4, Eq. 34)

A classic 2D potential with three minima. Training data: Langevin MD with `kT = 23`, `Δt = 0.005`,
`M = 0.5`, every 50th frame kept → 100k samples. (Friction `γ` is not stated; we use `γ = 1`.
We run 100 chains in parallel instead of one long chain.)

## What the notebook does

1. Plot the Müller–Brown potential and generate the training data (cached in `results/`).
2. Verify numerically that `∇ log p = F / kT` for the *true* potential, and that Langevin with the true
   force reproduces the data histogram (the reference "simulation").
3. Show that the stationary distribution doesn't depend on `γ` or `M`.
4. Compare `EnergyNet` and `ScoreNet` outputs: the curl of a learned score field.

## Check yourself
- Why is `∇ₓ log Z = 0`?
- If you train on normalised data `x̃ = (x - μ)/σ`, how do you convert the learned score back to a force
  in the original units? (Answer: `F(x) = kT · s̃(x̃)/σ`.)
