# Part 6 — Time-based mixture of experts (MoE)

**Paper sections:** §3.3 (Eq. 16), §5.2 "Advantages of mixture", App. C.2 (MoE vs. more parameters),
App. C.3 (runtime), Tables 3–6 (expert sizes)
**Notebook:** [06_mixture_of_experts.ipynb](06_mixture_of_experts.ipynb)
**Code:** [scoremd/moe.py](../scoremd/moe.py)

---

## Idea

Split diffusion time into disjoint intervals and give each its own network (Eq. 16):

$$ s_\theta(x,t) = s^\theta_i(x,t)\quad\text{for } t\in\mathcal I_i,\qquad \mathcal I = \{(0,0.1),\ [0.1,0.6),\ [0.6,1.0)\} $$

* **Simulation** only needs expert 0 (contains `t = 0`). You can even train *only* that one.
* **iid sampling** walks through all experts as `t` decreases.

## Why it helps

1. **Focus:** expert 0 sees only small `t` → all its capacity goes to the regime that matters for
   forces. Paper: Mixture improves simulation even *without* FP (Tables 10/11), and beats simply
   making one network wider/deeper (App. C.2, Table 7).
2. **No over-regularisation:** FP regularisation + conservative parameterisation are only applied to
   expert 0. Large-`t` experts are cheap unconstrained `ScoreNet`s → better iid quality and faster.
3. **Different mechanism than FP:** Fig. 6 shows Mixture improves consistency while its FP error at
   `t = 0` stays high → the two are complementary → **Both** (MoE + FP on expert 0) is the paper's
   final model ("Ours").
4. **Efficiency:** experts train independently/in parallel; only one is in memory at a time.

## The four variants used throughout the paper

| name | experts | FP regularisation | conservative |
|---|---|---|---|
| Diffusion | 1 on (0,1) | no | yes |
| Mixture | 3 | no | expert 0 only |
| Fokker–Planck | 1 on (0,1) | yes | yes |
| **Both** (= "Ours") | 3 | expert 0 only | expert 0 only |
| Two For One | = Diffusion, but simulated at `t = 0.02` | | |

## What the notebook does

1. Build `TimeMoE` from three experts and check the routing.
2. Train Mixture and Both on Müller–Brown (expert sizes from Table 9: `64,64,54` hidden, `3,2,2` layers).
3. Compare iid and simulation to the single-model variants from Part 5.
