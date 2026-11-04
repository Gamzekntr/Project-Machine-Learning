# Recreating "Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models"

Plainer, Wu, Klein, Günnemann, Noé — NeurIPS 2025 — [arXiv:2506.17139](https://arxiv.org/abs/2506.17139) — paper PDF: [PML.pdf](PML.pdf)
Official code: <https://github.com/noegroup/ScoreMD>

This folder is a **step-by-step study and re-implementation** of the paper. Each part has a `README.md`
(theory, equations, mapping to the paper) and a notebook (runnable experiment). All shared code is in
[scoremd/](scoremd/), written to be read alongside the paper.

## The paper in 5 sentences

1. A diffusion model trained on Boltzmann samples learns the score `∇log p_t(x)`; at diffusion time `t = 0`
   that score equals `force / kT` — so one model could do **iid sampling** *and* **molecular dynamics**.
2. In practice the two disagree: iid samples are right, but MD with the learned force converges to a wrong
   distribution, because the score is badly learned at very small `t`.
3. The authors show models violate the **Fokker–Planck equation** (which ties `log p_t` across `t`), most strongly
   near `t = 0`, and add an FP residual as a **regulariser** (made cheap with a weak-form estimator + finite differences).
4. A **time-based mixture of experts** lets them put all capacity, the energy parameterisation and the FP loss
   only on the small-`t` expert used for simulation.
5. Result: consistent sampling and simulation for Müller–Brown, alanine dipeptide, Chignolin, BBA and a
   transferable dipeptide model, without any force labels.

## Roadmap

| # | folder | paper | you will learn / reproduce |
|---|---|---|---|
| 1 | [01_diffusion_basics](01_diffusion_basics/) | §2.1, A.1 | VP SDE, denoising score matching, reverse-SDE sampling, why small `t` is hard |
| 2 | [02_score_force_boltzmann](02_score_force_boltzmann/) | §2.2, §3.2, B.1, C.1, C.4 | Boltzmann distribution, Langevin MD, score = force, conservative models |
| 3 | [03_inconsistency](03_inconsistency/) | §1 Fig. 1, Eq. 7, §5.1 | the iid-vs-simulation inconsistency; Two For One baseline |
| 4 | [04_fokker_planck](04_fokker_planck/) | §3.1, A.2, A.3, Fig. 6 | log-density Fokker–Planck eq., weak residual, unbiased estimator |
| 5 | [05_fp_regularized_training](05_fp_regularized_training/) | Eq. 10, Fig. 1, Table 1 | training with DSM + α·FP; consistency restored |
| 6 | [06_mixture_of_experts](06_mixture_of_experts/) | §3.3, C.2, C.3 | time-based MoE; "Mixture" and "Both" |
| 7 | [07_mueller_brown_benchmark](07_mueller_brown_benchmark/) | C.4, Table 10, Fig. 8 | full toy benchmark, PMF / JS metrics |
| 8 | [08_towards_molecules](08_towards_molecules/) | §3.2 Eq. 15, §5, A.5, B.3–B.6 | graph-transformer energy model, coarse-graining, alanine dipeptide pipeline |

Work through them in order — later notebooks reuse models cached by earlier ones (but each one will train what it
needs if run on its own).

## Setup

```bash
pip install -r requirements.txt
jupyter lab            # open 01_diffusion_basics/01_diffusion_basics.ipynb
```
Everything in Parts 1–7 runs on a laptop CPU (float64, small MLPs). Trained models, data and figures go to
[results/](results/). Delete a `results/*.pt` file to retrain that model.

## Code map ([scoremd/](scoremd/))

| file | contents | paper |
|---|---|---|
| `sde.py` | VP SDE: β(t), m(t), s(t), drift, perturbation | Eq. 1, 17–20 |
| `models.py` | `EnergyNet` (conservative), `ScoreNet`, `AnalyticModel` | §3.2 |
| `losses.py` | `dsm_loss`, exact & weak FP residual, `fp_loss`, finite-difference ∂ₜ | Eq. 3–4, 8–14, 30 |
| `train.py` | training loop for `L_DSM + α L_FP` | Eq. 10 |
| `sampling.py` | reverse-SDE Euler–Maruyama | Eq. 2 |
| `simulate.py` | Langevin integrator, score→force | Eq. 5–6, 33 |
| `moe.py` | `TimeMoE` | Eq. 16 |
| `metrics.py` | PMF error, JS divergence, gridded learned density | App. B.2 |
| `potentials.py` | Fig. 1 Gaussian mixture (with exact diffused density), Müller–Brown | Fig. 1, Eq. 34 |
| `molecules.py` | graph-transformer energy model, rotations, dihedrals, Ala2 CG | Eq. 15, B.3 |
| `experiments.py` | configs (Table 9), cached `get_model`, simulation & evaluation helpers | App. C.4 |

## Where this re-implementation differs from the paper (be aware when comparing numbers)

* **Training budget:** a few thousand steps instead of ~140k (Müller–Brown) → noisier, less converged models.
* **α and λ_FP:** the paper sets λ_FP = λ; with λ = s(t)² this would switch the FP term off at small t, so we use
  λ_FP = 1 and re-tuned α (0.01 for the Gaussian mixture, 0.15–0.5 for Müller–Brown, instead of 5e-4). The scale of α is not transferable between the two choices.
  At our budget results are sensitive to α — Part 5 has a sweep.
* **Unstated details we had to choose:** Langevin friction γ, simulation length, time embedding, LR schedule.
* **Molecular systems** (Part 8): architecture and pipeline are implemented, but full training needs GPUs and
  the datasets; use the official repo for exact reproduction.

Log your reproduced numbers next to the paper's in [RESULTS.md](RESULTS.md). It already has our first laptop
run: the problem (Fig. 1, Fig. 6, Two For One) reproduces clearly; the fix reproduces on the Gaussian mixture and
only partially on Müller–Brown at this training budget.
