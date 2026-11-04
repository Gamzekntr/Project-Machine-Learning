# Part 8 — From toys to molecules: architecture, coarse-graining and the real experiments

**Paper sections:** §3.2 "Architecture" (Eq. 15), §5.1–5.2, App. A.5 (coarse-graining), App. B.3–B.6
(datasets, hyper-parameters, compute), App. C.5–C.8 (molecular results)
**Notebook:** [08_towards_molecules.ipynb](08_towards_molecules.ipynb)
**Code:** [scoremd/molecules.py](../scoremd/molecules.py)

---

## 1. Coarse-graining (App. A.5)

Map all-atom coordinates `x` to beads `z = Ξ(x)` (here: keep a subset of atoms). The CG equilibrium
density is the marginal

$$ p(z) \propto \int \exp(-U(x)/k_BT)\,\delta(\Xi(x)-z)\,dx $$

whose potential (the potential of mean force) is unknown — but a diffusion model trained on CG
samples learns its score, hence the CG forces (Part 2).

| system | CG beads | data | paper App. |
|---|---|---|---|
| Alanine dipeptide | 5 atoms `[C, N, CA, C, N]` | 50k frames, implicit solvent, 300 K (`bgmol`) | B.3 |
| Chignolin | 10 (one per residue, Cα) | Lindorff-Larsen et al. 2011, 340 K | B.4 |
| BBA | 28 (one per residue) | Lindorff-Larsen et al. 2011, 325 K | B.4 |
| Dipeptides (2AA) | `[N, CA, CB, C, O]` per residue (≤10) | Klein et al. 2023a, 310 K | B.5 |

## 2. Architecture (Eq. 15)

A graph transformer (Shi et al. 2021) on the fully connected graph of beads:

$$ e_{ij} = x_i - x_j,\qquad n_i^{(0)} = [a_i, t],\qquad n^{(l+1)} = \phi^{(l)}(n^{(l)}, e),\qquad \log p_\theta = \sum_i \psi(n_i^{(L)}),\quad s_\theta = \nabla_x \log p_\theta $$

* **Permutation equivariant** — it's a graph network.
* **Translation invariant** — only relative vectors `x_i - x_j` enter.
* **Rotation equivariance is learned**, via random rotations of the training data (as in AlphaFold 3),
  not built in. Using raw difference vectors (not only distances) also keeps chirality / avoids the
  mirror-symmetry problem.
* **Conservative** — the score is the gradient of a sum of per-node scalar energies.
* Atom features `a_i`: one-hot atom type (single molecule); for the transferable model also atom
  number, residue index and amino-acid type.

## 3. Evaluation (§5, App. B.2)

* Dipeptides: 2D histogram of backbone dihedrals (φ, ψ) — Ramachandran plot.
* Proteins: TICA (time-lagged ICA) on pairwise distances + dihedrals → first two components.
* PMF error / JS divergence on these 2D histograms (Part 7), bond-length distributions (Fig. 3/12),
  contact maps and pairwise-distance JS (App. C.6), Markov-state transition matrices (App. C.6.4).

## 4. Hyper-parameters (App. B.3–B.5)

| | Ala2 | Chignolin | BBA | 2AA |
|---|---|---|---|---|
| hidden / layers (expert 0) | 96 / 2 | 128 / 3 | 128 / 3 | 128 / 3 |
| attention heads / feature dim | 8 / 16 | 8 / 16 | 8 / 16 | 8 / 16 |
| α (Both) | 1e-3 | 2e-3 | 2e-2 | 1e-4 |
| epochs (Both, per expert) | 7000/2000/1000 | 1700/200/100 | 1200/400/200 | 100/20/10 |
| batch size | 1024 | 1024 | 1024 | 1024 |
| Two-For-One `t` | 0.02 | 0.02 | 0.005 | 0.02 |
| sim timestep | 2 fs | 2 fs | 2 fs | 0.5 fs |
| GPUs | 1×A100 | 2×A100 | 4×A100 | 2×A100 |

Training times (App. C.3, Table 8): Ala2 "Both" ≈ 4 h, Chignolin ≈ 21 h, BBA ≈ 6.6 days on A100s.
→ **The molecular experiments are GPU work.** On a laptop you can reproduce the toy results
(Parts 1–7) fully and run alanine dipeptide in a reduced setting.

## 5. Recommended path for the molecular part

1. Run the notebook here: it implements the graph-transformer energy model in PyTorch, checks
   translation invariance / conservativeness / learned rotation equivariance, and contains the
   CG + dihedral + training + simulation pipeline for alanine dipeptide.
2. Get the alanine-dipeptide data: the paper uses the implicit-solvent trajectory distributed with
   [`bgmol`](https://github.com/noegroup/bgmol). The official repo
   [`noegroup/ScoreMD`](https://github.com/noegroup/ScoreMD) has data-loading code, configs and
   pretrained weights — use it as ground truth for any detail that differs from this re-implementation.
3. Compare against Table 11 (Ala2: Both sim PMF 0.091, Diffusion sim PMF 1.047) and Fig. 9/12.
