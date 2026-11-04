# Part 7 — Full toy benchmark: reproducing Table 10 / Figure 8 (Müller–Brown)

**Paper sections:** App. B.2 (metrics), App. C.4 (Table 9 hyper-parameters, Table 10, Fig. 8)
**Notebook:** [07_mueller_brown_benchmark.ipynb](07_mueller_brown_benchmark.ipynb)
**Code:** [scoremd/metrics.py](../scoremd/metrics.py)

---

## Metrics (App. B.2)

Both are computed on 2D histograms (dihedrals φ/ψ for peptides, TICA for proteins, raw `(x, y)` here):

* **PMF error** (Durumeric et al. 2024): bin proportions `p, q` (+`1e-6`), then
  `mean_bins (log p − log q)²`. Compares *free energies* `−log p`, so it is very sensitive to
  low-density regions (rarely visited states, barriers). 64 bins for dipeptides, 100 for proteins.
* **JS divergence**: on the flattened histograms with +1 pseudo-count per bin. Compares
  *probabilities*, dominated by the high-density regions.

**Important for fair comparisons:** both metrics depend on the number of samples (empty bins get
`log 1e-6`). Always compare sets of the same size, and report the **reference-vs-reference** value
(two independent MD runs on the true potential) as the noise floor — exactly what Table 10 does.

## Paper's numbers to compare against (Table 10, mean ± std over 10 models)

| Method | iid JS | sim JS | iid PMF | sim PMF |
|---|---|---|---|---|
| Reference | 0.0119 | | 0.087 | |
| Diffusion | 0.0122 | 0.0448 | 0.111 | 0.504 |
| Mixture | 0.0109 | 0.0254 | 0.097 | 0.247 |
| Fokker–Planck | 0.0130 | 0.0166 | 0.122 | 0.163 |
| Both | 0.0110 | 0.0108 | 0.098 | 0.099 |

Expect **the same ordering / qualitative pattern**, not the same digits: our training budget is far
smaller (CPU), the friction `γ` and simulation length are not given, and our sample sizes differ.
The main claim to test is: *sim error ≫ iid error for Diffusion; sim ≈ iid for Both.*

## What the notebook does

Runs all five methods (Diffusion, Two For One, Mixture, Fokker–Planck, Both) with `N_SEEDS` seeds,
produces a Fig. 8-style grid (iid row / sim row) and a Table-10-style results table, saved to
`results/table10_reproduction.csv`. Set `FAST = False` for longer training (closer to the paper).
