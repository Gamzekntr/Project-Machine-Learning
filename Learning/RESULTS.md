# Reproduction log

Fill in your numbers as you run the notebooks. Paper values are copied from the PDF.

## Claims to test

| # | claim (paper) | where | reproduced? | notes |
|---|---|---|---|---|
| 1 | DSM-only models give correct iid samples but a wrong density at t = 0 (Fig. 1) | Part 3 | ✅ yes | GMM small-mode mass: true 0.200, iid 0.191, learned p0 0.086. MB small basin: data 0.137, iid 0.124, sim 0.068 |
| 2 | The FP residual of DSM-only models is largest near t = 0 (Fig. 6) | Part 4/5 | ✅ yes | RMS residual ~600 at t≈0.003 vs ~10–100 at t>0.2; FP training brings it to ~1–5 at all t |
| 3 | FP regularisation reduces the FP error and makes iid ≈ sim (Fig. 1 bottom, Table 1) | Part 5 | ✅ GMM / ⚠️ MB partial | GMM α=0.01: iid 0.233, p0 0.194 (true 0.200). MB α=0.5: sim basin 0.154 (data 0.137) but iid degrades to 0.31; α=0.2: iid 0.16, sim 0.08. Trade-off is much sharper than in the paper at our ~4k-step budget |
| 4 | Two For One (t = 0.02) blurs structure (Fig. 3, 12) | Part 3 | ✅ yes (stronger than paper) | MB: small basin almost lost in sim (0.004), basin shape broadened |
| 5 | MoE improves simulation, beyond just adding parameters (Table 7) | Part 6 | ⚠️ mixed | Mixture lowers sim PMF (5.88 → 2.68) but small-basin mass gets worse (0.037). Wider-model control not run yet |
| 6 | Both (MoE + FP) is the most consistent model (Table 10) | Part 7 | ❌ not at laptop budget | Both's sim basin mass ranged 0.06–0.27 for α ∈ {0.05, 0.15, 0.3}: very sensitive to α. Needs longer training (`ex.STEPS_SCALE`), more seeds |
| 7 | Same on alanine dipeptide (Table 11) | Part 8 | – not run | needs data (+GPU for the full budget) |

## Müller–Brown (Table 10)

| Method | iid JS (paper) | sim JS (paper) | iid PMF (paper) | sim PMF (paper) | iid JS (ours) | sim JS (ours) | iid PMF (ours) | sim PMF (ours) |
|---|---|---|---|---|---|---|---|---|
| Reference | 0.0119 | | 0.087 | | | | | |
| Diffusion | 0.0122 | 0.0448 | 0.111 | 0.504 | | | | |
| Two For One | – | – | – | – | | | | |
| Mixture | 0.0109 | 0.0254 | 0.097 | 0.247 | | | | |
| Fokker–Planck | 0.0130 | 0.0166 | 0.122 | 0.163 | | | | |
| Both | 0.0110 | 0.0108 | 0.098 | 0.099 | | | | |

Our values depend on sample size and training budget — compare against our own reference row, not the paper's digits.

## Alanine dipeptide, 5 beads (Table 11)

| Method | iid PMF (paper) | sim PMF (paper) | ours iid | ours sim |
|---|---|---|---|---|
| Diffusion | 0.095 | 1.047 | | |
| Two For One | 0.098 | 0.206 | | |
| Mixture | 0.092 | 0.388 | | |
| Fokker–Planck | 0.098 | 0.104 | | |
| Both | 0.097 | 0.091 | | |

## Our first run (seed 0, default configs, laptop CPU, 4k training steps)

Müller–Brown, 18k iid samples and 18k simulation frames, 64 bins, equal sample sizes:

| Method | iid JS | sim JS | iid PMF | sim PMF | sim small-basin mass (data 0.137) |
|---|---|---|---|---|---|
| Reference (true force) | 0.0105 | | 2.28 | | 0.133–0.146 |
| Diffusion | 0.037 | 0.134 | 3.51 | 5.88 | 0.068 |
| Two For One | 0.037 | 0.113 | 3.51 | 5.36 | 0.004 |
| Mixture | 0.055 | 0.096 | 3.86 | 2.68 | 0.037 |
| Fokker–Planck (α=0.5) | 0.225 | 0.218 | 11.6 | 11.9 | 0.154 |
| Both (α=0.15) | 0.154 | 0.133 | 5.56 | 4.77 | 0.061 |

* The PMF noise floor (2.28) is far above the paper's 0.087 because we use fewer samples per histogram (many
  sparsely populated bins). The JS floor (0.0105) matches the paper's 0.0119.
* Our models are undertrained (iid JS 3.5× the floor for Diffusion vs ≈1× in the paper). That's the first thing
  to fix: set `ex.STEPS_SCALE = 5` or more, delete `results/mb_*.pt`, and re-run Part 7.
