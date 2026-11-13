# Alanine dipeptide (Milestone 1)

Issues #3 (`load_data`) and #4 (`compute_dihedrals`), plus figures for the report.

| File | Content |
|---|---|
| `data.py` | `load_data` for alanine dipeptide (`"aldp"`) and Mueller-Brown (`"toy"`), force field (`AldpEnergy`) |
| `dihedrals.py` | `compute_dihedrals` (phi, psi in PyTorch) |
| `transforms.py` | transformations for `load_data` |
| `metrics.py` | JS, KL, PMF error, Wasserstein |
| `analysis.py` | states, transitions, autocorrelation |
| `plotting.py` | Ramachandran and other plots |
| `aldp_overview.py` | figures and numbers for the report |
| `bgmol_compat.py` | makes bgmol work with current mdtraj |
| `tests/` | tests |

## Usage

```bash
pip install -r requirements.txt
python tests/test_dihedrals.py
python aldp_overview.py          # first run downloads the data to ~/pml-data
```

```python
from data import load_data, get_split
from dihedrals import compute_dihedrals
from transforms import aldp_transform, fit_scale

for x in load_data("aldp"):           # (1024, 22, 3) in nm
    angles = compute_dihedrals(x)     # (1024, 2)

sub = get_split("aldp", "train", n_samples=50_000)
sigma = fit_scale(sub[:, [4, 6, 8, 14, 16]])
train = load_data("aldp", transformation=aldp_transform(sigma), n_samples=50_000, n_epochs=None)

for x in load_data("toy"):            # (1024, 2), from data/mueller_brown_samples.npy
    ...
```

The splits are consecutive blocks of the trajectory (80/10/10 %, 1 ns gap between them).
