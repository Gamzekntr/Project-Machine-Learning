import sys; sys.path.insert(0, "../../Downloads")
import numpy as np, matplotlib; matplotlib.use("Agg")
import viz, standin as s

rng = np.random.default_rng(0)
#2D density on Muller-Brown-like data (fake samples around the minima)
mins = np.array([[-0.558, 1.442], [0.623, 0.028], [-0.05, 0.467]])
pts = mins[rng.choice(3, 20000, p=[.6, .25, .15])] + rng.normal(size=(20000, 2)) * 0.12
gen = mins[rng.choice(3, 20000, p=[.6, .3, .1])] + rng.normal(size=(20000, 2)) * 0.13
viz.plot_density_xy_grid({"Reference": pts, "Generated": gen}, potential_fn=s.muller_brown_potential,
                         kT=23.0, save="./xy_grid.png")
ext = (-1.8, 1.3, -0.4, 2.2)
E = viz.energy_grid(lambda p: s.muller_brown_potential(p) / 23.0, ext, n=120)
print("energy_grid", E.shape, E.min(), E.max().round(1))
viz.plot_energy_map(E, ext, title="analytic U/kT", potential_fn=s.muller_brown_potential, kT=23.0)

# Fihedral time series
t = np.arange(3000); phi = np.where((t // 400) % 2 == 0, -1.3, 1.1) + rng.normal(size=3000) * .2
viz.plot_dihedral_timeseries(phi, phi * 0.5, step=1, unit="frame")

# Full-atom renderer: methane (C + 4 H at 1.09 A) must give exactly 4 bonds
d = 1.09 / np.sqrt(3)
meth = np.array([[0, 0, 0], [d, d, d], [d, -d, -d], [-d, d, -d], [-d, -d, d]]) / 10.0   # nm
els = ["C", "H", "H", "H", "H"]
bonds = viz.guess_bonds(meth * 10, els)
print("methane bonds:", bonds)
assert len(bonds) == 4
viz.plot_molecule_mpl(meth, elements=els, save="./methane.png")

#22-atom ALDP element list + random coordinates (runs end-to-end, shape check)
x22 = rng.normal(size=(3, 22, 3)) * 0.2
viz.plot_molecule_mpl(x22, titles=["a", "b", "c"], save="./aldp_random.png")
try:
    viz.plot_molecule_mpl(x22[:, :20])
except ValueError as e:
    print("size mismatch caught:", e)
print("viz tests passed")
