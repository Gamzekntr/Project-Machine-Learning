"""Checks that the bgmol force field reproduces the stored energies (needs the real data):
python tests/check_force_field.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from data import AldpEnergy, load_aldp_arrays

arrays = load_aldp_arrays(with_forces=True)
idx = np.array([0, 1, 250_000, 500_000, 999_999]) % len(arrays["xyz"])
U, F = AldpEnergy()(np.asarray(arrays["xyz"][idx]))
U_ref = np.asarray(arrays["energies"][idx], dtype=np.float64)
F_ref = np.asarray(arrays["forces"][idx], dtype=np.float64)

print("frame   stored E    recomputed E   (kJ/mol)")
for i, a, b in zip(idx, U_ref, U):
    print(f"{i:7d} {a:11.3f} {b:13.3f}")
print("max |dE| =", np.abs(U - U_ref).max(), "kJ/mol")
print("max |dF| =", np.abs(F - F_ref).max(), "kJ/(mol nm)")
assert np.abs(U - U_ref).max() < 1.0, "energies differ: wrong force field or atom order?"
print("ok: the force field matches the data")
