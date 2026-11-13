"""Tests for compute_dihedrals: python tests/test_dihedrals.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mdtraj as md
import numpy as np
import torch

from dihedrals import BACKBONE_INDICES, PHI_INDICES, PSI_INDICES, compute_dihedrals
from transforms import random_rotation_matrices


def _random_frames(n=500, n_atoms=22, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(scale=0.3, size=(n, n_atoms, 3)).astype(np.float32)


def _mdtraj_reference(xyz):
    top = md.Topology()
    res = top.add_residue("X", top.add_chain())
    for _ in range(xyz.shape[1]):
        top.add_atom("X", md.element.carbon, res)
    return md.compute_dihedrals(md.Trajectory(xyz, top), [PHI_INDICES, PSI_INDICES])


def _angle_diff(a, b):
    return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def test_matches_mdtraj():
    xyz = _random_frames()
    ours = compute_dihedrals(torch.from_numpy(xyz)).numpy()
    ref = _mdtraj_reference(xyz)
    assert ours.shape == (500, 2)
    assert _angle_diff(ours, ref).max() < 1e-4


def test_range_and_shapes():
    xyz = _random_frames(100)
    a = compute_dihedrals(torch.from_numpy(xyz))
    assert torch.all(a > -np.pi - 1e-6) and torch.all(a <= np.pi + 1e-6)
    flat = compute_dihedrals(torch.from_numpy(xyz).reshape(100, -1))
    single = compute_dihedrals(xyz[0])
    assert torch.allclose(a, flat)
    assert torch.allclose(a[:1], single)
    assert torch.allclose(compute_dihedrals(xyz, degrees=True), torch.rad2deg(a))


def test_coarse_grained_input():
    xyz = torch.from_numpy(_random_frames(50))
    full = compute_dihedrals(xyz)
    cg = compute_dihedrals(xyz[:, list(BACKBONE_INDICES)])
    assert torch.allclose(full, cg, atol=1e-6)


def test_invariances():
    xyz = torch.from_numpy(_random_frames(200)).double()
    a = compute_dihedrals(xyz)
    R = random_rotation_matrices(200, torch.Generator().manual_seed(1), dtype=torch.float64)
    moved = torch.einsum("nij,naj->nai", R, xyz) + torch.randn(200, 1, 3, dtype=torch.float64)
    assert torch.allclose(compute_dihedrals(moved), a, atol=1e-8)
    # mirror image: both angles change sign
    mirrored = xyz * torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)
    assert torch.allclose(compute_dihedrals(mirrored), -a, atol=1e-8)


def test_known_angles():
    p = torch.tensor([[[1.0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]]], dtype=torch.float64)
    x = torch.zeros(1, 22, 3, dtype=torch.float64)
    x[0, list(PHI_INDICES)] = p[0]
    phi = compute_dihedrals(x)[0, 0]
    assert abs(phi.item() - np.pi / 2) < 1e-12


def test_gradient():
    xyz = torch.from_numpy(_random_frames(4)).double().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x: compute_dihedrals(x), (xyz,))
    out = compute_dihedrals(xyz).sin().sum()
    (g,) = torch.autograd.grad(out, xyz)
    assert torch.isfinite(g).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
