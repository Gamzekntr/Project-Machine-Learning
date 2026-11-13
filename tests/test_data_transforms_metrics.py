"""Tests for data.py, transforms.py and metrics.py on small fake data:
python tests/test_data_transforms_metrics.py"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

import data
import metrics
from dihedrals import compute_dihedrals
from transforms import (Center, Compose, Flatten, RandomRotation, Scale, Standardize,
                        aldp_transform, fit_scale, random_rotation_matrices)

N_FAKE = 20_000


def _fake_aldp_root():
    root = Path(tempfile.mkdtemp())
    cache = root / "AImplicitUnconstrained" / "cache"
    cache.mkdir(parents=True)
    rng = np.random.default_rng(0)
    xyz = (rng.normal(scale=0.2, size=(N_FAKE, 22, 3)) + 1.5).astype(np.float32)
    np.save(cache / "xyz.npy", xyz)
    np.save(cache / "time.npy", 1000.0 + np.arange(N_FAKE, dtype=np.float64))
    np.save(cache / "energies.npy", rng.normal(size=N_FAKE).astype(np.float32))
    names = ["CH3", "H1", "H2", "H3", "C", "O", "N", "H", "CA", "HA", "CB", "HB1", "HB2", "HB3",
             "C", "O", "N", "H", "C", "H1", "H2", "H3"]
    res = ["ACE"] * 6 + ["ALA"] * 10 + ["NME"] * 6
    np.savez(cache / "topology.npz", atom_names=np.array(names), residue_names=np.array(res),
             elements=np.array([n[0] for n in names]), bonds=np.zeros((0, 2), dtype=np.int64),
             temperature=np.array(300.0))
    return root, xyz


def test_split_indices_contiguous_disjoint():
    s = data.split_indices(1000, gap=10)
    assert len(s["train"]) == 800 and s["train"][0] == 0 and s["train"][-1] == 799
    assert s["val"][0] == 810 and s["test"][0] == 910 and s["test"][-1] == 999
    allidx = np.concatenate(list(s.values()))
    assert len(np.unique(allidx)) == len(allidx)


def test_load_data_aldp():
    root, xyz = _fake_aldp_root()
    data.load_aldp_arrays.cache_clear()
    batches = list(data.load_data("aldp", root=root, batch_size=1000, shuffle=False))
    x = torch.cat(batches)
    n_train = len(data.split_indices(N_FAKE)["train"])
    assert x.shape == (n_train, 22, 3) and x.dtype == torch.float32
    assert torch.allclose(x, torch.from_numpy(xyz[:n_train]))
    sub = data.get_split("aldp", "train", n_samples=500, root=root)
    assert sub.shape == (500, 22, 3)
    gen = data.load_data("aldp", root=root, n_epochs=None, batch_size=64, n_samples=100)
    assert sum(1 for _, _ in zip(range(10), gen)) == 10
    single = next(data.load_data("aldp", root=root, batch_size=None))
    assert single.shape == (22, 3)
    sigma = fit_scale(sub[:, [4, 6, 8, 14, 16]])
    t = aldp_transform(sigma, coarse_grain=True, augment=True)
    b = next(data.load_data("aldp", root=root, transformation=t, batch_size=256, split="test"))
    assert b.shape == (256, 15)
    assert abs(b.reshape(256, 5, 3).mean(1)).max() < 1e-5
    assert 0.5 < b.std().item() < 2.0


def test_load_data_mb_formats():
    tmp = Path(tempfile.mkdtemp())
    pts = np.random.default_rng(0).normal(size=(5000, 2)).astype(np.float32)
    np.save(tmp / "mb.npy", pts)
    np.savez(tmp / "mb.npz", x=pts)
    torch.save({"samples": torch.from_numpy(pts)}, tmp / "mb.pt")
    traj = pts.reshape(1000, 5, 2)
    np.save(tmp / "mb_traj.npy", traj)
    for f in ("mb.npy", "mb.npz", "mb.pt"):
        assert np.allclose(data.load_mb_arrays(tmp / f), pts)
    assert data.load_mb_arrays(tmp / "mb_traj.npy").shape == (5000, 2)
    batch = next(data.load_data("toy", mb_path=tmp / "mb.npy", batch_size=128))
    assert batch.shape == (128, 2)
    std = Standardize.fit(pts)
    b = next(data.load_data("mb", mb_path=tmp / "mb.npy", batch_size=4000, transformation=std))
    assert torch.allclose(std.inverse(b), torch.as_tensor(b) * std.std + std.mean)


def test_transforms():
    R = random_rotation_matrices(1000, torch.Generator().manual_seed(0), dtype=torch.float64)
    eye = torch.eye(3, dtype=torch.float64).expand(1000, 3, 3)
    assert torch.allclose(R @ R.transpose(1, 2), eye, atol=1e-10)
    assert torch.allclose(torch.linalg.det(R), torch.ones(1000, dtype=torch.float64))
    x = torch.randn(64, 22, 3, dtype=torch.float64) + 3.0
    c = Center()(x)
    assert c.mean(1).abs().max() < 1e-12
    rot = RandomRotation(seed=0)(c)
    assert torch.allclose(torch.cdist(rot, rot), torch.cdist(c, c), atol=1e-10)
    assert torch.allclose(compute_dihedrals(rot), compute_dihedrals(c), atol=1e-8)
    t = Compose(Center(), Scale(1 / 0.3), Flatten())
    y = t(x)
    assert y.shape == (64, 66)
    assert torch.allclose(t.inverse(y), c, atol=1e-12)


def test_metrics():
    rng = np.random.default_rng(0)
    a = rng.uniform(-np.pi, np.pi, size=(20000, 2))
    assert metrics.js_distance(a, a) == 0.0
    assert metrics.pmf_error(a, a) == 0.0
    assert metrics.kl_divergence(a, a) < 1e-12
    b = rng.uniform(-np.pi, np.pi, size=(20000, 2))
    assert metrics.js_distance(a, b) > 0
    # shift across the -pi/pi border
    theta = rng.vonmises(np.pi - 0.05, 20.0, size=50000)
    for delta in (0.1, 0.3):
        w = metrics.circular_wasserstein_1d(theta, (theta + delta + np.pi) % (2 * np.pi) - np.pi)
        assert abs(w - delta) < 0.01
    probs = metrics.boltzmann_bin_probabilities(lambda p: 0.5 * (p ** 2).sum(1),
                                                ((-4, 4), (-4, 4)), bins=32, kT=1.0)
    assert abs(probs.sum() - 1) < 1e-12
    from scipy.stats import norm
    edges = np.linspace(-4, 4, 33)
    p1 = np.diff(norm.cdf(edges))
    exact = np.outer(p1, p1)
    assert np.abs(probs - exact / exact.sum()).max() < 1e-4
    counts = metrics.histogram_2d(rng.normal(size=(200_000, 2)), bins=32, range=((-4, 4), (-4, 4)))
    counts2 = metrics.histogram_2d(rng.normal(size=(200_000, 2)), bins=32, range=((-4, 4), (-4, 4)))
    assert (metrics.js_distance_counts(counts, probs, pseudocount=0)
            < metrics.js_distance_counts(counts, counts2, pseudocount=0))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
