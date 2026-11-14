"""Transformations for load_data(..., transformation=...).

Each takes a batch tensor and returns a tensor. Positions have shape (N, n_atoms, 3).

    sigma = fit_scale(train_positions[:, BACKBONE_INDICES])
    transform = aldp_transform(sigma)
"""
import math

import torch

from dihedrals import BACKBONE_INDICES


class Compose:
    def __init__(self, *transforms):
        self.transforms = [t for t in transforms if t is not None]

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def inverse(self, x):
        for t in reversed(self.transforms):
            if hasattr(t, "inverse"):
                x = t.inverse(x)
        return x

    def __repr__(self):
        return "Compose(" + ", ".join(repr(t) for t in self.transforms) + ")"


class SelectAtoms:
    def __init__(self, indices=BACKBONE_INDICES):
        self.indices = list(indices)

    def __call__(self, x):
        return x[:, self.indices]

    def __repr__(self):
        return f"SelectAtoms({self.indices})"


def CoarseGrain():
    """Keep the backbone atoms [C, N, CA, C, N]."""
    return SelectAtoms(BACKBONE_INDICES)


class Center:
    """Subtract the mean position of each frame."""

    def __call__(self, x):
        return x - x.mean(dim=-2, keepdim=True)

    def __repr__(self):
        return "Center()"


class Scale:
    """x -> factor * x. Gradients/forces scale with 1 / factor."""

    def __init__(self, factor):
        self.factor = float(factor)

    def __call__(self, x):
        return x * self.factor

    def inverse(self, x):
        return x / self.factor

    def __repr__(self):
        return f"Scale({self.factor:.6g})"


def random_rotation_matrices(n, generator=None, dtype=torch.float32, device="cpu"):
    """n uniformly random rotation matrices (det = +1, no reflections)."""
    q = torch.randn(n, 4, generator=generator, dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(n, 3, 3)
    return R.to(dtype=dtype, device=device)


class RandomRotation:
    """Rotate every frame randomly (training data only, after Center)."""

    def __init__(self, seed=None):
        self.generator = torch.Generator().manual_seed(seed) if seed is not None else None

    def __call__(self, x):
        R = random_rotation_matrices(x.shape[0], self.generator, dtype=x.dtype, device=x.device)
        return torch.einsum("nij,naj->nai", R, x)

    def __repr__(self):
        return "RandomRotation()"


class Flatten:
    """(N, n_atoms, 3) -> (N, 3 * n_atoms)."""

    def __call__(self, x):
        return x.reshape(x.shape[0], -1)

    def inverse(self, x):
        return x.reshape(x.shape[0], -1, 3)

    def __repr__(self):
        return "Flatten()"


class Standardize:
    """Per-coordinate standardisation, e.g. for Mueller-Brown."""

    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)

    @classmethod
    def fit(cls, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return cls(x.mean(dim=0), x.std(dim=0))

    def __call__(self, x):
        return (x - self.mean.to(x)) / self.std.to(x)

    def inverse(self, x):
        return x * self.std.to(x) + self.mean.to(x)

    def __repr__(self):
        return "Standardize()"


def fit_scale(positions):
    """Standard deviation of the centred positions (one number for all atoms and axes)."""
    x = torch.as_tensor(positions, dtype=torch.float64)
    return float(math.sqrt(((x - x.mean(dim=-2, keepdim=True)) ** 2).mean()))


def aldp_transform(sigma, coarse_grain=True, augment=True, flatten=True, seed=None):
    """coarse-grain -> center -> scale to unit variance -> random rotation -> flatten."""
    return Compose(
        CoarseGrain() if coarse_grain else None,
        Center(),
        Scale(1.0 / sigma),
        RandomRotation(seed) if augment else None,
        Flatten() if flatten else None,
    )
