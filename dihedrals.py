"""Backbone dihedrals phi and psi of alanine dipeptide.

Atom indices in the bgmol dataset:
    phi = (C ACE, N ALA, CA ALA, C ALA) = (4, 6, 8, 14)
    psi = (N ALA, CA ALA, C ALA, N NME) = (6, 8, 14, 16)
The five atoms (4, 6, 8, 14, 16) are also the coarse-grained representation of the paper.
"""
import torch

PHI_INDICES = (4, 6, 8, 14)
PSI_INDICES = (6, 8, 14, 16)
BACKBONE_INDICES = (4, 6, 8, 14, 16)
PHI_INDICES_CG = (0, 1, 2, 3)
PSI_INDICES_CG = (1, 2, 3, 4)


def dihedral(p0, p1, p2, p3):
    """Dihedral angle of four points (..., 3) in (-pi, pi], same convention as mdtraj.

    atan2 keeps the sign and has finite gradients, unlike arccos.
    """
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)
    y = torch.linalg.norm(b2, dim=-1) * (b1 * n2).sum(dim=-1)
    x = (n1 * n2).sum(dim=-1)
    return torch.atan2(y, x)


def _as_positions(positions):
    x = torch.as_tensor(positions)
    if not torch.is_floating_point(x):
        x = x.float()
    if x.ndim == 2 and x.shape[-1] == 3:   # one frame
        x = x.unsqueeze(0)
    elif x.ndim == 2:                      # flattened (N, 3 * n_atoms)
        if x.shape[-1] % 3 != 0:
            raise ValueError(f"cannot reshape last dim {x.shape[-1]} into (n_atoms, 3)")
        x = x.reshape(x.shape[0], -1, 3)
    elif x.ndim == 1:
        x = x.reshape(1, -1, 3)
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"expected shape (N, n_atoms, 3), got {tuple(x.shape)}")
    return x


def compute_dihedrals(positions, phi_indices=None, psi_indices=None, degrees=False):
    """phi and psi for a batch of positions.

    positions: (N, 22, 3), (N, 5, 3) for the coarse-grained backbone, flattened, or one frame.
    Returns a tensor (N, 2) with columns phi, psi in radians (differentiable).
    """
    x = _as_positions(positions)
    n_atoms = x.shape[1]
    if phi_indices is None or psi_indices is None:
        if n_atoms == 5:
            default_phi, default_psi = PHI_INDICES_CG, PSI_INDICES_CG
        elif n_atoms == 22:
            default_phi, default_psi = PHI_INDICES, PSI_INDICES
        else:
            raise ValueError(f"no default indices for {n_atoms} atoms, pass them explicitly")
        phi_indices = phi_indices or default_phi
        psi_indices = psi_indices or default_psi

    phi = dihedral(*(x[:, i] for i in phi_indices))
    psi = dihedral(*(x[:, i] for i in psi_indices))
    angles = torch.stack([phi, psi], dim=-1)
    return torch.rad2deg(angles) if degrees else angles


def find_backbone_indices(atom_names, residue_names):
    """Indices of C ACE, N ALA, CA ALA, C ALA, N NME looked up by name."""
    wanted = [("ACE", "C"), ("ALA", "N"), ("ALA", "CA"), ("ALA", "C"), ("NME", "N")]
    lookup = {(r, a): i for i, (a, r) in enumerate(zip(atom_names, residue_names))}
    return tuple(lookup[key] for key in wanted)
