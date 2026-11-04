"""Molecular pieces (paper Sec. 3.2, Eq. 15; App. B.3): a conservative graph-transformer energy
model, random-rotation augmentation, dihedral angles and the alanine-dipeptide coarse-graining.

All models take *flattened* coordinates x of shape (B, N*3) so that the generic SDE / loss /
sampling / Langevin code from the toy experiments works unchanged.
"""
import math
import torch
import torch.nn as nn
from .models import FourierTimeEmbedding


# ----------------------------------------------------------------------------------------------
# Graph transformer energy model
# ----------------------------------------------------------------------------------------------
class GraphTransformerLayer(nn.Module):
    """Dense graph-transformer layer in the style of Shi et al. (2021) ('TransformerConv'):
        alpha_ij = softmax_j( q_i . (k_j + e_ij) / sqrt(d) ),   m_i = sum_j alpha_ij (v_j + e_ij)
    followed by a gated residual and a feed-forward block."""

    def __init__(self, hidden, heads=8, head_dim=16, edge_dim=32):
        super().__init__()
        self.h, self.d = heads, head_dim
        self.q = nn.Linear(hidden, heads * head_dim)
        self.k = nn.Linear(hidden, heads * head_dim)
        self.v = nn.Linear(hidden, heads * head_dim)
        self.e = nn.Linear(edge_dim, heads * head_dim, bias=False)
        self.o = nn.Linear(heads * head_dim, hidden)
        self.gate = nn.Linear(2 * hidden, 1)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.SiLU(), nn.Linear(2 * hidden, hidden))

    def forward(self, n, e, mask):
        B, N, _ = n.shape
        q = self.q(n).view(B, N, self.h, self.d)
        k = self.k(n).view(B, N, self.h, self.d)
        v = self.v(n).view(B, N, self.h, self.d)
        ee = self.e(e).view(B, N, N, self.h, self.d)
        logits = (q[:, :, None] * (k[:, None] + ee)).sum(-1) / math.sqrt(self.d)  # (B,N,N,h)
        logits = logits.masked_fill(~mask[None, :, :, None], float("-inf"))
        a = torch.softmax(logits, dim=2)
        msg = (a[..., None] * (v[:, None] + ee)).sum(2).reshape(B, N, -1)
        msg = self.o(msg)
        g = torch.sigmoid(self.gate(torch.cat([n, msg], -1)))
        n = self.norm1(n + g * msg)
        return self.norm2(n + self.ff(n))


class GraphEnergyNet(nn.Module):
    """Conservative, translation-invariant energy model (Eq. 15):
        e_ij = x_i - x_j,  n_i^(0) = [a_i, t],  n^(l+1) = phi(n^(l), e),  log p = sum_i psi(n_i^(L)),
        score = grad_x log p.
    Rotation equivariance is NOT built in: train with `random_rotation` augmentation."""

    conservative = True

    def __init__(self, atom_types, n_types=None, hidden=96, layers=2, heads=8, head_dim=16,
                 edge_dim=32, t_emb=32):
        super().__init__()
        atom_types = torch.as_tensor(atom_types, dtype=torch.long)
        self.register_buffer("atom_types", atom_types)
        self.N = len(atom_types)
        n_types = n_types or int(atom_types.max()) + 1
        self.atom_emb = nn.Embedding(n_types, hidden)
        self.temb = FourierTimeEmbedding(t_emb)
        self.inp = nn.Linear(hidden + t_emb, hidden)
        # edge features from the raw difference vector and its length
        self.edge_mlp = nn.Sequential(nn.Linear(4, edge_dim), nn.SiLU(), nn.Linear(edge_dim, edge_dim))
        self.layers = nn.ModuleList([GraphTransformerLayer(hidden, heads, head_dim, edge_dim)
                                     for _ in range(layers)])
        self.psi = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.register_buffer("mask", ~torch.eye(self.N, dtype=torch.bool))

    def log_p(self, x, t):
        B = x.shape[0]
        pos = x.view(B, self.N, 3)
        rel = pos[:, :, None, :] - pos[:, None, :, :]                       # e_ij = x_i - x_j
        dist = torch.sqrt((rel ** 2).sum(-1, keepdim=True) + 1e-8)
        e = self.edge_mlp(torch.cat([rel, dist], -1))
        a = self.atom_emb(self.atom_types)[None].expand(B, -1, -1)
        tt = self.temb(t)[:, None, :].expand(B, self.N, -1)
        n = self.inp(torch.cat([a, tt], -1))
        for layer in self.layers:
            n = layer(n, e, self.mask)
        return self.psi(n).squeeze(-1).sum(-1)                               # sum_i psi(n_i)

    def score(self, x, t, create_graph=None):
        if create_graph is None:
            create_graph = self.training
        with torch.enable_grad():
            xr = x.detach().requires_grad_(True)
            (s,) = torch.autograd.grad(self.log_p(xr, t).sum(), xr, create_graph=create_graph)
        return s if create_graph else s.detach()


# ----------------------------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------------------------
def random_rotation_matrices(B, dtype=None):
    """Uniformly random rotations via QR of a Gaussian matrix."""
    A = torch.randn(B, 3, 3, dtype=dtype or torch.get_default_dtype())
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))[:, None, :]
    det = torch.det(Q)
    Q[det < 0, :, 0] *= -1  # proper rotations only (det = +1): keep chirality
    return Q


def random_rotation(x):
    """Augmentation for flattened coordinates (B, N*3): rotate each sample about its centre."""
    B = x.shape[0]
    pos = x.view(B, -1, 3)
    c = pos.mean(1, keepdim=True)
    Rm = random_rotation_matrices(B, x.dtype)
    return ((pos - c) @ Rm.transpose(1, 2)).reshape(B, -1)


def dihedral(p0, p1, p2, p3):
    """Dihedral angle (radians) of four points, each (..., 3)."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / b1.norm(dim=-1, keepdim=True)
    v = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
    x = (v * w).sum(-1)
    y = (torch.cross(b1n, v, dim=-1) * w).sum(-1)
    return torch.atan2(y, x)


# Alanine dipeptide (ACE-ALA-NME, 22 atoms, standard ordering used in bgmol / bgflow tutorials).
# Backbone: 4 = C(ACE), 6 = N, 8 = CA, 14 = C, 16 = N(NME). phi = C-N-CA-C, psi = N-CA-C-N.
# !! Verify these indices against the topology of the data you download. !!
ALA2_CG_INDICES = [4, 6, 8, 14, 16]            # paper App. B.3: [C, N, CA, C, N]
ALA2_CG_ATOM_TYPES = [0, 1, 2, 0, 1]           # C, N, CA, C, N  (one-hot types)


def ala2_phi_psi_cg(z):
    """phi/psi from the 5-bead CG representation, z: (B, 5, 3) or (B, 15)."""
    z = z.view(z.shape[0], 5, 3)
    phi = dihedral(z[:, 0], z[:, 1], z[:, 2], z[:, 3])
    psi = dihedral(z[:, 1], z[:, 2], z[:, 3], z[:, 4])
    return torch.stack([phi, psi], -1)


def bond_lengths_cg(z):
    """Consecutive bead distances for a chain, z: (B, N, 3) -> (B, N-1)."""
    return (z[:, 1:] - z[:, :-1]).norm(dim=-1)
