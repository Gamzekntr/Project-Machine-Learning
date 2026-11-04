"""Networks for the toy experiments (paper Sec. 3.2 / Appendix C.4).

Two parameterisations (paper Sec. 3.2, "Conservative model parameterization"):

* ``EnergyNet``  : the network outputs a scalar  log p_theta(x, t)  (an unnormalised log density,
                   i.e. -U(x)/kT at t=0).  The score is its gradient  s = grad_x log p_theta.
                   -> conservative forces, access to the energy, required for the Fokker-Planck loss.
* ``ScoreNet``   : the network outputs the score vector directly (standard diffusion model).
                   -> no energy available, forces are not guaranteed to be conservative.

For molecules the paper uses a graph transformer (see part08); for 2D toys an MLP is enough.
"""
import math
import torch
import torch.nn as nn


class FourierTimeEmbedding(nn.Module):
    """Random Fourier features of the diffusion time t in [0, 1]."""

    def __init__(self, dim=32, scale=8.0):
        super().__init__()
        self.register_buffer("W", torch.randn(dim // 2) * scale)

    def forward(self, t):
        proj = 2 * math.pi * t[:, None] * self.W[None, :]
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


def mlp(in_dim, hidden, layers, out_dim):
    mods, d = [], in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.SiLU()]  # smooth activation: we need 2nd derivatives
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


class EnergyNet(nn.Module):
    """Conservative (energy-based) diffusion model: log p_theta(x,t) = NN(x,t), score = grad_x."""

    conservative = True

    def __init__(self, dim=2, hidden=92, layers=3, t_emb=32):
        super().__init__()
        self.temb = FourierTimeEmbedding(t_emb)
        self.net = mlp(dim + t_emb, hidden, layers, 1)

    def log_p(self, x, t):
        return self.net(torch.cat([x, self.temb(t)], dim=-1)).squeeze(-1)

    def score(self, x, t, create_graph=None):
        """grad_x log p_theta(x, t). create_graph=True is needed when the score enters a loss."""
        if create_graph is None:
            create_graph = self.training
        with torch.enable_grad():
            xr = x.detach().requires_grad_(True)
            lp = self.log_p(xr, t)
            (s,) = torch.autograd.grad(lp.sum(), xr, create_graph=create_graph)
        return s if create_graph else s.detach()


class ScoreNet(nn.Module):
    """Plain score model s_theta(x,t) = NN(x,t) in R^D (not conservative, no energy)."""

    conservative = False

    def __init__(self, dim=2, hidden=92, layers=3, t_emb=32):
        super().__init__()
        self.temb = FourierTimeEmbedding(t_emb)
        self.net = mlp(dim + t_emb, hidden, layers, dim)

    def score(self, x, t, create_graph=None):
        return self.net(torch.cat([x, self.temb(t)], dim=-1))

    def log_p(self, x, t):
        raise NotImplementedError("A score network has no energy / log-density.")


class AnalyticModel(nn.Module):
    """Wrap an analytic log p_t(x) (e.g. a Gaussian mixture diffused by the VP SDE) so it looks
    like a model. Useful for sanity-checking losses: the true density has zero FP residual."""

    conservative = True

    def __init__(self, log_p_fn):
        super().__init__()
        self.log_p_fn = log_p_fn

    def log_p(self, x, t):
        return self.log_p_fn(x, t)

    def score(self, x, t, create_graph=None):
        with torch.enable_grad():
            xr = x.detach().requires_grad_(True)
            (s,) = torch.autograd.grad(self.log_p(xr, t).sum(), xr, create_graph=bool(create_graph))
        return s if create_graph else s.detach()
