"""Toy systems: the 2D Gaussian mixture of Figure 1 and the Mueller-Brown potential (App. C.4)."""
import math
import torch
from . import sde
from .simulate import langevin


# ----------------------------------------------------------------------------------------------
# Figure 1: 1/5 N([-5,-5], I) + 4/5 N([5,5], I)
# ----------------------------------------------------------------------------------------------
class GaussianMixture:
    def __init__(self, means, weights, std=1.0):
        self.means = torch.as_tensor(means, dtype=torch.get_default_dtype())
        self.weights = torch.as_tensor(weights, dtype=torch.get_default_dtype())
        self.std = std

    def sample(self, n):
        k = torch.multinomial(self.weights, n, replacement=True)
        return self.means[k] + self.std * torch.randn(n, self.means.shape[1])

    def log_p_t(self, x, t, shift=None, scale=None):
        """Exact log-density of the *diffused* mixture at diffusion time t (VP SDE).

        If the data were normalised as (x - shift)/scale before training, pass the same
        shift/scale so this is the density of the normalised data.
        Each component N(mu, std^2 I) becomes N(m mu, (m^2 std^2 + s^2) I).
        """
        mu, std = self.means, self.std
        if shift is not None:
            mu = (mu - shift) / scale
            std = std / scale  # isotropic scale assumed (scalar)
        m = sde.marginal_mean_coef(t)[:, None]          # (B,1)
        var = (m ** 2) * std ** 2 + sde.marginal_std(t)[:, None] ** 2   # (B,1)
        D = x.shape[-1]
        diff = x[:, None, :] - m[:, :, None] * mu[None]  # (B,K,D)
        logc = (-0.5 * (diff ** 2).sum(-1) / var - 0.5 * D * torch.log(2 * math.pi * var)
                + torch.log(self.weights)[None])
        return torch.logsumexp(logc, dim=-1)


FIG1_GMM = dict(means=[[-5.0, -5.0], [5.0, 5.0]], weights=[0.2, 0.8], std=1.0)


# ----------------------------------------------------------------------------------------------
# Mueller-Brown potential (Eq. 34)
# ----------------------------------------------------------------------------------------------
def mueller_brown(x):
    X, Y = x[..., 0], x[..., 1]
    return (-200 * torch.exp(-(X - 1) ** 2 - 10 * Y ** 2)
            - 100 * torch.exp(-X ** 2 - 10 * (Y - 0.5) ** 2)
            - 170 * torch.exp(-6.5 * (0.5 + X) ** 2 + 11 * (X + 0.5) * (Y - 1.5) - 6.5 * (Y - 1.5) ** 2)
            + 15 * torch.exp(0.7 * (1 + X) ** 2 + 0.6 * (X + 1) * (Y - 1) + 0.7 * (Y - 1) ** 2))


def mueller_brown_force(x):
    with torch.enable_grad():
        xr = x.detach().requires_grad_(True)
        (g,) = torch.autograd.grad(mueller_brown(xr).sum(), xr)
    return -g


def generate_mueller_brown_data(n_samples=100_000, n_chains=100, kT=23.0, dt=0.005, mass=0.5,
                                gamma=1.0, save_every=50, burn_in=2000, seed=0):
    """Langevin MD on the Mueller-Brown potential (App. C.4: kT=23, dt=0.005, M=0.5, every 50th
    frame). The paper ran one chain for 5M steps; we run `n_chains` in parallel for the same total.
    The friction gamma is not given in the paper -> 1.0 is our choice."""
    g = torch.Generator().manual_seed(seed)
    x0 = torch.tensor([[-0.55, 1.45]]).repeat(n_chains, 1) + 0.05 * torch.randn(n_chains, 2, generator=g)
    n_steps = burn_in + (n_samples // n_chains) * save_every
    traj = langevin(mueller_brown_force, x0, n_steps=n_steps, dt=dt, gamma=gamma, kT=kT, mass=mass,
                    save_every=save_every, generator=g)
    traj = traj[burn_in // save_every:]                   # drop burn-in frames
    return traj.reshape(-1, 2)[:n_samples]
