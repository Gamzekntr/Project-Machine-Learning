"""
Energy-based VP diffusion model
* The network works on normalised coordinates x_n = (x - mu) / scale
  (use `Normalizer`); the VP schedule assumes roughly unit-variance data.
* E_theta(x_n, t) is DIMENSIONLESS (units of kT): as t -> 0,
      E_theta(x_n, 0) = -log p(x_n) = U(x) / kT + const.
* forward() returns the score  s_theta = -grad_{x_n} E_theta.
* Physical force from the score (needed when plugging the model into Langevin):
      F = -grad_x U = kT * grad_x log p = (kT / scale) * s_theta(x_n, t_min)
"""
import math
import time

import numpy as np
import torch
import torch.nn as nn

T_EPS = 1e-3     # smallest diffusion time used in training and sampling


# Normalisation
class Normalizer:

    def __init__(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        self.mu = x.mean(0, keepdim=True)
        self.scale = float((x - self.mu).std())

    def to_norm(self, x):
        return (x - self.mu.to(x.device)) / self.scale

    def to_phys(self, xn):
        return xn * self.scale + self.mu.to(xn.device)


class Model(nn.Module):

    def __init__(self, dim=2, hidden_dim=128, n_layers=3,
                 beta_min=0.1, beta_max=20.0):
        super().__init__()
        self.dim, self.beta_min, self.beta_max = dim, beta_min, beta_max
        layers = [nn.Linear(dim + 1, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    #VP-SDE: dx = -beta/2 x dt + sqrt(beta) dw,  beta(t) = bmin + t (bmax - bmin)
    def beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def marginal(self, t):
        ib = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2
        return torch.exp(-0.5 * ib), torch.sqrt(1 - torch.exp(-ib))

    def energy(self, X, t):
        #E_theta(x, t), shape (N,). t may be a scalar or shape (N,).
        t = torch.as_tensor(t, dtype=X.dtype, device=X.device)
        t = t.expand(X.shape[0]).reshape(-1, 1)
        return self.mlp(torch.cat([X, t], dim=-1)).squeeze(-1)

    def forward(self, X, t, create_graph=True):
        with torch.enable_grad():
            x = X if X.requires_grad else X.detach().requires_grad_(True)
            g, = torch.autograd.grad(self.energy(x, t).sum(), x,
                                     create_graph=create_graph)
        return -g


#Training
def dsm_loss(model, x0):
    #Denoising score matching
    t = T_EPS + (1 - T_EPS) * torch.rand(x0.shape[0], device=x0.device)
    m, sig = model.marginal(t)
    eps = torch.randn_like(x0)
    xt = m[:, None] * x0 + sig[:, None] * eps
    return ((sig[:, None] * model(xt, t) + eps) ** 2).sum(-1).mean()


def collect(data, max_samples=None):
    if isinstance(data, (torch.Tensor, np.ndarray)):
        out = torch.as_tensor(data, dtype=torch.float32)
    else:
        chunks, n = [], 0
        for item in data:
            item = torch.as_tensor(item, dtype=torch.float32)
            chunks.append(item if item.dim() > 1 else item[None])
            n += chunks[-1].shape[0]
            if max_samples is not None and n >= max_samples:
                break
        out = torch.cat(chunks)
    return out[:max_samples] if max_samples is not None else out


def train(model, data, epochs=150, batch_size=1024, lr_max=1e-3, lr_min=1e-5,
          clip=10.0, log_every=10):
    #Data: normalised (N, dim) tensor on the model's device. Returns the per-epoch mean loss.
    opt = torch.optim.AdamW(model.parameters(), lr=lr_max)
    n = data.shape[0]
    spe = max(1, n // batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * spe, eta_min=lr_min)
    losses, t0 = [], time.time()
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=data.device)
        run = 0.0
        for i in range(spe):
            loss = dsm_loss(model, data[perm[i * batch_size:(i + 1) * batch_size]])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step(); sched.step()
            run += loss.item()
        losses.append(run / spe)
        if ep % log_every == 0 or ep == epochs - 1:
            print(f"epoch {ep + 1:4d}/{epochs}  loss {losses[-1]:.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return losses


# Sampling
@torch.no_grad()
def sample_iid(model, n, n_steps=1000, device="cpu", seed=0):
    #Returns normalized samples (n, dim).
    g = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    x = torch.randn(n, model.dim, device=device, generator=g)
    ts = torch.linspace(1.0, T_EPS, n_steps + 1, device=device)
    for k in range(n_steps):
        t, h = ts[k], ts[k] - ts[k + 1]
        b = model.beta(t)
        s = model(x, t, create_graph=False).detach()
        x = x + (0.5 * b * x + b * s) * h
        if k < n_steps - 1:
            x = x + torch.sqrt(b * h) * torch.randn(x.shape, device=device, generator=g)
    return x


# ------------------------------------------------------------------ units / force
def force_physical(model, x_phys, norm, kT, t=T_EPS):
    xn = norm.to_norm(x_phys)
    return kT / norm.scale * model(xn, t, create_graph=False).detach()


def make_force_fn(model, norm, kT, t=T_EPS):#Plug into real langevin_integrator in the next milestone.

    return lambda x: force_physical(model, x, norm, kT, t)


def check_units(model, norm, x_phys, kT, potential_fn, force_fn):

    x_phys = torch.as_tensor(x_phys, dtype=torch.float32)
    F_true = torch.as_tensor(force_fn(x_phys), dtype=torch.float32)
    F_mod = force_physical(model, x_phys, norm, kT)
    cos = torch.nn.functional.cosine_similarity(F_mod, F_true, dim=-1).mean().item()
    ratio = (F_mod.norm(dim=-1) / F_true.norm(dim=-1)).median().item()
    with torch.no_grad():
        E = model.energy(norm.to_norm(x_phys), T_EPS)
    resid = E - torch.as_tensor(potential_fn(x_phys), dtype=torch.float32) / kT
    return dict(force_cosine=cos, force_magnitude_ratio=ratio,
                energy_residual_std_kT=resid.std().item())

 #Mean squared difference of -log bin proportions
def pmf_error(a, b, ranges, bins=64, floor=1e-6):
    a, b = np.asarray(a), np.asarray(b)
    ha, _, _ = np.histogram2d(a[:, 0], a[:, 1], bins=bins, range=ranges)
    hb, _, _ = np.histogram2d(b[:, 0], b[:, 1], bins=bins, range=ranges)
    pa, pb = ha / ha.sum() + floor, hb / hb.sum() + floor
    return float(np.mean((np.log(pa) - np.log(pb)) ** 2))
