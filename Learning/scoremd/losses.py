"""Training objectives.

* Denoising score matching (Eq. 3/4) with lambda(t) = s(t)^2  ->  || s(t) * score + eps ||^2
* Fokker-Planck residual (Eq. 8/9), its exact version (for evaluation in low dimension) and the
  cheap *weak* residual estimator (Eq. 12) with finite-difference time derivative (Eq. 14).
"""
import torch
from . import sde


# ----------------------------------------------------------------------------------------------
# Denoising score matching
# ----------------------------------------------------------------------------------------------
def dsm_loss(model, x0, t):
    """L_DSM with lambda(t) = sigma(t)^2 (the usual 'epsilon-prediction' weighting)."""
    xt, eps, std = sde.perturb(x0, t)
    s = model.score(xt, t, create_graph=True)
    return ((std * s + eps) ** 2).sum(-1).mean()


# ----------------------------------------------------------------------------------------------
# Fokker-Planck equation for log p_t  (Eq. 8)
#   d/dt log p_t = 1/2 g^2 [ div(s) + ||s||^2 ] - <f, s> - div(f)
# ----------------------------------------------------------------------------------------------
def dt_log_p_fd(model, x, t, hs=1e-3, hd=5e-4):
    """Non-uniform central finite difference for d/dt log p_theta (Eq. 14, (hs, hd) from A.3)."""
    lp_plus = model.log_p(x, t + hd)
    lp_0 = model.log_p(x, t)
    lp_minus = model.log_p(x, t - hs)
    return (hs ** 2 * lp_plus + (hd ** 2 - hs ** 2) * lp_0 - hd ** 2 * lp_minus) / (hs * hd * (hs + hd))


def dt_log_p_autograd(model, x, t):
    """Exact d/dt log p_theta via autograd (for evaluation)."""
    with torch.enable_grad():
        tr = t.detach().requires_grad_(True)
        (d,) = torch.autograd.grad(model.log_p(x, tr).sum(), tr)
    return d


def divergence_exact(model, x, t, create_graph=False):
    """div_x s_theta(x,t) = trace of the Hessian of log p. Loops over D -> only for small D."""
    with torch.enable_grad():
        xr = x.detach().requires_grad_(True)
        lp = model.log_p(xr, t)
        (s,) = torch.autograd.grad(lp.sum(), xr, create_graph=True)
        div = 0.0
        for i in range(x.shape[-1]):
            (gi,) = torch.autograd.grad(s[:, i].sum(), xr, create_graph=create_graph, retain_graph=True)
            div = div + gi[:, i]
    return div, s


def fp_residual_exact(model, x, t):
    """Strong residual R(x,t) = F_p(x,t) - d/dt log p(x,t)  (Eq. 9), computed exactly."""
    div, s = divergence_exact(model, x, t)
    b = sde.g2(t)
    rhs = 0.5 * b * (div + (s ** 2).sum(-1)) - (sde.drift(x, t) * s).sum(-1) - sde.div_drift(x, t)
    return (rhs - dt_log_p_autograd(model, x, t)).detach()


def fp_residual_weak(model, x, t, v, sigma=1e-4, hs=1e-3, hd=5e-4):
    """Single-sample weak residual  R~(x, t; v)  of Eq. 12 / Eq. 30.

    Only first derivatives are needed: the divergence is replaced by the antithetic estimator
        (v/sigma)^T [ s(x+v) - s(x-v) ] / (2 sigma),   v ~ N(0, sigma^2 I)
    and d/dt log p by finite differences.
    """
    xp, xm = x + v, x - v
    s_p = model.score(xp, t, create_graph=True)
    s_m = model.score(xm, t, create_graph=True)
    div_est = ((v / sigma) * (s_p - s_m) / (2 * sigma)).sum(-1)
    b = sde.g2(t)
    gamma = (0.5 * b * (s_p ** 2).sum(-1) - (sde.drift(xp, t) * s_p).sum(-1)
             - sde.div_drift(xp, t) - dt_log_p_fd(model, xp, t, hs, hd))
    return 0.5 * b * div_est + gamma


def fp_loss(model, x, t, sigma=1e-4, hs=1e-3, hd=5e-4, weight=None, unbiased=True):
    """L_FP = lambda_FP(t) * D^-2 * ||R~||^2  (Eq. 13).

    unbiased=True uses R~(v) * R~(v') with two independent v, v' (unbiased for ||E_v R||^2 but
    can be negative per-sample); unbiased=False uses R~(v)^2 (biased upward, always >= 0).
    """
    D = x.shape[-1]
    v1 = sigma * torch.randn_like(x)
    r1 = fp_residual_weak(model, x, t, v1, sigma, hs, hd)
    if unbiased:
        v2 = sigma * torch.randn_like(x)
        r2 = fp_residual_weak(model, x, t, v2, sigma, hs, hd)
        sq = r1 * r2
    else:
        sq = r1 ** 2
    w = 1.0 if weight is None else weight(t)
    return (w * sq / D ** 2).mean()
