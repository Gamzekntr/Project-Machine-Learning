"""Variance-Preserving (VP) SDE from Song et al. (2021), as used in the paper (Appendix A.1).

Forward SDE:   dx = f(x,t) dt + g(t) dw
    f(x,t) = -1/2 * beta(t) * x
    g(t)   = sqrt(beta(t))
    beta(t) = beta_min + t * (beta_max - beta_min),  (beta_min, beta_max) = (0.1, 20)

Transition kernel (Eq. 19-20):
    p_t(x(t) | x(0)) = N( m(t) * x(0),  s(t)^2 * I )
    m(t) = exp(-1/2 * int_0^t beta),   s(t)^2 = 1 - exp(-int_0^t beta)
"""
import torch

BETA_MIN = 0.1
BETA_MAX = 20.0


def beta(t):
    return BETA_MIN + t * (BETA_MAX - BETA_MIN)


def int_beta(t):
    """int_0^t beta(s) ds"""
    return BETA_MIN * t + 0.5 * t ** 2 * (BETA_MAX - BETA_MIN)


def marginal_mean_coef(t):
    """m(t): x(t) = m(t) x(0) + s(t) eps"""
    return torch.exp(-0.5 * int_beta(t))


def marginal_std(t):
    """s(t) (called sigma(t) in Eq. 4)"""
    return torch.sqrt(1.0 - torch.exp(-int_beta(t)))


def drift(x, t):
    """f(x, t) = -1/2 beta(t) x ; t has shape (B,)"""
    return -0.5 * beta(t)[:, None] * x


def g2(t):
    """g(t)^2 = beta(t)"""
    return beta(t)


def div_drift(x, t):
    """div_x f(x,t) = -1/2 beta(t) * D"""
    D = x.shape[-1]
    return -0.5 * beta(t) * D


def perturb(x0, t):
    """Sample x(t) ~ p_t(. | x0). Returns (x_t, eps, std)."""
    m = marginal_mean_coef(t)[:, None]
    s = marginal_std(t)[:, None]
    eps = torch.randn_like(x0)
    return m * x0 + s * eps, eps, s
