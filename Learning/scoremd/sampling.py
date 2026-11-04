"""iid sampling by integrating the reverse-time SDE (Eq. 2) with Euler-Maruyama."""
import math
import torch
from . import sde


@torch.no_grad()
def sample_reverse_sde(model, n, dim, steps=1000, t_min=1e-5, t_max=1.0, generator=None):
    """x_{t-dt} = x + [ 1/2 beta x + beta * s(x,t) ] dt + sqrt(beta dt) z   (no noise on the last step)."""
    dtype = torch.get_default_dtype()
    ts = torch.linspace(t_max, t_min, steps + 1, dtype=dtype)
    x = torch.randn(n, dim, generator=generator, dtype=dtype)
    for i in range(steps):
        t, dt = ts[i], ts[i] - ts[i + 1]
        tb = t.repeat(n)
        b = sde.beta(t)
        s = model.score(x, tb, create_graph=False)
        x = x + (0.5 * b * x + b * s) * dt
        if i < steps - 1:
            x = x + torch.sqrt(b * dt) * torch.randn(x.shape, generator=generator, dtype=dtype)
    return x
