"""Training loop for Eq. 10:  L = L_DSM + alpha * L_FP,  t ~ U(t_range)."""
import math
import time
import torch
from .losses import dsm_loss, fp_loss


def train(model, data, steps=6000, batch_size=128, lr=3e-3, min_lr=1e-5, t_range=(1e-5, 1.0),
          fp_alpha=0.0, fp_kwargs=None, fp_batch_size=None, log_every=1000, seed=0, verbose=True, augment=None):
    """Train `model` on `data` (N, D) (already normalised).

    fp_alpha > 0 adds the Fokker-Planck regulariser (needs an energy-based model).
    t is sampled uniformly in `t_range` -> use e.g. (1e-5, 0.1) to train a small-t MoE expert.
    The FP term needs log p at t - hs, so its t is clamped to >= hs + 1e-5.
    augment: optional callable applied to each batch (e.g. random rotations for molecules).
    Returns a dict of loss histories.
    """
    torch.manual_seed(seed)
    fp_kwargs = fp_kwargs or {}
    hs = fp_kwargs.get("hs", 1e-3)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=min_lr)
    t0, t1 = t_range
    hist = {"dsm": [], "fp": []}
    model.train()
    start = time.time()
    for step in range(1, steps + 1):
        x0 = data[torch.randint(0, len(data), (batch_size,))]
        if augment is not None:
            x0 = augment(x0)
        t = t0 + (t1 - t0) * torch.rand(batch_size, dtype=x0.dtype)
        loss = dsm_loss(model, x0, t)
        hist["dsm"].append(loss.item())
        if fp_alpha > 0:
            nb = fp_batch_size or batch_size
            # FP regulariser is evaluated on noised samples x(t) (Eq. 10: expectation over x(t)|x(0))
            from .sde import perturb
            tf = (t0 + (t1 - t0) * torch.rand(nb, dtype=x0.dtype)).clamp(min=hs + 1e-5)
            xt, _, _ = perturb(x0[:nb], tf)
            lfp = fp_loss(model, xt, tf, **fp_kwargs)
            hist["fp"].append(lfp.item())
            loss = loss + fp_alpha * lfp
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        sched.step()
        if verbose and (step % log_every == 0 or step == steps):
            msg = f"step {step:6d} | dsm {sum(hist['dsm'][-log_every:]) / len(hist['dsm'][-log_every:]):.4f}"
            if hist["fp"]:
                msg += f" | fp {sum(hist['fp'][-log_every:]) / len(hist['fp'][-log_every:]):.4f}"
            print(msg + f" | {time.time() - start:.0f}s")
    model.eval()
    return hist
