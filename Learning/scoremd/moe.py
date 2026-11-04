"""Time-based mixture of experts (paper Sec. 3.3, Eq. 16).

The diffusion time axis (0,1) is split into disjoint intervals I_0, I_1, ...; expert i is trained
only on t in I_i and the overall score is  s(x,t) = s_i(x,t) for t in I_i.
For simulation only expert 0 (the one containing t=0) is needed.
"""
import torch
import torch.nn as nn


class TimeMoE(nn.Module):
    def __init__(self, experts, boundaries):
        """experts: list of K models; boundaries: K-1 increasing split points, e.g. [0.1, 0.6]."""
        super().__init__()
        assert len(experts) == len(boundaries) + 1
        self.experts = nn.ModuleList(experts)
        self.register_buffer("boundaries", torch.tensor(boundaries, dtype=torch.get_default_dtype()))

    @property
    def conservative(self):
        return self.experts[0].conservative

    def expert_index(self, t):
        # t in [b_{i-1}, b_i) -> i  (matches intervals (0,0.1), [0.1,0.6), [0.6,1.0))
        return torch.bucketize(t, self.boundaries, right=True)

    def score(self, x, t, create_graph=None):
        idx = self.expert_index(t)
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = idx == i
            if mask.any():
                out[mask] = expert.score(x[mask], t[mask], create_graph=create_graph)
        return out

    def log_p(self, x, t):
        """Only defined where the active expert is energy-based (in practice: small t)."""
        idx = self.expert_index(t)
        if not (idx == 0).all():
            raise ValueError("log_p of the MoE is only available inside the first (energy-based) expert's interval")
        return self.experts[0].log_p(x, t)
