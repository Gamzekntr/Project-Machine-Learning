"""scoremd: a small, readable PyTorch re-implementation of the core ideas of
"Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models"
(Plainer, Wu, Klein, Guennemann, Noe - NeurIPS 2025), for the 2D toy experiments.

Official code (JAX + PyTorch notebooks): https://github.com/noegroup/ScoreMD
"""
from . import sde, models, losses, sampling, simulate, metrics, moe, train, potentials, plotting, molecules, experiments  # noqa: F401
