"""
Issue #2 - Underdamped Langevin integrator.

Langevin dynamics moves particles across an energy landscape. At every step a
particle feels three things:
  1. the force from the landscape (pushes it downhill),
  2. friction (slows it down),
  3. random kicks (heat; they let it climb over hills sometimes).

We use the same update rule as OpenMM (a popular simulation program):

    alpha = exp(-gamma * dt)
    v_new = alpha * v + (1 - alpha) / gamma * F / mass + sqrt(kT * (1 - alpha**2) / mass) * noise
    x_new = x + dt * v_new

`force_fn` can be ANY function that takes positions (N, 2) and returns forces
(N, 2). Today it is the Müller-Brown formula; later it can be a trained model.
"""
import math
import torch


def langevin_integrator(x0, v0, force_fn, dt, n_steps,
                        kT=1.0, gamma=1.0, mass=1.0,
                        save_every=1, seed=None):
    """
    Simulate underdamped Langevin dynamics.

    x0         : starting positions, shape (N, D) - N particles run side by side
    v0         : starting velocities, shape (N, D)
    force_fn   : function, positions (N, D) -> forces (N, D)
    dt         : time step (how far we jump in time each step)
    n_steps    : how many steps to take
    kT         : temperature (in energy units). Higher = more random kicks
    gamma      : friction coefficient
    mass       : particle mass
    save_every : keep only every k-th step (saves memory)
    seed       : number for the random generator, so results are repeatable

    returns    : trajectory of positions, shape (n_steps // save_every, N, D)
    """
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    # These numbers stay the same for every step, so compute them once.
    alpha = math.exp(-gamma * dt)
    noise_size = math.sqrt(kT * (1 - alpha**2) / mass)

    x = x0.clone()
    v = v0.clone()
    saved_positions = []

    for step in range(1, n_steps + 1):
        force = force_fn(x)
        noise = torch.randn(x.shape, generator=generator, dtype=x.dtype)

        v = alpha * v + (1 - alpha) / gamma * force / mass + noise_size * noise
        x = x + dt * v

        if step % save_every == 0:
            saved_positions.append(x.clone())

    trajectory = torch.stack(saved_positions)
    return trajectory


def random_velocities(n_particles, n_dims, kT, mass, seed=None):
    """
    Starting velocities that match the temperature
    (each component is random with spread sqrt(kT / mass)).
    """
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    return math.sqrt(kT / mass) * torch.randn(n_particles, n_dims, generator=generator)
