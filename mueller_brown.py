"""
Issue #1 - The Müller-Brown potential and its force.

The Müller-Brown potential is a famous 2D "toy landscape" used to test
simulation and machine-learning methods. It is a sum of 4 Gaussian-like bumps:

    U(x, y) = sum over k of  A[k] * exp( a[k]*(x - x0[k])**2
                                        + b[k]*(x - x0[k])*(y - y0[k])
                                        + c[k]*(y - y0[k])**2 )

The force is minus the slope of the landscape:  F = -dU/dx, -dU/dy.
We compute it with a formula we derived by hand (no automatic differentiation).
"""
import torch

# --- The 4 bumps that make up the landscape (standard textbook values) ---
A  = torch.tensor([-200.0, -100.0, -170.0, 15.0])   # height of each bump (negative = valley)
a  = torch.tensor([  -1.0,   -1.0,   -6.5,  0.7])   # how wide the bump is in x
b  = torch.tensor([   0.0,    0.0,   11.0,  0.6])   # how tilted the bump is (mixes x and y)
c  = torch.tensor([ -10.0,  -10.0,   -6.5,  0.7])   # how wide the bump is in y
x0 = torch.tensor([   1.0,    0.0,   -0.5, -1.0])   # x position of each bump's center
y0 = torch.tensor([   0.0,    0.5,    1.5,  1.0])   # y position of each bump's center


def _bump_values(points):
    """
    Helper used by both functions below.

    points: tensor of shape (N, 2), one (x, y) point per row.
    Returns dx, dy and the 4 exponential terms, each of shape (N, 4):
    one column per bump.
    """
    x = points[:, 0:1]          # shape (N, 1)
    y = points[:, 1:2]          # shape (N, 1)

    dx = x - x0                 # distance to each bump center in x, shape (N, 4)
    dy = y - y0                 # distance to each bump center in y, shape (N, 4)

    exponent = a * dx**2 + b * dx * dy + c * dy**2
    bump = A * torch.exp(exponent)   # value of each bump at each point, shape (N, 4)
    return dx, dy, bump


def mueller_brown_potential(points):
    """
    Potential energy U at each point.

    points: tensor of shape (N, 2)
    returns: tensor of shape (N,)
    """
    _, _, bump = _bump_values(points)
    energy = bump.sum(dim=1)         # add the 4 bumps together
    return energy


def mueller_brown_force(points):
    """
    Force F = -gradient of U, computed with a hand-derived formula.

    For one bump  A * exp(E)  with  E = a*dx^2 + b*dx*dy + c*dy^2 :
        d/dx  =  A * exp(E) * (2*a*dx + b*dy)
        d/dy  =  A * exp(E) * (b*dx + 2*c*dy)

    points: tensor of shape (N, 2)
    returns: tensor of shape (N, 2)
    """
    dx, dy, bump = _bump_values(points)

    dU_dx = (bump * (2 * a * dx + b * dy)).sum(dim=1)   # shape (N,)
    dU_dy = (bump * (b * dx + 2 * c * dy)).sum(dim=1)   # shape (N,)

    gradient = torch.stack([dU_dx, dU_dy], dim=1)       # shape (N, 2)
    force = -gradient                                    # force points downhill
    return force


def check_force_with_autograd(points):
    """
    Compare our hand-made force with PyTorch's automatic derivative.
    Returns the largest difference found (should be almost zero).
    """
    points = points.clone().requires_grad_(True)
    energy = mueller_brown_potential(points).sum()
    (gradient,) = torch.autograd.grad(energy, points)
    autograd_force = -gradient

    our_force = mueller_brown_force(points.detach())
    largest_difference = (our_force - autograd_force).abs().max().item()
    return largest_difference
