"""States, transitions and correlation times of the alanine dipeptide trajectory.
Angles are in radians; state boundaries are given in degrees.
"""
import numpy as np

STATE_NAMES = ("beta/PII", "alphaR", "alphaL", "C7ax")
STATE_LABELS = {
    "beta/PII": r"$\beta$/P$_\mathrm{II}$",
    "alphaR": r"$\alpha_\mathrm{R}$",
    "alphaL": r"$\alpha_\mathrm{L}$",
    "C7ax": r"C$_\mathrm{7ax}$",
}
PSI_ALPHA_R = (-120.0, 50.0)   # phi < 0: alphaR inside this psi range, beta/PII outside
PSI_ALPHA_L = (-30.0, 120.0)   # phi > 0: alphaL inside, C7ax outside
CORE_MAIN = (-170.0, -30.0)    # phi cores for counting transitions
CORE_RARE = (20.0, 120.0)


def wrap(angle):
    """Wrap to (-pi, pi]."""
    return -((-angle + np.pi) % (2 * np.pi) - np.pi)


def assign_states(phi, psi):
    """State index 0..3 per frame (order of STATE_NAMES)."""
    phi_d, psi_d = np.degrees(phi), np.degrees(psi)
    left = phi_d < 0
    in_ar = (psi_d > PSI_ALPHA_R[0]) & (psi_d < PSI_ALPHA_R[1])
    in_al = (psi_d > PSI_ALPHA_L[0]) & (psi_d < PSI_ALPHA_L[1])
    states = np.empty(len(phi_d), dtype=np.int8)
    states[left & ~in_ar] = 0
    states[left & in_ar] = 1
    states[~left & in_al] = 2
    states[~left & ~in_al] = 3
    return states


def state_populations(states, n_states=len(STATE_NAMES)):
    return np.bincount(states, minlength=n_states) / len(states)


def core_set_trajectory(phi):
    """0 = phi < 0 basin, 1 = phi > 0 basin.

    Frames between the cores keep the label of the last core, so short
    recrossings are not counted as transitions.
    """
    phi_d = np.degrees(phi)
    label = np.full(len(phi_d), -1, dtype=np.int8)
    label[(phi_d > CORE_MAIN[0]) & (phi_d < CORE_MAIN[1])] = 0
    label[(phi_d > CORE_RARE[0]) & (phi_d < CORE_RARE[1])] = 1
    idx = np.where(label >= 0, np.arange(len(label)), 0)
    np.maximum.accumulate(idx, out=idx)
    filled = label[idx]
    first = np.argmax(label >= 0)
    filled[:first] = label[first]
    return filled


def transitions(core_traj):
    """Frames where the label changes, and the dwell times (frames) with their labels.
    The first and last stay are cut off by the trajectory ends and left out."""
    change = np.flatnonzero(np.diff(core_traj) != 0) + 1
    bounds = np.concatenate([[0], change, [len(core_traj)]])
    dwell = np.diff(bounds)
    labels = core_traj[bounds[:-1]]
    return change, dwell[1:-1], labels[1:-1]


def autocorrelation(x, max_lag=None):
    """Normalised autocorrelation function, computed with the FFT."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    max_lag = n // 10 if max_lag is None else max_lag
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:max_lag + 1]
    return acf / acf[0]


def integrated_autocorrelation_time(x, max_lag=None, c=5.0):
    """tau in frames (Sokal's window); N / (2 tau) is the number of independent samples."""
    acf = autocorrelation(x, max_lag)
    taus = 0.5 + np.cumsum(acf[1:])
    window = np.arange(1, len(acf))
    ok = window >= c * taus
    m = np.argmax(ok) if ok.any() else len(taus) - 1
    return float(taus[m])


def block_populations(states, n_blocks=10, n_states=len(STATE_NAMES)):
    """Populations in consecutive blocks, shape (n_blocks, n_states)."""
    return np.stack([state_populations(b, n_states) for b in np.array_split(states, n_blocks)])


def chirality(xyz, n_idx=6, ca_idx=8, c_idx=14, cb_idx=10):
    """Sign of the volume spanned by N, C, CB around CA (L or D alanine)."""
    ca = xyz[:, ca_idx]
    v = np.einsum("ij,ij->i", xyz[:, n_idx] - ca, np.cross(xyz[:, c_idx] - ca, xyz[:, cb_idx] - ca))
    return np.sign(v)


def radius_of_gyration(xyz):
    centred = xyz - xyz.mean(axis=1, keepdims=True)
    return np.sqrt((centred ** 2).sum(-1).mean(-1))
