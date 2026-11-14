"""Figures and numbers for the alanine dipeptide part of the report.

    python aldp_overview.py [--root ~/pml-data] [--out figures]

Writes aldp_overview.pdf, aldp_structures.pdf, aldp_timeseries_zoom.pdf, aldp_quality.pdf,
metrics_sample_size.pdf, aldp_numbers.tex (LaTeX macros) and aldp_summary.txt.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

import analysis
import metrics
import plotting
import matplotlib
import matplotlib.pyplot as plt
from data import DEFAULT_ROOT, load_aldp_arrays, split_indices
from dihedrals import BACKBONE_INDICES, PHI_INDICES, PSI_INDICES, compute_dihedrals, find_backbone_indices

KB = 0.0083144626  # kJ/(mol K)


def dihedrals_all(xyz, chunk=200_000):
    out = []
    for s in range(0, len(xyz), chunk):
        x = torch.from_numpy(np.ascontiguousarray(xyz[s:s + chunk]))
        out.append(compute_dihedrals(x).numpy())
    a = np.concatenate(out)
    return a[:, 0].astype(np.float64), a[:, 1].astype(np.float64)


def check_against_mdtraj(xyz, n=2000, seed=0):
    """Largest difference to mdtraj on random frames, in radians."""
    import mdtraj as md
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(xyz), size=min(n, len(xyz)), replace=False))
    frames = np.asarray(xyz[idx])
    top = md.Topology()
    res = top.add_residue("X", top.add_chain())
    for _ in range(frames.shape[1]):
        top.add_atom("X", md.element.carbon, res)
    ref = md.compute_dihedrals(md.Trajectory(frames, top), [PHI_INDICES, PSI_INDICES])
    ours = compute_dihedrals(torch.from_numpy(frames)).numpy()
    return float(np.abs(analysis.wrap(ours - ref)).max())


def sample_size_study(angles, ns, n_repeats=5, seed=0):
    """JS and PMF error between n frames of the first half and n random frames of the second half.

    iid: random frames, md: one contiguous segment. Both histograms use the same n,
    because the pseudocount/floor makes the metrics depend on the sample size.
    """
    rng = np.random.default_rng(seed)
    half = len(angles) // 2
    first, second = angles[:half], angles[half:]
    res = {k: np.zeros((len(ns), n_repeats)) for k in ("js_iid", "js_md", "pmf_iid", "pmf_md")}
    for i, n in enumerate(ns):
        for r in range(n_repeats):
            ref_counts = metrics.histogram_2d(second[rng.choice(len(second), size=n, replace=False)])
            iid = first[rng.choice(half, size=n, replace=False)]
            start = rng.integers(0, half - n + 1)
            md_seg = first[start:start + n]
            for tag, s in (("iid", iid), ("md", md_seg)):
                c = metrics.histogram_2d(s)
                res[f"js_{tag}"][i, r] = metrics.js_distance_counts(c, ref_counts)
                res[f"pmf_{tag}"][i, r] = metrics.pmf_error_counts(c, ref_counts)
    return res


def fmt_int(n):
    return f"{int(n):,}".replace(",", "{,}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="folder with the bgmol dataset")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "figures"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plotting.set_style()
    t0 = time.time()
    N = {}  # LaTeX macro name -> value

    # data
    d = load_aldp_arrays(args.root)
    xyz, t_ps, energies = d["xyz"], np.asarray(d["time"]), np.asarray(d["energies"], dtype=np.float64)
    n_frames, n_atoms = xyz.shape[0], xyz.shape[1]
    T = float(d["temperature"])
    kT = KB * T
    dt = np.diff(t_ps)
    assert tuple(find_backbone_indices(d["atom_names"], d["residue_names"])) == BACKBONE_INDICES, \
        "atom order differs from the hard-coded indices in dihedrals.py"
    print(f"loaded {n_frames} frames x {n_atoms} atoms ({time.time() - t0:.0f}s)")

    N["aldpNframes"] = fmt_int(n_frames)
    N["aldpNatoms"] = str(n_atoms)
    N["aldpNheavy"] = str(int(np.sum(d["elements"] != "H")))
    N["aldpDt"] = f"{np.median(dt):g}"
    N["aldpDtUnique"] = "yes" if np.allclose(dt, dt[0]) else "no"
    N["aldpTstart"] = f"{t_ps[0] / 1000:g}"
    N["aldpTotalTime"] = f"{(t_ps[-1] - t_ps[0] + np.median(dt)) / 1e6:.3g}"
    N["aldpTemperature"] = f"{T:g}"
    N["aldpkT"] = f"{kT:.3f}"
    N["aldpTimestep"] = f"{d.get('timestep_fs', float('nan')):g}"
    N["aldpFriction"] = f"{d.get('friction_per_ps', float('nan')):g}"
    N["aldpBox"] = f"{np.asarray(d.get('box', [np.nan]))[0]:.4g}"
    N["aldpBoxConstant"] = "yes" if d.get("box_constant", True) else "no"
    N["aldpUsesPBC"] = "yes" if d.get("uses_pbc", False) else "no"
    N["aldpEnergyMean"] = f"{energies.mean():.1f}"
    N["aldpEnergyStd"] = f"{energies.std():.1f}"
    N["aldpEnergyStdkT"] = f"{energies.std() / kT:.1f}"
    tenth = n_frames // 10
    N["aldpEnergyDrift"] = f"{energies[-tenth:].mean() - energies[:tenth].mean():.2f}"

    # dihedrals & states
    phi, psi = dihedrals_all(xyz)
    angles = np.stack([phi, psi], axis=1)
    N["aldpDihedralCheck"] = f"{check_against_mdtraj(xyz):.1e}"
    states = analysis.assign_states(phi, psi)
    pops = analysis.state_populations(states)
    for name, macro in zip(analysis.STATE_NAMES, ("Beta", "AlphaR", "AlphaL", "CSevenAx")):
        N[f"aldpPop{macro}"] = f"{100 * pops[analysis.STATE_NAMES.index(name)]:.2f}"
    p_pos = float(np.mean(phi > 0))
    N["aldpPopPhiPos"] = f"{100 * p_pos:.2f}"
    N["aldpDeltaFPhiPos"] = f"{-np.log(p_pos / (1 - p_pos)):.1f}" if 0 < p_pos < 1 else "n/a"

    core = analysis.core_set_trajectory(phi)
    change, dwell, dwell_lab = analysis.transitions(core)
    n_into_rare = int(np.sum(core[change] == 1))
    N["aldpNtransitions"] = str(n_into_rare)
    N["aldpDwellMain"] = f"{np.mean(dwell[dwell_lab == 0]) * np.median(dt) / 1000:.1f}" if np.any(dwell_lab == 0) else "n/a"
    N["aldpDwellRare"] = f"{np.mean(dwell[dwell_lab == 1]) * np.median(dt):.0f}" if np.any(dwell_lab == 1) else "n/a"

    # correlation, convergence
    ind_rare = (phi > 0).astype(float)
    ind_ar = (states == 1).astype(float)
    tau_rare = analysis.integrated_autocorrelation_time(ind_rare) if ind_rare.std() > 0 else float("nan")
    tau_ar = analysis.integrated_autocorrelation_time(ind_ar) if ind_ar.std() > 0 else float("nan")
    N["aldpTauPhiPos"] = f"{tau_rare * np.median(dt) / 1000:.2g}"   # ns
    N["aldpTauAlphaR"] = f"{tau_ar * np.median(dt):.2g}"            # ps
    N["aldpNeffPhiPos"] = fmt_int(round(n_frames / (2 * tau_rare), -1)) if np.isfinite(tau_rare) else "n/a"
    blocks = analysis.block_populations(states, n_blocks=10)
    rare_blocks = 100 * (blocks[:, 2] + blocks[:, 3])
    N["aldpPopPhiPosBlockMin"] = f"{rare_blocks.min():.2f}"
    N["aldpPopPhiPosBlockMax"] = f"{rare_blocks.max():.2f}"
    N["aldpPopPhiPosBlockStd"] = f"{rare_blocks.std(ddof=1):.2f}"
    for k, macro in ((0, "Beta"), (1, "AlphaR")):
        N[f"aldpPop{macro}BlockMin"] = f"{100 * blocks[:, k].min():.1f}"
        N[f"aldpPop{macro}BlockMax"] = f"{100 * blocks[:, k].max():.1f}"
    N["aldpNblocks"] = "10"
    N["aldpBlockLength"] = f"{n_frames / 10 * np.median(dt) / 1000:g}"  # ns

    splits = split_indices(n_frames)
    for name in ("train", "val", "test"):
        idx = splits[name]
        cap = name.capitalize()
        N[f"aldpN{cap}"] = fmt_int(len(idx))
        N[f"aldpPopPhiPos{cap}"] = f"{100 * np.mean(phi[idx] > 0):.2f}"
    # test vs. an equally large part of train
    test_idx = splits["test"]
    train_sub = np.random.default_rng(1).choice(splits["train"], size=len(test_idx), replace=False)
    N["aldpJSTrainTest"] = f"{metrics.js_distance(angles[train_sub], angles[test_idx]):.4f}"
    N["aldpPMFTrainTest"] = f"{metrics.pmf_error(angles[train_sub], angles[test_idx]):.3f}"
    # same with the full training block, to show the effect of different sizes
    N["aldpJSTrainTestUnequal"] = f"{metrics.js_distance(angles[splits['train']], angles[test_idx]):.4f}"

    # geometry & preprocessing
    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(n_frames, size=min(100_000, n_frames), replace=False))
    xs = np.asarray(xyz[sub], dtype=np.float64)
    centroid_all = np.asarray(xyz, dtype=np.float32).mean(axis=1)
    disp = np.linalg.norm(centroid_all - centroid_all.mean(axis=0), axis=1)
    N["aldpCentroidMaxDisp"] = f"{disp.max():.3g}"
    N["aldpCentroidStd"] = f"{centroid_all.std(axis=0).mean():.3g}"
    rg = analysis.radius_of_gyration(xs)
    N["aldpRgMean"] = f"{rg.mean():.3f}"
    N["aldpRgStd"] = f"{rg.std():.3f}"
    N["aldpSigmaAll"] = f"{np.sqrt(((xs - xs.mean(1, keepdims=True)) ** 2).mean()):.3f}"
    xcg = xs[:, list(BACKBONE_INDICES)]
    N["aldpSigmaCG"] = f"{np.sqrt(((xcg - xcg.mean(1, keepdims=True)) ** 2).mean()):.3f}"
    chir = analysis.chirality(np.asarray(xyz))
    N["aldpChiralFrac"] = f"{100 * np.mean(chir == chir[0]):.2f}"
    cn = np.linalg.norm(np.asarray(xyz[:, 14]) - np.asarray(xyz[:, 16]), axis=1) * 10  # Angstrom
    N["aldpBondCNMean"] = f"{cn.mean():.3f}"
    N["aldpBondCNStd"] = f"{cn.std():.3f}"

    # metrics: noise floors
    half = n_frames // 2
    N["aldpJSHalves"] = f"{metrics.js_distance(angles[:half], angles[half:]):.4f}"
    N["aldpPMFHalves"] = f"{metrics.pmf_error(angles[:half], angles[half:]):.3f}"
    w_phi, w_psi = metrics.marginal_wasserstein_angles(angles[:half], angles[half:])
    N["aldpWPhiHalves"] = f"{np.degrees(w_phi):.2f}"
    N["aldpWPsiHalves"] = f"{np.degrees(w_psi):.2f}"
    ns = [n for n in (1_000, 3_000, 10_000, 30_000, 50_000, 100_000, 300_000) if n <= half // 2]
    study = sample_size_study(angles, ns)
    if 50_000 in ns:
        i50 = ns.index(50_000)
        for k, macro in (("js_iid", "JSiidFifty"), ("js_md", "JSmdFifty"),
                         ("pmf_iid", "PMFiidFifty"), ("pmf_md", "PMFmdFifty")):
            fmt = "{:.4f}" if k.startswith("js") else "{:.3f}"
            N[f"aldp{macro}"] = fmt.format(study[k][i50].mean())

    # figures
    t_ns = (t_ps - t_ps[0]) / 1000.0

    # figure: Ramachandran + time series
    fig = plt.figure(figsize=(6.3, 2.75))
    gs = fig.add_gridspec(2, 5, width_ratios=[1.0, 0.07, 0.05, 0.62, 1.3], wspace=0.05, hspace=0.12)
    ax0 = fig.add_subplot(gs[:, 0])
    cax = fig.add_subplot(gs[:, 2])
    plotting.plot_ramachandran(ax0, phi, psi, vmax=10, cax=cax)
    ax0.set_title("(a)", loc="left")
    step = max(1, n_frames // 20_000)
    N["aldpTimeseriesStride"] = str(step)
    ax1 = fig.add_subplot(gs[0, 4])
    plotting.plot_angle_timeseries(ax1, t_ns[::step], phi[::step], r"$\phi$", color="C0")
    ax1.tick_params(labelbottom=False)
    ax1.set_title("(b)", loc="left")
    ax2 = fig.add_subplot(gs[1, 4], sharex=ax1)
    plotting.plot_angle_timeseries(ax2, t_ns[::step], psi[::step], r"$\psi$", color="C1")
    ax2.set_xlabel("simulation time / ns")
    ax2.set_xlim(0, t_ns[-1])
    fig.savefig(out / "aldp_overview.pdf")
    plt.close(fig)

    # figure: first 5 ns
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 1.9), sharex=True)
    zoom = min(n_frames, 5000)
    plotting.plot_angle_timeseries(axes[0], t_ns[:zoom], phi[:zoom], r"$\phi$", s=1.0)
    plotting.plot_angle_timeseries(axes[1], t_ns[:zoom], psi[:zoom], r"$\psi$", color="C1", s=1.0)
    axes[1].set_xlabel("simulation time / ns")
    axes[1].set_xlim(0, t_ns[zoom - 1])
    fig.savefig(out / "aldp_timeseries_zoom.pdf")
    plt.close(fig)

    # figure: representative structures
    fig = plt.figure(figsize=(6.3, 1.9))
    for k, name in enumerate(analysis.STATE_NAMES):
        ax = fig.add_subplot(1, 4, k + 1, projection="3d")
        mask = np.flatnonzero(states == k)
        if len(mask) == 0:
            ax.set_axis_off()
            continue
        H, xe, ye = np.histogram2d(np.degrees(phi[mask]), np.degrees(psi[mask]), bins=36,
                                   range=[[-180, 180], [-180, 180]])
        i, j = np.unravel_index(np.argmax(H), H.shape)
        centre = np.radians([0.5 * (xe[i] + xe[i + 1]), 0.5 * (ye[j] + ye[j + 1])])
        dist = np.abs(analysis.wrap(angles[mask] - centre)).sum(1)
        frame = mask[np.argmin(dist)]
        plotting.plot_molecule(ax, xyz[frame], d["elements"], d["bonds"], highlight=BACKBONE_INDICES)
        ax.set_title(f"{analysis.STATE_LABELS[name]}\n"
                     rf"$\phi={np.degrees(phi[frame]):.0f}^\circ$, $\psi={np.degrees(psi[frame]):.0f}^\circ$",
                     fontsize=8)
    fig.subplots_adjust(wspace=0.0, left=0, right=1, top=0.82, bottom=0)
    fig.savefig(out / "aldp_structures.pdf")
    plt.close(fig)

    # figure: data quality (blocks + autocorrelation)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.3, 2.2), gridspec_kw=dict(wspace=0.35))
    block_mid = (np.arange(10) + 0.5) * t_ns[-1] / 10
    for k, name in enumerate(analysis.STATE_NAMES):
        ax0.plot(block_mid, 100 * blocks[:, k], "o-", ms=3, label=analysis.STATE_LABELS[name])
    for name, c in (("val", "0.88"), ("test", "0.75")):
        idx = splits[name]
        ax0.axvspan(t_ns[idx[0]], t_ns[idx[-1]], color=c, zorder=0, lw=0)
        ax0.text(0.5 * (t_ns[idx[0]] + t_ns[idx[-1]]), 1.02, name, transform=ax0.get_xaxis_transform(),
                 ha="center", fontsize=7)
    ax0.set_yscale("log")
    ax0.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
    ax0.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax0.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax0.set_xlabel("simulation time / ns (block centre)")
    ax0.set_ylabel("population / %")
    ax0.set_xlim(0, t_ns[-1])
    ax0.set_title("(a)", loc="left")
    fig.legend(*ax0.get_legend_handles_labels(), ncol=4, frameon=False, fontsize=8,
               loc="upper center", bbox_to_anchor=(0.29, -0.02))
    max_lag = min(n_frames // 10, 200_000)
    lags = np.arange(max_lag + 1) * np.median(dt)
    for series, lab, c in ((ind_rare, r"$\mathbb{1}[\phi>0]$", "C3"), (ind_ar, r"$\mathbb{1}[\alpha_\mathrm{R}]$", "C1")):
        if series.std() > 0:
            ax1.plot(lags[1:], analysis.autocorrelation(series, max_lag)[1:], label=lab, color=c)
    ax1.set_xscale("log")
    ax1.axhline(0, color="0.6", lw=0.6)
    ax1.set_xlabel("lag time / ps")
    ax1.set_ylabel("autocorrelation")
    ax1.legend(frameon=False)
    ax1.set_title("(b)", loc="left")
    fig.savefig(out / "aldp_quality.pdf")
    plt.close(fig)

    # figure: metrics vs number of samples
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.2), gridspec_kw=dict(wspace=0.35))
    for ax, key, ylabel, halves in ((axes[0], "js", "JS distance", N["aldpJSHalves"]),
                                    (axes[1], "pmf", r"PMF error / $(k_\mathrm{B}T)^2$", N["aldpPMFHalves"])):
        for tag, lab, c in (("iid", "random frames", "C0"), ("md", "contiguous MD segment", "C3")):
            v = study[f"{key}_{tag}"]
            ax.errorbar(ns, v.mean(1), yerr=v.std(1), fmt="o-", ms=3, capsize=2, color=c, label=lab)
        if float(halves) > 0:
            ax.axhline(float(halves), color="0.4", ls="--", lw=0.8, label="1st vs. 2nd half (500k each)")
        ax.axvline(50_000, color="0.7", ls=":", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("number of samples $n$")
        ax.set_ylabel(ylabel)
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].set_title("(a)", loc="left")
    axes[1].set_title("(b)", loc="left")
    fig.savefig(out / "metrics_sample_size.pdf")
    plt.close(fig)

    # numbers
    with open(out / "aldp_numbers.tex", "w") as f:
        f.write("% written by aldp_overview.py\n")
        for k, v in N.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    with open(out / "aldp_summary.txt", "w") as f:
        for k, v in N.items():
            f.write(f"{k:24s} {v.replace('{,}', ',')}\n")
        f.write("\nsample-size study (mean over repeats):\n")
        f.write("n        " + "  ".join(f"{k:>9s}" for k in study) + "\n")
        for i, n in enumerate(ns):
            f.write(f"{n:<8d} " + "  ".join(f"{study[k][i].mean():9.4f}" for k in study) + "\n")
    print((out / "aldp_summary.txt").read_text())
    print(f"done in {time.time() - t0:.0f}s, files in {out}")


if __name__ == "__main__":
    main()
