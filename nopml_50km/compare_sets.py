"""
compare_sets.py — side-by-side comparison of the three no-PML sets
==================================================================
Wavefields use a COMMON colour scale per time row (so amplitudes are
comparable across sets). Seismograms are overlaid on a common amplitude
scale per receiver.

Usage:
  python compare_sets.py                                   # default root, Vz
  python compare_sets.py --comp vx
  python compare_sets.py --root results/noPML_50km --view main
"""

import argparse, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

SETS = [
    ("1_homogeneous",                      "Homogeneous"),
    ("2_hetero_a0.3km_eps0.05",            "Hetero  a=0.3 km, ε=0.05"),
    ("3_anomaly_a0.3km_eps0.05_anom0.09",  "Hetero + anomaly  ε_anom=0.09"),
]
COLORS = ["black", "#0066cc", "#cc3300"]
ANOM_BOX = (18000.0, 26000.0, 16000.0, 24000.0)
COMP_LABEL = {"vx": "Vx (m/s)", "vz": "Vz (m/s)",
              "P": "P  ∇·v (1/s)", "S": "S  (∇×v)_y (1/s)"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/noPML_50km")
    p.add_argument("--comp", default="vz", choices=["vx", "vz", "P", "S"])
    p.add_argument("--view", default="full", choices=["full", "main"])
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--pct", type=float, default=99.0,
                   help="Percentile of |amplitude| used for the colour limit")
    args = p.parse_args()

    out_dir = os.path.join(args.root, "comparison")
    os.makedirs(out_dir, exist_ok=True)

    data = [np.load(os.path.join(args.root, f, "snapshots.npz")) for f, _ in SETS]
    d0 = data[0]
    times = d0["times"]
    nz, nx = d0[args.comp].shape[1:]
    x0, z0 = d0["origin"]; dx = float(d0["dx"]); dz = float(d0["dz"])
    extent = [x0/1000, (x0+nx*dx)/1000, (z0+nz*dz)/1000, z0/1000]
    mb = d0["main_box"] / 1000
    src = d0["source"] / 1000
    rnames, rpos = d0["rec_names"], d0["rec_pos"] / 1000

    # ── 1. Wavefield grid: rows = time, cols = set ───────────────────
    nt = len(times)
    fig, axes = plt.subplots(nt, 3, figsize=(16, 5.0*nt), squeeze=False)
    fig.suptitle(f"Wavefield comparison ({COMP_LABEL[args.comp]}) — no PML, "
                 f"50×50 km grid, common colour scale per row",
                 fontsize=15, fontweight="bold")
    for r, t in enumerate(times):
        vmax = max(np.percentile(np.abs(d[args.comp][r]), args.pct) for d in data) + 1e-30
        for c, (d, (_, label)) in enumerate(zip(data, SETS)):
            ax = axes[r, c]
            im = ax.imshow(d[args.comp][r], cmap="seismic", vmin=-vmax, vmax=vmax,
                           extent=extent, aspect="equal", interpolation="bilinear")
            ax.add_patch(patches.Rectangle((mb[0], mb[2]), mb[1]-mb[0], mb[3]-mb[2],
                         lw=1.0, ls="--", ec="black", fc="none"))
            if c == 2:
                ax.add_patch(patches.Rectangle(
                    (ANOM_BOX[0]/1000, ANOM_BOX[2]/1000),
                    (ANOM_BOX[1]-ANOM_BOX[0])/1000, (ANOM_BOX[3]-ANOM_BOX[2])/1000,
                    lw=1.0, ls=":", ec="green", fc="none"))
            ax.plot(*src, "r*", ms=10, markeredgecolor="k", markeredgewidth=0.4)
            for n, (rx, rz) in zip(rnames, rpos):
                ax.plot(rx, rz, "rv", ms=7, markeredgecolor="k", markeredgewidth=0.4)
                ax.annotate(str(n), (rx, rz), xytext=(4, -11),
                            textcoords="offset points", fontsize=7, color="darkred")
            if args.view == "main":
                ax.set_xlim(mb[0], mb[1]); ax.set_ylim(mb[3], mb[2])
            if r == 0:
                ax.set_title(label, fontsize=12, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"t = {t:.1f} s\n\nZ (km)", fontsize=11)
            if r == nt - 1:
                ax.set_xlabel("X (km)")
        fig.colorbar(im, ax=list(axes[r, :]), fraction=0.015, pad=0.01)\
           .set_label(COMP_LABEL[args.comp], fontsize=9)
    base = os.path.join(out_dir, f"wavefield_comparison_{args.comp}")
    fig.savefig(base + ".png", dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(base + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  - {base}.png / .pdf")

    # ── 2. Seismogram overlay per receiver ───────────────────────────
    comp = args.comp
    fig, axes = plt.subplots(len(rnames), 1, figsize=(12, 3.2*len(rnames)),
                             sharex=True, squeeze=False)
    fig.suptitle(f"Seismograms ({COMP_LABEL[comp]}) — three sets, common scale per receiver",
                 fontsize=13, fontweight="bold")
    for i, n in enumerate(rnames):
        ax = axes[i, 0]
        traces = [np.load(os.path.join(args.root, f, f"trace_{n}.npz")) for f, _ in SETS]
        peak = max(np.abs(tr[comp]).max() for tr in traces) or 1.0
        for tr, (_, label), col in zip(traces, SETS, COLORS):
            ax.plot(tr["t"], tr[comp] / peak, color=col, lw=0.9, label=label)
        ax.set_title(f"RCV.{n}", loc="left", fontsize=10, fontweight="bold")
        ax.set_ylabel("Amp / max")
        ax.grid(True, color="lightgray", lw=0.4)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")
    axes[-1, 0].set_xlabel("Time (s)")
    plt.tight_layout()
    base = os.path.join(out_dir, f"seismogram_comparison_{comp}")
    fig.savefig(base + ".png", dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(base + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  - {base}.png / .pdf")

    # ── 3. Energy in the 40x40 km area ───────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for (f, label), col in zip(SETS, COLORS):
        e = np.load(os.path.join(args.root, f, "energy.npz"))
        ax.semilogy(e["t"], e["E_tot"] + 1e-30, color=col, lw=1.3, label=label)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Energy in 40×40 km area")
    ax.set_title("Energy comparison (no PML)", fontweight="bold")
    ax.grid(True, which="both", color="lightgray", lw=0.4)
    ax.legend(fontsize=9)
    base = os.path.join(out_dir, "energy_comparison")
    fig.savefig(base + ".png", dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(base + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  - {base}.png / .pdf")


if __name__ == "__main__":
    main()
