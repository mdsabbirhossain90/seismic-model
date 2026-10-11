"""
thesis_figures.py — final comparison figures and numbers for Chapter 2
=======================================================================
Reads the finished no-PML runs and writes thesis-ready figures (PNG 300 dpi
+ vector PDF) matching the figure plan of Chapter 2, plus CSV tables with the
numbers needed for the [Insert ...] placeholders.

  Fig2_4   Homogeneous: (a) vz snapshots, (b) seismograms
  Fig2_5   Hetero a=0.3 km, eps=0.05: same layout, same colour scale as 2.4
  Fig2_6   eps = 0.01 (a = 1 km): (a) Vp model, (b) snapshots, (c) seismograms
  Fig2_7   eps = 0.05 (a = 1 km): same layout and colour scale as 2.6
  Fig2_8   eps = 0.10 (a = 1 km): same layout and colour scale as 2.6
  Fig2_9   Vp models for a = 0.3, 1, 5 km (eps = 0.05), common colour scale
  Fig2_10  (a)-(c) snapshots at one time for a = 0.3, 1, 5 km,
           (d) radial and (e) transverse seismograms at receiver C
  Fig2_11  Anomaly: (a) Case 3 snapshots, (b) Case 3 - Case 2.1, (c) seismograms
  FigS1    eps-series overview grid (3 models x 4 times)
  FigS2    a-series overview grid (3 models x 4 times)
  FigS3    Energy in the 40x40 km area and S/P energy ratio vs time

  Table2_4_velocity_stats.csv          Vp/Vs statistics in the 40x40 km area
  metrics_summary.csv                  dt, direct-wave amplitude, traveltime shift,
                                       coda/direct ratio, S/P energy ratio per case
  metrics_anomaly_vs_background.csv    Case 3 relative to Case 2.1 per receiver
  metrics_definitions.txt              exact definition of every number

Usage (from the noPML_50km folder):
  python thesis_figures.py
  python thesis_figures.py --comp P          # snapshots of div(v) instead of vz
  python thesis_figures.py --wave-view main  # crop snapshots to 40x40 km
  python thesis_figures.py --t-compare 6     # Fig 2.10 at 6 s instead of 4 s
"""

import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── fixed experiment constants (identical for all runs) ─────────────────
VP0, VS0, F0 = 6000.0, 3464.0, 4.78
T0 = 1.5 / F0                       # Ricker delay
ANOM_BOX = (18000.0, 26000.0, 16000.0, 24000.0)

CASES = {
    "1":   dict(folder="1_homogeneous",                    label="Homogeneous",          a=None, eps=0.00),
    "2.1": dict(folder="2_hetero_a0.3km_eps0.05",           label="a = 0.3 km, ε = 0.05", a=0.3,  eps=0.05),
    "2.2": dict(folder="2.2_hetero_a1km_eps0.05",           label="a = 1 km, ε = 0.05",   a=1.0,  eps=0.05),
    "2.3": dict(folder="2.3_hetero_a5km_eps0.05",           label="a = 5 km, ε = 0.05",   a=5.0,  eps=0.05),
    "2.4": dict(folder="2.4_hetero_a1km_eps0.01",           label="a = 1 km, ε = 0.01",   a=1.0,  eps=0.01),
    "2.5": dict(folder="2.5_hetero_a1km_eps0.1",            label="a = 1 km, ε = 0.10",   a=1.0,  eps=0.10),
    "3":   dict(folder="3_anomaly_a0.3km_eps0.05_anom0.09", label="a = 0.3 km, ε = 0.05 + anomaly ε = 0.09",
                a=0.3, eps=0.05, eps_anom=0.09),
}
EPS_SERIES = ["2.4", "2.2", "2.5"]
A_SERIES   = ["2.1", "2.2", "2.3"]
SERIES_COLORS = ["#0072B2", "#E69F00", "#D55E00"]       # colour-blind safe
COMP_LABEL = {"vx": r"$v_x$", "vz": r"$v_z$", "P": r"$\nabla\cdot\mathbf{v}$",
              "S": r"$(\nabla\times\mathbf{v})_y$"}

# ── thesis style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "axes.linewidth": 0.7, "lines.linewidth": 0.9,
    "savefig.dpi": 300, "figure.dpi": 100, "mathtext.fontset": "dejavuserif",
})
FULL_W = 6.3        # inches = 16 cm, full text width


# ═════════════════════════════════════════════════════════════════════════
# loading
# ═════════════════════════════════════════════════════════════════════════
def load_case(root, key):
    d = dict(CASES[key]); d["key"] = key
    folder = os.path.join(root, d["folder"]); d["dir"] = folder
    need = ["snapshots.npz", "energy.npz", "velocity_model.npz"]
    if not all(os.path.exists(os.path.join(folder, f)) for f in need):
        return None
    s = np.load(os.path.join(folder, "snapshots.npz"))
    d["snap"] = {k: s[k] for k in s.files}
    d["energy"] = dict(np.load(os.path.join(folder, "energy.npz")))
    d["vel"] = dict(np.load(os.path.join(folder, "velocity_model.npz")))
    d["traces"] = {}
    for n in d["snap"]["rec_names"]:
        f = os.path.join(folder, f"trace_{n}.npz")
        if os.path.exists(f):
            d["traces"][str(n)] = dict(np.load(f))
    if not d["traces"]:
        return None
    t = next(iter(d["traces"].values()))["t"]
    d["dt"] = float(t[1] - t[0]); d["nt"] = len(t)
    return d


def geometry(d):
    s = d["snap"]
    nz, nx = s["vz"].shape[1:]
    x0, z0 = s["origin"]; dx = float(s["dx"]); dz = float(s["dz"])
    return dict(nx=nx, nz=nz, x0=x0, z0=z0, dx=dx, dz=dz,
                extent=[x0/1e3, (x0+nx*dx)/1e3, (z0+nz*dz)/1e3, z0/1e3],
                box=s["main_box"], src=s["source"],
                rnames=[str(n) for n in s["rec_names"]], rpos=s["rec_pos"])


def crop_idx(g):
    b = g["box"]
    ix0 = int(round((b[0]-g["x0"])/g["dx"])); ix1 = int(round((b[1]-g["x0"])/g["dx"]))
    iz0 = int(round((b[2]-g["z0"])/g["dz"])); iz1 = int(round((b[3]-g["z0"])/g["dz"]))
    return slice(iz0, iz1), slice(ix0, ix1)


def snap_index(d, t):
    return int(np.argmin(np.abs(d["snap"]["times"] - t)))


def actual_snap_time(d, t):
    it = min(int(round(t / d["dt"])), d["nt"] - 1)
    return it * d["dt"]


# ═════════════════════════════════════════════════════════════════════════
# arrival-time markers
# ═════════════════════════════════════════════════════════════════════════
def arrivals(g, rx, rz):
    sx, sz = g["src"]
    dist = np.hypot(rx - sx, rz - sz)
    xa, xb = g["x0"], g["x0"] + (g["nx"]-1)*g["dx"]
    za, zb = g["z0"], g["z0"] + (g["nz"]-1)*g["dz"]
    imgs = [(2*xa - sx, sz), (2*xb - sx, sz), (sx, 2*za - sz), (sx, 2*zb - sz)]
    d_ref = min(np.hypot(rx - ix, rz - iz) for ix, iz in imgs)
    return dict(dist=dist, tP=T0 + dist/VP0, tS=T0 + dist/VS0, tR=T0 + d_ref/VP0)


# ═════════════════════════════════════════════════════════════════════════
# drawing helpers
# ═════════════════════════════════════════════════════════════════════════
def panel_label(ax, s, x=0.0, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontweight="bold", fontsize=9,
            ha="left" if x >= 0 else "right", va="bottom")


def group_clip(cases, t, comp, pct):
    """Common colour limit at time t: percentile of |field| inside the
    40x40 km area, maximised over all cases being compared."""
    vals = []
    for d in cases:
        g = geometry(d); cz, cx = crop_idx(g)
        f = d["snap"][comp][snap_index(d, t)][cz, cx]
        vals.append(np.percentile(np.abs(f), pct))
    return max(vals) + 1e-30


def draw_snapshot(ax, d, t, comp, clip, view, field=None, anomaly=False):
    g = geometry(d)
    f = d["snap"][comp][snap_index(d, t)] if field is None else field
    im = ax.imshow(f / clip, cmap="seismic", vmin=-1, vmax=1,
                   extent=g["extent"], aspect="equal", interpolation="bilinear")
    b = g["box"] / 1e3
    if view == "main":
        ax.set_xlim(b[0], b[1]); ax.set_ylim(b[3], b[2])
    else:
        ax.add_patch(patches.Rectangle((b[0], b[2]), b[1]-b[0], b[3]-b[2],
                     lw=0.6, ls="--", ec="k", fc="none"))
    if anomaly:
        ax.add_patch(patches.Rectangle((ANOM_BOX[0]/1e3, ANOM_BOX[2]/1e3),
                     (ANOM_BOX[1]-ANOM_BOX[0])/1e3, (ANOM_BOX[3]-ANOM_BOX[2])/1e3,
                     lw=0.7, ls=":", ec="#006400", fc="none"))
    ax.plot(*(g["src"]/1e3), marker="*", color="k", ms=5, mec="w", mew=0.3)
    for (rx, rz) in g["rpos"]:
        ax.plot(rx/1e3, rz/1e3, marker="v", color="k", ms=3.5, mec="w", mew=0.3)
    ax.text(0.03, 0.97, f"t = {t:.0f} s", transform=ax.transAxes, fontsize=7,
            va="top", ha="left", bbox=dict(fc="w", ec="none", alpha=0.75, pad=1))
    ax.text(0.97, 0.03, f"±{clip:.2e}", transform=ax.transAxes, fontsize=6,
            va="bottom", ha="right", bbox=dict(fc="w", ec="none", alpha=0.75, pad=1))
    ax.tick_params(length=2)
    return im


def vp_crop(d):
    v = d["vel"]; o = v["origin"]; dx = float(v["dx"]); b = v["main_box"]
    ix0 = int(round((b[0]-o[0])/dx)); ix1 = int(round((b[1]-o[0])/dx))
    iz0 = int(round((b[2]-o[1])/dx)); iz1 = int(round((b[3]-o[1])/dx))
    return v["vp"][iz0:iz1, ix0:ix1], b


def draw_vp(ax, d, vmin=None, vmax=None):
    vp, b = vp_crop(d)
    im = ax.imshow(vp, cmap="jet", vmin=vmin, vmax=vmax, aspect="equal",
                   extent=[b[0]/1e3, b[1]/1e3, b[3]/1e3, b[2]/1e3])
    ax.tick_params(length=2)
    return im


def seis_norm(cases, rname):
    """Normalisation per receiver: largest |vx| (radial component, which carries
    the direct P wave on the receiver line) over all cases in the group."""
    return max(np.abs(d["traces"][rname]["vx"]).max() for d in cases) + 1e-30


def draw_seis(ax, cases, rname, comp, norm, colors, labels, g, show_legend=False,
              ref=None):
    rpos = g["rpos"][g["rnames"].index(rname)]
    arr = arrivals(g, *rpos)
    if ref is not None:
        tr = ref["traces"][rname]
        ax.plot(tr["t"], tr[comp]/norm, color="0.65", lw=0.8, label="Homogeneous")
    for d, c, lab in zip(cases, colors, labels):
        tr = d["traces"][rname]
        ax.plot(tr["t"], tr[comp]/norm, color=c, lw=0.8, label=lab)
    ax.axvline(arr["tP"], color="#009E73", ls="--", lw=0.7)
    ax.axvline(arr["tS"], color="#CC79A7", ls="-.", lw=0.7)
    ax.axvspan(arr["tR"], 10.0, color="0.85", alpha=0.5, lw=0)
    ax.axhline(0, color="0.8", lw=0.4)
    ax.set_xlim(0, 10)
    ax.tick_params(length=2)
    ax.text(0.99, 0.95, f"RCV {rname} ({arr['dist']/1e3:.0f} km)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7)
    if show_legend:
        ax.legend(loc="center right", frameon=True, framealpha=0.9,
                  edgecolor="none", fontsize=6.5)
    return arr


def seis_block(fig, gs, cases, colors, labels, g, ref=None, first_label="(b)",
               norm_cases=None):
    """3 rows (receivers A, B, C) x 2 columns (radial vx | transverse vz)."""
    norm_cases = list(norm_cases or cases)
    names = g["rnames"]; axes = []
    for i, rn in enumerate(names):
        norm = seis_norm(norm_cases + ([ref] if ref is not None else []), rn)
        row = []
        for j, comp in enumerate(["vx", "vz"]):
            ax = fig.add_subplot(gs[i, j])
            draw_seis(ax, cases, rn, comp, norm, colors, labels, g,
                      show_legend=(i == 0 and j == 0), ref=ref)
            if i < len(names) - 1: ax.set_xticklabels([])
            else: ax.set_xlabel("Time (s)")
            if j == 0: ax.set_ylabel("Norm. amp.")
            if i == 0: ax.set_title("Radial ($v_x$)" if comp == "vx" else "Transverse ($v_z$)")
            lim = max(abs(np.array(ax.get_ylim()))); ax.set_ylim(-lim, lim)
            row.append(ax)
        axes.append(row)
    panel_label(axes[0][0], first_label, x=-0.12, y=1.08)
    return axes


def colorbar_shared(fig, im, axes, label):
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01, shrink=0.9)
    cb.set_label(label, fontsize=8); cb.ax.tick_params(labelsize=7, length=2)
    return cb


def save(fig, out_dir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"  - {name}.png / .pdf")


# ═════════════════════════════════════════════════════════════════════════
# figures
# ═════════════════════════════════════════════════════════════════════════
def fig_case(d, scale_group, name, args, with_model=False, ref=None, vp_lims=None):
    """Top: [Vp model] + snapshots at all saved times. Bottom: seismograms."""
    g = geometry(d)
    times = list(d["snap"]["times"])
    ntop = len(times) + (1 if with_model else 0)
    fig = plt.figure(figsize=(FULL_W, FULL_W * 1.0))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.6], hspace=0.2)
    top = outer[0].subgridspec(1, ntop, wspace=0.08)
    bot = outer[1].subgridspec(len(g["rnames"]), 2, hspace=0.12, wspace=0.18)
    lbl = iter("abcdefgh")
    if with_model:
        ax = fig.add_subplot(top[0, 0])
        im = draw_vp(ax, d, *(vp_lims or (None, None)))
        ax.set_xlabel("X (km)"); ax.set_ylabel("Depth (km)")
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.06, pad=0.22)
        cb.set_label(r"$V_P$ (m/s)", fontsize=7); cb.ax.tick_params(labelsize=6, length=2)
        cb.ax.locator_params(nbins=3)
        panel_label(ax, f"({next(lbl)})")
    snap_axes = []
    for k, t in enumerate(times):
        ax = fig.add_subplot(top[0, k + (1 if with_model else 0)])
        clip = group_clip(scale_group, t, args.comp, args.pct)
        im = draw_snapshot(ax, d, t, args.comp, clip, args.wave_view)
        ax.set_xlabel("X (km)")
        if k == 0 and not with_model: ax.set_ylabel("Depth (km)")
        else: ax.set_yticklabels([])
        snap_axes.append(ax)
    panel_label(snap_axes[0], f"({next(lbl)})")
    colorbar_shared(fig, im, snap_axes, f"{COMP_LABEL[args.comp]} / clip")
    seis_block(fig, bot, [d], ["k"], [d["label"]], g, ref=ref,
               first_label=f"({next(lbl)})", norm_cases=scale_group)
    save(fig, args.out, name)


def fig_vp_series(cases, name, args):
    vps = [vp_crop(d)[0] for d in cases]
    vmin = min(v.min() for v in vps); vmax = max(v.max() for v in vps)
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, FULL_W * 0.40))
    for i, (ax, d) in enumerate(zip(axes, cases)):
        im = draw_vp(ax, d, vmin, vmax)
        ax.set_xlabel("X (km)")
        if i == 0: ax.set_ylabel("Depth (km)")
        else: ax.set_yticklabels([])
        ax.set_title(f"a = {d['a']:g} km", fontsize=8)
        panel_label(ax, f"({'abc'[i]})")
    colorbar_shared(fig, im, list(axes), r"$V_P$ (m/s)")
    save(fig, args.out, name)


def fig_a_snapshots(cases, ref, name, args):
    g = geometry(cases[0]); t = args.t_compare
    fig = plt.figure(figsize=(FULL_W, FULL_W * 0.86))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.9], hspace=0.22)
    top = outer[0].subgridspec(1, 3, wspace=0.08)
    bot = outer[1].subgridspec(2, 1, hspace=0.22)
    clip = group_clip(cases, t, args.comp, args.pct)
    axs = []
    for i, d in enumerate(cases):
        ax = fig.add_subplot(top[0, i])
        im = draw_snapshot(ax, d, t, args.comp, clip, args.wave_view)
        ax.set_title(f"a = {d['a']:g} km", fontsize=8); ax.set_xlabel("X (km)")
        if i == 0: ax.set_ylabel("Depth (km)")
        else: ax.set_yticklabels([])
        panel_label(ax, f"({'abc'[i]})"); axs.append(ax)
    colorbar_shared(fig, im, axs, f"{COMP_LABEL[args.comp]} / clip")
    rc = g["rnames"][-1]
    norm = seis_norm(cases + [ref], rc)
    labels = [f"a = {d['a']:g} km" for d in cases]
    for j, comp in enumerate(["vx", "vz"]):
        ax = fig.add_subplot(bot[j, 0])
        draw_seis(ax, cases, rc, comp, norm, SERIES_COLORS, labels, g,
                  show_legend=(j == 0), ref=ref)
        ax.set_ylabel("Radial $v_x$" if comp == "vx" else "Transverse $v_z$")
        if j == 0: ax.set_xticklabels([])
        else: ax.set_xlabel("Time (s)")
        lim = max(abs(np.array(ax.get_ylim()))); ax.set_ylim(-lim, lim)
        panel_label(ax, f"({'de'[j]})", x=-0.06)
    save(fig, args.out, name)


def fig_anomaly(d3, d21, name, args):
    g = geometry(d3); times = list(d3["snap"]["times"])
    fig = plt.figure(figsize=(FULL_W, FULL_W * 1.32))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.7], hspace=0.2)
    r1 = outer[0].subgridspec(1, 4, wspace=0.08)
    r2 = outer[1].subgridspec(1, 4, wspace=0.08)
    bot = outer[2].subgridspec(len(g["rnames"]), 2, hspace=0.12, wspace=0.18)
    a1, a2 = [], []
    print("    snapshot timing check, Case 3 vs Case 2.1 (affects panel b only):")
    for k, t in enumerate(times):
        clip = group_clip([d3, d21], t, args.comp, args.pct)
        ax = fig.add_subplot(r1[0, k])
        im = draw_snapshot(ax, d3, t, args.comp, clip, args.wave_view, anomaly=True)
        diff = (d3["snap"][args.comp][snap_index(d3, t)]
                - d21["snap"][args.comp][snap_index(d21, t)])
        bx = fig.add_subplot(r2[0, k])
        draw_snapshot(bx, d3, t, args.comp, clip, args.wave_view, field=diff, anomaly=True)
        for axx in (ax, bx):
            if k == 0: axx.set_ylabel("Depth (km)")
            else: axx.set_yticklabels([])
        ax.set_xticklabels([]); bx.set_xlabel("X (km)")
        a1.append(ax); a2.append(bx)
        mis = abs(actual_snap_time(d3, t) - actual_snap_time(d21, t))
        print(f"      t = {t:.0f} s: time mismatch {mis*1e3:.2f} ms -> possible residual "
              f"ring on the direct wavefront up to ~{2*np.pi*F0*mis*100:.1f} % of its amplitude")
    panel_label(a1[0], "(a)"); panel_label(a2[0], "(b)")
    colorbar_shared(fig, im, a1, f"{COMP_LABEL[args.comp]} / clip")
    colorbar_shared(fig, im, a2, "Difference / clip")
    seis_block(fig, bot, [d21, d3], ["#0072B2", "#D55E00"],
               ["Background only (Case 2.1)", "With anomaly (Case 3)"], g,
               first_label="(c)")
    save(fig, args.out, name)


def fig_overview(cases, name, args, title_fn):
    times = list(cases[0]["snap"]["times"])
    fig, axes = plt.subplots(len(cases), len(times),
                             figsize=(FULL_W, FULL_W * 0.27 * len(cases) + 0.4))
    for j, t in enumerate(times):
        clip = group_clip(cases, t, args.comp, args.pct)
        for i, d in enumerate(cases):
            ax = axes[i, j]
            im = draw_snapshot(ax, d, t, args.comp, clip, args.wave_view)
            if j == 0: ax.set_ylabel(f"{title_fn(d)}\nDepth (km)", fontsize=8)
            else: ax.set_yticklabels([])
            if i == len(cases) - 1: ax.set_xlabel("X (km)")
            else: ax.set_xticklabels([])
    colorbar_shared(fig, im, list(axes.ravel()), f"{COMP_LABEL[args.comp]} / clip")
    save(fig, args.out, name)


def fig_energy(C, args):
    ref = C.get("1")
    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, FULL_W * 0.62), sharex=True)
    for row, (series, lab_fn) in enumerate([
            (EPS_SERIES, lambda d: f"ε = {d['eps']:g} (a = 1 km)"),
            (A_SERIES,   lambda d: f"a = {d['a']:g} km (ε = 0.05)")]):
        for k, c in zip(series, SERIES_COLORS):
            d = C.get(k)
            if d is None: continue
            e = d["energy"]
            axes[row, 0].semilogy(e["t"], e["E_tot"] + 1e-30, color=c, label=lab_fn(d))
            axes[row, 1].semilogy(e["t"], (e["E_S"] + 1e-30) / (e["E_P"] + 1e-30), color=c)
        if ref is not None:
            e = ref["energy"]
            axes[row, 0].semilogy(e["t"], e["E_tot"] + 1e-30, color="0.55", label="Homogeneous")
        axes[row, 0].set_ylabel("Energy in 40×40 km")
        axes[row, 1].set_ylabel(r"$E_S\,/\,E_P$")
        for ax in axes[row]:
            ax.axvspan(4.0, 10.0, color="0.85", alpha=0.5, lw=0)
            ax.grid(True, color="0.9", lw=0.4); ax.tick_params(length=2)
        axes[row, 0].legend(frameon=False, fontsize=6.5)
    for ax in axes[1]: ax.set_xlabel("Time (s)"); ax.set_xlim(0, 10)
    for ax, l in zip(axes.ravel(), "abcd"): panel_label(ax, f"({l})", x=-0.1)
    save(fig, args.out, "FigS3_energy_eps_and_a_series")


# ═════════════════════════════════════════════════════════════════════════
# numbers for the thesis
# ═════════════════════════════════════════════════════════════════════════
def window(tr, t1, t2):
    return (tr["t"] >= t1) & (tr["t"] <= t2)


def xcorr_shift(t_ref, x_ref, t_y, y, t1, t2, dt=5e-4, maxlag=0.3):
    """Delay of y relative to x_ref (s), positive = y later."""
    tt = np.arange(t1, t2, dt)
    a = np.interp(tt, t_ref, x_ref); b = np.interp(tt, t_y, y)
    n = len(tt); L = int(maxlag / dt)
    cc = np.array([np.dot(a[:n-k], b[k:]) if k >= 0 else np.dot(a[-k:], b[:n+k])
                   for k in range(-L, L + 1)])
    i = int(np.argmax(cc)); frac = 0.0
    if 0 < i < len(cc) - 1:
        y0, y1, y2 = cc[i-1], cc[i], cc[i+1]
        den = y0 - 2*y1 + y2
        frac = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    return (i - L + frac) * dt


def velocity_stats(d):
    v = d["vel"]; o = v["origin"]; dx = float(v["dx"]); b = v["main_box"]
    ix0 = int(round((b[0]-o[0])/dx)); ix1 = int(round((b[1]-o[0])/dx))
    iz0 = int(round((b[2]-o[1])/dx)); iz1 = int(round((b[3]-o[1])/dx))
    out = {}
    for k in ("vp", "vs"):
        a = v[k][iz0:iz1, ix0:ix1].astype(float)
        out.update({f"{k}_min": a.min(), f"{k}_max": a.max(),
                    f"{k}_mean": a.mean(), f"{k}_std": a.std()})
    if d["key"] == "3":
        X = o[0] + np.arange(v["vp"].shape[1]) * dx
        Z = o[1] + np.arange(v["vp"].shape[0]) * dx
        m = ((X[None, :] >= ANOM_BOX[0]) & (X[None, :] <= ANOM_BOX[1]) &
             (Z[:, None] >= ANOM_BOX[2]) & (Z[:, None] <= ANOM_BOX[3]))
        a = v["vp"][m].astype(float)
        out.update({"vp_min_anom": a.min(), "vp_max_anom": a.max(), "vp_std_anom": a.std()})
    return out


def direct_peak(tr, arr):
    w = window(tr, arr["tP"] - 0.4, arr["tP"] + 0.4)
    return float(np.sqrt((tr["vx"]**2 + tr["vz"]**2)[w].max()))


def metrics(C, args):
    ref = C.get("1")
    rows, stats_rows = [], []
    for key, d in C.items():
        if d is None: continue
        g = geometry(d)
        st = velocity_stats(d)
        stats_rows.append(dict(case=key, a_km=d["a"] if d["a"] else "", eps=d["eps"],
                               eps_anom=d.get("eps_anom", ""),
                               **{k: round(float(v), 1) for k, v in st.items()}))
        r = dict(case=key, label=d["label"], dt_ms=round(d["dt"]*1e3, 4))
        e = d["energy"]; i3 = int(np.argmin(np.abs(e["t"] - 3.0)))
        r["ES_over_EP_at_3s"] = float(e["E_S"][i3] / (e["E_P"][i3] + 1e-30))
        if ref is not None:
            er = ref["energy"]; j3 = int(np.argmin(np.abs(er["t"] - 3.0)))
            r["E_area_at_3s_rel_homog"] = float(e["E_tot"][i3] / (er["E_tot"][j3] + 1e-30))
        for rn, (rx, rz) in zip(g["rnames"], g["rpos"]):
            tr = d["traces"][rn]; arr = arrivals(g, rx, rz)
            amp2 = tr["vx"]**2 + tr["vz"]**2
            wd = window(tr, arr["tP"] - 0.4, arr["tP"] + 0.4)
            wc = window(tr, arr["tP"] + 0.5, arr["tP"] + 1.5)
            peak = direct_peak(tr, arr)
            r[f"{rn}_tP_predicted_s"] = round(arr["tP"], 3)
            r[f"{rn}_direct_peak"] = peak
            r[f"{rn}_coda_to_direct_energy"] = float(amp2[wc].mean() / (amp2[wd].mean() + 1e-30))
            r[f"{rn}_transverse_fraction_coda"] = float((tr["vz"][wc]**2).mean() / (amp2[wc].mean() + 1e-30))
            if ref is not None:
                trr = ref["traces"][rn]
                r[f"{rn}_direct_peak_rel_homog"] = peak / (direct_peak(trr, arr) + 1e-30)
                r[f"{rn}_traveltime_shift_ms"] = 1e3 * xcorr_shift(
                    trr["t"], trr["vx"], tr["t"], tr["vx"], arr["tP"] - 0.5, arr["tP"] + 0.5)
        rows.append(r)

    anom_rows = []
    d3, d21 = C.get("3"), C.get("2.1")
    if d3 is not None and d21 is not None:
        g = geometry(d3)
        for rn, (rx, rz) in zip(g["rnames"], g["rpos"]):
            arr = arrivals(g, rx, rz)
            t3, t2 = d3["traces"][rn], d21["traces"][rn]
            e3 = t3["vx"]**2 + t3["vz"]**2; e2 = t2["vx"]**2 + t2["vz"]**2
            late1, late2 = arr["tP"] + 1.0, min(arr["tR"] - 0.2, arr["tP"] + 4.0)
            s3 = (t3["S"]**2)[window(t3, arr["tP"], late2)].sum()
            s2 = (t2["S"]**2)[window(t2, arr["tP"], late2)].sum()
            anom_rows.append(dict(
                receiver=rn,
                direct_peak_ratio=direct_peak(t3, arr) / (direct_peak(t2, arr) + 1e-30),
                late_coda_energy_ratio=float(e3[window(t3, late1, late2)].mean()
                                             / (e2[window(t2, late1, late2)].mean() + 1e-30)),
                late_window_s=f"{late1:.2f}-{late2:.2f}",
                S_energy_ratio=float(s3 / (s2 + 1e-30))))

    def write(fn, rws):
        if not rws: return
        keys = []
        for rw in rws:
            for k in rw:
                if k not in keys: keys.append(k)
        with open(os.path.join(args.out, fn), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for rw in rws:
                w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in rw.items()})
        print(f"  - {fn}")

    write("Table2_4_velocity_stats.csv", stats_rows)
    write("metrics_summary.csv", rows)
    write("metrics_anomaly_vs_background.csv", anom_rows)
    with open(os.path.join(args.out, "metrics_definitions.txt"), "w") as f:
        f.write(METRIC_TEXT)
    print("  - metrics_definitions.txt")
    return stats_rows, rows, anom_rows


METRIC_TEXT = """Definitions of the numbers in the CSV files (no-PML runs)
=========================================================
Table2_4_velocity_stats.csv
  vp_* / vs_*   min, max, mean and standard deviation of the model inside the
                40 x 40 km model area (m/s). Case 3 also lists Vp min, max and
                std inside the anomaly box (18-26 km x 16-24 km).

metrics_summary.csv
  dt_ms                          time step of the run (ms)
  ES_over_EP_at_3s               S-energy / P-energy measure in the 40x40 km area at t = 3 s
                                 (measures proportional to (lambda+2mu)(div v)^2 and mu(curl v)^2)
  E_area_at_3s_rel_homog         total elastic energy in the 40x40 km area at 3 s,
                                 divided by the homogeneous value at 3 s
  <R>_tP_predicted_s             predicted direct P time: 0.31 s + distance / 6000 m/s
  <R>_direct_peak                peak of |v| = sqrt(vx^2 + vz^2) in tP +/- 0.4 s
  <R>_direct_peak_rel_homog      the same divided by the homogeneous value
  <R>_traveltime_shift_ms        delay of the direct P wave relative to the homogeneous run,
                                 from cross-correlation of vx in tP +/- 0.5 s
                                 (positive = arrives later than in the homogeneous model)
  <R>_coda_to_direct_energy      mean |v|^2 in [tP+0.5, tP+1.5] s divided by mean |v|^2 in tP +/- 0.4 s
  <R>_transverse_fraction_coda   share of vz^2 in that coda window (vx is radial on the receiver line)

metrics_anomaly_vs_background.csv   (Case 3 divided by Case 2.1, same receiver)
  direct_peak_ratio        peak |v| in tP +/- 0.4 s
  late_coda_energy_ratio   mean |v|^2 in [tP + 1 s, min(tP + 4 s, first boundary reflection - 0.2 s)]
  S_energy_ratio           sum of (curl v)^2 at the receiver from tP to the end of that window

All receiver windows end before the first boundary reflection at that receiver.
"""


# ═════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/noPML_50km")
    p.add_argument("--out", default=None)
    p.add_argument("--comp", default="vz", choices=["vx", "vz", "P", "S"],
                   help="field shown in the wavefield snapshots (default vz)")
    p.add_argument("--wave-view", default="full", choices=["full", "main"])
    p.add_argument("--t-compare", type=float, default=4.0)
    p.add_argument("--pct", type=float, default=99.0)
    args = p.parse_args()
    args.out = args.out or os.path.join(args.root, "thesis_figures")
    os.makedirs(args.out, exist_ok=True)

    print(f"Loading runs from {args.root}/")
    C = {}
    for k in CASES:
        C[k] = load_case(args.root, k)
        print(f"  Case {k:<4} {CASES[k]['folder']:<38} {'ok' if C[k] else 'MISSING'}")
    have = lambda *ks: all(C.get(k) is not None for k in ks)
    ref = C.get("1")
    print(f"\nWriting to {args.out}/")

    if have("1", "2.1"):
        grp = [C["1"], C["2.1"]]
        fig_case(C["1"], grp, "Fig2_4_homogeneous", args)
        fig_case(C["2.1"], grp, "Fig2_5_hetero_a0.3km_eps0.05", args, ref=ref)
    else:
        print("  (skip Fig 2.4 / 2.5: need Cases 1 and 2.1)")

    if have(*EPS_SERIES):
        grp = [C[k] for k in EPS_SERIES]
        vps = [vp_crop(d)[0] for d in grp]
        lims = (min(v.min() for v in vps), max(v.max() for v in vps))
        for k, n in zip(EPS_SERIES, ["Fig2_6_eps0.01_a1km", "Fig2_7_eps0.05_a1km",
                                     "Fig2_8_eps0.10_a1km"]):
            fig_case(C[k], grp, n, args, with_model=True, ref=ref, vp_lims=lims)
        fig_overview(grp, "FigS1_eps_series_overview", args, lambda d: f"ε = {d['eps']:g}")
    else:
        print("  (skip Fig 2.6-2.8: need Cases 2.4, 2.2, 2.5)")

    if have(*A_SERIES):
        grp = [C[k] for k in A_SERIES]
        fig_vp_series(grp, "Fig2_9_vp_models_a_series", args)
        if ref is not None:
            fig_a_snapshots(grp, ref, f"Fig2_10_wavefield_a_series_t{args.t_compare:g}s", args)
        fig_overview(grp, "FigS2_a_series_overview", args, lambda d: f"a = {d['a']:g} km")
    else:
        print("  (skip Fig 2.9 / 2.10: need Cases 2.1, 2.2, 2.3)")

    if have("3", "2.1"):
        fig_anomaly(C["3"], C["2.1"], "Fig2_11_anomaly", args)
    else:
        print("  (skip Fig 2.11: need Cases 3 and 2.1)")

    fig_energy(C, args)
    stats, rows, anom = metrics(C, args)

    print("\nTable 2.4 — velocity statistics in the 40x40 km area:")
    print(f"  {'Case':<5}{'Vp min':>8}{'Vp max':>8}{'Vp mean':>9}{'Vp std':>8}{'Vs min':>8}{'Vs max':>8}")
    for s in stats:
        print(f"  {s['case']:<5}{s['vp_min']:>8.0f}{s['vp_max']:>8.0f}{s['vp_mean']:>9.0f}"
              f"{s['vp_std']:>8.0f}{s['vs_min']:>8.0f}{s['vs_max']:>8.0f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
