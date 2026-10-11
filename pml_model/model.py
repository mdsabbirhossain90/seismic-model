"""
model.py — 2D Elastic Simulation with optional anomaly region
==============================================================
NEW: --anomaly mode  builds a background medium (eps-bg) with a square
                     scattering anomaly (eps-anom) at a configurable bbox.

Run examples:
  python model.py --anomaly --corr 6   --output-dir results/anom_a0.3km
  python model.py --anomaly --corr 40  --output-dir results/anom_a2km
  python model.py --anomaly --corr 200 --output-dir results/anom_a10km
"""

import argparse, time, os
import numpy as np
import matplotlib.pyplot as plt

from geometry       import load_receivers, geometry_summary, plot_geometry
from velocity_model import (heterogeneous_random, heterogeneous_with_anomaly,
                             homogeneous, plot_velocity_model)
from elastic_solver import ElasticSolver2D


# Domain
NX, NZ   = 800, 800
DX, DZ   = 50.0, 50.0
VP_MEAN  = 6000.0
VS_MEAN  = 3464.0
RHO_MEAN = 2700.0

# Source / receivers per Fig 3.3
SOURCE   = (12000.0, 20000.0)
RECEIVERS = [
    ("A", 15000.0, 20000.0),     # BEFORE  the anomaly (3 km from source)
    ("B", 22000.0, 20000.0),     # WITHIN  the anomaly (10 km from source)
    ("C", 30000.0, 20000.0),     # BEHIND  the anomaly (18 km from source)
]

# Anomaly box — covers RCV.B, between RCV.A and RCV.C
# 8 km × 8 km box centred on (22, 20) km
ANOM_X = (18000.0, 26000.0)      # X-range in metres
ANOM_Z = (16000.0, 24000.0)      # Z-range in metres

# Sim params
F0       = 4.78
T_TOTAL  = 10.0
SNAPS    = [2.0, 4.0, 6.0, 8.0]
SEED     = 42
VS_FLOOR = 1800.0


def sep(c="=", n=72): print(c*n)


def run_single(model_type, perturbation, corr_cells, seed, output_dir,
               snap_times=None, geometry_only=False, order=4,
               psdf='exp', t_total=T_TOTAL, f0=F0,
               eps_bg=0.0, eps_anom=0.10, anomaly=False):
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()
    sep()
    if anomaly:
        print(f"  ANOMALY MODE  |  PSDF={psdf}  |  eps_bg={eps_bg}  "
              f"eps_anom={eps_anom}  a={corr_cells*DX:.0f}m")
    else:
        print(f"  {model_type}  |  PSDF={psdf}  |  ε={perturbation}  "
              f"a={corr_cells*DX:.0f}m")
    print(f"  T={t_total}s  f0={f0}Hz")
    sep()

    # Geometry
    receivers = load_receivers(RECEIVERS)
    distances = geometry_summary(SOURCE, receivers)
    plot_geometry(SOURCE, receivers, distances, output_dir=output_dir)

    # Velocity model
    anomaly_bbox = None
    if anomaly:
        vp, vs, rho, anomaly_bbox = heterogeneous_with_anomaly(
            NX, NZ, dx=DX, dz=DZ,
            vp_mean=VP_MEAN, vs_mean=VS_MEAN, rho_mean=RHO_MEAN,
            eps_bg=eps_bg, eps_anom=eps_anom,
            corr_x=corr_cells, corr_z=corr_cells,
            seed=seed, psdf=psdf,
            anom_x=ANOM_X, anom_z=ANOM_Z,
            vs_floor=VS_FLOOR,
        pml_cells=40,
        )
        title = (f"Anomaly  PSDF={psdf}  ε_bg={eps_bg}  ε_anom={eps_anom}  "
                 f"a={corr_cells*DX/1000:.1f} km")
    elif model_type == "homogeneous" or perturbation == 0.0:
        vp, vs, rho = homogeneous(NX, NZ, vp=VP_MEAN, rho=RHO_MEAN)
        title = f"Homogeneous  Vp={VP_MEAN:.0f}  Vs={VS_MEAN:.0f} m/s"
    else:
        vp, vs, rho = heterogeneous_random(
            NX, NZ, dx=DX, dz=DZ,
            vp_mean=VP_MEAN, vs_mean=VS_MEAN, rho_mean=RHO_MEAN,
            perturbation=perturbation,
            corr_x=corr_cells, corr_z=corr_cells,
            seed=seed, psdf=psdf, vs_floor=VS_FLOOR,
        pml_cells=40,
        )
        title = (f"Heterogeneous  PSDF={psdf}  ε={perturbation}  "
                 f"a={corr_cells*DX/1000:.1f} km")

    print(f"  Vp  {vp.min():.0f}–{vp.max():.0f}  mean={vp.mean():.0f} m/s")
    print(f"  Vs  {vs.min():.0f}–{vs.max():.0f}  mean={vs.mean():.0f} m/s")
    plot_velocity_model(vp, vs, DX, DZ, SOURCE, RECEIVERS,
                        output_dir=output_dir, title=title,
                        anomaly_bbox=anomaly_bbox)

    if geometry_only:
        print(f"\nDone (geometry only).  {time.time()-t0:.1f}s")
        return None

    # Solver
    solver = ElasticSolver2D(
    nx=NX, nz=NZ,
    dx=DX, dz=DZ,
    vp=vp, vs=vs, rho=rho,
    pml=40,
    order=order,
    f0_pml=f0
)
    solver.add_source(*SOURCE)
    for name, rx, rz in RECEIVERS:
        solver.add_receiver(name, rx, rz)

    src_type = "ricker"
    snaps = snap_times if snap_times else SNAPS
    print(f"\n[Sim] T={t_total}s f0={f0}Hz source={src_type}")
    snapshots, times, E_tot, E_P, E_S = solver.run(
        t_total=t_total, f0=f0,
        snap_times_s=snaps, output_dir=output_dir,
        source_type=src_type,
    )

    plot_energy(times, E_tot, E_P, E_S, title, output_dir)
    np.savez(os.path.join(output_dir, "energy.npz"),
             t=np.asarray(times), E_tot=np.asarray(E_tot),
             E_P=np.asarray(E_P), E_S=np.asarray(E_S))

    elapsed = time.time() - t0
    sep()
    print(f"  Outputs → '{output_dir}/'   ({elapsed/60:.1f} min)")
    sep()
    return times, E_tot, E_P, E_S


def smooth(x, w=40):
    out = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        out[i] = x[max(0, i-w+1):i+1].mean()
    return out

def plot_energy(times, E_tot, E_P, E_S, label, output_dir):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=False)
    fig.patch.set_facecolor("white")
    fig.suptitle(f"Domain Energy — {label}", fontsize=12, fontweight="bold")
    panels = [(E_tot, "Total", "-", "black"),
              (E_P,   "P-wave  ½ρVp²|div v|²", "--", "#0066aa"),
              (E_S,   "S-wave  ½ρVs²|curl v|²", "-.", "#aa0033")]
    times = np.asarray(times)
    for row, (E, name, ls, color) in enumerate(panels):
        E = np.asarray(E)
        # Left: full range, log scale
        axes[row, 0].semilogy(times, E + 1e-30, color=color, lw=1.4, ls=ls)
        axes[row, 0].set_ylabel("E (log)")
        axes[row, 0].set_title(f"{name}  — full", fontsize=10, loc="left")
        axes[row, 0].grid(True, which="both", color="lightgray", lw=0.4)
        # Right: coda window (t > 3s), linear
        mask = times > 3.0
        axes[row, 1].plot(times[mask], E[mask], color=color, lw=1.4, ls=ls)
        axes[row, 1].set_ylabel("E")
        axes[row, 1].set_title(f"{name}  — coda (t > 3s)", fontsize=10, loc="left")
        axes[row, 1].grid(True, color="lightgray", lw=0.4)
    axes[-1, 0].set_xlabel("Time (s)"); axes[-1, 1].set_xlabel("Time (s)")
    plt.tight_layout()
    out = os.path.join(output_dir, "energy.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  - {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="heterogeneous",
                    choices=["homogeneous", "heterogeneous"])
    p.add_argument("--psdf", default="exp", choices=["exp", "gauss"])
    p.add_argument("--perturbation", type=float, default=0.10)
    p.add_argument("--corr", type=int, default=20,
                    help="Correlation length in cells (dx=50m → 6=0.3km, "
                         "40=2km, 200=10km)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--snap", nargs="+", type=float, default=None)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--geometry-only", action="store_true")
    p.add_argument("--order", type=int, default=4, choices=[2, 4])
    p.add_argument("--t-total", type=float, default=T_TOTAL)
    p.add_argument("--f0", type=float, default=F0)
    # NEW: anomaly mode
    p.add_argument("--anomaly", action="store_true",
                    help="Use heterogeneous_with_anomaly: bg medium + box of "
                         "different ε.  RCV.B sits inside the anomaly.")
    p.add_argument("--eps-bg", type=float, default=0.0,
                    help="Background ε (default 0 = homogeneous outside box)")
    p.add_argument("--eps-anom", type=float, default=0.10,
                    help="Anomaly ε  (default 0.10)")
    args = p.parse_args()

    delta = 0.0 if args.model == "homogeneous" else args.perturbation
    run_single(
        model_type=args.model, perturbation=delta,
        corr_cells=args.corr, seed=args.seed,
        output_dir=args.output_dir, snap_times=args.snap,
        geometry_only=args.geometry_only, order=args.order,
        psdf=args.psdf, t_total=args.t_total, f0=args.f0,
        eps_bg=args.eps_bg, eps_anom=args.eps_anom, anomaly=args.anomaly,
    )


if __name__ == "__main__":
    main()