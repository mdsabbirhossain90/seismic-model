"""
model.py — 2D Elastic Simulation, NO PML, padded computational domain
======================================================================
Model area (what you interpret)  : 40 x 40 km,  x,z in [0, 40] km
Computational grid (what is run) : 50 x 50 km,  x,z in [-5, 45] km
The 5 km pad on every side replaces the old 2 km PML. There is no absorbing
layer: the outer grid edges are rigid walls.  Source, receivers and anomaly
keep their original 40 km-frame coordinates.

Run examples:
  python model.py --model homogeneous --output-dir results/test_homo
  python model.py --perturbation 0.05 --corr 6 --output-dir results/test_het
  python model.py --anomaly --eps-bg 0.05 --eps-anom 0.09 --corr 6 --output-dir results/test_anom
"""

import argparse, time, os
import numpy as np
import matplotlib.pyplot as plt

from geometry       import load_receivers, geometry_summary, plot_geometry
from velocity_model import (heterogeneous_random, heterogeneous_with_anomaly,
                             homogeneous, plot_velocity_model)
from elastic_solver import ElasticSolver2D


# Model area (40 x 40 km) — unchanged
MAIN_X   = (0.0, 40000.0)
MAIN_Z   = (0.0, 40000.0)
MAIN_BOX = (MAIN_X[0], MAIN_X[1], MAIN_Z[0], MAIN_Z[1])

# Computational padding around the model area (each side)
PAD      = 5000.0            # 5 km  -> 50 x 50 km grid
PML      = 0                 # NO PML

DX, DZ   = 50.0, 50.0
VP_MEAN  = 6000.0
VS_MEAN  = 3464.0
RHO_MEAN = 2700.0

# Source / receivers (40 km-frame coordinates, unchanged)
SOURCE   = (12000.0, 20000.0)
RECEIVERS = [
    ("A", 15000.0, 20000.0),     # BEFORE  the anomaly (3 km from source)
    ("B", 22000.0, 20000.0),     # WITHIN  the anomaly (10 km from source)
    ("C", 30000.0, 20000.0),     # BEHIND  the anomaly (18 km from source)
]

# Anomaly box — 8 km x 8 km centred on (22, 20) km
ANOM_X = (18000.0, 26000.0)
ANOM_Z = (16000.0, 24000.0)

# Sim params (unchanged)
F0       = 4.78
T_TOTAL  = 10.0
SNAPS    = [2.0, 4.0, 6.0, 8.0]
SEED     = 42
VS_FLOOR = 1800.0


def grid_setup(pad):
    """Grid size and origin for a model area padded by `pad` metres."""
    nx = int(round((MAIN_X[1] - MAIN_X[0] + 2*pad) / DX))
    nz = int(round((MAIN_Z[1] - MAIN_Z[0] + 2*pad) / DZ))
    origin = (MAIN_X[0] - pad, MAIN_Z[0] - pad)
    return nx, nz, origin


def sep(c="=", n=72): print(c*n)


def run_single(model_type, perturbation, corr_cells, seed, output_dir,
               snap_times=None, geometry_only=False, order=4,
               psdf='exp', t_total=T_TOTAL, f0=F0,
               eps_bg=0.0, eps_anom=0.10, anomaly=False,
               pml=PML, pad=PAD, view="full"):
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()
    NX, NZ, ORIGIN = grid_setup(pad)

    sep()
    if anomaly:
        print(f"  ANOMALY MODE  |  PSDF={psdf}  |  eps_bg={eps_bg}  "
              f"eps_anom={eps_anom}  a={corr_cells*DX:.0f}m")
    else:
        print(f"  {model_type}  |  PSDF={psdf}  |  ε={perturbation}  "
              f"a={corr_cells*DX:.0f}m")
    print(f"  T={t_total}s  f0={f0}Hz  seed={seed}")
    print(f"  Model area  : {MAIN_X[0]/1000:.0f}-{MAIN_X[1]/1000:.0f} km x "
          f"{MAIN_Z[0]/1000:.0f}-{MAIN_Z[1]/1000:.0f} km")
    print(f"  Grid        : {NX} x {NZ} cells, {NX*DX/1000:.0f} x {NZ*DZ/1000:.0f} km "
          f"(pad {pad/1000:.1f} km each side, PML = {pml} cells)")
    sep()

    # Geometry
    receivers = load_receivers(RECEIVERS)
    distances = geometry_summary(SOURCE, receivers)
    plot_geometry(SOURCE, receivers, distances, output_dir=output_dir)

    # Velocity model (built on the full padded grid)
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
            pml_cells=pml, origin=ORIGIN,
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
            pml_cells=pml,
        )
        title = (f"Heterogeneous  PSDF={psdf}  ε={perturbation}  "
                 f"a={corr_cells*DX/1000:.1f} km")

    print(f"  Vp  {vp.min():.0f}–{vp.max():.0f}  mean={vp.mean():.0f} m/s")
    print(f"  Vs  {vs.min():.0f}–{vs.max():.0f}  mean={vs.mean():.0f} m/s")
    plot_velocity_model(vp, vs, DX, DZ, SOURCE, RECEIVERS,
                        output_dir=output_dir, title=title,
                        anomaly_bbox=anomaly_bbox, origin=ORIGIN,
                        main_box=MAIN_BOX, view=view)
    np.savez_compressed(os.path.join(output_dir, "velocity_model.npz"),
                        vp=vp.astype(np.float32), vs=vs.astype(np.float32),
                        origin=np.array(ORIGIN), dx=DX, dz=DZ,
                        main_box=np.array(MAIN_BOX))

    if geometry_only:
        print(f"\nDone (geometry only).  {time.time()-t0:.1f}s")
        return None

    # Solver
    solver = ElasticSolver2D(
        nx=NX, nz=NZ, dx=DX, dz=DZ,
        vp=vp, vs=vs, rho=rho,
        pml=pml, order=order, f0_pml=f0,
        origin=ORIGIN, main_box=MAIN_BOX, plot_view=view,
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
    E_dom = solver.E_domain

    plot_energy(times, E_tot, E_P, E_S, E_dom, title, output_dir)
    np.savez(os.path.join(output_dir, "energy.npz"),
             t=np.asarray(times), E_tot=np.asarray(E_tot),
             E_P=np.asarray(E_P), E_S=np.asarray(E_S),
             E_domain=np.asarray(E_dom))

    elapsed = time.time() - t0
    sep()
    print(f"  Outputs → '{output_dir}/'   ({elapsed/60:.1f} min)")
    sep()
    return times, E_tot, E_P, E_S


def plot_energy(times, E_tot, E_P, E_S, E_dom, label, output_dir):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=False)
    fig.patch.set_facecolor("white")
    fig.suptitle(f"Energy in 40×40 km model area — {label}",
                 fontsize=12, fontweight="bold")
    panels = [(E_tot, "Total", "-", "black"),
              (E_P,   "P-wave  ½ρVp²|div v|²", "--", "#0066aa"),
              (E_S,   "S-wave  ½ρVs²|curl v|²", "-.", "#aa0033")]
    times = np.asarray(times)
    for row, (E, name, ls, color) in enumerate(panels):
        E = np.asarray(E)
        axes[row, 0].semilogy(times, E + 1e-30, color=color, lw=1.4, ls=ls,
                              label="40×40 km area")
        if row == 0:
            axes[row, 0].semilogy(times, np.asarray(E_dom) + 1e-30, color="0.6",
                                  lw=1.0, label="whole 50×50 km grid")
            axes[row, 0].legend(fontsize=8)
        axes[row, 0].set_ylabel("E (log)")
        axes[row, 0].set_title(f"{name}  — full", fontsize=10, loc="left")
        axes[row, 0].grid(True, which="both", color="lightgray", lw=0.4)
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
                    help="Correlation length in cells (dx=50m → 6=0.3km)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--snap", nargs="+", type=float, default=None)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--geometry-only", action="store_true")
    p.add_argument("--order", type=int, default=4, choices=[2, 4])
    p.add_argument("--t-total", type=float, default=T_TOTAL)
    p.add_argument("--f0", type=float, default=F0)
    p.add_argument("--anomaly", action="store_true")
    p.add_argument("--eps-bg", type=float, default=0.0)
    p.add_argument("--eps-anom", type=float, default=0.10)
    # NEW
    p.add_argument("--pml", type=int, default=PML,
                    help="Absorbing layer thickness in cells (default 0 = no PML)")
    p.add_argument("--pad-km", type=float, default=PAD/1000,
                    help="Padding around the 40x40 km area, each side (default 5)")
    p.add_argument("--view", default="full", choices=["full", "main"],
                    help="Plot whole grid (full) or only the 40x40 km area (main)")
    args = p.parse_args()

    delta = 0.0 if args.model == "homogeneous" else args.perturbation
    run_single(
        model_type=args.model, perturbation=delta,
        corr_cells=args.corr, seed=args.seed,
        output_dir=args.output_dir, snap_times=args.snap,
        geometry_only=args.geometry_only, order=args.order,
        psdf=args.psdf, t_total=args.t_total, f0=args.f0,
        eps_bg=args.eps_bg, eps_anom=args.eps_anom, anomaly=args.anomaly,
        pml=args.pml, pad=args.pad_km*1000, view=args.view,
    )


if __name__ == "__main__":
    main()
