"""
smoke_test.py — fast sanity check for the PML fixes (no 21-min full run).
Small 200x200 grid, ~2.5 s sim. Verifies:
  [1] perturbation tapered to zero inside the PML band (flat background there)
  [2] no RuntimeWarning, no NaN/Inf during stepping
  [3] energy decays after the source (PML removing energy, not blowing up)
  [4] homogeneous control: field near the boundary is tiny after the wave exits
"""
import warnings, sys
import numpy as np
from velocity_model import heterogeneous_random, homogeneous, pml_taper_2d
from elastic_solver import ElasticSolver2D

NX = NZ = 200
DX = DZ = 50.0
PML = 20
VP, RHO = 6000.0, 2700.0
fails = []

# ── [1] taper zeroes the PML band ────────────────────────────────────
vp, vs, rho = heterogeneous_random(NX, NZ, dx=DX, dz=DZ, vp_mean=VP,
                                   perturbation=0.05, corr_x=6, corr_z=6,
                                   psdf='exp', pml_cells=PML)
mid = NZ // 2
band = np.r_[vp[mid, :PML], vp[mid, -PML:]]          # PML columns at mid-height
flat = np.allclose(band, VP, atol=1.0)
print(f"[1] PML-band Vp: min={band.min():.1f} max={band.max():.1f} "
      f"(expect {VP:.0f})  -> {'PASS' if flat else 'FAIL'}")
interior_spread = vp[mid, NX//2-20:NX//2+20].std()
print(f"    interior Vp std={interior_spread:.1f} m/s (should be > 0, medium still random)")
if not flat: fails.append("taper")
if interior_spread < 1.0: fails.append("interior-flattened")

# ── [2]+[3] run heterogeneous, watch for warnings / NaN / energy ─────
with warnings.catch_warnings():
    warnings.simplefilter("error", RuntimeWarning)   # any divide warning -> exception
    try:
        solver = ElasticSolver2D(NX, NZ, DX, DZ, vp, vs, rho, pml=PML, order=4)
        solver.add_source(NX*DX/2, NZ*DZ/2)
        snaps, t, Et, Ep, Es = solver.run(t_total=2.5, f0=4.0,
                                          snap_times_s=[2.4], output_dir="smoke_out")
        warn_ok = True
    except RuntimeWarning as e:
        warn_ok = False
        print(f"[2] RuntimeWarning raised: {e}")
print(f"[2] no RuntimeWarning during run -> {'PASS' if warn_ok else 'FAIL'}")
if not warn_ok: fails.append("runtimewarning")

finite = np.all(np.isfinite(Et))
peak_i = int(np.argmax(Et))
decayed = Et[-1] < 0.2 * Et[peak_i]
print(f"[3] energy finite={finite}  peak@step{peak_i}  "
      f"E_end/E_peak={Et[-1]/Et[peak_i]:.3e} -> "
      f"{'PASS' if (finite and decayed) else 'FAIL'}")
if not (finite and decayed): fails.append("energy")

# ── [4] homogeneous control: PML absorbs essentially all energy ──────
vp2, vs2, rho2 = homogeneous(NX, NZ, vp=VP, rho=RHO)
solver2 = ElasticSolver2D(NX, NZ, DX, DZ, vp2, vs2, rho2, pml=PML, order=4)
solver2.add_source(NX*DX/2, NZ*DZ/2)
_, t2, Et2, _, _ = solver2.run(t_total=3.0, f0=4.0,
                               snap_times_s=[2.9], output_dir="smoke_out_homo")
resid = Et2[-1] / Et2[int(np.argmax(Et2))]
clean = resid < 1e-4                    # <0.01% energy left = no boundary leak
print(f"[4] homogeneous residual E_end/E_peak = {resid:.3e} "
      f"(<1e-4 = PML clean) -> {'PASS' if clean else 'FAIL'}")
if not clean: fails.append("boundary")

print("\n" + "="*50)
print("SMOKE TEST: " + ("ALL PASS ✓" if not fails else f"FAIL -> {fails}"))
print("="*50)
sys.exit(1 if fails else 0)