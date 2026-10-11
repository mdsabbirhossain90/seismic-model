"""
elastic_solver.py  (v6 — no PML, padded 50 x 50 km computational domain)
========================================================================
2D Elastic Wave Solver — Virieux (1986) velocity-stress staggered-grid FD.

CHANGES vs v5
-------------
1. PML removed by default (pml=0).  The outer edges are rigid walls
   (vx = vz = 0 on the outermost 2 cells), so boundary reflections are total.
   The C-PML code is still present but only switches on if pml > 0.
2. Physical-coordinate origin: the grid can start at negative coordinates
   (origin = (-5000, -5000) m) so the 40 x 40 km model area keeps its
   original coordinates (source at x=12 km etc.) inside a 50 x 50 km grid.
3. Plots show the FULL 50 x 50 km domain with the 40 x 40 km model area
   outlined (plot_view="full"), or only the 40 x 40 km area ("main").
4. Energy: strain energy uses the exact plane-strain compliance.
   Energy is reported inside the 40 x 40 km area (ROI) and for the whole
   domain (the latter should stay ~constant with rigid walls: a check).
5. Time axis is exactly it*dt (was linspace(0, T, nt)).
6. Seismogram markers include the Ricker delay t0 = 1.5/f0, plus a marker
   for the first boundary-reflected P arrival (image-source estimate).
7. Snapshots saved to snapshots.npz for common-scale comparison figures.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class ElasticSolver2D:

    _C1 = 9.0 / 8.0
    _C2 = -1.0 / 24.0
    _NB = 2                       # rigid-wall thickness (cells) when pml == 0

    def __init__(self, nx, nz, dx, dz, vp, vs, rho,
                 dt=None, pml=0, order=4, f0_pml=1.0,
                 origin=(0.0, 0.0), main_box=None, plot_view="full"):
        self.nx, self.nz = nx, nz
        self.dx, self.dz = dx, dz
        self.pml   = int(pml)
        self.order = order
        self.x0, self.z0 = float(origin[0]), float(origin[1])
        # main_box = (xmin, xmax, zmin, zmax) in metres (physical coords)
        self.main_box  = main_box
        self.plot_view = plot_view

        def _arr(v):
            return np.full((nz, nx), float(v)) if np.isscalar(v) else np.asarray(v, float)

        vp  = np.clip(_arr(vp),  500.0, 12000.0)
        vs  = np.clip(_arr(vs),  200.0,  7000.0)
        rho = _arr(rho)

        self.lam    = rho * (vp**2 - 2.0*vs**2)
        self.mu     = rho * vs**2
        self.rho    = rho
        self.lam2mu = self.lam + 2.0*self.mu
        self.vp_arr = vp
        self.vs_arr = vs
        self._inv_rho = 1.0 / rho

        vmax       = vp.max()
        cfl_factor = 6.0/7.0 if order == 4 else 1.0
        dt_max     = cfl_factor * min(dx, dz) / (vmax * np.sqrt(2.0))
        safety     = 0.6 if order == 4 else 0.75
        self.dt    = float(dt) if dt else safety * dt_max

        x1 = self.x0 + nx*dx; z1 = self.z0 + nz*dz
        print(f"  Grid      : {nx} x {nz}  ({nx*dx/1000:.0f} km x {nz*dz/1000:.0f} km)"
              f"   x: {self.x0/1000:.1f}..{x1/1000:.1f} km   z: {self.z0/1000:.1f}..{z1/1000:.1f} km")
        print(f"  FD order  : {order} (spatial),  2 (temporal)")
        print(f"  Vp range  : {vp.min():.0f} - {vp.max():.0f} m/s")
        print(f"  Vs range  : {vs.min():.0f} - {vs.max():.0f} m/s")
        print(f"  CFL limit : {dt_max*1000:.3f} ms  =>  dt = {self.dt*1000:.3f} ms  (x{safety})")
        if self.pml > 0:
            print(f"  C-PML zone: {self.pml} cells = {self.pml*dx:.0f} m from each edge")
        else:
            print(f"  Boundary  : NO PML — rigid walls (fully reflecting) at the grid edges")

        shape = (nz, nx)
        self.vx  = np.zeros(shape); self.vz  = np.zeros(shape)
        self.sxx = np.zeros(shape); self.szz = np.zeros(shape)
        self.sxz = np.zeros(shape)

        # exact plane-strain strain-energy coefficients
        lam_mu = np.maximum(self.lam + self.mu, 1e-30)
        mu_s   = np.maximum(self.mu, 1e-30)
        self._ce1 = self.lam2mu / (8.0 * mu_s * lam_mu)      # (sxx^2 + szz^2)
        self._ce2 = -self.lam  / (4.0 * mu_s * lam_mu)       # sxx*szz
        self._ce3 = 1.0 / (2.0 * mu_s)                       # sxz^2

        # 40 x 40 km region-of-interest index window
        if main_box is not None:
            ix0 = max(0, int(round((main_box[0] - self.x0) / dx)))
            ix1 = min(nx, int(round((main_box[1] - self.x0) / dx)) + 1)
            iz0 = max(0, int(round((main_box[2] - self.z0) / dz)))
            iz1 = min(nz, int(round((main_box[3] - self.z0) / dz)) + 1)
            self._roi = (slice(iz0, iz1), slice(ix0, ix1))
        else:
            self._roi = (slice(None), slice(None))

        if self.pml > 0:
            self._init_cpml(vmax, f0_pml)

        self._receivers = []
        self._traces    = {}
        self._E_total = []; self._E_P = []; self._E_S = []; self._E_domain = []
        self._peak_v = 0.0
        self._src_iz = self._src_ix = None
        self._f0 = None

    # ── coordinates ──────────────────────────────────────────────────
    def _to_grid(self, x_m, z_m):
        return (int(round((x_m - self.x0) / self.dx)),
                int(round((z_m - self.z0) / self.dz)))

    def _to_phys(self, ix, iz):
        return self.x0 + ix*self.dx, self.z0 + iz*self.dz

    # ── C-PML (only used if pml > 0) ─────────────────────────────────
    def _init_cpml(self, vmax, f0_pml):
        pml = self.pml
        L_pml_x = pml * self.dx
        L_pml_z = pml * self.dz
        R_target, N_pml, kappa_max = 1e-8, 4, 10.0
        d_max_x = -(N_pml + 1) * vmax * np.log(R_target) / (2.0 * L_pml_x)
        d_max_z = -(N_pml + 1) * vmax * np.log(R_target) / (2.0 * L_pml_z)
        alpha_max = np.pi * f0_pml

        d_x = np.zeros(self.nx); alpha_x = np.zeros(self.nx); kappa_x = np.ones(self.nx)
        d_z = np.zeros(self.nz); alpha_z = np.zeros(self.nz); kappa_z = np.ones(self.nz)
        for i in range(pml):
            xi   = (pml - i) / pml
            prof = xi**N_pml
            kap  = 1.0 + (kappa_max - 1.0) * prof
            alf  = alpha_max * (1.0 - xi)
            for idx in (i, self.nx - 1 - i):
                d_x[idx] = d_max_x * prof; alpha_x[idx] = alf; kappa_x[idx] = kap
            for idx in (i, self.nz - 1 - i):
                d_z[idx] = d_max_z * prof; alpha_z[idx] = alf; kappa_z[idx] = kap

        self.b_x = np.exp(-(d_x / kappa_x + alpha_x) * self.dt)
        self.b_z = np.exp(-(d_z / kappa_z + alpha_z) * self.dt)
        denom_x  = kappa_x * (d_x + kappa_x * alpha_x)
        denom_z  = kappa_z * (d_z + kappa_z * alpha_z)
        self.a_x = np.zeros(self.nx); mx = d_x > 1e-12
        self.a_z = np.zeros(self.nz); mz = d_z > 1e-12
        self.a_x[mx] = d_x[mx] / denom_x[mx] * (self.b_x[mx] - 1.0)
        self.a_z[mz] = d_z[mz] / denom_z[mz] * (self.b_z[mz] - 1.0)
        self.kappa_x = kappa_x
        self.kappa_z = kappa_z

        s = (self.nz, self.nx)
        self.psi_vx_x  = np.zeros(s); self.psi_vz_z  = np.zeros(s)
        self.psi_vx_z  = np.zeros(s); self.psi_vz_x  = np.zeros(s)
        self.psi_sxx_x = np.zeros(s); self.psi_sxz_z = np.zeros(s)
        self.psi_sxz_x = np.zeros(s); self.psi_szz_z = np.zeros(s)

    # ── source / receiver registration ───────────────────────────────
    def add_source(self, x_m, z_m):
        ix, iz = self._to_grid(x_m, z_m)
        margin = (self.pml if self.pml > 0 else self._NB) + 5
        if iz < margin or iz > self.nz-margin or ix < margin or ix > self.nx-margin:
            raise ValueError(f"Source [{ix},{iz}] too close to the grid edge.")
        self._src_ix = ix; self._src_iz = iz
        print(f"  Source    : ({x_m:.0f}, {z_m:.0f}) m  -> grid [{ix}, {iz}]")

    def add_receiver(self, name, x_m, z_m):
        ix, iz = self._to_grid(x_m, z_m)
        self._receivers.append((name, iz, ix))
        self._traces[name] = {"vx": [], "vz": [], "P": [], "S": []}
        print(f"  Receiver  : {name:<10} ({x_m:.0f}, {z_m:.0f}) m  -> grid [{ix}, {iz}]")

    # ── wavelets ─────────────────────────────────────────────────────
    @staticmethod
    def ricker(t, f0, t0=None):
        t0 = t0 or (1.5 / f0); tau = np.pi * f0 * (t - t0)
        return (1.0 - 2.0*tau**2) * np.exp(-tau**2)

    @staticmethod
    def gaussian(t, f0, t0=None):
        t0 = t0 or (1.5 / f0); tau = np.pi * f0 * (t - t0)
        return -2.0 * tau * np.exp(-tau**2)

    # ── stencils ─────────────────────────────────────────────────────
    def _dx_fwd_2(self, f):
        d = np.zeros_like(f); d[:, :-1] = (f[:, 1:] - f[:, :-1]) / self.dx; return d
    def _dx_bwd_2(self, f):
        d = np.zeros_like(f); d[:, 1:]  = (f[:, 1:] - f[:, :-1]) / self.dx; return d
    def _dz_fwd_2(self, f):
        d = np.zeros_like(f); d[:-1, :] = (f[1:, :] - f[:-1, :]) / self.dz; return d
    def _dz_bwd_2(self, f):
        d = np.zeros_like(f); d[1:, :]  = (f[1:, :] - f[:-1, :]) / self.dz; return d

    def _dx_fwd_4(self, f):
        d = np.zeros_like(f)
        d[:, 1:-2] = (self._C1 * (f[:, 2:-1] - f[:, 1:-2])
                    + self._C2 * (f[:, 3:]   - f[:, :-3])) / self.dx
        d[:, 0]    = (f[:, 1]  - f[:, 0])  / self.dx
        d[:, -2]   = (f[:, -1] - f[:, -2]) / self.dx
        return d
    def _dx_bwd_4(self, f):
        d = np.zeros_like(f)
        d[:, 2:-1] = (self._C1 * (f[:, 2:-1] - f[:, 1:-2])
                    + self._C2 * (f[:, 3:]   - f[:, :-3])) / self.dx
        d[:, 1]    = (f[:, 1]  - f[:, 0])  / self.dx
        d[:, -1]   = (f[:, -1] - f[:, -2]) / self.dx
        return d
    def _dz_fwd_4(self, f):
        d = np.zeros_like(f)
        d[1:-2, :] = (self._C1 * (f[2:-1, :] - f[1:-2, :])
                    + self._C2 * (f[3:, :]   - f[:-3, :])) / self.dz
        d[0,  :]   = (f[1, :]  - f[0, :])  / self.dz
        d[-2, :]   = (f[-1, :] - f[-2, :]) / self.dz
        return d
    def _dz_bwd_4(self, f):
        d = np.zeros_like(f)
        d[2:-1, :] = (self._C1 * (f[2:-1, :] - f[1:-2, :])
                    + self._C2 * (f[3:, :]   - f[:-3, :])) / self.dz
        d[1,  :]   = (f[1, :]  - f[0, :])  / self.dz
        d[-1, :]   = (f[-1, :] - f[-2, :]) / self.dz
        return d

    def _dx_fwd(self, f): return self._dx_fwd_4(f) if self.order == 4 else self._dx_fwd_2(f)
    def _dx_bwd(self, f): return self._dx_bwd_4(f) if self.order == 4 else self._dx_bwd_2(f)
    def _dz_fwd(self, f): return self._dz_fwd_4(f) if self.order == 4 else self._dz_fwd_2(f)
    def _dz_bwd(self, f): return self._dz_bwd_4(f) if self.order == 4 else self._dz_bwd_2(f)

    # ── diagnostics ──────────────────────────────────────────────────
    def _ps_at_receiver(self, iz, ix):
        ix0 = max(1, min(ix, self.nx-2))
        iz0 = max(1, min(iz, self.nz-2))
        dvx_dx = (self.vx[iz0, ix0+1] - self.vx[iz0, ix0-1]) / (2*self.dx)
        dvz_dz = (self.vz[iz0+1, ix0] - self.vz[iz0-1, ix0]) / (2*self.dz)
        dvz_dx = (self.vz[iz0, ix0+1] - self.vz[iz0, ix0-1]) / (2*self.dx)
        dvx_dz = (self.vx[iz0+1, ix0] - self.vx[iz0-1, ix0]) / (2*self.dz)
        return dvx_dx + dvz_dz, dvz_dx - dvx_dz

    def _compute_energies(self):
        """Returns (E_total in ROI, E_P in ROI, E_S in ROI, E_total whole domain)."""
        dA = self.dx * self.dz
        w = (0.5 * self.rho * (self.vx**2 + self.vz**2)
             + self._ce1 * (self.sxx**2 + self.szz**2)
             + self._ce2 * self.sxx * self.szz
             + self._ce3 * self.sxz**2)
        E_dom = float(w.sum()) * dA
        r = self._roi
        E_roi = float(w[r].sum()) * dA
        div_v  = self._dx_bwd(self.vx) + self._dz_bwd(self.vz)
        curl_v = self._dx_bwd(self.vz) - self._dz_bwd(self.vx)
        E_P = 0.5 * float(np.sum(self.lam2mu[r] * div_v[r]**2)) * dA
        E_S = 0.5 * float(np.sum(self.mu[r]     * curl_v[r]**2)) * dA
        return E_roi, E_P, E_S, E_dom

    def _rigid_walls(self):
        nb = self._NB
        for f in (self.vx, self.vz):
            f[:, :nb] = 0.0; f[:, -nb:] = 0.0
            f[:nb, :] = 0.0; f[-nb:, :] = 0.0

    # ── main loop ────────────────────────────────────────────────────
    def run(self, t_total, f0=5.0, snap_times_s=None, output_dir="outputs",
            source_type="ricker"):
        os.makedirs(output_dir, exist_ok=True)
        dt = self.dt
        nt = int(t_total / dt)
        self._f0 = f0
        use_pml = self.pml > 0

        ppw_p = self.vp_arr.min() / (f0 * self.dx)
        ppw_s = self.vs_arr.min() / (f0 * self.dx)
        need  = 6 if self.order == 4 else 10
        ok    = "OK" if ppw_s >= need else f"WARNING — undersampled (need ≥{need})"
        print(f"  PPW (P)   : {ppw_p:.1f}")
        print(f"  PPW (S)   : {ppw_s:.1f}   {ok}")

        if snap_times_s is None:
            snap_times_s = [t_total*0.4, t_total*0.75, t_total]
        snap_steps = {min(int(round(ts/dt)), nt - 1): ts for ts in snap_times_s}
        snapshots  = []

        print(f"\n  Running {nt} steps  (T={t_total} s, dt={dt*1000:.3f} ms, "
              f"f0={f0} Hz, source={source_type}, "
              f"{'C-PML on' if use_pml else 'no PML, rigid walls'})")

        src_fn = self.gaussian if source_type == "gaussian" else self.ricker

        if use_pml:
            bx, ax = self.b_x[None, :], self.a_x[None, :]
            bz, az = self.b_z[:, None], self.a_z[:, None]
            kx, kz = self.kappa_x[None, :], self.kappa_z[:, None]

        import time as _time
        t_wall = _time.time()

        for it in range(nt):
            t = it * dt

            # ── Stress update ────────────────────────────────────────
            dvx_dx = self._dx_bwd(self.vx)
            dvz_dz = self._dz_bwd(self.vz)
            dvx_dz = self._dz_fwd(self.vx)
            dvz_dx = self._dx_fwd(self.vz)

            if use_pml:
                self.psi_vx_x = bx * self.psi_vx_x + ax * dvx_dx
                self.psi_vz_z = bz * self.psi_vz_z + az * dvz_dz
                self.psi_vx_z = bz * self.psi_vx_z + az * dvx_dz
                self.psi_vz_x = bx * self.psi_vz_x + ax * dvz_dx
                dvx_dx = dvx_dx / kx + self.psi_vx_x
                dvz_dz = dvz_dz / kz + self.psi_vz_z
                dvx_dz = dvx_dz / kz + self.psi_vx_z
                dvz_dx = dvz_dx / kx + self.psi_vz_x

            self.sxx += dt * (self.lam2mu * dvx_dx + self.lam    * dvz_dz)
            self.szz += dt * (self.lam    * dvx_dx + self.lam2mu * dvz_dz)
            self.sxz += dt * self.mu * (dvx_dz + dvz_dx)

            # ── explosive source ─────────────────────────────────────
            if self._src_iz is not None:
                amp = src_fn(t, f0) * 1e6
                self.sxx[self._src_iz, self._src_ix] += dt * amp
                self.szz[self._src_iz, self._src_ix] += dt * amp

            # ── Velocity update ──────────────────────────────────────
            dsxx_dx = self._dx_fwd(self.sxx)
            dsxz_dz = self._dz_bwd(self.sxz)
            dsxz_dx = self._dx_bwd(self.sxz)
            dszz_dz = self._dz_fwd(self.szz)

            if use_pml:
                self.psi_sxx_x = bx * self.psi_sxx_x + ax * dsxx_dx
                self.psi_sxz_z = bz * self.psi_sxz_z + az * dsxz_dz
                self.psi_sxz_x = bx * self.psi_sxz_x + ax * dsxz_dx
                self.psi_szz_z = bz * self.psi_szz_z + az * dszz_dz
                dsxx_dx = dsxx_dx / kx + self.psi_sxx_x
                dsxz_dz = dsxz_dz / kz + self.psi_sxz_z
                dsxz_dx = dsxz_dx / kx + self.psi_sxz_x
                dszz_dz = dszz_dz / kz + self.psi_szz_z

            self.vx += dt * self._inv_rho * (dsxx_dx + dsxz_dz)
            self.vz += dt * self._inv_rho * (dsxz_dx + dszz_dz)

            if not use_pml:
                self._rigid_walls()

            self._peak_v = max(self._peak_v,
                               float(np.abs(self.vx).max()),
                               float(np.abs(self.vz).max()))

            # ── diagnostics ──────────────────────────────────────────
            Et, Ep, Es, Ed = self._compute_energies()
            self._E_total.append(Et); self._E_P.append(Ep)
            self._E_S.append(Es);     self._E_domain.append(Ed)

            for name, iz, ix in self._receivers:
                self._traces[name]["vx"].append(float(self.vx[iz, ix]))
                self._traces[name]["vz"].append(float(self.vz[iz, ix]))
                P, S = self._ps_at_receiver(iz, ix)
                self._traces[name]["P"].append(P)
                self._traces[name]["S"].append(S)

            if it in snap_steps:
                P_field = self._dx_bwd(self.vx) + self._dz_bwd(self.vz)
                S_field = self._dx_bwd(self.vz) - self._dz_bwd(self.vx)
                snapshots.append((snap_steps[it],
                                  self.vx.copy(), self.vz.copy(),
                                  P_field.copy(), S_field.copy()))
                print(f"    Snap t={snap_steps[it]:.2f}s  "
                      f"|vx|={np.abs(self.vx).max():.3e}  "
                      f"|P|={np.abs(P_field).max():.3e}  "
                      f"|S|={np.abs(S_field).max():.3e}")

            if it % max(1, nt//20) == 0:
                el = _time.time() - t_wall
                eta = el / (it + 1) * (nt - it - 1)
                print(f"    Step {it:6d}/{nt}  t={t:6.2f} s   "
                      f"elapsed {el/60:5.1f} min   ETA {eta/60:5.1f} min", flush=True)

        print("  Simulation complete.")
        self._save_snapshots(snapshots, output_dir)
        self._plot_wavefield(snapshots, output_dir)
        self._plot_ps_wavefield(snapshots, output_dir)
        self._plot_seismograms(nt, t_total, output_dir)

        times = np.arange(nt) * dt
        self.E_domain = np.array(self._E_domain)
        return (snapshots, times,
                np.array(self._E_total),
                np.array(self._E_P),
                np.array(self._E_S))

    # ── saving ───────────────────────────────────────────────────────
    def _save_snapshots(self, snapshots, output_dir):
        if not snapshots:
            return
        names = [r[0] for r in self._receivers]
        rpos  = [self._to_phys(ix, iz) for _, iz, ix in self._receivers]
        sx, sz = self._to_phys(self._src_ix, self._src_iz)
        out = os.path.join(output_dir, "snapshots.npz")
        np.savez_compressed(
            out,
            times=np.array([s[0] for s in snapshots]),
            vx=np.stack([s[1] for s in snapshots]).astype(np.float32),
            vz=np.stack([s[2] for s in snapshots]).astype(np.float32),
            P=np.stack([s[3] for s in snapshots]).astype(np.float32),
            S=np.stack([s[4] for s in snapshots]).astype(np.float32),
            origin=np.array([self.x0, self.z0]),
            dx=self.dx, dz=self.dz,
            main_box=np.array(self.main_box if self.main_box is not None
                              else [self.x0, self.x0+self.nx*self.dx,
                                    self.z0, self.z0+self.nz*self.dz]),
            source=np.array([sx, sz]),
            rec_names=np.array(names, dtype=str),
            rec_pos=np.array(rpos),
        )
        print(f"  - {out}")

    # ── plotting helpers ─────────────────────────────────────────────
    def _extent_km(self):
        return [self.x0/1000, (self.x0 + self.nx*self.dx)/1000,
                (self.z0 + self.nz*self.dz)/1000, self.z0/1000]

    def _decorate(self, ax, marker_color="r", edge="black", text_color="darkred"):
        if self._src_ix is not None:
            sx, sz = self._to_phys(self._src_ix, self._src_iz)
            ax.plot(sx/1000, sz/1000, marker_color + "*", ms=13, zorder=7,
                    markeredgecolor=edge, markeredgewidth=0.5, label="Source")
        for name, iz, ix in self._receivers:
            rx, rz = self._to_phys(ix, iz)
            ax.plot(rx/1000, rz/1000, marker_color + "v", ms=9, zorder=7,
                    markeredgecolor=edge, markeredgewidth=0.5)
            ax.annotate(name, (rx/1000, rz/1000), textcoords="offset points",
                        xytext=(5, -13), fontsize=8, color=text_color)
        if self.main_box is not None:
            xmin, xmax, zmin, zmax = [v/1000 for v in self.main_box]
            ax.add_patch(patches.Rectangle((xmin, zmin), xmax-xmin, zmax-zmin,
                         lw=1.2, ls="--", ec="black", fc="none", zorder=6,
                         label=f"{xmax-xmin:.0f}×{zmax-zmin:.0f} km model area"))
            if self.plot_view == "main":
                ax.set_xlim(xmin, xmax); ax.set_ylim(zmax, zmin)

    _FLOOR_RATIO = 3e-4

    def _annotate_amp(self, ax, data, ref, ref_label, allow_floor):
        pk = float(np.abs(data).max())
        ref = ref if ref > 0 else pk
        ratio = pk / ref
        ax.text(0.015, 0.985, f"peak = {pk:.2e}\n{ratio:.1e} x {ref_label}",
                transform=ax.transAxes, fontsize=8, ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                          ec="gray", alpha=0.9))
        if allow_floor and ratio < self._FLOOR_RATIO:
            ax.text(0.5, 0.5, "RESIDUAL FLOOR\n(< control floor)",
                    transform=ax.transAxes, fontsize=13, ha="center",
                    va="center", color="0.45", fontweight="bold", alpha=0.6,
                    rotation=20)

    def _plot_wavefield(self, snapshots, output_dir):
        for snap in snapshots:
            t, vx_s, vz_s = snap[0], snap[1], snap[2]
            fig, axes = plt.subplots(1, 2, figsize=(15, 7))
            fig.patch.set_facecolor("white")
            fig.suptitle(f"Wavefield at t = {t:.2f} s", fontsize=14, fontweight="bold")
            for ax, data, label in zip(axes, [vx_s, vz_s], ["Vx", "Vz"]):
                vmax = np.percentile(np.abs(data), 99) + 1e-30
                im = ax.imshow(data, cmap="seismic", vmin=-vmax, vmax=vmax,
                               extent=self._extent_km(), aspect="equal",
                               interpolation="bilinear")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)\
                   .set_label("Amplitude (m/s)", fontsize=9)
                ax.set_title(label, fontsize=13, fontweight="bold")
                ax.set_xlabel("X (km)"); ax.set_ylabel("Z (km)")
                self._decorate(ax)
                ax.legend(fontsize=8, loc="upper right")
                self._annotate_amp(ax, data, self._peak_v, "incident", self.pml > 0)
            plt.tight_layout()
            fname = os.path.join(output_dir, f"wavefield_t{t:.2f}s.png")
            plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close()
            print(f"  - {fname}")

    def _plot_ps_wavefield(self, snapshots, output_dir):
        refP = max((float(np.abs(s[3]).max()) for s in snapshots), default=1.0) or 1.0
        refS = max((float(np.abs(s[4]).max()) for s in snapshots), default=1.0) or 1.0
        for snap in snapshots:
            t, _, _, P_s, S_s = snap
            fig, axes = plt.subplots(1, 2, figsize=(15, 7))
            fig.patch.set_facecolor("white")
            fig.suptitle(f"P / S wavefield decomposition at t = {t:.2f} s",
                         fontsize=14, fontweight="bold")
            for ax, data, label in zip(
                axes, [P_s, S_s],
                ["P-wavefield   ∇·v   (compressional)",
                 "S-wavefield   (∇×v)$_y$   (shear)"]):
                vmax = np.percentile(np.abs(data), 99) + 1e-30
                im = ax.imshow(data, cmap="seismic", vmin=-vmax, vmax=vmax,
                               extent=self._extent_km(), aspect="equal",
                               interpolation="bilinear")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)\
                   .set_label(r"Amplitude  (s$^{-1}$)", fontsize=9)
                ax.set_title(label, fontsize=12, fontweight="bold")
                ax.set_xlabel("X (km)"); ax.set_ylabel("Z (km)")
                self._decorate(ax, marker_color="k", edge="white", text_color="black")
                ax.legend(fontsize=8, loc="upper right")
                _ref, _lab = (refP, "P max") if label.startswith("P") else (refS, "S max")
                self._annotate_amp(ax, data, _ref, _lab, False)
            plt.tight_layout()
            fname = os.path.join(output_dir, f"wavefield_PS_t{t:.2f}s.png")
            plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close()
            print(f"  - {fname}")

    def _arrival_markers(self, iz, ix):
        """Direct P time and first boundary-reflected P time (image sources),
        both including the Ricker delay t0 = 1.5/f0."""
        t0 = 1.5 / self._f0 if self._f0 else 0.0
        sx, sz = self._to_phys(self._src_ix, self._src_iz)
        rx, rz = self._to_phys(ix, iz)
        v = self.vp_arr[self._src_iz, self._src_ix]
        dist = np.hypot(rx - sx, rz - sz)
        xa, xb = self.x0, self.x0 + (self.nx - 1)*self.dx
        za, zb = self.z0, self.z0 + (self.nz - 1)*self.dz
        images = [(2*xa - sx, sz), (2*xb - sx, sz), (sx, 2*za - sz), (sx, 2*zb - sz)]
        d_ref = min(np.hypot(rx - ix_, rz - iz_) for ix_, iz_ in images)
        return dist, t0 + dist / v, t0 + d_ref / v

    def _plot_seismograms(self, nt, t_total, output_dir):
        if not self._receivers: return
        times = np.arange(nt) * self.dt
        for name in self._traces:
            np.savez(os.path.join(output_dir, f"trace_{name}.npz"),
                     t=times,
                     vx=np.asarray(self._traces[name]["vx"]),
                     vz=np.asarray(self._traces[name]["vz"]),
                     P=np.asarray(self._traces[name]["P"]),
                     S=np.asarray(self._traces[name]["S"]))
            print(f"  - {os.path.join(output_dir, f'trace_{name}.npz')}")

        show_refl = self.pml == 0
        n_recv = len(self._receivers)
        components  = ["vx", "vz", "P", "S"]
        comp_labels = ["Vx  (horizontal velocity)", "Vz  (vertical velocity)",
                       "P   (divergence  dvx/dx + dvz/dz)",
                       "S   (curl  dvz/dx - dvx/dz)"]

        def _markers(ax, t_p, t_r, label=True):
            if t_p < t_total:
                ax.axvline(t_p, color="#009944", lw=1.2, ls="--", alpha=0.85)
                if label:
                    ax.text(t_p + t_total*0.01, 0.88, f"P  {t_p:.2f}s", color="#009944",
                            fontsize=8, transform=ax.get_xaxis_transform())
            if show_refl and t_r < t_total:
                ax.axvline(t_r, color="#cc0000", lw=1.2, ls=":", alpha=0.9)
                if label:
                    ax.text(t_r + t_total*0.01, 0.72, f"boundary refl. {t_r:.2f}s",
                            color="#cc0000", fontsize=8,
                            transform=ax.get_xaxis_transform())

        for name, iz, ix in self._receivers:
            dist, t_p, t_r = self._arrival_markers(iz, ix)
            fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
            fig.patch.set_facecolor("white")
            fig.suptitle(f"Seismograms — {name}  (dist = {dist/1000:.1f} km)",
                         fontsize=13, fontweight="bold")
            for ax, comp, clabel in zip(axes, components, comp_labels):
                trace = np.array(self._traces[name][comp])
                t_arr = times[:len(trace)]
                peak  = np.abs(trace).max() or 1.0
                tn    = trace / peak
                ax.plot(t_arr, tn, color="black", linewidth=0.9)
                ax.axhline(0, color="lightgray", linewidth=0.6)
                ax.fill_between(t_arr, tn, 0, where=(tn > 0), color="black", alpha=0.25, linewidth=0)
                _markers(ax, t_p, t_r)
                ax.set_ylabel("Norm. Amp.", fontsize=9); ax.set_ylim(-1.35, 1.35)
                ax.set_title(clabel, fontsize=10, loc="left", pad=3)
                ax.grid(True, color="lightgray", linewidth=0.4)
                ax.text(0.99, 0.94, f"peak = {peak:.2e}", transform=ax.transAxes,
                        fontsize=8, ha="right", va="top",
                        bbox=dict(boxstyle="round,pad=0.25", fc="lightyellow", ec="gray", alpha=0.85))
            axes[-1].set_xlabel("Time (s)", fontsize=10); axes[-1].set_xlim(0, t_total)
            plt.tight_layout()
            fname = os.path.join(output_dir, f"seismogram_{name}.png")
            plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close()
            print(f"  - {fname}")

        fig, axes = plt.subplots(n_recv, 2, figsize=(14, 4*n_recv), sharex=True)
        fig.patch.set_facecolor("white")
        fig.suptitle("Seismograms — Vx & Vz (all receivers)", fontsize=13, fontweight="bold")
        if n_recv == 1: axes = axes.reshape(1, -1)
        for row, (name, iz, ix) in enumerate(self._receivers):
            dist, t_p, t_r = self._arrival_markers(iz, ix)
            for col, comp in enumerate(["vx", "vz"]):
                ax = axes[row][col]
                trace = np.array(self._traces[name][comp]); t_arr = times[:len(trace)]
                peak  = np.abs(trace).max() or 1.0; tn = trace / peak
                ax.plot(t_arr, tn, color="black", linewidth=0.9)
                ax.axhline(0, color="lightgray", linewidth=0.5)
                ax.fill_between(t_arr, tn, 0, where=(tn > 0), color="black", alpha=0.22, linewidth=0)
                _markers(ax, t_p, t_r, label=False)
                ax.set_title(f"{name}  —  {'Vx' if comp=='vx' else 'Vz'}", fontsize=10, fontweight="bold")
                ax.set_ylabel("Norm. Amp.", fontsize=9); ax.set_ylim(-1.35, 1.35)
                ax.grid(True, color="lightgray", linewidth=0.4)
                ax.text(0.99, 0.94, f"dist={dist/1000:.1f} km  peak={peak:.2e}",
                        transform=ax.transAxes, fontsize=7.5, ha="right", va="top",
                        bbox=dict(boxstyle="round,pad=0.25", fc="lightyellow", ec="gray", alpha=0.85))
        for ax in axes[-1]:
            ax.set_xlabel("Time (s)", fontsize=10); ax.set_xlim(0, t_total)
        plt.tight_layout()
        fname = os.path.join(output_dir, "seismograms_all.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  - {fname}")
