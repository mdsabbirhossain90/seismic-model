"""
mc_solver.py
============
2D scalar Monte Carlo radiative transfer simulation for elastic seismic
energy in random heterogeneous media.

Each phonon carries unit energy and propagates in straight lines between
scattering events. Distance to next scattering event is exponentially
distributed with mean = mean free path ℓ. Direction after each event is
randomized (isotropic-scattering kernel — Sato & Fehler 2012, Ch 7).

Output: energy envelope E(t) at each receiver, directly comparable to
the squared-and-smoothed seismogram trace from the FD solver.

References:
    Sato, H., Fehler, M., Maeda, T. (2012). Seismic Wave Propagation and
    Scattering in the Heterogeneous Earth, 2nd ed. Springer. (Ch 4–7)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time


# ──────────────────────────────────────────────────────────────────────
# Mean free path from random-medium statistics (Born approximation)
# ──────────────────────────────────────────────────────────────────────

def mean_free_path_gaussian_2d(V0, f0, eps, a):
    """Mean free path for a 2D Gaussian-correlated random medium under the
    Born (single-scattering) approximation.

        g_total ≈ π · ε² · k² · a · exp(-k²a²/4)        [1/length]
        ℓ       = 1 / g_total

    where k = 2π f₀ / V₀ is the dominant wavenumber.

    Args
    ----
    V0  : mean wave speed  (m/s)
    f0  : dominant source frequency  (Hz)
    eps : RMS velocity perturbation  δ  (dimensionless)
    a   : correlation length  (m)

    Returns
    -------
    ℓ  : mean free path  (m)
    """
    k = 2.0 * np.pi * f0 / V0
    g_total = np.pi * eps**2 * k**2 * a * np.exp(-(k * a)**2 / 4.0)
    if g_total <= 0:
        return np.inf
    return 1.0 / g_total


# ──────────────────────────────────────────────────────────────────────
# Monte Carlo phonon-shooting solver
# ──────────────────────────────────────────────────────────────────────

class RTMonteCarlo2D:
    """2D Monte Carlo radiative transfer phonon-shooting simulation.

    Convention:
        domain origin at (0, 0), domain extent (Lx, Lz).
        Source emits phonons at `source_pos` at time `t_source` with
        random isotropic direction.
        Each phonon propagates at constant speed V0 in straight lines
        between scattering events. Mean free path is `mean_free_path`.
        At each event, direction is reset to a uniform random angle.
        Energy at each receiver is recorded as a histogram of phonon
        passages (density of phonons inside the receiver disk).

    Compared to FD:
        FD solves the wave equation; this solves the Boltzmann transport
        equation for elastic energy under the assumption of incoherent
        single-mode scattering. The two should agree at long times in the
        diffusive regime; deviations indicate coherent (interference)
        effects that RT cannot capture.
    """

    def __init__(self, V0, mean_free_path, domain_size, source_pos,
                 n_phonons=100_000, t_source=1.5, seed=42):
        self.V0           = float(V0)
        self.l_mfp        = float(mean_free_path)
        self.domain_size  = np.asarray(domain_size, float)        # (Lx, Lz)
        self.source_pos   = np.asarray(source_pos, float)         # (sx, sz)
        self.n_phonons    = int(n_phonons)
        self.t_source     = float(t_source)
        self.seed         = int(seed)
        self.receivers    = []

    def add_receiver(self, name, x, z, radius=500.0):
        self.receivers.append({
            'name'  : name,
            'pos'   : np.array([float(x), float(z)]),
            'radius': float(radius),
        })

    def run(self, t_total, dt=0.02, verbose=True):
        n_steps = int(t_total / dt)
        rng     = np.random.default_rng(self.seed)
        n       = self.n_phonons

        # Init phonon positions, directions, scatter distances
        pos        = np.tile(self.source_pos, (n, 1))
        theta      = rng.uniform(0, 2*np.pi, n)
        direction  = np.column_stack([np.cos(theta), np.sin(theta)])
        d_scatter  = -self.l_mfp * np.log(rng.uniform(1e-12, 1.0, n))
        alive      = np.ones(n, dtype=bool)

        # Init receiver bins
        time_axis = np.arange(n_steps) * dt
        for rcv in self.receivers:
            rcv['energy'] = np.zeros(n_steps)
            rcv['time']   = time_axis

        if verbose:
            print(f"  MC simulation:")
            print(f"    n_phonons     : {n}")
            print(f"    mean free path: {self.l_mfp:.0f} m")
            print(f"    n_steps       : {n_steps}  (dt={dt*1000:.0f} ms)")
            print(f"    source delay  : {self.t_source} s")

        t_start = time.time()
        ds      = self.V0 * dt

        for step in range(n_steps):
            t_now    = step * dt
            emitted  = alive & (t_now >= self.t_source)
            n_emit   = int(np.sum(emitted))

            if n_emit == 0:
                continue

            # advance emitted phonons
            pos[emitted]       += direction[emitted] * ds
            d_scatter[emitted] -= ds

            # count phonons inside each receiver disk
            for rcv in self.receivers:
                dx = pos[emitted, 0] - rcv['pos'][0]
                dz = pos[emitted, 1] - rcv['pos'][1]
                rcv['energy'][step] = np.sum(dx*dx + dz*dz < rcv['radius']**2)

            # rescatter phonons whose distance ran out
            scatter_now = emitted & (d_scatter <= 0)
            n_scatter   = int(np.sum(scatter_now))
            if n_scatter > 0:
                new_theta                  = rng.uniform(0, 2*np.pi, n_scatter)
                direction[scatter_now, 0]  = np.cos(new_theta)
                direction[scatter_now, 1]  = np.sin(new_theta)
                d_scatter[scatter_now]     = -self.l_mfp * np.log(
                                                rng.uniform(1e-12, 1.0, n_scatter))

            # kill phonons that left domain
            out  = ((pos[:, 0] < 0) | (pos[:, 0] > self.domain_size[0]) |
                    (pos[:, 1] < 0) | (pos[:, 1] > self.domain_size[1]))
            alive[out] = False

            if verbose and (step % max(1, n_steps//10) == 0):
                n_alive = int(np.sum(alive))
                print(f"    step {step:5d}/{n_steps}  alive={n_alive:6d}  "
                      f"t={t_now:.2f} s")

        if verbose:
            print(f"  MC done in {time.time()-t_start:.1f} s")

        # Normalise: phonon count → energy density (phonons / m² / s / source phonon)
        for rcv in self.receivers:
            A_rcv = np.pi * rcv['radius']**2
            rcv['energy'] = rcv['energy'] / (self.n_phonons * dt * A_rcv)

        return self.receivers

    # ── plotting ──────────────────────────────────────────────────────

    def plot_envelopes(self, output_dir, title="Monte Carlo Energy Envelopes",
                       log_scale=True):
        os.makedirs(output_dir, exist_ok=True)
        n   = len(self.receivers)
        fig, axes = plt.subplots(n, 1, figsize=(10, 3*n), sharex=True)
        if n == 1: axes = [axes]
        fig.patch.set_facecolor("white")
        fig.suptitle(title, fontsize=12, fontweight="bold")

        for ax, rcv in zip(axes, self.receivers):
            ax.set_facecolor("white")
            dist = np.linalg.norm(rcv['pos'] - self.source_pos)
            t_geo = self.t_source + dist / self.V0
            ax.plot(rcv['time'], rcv['energy'], color="black", lw=1.0)
            ax.axvline(t_geo, ls='--', color='#009944', lw=1.2,
                       label=f'ballistic t={t_geo:.2f}s')
            if log_scale:
                ax.set_yscale("log")
                ax.set_ylim(1e-12, None)
            ax.set_ylabel(r"E(t)  (a.u.)")
            ax.set_title(f"{rcv['name']}   dist = {dist/1000:.1f} km",
                         loc="left", fontsize=10)
            ax.grid(True, color="lightgray", lw=0.4)
            ax.legend(fontsize=8, loc="upper right")
        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        out = os.path.join(output_dir, "mc_envelopes.png")
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  - {out}")

    def save_envelopes(self, npz_path):
        """Save envelopes for later overlay with FD."""
        os.makedirs(os.path.dirname(npz_path) or '.', exist_ok=True)
        save_dict = {'time': self.receivers[0]['time']}
        for rcv in self.receivers:
            save_dict[rcv['name']] = rcv['energy']
        np.savez(npz_path, **save_dict)
        print(f"  - {npz_path}")


# ──────────────────────────────────────────────────────────────────────
# Helper: compute energy envelope from FD seismogram trace
# ──────────────────────────────────────────────────────────────────────

def envelope_from_trace(trace, dt, smooth_t=0.2):
    """Energy envelope ∝ trace²(t) smoothed by a moving average of length
    smooth_t (in seconds). Use smooth_t ~ 1/f₀ for a clean comparison
    against MC envelopes."""
    sq = np.asarray(trace, float)**2
    n_smooth = max(1, int(smooth_t / dt))
    # padded moving average
    cum = np.cumsum(np.insert(sq, 0, 0))
    env = (cum[n_smooth:] - cum[:-n_smooth]) / n_smooth
    pad = np.zeros(n_smooth - 1)
    env = np.concatenate([pad, env]) if len(env) < len(sq) else env[:len(sq)]
    return env


# ──────────────────────────────────────────────────────────────────────
# Stand-alone demo / single-shot run
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Match FD setup exactly
    V0   = 4000.0
    f0   = 1.0
    eps  = 0.15
    a    = 1200.0
    L    = 40000.0

    mfp = mean_free_path_gaussian_2d(V0, f0, eps, a)
    print(f"Born-approximation mean free path: {mfp/1000:.2f} km   "
          f"(domain crossings: {L/mfp:.2f})")

    mc = RTMonteCarlo2D(V0=V0, mean_free_path=mfp,
                        domain_size=(L, L),
                        source_pos=(20000.0, 20000.0),
                        n_phonons=200_000)
    mc.add_receiver("RCV.A", 10000.0, 20000.0, radius=500.0)
    mc.add_receiver("RCV.B", 25000.0, 20000.0, radius=500.0)
    mc.add_receiver("RCV.C", 30000.0, 20000.0, radius=500.0)

    mc.run(t_total=8.0, dt=0.02)
    mc.plot_envelopes("mc_outputs/heterogeneous",
                       title=f"MC envelopes — δ={eps}, a={a:.0f} m, ℓ={mfp/1000:.1f} km")
    mc.save_envelopes("mc_outputs/heterogeneous/envelopes.npz")