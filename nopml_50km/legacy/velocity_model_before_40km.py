"""
velocity_model.py  — supports Gaussian/Exponential PSDF + anomaly mode
====================================================================
Background: Vp=6000 m/s, Vs=3464 m/s (Poisson solid), ρ=2700 kg/m³.

Functions:
  homogeneous(...)              uniform medium
  heterogeneous_random(...)     full-domain random medium (one ε, one a)
  heterogeneous_with_anomaly(...) bg ε + box of different ε inside (anomaly)

v5: perturbation is tapered to zero across the C-PML band so the absorbing
    layer sees a laterally-invariant background (kills boundary re-radiation).
v6: no-PML mode — pml_cells defaults to 0 (no taper, the random medium
    extends to the grid edge). The anomaly box is placed in PHYSICAL
    coordinates via `origin`, so a padded grid (starting at -5 km) keeps the
    box at 18–26 km. Velocity plot shows the whole computational domain with
    the 40 x 40 km model area outlined.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


def pml_taper_2d(nz, nx, npml, ramp=12):
    """Weight field: 0 across the PML band, cosine-rising to 1 over `ramp`
    cells, 1 in the interior.  Multiply the perturbation field xi by this so
    velocities relax to the background mean inside the absorbing layer."""
    rise = 0.5 * (1.0 - np.cos(np.pi * np.arange(1, ramp + 1) / (ramp + 1)))
    edge = np.concatenate([np.zeros(npml), rise])      # length npml+ramp
    n = npml + ramp
    wx = np.ones(nx); wz = np.ones(nz)
    wx[:n] = edge; wx[nx - n:] = edge[::-1]
    wz[:n] = edge; wz[nz - n:] = edge[::-1]
    return wz[:, None] * wx[None, :]


def _make_random_field(nz, nx, dx, dz, a_m, epsilon, psdf, seed):
    """Generate one random field xi with RMS = epsilon."""
    rng = np.random.default_rng(seed)
    eta = rng.standard_normal((nz, nx))
    Eta = np.fft.fft2(eta)
    kx = 2*np.pi*np.fft.fftfreq(nx, d=dx)
    kz = 2*np.pi*np.fft.fftfreq(nz, d=dz)
    KX, KZ = np.meshgrid(kx, kz)
    K2 = KX**2 + KZ**2
    if psdf == 'gauss':
        P = np.pi*epsilon**2*a_m**2 * np.exp(-K2*a_m**2/4.0)
    else:  # 'exp'
        P = 2*np.pi*epsilon**2*a_m**2 / (1.0 + K2*a_m**2)**1.5
    P[0, 0] = 0.0
    xi = np.fft.ifft2(Eta * np.sqrt(np.maximum(P, 0))).real
    return xi * (epsilon / xi.std())


def heterogeneous_random(nx, nz, dx=50.0, dz=50.0,
                          vp_mean=6000.0, vs_mean=None, rho_mean=2700.0,
                          perturbation=0.10, corr_x=20, corr_z=20, seed=42,
                          psdf='exp',
                          vp_floor=1500.0, vp_ceil=8500.0,
                          vs_floor=1800.0, vs_ceil=5000.0,
                          pml_cells=0, pml_ramp=12):
    if perturbation == 0.0:
        return homogeneous(nx, nz, vp=vp_mean, rho=rho_mean)

    a_m = 0.5 * (corr_x*dx + corr_z*dz)
    xi  = _make_random_field(nz, nx, dx, dz, a_m, perturbation, psdf, seed)
    if pml_cells:                                          # taper out of PML
        xi = xi * pml_taper_2d(nz, nx, pml_cells, pml_ramp)

    vs_m_ = vs_mean if vs_mean else vp_mean/np.sqrt(3)
    vp_m  = np.clip(vp_mean*(1.0+xi), vp_floor, vp_ceil)
    vs_m  = np.clip(vs_m_  *(1.0+xi), vs_floor, vs_ceil)
    rho_m = np.full((nz, nx), rho_mean)
    return vp_m, vs_m, rho_m


def heterogeneous_with_anomaly(
        nx, nz, dx=50.0, dz=50.0,
        vp_mean=6000.0, vs_mean=None, rho_mean=2700.0,
        eps_bg=0.0, eps_anom=0.10,
        corr_x=40, corr_z=40, seed=42, psdf='exp',
        anom_x=(18000.0, 26000.0),    # metres  (Xmin, Xmax)
        anom_z=(16000.0, 24000.0),    # metres  (Zmin, Zmax)
        vp_floor=1500.0, vp_ceil=8500.0,
        vs_floor=1800.0, vs_ceil=5000.0,
        pml_cells=0, pml_ramp=12, origin=(0.0, 0.0)):
    """Background medium (ε_bg) with a rectangular anomaly region of ε_anom.
    The background random field is tapered out of the PML band; the anomaly
    box is interior so it is unaffected."""
    a_m = 0.5 * (corr_x*dx + corr_z*dz)
    vs_m_ = vs_mean if vs_mean else vp_mean/np.sqrt(3)
    taper = pml_taper_2d(nz, nx, pml_cells, pml_ramp) if pml_cells else 1.0

    # Background medium
    if eps_bg == 0.0:
        vp_m = np.full((nz, nx), vp_mean)
        vs_m = np.full((nz, nx), vs_m_)
    else:
        xi_bg = _make_random_field(nz, nx, dx, dz, a_m, eps_bg, psdf, seed) * taper
        vp_m  = np.clip(vp_mean*(1.0+xi_bg), vp_floor, vp_ceil)
        vs_m  = np.clip(vs_m_  *(1.0+xi_bg), vs_floor, vs_ceil)

    # Anomaly random field (different seed so it's independent)
    xi_an = _make_random_field(nz, nx, dx, dz, a_m, eps_anom, psdf, seed+1000)
    vp_an = np.clip(vp_mean*(1.0+xi_an), vp_floor, vp_ceil)
    vs_an = np.clip(vs_m_  *(1.0+xi_an), vs_floor, vs_ceil)

    # Build anomaly mask
    X = origin[0] + np.arange(nx)[None, :]*dx
    Z = origin[1] + np.arange(nz)[:, None]*dz
    mask = ((X >= anom_x[0]) & (X <= anom_x[1]) &
            (Z >= anom_z[0]) & (Z <= anom_z[1]))
    vp_m = np.where(mask, vp_an, vp_m)
    vs_m = np.where(mask, vs_an, vs_m)

    rho_m = np.full((nz, nx), rho_mean)
    bbox = (anom_x[0], anom_x[1], anom_z[0], anom_z[1])
    return vp_m, vs_m, rho_m, bbox


def homogeneous(nx, nz, vp=6000.0, rho=2700.0):
    vp_m  = np.full((nz, nx), vp)
    vs_m  = np.full((nz, nx), vp/np.sqrt(3))
    rho_m = np.full((nz, nx), rho)
    return vp_m, vs_m, rho_m


def plot_velocity_model(vp_m, vs_m, dx, dz, source, receivers,
                         output_dir="outputs", title="Velocity Model",
                         anomaly_bbox=None, origin=(0.0, 0.0),
                         main_box=None, view="full"):
    """Clean velocity-model figure: only the model inside the plot.
    Model-area limits are marked in red OUTSIDE the plot (top and right
    rulers); source / receiver / anomaly positions are listed below."""
    os.makedirs(output_dir, exist_ok=True)
    nz, nx = vp_m.shape
    x0, z0 = origin
    extent = [x0/1000, (x0 + nx*dx)/1000, (z0 + nz*dz)/1000, z0/1000]
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    for ax, data, label, cmap in zip(
        axes, [vp_m, vs_m], ["Vp (m/s)", "Vs (m/s)"], ["jet", "viridis"]):
        im = ax.imshow(data, cmap=cmap, extent=extent, aspect="equal")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.09).set_label(label, fontsize=10)
        ax.set_xlabel("X (km)", fontsize=10); ax.set_ylabel("Depth (km)", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        if main_box is not None:
            xmin, xmax, zmin, zmax = [v/1000 for v in main_box]
            if view == "main":
                ax.set_xlim(xmin, xmax); ax.set_ylim(zmax, zmin)
            top = ax.secondary_xaxis("top")
            top.set_xticks([xmin, xmax])
            top.set_xticklabels([f"{xmin:g}", f"{xmax:g}"])
            top.tick_params(colors="red", length=9, width=1.8, labelsize=9)
            top.spines["top"].set_visible(False)
            top.set_xlabel(f"model area {xmin:g}–{xmax:g} km", color="red", fontsize=9)
            right = ax.secondary_yaxis("right")
            right.set_yticks([zmin, zmax])
            right.set_yticklabels([f"{zmin:g}", f"{zmax:g}"])
            right.tick_params(colors="red", length=9, width=1.8, labelsize=9)
            right.spines["right"].set_visible(False)

    # reference positions (outside the plots)
    sx, sz = source
    ref = [f"Source ({sx/1000:g}, {sz/1000:g}) km"]
    for i, (name, rx, rz) in enumerate(receivers, 1):
        disp = f"RCV.{name}" if isinstance(name, str) else f"RCV.{i}"
        ref.append(f"{disp} ({rx/1000:g}, {rz/1000:g}) km")
    if anomaly_bbox is not None:
        a = [v/1000 for v in anomaly_bbox]
        ref.append(f"Anomaly box x {a[0]:g}–{a[1]:g} km, z {a[2]:g}–{a[3]:g} km")
    fig.text(0.5, 0.015, "     |     ".join(ref), ha="center", va="bottom",
             fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    out = os.path.join(output_dir, "velocity_model.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  - {out}")
