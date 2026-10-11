"""
geometry.py — Source-Receiver Geometry
"""
import numpy as np
import matplotlib.pyplot as plt
import os


def load_receivers(receiver_list):
    print("Receivers loaded:")
    for i, (name, x, z) in enumerate(receiver_list, 1):
        print(f"  {i}. {name}: ({x:.1f}, {z:.1f}) m")
    return receiver_list


def geometry_summary(source, receivers):
    sx, sz = source
    sep  = "=" * 72
    dash = "-" * 72
    print(f"\n{sep}\nSOURCE-RECEIVER GEOMETRY SUMMARY\n{sep}")
    print(f"\nSource Location: ({sx:.1f}, {sz:.1f}) m")
    print(f"\nReceivers ({len(receivers)} total):")
    print(dash)
    print(f"{'No.':<6} {'Name':<12} {'X (m)':>12} {'Z (m)':>12} {'Distance (m)':>14}")
    print(dash)
    distances = []
    for i, (name, rx, rz) in enumerate(receivers, 1):
        d = np.sqrt((rx-sx)**2 + (rz-sz)**2)
        distances.append(d)
        print(f"{i:<6} {name:<12} {rx:>12.1f} {rz:>12.1f} {d:>14.1f}")
    print(dash)
    print(f"\nDistance Statistics:")
    print(f"  Min distance: {np.min(distances):.1f} m")
    print(f"  Max distance: {np.max(distances):.1f} m")
    print(f"  Avg distance: {np.mean(distances):.1f} m")
    print(f"  Std deviation: {np.std(distances):.1f} m")
    return distances


def plot_geometry(source, receivers, distances, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    sx, sz = source

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0a0a0a")
    fig.suptitle("Source-Receiver Geometry", fontsize=13,
                 fontweight="bold", color="white")

    # Map
    ax = axes[0]
    ax.set_facecolor("#0a0a14")
    ax.scatter(sx/1000, sz/1000, c="red", s=300, marker="*",
               zorder=6, label="Source", edgecolors="white", linewidths=0.5)
    ax.annotate(f"Source\n({sx:.0f}, {sz:.0f}) m", (sx/1000, sz/1000),
                textcoords="offset points", xytext=(8, 8),
                fontsize=8, color="#ff6666", fontweight="bold")
    colors = ["#00cfff", "#00ff99", "#ffaa00"]
    for i, (name, rx, rz) in enumerate(receivers, 1):
        c = colors[(i-1) % len(colors)]
        ax.scatter(rx/1000, rz/1000, c=c, s=150, marker="v", zorder=5,
                   edgecolors="white", linewidths=0.5)
        ax.plot([sx/1000, rx/1000], [sz/1000, rz/1000], "--",
                color=c, alpha=0.35, lw=0.9)
        ax.annotate(f"RCV.{i}", (rx/1000, rz/1000),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=8, color=c)
    ax.set_xlabel("X (km)", color="white"); ax.set_ylabel("Z (km)", color="white")
    ax.set_title("Geometry Map", color="white", fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, color="#222244", alpha=0.5)
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#333355")
    ax.legend(facecolor="#111111", edgecolor="gray", labelcolor="white")

    # Distances
    ax2 = axes[1]
    ax2.set_facecolor("#0a0a14")
    names = [f"RCV.{i+1}" for i in range(len(receivers))]
    bars  = ax2.bar(names, [d/1000 for d in distances],
                    color=colors[:len(receivers)], edgecolor="white",
                    linewidth=0.5, alpha=0.85)
    for bar, d in zip(bars, distances):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f"{d/1000:.1f} km", ha="center", fontsize=9, color="white")
    ax2.set_xlabel("Receiver", color="white")
    ax2.set_ylabel("Distance from Source (km)", color="white")
    ax2.set_title("Source-Receiver Distances", color="white", fontweight="bold")
    ax2.grid(True, axis="y", color="#222244", alpha=0.5)
    ax2.tick_params(colors="white")
    for sp in ax2.spines.values(): sp.set_edgecolor("#333355")

    plt.tight_layout()
    out1 = os.path.join(output_dir, "source_receiver_geometry.png")
    out2 = os.path.join(output_dir, "source_receiver_distances.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    fig2, ax3 = plt.subplots(figsize=(8, 5), facecolor="#0a0a0a")
    ax3.set_facecolor("#0a0a14")
    b2 = ax3.bar(names, [d/1000 for d in distances],
                 color=colors[:len(receivers)], edgecolor="white",
                 linewidth=0.5, alpha=0.85)
    for bar, d in zip(b2, distances):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.2,
                 f"{d/1000:.1f} km", ha="center", fontsize=9, color="white")
    ax3.set_xlabel("Receiver", color="white")
    ax3.set_ylabel("Distance (km)", color="white")
    ax3.set_title("Source-Receiver Distances", color="white", fontweight="bold")
    ax3.grid(True, axis="y", color="#222244", alpha=0.5)
    ax3.tick_params(colors="white")
    for sp in ax3.spines.values(): sp.set_edgecolor("#333355")
    fig2.tight_layout()
    fig2.savefig(out2, dpi=150, bbox_inches="tight",
                 facecolor=fig2.get_facecolor())
    plt.close()

    print(f"\nPlots saved:")
    print(f"  - {out1}")
    print(f"  - {out2}")
