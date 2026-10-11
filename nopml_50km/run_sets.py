"""
run_sets.py — run the three no-PML simulation sets, then build comparisons
==========================================================================
  Set 1  Homogeneous
  Set 2  Heterogeneous, no anomaly     a = 0.3 km, ε = 0.05
  Set 3  Heterogeneous + anomaly box   a = 0.3 km, ε_bg = 0.05, ε_anom = 0.09

Common to all: no PML, 50x50 km grid (40x40 km model area + 5 km pad),
exp PSDF, seed 42, f0 = 4.78 Hz, T = 10 s, snapshots 2/4/6/8 s.

Usage:
  python run_sets.py              # all three sets (long)
  python run_sets.py --quick      # 1-second test of all three (checks pipeline)
  python run_sets.py --only 2 3   # just sets 2 and 3
"""

import argparse, os, subprocess, sys, time

ROOT_DEFAULT = "results/noPML_50km"

COMMON = ["--pml", "0", "--pad-km", "5", "--psdf", "exp", "--seed", "42",
          "--f0", "4.78", "--order", "4"]

SETS = {
    1: ("1_homogeneous",
        "Homogeneous",
        ["--model", "homogeneous"]),
    2: ("2_hetero_a0.3km_eps0.05",
        "Heterogeneous, no anomaly (a = 0.3 km, ε = 0.05)",
        ["--model", "heterogeneous", "--perturbation", "0.05", "--corr", "6"]),
    3: ("3_anomaly_a0.3km_eps0.05_anom0.09",
        "Heterogeneous + anomaly (a = 0.3 km, ε_bg = 0.05, ε_anom = 0.09)",
        ["--anomaly", "--eps-bg", "0.05", "--eps-anom", "0.09", "--corr", "6"]),
}


def banner(text, char="=", width=76):
    print("\n" + char*width + f"\n  {text}\n" + char*width + "\n", flush=True)


def run_logged(cmd, log_path):
    """Run cmd, stream output to the terminal and to log_path."""
    with open(log_path, "w") as log:
        log.write("Command: " + " ".join(cmd) + "\n\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return proc.wait()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="+", type=int, choices=[1, 2, 3],
                   default=[1, 2, 3])
    p.add_argument("--quick", action="store_true",
                   help="1-second runs into results/_quicktest (pipeline check)")
    p.add_argument("--root", default=None)
    p.add_argument("--view", default="full", choices=["full", "main"])
    args = p.parse_args()

    root = args.root or ("results/_quicktest" if args.quick else ROOT_DEFAULT)
    if args.quick:
        timing = ["--t-total", "1.0", "--snap", "0.5", "1.0"]
    else:
        timing = ["--t-total", "10", "--snap", "2", "4", "6", "8"]

    os.makedirs(root, exist_ok=True)
    banner(f"NO-PML SIMULATION SETS  →  {root}")
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  Sets    : {args.only}")
    print(f"  Started : {time.strftime('%Y-%m-%d %H:%M:%S')}")

    t_all = time.time()
    for k in args.only:
        folder, label, extra = SETS[k]
        out = os.path.join(root, folder)
        os.makedirs(out, exist_ok=True)
        cmd = ([sys.executable, "-u", "model.py"] + COMMON + extra + timing
               + ["--view", args.view, "--output-dir", out])
        banner(f"[Set {k}]  {label}")
        print("  Command : " + " ".join(cmd) + "\n", flush=True)
        t0 = time.time()
        rc = run_logged(cmd, os.path.join(out, "run_log.txt"))
        if rc != 0:
            print(f"\n  ERROR: set {k} failed (exit code {rc}). "
                  f"See {out}/run_log.txt")
            sys.exit(rc)
        print(f"\n  Set {k} finished in {(time.time()-t0)/60:.1f} min")

    # Comparison figures once all three exist
    have_all = all(os.path.exists(os.path.join(root, SETS[k][0], "snapshots.npz"))
                   for k in SETS)
    if have_all:
        banner("Building comparison figures")
        subprocess.run([sys.executable, "compare_sets.py", "--root", root,
                        "--view", args.view])
    else:
        print("\n  (Comparison figures skipped — not all three sets exist yet.)")

    banner("DONE")
    print(f"  Total time : {(time.time()-t_all)/60:.1f} min")
    print(f"  Results    : {root}/")


if __name__ == "__main__":
    main()
