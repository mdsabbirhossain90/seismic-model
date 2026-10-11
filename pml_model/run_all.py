"""
run_all.py  —  Execute both simulation runs sequentially
=========================================================
Run A: Homogeneous  -> outputs/homogeneous/
Run B: Heterogeneous (seed=42) -> outputs/heterogeneous/

Usage:
    python run_all.py

Then paste both terminal outputs into Claude for the technical report.
"""

import subprocess
import sys
import time
import os

RUNS = [
    {
        "label"  : "RUN A — Homogeneous",
        "args"   : ["--model", "homogeneous", "--output-dir", "outputs/homogeneous"],
    },
    {
        "label"  : "RUN B — Heterogeneous (seed=42, perturbation=0.15)",
        "args"   : ["--model", "heterogeneous", "--seed", "42",
                    "--output-dir", "outputs/heterogeneous"],
    },
]

def banner(text, char="=", width=72):
    print()
    print(char * width)
    print(f"  {text}")
    print(char * width)
    print()

def main():
    banner("SEISMIC SIMULATION  —  FULL REPRODUCIBLE DUAL RUN")
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  Runs    : {len(RUNS)}")
    print(f"  Started : {time.strftime('%Y-%m-%d %H:%M:%S')}")

    total_start = time.time()

    for i, run in enumerate(RUNS, 1):
        banner(f"[{i}/{len(RUNS)}]  {run['label']}")
        cmd = [sys.executable, "model.py"] + run["args"]
        print(f"  Command : {' '.join(cmd)}\n")

        t_start = time.time()
        result  = subprocess.run(cmd)
        elapsed = time.time() - t_start

        if result.returncode != 0:
            print(f"\n  ERROR: Run {i} failed with code {result.returncode}")
            sys.exit(1)

        print(f"\n  Finished in {elapsed:.1f} s")

    total = time.time() - total_start
    banner("ALL RUNS COMPLETE")
    print(f"  Total time : {total:.1f} s  ({total/60:.2f} min)")
    print(f"  Outputs    : outputs/homogeneous/  &  outputs/heterogeneous/")
    print(f"  Completed  : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

if __name__ == "__main__":
    main()