# pml_model — C-PML configuration

40 × 40 km model (800 × 800 cells, 50 m) with a convolutional PML using damping,
frequency shift and grid stretching (d, α, κ). Includes the Monte Carlo
radiative-transfer solver and fast sanity tests.

| File | Role |
|---|---|
| `model.py` | Runs one simulation (CLI; supports `--anomaly`) |
| `elastic_solver.py` | Staggered-grid FD solver with C-PML |
| `velocity_model.py` | Random media with the perturbation tapered out of the PML |
| `geometry.py` | Source–receiver layout |
| `mc_solver.py` | 2D Monte Carlo radiative transfer (energy envelopes) |
| `run_all.py` | Homogeneous + heterogeneous batch |
| `smoke_test.py` | Quick numerical checks (taper, stability, energy decay) |

```bash
python model.py --geometry-only
python model.py --perturbation 0.05 --corr 6 --output-dir results/hetero
python smoke_test.py
```
