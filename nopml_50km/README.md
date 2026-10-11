# nopml_50km — final research configuration

The 40 × 40 km model area is embedded in a 50 × 50 km grid (5 km pad on every side,
rigid walls, no absorbing layer). All analysis windows end before the first boundary
reflection, so the results are free of absorbing-boundary artefacts.

| File | Role |
|---|---|
| `model.py` | Runs one simulation (CLI; `python model.py -h`) |
| `elastic_solver.py` | Staggered-grid FD solver, P/S separation, energy diagnostics, plots |
| `velocity_model.py` | Homogeneous, random (exp/Gaussian PSDF) and anomaly media |
| `geometry.py` | Source–receiver layout and distance plots |
| `run_sets.py` | Runs the simulation sets with per-run logs (`--quick` for a pipeline test) |
| `compare_sets.py` | Side-by-side wavefields and seismograms on a common scale |
| `thesis_figures.py` | Publication figures (PNG + PDF) and metric tables (CSV) |
| `legacy/` | Earlier velocity-model versions kept for traceability |

```bash
python run_sets.py --quick
python run_sets.py
python thesis_figures.py
```

Outputs go to `results/` (git-ignored; regenerate with the commands above).
