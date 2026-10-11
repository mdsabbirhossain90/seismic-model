# Seismic Model — 2D Elastic Wave Propagation in Heterogeneous Media

Finite-difference simulation of **2D P-SV elastic wave propagation** using the
velocity–stress staggered-grid method (Virieux, 1986), with random
heterogeneous velocity models, localized scattering anomalies, and a
Monte Carlo radiative-transfer solver for energy-envelope comparison.

## Features

- 4th/2nd-order staggered-grid velocity–stress FD solver (leap-frog in time)
- Velocity models: homogeneous, random heterogeneous
  (exponential or Gaussian PSDF), and heterogeneous background + anomaly box
- Absorbing boundaries via PML (`pml_model/`) or a padded, PML-free domain (`nopml_50km/`)
- Outputs: Vx/Vz and P/S wavefield snapshots, seismograms, energy curves,
  velocity-model and source–receiver geometry plots
- Monte Carlo radiative-transfer energy envelopes (Sato & Fehler, 2012)

## Repository structure

```
seismic-model/
├── pml_model/               # Main model with PML absorbing boundaries (40 × 40 km)
│   ├── model.py             #   CLI entry point
│   ├── elastic_solver.py    #   Staggered-grid elastic FD solver
│   ├── velocity_model.py    #   Velocity-model builders + plotting
│   ├── geometry.py          #   Source/receiver geometry
│   ├── mc_solver.py         #   Monte Carlo radiative transfer
│   ├── run_all.py           #   Homogeneous + heterogeneous batch run
│   └── smoke_test.py        #   Quick sanity check
├── nopml_50km/              # PML-free variant: 40 × 40 km model on a 50 × 50 km padded grid
│   ├── model.py
│   ├── elastic_solver.py
│   ├── velocity_model.py
│   ├── geometry.py
│   ├── run_sets.py          #   Runs the three simulation sets
│   ├── compare_sets.py      #   Side-by-side comparisons
│   ├── thesis_figures.py    #   Final figures and metrics
│   └── legacy/              #   Earlier velocity-model versions (reference only)
├── pml_comparison/
│   └── comparison.py        # PML 20 / 30 / 40 wavefield comparison figure
├── archive/
│   └── setup_noPML.py       # One-shot generator that originally created nopml_50km/
├── requirements.txt
└── README.md
```

Simulation outputs (`results/`, `outputs/`, `*.npz`, …) are large and fully
regenerable, so they are excluded from version control via `.gitignore`.

## Installation

```bash
git clone https://github.com/mdsabbirhossain90/seismic-model.git
cd seismic-model
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### PML model (`pml_model/`)

```bash
cd pml_model

# Geometry + velocity model only (fast preview)
python model.py --geometry-only

# Full simulation
python model.py --model heterogeneous --perturbation 0.05 --corr 6 --output-dir results/hetero

# Heterogeneous background with a scattering anomaly
python model.py --anomaly --eps-bg 0.05 --eps-anom 0.09 --corr 6 --output-dir results/anom_a0.3km

# Homogeneous + heterogeneous batch
python run_all.py

# Sanity check
python smoke_test.py
```

Main options: `--model {homogeneous,heterogeneous}`, `--anomaly`, `--eps-bg`, `--eps-anom`,
`--psdf {exp,gauss}`, `--perturbation`, `--corr` (correlation length in cells),
`--seed`, `--snap`, `--order {2,4}`, `--t-total`, `--f0`, `--output-dir`.

### PML-free model (`nopml_50km/`)

```bash
cd nopml_50km
python run_sets.py --quick     # 1-second pipeline check
python run_sets.py             # full runs (long)
python compare_sets.py         # comparison figures
python thesis_figures.py       # final figures + metrics
```

### PML comparison

Requires the PML20/30/40 runs in `pml_model/results/`.

```bash
cd pml_comparison
mkdir -p output
python comparison.py
```

## Theory

Velocity–stress form of the 2D elastic wave equations:

```
ρ ∂vx/∂t  = ∂σxx/∂x + ∂σxz/∂z
ρ ∂vz/∂t  = ∂σxz/∂x + ∂σzz/∂z
∂σxx/∂t   = (λ+2μ) ∂vx/∂x + λ ∂vz/∂z
∂σzz/∂t   = λ ∂vx/∂x + (λ+2μ) ∂vz/∂z
∂σxz/∂t   = μ (∂vx/∂z + ∂vz/∂x)
```

## References

- Virieux, J. (1986). P-SV wave propagation in heterogeneous media:
  Velocity-stress finite-difference method. *Geophysics*, 51(4), 889–901.
- Sato, H., Fehler, M., & Maeda, T. (2012). *Seismic Wave Propagation and
  Scattering in the Heterogeneous Earth* (2nd ed.). Springer.

## Author

[@mdsabbirhossain90](https://github.com/mdsabbirhossain90)
