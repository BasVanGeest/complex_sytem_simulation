# Bornholdt Baseline Spin-Market Model

This repository contains a 2D Bornholdt spin-market model implemented on an L×L periodic lattice with asynchronous single site heat-bath updates, plus scripts to run simulations and reproduce paper-style figures.

## Folder Overview

- **`bornholdt_model.py`**  
  Core 2D Bornholdt spin-market model.  
  Creates `simulate`, `sweep`, and helpers (e.g., magnetization `M`, `frac_chartist`, etc.).

- **`run_baseline.py`**  
  CLI to run a single lattice simulation and write one CSV (default: `intermediate.csv`).  
  Controls: `steps`, `burn_in`, `thin`, `seed`, and model parameters (`L`, `J`, `alpha`, `T`).  
  Outputs:  
  `t, M, r, abs_r, C_mean, chartist_frac, fundamentalist_frac`

- **`plot_paper_figures.py`**  
  Reads the baseline CSV and generates the figures listed below into `paper_figures/`:
  1. Returns time series  
  2. CCDF of `|r|`  
  3. Volatility autocorrelation  
  4. Chartist fraction vs volatility

- **`visualize_lattice.py`**  
  A live Matplotlib animation of the lattice. Press **`t`** to toggle between:
  - decision spins **S**
  - strategy spins **C**

- **`run_models.py`**  
  General runner that can simulate either:
  - the lattice (`Bornholdt2D`), or
  - network topologies (BA/ER/WS) via the package `bornholdt_network`.

  Writes a CSV with the same schema as `run_baseline.py`.  
  If `--out` is not provided, it auto-builds an output name under `../../data/`.

- **`__init__.py`**  
  Empty marker so the directory can be imported as a package.

## Dependencies

- `numpy`
- `matplotlib`
- `tqdm`
- `networkx`

No external data files are required. Outputs will be created locally.

## Run Order

Generate data: intermediate.csv --steps 200000 --burn_in 10000 --thin 1 --L 32 --J 1 --alpha 8 --T 1.5 --seed 0 \
Creates data/ if missing; first row(s) have NaN for returns because r(t) needs two magnetization points. \
Make figures: python plot_paper_figures.py \
Expects intermediate.csv; outputs PNGs to paper_figures/. \
Optional visualization: python visualize_lattice.py \
Uses defaults mirroring the paper unless overridden in params.json; interactive window, no files written. \
Alternative network runs: ba_20k.csv
