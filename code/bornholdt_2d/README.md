# Bornholdt Baseline Spin-Market Model

This repository contains a 2D Bornholdt spin-market model implemented on an L×L periodic lattice with asynchronous single site heat-bath updates, plus scripts to run simulations and reproduce paper-style figures. We recommend not simulating more than 200,000 steps to keep run time under 30 minutes.

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

- **`__init__.py`**  
  Empty marker so the directory can be imported as a package.

## Dependencies

- `numpy`
- `matplotlib`
- `tqdm`
- `networkx`

No external data files are required. Outputs will be created locally.

## Run Order

Run a baseline lattice simulation: python code/bornholdt_2d/run_baseline.py
This will: a csv file called "lattice_data_results_50000.csv" in the data folder.
Example with parameters:
python code/bornholdt_2d/run_baseline.py --steps 50000 --burn_in 10000  --thin 1 --L 32 --J 1.0 --alpha 8.0  --T 1.5 --seed 0

To generate the figures:
python code/bornholdt_2d/plot_paper_figures.py --data data/lattice_data_results_50000.csv
This will save the figures in the results folder.

To visualize the lattice:
python code/bornholdt_2d/visualize_lattice.py
(Override parameters if desired: --L 64 --alpha 8 --T 1.2)
