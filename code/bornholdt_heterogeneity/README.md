# Bornholdt Heterogeneity Model

Asynchronous 2D Bornholdt market-spin model with **quenched heterogeneous** \(\alpha_{ij}\).  
The heterogeneity field \(\alpha_{ij}\) is drawn from a Normal(mean, std) distribution and clipped to be > 0, then held fixed for the full simulation.

Produces time series of magnetization, and strategy composition, plus scripts for overlay plots and interactive lattice animation.

---

## Folder Overview

- **`bornholdt_heterogeneity.py`**  
  Core `Bornholdt2D_heterogeneity` model:
  - draws and stores the quenched \(\alpha_{ij}\) field
  - performs one sweep (random-serial asynchronous updates)
  - runs simulations
  - computes observables (e.g., `M`, return `r`, `C_mean`, chartist/fundamentalist fractions)

- **`run_heterogeneity.py`**  
  CLI runner for a single heterogeneity simulation.  
  Writes one CSV with headers and includes the run parameters in CSV **comment lines**.

- **`plot_overlap_figures.py`**  
  Reads:
  - baseline `intermediate.csv`
  - heterogeneity `intermediate_heterogeneity.csv`  
  and creates overlay figures + an \(\alpha\) histogram into `paper_figures/`.

- **`visualize_latt_heterogeneity.py`**  
  Interactive Matplotlib animation of the lattice fields:
  - `t` toggles S ↔ C
  - `a` toggles α view

- **`__init__.py`**  
  Empty marker so the directory can be imported as a package.

---

## Dependencies

- `numpy`
- `matplotlib`

Standard library: `argparse`, `os`, `re`, `io`

## Run Order

Generate heterogeneous-α simulation data:
python code/bornholdt_heterogeneity/run_heterogeneity.py --steps 50000
This will save the csv output in the data folder and it will be called heterogeneity_data_results_50000.csv
Key parameters: --L (lattice size), --steps (number of sweeps), --burn_in, --thin, --alpha_mean, --alpha_std, --alpha_min, --T, --J, --seed. 
To generate overlap figures (baseline vs heterogeneity):
Default command:python code/bornholdt_heterogeneity/plot_overlap_figures.py
By default, this script reads:
data/lattice_data_results_100000.csv
data/heterogeneity_data_results_100000.csv
and writes overlay figures in the results folder.
To specify different files:
python code/bornholdt_heterogeneity/plot_overlap_figures.py --baseline data/lattice_data_results_50000.csv  --hetero   data/heterogeneity_data_results_50000.csv

Generated figures include:
 - overlay of returns time series,
 - overlay of CCDF(|r|),
 - overlay of volatility autocorrelation.

(Optional) Interactive lattice visualization:
python code/bornholdt_heterogeneity/visualize_lattice_heterogeneity.py
Override parameters if desired:
python code/bornholdt_heterogeneity/visualize_lattice_heterogeneity.py --alpha-mean 8 --alpha-std 2 --T 1.2

