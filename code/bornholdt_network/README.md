# Bornholdt Network Model

Bornholdt spin-market model implemented on **arbitrary network topologies** (BA / ER / WS) using the same asynchronous heat-bath updates.  
Includes a CLI runner to generate a standardized time-series CSV, a plotting script to reproduce paper-style figures, and an interactive network visualization.

To keep run times below 30 minutes we recommend not excedding 200,000 steps. 

Topology specific parameters:
- BA: --m
- ER: --p (edge probability)
- WS: --k (neighbors), --p (rewiring probability)

## Folder Overview

- **`model_network.py`**  
  Bornholdt spin-market model on an arbitrary network:
  - fully asynchronous updates
  - heat-bath spin flips
  - magnetization/strategy helpers
  - returns-from-magnetization utility

- **`networks.py`**  
  Small factory to build network topologies (BA, ER, WS) via NetworkX with convenience defaults.

- **`run_networks.py`**  
  CLI runner to simulate the model and write one time-series CSV (default: `network_run.csv`).  
  CSV columns:
  `t, M, r, abs_r, C_mean, chartist_frac, fundamentalist_frac`

- **`plot_figures_network.py`**  
  CLI to load the CSV and reproduce paper-style figures into `paper_figures_network/` by default:
  - returns time series
  - CCDF of `|r|`
  - volatility autocorrelation (ACF)
  - chartist fraction vs volatility

- **`visualize_network.py`**  
  Interactive Matplotlib visualization of spins on the network.  
  Loads defaults from `params_network.json` if present and allows keyboard toggling between:
  - decision spins
  - strategy spins
  
  Press `t` while the window is active to toggle S ↔ C.

  run_networks.py uses N = L×L nodes regardless of topology; adjust --L to scale system size.
  Returns are log changes of |M| with a small epsilon; the first row is NaN by design.
  plot_figures_network.py auto-detects the CSV path if --data is not provided (network_run.csv preferred).
  visualize_network.py reads overrides from params_network.json if present; otherwise uses baked defaults:

- **`__init__`**  
  Marks this directory as a Python module.

## Dependencies

Packages:
- `numpy`
- `networkx`
- `matplotlib`

Stdlib:
- `argparse`, `os`, `json`, `io`
- randomness handled via NumPy RNG

Scripts assume a writable `data/` and `paper_figures_network/` directory.

## Run Order

Run network simulations:
python code/bornholdt_network/run_networks.py --steps 50000

By default, this runs all three topologies (ER, BA, WS) and produces:
data/ER_data_results_50000.csv
data/BA_data_results_50000.csv
data/WS_data_results_50000.csv

Each CSV contains the same schema as the lattice baseline, enabling direct comparison.

How to run one topology only:
python code/bornholdt_network/run_networks.py --steps 50000 --topology ER

Example with explicit parameters:
python run_networks.py  --steps 100000  --topology WS  --L 32  --k 6  --p 0.1  --alpha 8  --T 1.5  --seed 0
(The number of nodes is always N = L × L, to keep same nodes as cells in the lattice)

How to generate figures from a network run:
python code/bornholdt_network/plot_figures_networks.py --data data/ER_data_results_50000.csv
This generates and saves the following figures into results folder:
ER_returns_timeseries.png
ER_ccdf_abs_returns.png
ER_volatility_autocorr.png
ER_chartist_fraction_vs_volatility.png

(Optional) Interactive network visualization: 
python visualize_networks.py --topology ER --p 0.01
python visualize_networks.py --topology BA --m 2
python visualize_networks.py --topology WS --k 6 --p 0.05