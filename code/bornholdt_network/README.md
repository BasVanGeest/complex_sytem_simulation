# Bornholdt Network Model

Bornholdt spin-market model implemented on **arbitrary network topologies** (BA / ER / WS) using the same asynchronous heat-bath updates.  
Includes a CLI runner to generate a standardized time-series CSV, a plotting script to reproduce paper-style figures, and an interactive network visualization.

To keep run times below 30 minutes we recommend not excedding 200,000 steps. 

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
  Topology-specific parameters:
  BA: --m
  ER: --p (edge probability)
  WS: --k (neighbors), --p (rewiring probability)
  Returns are log changes of |M| with a small epsilon; the first row is NaN by design.
  Thinning and burn-in are handled in-loop before writing.
  plot_figures_network.py auto-detects the CSV path if --data is not provided (network_run.csv preferred).
  visualize_network.py reads overrides from params_network.json if present; otherwise uses baked defaults:

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

Simulate with run_networks.py (uses model_network.py to generate network_run.csv
Generate figures from the simulation CSV with plot_figures_network.py, network_run.csv --outdir paper_figures_network --prefix network_
Optional Live animation using the same model, python visualize_network.py
