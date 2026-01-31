## Bornholdt Network Model

This folder contains an extension of the Bornholdt spin-market model implemented on **arbitrary network topologies** (Barabási–Albert, Erdős–Rényi, Watts–Strogatz). The microscopic update rules are identical to the lattice baseline: asynchronous single-site heat-bath updates, with returns defined from changes in the global magnetization.

The folder includes a command-line runner to generate standardized time-series CSV files, scripts to reproduce paper-style figures, and an interactive network visualization.

> **Runtime note:** To keep execution time below ~30 minutes on a standard laptop, we recommend not exceeding **200,000 Monte Carlo sweeps**.

### Topology-specific parameters

-   **BA:** `--m` (number of edges per added node)
    
-   **ER:** `--p` (edge probability)
    
-   **WS:** `--k` (initial number of neighbors), `--p` (rewiring probability)
    

----------

## Folder Contents

-   **`model_network.py`**  
    Core Bornholdt spin-market model defined on an arbitrary graph:
    
    -   fully asynchronous heat-bath updates
        
    -   spin interactions along network edges
        
    -   magnetization and strategy observables
        
    -   returns computed from changes in global magnetization
        
-   **`networks.py`**  
    Lightweight factory for generating BA, ER, and WS networks using NetworkX.
    
-   **`run_networks.py`**  
    Command-line interface to run network simulations and write time-series CSV files.  
    Output columns:
    
    `t, M, r, abs_r, C_mean, chartist_frac, fundamentalist_frac` 
    
-   **`plot_figures_network.py`**  
    Reads a network CSV file and reproduces paper-style figures in `paper_figures_network/`:
    
    -   return time series
        
    -   CCDF of |r|
        
    -   volatility autocorrelation
        
    -   chartist fraction versus volatility
        
-   **`visualize_network.py`**  
    Interactive Matplotlib visualization of spins on the network.  
    Press `t` to toggle between decision spins (S) and strategy spins (C).
    
-   **`__init__.py`**  
    Marks the directory as a Python package.
    

----------

## Dependencies

Required packages:

-   `numpy`
    
-   `networkx`
    
-   `matplotlib`
    

Standard library modules: `argparse`, `os`, `json`, `io`

Scripts assume writable `data/` and `paper_figures_network/` directories.

----------

## How to Run

### 1. Run network simulations

`python code/bornholdt_network/run_networks.py --steps 50000` 

By default, this runs all three topologies (ER, BA, WS) and produces:

`data/ER_data_results_50000.csv data/BA_data_results_50000.csv data/WS_data_results_50000.csv` 

Each CSV uses the same schema as the lattice baseline, enabling direct comparison.

To run a single topology only:

`python code/bornholdt_network/run_networks.py --steps 50000 --topology ER` 

Example with explicit parameters:

`python code/bornholdt_network/run_networks.py \
  --steps 100000 \
  --topology WS \
  --L 32 \
  --k 6 \
  --p 0.1 \
  --alpha 8.0 \
  --T 1.5 \
  --seed 0` 

The number of nodes is always set to N=L×LN = L \times LN=L×L to match the lattice baseline.

----------

### 2. Generate figures from a network run

`python code/bornholdt_network/plot_figures_network.py \
  --data data/ER_data_results_50000.csv` 

Figures are saved to the `results/` folder, for example:

-   `ER_returns_timeseries.png`
    
-   `ER_ccdf_abs_returns.png`
    
-   `ER_volatility_autocorr.png`
    
-   `ER_chartist_fraction_vs_volatility.png`
    

----------

### 3. Interactive network visualization (optional)

`python code/bornholdt_network/visualize_network.py --topology ER --p 0.01
python code/bornholdt_network/visualize_network.py --topology BA --m 2
python code/bornholdt_network/visualize_network.py --topology WS --k 6 --p 0.05` 

----------

### Notes

-   Returns are defined as logarithmic changes of |M| with a small numerical offset; the first value is NaN by design.
    
-   Default parameters balance qualitative behavior and computational cost.