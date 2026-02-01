## Bornholdt Model with Agent Heterogeneity
This folder contains an extension of the 2D Bornholdt spin-market model in which agents differ in their sensitivity to global market pressure. Each agent is assigned an **agent-specific contrarian strength** αij\alpha_{ij}αij​, drawn at initialization and kept fixed throughout the simulation.

The model uses asynchronous single-site heat-bath updates on a periodic L×LL \times LL×L lattice and produces time series of magnetization, returns, and strategy composition. Additional scripts allow direct comparison with the baseline model.

----------

## Folder Contents

-   **`bornholdt_heterogeneity.py`**  
    Core implementation of the heterogeneous 2D Bornholdt model.  
    Responsibilities include:
    
    -   drawing and storing the agent-specific αij\alpha_{ij}αij​ field
        
    -   performing random-serial asynchronous sweeps
        
    -   running simulations
        
    -   computing observables such as magnetization `M`, returns `r`, mean strategy `C_mean`, and chartist/fundamentalist fractions
        
-   **`run_heterogeneity.py`**  
    Command-line interface to run a single heterogeneous simulation.  
    Writes one CSV file with headers and records all run parameters as comment lines at the top of the file.
    
-   **`plot_overlap_figures.py`**  
    Reads a baseline CSV and a heterogeneity CSV and produces overlay figures in `paper_figures/`, including:
    
    -   return time series overlays
        
    -   CCDFs of |r|
        
    -   volatility autocorrelation overlays
        
    -   histogram of the α\alphaα distribution
        
-   **`visualize_lattice_heterogeneity.py`**  
    Interactive Matplotlib visualization of the lattice fields.  
    Keyboard controls:
    
    -   `t`: toggle between decision spins SSS and strategy spins CCC
        
    -   `a`: toggle visualization of the α\alphaα field
        
-   **`__init__.py`**  
    Marks the directory as a Python package.
    

----------

## Dependencies

-   `numpy`
    
-   `matplotlib`
    

Standard library modules: `argparse`, `os`, `re`, `io`

----------

## How to Run

Important:
The simulation should run for ≥ 20,000 steps.
The first 10,000 steps act as burn-in, and multiple plots are evaluated using data from steps 10,000 to 20,000. Runs shorter than this will not reproduce the intended results.

### 1. Run a heterogeneous simulation

`python code/bornholdt_heterogeneity/run_heterogeneity.py --steps 50000` 

This generates a CSV file named:

`heterogeneity_data_results_50000.csv` 

in the `data/` folder.

Key parameters include:  
`--L`, `--steps`, `--burn_in`, `--thin`, `--alpha_mean`, `--alpha_std`, `--alpha_min`, `--T`, `--J`, `--seed`.

----------

### 2. Generate overlay figures (baseline vs heterogeneity)

Default usage:

`python code/bornholdt_heterogeneity/plot_overlap_figures.py` 

By default, this reads:

`data/lattice_data_results_100000.csv
data/heterogeneity_data_results_100000.csv` 

and writes overlay figures to the `results/` folder.

To specify different files:

`python code/bornholdt_heterogeneity/plot_overlap_figures.py \
  --baseline data/lattice_data_results_50000.csv \
  --hetero data/heterogeneity_data_results_50000.csv` 

----------

### 3. Interactive lattice visualization (optional)

`python code/bornholdt_heterogeneity/visualize_lattice_heterogeneity.py` 

Optional parameter overrides:

`python code/bornholdt_heterogeneity/visualize_lattice_heterogeneity.py \
  --alpha-mean 8 --alpha-std 2 --T 1.2` 

----------

### Notes

-   Agent heterogeneity is **fixed over time** and assigned at initialization.
    
-   The same observables and figures as in the baseline model are used, enabling direct comparison.
    
-   Recommended parameter ranges are chosen to balance qualitative behavior and runtime.
  