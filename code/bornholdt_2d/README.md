## Bornholdt Baseline 2D Spin-Market Model

This folder contains an implementation of the **baseline Bornholdt spin-market model** on a two-dimensional L×L periodic lattice. Agents are updated asynchronously using single-site heat-bath dynamics. The code is designed to reproduce the qualitative figures reported in the original Bornholdt and Yamano papers.

Simulation scripts are provided to generate time series data and paper-style figures.

> **Runtime note:** To keep execution time below ~30 minutes on a standard laptop, we recommend running no more than **200,000 Monte Carlo sweeps**.

----------

## Folder Contents

-   **`bornholdt_model.py`**  
    Core implementation of the 2D Bornholdt model.  
    Defines the main simulation logic, including:
    
    -   `simulate` and `sweep` routines
        
    -   observables such as magnetization `M`
        
    -   strategy composition (`frac_chartist`, etc.)
        
-   **`run_baseline.py`**  
    Command-line interface to run a single lattice simulation and write results to CSV.  
    Configurable parameters include:
    
    -   simulation control: `steps`, `burn_in`, `thin`, `seed`
        
    -   model parameters: `L`, `J`, `alpha`, `T`
        
    
    **Output columns:**
    
    `t, M, r, abs_r, C_mean, chartist_frac, fundamentalist_frac` 
    
-   **`plot_paper_figures.py`**  
    Reads a baseline CSV file and generates paper-style figures in `paper_figures/`:
    
    1.  Return time series
        
    2.  CCDF of |r|
        
    3.  Volatility autocorrelation (of |r|)
        
    4.  Chartist fraction versus volatility
        
-   **`visualize_lattice.py`**  
    Interactive Matplotlib visualization of the lattice.  
    Press **`t`** to toggle between:
    
    -   decision spins SSS
        
    -   strategy spins CCC
        
-   **`__init__.py`**  
    Allows the directory to be imported as a Python package.
    

----------

## Dependencies

Required Python packages:

-   `numpy`
    
-   `matplotlib`
    
-   `tqdm`
    
-   `networkx`
    

No external data files are required. All outputs are generated locally.

----------

## How to Run

Important:
The simulation should run for ≥ 20,000 steps.
The first 10,000 steps act as burn-in, and multiple plots are evaluated using data from steps 10,000 to 20,000. Runs shorter than this will not reproduce the intended results.

### 1. Run a baseline lattice simulation

From the repository root:

`python code/bornholdt_2d/run_baseline.py` 

This generates a CSV file (by default  
`lattice_data_results_50000.csv`) in the `data/` folder.

**Example with explicit parameters:**

`python code/bornholdt_2d/run_baseline.py \
  --steps 50000 \
  --burn_in 10000 \
  --thin 1 \
  --L 32 \
  --J 1.0 \
  --alpha 8.0 \
  --T 1.5 \
  --seed 0` 

----------

### 2. Generate paper-style figures

`python code/bornholdt_2d/plot_paper_figures.py \
  --data data/lattice_data_results_50000.csv` 

Figures are saved to the `results/` folder.

----------

### 3. Visualize the lattice dynamics (optional)

`python code/bornholdt_2d/visualize_lattice.py` 

Optional parameter overrides:

`python code/bornholdt_2d/visualize_lattice.py --L 64 --alpha 8 --T 1.2` 

----------

## Notes on Parameter Choices

-   Intermittent, market-like dynamics is typically observed for temperatures **below but close to the critical temperature**.
    
-   Larger lattice sizes and longer runs improve statistics but increase runtime.
    
-   The provided defaults are chosen to balance qualitative accuracy and computational cost.