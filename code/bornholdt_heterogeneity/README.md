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

Generate heterogeneity data: intermediate_heterogeneity.csv 
Key options: --L (size), --steps (sweeps), --burn_in, --thin, --alpha_mean/std/min, --seed.
CSV columns: t, M, r, abs_r, C_mean, chartist_frac, fundamentalist_frac.
Ensure baseline CSV intermediate.csv exists (comes from baseline model in code/bornholdt_2d).
Make overlay figures: plot_overlap_figures.py
Outputs: overlap_fig2_returns_timeseries.png, overlap_fig3_ccdf_abs_returns.png, overlap_fig4_volatility_acf.png, heterogeneity_alpha_hist.png.
Optional live visualization: visualize_latt_heterogeneity.py.
