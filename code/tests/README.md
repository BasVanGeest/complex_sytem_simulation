# Tests

This repo includes two pytest files for sanity checks. 

## File Overview

- **`test_bornholdt2d.py`**  
  Sanity checks for the baseline 2D lattice model (`Bornholdt2D`). It verifies:
  - spin array shapes and allowed values
  - magnetization bounds (within `[-1, 1]`)
  - heat-bath probability behavior at extreme inputs
  - invariant/cached counters remain consistent after sweeps
  - typical state changes after a few sweeps (i.e., system actually evolves)
  - `simulate()` output length and key consistency
  - reproducibility when seeds match (RNG determinism)

- **`test_heterogeneity.py`**  
  Sanity checks for the heterogeneous-\(\alpha\) lattice model (`Bornholdt2D_heterogeneity`). It verifies:
  - \(\alpha\) field shape and clipping constraints
  - spin shapes and allowed values
  - cached counter consistency after a sweep
  - `simulate()` output lengths and value ranges:
    - fractions in `[0, 1]`
    - magnetization in `[-1, 1]`

