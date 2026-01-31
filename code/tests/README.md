# Tests

This folder contains pytest-based sanity tests for the main components of the Bornholdt spin-market project.

The tests are intentionally lightweight and fast: they verify model correctness, invariants, and basic reproducibility without running long simulations.

# How to Run the Tests (IMPORTANT)
You must run pytest from inside the code/ directory, and you must use Python’s module invocation.
Correct procedure:From the repository root:
cd code
python -m pytest .

# Test Files Overview
 - test_bornholdt2d.py
Sanity tests for the baseline 2D lattice model (Bornholdt2D):

decision (S) and strategy (C) spins have correct shape and values (±1)

magnetization M always lies in [-1, 1]

heat-bath probability behaves correctly at extreme fields

cached counters remain consistent after a sweep

These tests ensure the core lattice dynamics are implemented correctly.

- test_heterogeneity.py

Sanity tests for the heterogeneous-α lattice model (Bornholdt2D_heterogeneity):

quenched α-field has correct shape and is clipped to alpha_min

cached counters remain consistent after a sweep

magnetization stays in [-1, 1]

chartist fraction stays in [0, 1]

These tests verify that introducing heterogeneity does not break core invariants.

- test_network.py 

Sanity tests for the network-based Bornholdt model:

decision and strategy spins have correct length and values

cached counters stay consistent after a sweep

magnetization and chartist fraction remain within valid bounds

heat-bath probability saturates correctly