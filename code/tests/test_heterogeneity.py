import numpy as np
import pytest

from bornholdt_heterogeneity.bornholdt_heterogeneity import Bornholdt2D_heterogeneity


def test_alpha_field_shape_and_bounds():
    """alpha should be an LxL field and clipped to be >= alpha_min."""
    L = 12
    m = Bornholdt2D_heterogeneity(L=L, seed=0, alpha_mean=4.0, alpha_std=1.0, alpha_min=0.1)
    assert m.alpha.shape == (L, L)
    assert np.all(m.alpha >= 0.1)


def test_spins_shapes_and_values():
    """S and C should be LxL with values in {-1, +1}."""
    L = 10
    m = Bornholdt2D_heterogeneity(L=L, seed=1)
    assert m.S.shape == (L, L)
    assert m.C.shape == (L, L)
    assert set(np.unique(m.S)).issubset({-1, +1})
    assert set(np.unique(m.C)).issubset({-1, +1})


def test_sweep_preserves_caches():
    """Cached counters should remain consistent after a sweep."""
    m = Bornholdt2D_heterogeneity(L=15, seed=2)
    m.sweep()
    assert m._sumS == int(m.S.sum())
    assert m._sumC == int(m.C.sum())
    assert m._n_chartist == int((m.C == -1).sum())


def test_simulate_lengths_and_ranges():
    """simulate should return consistent lengths and valid ranges."""
    m = Bornholdt2D_heterogeneity(L=10, seed=3)
    out = m.simulate(n_sweeps=40, burn_in=10, thin=2)

    # keys
    for k in ["M", "r", "abs_r", "frac_chartist", "frac_fundamentalist"]:
        assert k in out

    M = out["M"]
    r = out["r"]
    abs_r = out["abs_r"]
    fc = out["frac_chartist"]
    ff = out["frac_fundamentalist"]

    assert len(M) == len(fc) == len(ff)
    assert len(abs_r) == len(r)
    assert len(r) == max(0, len(M) - 1)

    assert np.all((0.0 <= fc) & (fc <= 1.0))
    assert np.allclose(fc + ff, 1.0)
    assert np.all((-1.0 <= M) & (M <= 1.0))
