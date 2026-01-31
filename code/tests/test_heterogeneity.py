import numpy as np

from bornholdt_heterogeneity.bornholdt_heterogeneity import Bornholdt2D_heterogeneity


def test_alpha_shape_and_min_clip():
    """alpha is LxL and clipped to be >= alpha_min."""
    L = 10
    m = Bornholdt2D_heterogeneity(L=L, seed=0, alpha_mean=4.0, alpha_std=1.0, alpha_min=0.1)
    assert m.alpha.shape == (L, L)
    assert np.all(m.alpha >= 0.1)


def test_sweep_keeps_caches_consistent_and_bounds_ok():
    """After a sweep, caches match arrays; M in [-1,1]; chartist fraction in [0,1]."""
    m = Bornholdt2D_heterogeneity(L=12, seed=1)
    m.sweep()

    assert m._sumS == int(m.S.sum())
    assert m._n_chartist == int((m.C == -1).sum())

    M = m.M()
    fc = m.frac_chartist()
    assert -1.0 <= M <= 1.0
    assert 0.0 <= fc <= 1.0
