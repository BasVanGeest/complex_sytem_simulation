import numpy as np

from bornholdt_2d.bornholdt_model import Bornholdt2D


def test_init_shapes_and_spin_values():
    """S and C are LxL and only contain ±1."""
    L = 10
    m = Bornholdt2D(L=L, seed=0)

    assert m.S.shape == (L, L)
    assert m.C.shape == (L, L)
    assert set(np.unique(m.S)).issubset({-1, +1})
    assert set(np.unique(m.C)).issubset({-1, +1})


def test_p_up_basic_behavior():
    """Heat-bath probability saturates for extreme fields and is 0.5 at h=0."""
    beta = 1.0
    assert Bornholdt2D._p_up(beta, 1000.0) == 1.0
    assert Bornholdt2D._p_up(beta, -1000.0) == 0.0
    assert abs(Bornholdt2D._p_up(beta, 0.0) - 0.5) < 1e-12


def test_sweep_keeps_caches_consistent_and_M_in_bounds():
    """After a sweep, cached counters match array sums; magnetization stays in [-1, 1]."""
    m = Bornholdt2D(L=12, seed=1)
    m.sweep()

    assert m._sumS == int(m.S.sum())
    assert m._sumC == int(m.C.sum())
    assert m._n_chartist == int((m.C == -1).sum())

    M = m.M()
    assert -1.0 <= M <= 1.0


