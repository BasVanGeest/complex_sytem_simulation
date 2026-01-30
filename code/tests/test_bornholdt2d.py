import numpy as np
import pytest

# ✅ 如果你的文件名不是 bornholdt_model.py，请改这一行
from bornholdt_2d.bornholdt_model import Bornholdt2D


def test_init_shapes():
    """S and C should be LxL arrays."""
    L = 10
    m = Bornholdt2D(L=L, seed=0)
    assert m.S.shape == (L, L)
    assert m.C.shape == (L, L)


def test_init_values_are_pm_one():
    """S and C entries must be in {-1, +1}."""
    m = Bornholdt2D(L=8, seed=1)
    assert set(np.unique(m.S)).issubset({-1, +1})
    assert set(np.unique(m.C)).issubset({-1, +1})


def test_magnetization_range():
    """M(t) must be within [-1, 1]."""
    m = Bornholdt2D(L=12, seed=2)
    M = m.M()
    assert -1.0 <= M <= 1.0


def test_p_up_extremes():
    """_p_up should saturate to 0 or 1 for extreme x=2*beta*h."""
    # beta > 0
    beta = 1.0
    # very large positive h -> prob 1
    assert Bornholdt2D._p_up(beta, h=1000.0) == 1.0
    # very large negative h -> prob 0
    assert Bornholdt2D._p_up(beta, h=-1000.0) == 0.0
    # h=0 -> prob 0.5 (within float tolerance)
    assert abs(Bornholdt2D._p_up(beta, h=0.0) - 0.5) < 1e-12


def test_sweep_preserves_invariants_counts():
    """
    After a sweep, cached counters should match recomputed values.
    (invariants / consistency check)
    """
    m = Bornholdt2D(L=15, seed=3)
    m.sweep()
    # recompute from arrays (O(N)) only in tests
    assert m._sumS == int(m.S.sum())
    assert m._sumC == int(m.C.sum())
    assert m._n_chartist == int((m.C == -1).sum())


def test_sweep_changes_state_typically():
    """With non-trivial parameters, the state should usually change after some sweeps."""
    m = Bornholdt2D(L=20, seed=4, J=1.0, alpha=4.0, T=1.5)
    S_before = m.S.copy()
    # do a few sweeps to reduce fluke of no-change
    for _ in range(3):
        m.sweep()
    assert not np.array_equal(m.S, S_before)


def test_simulate_output_keys_and_lengths():
    """simulate should return expected keys and consistent lengths."""
    m = Bornholdt2D(L=10, seed=5)
    out = m.simulate(n_sweeps=50, burn_in=10, thin=2)
    # keys
    for k in ["M", "r", "abs_r", "frac_chartist", "frac_fundamentalist"]:
        assert k in out

    M = out["M"]
    r = out["r"]
    abs_r = out["abs_r"]
    fc = out["frac_chartist"]
    ff = out["frac_fundamentalist"]

    # length relations
    assert len(M) == len(fc) == len(ff)
    assert len(abs_r) == len(r)
    assert len(r) == max(0, len(M) - 1)

    # fractions in [0,1] and sum to 1
    assert np.all((0.0 <= fc) & (fc <= 1.0))
    assert np.allclose(fc + ff, 1.0)


def test_reproducibility_with_seed():
    """
    Same seed + same params -> identical initial states and identical short simulation output.
    (reproducibility)
    """
    params = dict(L=12, J=1.0, alpha=4.0, T=1.5, seed=123)
    m1 = Bornholdt2D(**params)
    m2 = Bornholdt2D(**params)

    # initial arrays should match
    assert np.array_equal(m1.S, m2.S)
    assert np.array_equal(m1.C, m2.C)

    # short run should match (since RNG streams start identical)
    out1 = m1.simulate(n_sweeps=30, burn_in=0, thin=1)
    out2 = m2.simulate(n_sweeps=30, burn_in=0, thin=1)
    assert np.allclose(out1["M"], out2["M"])
    assert np.allclose(out1["frac_chartist"], out2["frac_chartist"])
    assert np.allclose(out1["r"], out2["r"])
