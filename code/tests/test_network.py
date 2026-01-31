import numpy as np
import networkx as nx

from bornholdt_network.model_network import BornholdtNetwork


def test_init_shapes_and_spin_values():
    """S and C have length N and contain only ±1."""
    G = nx.path_graph(25)  # nodes 0..24
    m = BornholdtNetwork(G, seed=0)

    assert m.S.shape == (m.N,)
    assert m.C.shape == (m.N,)
    assert m.N == 25
    assert set(np.unique(m.S)).issubset({-1, +1})
    assert set(np.unique(m.C)).issubset({-1, +1})


def test_heat_bath_prob_extremes():
    """Heat-bath probability saturates for extreme fields and is 0.5 at h=0."""
    beta = 1.0
    assert BornholdtNetwork._heat_bath_prob_up(beta, 1000.0) == 1.0
    assert BornholdtNetwork._heat_bath_prob_up(beta, -1000.0) == 0.0
    assert abs(BornholdtNetwork._heat_bath_prob_up(beta, 0.0) - 0.5) < 1e-12


def test_sweep_keeps_caches_consistent_and_bounds_ok():
    """After a sweep, cached counters match arrays; M in [-1,1]; chartist frac in [0,1]."""
    G = nx.erdos_renyi_graph(64, 0.1, seed=1)  # nodes 0..63
    m = BornholdtNetwork(G, seed=1)

    m.sweep()

    assert m._sumS == int(m.S.sum())
    assert m._n_chartist == int((m.C == -1).sum())

    M = m.M()
    fc = m.frac_chartist()
    assert -1.0 <= M <= 1.0
    assert 0.0 <= fc <= 1.0