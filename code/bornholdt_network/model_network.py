import numpy as np
import networkx as nx


class BornholdtNetwork:
    """
    Bornholdt Network spin market model with two spins per agent, updated asynchronously, matching the Bornholdt's paper logic.
    """

    def __init__(self, G, J: float = 1.0, alpha: float = 4.0, T: float = 1.5, seed: int | None = None):
        """Initialize the Bornholdt spin-market model on a NetworkX graph."""
        self.G = G
        self.N = int(nx.number_of_nodes(G))

        assert self.N > 0, "Graph must contain at least one node"            # ASSERT

        nodes = list(self.G.nodes())
        if nodes:
            assert min(nodes) == 0 and max(nodes) == self.N - 1, (           # ASSERT
                "Nodes must be labeled 0..N-1. "
                "Use nx.convert_node_labels_to_integers(G) if needed."
            )

        self.J = float(J)
        self.alpha = float(alpha)
        self.T = float(T)

        assert self.alpha > 0, "alpha must be positive"                      # ASSERT
        assert self.T > 0, "Temperature T must be > 0"                       # ASSERT

        self.beta = 1.0 / self.T
        self.rng = np.random.default_rng(seed)

        # spins
        self.S = self.rng.choice([-1, +1], size=self.N)
        self.C = self.rng.choice([-1, +1], size=self.N)

        # ASSERT: spin values valid
        assert set(np.unique(self.S)).issubset({-1, +1})                     # ASSERT
        assert set(np.unique(self.C)).issubset({-1, +1})                     # ASSERT

        # cached counters
        self._sumS = int(self.S.sum())
        self._n_chartist = int((self.C == -1).sum())

        # ASSERT: cached counters consistent
        assert self._sumS == int(self.S.sum())                               # ASSERT
        assert self._n_chartist == int((self.C == -1).sum())                 # ASSERT

        # precompute neighbor arrays for speed
        self.neigh = [np.fromiter(self.G.neighbors(i), dtype=int) for i in range(self.N)]

    # Observables (baseline-compatible)
    def magnetization_M(self) -> float:
        """Return the current magnetization (mean of decision spins S)."""
        M = self._sumS / self.N
        assert -1.0 <= M <= 1.0                                              # ASSERT
        return M

    def mean_strategy_C(self) -> float:
        """Return the mean strategy spin C."""
        return float(self.C.mean())

    # Lattice-style convenience
    def M(self) -> float:
        """Alias for magnetization."""
        return self.magnetization_M()

    def frac_chartist(self) -> float:
        """Fraction of chartists (C == -1)."""
        f = self._n_chartist / self.N
        assert 0.0 <= f <= 1.0                                               # ASSERT
        return f

    def sweep(self) -> None:
        """Alias for one sweep."""
        self.monte_carlo_sweep()

    # Heat-bath helper
    @staticmethod
    def _heat_bath_prob_up(beta: float, h: float) -> float:
        """Return heat-bath probability of setting S=+1."""
        x = 2.0 * beta * h
        if x >= 50.0:
            return 1.0
        if x <= -50.0:
            return 0.0
        p = 1.0 / (1.0 + np.exp(-x))
        assert 0.0 <= p <= 1.0                                               # ASSERT
        return p

    def _neighbors_sum_S(self, node: int) -> int:
        """Return the sum of decision spins over neighbors of a node."""
        nbrs = self.neigh[node]
        if nbrs.size == 0:
            return 0
        return int(self.S[nbrs].sum())

    def _local_field(self, node: int, M_now: float) -> float:
        """Compute local field at a node."""
        return self.J * self._neighbors_sum_S(node) - self.alpha * self.C[node] * M_now

    # Fully asynchronous single-node update
    def _update_node(self, node: int) -> None:
        """Update S then C for a single node (async)."""
        M_before = self._sumS / self.N

        # update S
        S_old = int(self.S[node])
        h = self._local_field(node, M_before)
        p_up = self._heat_bath_prob_up(self.beta, h)
        S_new = +1 if (self.rng.random() < p_up) else -1

        if S_new != S_old:
            self.S[node] = S_new
            self._sumS += (S_new - S_old)

        # update C
        C_old = int(self.C[node])
        if (int(self.S[node]) * C_old * self._sumS) < 0:
            self.C[node] = -C_old
            if C_old == -1:
                self._n_chartist -= 1
            else:
                self._n_chartist += 1

        # ASSERT: cached counters remain consistent
        assert self._sumS == int(self.S.sum())                               # ASSERT
        assert self._n_chartist == int((self.C == -1).sum())                 # ASSERT

    # One sweep / time step
    def monte_carlo_sweep(self) -> None:
        """One sweep = N random-serial node updates."""
        order = self.rng.permutation(self.N)
        for node in order:
            self._update_node(int(node))

    def time_step(self) -> None:
        """Runner-compatible time step."""
        self.monte_carlo_sweep()

    # Returns helper
    @staticmethod
    def returns_from_magnetization(M_series: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Compute log returns from a magnetization series."""
        assert eps > 0                                                       # ASSERT
        M_series = np.asarray(M_series, dtype=float)
        if M_series.size < 2:
            return np.array([], dtype=float)
        P = np.abs(M_series) + eps
        return np.log(P[1:]) - np.log(P[:-1])



