import numpy as np
import networkx as nx


class BornholdtNetwork:
    """
    Bornholdt Network spin market model with two spins per agent, updated *fully asynchronously*
    (matching the lattice async Bornholdt2D logic):

    - Graph topology defines neighbors
    - Decision spins S_i ∈ {+1,-1}
    - Strategy spins C_i ∈ {+1,-1}

    Local field:
        h_i = J * sum_{j in neigh(i)} S_j - alpha * C_i * M
    where M = (1/N) sum_i S_i

    Asynchronous update (per node, random serial):
      1) compute M_before from current sumS
      2) update S_i by heat-bath using M_before and current C_i
      3) immediately update C_i using updated S_i and updated sumS:
            flip C_i if S_i * C_i * sumS < 0   (equiv to S_i * C_i * M < 0)

    NOTE:
    - `time_step()` performs ONE "sweep": N random-serial single-node updates,
      but each update includes BOTH S and C updates immediately (async).
    - Method names are aligned with your baseline runner/plotter expectations.
    """

    def __init__(self, G, J: float = 1.0, alpha: float = 4.0, T: float = 1.5, seed: int | None = None):
        self.G = G
        self.N = int(nx.number_of_nodes(G))

        nodes = list(self.G.nodes())
        if nodes and (min(nodes) != 0 or max(nodes) != self.N - 1):
            raise ValueError(
                "BornholdtNetwork assumes nodes are labeled 0..N-1. "
                "Use nx.convert_node_labels_to_integers(G) if needed."
            )

        self.J = float(J)
        self.alpha = float(alpha)
        self.T = float(T)
        if self.T <= 0:
            raise ValueError("Temperature T must be > 0.")
        self.beta = 1.0 / self.T

        self.rng = np.random.default_rng(seed)

        # spins
        self.S = self.rng.choice([-1, +1], size=self.N)
        self.C = self.rng.choice([-1, +1], size=self.N)

        # maintain sumS for O(1) magnetization
        self._sumS = int(self.S.sum())

        # (Optional bookkeeping; not required by your runner but useful)
        self._n_chartist = int((self.C == -1).sum())

        # precompute neighbor arrays for speed
        self.neigh = [np.fromiter(self.G.neighbors(i), dtype=int) for i in range(self.N)]

    # ---------------------------------------------------------------------
    # Baseline-compatible names (so your runner/plotter don't change)
    # ---------------------------------------------------------------------
    def magnetization_M(self) -> float:
        return self._sumS / self.N

    def mean_strategy_C(self) -> float:
        return float(self.C.mean())

    # ---------------------------------------------------------------------
    # Heat-bath helper
    # ---------------------------------------------------------------------
    @staticmethod
    def _heat_bath_prob_up(beta: float, h: float) -> float:
        x = 2.0 * beta * h
        if x >= 50.0:
            return 1.0
        if x <= -50.0:
            return 0.0
        return 1.0 / (1.0 + np.exp(-x))

    def _neighbors_sum_S(self, node: int) -> int:
        nbrs = self.neigh[node]
        if nbrs.size == 0:
            return 0
        return int(self.S[nbrs].sum())

    def _local_field(self, node: int, M_now: float) -> float:
        return self.J * self._neighbors_sum_S(node) - self.alpha * self.C[node] * M_now

    # ---------------------------------------------------------------------
    # Fully asynchronous single-node update (S then C immediately)
    # ---------------------------------------------------------------------
    def _update_node(self, node: int) -> None:
        # magnetization BEFORE updating this node
        M_before = self._sumS / self.N

        # 1) update S_node via heat-bath
        S_old = int(self.S[node])
        h = self._local_field(node, M_before)
        p_up = self._heat_bath_prob_up(self.beta, h)
        S_new = +1 if (self.rng.random() < p_up) else -1

        if S_new != S_old:
            self.S[node] = S_new
            self._sumS += (S_new - S_old)

        # 2) immediately update C_node using UPDATED S and UPDATED sumS
        C_old = int(self.C[node])
        if (int(self.S[node]) * C_old * self._sumS) < 0:
            C_new = -C_old
            self.C[node] = C_new

            # optional bookkeeping
            if C_old == -1:
                self._n_chartist -= 1
            else:
                self._n_chartist += 1

    # ---------------------------------------------------------------------
    # One "sweep" / one time step in the runner sense
    # ---------------------------------------------------------------------
    def monte_carlo_sweep(self) -> None:
        """One sweep = N random-serial asynchronous node updates (each updates S then C)."""
        order = self.rng.permutation(self.N)
        for node in order:
            self._update_node(int(node))

    def time_step(self) -> None:
        """Runner expects time_step(); here it's one full sweep."""
        self.monte_carlo_sweep()

    # ---------------------------------------------------------------------
    # Returns helper (same as lattice version)
    # ---------------------------------------------------------------------
    @staticmethod
    def returns_from_magnetization(M_series: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        M_series = np.asarray(M_series, dtype=float)
        if M_series.size < 2:
            return np.array([], dtype=float)
        P = np.abs(M_series) + float(eps)
        return np.log(P[1:]) - np.log(P[:-1])




