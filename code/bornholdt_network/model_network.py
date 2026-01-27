import numpy as np
import networkx as nx

class BornholdtNetwork:
    """
    Bornholdt Network spin market model with two spins per agent:

    S_ij in {+1,-1}: buy/sell decision
    C_ij in {+1,-1}: strategy (e.g., chartist/fundamentalist)

    Local field:
        h_ij = J * sum_nn(S) - alpha * C_ij * M
    where M = mean(S).

    Updates:
    - Keep the same pipeline as the lattice baseline:
    - Class re-writtent to be a drop-in sibling of Bornholdt2D so that the same
      run_baseline.py + plot_paper_figures.py can be reused for both lattice and networks.
    """

    def __init__(self, G, J: float= 1.0, alpha: float = 4.0, T: float = 1.5, seed: int | None = None):
        self.G = G
        self.N = int(nx.number_of_nodes(G))
        nodes = list(self.G.nodes())
        if nodes and (min(nodes) != 0 or max(nodes) != self.N - 1):
            raise ValueError(
                "BornholdtNetwork assumes nodes are labeled 0..N-1. "
                "Use nx.convert_node_labels_to_integers(G) if needed."
            )
        self.J = J
        self.alpha = alpha
        self.T = T
        self.beta = 1.0 / T
        self.rng = np.random.default_rng(seed)

        # Decision spins (buy/sell)
        self.S = self.rng.choice([-1, +1], size=self.N)

        # Strategy spins (chartist/fundamentalist)
        self.C = self.rng.choice([-1, +1], size=self.N)

        #Cache sum(S) so magnetization is O(1), needed for long runs (Fig 3 uses 10^6 sweeps)
        self._sumS = int(self.S.sum())

        #Precompute neighbor lists for speed
        self.neigh = [np.fromiter(self.G.neighbors(i), dtype=int) for i in range(self.N)]
  
    #Baseline-compatible (same names as Bornholdt2D)
    def magnetization_M(self) -> float:
        return self._sumS / self.N

    def mean_strategy_C(self) -> float:
        return float(self.C.mean())

    def magnetization(self) -> float:
        return self.magnetization_M()
    
    #Heat-bath micro-update
    @staticmethod
    def _heat_bath_prob_up(beta: float, h: float) -> float:
        """
        Matches Bornholdt2Ds helper
        """
        x = 2.0 * beta * h
        if x >= 50.0:
            return 1.0
        if x <= -50.0:
            return 0.0
        return 1.0 / (1.0 + np.exp(-x))

    def _neighbors_sum_S(self, node: int) -> float:
        nbrs = self.neigh[node]
        if nbrs.size == 0:
            return 0.0
        return float(self.S[nbrs].sum())

    def _local_field(self, node: int, M: float) -> float:
        """
        paper Eq.(2):
        """
        return self.J * self._neighbors_sum_S(node) - self.alpha * self.C[node] * M

    def update_decision_spin_S(self, node: int, M: float) -> None:
        """
        Rename + behavior to match lattice baseline naming.
        Update S_node by heat-bath, with C held fixed during sweep.
        """
        S_old = int(self.S[node])
        h = self._local_field(node, M)
        p_up = self._heat_bath_prob_up(self.beta, h)
        S_new = 1 if (self.rng.random() < p_up) else -1

        if S_new != S_old:
            self.S[node] = S_new
            self._sumS += (S_new - S_old)

    def update_strategy_spins_C_synchronously(self, M: float) -> None:
        """
        Strategy switching rule (paper Eq.(3)):
          if C_i * S_i * M < 0 then flip C_i
        Apply this synchronously once per sweep to match the baseline time-step logic.
        """
        flip = (self.C * self.S * M) < 0
        self.C[flip] *= -1


    #time step = one sweep + synchronous C update
    def monte_carlo_sweep(self) -> None:
        """
        One Monte Carlo sweep = N random-serial single-node S updates.
        C is held fixed during the sweep.
        """
        order = self.rng.permutation(self.N)
        for node in order:
            M_inst = self.magnetization_M()
            self.update_decision_spin_S(int(node), M_inst)

    def time_step(self) -> None:
        """
         Matches the same t -> t+1 semantics used in Bornholdt2D baseline:
            1) sweep update of S with C fixed
            2) compute M(t+1)
            3) synchronous update of all C using M(t+1)
        """
        self.monte_carlo_sweep()
        M_new = self.magnetization_M()
        self.update_strategy_spins_C_synchronously(M_new)

    # Returns (same helper as Bornholdt2D)
    @staticmethod
    def returns_from_magnetization(M_series: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        M_series = np.asarray(M_series, dtype=float)
        if M_series.size < 2:
            return np.array([], dtype=float)
        P = np.abs(M_series) + float(eps)
        return np.log(P[1:]) - np.log(P[:-1])



