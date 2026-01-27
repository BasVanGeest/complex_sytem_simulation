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
    - Heat-bath update for S
    - Strategy switching rule for C: flip if C*S*M < 0
    """

    def __init__(self, G, J: float= 1.0, alpha: float = 4.0, T: float = 1.5, seed: int | None = None):
        self.G = G
        self.N = nx.number_of_nodes(G)
        self.J = J
        self.alpha = alpha
        self.T = T
        self.beta = 1.0 / T
        self.rng = np.random.default_rng(seed)

        # Decision spins (buy/sell)
        self.S = self.rng.choice([-1, +1], size=self.N)

        # Strategy spins (chartist/fundamentalist)
        self.C = self.rng.choice([-1, +1], size=self.N)

    def magnetization(self):
        return float(np.mean(self.S))
    
    def _neighbors_sum_S(self, n):
        """Sum of spins S of all neighbors of node n."""
        return sum(self.S[neighbor] for neighbor in self.G.neighbors(n))
    
    def _local_field(self, node: int, M: float):
        """Compute local field for a single node on a network."""
        return self.J * self._neighbors_sum_S(node) - self.alpha * self.C[node] * M


    def update_one_site(self):
        """One random-site update on a network."""
        node = self.rng.integers(0, self.N) 

        M = self.magnetization()
        h = self._local_field(node, M)

        # Heat-bath: P(S=+1) = 1/(1+exp(-2*beta*h))
        p_up = 1.0 / (1.0 + np.exp(-2.0 * self.beta * h))
        self.S[node] = 1 if self.rng.random() < p_up else -1

        # Strategy switching: flip C if C*S*M < 0
        if self.C[node] * self.S[node] * M < 0:
            self.C[node] *= -1


    def sweep(self):
        """One Monte Carlo sweep = N random-site updates."""
        for _ in range(self.N):
            self.update_one_site()


# simple test
G = nx.barabasi_albert_graph(n=10, m=2, seed=42)
model = BornholdtNetwork(G, J=1.0, alpha=4.0, T=1.5, seed=42)
for t in range(5):
    model.sweep()
    print(f"Sweep {t+1}, magnetization: {model.magnetization():.3f}")


