import numpy as np

class Bornholdt2D:
    """
    Bornholdt 2D spin market model with two spins per agent:

    S_ij in {+1,-1}: buy/sell decision
    C_ij in {+1,-1}: strategy (e.g., chartist/fundamentalist)

    Local field:
        h_ij = J * sum_nn(S) - alpha * C_ij * M
    where M = mean(S).

    Updates:
    - Heat-bath update for S
    - Strategy switching rule for C: flip if C*S*M < 0
    """

    def __init__(self, L: int, J: float= 1.0, alpha: float = 4.0, T: float = 1.5, seed: int | None = None):
        self.L = L
        self.N = L*L
        self.J = J
        self.alpha = alpha
        self.T = T
        self.beta = 1.0 / T
        self.rng = np.random.default_rng(seed)

        # Decision spins (buy/sell)
        self.S = self.rng.choice([-1,+1], size=(L,L))

        # Strategy spins (chartist/fundamentalist)
        self.C = self.rng.choice([-1,+1], size=(L,L))

    def magnetization(self):
        return float(np.mean(self.S))
    
    def _neighbors_sum_S(self, i: int, j: int):
        """Sum of 4 nearest neighbors of S with periodic boundaries."""
        L = self.L
        return(
            self.S[(i + 1) % L, j] +
            self.S[(i - 1) % L, j] +
            self.S[i, (j + 1) % L] +
            self.S[i, (j - 1) % L]
        )
    
    def _local_field(self, i: int, j: int, M: float):
        return self.J * self._neighbors_sum_S(i, j) - self.alpha * self.C[i, j] * M
    
    def update_one_site(self):
        """
        One random-site update:
        1) compute current magnetization M
        2) update S_ij with heat-bath probability using h_ij
        3) update C_ij with switching rule
        """
        i = self.rng.integers(0, self.L)
        j = self.rng.integers(0, self.L)

        M = self.magnetization()
        h = self._local_field(i, j, M)

        # Heat-bath: P(S=+1) = 1/(1+exp(-2*beta*h))
        p_up = 1.0 / (1.0 + np.exp(-2.0 * self.beta * h))
        self.S[i, j] = 1 if self.rng.random() < p_up else -1

        # Strategy switching: flip C if C*S*M < 0
        if self.C[i, j] * self.S[i, j] * M < 0:
            self.C[i, j] *= -1
    
    def sweep(self):
        """One Monte Carlo sweep = N random-site updates."""
        for _ in range(self.N):
            self.update_one_site()
    


