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
    
    def update_one_site(self): #UPDATED VERSION
        #THERE WAS A DISCREPANCY BETWEEN THIS MODEL AND MODEL EXPLAIN IN THE BOOK, IT USED OLD MAGNETIZATION. 
        #IF YOU WANT YOU CAN COMPARE THIS NEW MODEL WITH OLD MODEL; IN THE FOLDER "bornholdt"
        """
        One random-site update (Option A):
        1) compute current magnetization M_old
        2) update S_ij with heat-bath using M_old
        3) update magnetization to M_new (incremental, exact)
        4) update C_ij using the new S_ij and new M_new
        """
        i = self.rng.integers(0, self.L)
        j = self.rng.integers(0, self.L)

        # 1) old magnetization
        M_old = self.magnetization()

        # Save old spin before update (needed for incremental M update)
        S_old = self.S[i, j]

        # 2) update S using local field computed with M_old
        h = self._local_field(i, j, M_old)
        p_up = 1.0 / (1.0 + np.exp(-2.0 * self.beta * h))
        S_new = 1 if self.rng.random() < p_up else -1
        self.S[i, j] = S_new

        # 3) compute new magnetization exactly, without recomputing mean(S)
        # M = (1/N) sum S, so flipping one spin changes M by (S_new - S_old)/N
        M_new = M_old + (S_new - S_old) / self.N

        # 4) strategy switching using NEW S and NEW M
        if self.C[i, j] * self.S[i, j] * M_new < 0:
            self.C[i, j] *= -1

    
    def sweep(self):
        """One Monte Carlo sweep = N random-site updates."""
        for _ in range(self.N):
            self.update_one_site()
    


