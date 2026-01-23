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

        #Maintain a glovbal sum of S values for efficiency
        self.sumS = int(np.sum(self.S)) 

        # Strategy spins (chartist/fundamentalist)
        self.C = self.rng.choice([-1,+1], size=(L,L))

    def magnetization(self):
        return self.sumS / self.N
    
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
        # Using the difference between new and old to update sumS efficiently
        p_up = 1.0 / (1.0 + np.exp(-2.0 * self.beta * h))
        old = self.S[i, j]
        new = 1 if self.rng.random() < p_up else -1
        if new != old:
            self.S[i, j] = new
            self.sumS += int(new - old)

        # Strategy switching: flip C if C*S*M < 0
        if self.C[i, j] * self.S[i, j] * M < 0:
            self.C[i, j] *= -1
    
    def sweep(self):
        """One Monte Carlo sweep = N random-site updates."""
        for _ in range(self.N):
            self.update_one_site()
    

    def price_proxy(self, M: float, eps: float = 1e-6) -> float:
            """
            Price proxy used in the paper-style definition:
                P(t) ~ |M(t)|
            We use |M| + eps to avoid log(0).
            """
            return abs(M) + eps
    
    @staticmethod
    def compute_returns_from_M(M_series: np.ndarray, mode: str = "log_absM", eps: float = 1e-6) -> np.ndarray:
        """
        Compute return series from magnetization series.

        mode:
          - "log_absM": r(t) = log(|M(t)|+eps) - log(|M(t-1)|+eps)
          - "diffM":    r(t) = M(t) - M(t-1)   (more robust, no log)
        """
        M_series = np.asarray(M_series, dtype=float)
        if M_series.size < 2:
            return np.array([], dtype=float)

        if mode == "log_absM":
            P = np.abs(M_series) + eps
            return np.log(P[1:]) - np.log(P[:-1])
        elif mode == "diffM":
            return M_series[1:] - M_series[:-1]
        else:
            raise ValueError(f"Unknown return mode: {mode}")
    
    def run(
        self,
        n_sweeps: int,
        burn_in: int = 0,
        thin: int = 1,
        return_mode: str = "log_absM",
        eps: float = 1e-6,
    ):
        """
        Run simulation and collect time series.

        Parameters
        ----------
        n_sweeps : total sweeps to execute (including burn-in)
        burn_in  : number of initial sweeps to discard
        thin     : keep one sample every 'thin' sweeps after burn-in
        return_mode : "log_absM" (paper-like) or "diffM"
        eps      : small constant to avoid log(0)

        Returns
        -------
        dict with:
          - "M": magnetization samples
          - "r": returns aligned with M (length len(M)-1)
          - "abs_r": |returns|
        """
        if n_sweeps <= 0:
            raise ValueError("n_sweeps must be positive")
        if burn_in < 0 or burn_in >= n_sweeps:
            raise ValueError("burn_in must satisfy 0 <= burn_in < n_sweeps")
        if thin <= 0:
            raise ValueError("thin must be positive")

        Ms = []

        for s in range(n_sweeps):
            self.sweep()

            if s < burn_in:
                continue

            if ((s - burn_in) % thin) == 0:
                Ms.append(self.magnetization())

        M_series = np.array(Ms, dtype=float)
        r_series = self.compute_returns_from_M(M_series, mode=return_mode, eps=eps)
        abs_r = np.abs(r_series)

        return {"M": M_series, "r": r_series, "abs_r": abs_r}