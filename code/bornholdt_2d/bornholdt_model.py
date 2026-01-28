import numpy as np


class Bornholdt2D:
    """
    Bornholdt (2001) 2D market spin model with *asynchronous* updates:

    - LxL periodic lattice
    - decision spin S_ij ∈ {+1,-1}
    - strategy spin C_ij ∈ {+1,-1}  (+1=fundamentalist, -1=chartist)

    Per Bornholdt description:
      - random serial, asynchronous heat-bath updates of single sites
      - for every updated S_ij, update C_ij *subsequently* (immediately)

    Local field (general version eq.(2) in Bornholdt text):
      h_ij = J * sum_nn(S) - alpha * C_ij * M
    where M = (1/N) sum(S)

    Strategy update (Bornholdt eq.(3) logic, site-local, applied immediately):
      flip C_ij if  alpha * S_ij * C_ij * sum(S) < 0
    (equivalently: S_ij * C_ij * M < 0 since alpha>0, N>0)
    """

    def __init__(self, L, J=1.0, alpha=4.0, T=1.5, seed=None,
                 init_S="random", init_C="random"):
        self.L = int(L)
        self.N = self.L * self.L
        self.J = float(J)
        self.alpha = float(alpha)
        self.T = float(T)
        if self.T <= 0:
            raise ValueError("T must be > 0.")
        self.beta = 1.0 / self.T
        self.rng = np.random.default_rng(seed)

        # spins S
        if init_S == "random":
            self.S = self.rng.choice([-1, +1], size=(self.L, self.L))
        elif init_S == "all_up":
            self.S = np.ones((self.L, self.L), dtype=int)
        elif init_S == "all_down":
            self.S = -np.ones((self.L, self.L), dtype=int)
        else:
            raise ValueError(f"Unknown init_S={init_S!r}")

        # strategy spins C
        if init_C == "random":
            self.C = self.rng.choice([-1, +1], size=(self.L, self.L))
        elif init_C == "all_fundamentalist":
            self.C = np.ones((self.L, self.L), dtype=int)   # +1
        elif init_C == "all_chartist":
            self.C = -np.ones((self.L, self.L), dtype=int)  # -1
        else:
            raise ValueError(f"Unknown init_C={init_C!r}")

        # maintain sums for O(1) M and strategy ratios
        self._sumS = int(self.S.sum())
        self._sumC = int(self.C.sum())
        self._n_chartist = int((self.C == -1).sum())  # explicit ratio tracking

    # ---- observables ----
    def M(self) -> float:
        return self._sumS / self.N

    def frac_chartist(self) -> float:
        """Fraction of chartists = P(C=-1)."""
        return self._n_chartist / self.N

    def frac_fundamentalist(self) -> float:
        """Fraction of fundamentalists = P(C=+1)."""
        return 1.0 - self.frac_chartist()

    # ---- lattice helpers ----
    def _nn_sumS(self, i, j) -> int:
        L = self.L
        S = self.S
        return S[(i + 1) % L, j] + S[(i - 1) % L, j] + S[i, (j + 1) % L] + S[i, (j - 1) % L]

    def _h(self, i, j, M_now) -> float:
        return self.J * self._nn_sumS(i, j) - self.alpha * self.C[i, j] * M_now

    @staticmethod
    def _p_up(beta, h) -> float:
        x = 2.0 * beta * h
        if x >= 50.0:
            return 1.0
        if x <= -50.0:
            return 0.0
        return 1.0 / (1.0 + np.exp(-x))

    # ---- single-site asynchronous update (Bornholdt-style) ----
    def _update_site(self, i, j) -> None:
        # use instantaneous magnetization BEFORE updating this site
        M_before = self._sumS / self.N

        # 1) heat-bath update S_ij
        Sold = int(self.S[i, j])
        h = self._h(i, j, M_before)
        Snew = +1 if (self.rng.random() < self._p_up(self.beta, h)) else -1
        if Snew != Sold:
            self.S[i, j] = Snew
            self._sumS += (Snew - Sold)

        # 2) subsequently update C_ij immediately (as described)
        # Bornholdt eq.(3) condition: alpha * S_i * C_i * sum(S) < 0
        # Use UPDATED S_ij and UPDATED sum(S) (asynchronous).
        Ci_old = int(self.C[i, j])
        if (self.S[i, j] * Ci_old * self._sumS) < 0:
            # flip C
            Ci_new = -Ci_old
            self.C[i, j] = Ci_new

            # maintain counters
            self._sumC += (Ci_new - Ci_old)
            if Ci_old == -1:
                self._n_chartist -= 1
            else:
                self._n_chartist += 1

    # ---- one Monte Carlo sweep (N random-serial site updates) ----
    def sweep(self) -> None:
        order = self.rng.permutation(self.N)
        L = self.L
        for k in order:
            i, j = divmod(int(k), L)
            self._update_site(i, j)

    # ---- simulation driver ----
    def simulate(self, n_sweeps, burn_in=0, thin=1, eps=1e-6):
        """
        Runs n_sweeps sweeps.
        Records:
          - M(t)
          - frac_chartist(t) (explicit "ratio of strategy choices")
        """
        if n_sweeps <= 0:
            raise ValueError("n_sweeps must be > 0.")
        if burn_in < 0 or burn_in >= n_sweeps:
            raise ValueError("burn_in must satisfy 0 <= burn_in < n_sweeps.")
        if thin <= 0:
            raise ValueError("thin must be > 0.")

        M_series = []
        chartist_series = []

        for t in range(n_sweeps):
            self.sweep()

            if t < burn_in:
                continue
            if ((t - burn_in) % thin) != 0:
                continue

            M_series.append(self.M())
            chartist_series.append(self.frac_chartist())

        M_series = np.asarray(M_series, dtype=float)
        chartist_series = np.asarray(chartist_series, dtype=float)

        # Bornholdt paper plots often use log-return of |M|
        P = np.abs(M_series) + float(eps)
        r = np.log(P[1:]) - np.log(P[:-1]) if len(P) >= 2 else np.array([], dtype=float)

        return {
            "M": M_series,
            "r": r,
            "abs_r": np.abs(r),
            "frac_chartist": chartist_series,
            "frac_fundamentalist": 1.0 - chartist_series,
        }
