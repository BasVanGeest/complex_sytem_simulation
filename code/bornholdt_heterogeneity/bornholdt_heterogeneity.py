import numpy as np


class Bornholdt2D_heterogeneity:
    """
    Bornholdt (2001) 2D market spin model with asynchronous updates,
    extended with heterogeneous alpha_ij drawn once at init:
        alpha_ij ~ Normal(alpha_mean, alpha_std^2), clipped to alpha_min > 0
    """

    def __init__(
        self,
        L: int,
        J: float = 1.0,
        alpha_mean: float = 8.0,
        alpha_std: float = 2.0,
        alpha_min: float = 1e-6,
        T: float = 1.5,
        seed=None,
        init_S: str = "random",
        init_C: str = "random",
    ):
        self.L = int(L)
        assert self.L > 0, "Lattice size L must be positive"                      # ASSERT

        self.N = self.L * self.L
        self.J = float(J)

        self.alpha_mean = float(alpha_mean)
        self.alpha_std = float(alpha_std)
        self.alpha_min = float(alpha_min)

        assert self.alpha_min > 0, "alpha_min must be > 0"                        # ASSERT
        assert self.alpha_std >= 0, "alpha_std must be >= 0"                       # ASSERT

        self.T = float(T)
        if self.T <= 0:
            raise ValueError("T must be > 0.")
        self.beta = 1.0 / self.T
        self.rng = np.random.default_rng(seed)

        # quenched alpha field (positive)
        alpha_field = self.rng.normal(self.alpha_mean, self.alpha_std, size=(self.L, self.L))
        self.alpha = np.clip(alpha_field, self.alpha_min, None)

        assert self.alpha.shape == (self.L, self.L)                               # ASSERT
        assert np.all(self.alpha >= self.alpha_min), "alpha clipping failed"      # ASSERT

        # decision spins S
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

        # ASSERT: spin values valid
        assert set(np.unique(self.S)).issubset({-1, +1})                          # ASSERT
        assert set(np.unique(self.C)).issubset({-1, +1})                          # ASSERT

        self._sumS = int(self.S.sum())
        self._n_chartist = int((self.C == -1).sum())

    # ---- observables ----
    def M(self) -> float:
        """Return current magnetization (mean decision spin)."""
        M = self._sumS / self.N
        assert -1.0 <= M <= 1.0                                                   # ASSERT
        return M

    def frac_chartist(self) -> float:
        """Return current fraction of chartists (C == -1)."""
        f = self._n_chartist / self.N
        assert 0.0 <= f <= 1.0                                                    # ASSERT
        return f

    def frac_fundamentalist(self) -> float:
        """Return current fraction of fundamentalists (C == +1)."""
        return 1.0 - self.frac_chartist()

    # ---- lattice helpers ----
    def _nn_sumS(self, i: int, j: int) -> int:
        """Sum of four nearest-neighbor decision spins (periodic boundaries)."""
        L = self.L
        S = self.S
        return (
            S[(i + 1) % L, j] + S[(i - 1) % L, j] +
            S[i, (j + 1) % L] + S[i, (j - 1) % L]
        )

    def _h(self, i: int, j: int, M_now: float) -> float:
        """Local field at site (i, j)."""
        return self.J * self._nn_sumS(i, j) - self.alpha[i, j] * self.C[i, j] * M_now

    @staticmethod
    def _p_up(beta: float, h: float) -> float:
        """Heat-bath probability of setting S=+1 (with clipping for stability)."""
        x = 2.0 * beta * h
        if x >= 50.0:
            return 1.0
        if x <= -50.0:
            return 0.0
        p = 1.0 / (1.0 + np.exp(-x))
        assert 0.0 <= p <= 1.0                                                    # ASSERT
        return p

    def _update_site(self, i: int, j: int) -> None:
        """One asynchronous update at (i, j): update S then possibly flip C."""
        M_before = self._sumS / self.N

        # 1) update S
        Sold = int(self.S[i, j])
        h = self._h(i, j, M_before)
        Snew = +1 if (self.rng.random() < self._p_up(self.beta, h)) else -1
        if Snew != Sold:
            self.S[i, j] = Snew
            self._sumS += (Snew - Sold)

        # 2) update C 
        Cold = int(self.C[i, j])
        if (self.S[i, j] * Cold * self._sumS) < 0:
            self.C[i, j] = -Cold
            if Cold == -1:
                self._n_chartist -= 1
            else:
                self._n_chartist += 1

        # ASSERT: cached counter stays consistent (cheap check)
        assert self._n_chartist == int((self.C == -1).sum())                      # ASSERT

    def sweep(self) -> None:
        """One Monte Carlo sweep (visit each site once in random order)."""
        order = self.rng.permutation(self.N)
        L = self.L
        for k in order:
            i, j = divmod(int(k), L)
            self._update_site(i, j)

        # ASSERT: cached sum stays consistent after a sweep
        assert self._sumS == int(self.S.sum())                                    # ASSERT

    def simulate(self, n_sweeps: int, burn_in: int = 0, thin: int = 1, eps: float = 1e-6):
        """
        Run the model for n_sweeps sweeps and return time series observables.

        Returns a dict with:
          M, r, abs_r, frac_chartist, frac_fundamentalist
        """
        # ASSERT: input validation (bonus-friendly, and matches your ValueErrors)
        assert n_sweeps > 0                                                       # ASSERT
        assert 0 <= burn_in < n_sweeps                                            # ASSERT
        assert thin > 0                                                           # ASSERT
        assert eps > 0                                                            # ASSERT

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
            if (t - burn_in) % thin != 0:
                continue

            M_series.append(self.M())
            chartist_series.append(self.frac_chartist())

        M_series = np.asarray(M_series, dtype=float)
        chartist_series = np.asarray(chartist_series, dtype=float)

        P = np.abs(M_series) + float(eps)
        if len(P) >= 2:
            r = np.log(P[1:]) - np.log(P[:-1])
        else:
            r = np.array([], dtype=float)

        return {
            "M": M_series,
            "r": r,
            "abs_r": np.abs(r),
            "frac_chartist": chartist_series,
            "frac_fundamentalist": 1.0 - chartist_series,
        }
