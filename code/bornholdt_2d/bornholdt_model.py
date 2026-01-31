import numpy as np


class Bornholdt2D:
    """
    Bornholdt (2001) 2D market spin model with *asynchronous* updates.
    """

    def __init__(self, L, J=1.0, alpha=4.0, T=1.5, seed=None,
                 init_S="random", init_C="random"):
        self.L = int(L)
        assert self.L > 0, "Lattice size L must be positive"              # ASSERT

        self.N = self.L * self.L
        self.J = float(J)
        self.alpha = float(alpha)
        self.T = float(T)

        assert self.alpha > 0, "alpha must be positive"                  # ASSERT
        assert self.T > 0, "Temperature T must be > 0"                   # ASSERT

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
            self.C = np.ones((self.L, self.L), dtype=int)
        elif init_C == "all_chartist":
            self.C = -np.ones((self.L, self.L), dtype=int)
        else:
            raise ValueError(f"Unknown init_C={init_C!r}")

        # ASSERT: spin values valid
        assert set(np.unique(self.S)).issubset({-1, +1})                 # ASSERT
        assert set(np.unique(self.C)).issubset({-1, +1})                 # ASSERT

        self._sumS = int(self.S.sum())
        self._sumC = int(self.C.sum())
        self._n_chartist = int((self.C == -1).sum())

    # ---- observables ----
    def M(self) -> float:
        M = self._sumS / self.N
        assert -1.0 <= M <= 1.0                                           # ASSERT
        return M

    def frac_chartist(self) -> float:
        f = self._n_chartist / self.N
        assert 0.0 <= f <= 1.0                                            # ASSERT
        return f

    def frac_fundamentalist(self) -> float:
        return 1.0 - self.frac_chartist()

    # ---- lattice helpers ----
    def _nn_sumS(self, i, j) -> int:
        L = self.L
        S = self.S
        return (
            S[(i + 1) % L, j] +
            S[(i - 1) % L, j] +
            S[i, (j + 1) % L] +
            S[i, (j - 1) % L]
        )

    def _h(self, i, j, M_now) -> float:
        return self.J * self._nn_sumS(i, j) - self.alpha * self.C[i, j] * M_now

    @staticmethod
    def _p_up(beta, h) -> float:
        x = 2.0 * beta * h
        if x >= 50.0:
            return 1.0
        if x <= -50.0:
            return 0.0
        p = 1.0 / (1.0 + np.exp(-x))
        assert 0.0 <= p <= 1.0                                            # ASSERT
        return p

    # ---- single-site asynchronous update ----
    def _update_site(self, i, j) -> None:
        M_before = self._sumS / self.N

        Sold = int(self.S[i, j])
        h = self._h(i, j, M_before)
        Snew = +1 if (self.rng.random() < self._p_up(self.beta, h)) else -1

        if Snew != Sold:
            self.S[i, j] = Snew
            self._sumS += (Snew - Sold)

        Ci_old = int(self.C[i, j])
        if (self.S[i, j] * Ci_old * self._sumS) < 0:
            Ci_new = -Ci_old
            self.C[i, j] = Ci_new
            self._sumC += (Ci_new - Ci_old)
            self._n_chartist += (-1 if Ci_old == -1 else 1)

        # ASSERT: cached counters consistent
        assert self._sumS == int(self.S.sum())                            # ASSERT
        assert self._sumC == int(self.C.sum())                            # ASSERT
        assert self._n_chartist == int((self.C == -1).sum())              # ASSERT

    # ---- one Monte Carlo sweep ----
    def sweep(self) -> None:
        order = self.rng.permutation(self.N)
        L = self.L
        for k in order:
            i, j = divmod(int(k), L)
            self._update_site(i, j)

    # ---- simulation driver ----
    def simulate(self, n_sweeps, burn_in=0, thin=1, eps=1e-6):
        assert n_sweeps > 0                                                # ASSERT
        assert 0 <= burn_in < n_sweeps                                     # ASSERT
        assert thin > 0                                                     # ASSERT
        assert eps > 0                                                      # ASSERT

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

        P = np.abs(M_series) + eps
        r = np.log(P[1:]) - np.log(P[:-1]) if len(P) >= 2 else np.array([])

        return {
            "M": M_series,
            "r": r,
            "abs_r": np.abs(r),
            "frac_chartist": chartist_series,
            "frac_fundamentalist": 1.0 - chartist_series,
        }

