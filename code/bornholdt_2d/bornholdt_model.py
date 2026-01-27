import numpy as np


class Bornholdt2D:
    """
    Bornholdt (2001) 2D spin market model (two spins per agent), written to match
    the paper's *time-step logic*:

    Variables on a LxL lattice (periodic BC):
      - decision spin S_i(t) ∈ {+1, -1}  (buy/sell)
      - strategy spin C_i(t) ∈ {+1, -1}  (fundamentalist/chartist indicator)

    Magnetization:
      M(t) = (1/N) * Σ_i S_i(t)

    Local field (paper Eq.(2)-style):
      h_i(t) = J * Σ_{j nn(i)} S_j(t)  -  α * C_i(t) * M(t)

    Heat-bath update for S (paper Eq.(1)-style):
      P(S_i(t+1)=+1) = 1 / (1 + exp(-2β h_i(t))), β = 1/T

    Strategy switching (paper Eq.(3)-style), applied *synchronously once per sweep*:
      if  S_i(t+1) * C_i(t) * M(t+1) < 0  then  C_i(t+1) = - C_i(t)
      else                                  C_i(t+1) =   C_i(t)

    Key fidelity choice vs many implementations:
      - During a Monte Carlo sweep, C is held fixed (represents C(t)).
      - After the sweep, we compute M(t+1) and then update all C synchronously to C(t+1).
      This matches the paper's notion of a discrete time step t -> t+1.
    """

    def __init__(
        self,
        L: int,
        J: float = 1.0,
        alpha: float = 4.0,
        T: float = 1.5,
        seed: int | None = None,
        init_S: str = "random",   # "random", "all_up", "all_down"
        init_C: str = "random",   # "random", "all_fundamentalist", "all_chartist"
    ):
        self.L = int(L)
        self.N = self.L * self.L

        self.J = float(J)
        self.alpha = float(alpha)
        self.T = float(T)
        if self.T <= 0:
            raise ValueError("Temperature T must be > 0.")
        self.beta = 1.0 / self.T

        self.rng = np.random.default_rng(seed)

        # --- initialize S ---
        if init_S == "random":
            self.S = self.rng.choice([-1, +1], size=(self.L, self.L))
        elif init_S == "all_up":
            self.S = np.ones((self.L, self.L), dtype=int)
        elif init_S == "all_down":
            self.S = -np.ones((self.L, self.L), dtype=int)
        else:
            raise ValueError(f"Unknown init_S='{init_S}'")

        # maintain Σ S for O(1) magnetization
        self._sumS = int(self.S.sum())

        # --- initialize C ---
        if init_C == "random":
            self.C = self.rng.choice([-1, +1], size=(self.L, self.L))
        elif init_C == "all_fundamentalist":
            # choose a convention: C=+1 means (say) fundamentalist
            self.C = np.ones((self.L, self.L), dtype=int)
        elif init_C == "all_chartist":
            self.C = -np.ones((self.L, self.L), dtype=int)
        else:
            raise ValueError(f"Unknown init_C='{init_C}'")

    # ---------------------------------------------------------------------
    # Theory-facing quantities
    # ---------------------------------------------------------------------
    def magnetization_M(self) -> float:
        """M(t) = (1/N) Σ_i S_i(t)."""
        return self._sumS / self.N

    def mean_strategy_C(self) -> float:
        """⟨C⟩(t) = (1/N) Σ_i C_i(t) (useful for the paper's 'fraction of chartists' proxy)."""
        return float(self.C.mean())

    # ---------------------------------------------------------------------
    # Lattice helpers (periodic boundary conditions)
    # ---------------------------------------------------------------------
    def sum_nearest_neighbors_S(self, i: int, j: int) -> int:
        """Σ_{nn} S_j for the four nearest neighbors on a periodic 2D lattice."""
        L = self.L
        return (
            self.S[(i + 1) % L, j] +
            self.S[(i - 1) % L, j] +
            self.S[i, (j + 1) % L] +
            self.S[i, (j - 1) % L]
        )

    def local_field_h(self, i: int, j: int, M: float) -> float:
        """h_i(t) = J Σ_{nn} S - α C_i(t) M(t)."""
        return self.J * self.sum_nearest_neighbors_S(i, j) - self.alpha * self.C[i, j] * M

    # ---------------------------------------------------------------------
    # Microscopic update rules
    # ---------------------------------------------------------------------
    @staticmethod
    def _heat_bath_prob_up(beta: float, h: float) -> float:
        """
        Heat-bath probability P(S=+1) = 1/(1+exp(-2βh)), computed stably.
        """
        x = 2.0 * beta * h
        # avoid overflow in exp for large |x|
        if x >= 50.0:
            return 1.0
        if x <= -50.0:
            return 0.0
        return 1.0 / (1.0 + np.exp(-x))

    def update_decision_spin_S(self, i: int, j: int, M: float) -> None:
        """
        Update S_i by heat-bath using the *current* M and the *current* C (held fixed during sweep).
        This is the micro-update inside a Monte Carlo sweep.
        """
        S_old = int(self.S[i, j])
        h = self.local_field_h(i, j, M)
        p_up = self._heat_bath_prob_up(self.beta, h)
        S_new = 1 if (self.rng.random() < p_up) else -1
        if S_new != S_old:
            self.S[i, j] = S_new
            self._sumS += (S_new - S_old)

    def update_strategy_spins_C_synchronously(self, M: float) -> None:
        """
        Apply the Bornholdt strategy switching rule (paper Eq.(3)) *synchronously*:
          if C_i * S_i * M < 0 then flip C_i
        """
        # vectorized flip mask
        flip = (self.C * self.S * M) < 0
        self.C[flip] *= -1

    # ---------------------------------------------------------------------
    # One model time step (paper t -> t+1)
    # ---------------------------------------------------------------------
    def monte_carlo_sweep(self) -> None:
        """
        One Monte Carlo sweep = N random-serial decision-spin updates.
        Strategy spins are NOT updated here (C is held fixed during the sweep).
        """
        order = self.rng.permutation(self.N)
        L = self.L
        for k in order:
            i, j = divmod(int(k), L)
            # use instantaneous M during the sweep (based on current ΣS)
            M_inst = self.magnetization_M()
            self.update_decision_spin_S(i, j, M_inst)

    def time_step(self) -> None:
        """
        One Bornholdt time step t -> t+1:
          1) update S via one Monte Carlo sweep with C fixed
          2) compute M(t+1)
          3) update all C synchronously using S(t+1), M(t+1)
        """
        self.monte_carlo_sweep()
        M_new = self.magnetization_M()
        self.update_strategy_spins_C_synchronously(M_new)

    # ---------------------------------------------------------------------
    # Observables: returns from magnetization (paper-style)
    # ---------------------------------------------------------------------
    @staticmethod
    def returns_from_magnetization(M_series: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Paper-style practical return:
          r(t) = log(|M(t)| + eps) - log(|M(t-1)| + eps)
        """
        M_series = np.asarray(M_series, dtype=float)
        if M_series.size < 2:
            return np.array([], dtype=float)
        P = np.abs(M_series) + float(eps)
        return np.log(P[1:]) - np.log(P[:-1])

    # ---------------------------------------------------------------------
    # Simulation driver
    # ---------------------------------------------------------------------
    def simulate(
        self,
        n_steps: int,
        burn_in: int = 0,
        thin: int = 1,
        eps: float = 1e-6,
        record_C: bool = True,
    ) -> dict:
        """
        Run for n_steps Bornholdt time steps (each is sweep + synchronous C update).
        Records M(t) (and optionally ⟨C⟩(t)) after burn-in, with thinning.
        """
        if n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if burn_in < 0 or burn_in >= n_steps:
            raise ValueError("burn_in must satisfy 0 <= burn_in < n_steps.")
        if thin <= 0:
            raise ValueError("thin must be positive.")

        M_list: list[float] = []
        C_list: list[float] = []

        for t in range(n_steps):
            self.time_step()

            if t < burn_in:
                continue
            if ((t - burn_in) % thin) != 0:
                continue

            M_list.append(self.magnetization_M())
            if record_C:
                C_list.append(self.mean_strategy_C())

        M = np.array(M_list, dtype=float)
        r = self.returns_from_magnetization(M, eps=eps)
        abs_r = np.abs(r)

        out = {"M": M, "r": r, "abs_r": abs_r}
        if record_C:
            out["C_mean"] = np.array(C_list, dtype=float)
        return out
