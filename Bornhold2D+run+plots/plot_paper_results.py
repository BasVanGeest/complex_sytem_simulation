import os
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_timeseries(csv_path: str):
    """
    Loads timeseries.csv written by your runner script.
    Expected baseline columns:
      t_M, M, t_r, r, abs_r

    Optional (for Bornholdt Fig. 5):
      C_mean or chartist_frac
    """
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")

    cols = set(data.dtype.names)

    # Required series
    M = data["M"] if "M" in cols else None
    r = data["r"] if "r" in cols else None
    abs_r = data["abs_r"] if "abs_r" in cols else None

    # Optional series for Fig 5
    C_mean = data["C_mean"] if "C_mean" in cols else None
    chartist_frac = data["chartist_frac"] if "chartist_frac" in cols else None

    # Drop NaNs for r/abs_r because last CSV row is padded with blanks
    if r is not None:
        r = r[~np.isnan(r)]
    if abs_r is not None:
        abs_r = abs_r[~np.isnan(abs_r)]

    # If optional series exist, also drop NaNs safely
    if C_mean is not None:
        C_mean = C_mean[~np.isnan(C_mean)]
    if chartist_frac is not None:
        chartist_frac = chartist_frac[~np.isnan(chartist_frac)]

    return M, r, abs_r, C_mean, chartist_frac


def ccdf_abs_returns(abs_r: np.ndarray):
    """
    Empirical CCDF: P(|r| >= x).
    Returns x_sorted (ascending) and ccdf (descending from ~1 to 1/n).
    """
    x = np.sort(np.abs(abs_r))
    n = len(x)
    # CCDF at each sorted point: fraction of samples >= x_k
    ccdf = 1.0 - (np.arange(n) / n)
    return x, ccdf


def acf(x: np.ndarray, max_lag: int):
    """
    Simple normalized autocorrelation function:
      ACF(lag) = sum_{t} (x_t - mean)(x_{t+lag} - mean) / sum_{t} (x_t - mean)^2

    This matches the spirit of Bornholdt Fig 4 (volatility autocorrelation).
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    denom = np.sum(x * x)
    if denom == 0:
        return np.zeros(max_lag + 1, dtype=float)

    out = np.empty(max_lag + 1, dtype=float)
    out[0] = 1.0
    for lag in range(1, max_lag + 1):
        out[lag] = np.sum(x[:-lag] * x[lag:]) / denom
    return out


def main():
    in_dir = os.path.join("results")
    csv_path = os.path.join(in_dir, "timeseries.csv")
    fig_dir = os.path.join(in_dir, "paper_figures")
    ensure_dir(fig_dir)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}. Run the simulation script first.")

    M, r, abs_r, C_mean, chartist_frac = load_timeseries(csv_path)

    # -------------------------
    # Bornholdt Fig. 2: returns time series
    # -------------------------
    if r is None or len(r) == 0:
        print("No returns 'r' found in CSV; cannot plot Fig. 2.")
    else:
        plt.figure(figsize=(10, 3))
        plt.plot(r, linewidth=0.8)
        plt.xlabel("time (sample index)")
        plt.ylabel("returns r(t)")
        plt.title("Bornholdt Fig. 2 style: Returns time series")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "fig2_returns_timeseries.png"), dpi=200)
        plt.close()

    # -------------------------
    # Bornholdt Fig. 3: cumulative distribution (CCDF) of absolute returns
    # -------------------------
    if abs_r is None or len(abs_r) == 0:
        print("No abs_r found in CSV; cannot plot Fig. 3.")
    else:
        x, ccdf = ccdf_abs_returns(abs_r)

        # Avoid log(0) issues: drop zeros if they exist
        mask = (x > 0) & (ccdf > 0)
        x = x[mask]
        ccdf = ccdf[mask]

        plt.figure()
        plt.loglog(x, ccdf, marker=".", linestyle="none", markersize=2)
        plt.xlabel("|r|")
        plt.ylabel("P(|r| ≥ x)")
        plt.title("Bornholdt Fig. 3 style: CCDF of absolute returns")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "fig3_ccdf_abs_returns.png"), dpi=200)
        plt.close()

    # -------------------------
    # Bornholdt Fig. 4: autocorrelation of absolute returns (volatility clustering)
    # -------------------------
    if abs_r is None or len(abs_r) == 0:
        print("No abs_r found in CSV; cannot plot Fig. 4.")
    else:
        # Choose a lag range that makes sense for your sample size
        max_lag = min(2000, len(abs_r) - 1)  # adjust if you want longer
        acf_abs = acf(abs_r, max_lag=max_lag)

        # Bornholdt shows decay clearly; log-x is helpful
        lags = np.arange(max_lag + 1)
        # Avoid lag=0 for log scale (optional)
        lags_plot = lags[1:]
        acf_plot = acf_abs[1:]

        plt.figure()
        plt.semilogx(lags_plot, acf_plot, linewidth=1.0)
        plt.xlabel("lag")
        plt.ylabel("ACF(|r|)")
        plt.title("Bornholdt Fig. 4 style: Volatility autocorrelation")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "fig4_acf_abs_returns.png"), dpi=200)
        plt.close()

    # -------------------------
    # Bornholdt Fig. 5: strategy composition vs time
    # Requires C_mean or chartist_frac columns in CSV.
    # -------------------------
    if C_mean is not None and len(C_mean) > 0:
        plt.figure(figsize=(10, 3))
        plt.plot(C_mean, linewidth=0.8)
        plt.xlabel("time (sample index)")
        plt.ylabel("<C>(t)")
        plt.title("Bornholdt Fig. 5 style: Strategy balance (mean C)")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "fig5_strategy_balance_Cmean.png"), dpi=200)
        plt.close()
    elif chartist_frac is not None and len(chartist_frac) > 0:
        plt.figure(figsize=(10, 3))
        plt.plot(chartist_frac, linewidth=0.8)
        plt.xlabel("time (sample index)")
        plt.ylabel("fraction chartists")
        plt.title("Bornholdt Fig. 5 style: Fraction of chartists")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "fig5_fraction_chartists.png"), dpi=200)
        plt.close()
    else:
        print(
            "Fig. 5 skipped: CSV does not contain 'C_mean' or 'chartist_frac'.\n"
            "To plot Fig. 5, modify the simulation run() to save strategy composition into timeseries.csv."
        )

    print("Saved paper-style figures to:", fig_dir)

        # -------------------------
    # NEW: Conditional volatility (regime-dependent volatility)
    # Shows that volatility is higher on average when chartists dominate,
    # even if spikes do not align 1-to-1 with peaks of C_mean.
    # Requires C_mean and abs_r.
    # -------------------------
    if C_mean is not None and len(C_mean) > 1 and abs_r is not None and len(abs_r) > 0:
        # Align: abs_r has length len(M)-1; C_mean has length len(M)
        C_aligned = C_mean[:-1]
        abs_r_aligned = abs_r  # already length len(M)-1

        chartist_mask = C_aligned < 0
        fundamentalist_mask = C_aligned > 0

        # Guard against empty groups (can happen rarely depending on parameters)
        if np.any(chartist_mask) and np.any(fundamentalist_mask):
            mean_vol_chartist = float(np.mean(abs_r_aligned[chartist_mask]))
            mean_vol_fund = float(np.mean(abs_r_aligned[fundamentalist_mask]))

            plt.figure()
            plt.bar(
                ["Fundamentalist (C>0)", "Chartist (C<0)"],
                [mean_vol_fund, mean_vol_chartist],
            )
            plt.ylabel("Mean |r|")
            plt.title("Conditional volatility: mean(|r|) by strategy regime")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "fig6_conditional_volatility.png"), dpi=200)
            plt.close()

            print("Conditional volatility (mean |r|):")
            print(f"  Fundamentalist (C<0): {mean_vol_fund:.6f}")
            print(f"  Chartist (C>0):       {mean_vol_chartist:.6f}")
        else:
            print("Fig. 6 skipped: not enough samples in one of the regimes (C>0 or C<0).")
    else:
        print("Fig. 6 skipped: requires both C_mean and abs_r.")



if __name__ == "__main__":
    main()
