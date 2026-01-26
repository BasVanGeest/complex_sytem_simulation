import os
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_timeseries(csv_path: str):
    """
    Loads timeseries.csv written by your runner script.

    Required columns:
      M, r, abs_r

    Optional (for Bornholdt Fig. 5 diagnostics):
      C_mean (mean strategy spin; C=-1 chartist, C=+1 fundamentalist in Bornholdt)
      chartist_frac (fraction of agents with C=-1)
    """
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    cols = set(data.dtype.names)

    M = data["M"] if "M" in cols else None
    r = data["r"] if "r" in cols else None
    abs_r = data["abs_r"] if "abs_r" in cols else None

    C_mean = data["C_mean"] if "C_mean" in cols else None
    chartist_frac = data["chartist_frac"] if "chartist_frac" in cols else None

    # Drop NaNs (some pipelines pad last row)
    def drop_nans(x):
        if x is None:
            return None
        x = np.asarray(x, dtype=float)
        return x[~np.isnan(x)]

    M = drop_nans(M)
    r = drop_nans(r)
    abs_r = drop_nans(abs_r)
    C_mean = drop_nans(C_mean)
    chartist_frac = drop_nans(chartist_frac)

    return M, r, abs_r, C_mean, chartist_frac


def ccdf_abs_returns(abs_r: np.ndarray):
    """Empirical CCDF: P(|r| >= x)."""
    x = np.sort(np.abs(abs_r))
    n = len(x)
    ccdf = 1.0 - (np.arange(n) / n)
    return x, ccdf


def acf(x: np.ndarray, max_lag: int):
    """Normalized autocorrelation function."""
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


def rolling_mean(x: np.ndarray, window: int):
    """Simple rolling mean (valid mode)."""
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x.copy()
    if len(x) < window:
        return np.array([], dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def pearson_corr(x: np.ndarray, y: np.ndarray):
    """Pearson correlation with basic guards."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 2:
        return np.nan
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom == 0:
        return np.nan
    return float(np.sum(x * y) / denom)


def main():
    in_dir = "results"
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
        print("No returns 'r' found; cannot plot Fig. 2.")
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
    # Bornholdt Fig. 3: CCDF of absolute returns
    # -------------------------
    if abs_r is None or len(abs_r) == 0:
        print("No abs_r found; cannot plot Fig. 3.")
    else:
        x, ccdf = ccdf_abs_returns(abs_r)
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
    # Bornholdt Fig. 4: autocorrelation of |r|
    # -------------------------
    if abs_r is None or len(abs_r) == 0:
        print("No abs_r found; cannot plot Fig. 4.")
    else:
        max_lag = min(2000, len(abs_r) - 1)
        acf_abs = acf(abs_r, max_lag=max_lag)

        lags = np.arange(max_lag + 1)
        plt.figure()
        plt.semilogx(lags[1:], acf_abs[1:], linewidth=1.0)
        plt.xlabel("lag")
        plt.ylabel("ACF(|r|)")
        plt.title("Bornholdt Fig. 4 style: Volatility autocorrelation")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "fig4_acf_abs_returns.png"), dpi=200)
        plt.close()

    # -------------------------
    # Bornholdt Fig. 5 (IMPROVED): strategy vs volatility in time
    # -------------------------
    # We prefer to work with chartist_frac if available.
    # In Bornholdt, chartists correspond to C=-1 (ferromagnetic to magnetization term).
    # If only C_mean is available: a higher chartist presence corresponds to LOWER C_mean.
    # We'll define a "chartist_indicator" that increases when chartists dominate:
    #   chartist_indicator = chartist_frac (if provided)
    #   chartist_indicator = -C_mean (if only C_mean provided)
    if abs_r is None or len(abs_r) == 0:
        print("Fig. 5 skipped: requires abs_r.")
        return

    chartist_indicator = None
    strategy_label = None

    if chartist_frac is not None and len(chartist_frac) > 1:
        # align lengths later
        chartist_indicator = chartist_frac
        strategy_label = "fraction chartists"
    elif C_mean is not None and len(C_mean) > 1:
        chartist_indicator = -C_mean
        strategy_label = "- <C>(t)  (higher = more chartists)"
    else:
        print("Fig. 5 skipped: CSV does not contain 'C_mean' or 'chartist_frac'.")
        print("Add C_mean (or chartist_frac) to timeseries.csv to validate Bornholdt Fig. 5.")
        return

    # Align time series:
    # abs_r length is len(M)-1, and typically r/abs_r are sampled per recorded M step.
    # strategy series is length len(M). We'll drop last strategy sample for alignment.
    min_len = min(len(abs_r), len(chartist_indicator) - 1)
    if min_len <= 10:
        print("Fig. 5 skipped: not enough aligned samples.")
        return

    vol = abs_r[:min_len]
    strat = chartist_indicator[:min_len]  # use first min_len (after dropping last implicitly)

    t = np.arange(min_len)

    # --- Fig 5a: overlay raw volatility and strategy indicator (two y-axes)
    fig, ax1 = plt.subplots(figsize=(10, 3))
    ax1.plot(t, vol, linewidth=0.8)
    ax1.set_xlabel("time (sample index)")
    ax1.set_ylabel("|r(t)| (volatility proxy)")

    ax2 = ax1.twinx()
    ax2.plot(t, strat, linewidth=0.8)
    ax2.set_ylabel(strategy_label)

    plt.title("Bornholdt Fig. 5 improved: Volatility and strategy composition over time")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig5a_volatility_vs_strategy_overlay.png"), dpi=200)
    plt.close()

    # --- Fig 5b: rolling volatility vs strategy indicator (cleaner)
    window = max(10, min(200, min_len // 50))  # adaptive but reasonable
    vol_roll = rolling_mean(vol, window=window)
    strat_roll = rolling_mean(strat, window=window)
    Lr = min(len(vol_roll), len(strat_roll))
    if Lr > 10:
        t2 = np.arange(Lr)
        fig, ax1 = plt.subplots(figsize=(10, 3))
        ax1.plot(t2, vol_roll[:Lr], linewidth=1.0)
        ax1.set_xlabel("time (rolling index)")
        ax1.set_ylabel(f"rolling mean |r| (window={window})")

        ax2 = ax1.twinx()
        ax2.plot(t2, strat_roll[:Lr], linewidth=1.0)
        ax2.set_ylabel(f"rolling mean of {strategy_label}")

        plt.title("Bornholdt Fig. 5 improved: Rolling volatility vs rolling strategy")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "fig5b_rolling_volatility_vs_strategy.png"), dpi=200)
        plt.close()

    # --- Fig 5c: scatter + correlation (quantitative support)
    corr = pearson_corr(vol, strat)
    plt.figure(figsize=(4.5, 4))
    plt.scatter(strat, vol, s=6)
    plt.xlabel(strategy_label)
    plt.ylabel("|r(t)|")
    plt.title(f"Strategy vs volatility scatter (corr={corr:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig5c_scatter_strategy_vs_volatility.png"), dpi=200)
    plt.close()

    print("Saved paper-style figures to:", fig_dir)
    print(f"Strategy–volatility Pearson corr (aligned): {corr:.4f}")
    print("Note: higher strategy indicator = more chartists (either chartist_frac or -C_mean).")


if __name__ == "__main__":
    main()
