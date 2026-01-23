import os
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_timeseries(csv_path: str):
    # reads columns: t_M,M,t_r,r,abs_r
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    M = data["M"]
    r = data["r"]
    abs_r = data["abs_r"]

    # last row has empty r/abs_r -> becomes nan
    r = r[~np.isnan(r)]
    abs_r = abs_r[~np.isnan(abs_r)]
    return M, r, abs_r


def acf(x: np.ndarray, max_lag: int):
    """
    Simple normalized autocorrelation function.
    ACF(lag) = cov(x_t, x_{t+lag}) / var(x)
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
    fig_dir = os.path.join(in_dir, "figures")
    ensure_dir(fig_dir)

    M, r, abs_r = load_timeseries(csv_path)

    # 1) M(t) timeseries
    plt.figure()
    plt.plot(M)
    plt.xlabel("sample index")
    plt.ylabel("M(t)")
    plt.title("Magnetization time series M(t)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_M_timeseries.png"), dpi=200)

    # 2) Return histogram (heavy tails)
    plt.figure()
    plt.hist(r, bins=80, density=True)
    plt.xlabel("r(t)")
    plt.ylabel("density")
    plt.title("Return distribution (histogram)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_return_hist.png"), dpi=200)

    # 3) ACF of |r| (volatility clustering)
    max_lag = 200
    acf_abs = acf(abs_r, max_lag=max_lag)

    plt.figure()
    plt.plot(np.arange(max_lag + 1), acf_abs)
    plt.xlabel("lag")
    plt.ylabel("ACF(|r|)")
    plt.title("Volatility clustering: ACF of |returns|")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_acf_abs_returns.png"), dpi=200)

    print("Saved figures to:", fig_dir)


if __name__ == "__main__":
    main()
