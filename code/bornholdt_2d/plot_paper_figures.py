# plot_paper_figures.py
import argparse
import os
from io import StringIO

import numpy as np
import matplotlib.pyplot as plt


def load_csv(path: str) -> np.ndarray:
    """Load CSV produced by run_baseline.py (supports '#' comment header lines)."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    if len(lines) < 2:
        raise ValueError(f"{path} does not contain a header + data rows.")

    header = [c.strip() for c in lines[0].split(",")]
    buf = StringIO("\n".join(lines[1:]))

    return np.genfromtxt(buf, delimiter=",", names=header, dtype=float)


def ccdf(x: np.ndarray):
    """Return sorted x and CCDF y = P(X >= x)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = np.sort(x)
    n = len(x)
    if n == 0:
        return np.array([]), np.array([])
    y = 1.0 - (np.arange(n) / n)
    return x, y


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalized autocorrelation for lags 0..max_lag."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.full(max_lag + 1, np.nan)

    x = x - x.mean()
    var = np.var(x)
    if var == 0:
        return np.full(max_lag + 1, np.nan)

    acf = np.empty(max_lag + 1, dtype=float)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = np.mean(x[:-lag] * x[lag:]) / var
    return acf


def window_by_t(data: np.ndarray, t_min: int, t_max: int) -> np.ndarray:
    """Filter structured array by t in [t_min, t_max]."""
    t = data["t"]
    return data[(t >= t_min) & (t <= t_max)]


def plot_fig2_returns_timeseries(data: np.ndarray, outdir: str, t_min=10_000, t_max=20_000):
    d = window_by_t(data, t_min, t_max)
    t = d["t"]
    r = d["r"]
    mask = np.isfinite(r)

    plt.figure(figsize=(10, 3))
    plt.plot(t[mask], r[mask], lw=0.5, color="black")
    plt.xlabel("time (steps)")
    plt.ylabel("return r(t)")
    plt.title("Returns time series")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "baseline_returns_timeseries.png"), dpi=300)
    plt.close()


def plot_fig3_ccdf_abs_returns(data: np.ndarray, outdir: str, scale=100.0):
    abs_r = data["abs_r"]
    abs_r = abs_r[np.isfinite(abs_r)]
    x, y = ccdf(scale * abs_r)
    if x.size == 0:
        raise ValueError("No finite abs_r values for CCDF plot.")

    plt.figure(figsize=(5, 5))
    plt.loglog(x, y, ".", markersize=2, color="black")
    plt.xlabel(f"{scale:g} × |ret|")
    plt.ylabel(f"P({scale:g}×|ret| ≥ x)")
    plt.title("CCDF of absolute returns")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "baseline_ccdf_abs_returns.png"), dpi=300)
    plt.close()


def plot_fig4_volatility_autocorr(data: np.ndarray, outdir: str, max_lag=2000):
    abs_r = data["abs_r"]
    abs_r = abs_r[np.isfinite(abs_r)]
    acf = autocorrelation(abs_r, max_lag)

    lags = np.arange(1, len(acf), dtype=float)
    vals = acf[1:]

    mask = np.isfinite(vals) & (vals > 0)
    lags = lags[mask]
    vals = vals[mask]

    plt.figure(figsize=(6, 4))
    plt.loglog(lags, vals, lw=1.0, color="black")
    plt.xlabel("lag")
    plt.ylabel("autocorrelation")
    plt.title("Volatility autocorrelation")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "baseline_volatility_autocorr.png"), dpi=300)
    plt.close()


def plot_fig5_chartist_vs_volatility(data: np.ndarray, outdir: str, t_min=10_000, t_max=20_000):
    d = window_by_t(data, t_min, t_max)
    t = d["t"]
    abs_r = d["abs_r"]
    chartist_frac = d["chartist_frac"]
    mask = np.isfinite(abs_r) & np.isfinite(chartist_frac)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(t[mask], abs_r[mask], lw=0.5, color="black")
    ax1.set_xlabel("time (steps)")
    ax1.set_ylabel("|ret(t)|")

    ax2 = ax1.twinx()
    ax2.plot(t[mask], chartist_frac[mask], lw=0.8, color="red")
    ax2.set_ylabel("fraction chartists (C=-1)", color="red")

    plt.title("Volatility and fraction of chartists")
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, "baseline_chartist_fraction_vs_volatility.png"), dpi=300)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate paper-style figures from a baseline lattice CSV.")
    ap.add_argument(
        "--data",
        type=str,
        default=os.path.join("data", "lattice_data_result_100000.csv"),
        help="Input CSV path (default: data/lattice_data_result_100000.csv).",
    )
    ap.add_argument("--outdir", type=str, default="results", help="Output directory for figures (default: results).")
    ap.add_argument("--tmin", type=int, default=10_000, help="Window start for time-series style plots.")
    ap.add_argument("--tmax", type=int, default=20_000, help="Window end for time-series style plots.")
    ap.add_argument("--max_lag", type=int, default=2000, help="Max lag for volatility autocorrelation.")
    ap.add_argument("--scale", type=float, default=100.0, help="Scale factor for CCDF x-axis (Yamano-style).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    data = load_csv(args.data)

    required = {"t", "M", "r", "abs_r", "C_mean", "chartist_frac", "fundamentalist_frac"}
    missing = required - set(data.dtype.names)
    if missing:
        raise ValueError(f"Missing columns {missing}. Found: {data.dtype.names}")

    plot_fig2_returns_timeseries(data, args.outdir, t_min=args.tmin, t_max=args.tmax)
    plot_fig3_ccdf_abs_returns(data, args.outdir, scale=args.scale)
    plot_fig4_volatility_autocorr(data, args.outdir, max_lag=args.max_lag)
    plot_fig5_chartist_vs_volatility(data, args.outdir, t_min=args.tmin, t_max=args.tmax)

    print(f"Figures saved to '{args.outdir}/'")


if __name__ == "__main__":
    main()

