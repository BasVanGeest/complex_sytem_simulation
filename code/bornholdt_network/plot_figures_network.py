# plot_figures_network.py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def load_csv(path: str):
    """
    Load CSV written by run_network / run_networks:
      - optional comment lines starting with '#'
      - then a normal CSV header row:
          t,M,r,abs_r,C_mean,chartist_frac,fundamentalist_frac
      - then numeric rows

    Returns a structured numpy array with named columns.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    # drop comment lines
    lines = [ln for ln in lines if not ln.startswith("#")]
    if len(lines) < 2:
        raise ValueError(f"CSV {path} doesn't look like: [header row] + [numeric rows].")

    header = [c.strip() for c in lines[0].split(",")]
    data_lines = lines[1:]

    buf = StringIO("\n".join(data_lines))
    data = np.genfromtxt(buf, delimiter=",", names=header, dtype=float)
    return data


def resolve_default_data_path(explicit: str | None) -> str:
    if explicit:
        return explicit

    # common outputs from run_network / run_networks
    candidates = [
        os.path.join("data", "network_run.csv"),
        os.path.join("data", "intermediate_network.csv"),
        os.path.join("data", "intermediate.csv"),  # fallback (baseline-like)
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # If nothing exists, return the first expected path so error is clear.
    return candidates[0]


def ensure_required_columns(data, required):
    names = set(data.dtype.names or [])
    missing = required - names
    if missing:
        raise ValueError(f"Missing columns {missing}. Found: {data.dtype.names}")


def ccdf(x):
    """Complementary CDF: P(X >= x). Returns sorted x and CCDF y."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = np.sort(x)
    n = len(x)
    if n == 0:
        return np.array([]), np.array([])
    y = 1.0 - (np.arange(n) / n)
    return x, y


def autocorrelation(x, max_lag):
    """Normalized autocorrelation function."""
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


def window_by_t(data, t_min, t_max):
    t = data["t"]
    mask = (t >= t_min) & (t <= t_max)
    return data[mask]


# ---------------------------------------------------------------------
# FIGURE 2 — Returns time series
# ---------------------------------------------------------------------
def plot_fig2(data, outdir, t_min=10000, t_max=20000, prefix="network_"):
    d = window_by_t(data, t_min, t_max)
    t = d["t"]
    r = d["r"]
    mask = np.isfinite(r)

    plt.figure(figsize=(10, 3))
    plt.plot(t[mask], r[mask], lw=0.5, color="black")
    plt.xlabel("time (sweeps)")
    plt.ylabel("return r(t)")
    plt.title("Fig. 2 — Returns time series (network)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{prefix}fig2_returns_timeseries.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# FIGURE 3 — Yamano-style CCDF of absolute returns
# ---------------------------------------------------------------------
def plot_fig3_yamano(data, outdir, scale=100.0, prefix="network_"):
    """
    Yamano (2002) definition:
      ret(t) = ln|M(t)| - ln|M(t-1)|
    Here abs_r is already |ret(t)| (first entry NaN).
    Plot = CCDF of X = scale * abs_r on log-log axes.
    """
    abs_r = data["abs_r"]
    abs_r = abs_r[np.isfinite(abs_r)]
    X = scale * abs_r

    x, y = ccdf(X)
    if x.size == 0:
        raise ValueError("No finite abs_r values found for Fig. 3.")

    plt.figure(figsize=(5, 5))
    plt.loglog(x, y, ".", markersize=2, color="black")
    plt.xlabel(f"{scale:g} × |ret|")
    plt.ylabel(f"P({scale:g}×|ret| ≥ x)")
    plt.title("Fig. 3 — CCDF of absolute returns (network, Yamano style)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{prefix}fig3_ccdf_abs_returns_yamano.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# FIGURE 4 — Volatility autocorrelation
# ---------------------------------------------------------------------
def plot_fig4(data, outdir, max_lag=2000, prefix="network_"):
    abs_r = data["abs_r"]
    abs_r = abs_r[np.isfinite(abs_r)]

    # Volatility V(t) = |ret(t)|
    V = abs_r

    acf = autocorrelation(V, max_lag)

    # start from lag 1 for log-log
    lags = np.arange(1, len(acf), dtype=float)
    vals = acf[1:]

    mask = np.isfinite(vals) & (vals > 0)
    lags = lags[mask]
    vals = vals[mask]

    plt.figure(figsize=(6, 4))
    plt.loglog(lags, vals, lw=1.0, color="black")
    plt.xlabel("Time")
    plt.ylabel("Autocorrelation function")
    plt.title("Fig. 4 — Volatility autocorrelation (network, Yamano style)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{prefix}fig4_volatility_autocorr_yamano.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# FIGURE 5 — Fraction of chartists vs volatility
# ---------------------------------------------------------------------
def plot_fig5(data, outdir, t_min=10000, t_max=20000, prefix="network_"):
    d = window_by_t(data, t_min, t_max)

    t = d["t"]
    abs_r = d["abs_r"]
    chartist_frac = d["chartist_frac"]

    mask = np.isfinite(abs_r) & np.isfinite(chartist_frac)

    fig, ax1 = plt.subplots(figsize=(10, 4))

    ax1.plot(t[mask], abs_r[mask], color="black", lw=0.5)
    ax1.set_xlabel("time (sweeps)")
    ax1.set_ylabel("|ret(t)|", color="black")

    ax2 = ax1.twinx()
    ax2.plot(t[mask], chartist_frac[mask], color="red", lw=0.8)
    ax2.set_ylabel("fraction of chartists (C=-1)", color="red")

    plt.title("Fig. 5 — Volatility and fraction of chartists (network)")
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, f"{prefix}fig5_chartist_fraction_vs_volatility.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Plot network figures from run_network/run_networks CSV output.")
    ap.add_argument("--data", type=str, default=None,
                    help="Path to CSV produced by run_network/run_networks.")
    ap.add_argument("--outdir", type=str, default="paper_figures_network",
                    help="Directory to save figures.")
    ap.add_argument("--prefix", type=str, default="network_",
                    help="Filename prefix for saved figures.")

    ap.add_argument("--t_min", type=int, default=10000)
    ap.add_argument("--t_max", type=int, default=20000)
    ap.add_argument("--scale", type=float, default=100.0)
    ap.add_argument("--max_lag", type=int, default=2000)
    args = ap.parse_args()

    data_path = resolve_default_data_path(args.data)
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    data = load_csv(data_path)

    required = {"t", "M", "r", "abs_r", "C_mean", "chartist_frac", "fundamentalist_frac"}
    ensure_required_columns(data, required)

    plot_fig2(data, outdir, t_min=args.t_min, t_max=args.t_max, prefix=args.prefix)
    plot_fig3_yamano(data, outdir, scale=args.scale, prefix=args.prefix)
    plot_fig4(data, outdir, max_lag=args.max_lag, prefix=args.prefix)
    plot_fig5(data, outdir, t_min=args.t_min, t_max=args.t_max, prefix=args.prefix)

    print(f"[OK] Figures saved to '{outdir}/' from '{data_path}'")


if __name__ == "__main__":
    main()