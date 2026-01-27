# plot_paper_figures.py
import os
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def load_csv(path: str):
    """
    Load CSV written by the NEW run_baseline.py:
      - optional comment lines starting with '#'
      - then a normal CSV header row: t,M,r,abs_r,C_mean,chartist_frac,fundamentalist_frac
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


def ccdf(x):
    """Complementary CDF: P(X >= x). Returns sorted x and CCDF y."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = np.sort(x)
    n = len(x)
    if n == 0:
        return np.array([]), np.array([])
    # y[i] = fraction with value >= x[i]
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


# ---------------------------------------------------------------------
# FIGURE 2 — Returns time series
# ---------------------------------------------------------------------
def plot_fig2(data, outdir):
    t = data["t"]
    r = data["r"]
    mask = np.isfinite(r)

    plt.figure(figsize=(10, 3))
    plt.plot(t[mask], r[mask], lw=0.5, color="black")
    plt.xlabel("time (sweeps)")
    plt.ylabel("return r(t)")
    plt.title("Fig. 2 — Returns time series")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig2_returns_timeseries.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# FIGURE 3 — Yamano-style CCDF of absolute returns
# ---------------------------------------------------------------------
def plot_fig3_yamano(data, outdir, scale=100.0):
    """
    Yamano (2002) definition:
      ret(t) = ln|M(t)| - ln|M(t-1)|
    and then they plot the cumulative distribution of absolute returns,
    shown as "100 * ret" in the caption/axis for readability.

    In our pipeline, abs_r is already |ret(t)| (padded first entry NaN).
    So Yamano-style plot = CCDF of X = scale * abs_r on log-log axes.
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
    plt.title("Fig. 3 — CCDF of absolute returns (Yamano style)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig3_ccdf_abs_returns_yamano.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# FIGURE 4 — Volatility autocorrelation
# ---------------------------------------------------------------------
def plot_fig4(data, outdir, max_lag=2000):
    abs_r = data["abs_r"]
    abs_r = abs_r[np.isfinite(abs_r)]

    acf = autocorrelation(abs_r, max_lag)

    plt.figure(figsize=(6, 4))
    plt.semilogx(np.arange(len(acf)), acf, lw=1.0, color="black")
    plt.xlabel("lag T")
    plt.ylabel("ACF(|ret|)")
    plt.title("Fig. 4 — Volatility autocorrelation")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig4_volatility_acf.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# FIGURE 5 — Fraction of chartists vs volatility
# ---------------------------------------------------------------------
def plot_fig5(data, outdir):
    t = data["t"]
    abs_r = data["abs_r"]
    chartist_frac = data["chartist_frac"]

    mask = np.isfinite(abs_r) & np.isfinite(chartist_frac)

    fig, ax1 = plt.subplots(figsize=(10, 4))

    # Volatility (|ret|)
    ax1.plot(t[mask], abs_r[mask], color="black", lw=0.5)
    ax1.set_xlabel("time (sweeps)")
    ax1.set_ylabel("|ret(t)|", color="black")

    # Strategy fraction (second axis)
    ax2 = ax1.twinx()
    ax2.plot(t[mask], chartist_frac[mask], color="red", lw=0.8)
    ax2.set_ylabel("fraction of chartists (C=-1)", color="red")

    plt.title("Fig. 5 — Volatility and fraction of chartists")
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, "fig5_chartist_fraction_vs_volatility.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    data_path = os.path.join("data", "intermediate.csv")  # single CSV from run_baseline.py
    outdir = "paper_figures"
    os.makedirs(outdir, exist_ok=True)

    data = load_csv(data_path)

    required = {"t", "M", "r", "abs_r", "C_mean", "chartist_frac", "fundamentalist_frac"}
    missing = required - set(data.dtype.names)
    if missing:
        raise ValueError(f"Missing columns {missing}. Found: {data.dtype.names}")

    plot_fig2(data, outdir)
    plot_fig3_yamano(data, outdir, scale=100.0)  # Yamano-style Fig 3
    plot_fig4(data, outdir, max_lag=2000)
    plot_fig5(data, outdir)

    print(f"Figures saved to '{outdir}/'")


if __name__ == "__main__":
    main()
