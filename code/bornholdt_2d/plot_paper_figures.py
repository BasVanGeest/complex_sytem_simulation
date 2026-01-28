# plot_paper_figures.py
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from io import StringIO


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def load_csv(path: str):
    """
    Load CSV written by run_baseline.py (with a 3-line text header).

    Expected header contains a line like:
        columns: t,M,r,abs_r,C_mean,chartist_frac,fundamentalist_frac

    Then numeric rows follow.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    # find the columns line
    col_line_idx = None
    colnames = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("columns:"):
            col_line_idx = i
            colnames = [c.strip() for c in line.split(":", 1)[1].split(",")]
            break

    if col_line_idx is None or not colnames:
        raise ValueError(
            f"Could not find a 'columns:' line in {path}. "
            "Your CSV must include a line like: columns: t,M,r,..."
        )

    # numeric data begins after the 'columns:' line
    data_lines = lines[col_line_idx + 1 :]
    if not data_lines:
        raise ValueError(f"No numeric data found after 'columns:' line in {path}")

    # parse numeric block
    buf = StringIO("\n".join(data_lines))
    data = np.genfromtxt(buf, delimiter=",", names=colnames, dtype=float)

    return data


def ccdf(x):
    """Complementary CDF: P(X >= x)."""
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


# ---------------------------------------------------------------------
# FIGURE 2 — Returns time series
# ---------------------------------------------------------------------
def plot_fig2(data, outdir, prefix):
    t = data["t"]
    r = data["r"]
    mask = np.isfinite(r)

    plt.figure(figsize=(10, 3))
    plt.plot(t[mask], r[mask], lw=0.5, color="black")
    plt.xlabel("time (sweeps)")
    plt.ylabel("return r(t)")
    plt.title("Fig. 2 — Returns time series (Bornholdt 2001)")
    plt.tight_layout()
    
    save_path = os.path.join(outdir, f"{prefix}_fig_2_returns_timeseries.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# FIGURE 3 — CCDF of |returns|
# ---------------------------------------------------------------------
def plot_fig3(data, outdir, prefix):
    abs_r = data["abs_r"]
    abs_r = abs_r[np.isfinite(abs_r)]

    x, y = ccdf(abs_r)

    plt.figure(figsize=(5, 5))
    plt.loglog(x, y, ".", markersize=2, color="black")
    plt.xlabel("|r|")
    plt.ylabel("P(|r| ≥ x)")
    plt.title("Fig. 3 — CCDF of absolute returns")
    plt.tight_layout()
    
    save_path = os.path.join(outdir, f"{prefix}_fig_3_ccdf_abs_returns.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# FIGURE 4 — Volatility autocorrelation
# ---------------------------------------------------------------------
def plot_fig4(data, outdir, prefix):
    abs_r = data["abs_r"]
    abs_r = abs_r[np.isfinite(abs_r)]

    max_lag = 2000 
    acf = autocorrelation(abs_r, max_lag)

    plt.figure(figsize=(6, 4))
    plt.semilogx(np.arange(len(acf)), acf, lw=1.0, color="black")
    plt.xlabel("lag T")
    plt.ylabel("ACF(|r|)")
    plt.title("Fig. 4 — Volatility autocorrelation")
    plt.tight_layout()
    
    save_path = os.path.join(outdir, f"{prefix}_fig_4_volatility_acf.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# FIGURE 5 — Fraction of chartists vs volatility
# ---------------------------------------------------------------------
def plot_fig5(data, outdir, prefix):
    t = data["t"]
    abs_r = data["abs_r"]
    chartist_frac = data["chartist_frac"]

    mask = np.isfinite(abs_r)

    fig, ax1 = plt.subplots(figsize=(10, 4))

    # Volatility (|r|)
    ax1.plot(t[mask], abs_r[mask], color="black", lw=0.5)
    ax1.set_xlabel("time (sweeps)")
    ax1.set_ylabel("|r(t)|", color="black")

    # Strategy fraction (second axis)
    ax2 = ax1.twinx()
    ax2.plot(t[mask], chartist_frac[mask], color="red", lw=0.8)
    ax2.set_ylabel("fraction of chartists", color="red")

    plt.title("Fig. 5 — Volatility and fraction of chartists")
    fig.tight_layout()
    
    save_path = os.path.join(outdir, f"{prefix}_fig_5_chartist_fraction_vs_volatility.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default=None, help="Path to the CSV data file")
    parser.add_argument("--outdir", type=str, default="graphs/paper_figures", help="Directory to save figure PNGs")
    
    # Arguments to help construct the prefix if data_file is not provided
    parser.add_argument("--topology", type=str, default="lattice")
    parser.add_argument("--steps", type=int, default=200000)
    
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Determine data file and prefix
    if args.data_file is None:
        prefix = f"{args.topology.lower()}_{args.steps}"
        data_path = os.path.join("/data", f"{prefix}_intermediate.csv")
    else:
        data_path = args.data_file
        # Extract prefix from filename if possible, otherwise use a generic one
        base = os.path.basename(data_path)
        if "_intermediate.csv" in base:
            prefix = base.replace("_intermediate.csv", "")
        else:
            prefix = "custom_run"

    data = load_csv(data_path)

    # Sanity check
    required = {"t", "M", "r", "abs_r", "C_mean", "chartist_frac", "fundamentalist_frac"}
    missing = required - set(data.dtype.names)
    if missing:
        raise ValueError(f"Data file is missing columns: {missing}")

    # Generate plots with the prefix
    plot_fig2(data, args.outdir, prefix)
    plot_fig3(data, args.outdir, prefix)
    plot_fig4(data, args.outdir, prefix)
    plot_fig5(data, args.outdir, prefix)

    print(f"Figures saved to '{args.outdir}/' with prefix '{prefix}'")


if __name__ == "__main__":
    main()