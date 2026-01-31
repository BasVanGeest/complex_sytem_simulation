import argparse
import os
from io import StringIO

import numpy as np
import matplotlib.pyplot as plt


def load_csv(path: str) -> np.ndarray:
    """Load CSV with optional '#' comment header lines; returns structured array."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    if len(lines) < 2:
        raise ValueError(f"{path} must contain a CSV header row + data rows.")

    header = [c.strip() for c in lines[0].split(",")]
    buf = StringIO("\n".join(lines[1:]))
    return np.genfromtxt(buf, delimiter=",", names=header, dtype=float)


def ccdf(x: np.ndarray):
    """Return sorted x and CCDF y=P(X>=x)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = np.sort(x)
    n = x.size
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
    t = data["t"]
    return data[(t >= t_min) & (t <= t_max)]


def infer_topology_from_path(path: str) -> str:
    """Infer topology from filename prefix like 'ER_data_results_50000.csv'."""
    base = os.path.basename(path)
    for topo in ("ER", "BA", "WS"):
        if base.startswith(f"{topo}_"):
            return topo
    return "NETWORK"


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot figures for a single network CSV.")
    ap.add_argument(
        "--data",
        type=str,
        required=True,
        help="Input CSV path (e.g. data/ER_data_results_50000.csv).",
    )
    ap.add_argument(
        "--topology",
        type=str,
        default=None,
        help="Topology label used in output filenames (ER/BA/WS). If omitted, inferred from --data filename.",
    )
    ap.add_argument("--outdir", type=str, default="results", help="Output folder for PNGs (default: results).")
    ap.add_argument("--tmin", type=int, default=10_000)
    ap.add_argument("--tmax", type=int, default=20_000)
    ap.add_argument("--scale", type=float, default=100.0)
    ap.add_argument("--max_lag", type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    topo = (args.topology or infer_topology_from_path(args.data)).upper()

    data = load_csv(args.data)

    required = {"t", "M", "r", "abs_r", "chartist_frac"}
    missing = required - set(data.dtype.names or [])
    if missing:
        raise ValueError(f"Missing columns {missing}. Found: {data.dtype.names}")

    # --------------------
    # 1) Returns time series
    # --------------------
    d = window_by_t(data, args.tmin, args.tmax)
    t = d["t"]
    r = d["r"]
    mask = np.isfinite(r)

    plt.figure(figsize=(10, 3))
    plt.plot(t[mask], r[mask], lw=0.5, color="black")
    plt.xlabel("time (steps)")
    plt.ylabel("return r(t)")
    plt.title(f"{topo} — Returns time series")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{topo}_returns_timeseries.png"), dpi=300)
    plt.close()

    # --------------------
    # 2) CCDF of |returns|
    # --------------------
    abs_r = data["abs_r"]
    abs_r = abs_r[np.isfinite(abs_r)]
    X = args.scale * abs_r
    x, y = ccdf(X)
    if x.size == 0:
        raise ValueError("No finite abs_r values to plot CCDF.")

    plt.figure(figsize=(5, 5))
    plt.loglog(x, y, ".", markersize=2, color="black")
    plt.xlabel(f"{args.scale:g} × |ret|")
    plt.ylabel(f"P({args.scale:g}×|ret| ≥ x)")
    plt.title(f"{topo} — CCDF of absolute returns")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{topo}_ccdf_abs_returns.png"), dpi=300)
    plt.close()

    # --------------------
    # 3) Volatility autocorrelation
    # --------------------
    acf = autocorrelation(abs_r, args.max_lag)
    lags = np.arange(1, len(acf), dtype=float)
    vals = acf[1:]
    mask = np.isfinite(vals) & (vals > 0)

    plt.figure(figsize=(6, 4))
    plt.loglog(lags[mask], vals[mask], lw=1.0, color="black")
    plt.xlabel("lag")
    plt.ylabel("autocorrelation of |ret|")
    plt.title(f"{topo} — Volatility autocorrelation")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{topo}_volatility_autocorr.png"), dpi=300)
    plt.close()

    # --------------------
    # 4) Chartist fraction vs volatility (windowed)
    # --------------------
    abs_r_win = d["abs_r"]
    cf = d["chartist_frac"]
    mask = np.isfinite(abs_r_win) & np.isfinite(cf)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(d["t"][mask], abs_r_win[mask], lw=0.5, color="black")
    ax1.set_xlabel("time (steps)")
    ax1.set_ylabel("|ret(t)|", color="black")

    ax2 = ax1.twinx()
    ax2.plot(d["t"][mask], cf[mask], lw=0.8, color="red")
    ax2.set_ylabel("fraction chartists (C=-1)", color="red")

    plt.title(f"{topo} — Volatility and chartist fraction")
    fig.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{topo}_chartist_fraction_vs_volatility.png"), dpi=300)
    plt.close()

    print(f"[OK] Saved figures to '{args.outdir}/' for topology='{topo}' from '{args.data}'")


if __name__ == "__main__":
    main()
