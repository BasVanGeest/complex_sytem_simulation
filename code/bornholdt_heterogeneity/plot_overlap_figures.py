import argparse
import os
from io import StringIO

import numpy as np
import matplotlib.pyplot as plt


def load_csv(path: str) -> np.ndarray:
    """Load CSV that may start with '#' comment lines; returns structured array."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]

    if len(lines) < 2:
        raise ValueError(f"{path} doesn't look like [header] + [data rows].")

    header = [c.strip() for c in lines[0].split(",")]
    buf = StringIO("\n".join(lines[1:]))
    return np.genfromtxt(buf, delimiter=",", names=header, dtype=float)


def ensure_abs_r(data: np.ndarray) -> np.ndarray:
    """Ensure data has abs_r; compute it from r if needed."""
    names = set(data.dtype.names)
    if "abs_r" in names:
        return data
    if "r" not in names:
        raise ValueError(f"Need 'abs_r' or 'r' in columns, found: {data.dtype.names}")

    abs_r = np.abs(data["r"])
    newdtype = data.dtype.descr + [("abs_r", "f8")]
    out = np.empty(data.shape, dtype=newdtype)
    for n in data.dtype.names:
        out[n] = data[n]
    out["abs_r"] = abs_r
    return out


def ccdf(x: np.ndarray):
    """Return sorted x and CCDF y=P(X>=x)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0]
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
    if x.size < 5:
        return np.full(max_lag + 1, np.nan)

    x = x - x.mean()
    var = np.var(x)
    if var == 0:
        return np.full(max_lag + 1, np.nan)

    max_lag = min(max_lag, x.size - 2)
    acf = np.empty(max_lag + 1, dtype=float)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = np.mean(x[:-lag] * x[lag:]) / var
    return acf


def window_by_t(data: np.ndarray, t_min: int, t_max: int) -> np.ndarray:
    """Filter structured array by t in [t_min, t_max]."""
    t = data["t"]
    m = np.isfinite(t) & (t >= t_min) & (t <= t_max)
    return data[m]


def plot_overlay_returns(base: np.ndarray, het: np.ndarray, outpath: str, t_min: int, t_max: int) -> None:
    """Overlay returns r(t) in a shared time window; align on common t values."""
    b = window_by_t(base, t_min, t_max)
    h = window_by_t(het, t_min, t_max)

    if b.size == 0 or h.size == 0:
        raise ValueError(f"No data in window [{t_min},{t_max}]. baseline={b.size}, hetero={h.size}")

    bt, br = b["t"], b["r"]
    ht, hr = h["t"], h["r"]

    common_t = np.intersect1d(bt[np.isfinite(bt)], ht[np.isfinite(ht)])
    common_t = common_t[(common_t >= t_min) & (common_t <= t_max)]
    if common_t.size == 0:
        raise ValueError(f"No common t values in [{t_min},{t_max}].")

    b_map = {int(t): r for t, r in zip(bt, br)}
    h_map = {int(t): r for t, r in zip(ht, hr)}
    r_base = np.array([b_map.get(int(t), np.nan) for t in common_t], dtype=float)
    r_het = np.array([h_map.get(int(t), np.nan) for t in common_t], dtype=float)

    mb = np.isfinite(r_base)
    mh = np.isfinite(r_het)

    plt.figure(figsize=(10, 3))
    plt.plot(common_t[mb], r_base[mb], lw=0.6, color="black", label="baseline")
    plt.plot(common_t[mh], r_het[mh], lw=0.6, color="red", alpha=0.8, label="heterogeneous α")
    plt.xlabel("time (steps)")
    plt.ylabel("return r(t)")
    plt.title("Overlay — Returns time series")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_overlay_ccdf(base: np.ndarray, het: np.ndarray, outpath: str, scale: float) -> None:
    """Overlay CCDF of scale*abs_r."""
    Xb = scale * base["abs_r"]
    Xh = scale * het["abs_r"]

    xb, yb = ccdf(Xb)
    xh, yh = ccdf(Xh)

    plt.figure(figsize=(6, 5))
    plt.loglog(xb, yb, ".", markersize=2, label="baseline", color="black")
    plt.loglog(xh, yh, ".", markersize=2, label="heterogeneous α", color="red")
    plt.xlabel(f"{scale:g} × |ret|")
    plt.ylabel(f"P({scale:g}×|ret| ≥ x)")
    plt.title("Overlay — CCDF of absolute returns")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_overlay_acf(base: np.ndarray, het: np.ndarray, outpath: str, max_lag: int) -> None:
    """Overlay volatility autocorrelation on log-log axes."""
    Vb = base["abs_r"][np.isfinite(base["abs_r"])]
    Vh = het["abs_r"][np.isfinite(het["abs_r"])]

    acb = autocorrelation(Vb, max_lag=max_lag)
    ach = autocorrelation(Vh, max_lag=max_lag)

    lags_b = np.arange(1, len(acb), dtype=float)
    vals_b = acb[1:]
    mb = np.isfinite(vals_b) & (vals_b > 0)
    lags_b, vals_b = lags_b[mb], vals_b[mb]

    lags_h = np.arange(1, len(ach), dtype=float)
    vals_h = ach[1:]
    mh = np.isfinite(vals_h) & (vals_h > 0)
    lags_h, vals_h = lags_h[mh], vals_h[mh]

    plt.figure(figsize=(6, 5))
    plt.loglog(lags_b, vals_b, lw=1.0, label="baseline", color="black")
    plt.loglog(lags_h, vals_h, lw=1.0, label="heterogeneous α", color="red")
    plt.xlabel("lag")
    plt.ylabel("autocorrelation of |ret|")
    plt.title("Overlay — Volatility autocorrelation")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Overlay baseline vs heterogeneity figures.")
    ap.add_argument("--baseline", type=str, default=os.path.join("data", "lattice_data_results_50000.csv"),
                    help="Baseline CSV path.")
    ap.add_argument("--hetero", type=str, default=os.path.join("data", "heterogeneity_data_results_50000.csv"),
                    help="Heterogeneity CSV path.")
    ap.add_argument("--outdir", type=str, default="results", help="Output directory for figures.")
    ap.add_argument("--tmin", type=int, default=10_000)
    ap.add_argument("--tmax", type=int, default=20_000)
    ap.add_argument("--max_lag", type=int, default=2000)
    ap.add_argument("--scale", type=float, default=100.0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    base = ensure_abs_r(load_csv(args.baseline))
    het = ensure_abs_r(load_csv(args.hetero))

    # returns overlay needs r and t
    for col in ("t", "r"):
        if col not in base.dtype.names or col not in het.dtype.names:
            raise ValueError(f"Both CSVs must contain '{col}' for returns overlay.")

    plot_overlay_returns(base, het, os.path.join(args.outdir, "overlap_returns_timeseries.png"), args.tmin, args.tmax)
    plot_overlay_ccdf(base, het, os.path.join(args.outdir, "overlap_ccdf_abs_returns.png"), args.scale)
    plot_overlay_acf(base, het, os.path.join(args.outdir, "overlap_volatility_acf.png"), args.max_lag)

    print(f"Figures saved to '{args.outdir}/'")


if __name__ == "__main__":
    main()

