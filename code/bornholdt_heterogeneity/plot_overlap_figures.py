# plot_overlap_figures.py
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO


# -----------------------------
# CSV utilities
# -----------------------------
def load_csv_with_comments(path: str):
    """
    Loads a CSV that may start with comment lines '# ...'
    Returns: (data_structured_array, comment_lines_list)
    """
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [ln.rstrip("\n") for ln in f.readlines() if ln.strip()]

    comments = [ln for ln in raw_lines if ln.lstrip().startswith("#")]
    lines = [ln for ln in raw_lines if not ln.lstrip().startswith("#")]

    if len(lines) < 2:
        raise ValueError(f"CSV {path} doesn't look like [header] + [data].")

    header = [c.strip() for c in lines[0].split(",")]
    data_lines = lines[1:]

    buf = StringIO("\n".join(data_lines))
    data = np.genfromtxt(buf, delimiter=",", names=header, dtype=float)

    return data, comments


def normalize_fieldnames(data):
    """
    Some runs may use slightly different column names.
    This function just ensures we can access the needed fields robustly.
    """
    names = set(data.dtype.names)

    # need at least r or abs_r
    required_any = {"abs_r", "r"}
    if not (required_any & names):
        raise ValueError(f"Expected at least one of {required_any}, found {data.dtype.names}")

    # If abs_r missing but r exists, compute abs_r
    if "abs_r" not in names and "r" in names:
        abs_r = np.abs(data["r"])
        data = append_field(data, "abs_r", abs_r)

    return data


def append_field(data, name, values):
    """Append a new float field to a structured array."""
    values = np.asarray(values, dtype=float)
    newdtype = data.dtype.descr + [(name, "f8")]
    out = np.empty(data.shape, dtype=newdtype)
    for n in data.dtype.names:
        out[n] = data[n]
    out[name] = values
    return out


# -----------------------------
# Stats utilities
# -----------------------------
def ccdf(x):
    """Return sorted x and CCDF y = P(X >= x)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0]  # avoid log(0)
    x = np.sort(x)
    n = x.size
    if n == 0:
        return np.array([]), np.array([])
    y = 1.0 - (np.arange(n) / n)
    return x, y


def autocorrelation(x, max_lag):
    """
    Normalized autocorrelation:
      C(lag) = <(x_t-mean)(x_{t+lag}-mean)> / var
    """
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


def window_by_t(data, t_min, t_max):
    t = data["t"]
    m = np.isfinite(t) & (t >= t_min) & (t <= t_max)
    return data[m]


# -----------------------------
# Parse heterogeneity params from comment lines
# -----------------------------
def parse_params_from_comments(comment_lines):
    """
    Tries to parse a line like:
      # params: L=32, J=1.0, T=1.5, seed=0, alpha_mean=8.0, alpha_std=2.0, alpha_min=1e-06, steps=200000, ...
    Returns dict with whatever it finds.
    """
    txt = "\n".join(comment_lines)

    kv = re.findall(r"(\w+)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", txt)
    out = {}
    for k, v in kv:
        if k in {"L", "seed", "steps", "burn_in", "thin"}:
            try:
                out[k] = int(float(v))
            except Exception:
                out[k] = float(v)
        else:
            out[k] = float(v)
    return out


def reconstruct_alpha_grid_from_params(params):
    """
    Rebuild alpha_ij:
      alpha_ij ~ Normal(alpha_mean, alpha_std), clipped to alpha_min
    Uses (seed) and (L) if available.
    """
    L = int(params.get("L", 32))
    seed = int(params.get("seed", 0))
    alpha_mean = float(params.get("alpha_mean", 8.0))
    alpha_std = float(params.get("alpha_std", 2.0))
    alpha_min = float(params.get("alpha_min", 1e-6))

    rng = np.random.default_rng(seed)
    alpha_grid = rng.normal(loc=alpha_mean, scale=alpha_std, size=(L, L))
    alpha_grid = np.clip(alpha_grid, alpha_min, None)
    return alpha_grid


# -----------------------------
# Plotting
# -----------------------------
def plot_overlay_returns(base, het, outpath, t_min=10000, t_max=20000):
    """
    Overlay Fig.2-style returns r(t): baseline vs heterogeneity on the same time window.
    Uses intersection of available t values to avoid mismatches due to burn-in.
    """
    b = window_by_t(base, t_min, t_max)
    h = window_by_t(het, t_min, t_max)

    if b.size == 0 or h.size == 0:
        raise ValueError(
            f"No overlap data for returns in [{t_min},{t_max}]. "
            f"Baseline rows={b.size}, Hetero rows={h.size}."
        )

    # build maps t -> r for safe alignment
    bt = b["t"]
    br = b["r"] if "r" in b.dtype.names else np.nan * np.ones_like(bt)

    ht = h["t"]
    hr = h["r"] if "r" in h.dtype.names else np.nan * np.ones_like(ht)

    # intersection times
    common_t = np.intersect1d(bt[np.isfinite(bt)], ht[np.isfinite(ht)])
    common_t = common_t[(common_t >= t_min) & (common_t <= t_max)]

    if common_t.size == 0:
        raise ValueError(f"No common t values in [{t_min},{t_max}] between baseline and hetero.")

    # index via dict (fast enough for these sizes)
    b_map = {int(t): r for t, r in zip(bt, br)}
    h_map = {int(t): r for t, r in zip(ht, hr)}

    r_base = np.array([b_map.get(int(t), np.nan) for t in common_t], dtype=float)
    r_het = np.array([h_map.get(int(t), np.nan) for t in common_t], dtype=float)

    mb = np.isfinite(r_base)
    mh = np.isfinite(r_het)

    plt.figure(figsize=(10, 3))
    plt.plot(common_t[mb], r_base[mb], lw=0.6, color="black", label="baseline")
    plt.plot(common_t[mh], r_het[mh], lw=0.6, color="red", alpha=0.8, label="heterogeneous α")
    plt.xlabel("time (sweeps)")
    plt.ylabel("return r(t)")
    plt.title("Overlay — Returns time series (Fig. 2 style)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_overlay_ccdf(abs_r_base, abs_r_het, outpath, scale=100.0):
    Xb = scale * abs_r_base[np.isfinite(abs_r_base)]
    Xh = scale * abs_r_het[np.isfinite(abs_r_het)]

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


def plot_overlay_acf(abs_r_base, abs_r_het, outpath, max_lag=2000):
    Vb = abs_r_base[np.isfinite(abs_r_base)]
    Vh = abs_r_het[np.isfinite(abs_r_het)]

    acb = autocorrelation(Vb, max_lag=max_lag)
    ach = autocorrelation(Vh, max_lag=max_lag)

    # Yamano-style: start at T=1 and only positive values for log-log.
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
    plt.xlabel("Time (lag T)")
    plt.ylabel("Autocorrelation function of volatility")
    plt.title("Overlay — Volatility autocorrelation (log–log)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_alpha_hist(alpha_grid, outpath):
    a = alpha_grid.ravel()
    a = a[np.isfinite(a)]

    plt.figure(figsize=(6, 4))
    plt.hist(a, bins=40, edgecolor="black")
    plt.xlabel("α")
    plt.ylabel("count")
    plt.title("Heterogeneity check — histogram of αᵢⱼ")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    baseline_path = os.path.join("data", "intermediate.csv")
    hetero_path = os.path.join("data", "intermediate_heterogeneity.csv")

    outdir = "paper_figures"
    os.makedirs(outdir, exist_ok=True)

    base, _ = load_csv_with_comments(baseline_path)
    het, het_comments = load_csv_with_comments(hetero_path)

    base = normalize_fieldnames(base)
    het = normalize_fieldnames(het)

    # --- Plot 0: Fig 2 overlay (returns time series) ---
    plot_overlay_returns(
        base=base,
        het=het,
        outpath=os.path.join(outdir, "overlap_fig2_returns_timeseries.png"),
        t_min=10000,
        t_max=20000,
    )

    # --- Plot 1: CCDF overlay ---
    plot_overlay_ccdf(
        abs_r_base=base["abs_r"],
        abs_r_het=het["abs_r"],
        outpath=os.path.join(outdir, "overlap_fig3_ccdf_abs_returns.png"),
        scale=100.0,
    )

    # --- Plot 2: ACF overlay ---
    plot_overlay_acf(
        abs_r_base=base["abs_r"],
        abs_r_het=het["abs_r"],
        outpath=os.path.join(outdir, "overlap_fig4_volatility_acf.png"),
        max_lag=2000,
    )

    # --- Plot 3: alpha histogram ---
    params = parse_params_from_comments(het_comments)
    alpha_grid = reconstruct_alpha_grid_from_params(params)
    plot_alpha_hist(
        alpha_grid=alpha_grid,
        outpath=os.path.join(outdir, "heterogeneity_alpha_hist.png"),
    )

    print("[saved]")
    print(f" - {os.path.join(outdir, 'overlap_fig2_returns_timeseries.png')}")
    print(f" - {os.path.join(outdir, 'overlap_fig3_ccdf_abs_returns.png')}")
    print(f" - {os.path.join(outdir, 'overlap_fig4_volatility_acf.png')}")
    print(f" - {os.path.join(outdir, 'heterogeneity_alpha_hist.png')}")


if __name__ == "__main__":
    main()

