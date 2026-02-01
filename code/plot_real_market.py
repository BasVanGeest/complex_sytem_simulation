import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DEFAULT_OUTDIR = os.path.join(REPO_ROOT, "results")
DEFAULT_BTC_PATH = os.path.join(REPO_ROOT, "data", "BTC-USD.csv")
DEFAULT_SPX_PATH = os.path.join(REPO_ROOT, "data", "GSPC.csv")


def load_market_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df = df.rename(columns={"Price": "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"].notna()].copy()

    for col in ["Close", "High", "Low", "Open", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["Close"].notna()].copy()
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float).values
    r = np.log(close[1:]) - np.log(close[:-1])
    out = df.iloc[1:].copy()
    out["log_return"] = r
    out["abs_log_return"] = np.abs(r)
    return out


def ccdf(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = np.sort(x)
    n = x.size
    if n == 0:
        return np.array([]), np.array([])
    y = 1.0 - (np.arange(n) / n)
    return x, y


def plot_returns_timeseries(df_ret: pd.DataFrame, title: str, outpath: str):
    plt.figure(figsize=(12, 4))
    plt.plot(df_ret["Date"].values, df_ret["log_return"].values, lw=0.6)
    plt.title(title)
    plt.xlabel("time")
    plt.ylabel("log return")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_ccdf_abs_returns(df_ret: pd.DataFrame, title: str, outpath: str):
    x, y = ccdf(df_ret["abs_log_return"].values)
    if x.size == 0:
        raise ValueError("No valid returns to plot CCDF.")

    plt.figure(figsize=(6, 6))
    plt.loglog(x, y, lw=1.5)
    plt.title(title)
    plt.xlabel("|log return|")
    plt.ylabel("P(|return| > x)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser(
        description="Plot CCDF + time-series log returns for BTC and S&P500 CSVs."
    )
    ap.add_argument(
        "--btc",
        type=str,
        default=DEFAULT_BTC_PATH,
        help="Path to BTC CSV (default: data/BTC-USD.csv)",
    )
    ap.add_argument(
        "--spx",
        type=str,
        default=DEFAULT_SPX_PATH,
        help="Path to S&P500 CSV (default: data/GSPC.csv)",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default=DEFAULT_OUTDIR,
        help="Output folder for PNGs (default: <repo_root>/results)",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    btc_raw = load_market_csv(args.btc)
    btc = compute_log_returns(btc_raw)
    plot_returns_timeseries(
        btc,
        title="BTC-USD: log returns vs time",
        outpath=os.path.join(args.outdir, "btc_returns_timeseries.png"),
    )
    plot_ccdf_abs_returns(
        btc,
        title="BTC-USD: CCDF of |log returns| (log-log)",
        outpath=os.path.join(args.outdir, "btc_ccdf_abs_logreturns.png"),
    )

    spx_raw = load_market_csv(args.spx)
    spx = compute_log_returns(spx_raw)
    plot_returns_timeseries(
        spx,
        title="S&P 500 (^GSPC): log returns vs time",
        outpath=os.path.join(args.outdir, "spx_returns_timeseries.png"),
    )
    plot_ccdf_abs_returns(
        spx,
        title="S&P 500 (^GSPC): CCDF of |log returns| (log-log)",
        outpath=os.path.join(args.outdir, "spx_ccdf_abs_logreturns.png"),
    )

    print(f"[OK] Saved 4 figures to: {args.outdir}")


if __name__ == "__main__":
    main()