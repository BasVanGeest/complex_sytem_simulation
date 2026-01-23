import os
import json
import numpy as np

from add_run_return import Bornholdt2D


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    # ===== parameters (paper-style baseline) =====
    L = 32
    J = 1.0
    alpha = 4.0
    T = 1.5
    seed = 0

    # ===== run control =====
    n_sweeps = 50_000
    burn_in = 10_000
    thin = 1
    return_mode = "log_absM"   # or "diffM"
    eps = 1e-6

    # ===== output =====
    out_dir = os.path.join("results")
    ensure_dir(out_dir)

    params = {
        "L": L, "J": J, "alpha": alpha, "T": T, "seed": seed,
        "n_sweeps": n_sweeps, "burn_in": burn_in, "thin": thin,
        "return_mode": return_mode, "eps": eps,
    }
    with open(os.path.join(out_dir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    # ===== simulate =====
    model = Bornholdt2D(L=L, J=J, alpha=alpha, T=T, seed=seed)
    ts = model.run(n_sweeps=n_sweeps, burn_in=burn_in, thin=thin, return_mode=return_mode, eps=eps)

    M = ts["M"]
    r = ts["r"]
    abs_r = ts["abs_r"]

    # Align lengths for saving:
    # M has length K; r/abs_r have length K-1
    # We'll save t index for M and a separate t index for r.
    k = len(M)
    t_M = np.arange(k)
    t_r = np.arange(k - 1)

    # Save timeseries.csv
    # Columns: t_M, M, t_r, r, abs_r  (padded with empty for last row if needed)
    csv_path = os.path.join(out_dir, "timeseries.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("t_M,M,t_r,r,abs_r\n")
        for i in range(k):
            if i < k - 1:
                f.write(f"{t_M[i]},{M[i]},{t_r[i]},{r[i]},{abs_r[i]}\n")
            else:
                f.write(f"{t_M[i]},{M[i]},,,\n")

    print(f"Saved: {csv_path}")
    print(f"M samples: {len(M)} | r samples: {len(r)}")


if __name__ == "__main__":
    main()
