import os
import sys

# allow imports from parent folder (Bornholdt2D+run+plots)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import json
import numpy as np

from add_run_return import Bornholdt2D


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def run_and_save(out_dir: str,
                 L=32, J=1.0, alpha=4.0, T=1.5, seed=0,
                 n_sweeps=50000, burn_in=5000, thin=1,
                 return_mode="log_absM", eps=1e-6,
                 update_C=True):

    results_dir = os.path.join(out_dir, "results")
    ensure_dir(os.path.join(results_dir, "paper_figures"))

    model = Bornholdt2D(L=L, J=J, alpha=alpha, T=T, seed=seed, update_C=update_C)

    # Use the class' own run() method so it matches your baseline exactly
    series = model.run(
        n_sweeps=n_sweeps,
        burn_in=burn_in,
        thin=thin,
        return_mode=return_mode,
        eps=eps,
    )

    # Save CSV in the same format your plot script expects
    csv_path = os.path.join(results_dir, "timeseries.csv")
    header = "M,C_mean,r,abs_r\n"
    M = np.asarray(series["M"])
    C = np.asarray(series["C_mean"])
    r = np.asarray(series["r"])
    abs_r = np.asarray(series["abs_r"])

    # Align: r(t) corresponds to transition M(t-1)->M(t), so use M[1:]
    n = min(len(r), len(abs_r), len(M) - 1, len(C) - 1)

    data = np.column_stack([M[1:1+n], C[1:1+n], r[:n], abs_r[:n]])


    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(header)
        np.savetxt(f, data, delimiter=",")

    # Save params used (reproducibility)
    params_used = dict(
        L=L, J=J, alpha=alpha, T=T, seed=seed,
        n_sweeps=n_sweeps, burn_in=burn_in, thin=thin,
        return_mode=return_mode, eps=eps,
        update_C=update_C
    )
    with open(os.path.join(results_dir, "params_used.json"), "w", encoding="utf-8") as f:
        json.dump(params_used, f, indent=2)

    print(f"[OK] wrote {csv_path}")


def main():
    # Match your baseline params here (edit if your baseline uses different ones)
    base = dict(
        L=32, J=1.0, alpha=4.0, T=1.5, seed=0,
        n_sweeps=50000, burn_in=5000, thin=1,
        return_mode="log_absM", eps=1e-6
    )

    # Control A: remove global coupling
    run_and_save("control_experiment/alpha0", **{**base, "alpha": 0.0, "update_C": True})

    # Control B: freeze strategy switching
    run_and_save("control_experiment/freezeC", **{**base, "alpha": 4.0, "update_C": False})


if __name__ == "__main__":
    main()
