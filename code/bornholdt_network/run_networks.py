# run_network.py

import os
import argparse
import numpy as np

from networks import create_network
from model_network import BornholdtNetwork


def main():
    """CLI entry point to run the Bornholdt spin-market model on a generated network and write a CSV.

    Constructs a BA/ER/WS graph for N = L*L, initializes the BornholdtNetwork model, runs asynchronous
    updates for the requested number of steps (with optional burn-in and thinning), computes returns
    from changes in |magnetization| (using eps for numerical stability), and writes a time series CSV
    with: step, M, r, abs_r, mean_strategy_C, frac_chartist, frac_fundamentalist.
    """

    parser = argparse.ArgumentParser(description="Run Bornholdt model on a network")

    # --- Network / system size ---
    parser.add_argument("--topology", type=str, required=True,
                        choices=["BA", "ER", "WS"])
    parser.add_argument("--L", type=int, default=32,
                        help="Linear size (N = L*L)")
    parser.add_argument("--seed", type=int, default=0)

    # --- Network parameters ---
    parser.add_argument("--m", type=int, default=2,
                        help="BA: edges per new node")
    parser.add_argument("--k", type=int, default=4,
                        help="WS: nearest neighbors")
    parser.add_argument("--p", type=float, default=0.1,
                        help="ER: connection prob / WS: rewiring prob")

    # --- Model parameters ---
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--T", type=float, default=1.5)

    # --- Simulation parameters ---
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--burn_in", type=int, default=1000)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--eps", type=float, default=1e-6)

    parser.add_argument("--out", type=str,
                        default="data/network_run.csv")

    args = parser.parse_args()

    # ============================
    # Network construction (KEY)
    # ============================
    N = args.L * args.L

    G = create_network(
        net_type=args.topology,
        n=N,
        seed=args.seed,
        m=args.m,
        k=args.k,
        p=args.p
    )

    model = BornholdtNetwork(
        G=G,
        J=args.J,
        alpha=args.alpha,
        T=args.T,
        seed=args.seed
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with open(args.out, "w") as f:
        f.write("t,M,r,abs_r,C_mean,chartist_frac,fundamentalist_frac\n")

        last_M = None

        for step in range(args.steps):
            model.time_step()

            if step < args.burn_in:
                continue
            if (step - args.burn_in) % args.thin != 0:
                continue

            M = float(model.magnetization_M())

            if last_M is None:
                r = np.nan
                abs_r = np.nan
            else:
                r = np.log(abs(M) + args.eps) - np.log(abs(last_M) + args.eps)
                abs_r = abs(r)

            last_M = M

            C_mean = float(model.mean_strategy_C())
            frac_ch = float(np.mean(model.C == -1))
            frac_fu = 1.0 - frac_ch

            f.write(f"{step},{M},{r},{abs_r},{C_mean},{frac_ch},{frac_fu}\n")

    print(f"[OK] Saved to {args.out}")


if __name__ == "__main__":
    main()
