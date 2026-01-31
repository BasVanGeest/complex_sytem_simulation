# run_baseline.py
import argparse
import os
import numpy as np

from bornholdt_model import Bornholdt2D


def main() -> None:
    """Run one Bornholdt2D baseline simulation and save one CSV in ./data/."""
    ap = argparse.ArgumentParser(
        description="Run Bornholdt2D baseline and save a CSV in ./data/."
    )
    ap.add_argument("--steps", type=int, default=50_000,
                    help="Total time steps (underscores allowed: 50_000).")
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--J", type=float, default=1.0)
    ap.add_argument("--T", type=float, default=1.5)
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--burn_in", type=int, default=5_000)
    ap.add_argument("--thin", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=1e-6,
                    help="Epsilon for log(|M|+eps) return definition.")
    args = ap.parse_args()

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", f"lattice_data_results_{args.steps}.csv")

    model = Bornholdt2D(L=args.L, J=args.J, alpha=args.alpha, T=args.T, seed=args.seed)

    cols = ["t", "M", "r", "abs_r", "C_mean", "chartist_frac", "fundamentalist_frac"]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Bornholdt2D baseline run\n")
        f.write(
            f"# params: L={args.L}, J={args.J}, T={args.T}, alpha={args.alpha}, seed={args.seed}, "
            f"steps={args.steps}, burn_in={args.burn_in}, thin={args.thin}, eps={args.eps}\n"
        )
        f.write(",".join(cols) + "\n")

        last_M = None
        kept = 0

        for step in range(args.steps):
            model.sweep()  # one Monte Carlo sweep

            if step < args.burn_in:
                continue
            if (step - args.burn_in) % args.thin != 0:
                continue

            t = step + 1
            M = float(model.M())

            if last_M is None:
                r = np.nan
                abs_r = np.nan
            else:
                r = float(np.log(abs(M) + args.eps) - np.log(abs(last_M) + args.eps))
                abs_r = abs(r)
            last_M = M

            C_mean = float(np.mean(model.C))
            chartist_frac = float(np.mean(model.C == -1))
            fundamentalist_frac = 1.0 - chartist_frac

            f.write(f"{t},{M},{r},{abs_r},{C_mean},{chartist_frac},{fundamentalist_frac}\n")
            kept += 1

    print(f"[saved] {out_path} (rows={kept})")


if __name__ == "__main__":
    main()