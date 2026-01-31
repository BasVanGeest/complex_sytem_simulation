import argparse
import os
import numpy as np

from bornholdt_heterogeneity import Bornholdt2D_heterogeneity


def main() -> None:
    """Run heterogeneous-alpha Bornholdt2D and save one CSV into ./data/."""
    ap = argparse.ArgumentParser(description="Run heterogeneous-alpha Bornholdt2D and save a CSV in ./data/.")
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--burn_in", type=int, default=10_000)
    ap.add_argument("--thin", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=1e-6)

    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--J", type=float, default=1.0)
    ap.add_argument("--T", type=float, default=1.5)

    ap.add_argument("--alpha_mean", type=float, default=8.0)
    ap.add_argument("--alpha_std", type=float, default=2.0)
    ap.add_argument("--alpha_min", type=float, default=1e-6)

    args = ap.parse_args()

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", f"heterogeneity_data_results_{args.steps}.csv")

    model = Bornholdt2D_heterogeneity(
        L=args.L,
        J=args.J,
        T=args.T,
        seed=args.seed,
        alpha_mean=args.alpha_mean,
        alpha_std=args.alpha_std,
        alpha_min=args.alpha_min,
    )

    cols = ["t", "M", "r", "abs_r", "C_mean", "chartist_frac", "fundamentalist_frac"]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Bornholdt2D heterogeneous-alpha run\n")
        f.write(
            f"# params: L={args.L}, J={args.J}, T={args.T}, seed={args.seed}, "
            f"alpha_mean={args.alpha_mean}, alpha_std={args.alpha_std}, alpha_min={args.alpha_min}, "
            f"steps={args.steps}, burn_in={args.burn_in}, thin={args.thin}, eps={args.eps}\n"
        )
        f.write(",".join(cols) + "\n")

        last_M = None
        kept = 0

        for step in range(args.steps):
            model.sweep()

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
            chartist_frac = float(model.frac_chartist())
            fundamentalist_frac = 1.0 - chartist_frac

            f.write(f"{t},{M},{r},{abs_r},{C_mean},{chartist_frac},{fundamentalist_frac}\n")
            kept += 1

    print(f"[saved] {out_path} (rows={kept})")


if __name__ == "__main__":
    main()