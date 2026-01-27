# run_heterogeneity.py
import os
import argparse
import numpy as np

from bornholdt_heterogeneity import Bornholdt2D_heterogeneity


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run ONE Bornholdt2D heterogeneous-alpha configuration "
            "(quenched alpha_ij ~ Normal) and save ONE CSV used for all plots."
        )
    )

    ap.add_argument("--out", type=str, default="data/intermediate_heterogeneity.csv",
                    help="Output CSV path.")
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--J", type=float, default=1.0)

    # Temperature as in baseline comparison
    ap.add_argument("--T", type=float, default=1.5)

    # --- heterogeneous alpha parameters ---
    ap.add_argument("--alpha_mean", type=float, default=8.0,
                    help="Mean of Normal distribution for quenched alpha_ij.")
    ap.add_argument("--alpha_std", type=float, default=2.0,
                    help="Std dev of Normal distribution for quenched alpha_ij.")
    ap.add_argument("--alpha_min", type=float, default=1e-6,
                    help="Minimum alpha value (clipping to enforce positivity).")

    # Simulation length
    ap.add_argument("--steps", type=int, default=200_000,
                    help="Total sweeps (1 sweep = 1 time step).")
    ap.add_argument("--burn_in", type=int, default=10_000)
    ap.add_argument("--thin", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=1e-6,
                    help="Epsilon for log(|M|+eps) return definition.")

    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    model = Bornholdt2D_heterogeneity(
        L=args.L,
        J=args.J,
        T=args.T,
        seed=args.seed,
        alpha_mean=args.alpha_mean,
        alpha_std=args.alpha_std,
        alpha_min=args.alpha_min,
    )

    cols = [
        "t", "M", "r", "abs_r",
        "C_mean", "chartist_frac", "fundamentalist_frac",
    ]

    with open(args.out, "w", encoding="utf-8") as f:
        # Informative header
        f.write("# Bornholdt2D heterogeneous-alpha single-run dataset\n")
        f.write(
            "# params: "
            f"L={args.L}, J={args.J}, T={args.T}, seed={args.seed}, "
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
            if ((step - args.burn_in) % args.thin) != 0:
                continue

            t = step + 1
            M = float(model.M())

            # Bornholdt-style returns
            if last_M is None:
                r = np.nan
                abs_r = np.nan
            else:
                r = float(
                    np.log(abs(M) + args.eps) -
                    np.log(abs(last_M) + args.eps)
                )
                abs_r = abs(r)

            last_M = M

            C_mean = float(np.mean(model.C))
            frac_ch = float(model.frac_chartist())
            frac_fu = 1.0 - frac_ch

            f.write(
                f"{t},{M},{r},{abs_r},{C_mean},{frac_ch},{frac_fu}\n"
            )
            kept += 1

    print(f"[saved] {args.out}  (rows={kept})")


if __name__ == "__main__":
    main()