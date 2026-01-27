# run_baseline.py
import os
import argparse
import numpy as np

from tqdm import tqdm
from bornholdt_model import Bornholdt2D


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run ONE Bornholdt2D configuration (async Bornholdt 2001) and save ONE CSV used for all plots."
    )
    ap.add_argument("--out", type=str, default="data/intermediate.csv", help="Output CSV path.")
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--J", type=float, default=1.0)

    # Intermediary choice between (T=1.5, alpha=4) and (T=1.0, alpha=8)
    ap.add_argument("--T", type=float, default=1.5)
    ap.add_argument("--alpha", type=float, default=8.0)

    # Larger than 50k, far smaller than 1e6
    ap.add_argument("--steps", type=int, default=200_000, help="Total sweeps (1 sweep = 1 time step).")
    ap.add_argument("--burn_in", type=int, default=10_000)
    ap.add_argument("--thin", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=1e-6, help="Epsilon for log(|M|+eps) return definition.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    model = Bornholdt2D(L=args.L, J=args.J, alpha=args.alpha, T=args.T, seed=args.seed)

    cols = [
        "t", "M", "r", "abs_r",
        "C_mean", "chartist_frac", "fundamentalist_frac"
    ]

    with open(args.out, "w", encoding="utf-8") as f:
        # Minimal but informative header (comment lines)
        f.write("# Bornholdt2D single-run intermediate dataset\n")
        f.write(
            f"# params: L={args.L}, J={args.J}, alpha={args.alpha}, T={args.T}, seed={args.seed}, "
            f"steps={args.steps}, burn_in={args.burn_in}, thin={args.thin}, eps={args.eps}\n"
        )
        f.write(",".join(cols) + "\n")

        last_M = None
        kept = 0

        for step in tqdm(range(args.steps)):
            model.sweep()  # 1 sweep = N random-serial async single-site heat-bath updates; C updated immediately after each S

            if step < args.burn_in:
                continue
            if ((step - args.burn_in) % args.thin) != 0:
                continue

            t = step + 1
            M = float(model.M())

            # Bornholdt-style returns: log(|M|+eps)
            if last_M is None:
                r = np.nan
                abs_r = np.nan
            else:
                r = float(np.log(abs(M) + args.eps) - np.log(abs(last_M) + args.eps))
                abs_r = abs(r)
            last_M = M

            C_mean = float(np.mean(model.C))
            frac_ch = float(model.frac_chartist()) if hasattr(model, "frac_chartist") else float(np.mean(model.C == -1))
            frac_fu = 1.0 - frac_ch

            f.write(f"{t},{M},{r},{abs_r},{C_mean},{frac_ch},{frac_fu}\n")
            kept += 1

    print(f"[saved] {args.out}  (rows={kept})")


if __name__ == "__main__":
    main()