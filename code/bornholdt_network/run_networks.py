import argparse
import os
import numpy as np

from networks import create_network
from model_network import BornholdtNetwork


def run_one(topology: str, args) -> str:
    N = args.L * args.L
    G = create_network(
        net_type=topology,
        n=N,
        seed=args.seed,
        m=args.m,
        k=args.k,
        p=args.p,
    )

    model = BornholdtNetwork(G=G, J=args.J, alpha=args.alpha, T=args.T, seed=args.seed)

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", f"{topology}_data_results_{args.steps}.csv")

    cols = ["t", "M", "r", "abs_r", "C_mean", "chartist_frac", "fundamentalist_frac"]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Bornholdt network run\n")
        f.write(
            f"# topology={topology}, N={N}, L={args.L}, J={args.J}, alpha={args.alpha}, T={args.T}, seed={args.seed}, "
            f"steps={args.steps}, burn_in={args.burn_in}, thin={args.thin}, eps={args.eps}, "
            f"m={args.m}, k={args.k}, p={args.p}\n"
        )
        f.write(",".join(cols) + "\n")

        last_M = None
        kept = 0

        for step in range(args.steps):
            model.time_step()  # or model.sweep() if you rename it

            if step < args.burn_in:
                continue
            if (step - args.burn_in) % args.thin != 0:
                continue

            t = step + 1
            M = float(model.magnetization_M())

            if last_M is None:
                r = np.nan
                abs_r = np.nan
            else:
                r = float(np.log(abs(M) + args.eps) - np.log(abs(last_M) + args.eps))
                abs_r = abs(r)
            last_M = M

            C_mean = float(np.mean(model.C))
            frac_ch = float(np.mean(model.C == -1))
            frac_fu = 1.0 - frac_ch

            f.write(f"{t},{M},{r},{abs_r},{C_mean},{frac_ch},{frac_fu}\n")
            kept += 1

    print(f"[saved] {out_path} (rows={kept})")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Bornholdt model on ER/BA/WS networks and save CSVs in ./data/.")
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--burn_in", type=int, default=1_000)
    ap.add_argument("--thin", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=1e-6)

    ap.add_argument("--L", type=int, default=32, help="N = L*L nodes")
    ap.add_argument("--J", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--T", type=float, default=1.5)

    # network params (used depending on topology)
    ap.add_argument("--m", type=int, default=2, help="BA: edges per new node")
    ap.add_argument("--k", type=int, default=4, help="WS: nearest neighbors (must be even)")
    ap.add_argument("--p", type=float, default=0.01, help="ER: p, WS: rewiring p")

    # optional: restrict to one topology
    ap.add_argument("--topology", choices=["ER", "BA", "WS"], default=None,
                    help="If set, run only this topology. Otherwise runs ER, BA, and WS.")

    args = ap.parse_args()

    to_run = [args.topology] if args.topology else ["ER", "BA", "WS"]

    for topo in to_run:
        run_one(topo, args)


if __name__ == "__main__":
    main()