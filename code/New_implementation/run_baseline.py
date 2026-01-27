# run_baseline.py
import os
import argparse
import numpy as np

from bornholdt_model import Bornholdt2D
from model_network import BornholdtNetwork
from networks import create_network

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))



def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _make_model(
    *,
    topology: str,
    L: int,
    n: int | None,
    J: float,
    alpha: float,
    T: float,
    seed: int,
    net_params: dict,
):
    """
    Central model factory so the rest of the pipeline stays unchanged.

    topology:
      - "lattice" is Bornholdt2D (paper baseline)
      - "BA"/"ER"/"WS" is BornholdtNetwork on that topology
    """
    topo = topology.strip().upper()

    if topo == "LATTICE":
        return Bornholdt2D(L=L, J=J, alpha=alpha, T=T, seed=seed)

    # network case
    N = int(n) if (n is not None) else int(L * L)
    G = create_network(net_type=topo, n=N, seed=seed, **net_params)
    return BornholdtNetwork(G, J=J, alpha=alpha, T=T, seed=seed)


def run_and_collect(
    *,
    topology: str,
    L: int,
    n: int | None,
    net_params: dict,
    J: float,
    alpha: float,
    T: float,
    seed: int,
    n_steps: int,
    burn_in: int,
    thin: int,
    eps: float,
) -> dict:
    """
    Run Bornholdt2D for n_steps "paper time steps" (each = sweep + synchronous C update),
    then collect time series needed for Figs. 2 to 5:
      - M(t)
      - returns r(t) from M(t)
      - abs returns |r(t)|
      - C_mean(t) and strategy fractions (for Fig. 5)

    Update:
    Run either Bornholdt2D or BornholdtNetwork for n_steps (each = sweep + synchronous C update) and collect time series needed for Figs. 2-5
    Same output format regardless of topology.
    """
    
    model = _make_model(
        topology=topology,
        L=L,
        n=n,
        J=J,
        alpha=alpha,
        T=T,
        seed=seed,
        net_params=net_params,
    )

    t_list: list[int] = []
    M_list: list[float] = []
    Cmean_list: list[float] = []
    chartist_frac_list: list[float] = []
    fundamentalist_frac_list: list[float] = []

    for t in range(n_steps):
        model.time_step()

        if t < burn_in:
            continue
        if ((t - burn_in) % thin) != 0:
            continue

        M = model.magnetization_M()
        C_mean = model.mean_strategy_C()

        # Convention (document in report!):
        # here we store BOTH mean(C) and fractions of C=+1 / C=-1 so you can plot whichever matches.
        frac_C_plus = float(np.mean(model.C == 1))
        frac_C_minus = float(np.mean(model.C == -1))

        t_list.append(t + 1)  # time step index starting at 1 (after applying the step)
        M_list.append(float(M))
        Cmean_list.append(float(C_mean))
        fundamentalist_frac_list.append(frac_C_plus)
        chartist_frac_list.append(frac_C_minus)

    t_arr = np.array(t_list, dtype=int)
    M_arr = np.array(M_list, dtype=float)
    Cmean_arr = np.array(Cmean_list, dtype=float)
    chartist_frac_arr = np.array(chartist_frac_list, dtype=float)
    fundamentalist_frac_arr = np.array(fundamentalist_frac_list, dtype=float)

    #Returns from magnetization 
    #call the model class helper so lattice+network use the same definition
    r = model.__class__.returns_from_magnetization(M_arr, eps=eps)
    abs_r = np.abs(r)

    # Align lengths: r has length len(M)-1. We'll pad the first entry with NaN.
    r_padded = np.empty_like(M_arr, dtype=float)
    abs_r_padded = np.empty_like(M_arr, dtype=float)
    r_padded[:] = np.nan
    abs_r_padded[:] = np.nan
    if r.size > 0:
        r_padded[1:] = r
        abs_r_padded[1:] = abs_r

    #add topology + network params into the metadata header
    params = dict(
        topology=topology,
        L=L,
        n=(int(n) if n is not None else None),
        net_params=net_params,
        J=J,
        alpha=alpha,
        T=T,
        seed=seed,
        n_steps=n_steps,
        burn_in=burn_in,
        thin=thin,
        eps=eps,
    )

    return dict(
        t=t_arr,
        M=M_arr,
        r=r_padded,
        abs_r=abs_r_padded,
        C_mean=Cmean_arr,
        chartist_frac=chartist_frac_arr,
        fundamentalist_frac=fundamentalist_frac_arr,
        params=params,
    )


def save_csv(out_path: str, data: dict) -> None:
    """
    Save time series to CSV with a clear header.

    Update:
    Header is now topology-agnostic
    """
    cols = ["t", "M", "r", "abs_r", "C_mean", "chartist_frac", "fundamentalist_frac"]
    arr = np.column_stack([data[c] for c in cols])

    header_lines = [
        "Bornholdt baseline run (lattice or network)",
        f"params: {data['params']}",
        "columns: " + ",".join(cols),
    ]
    header = "\n".join(header_lines)

    np.savetxt(out_path, arr, delimiter=",", header=header, comments="")
    print(f"[saved] {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Bornholdt baseline (Bornholdt 2001) on lattice OR network and save CSVs for Figs. 2–5."
    )
    parser.add_argument("--out_dir", type=str, default="data", help="Output directory for CSV files.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for both runs.")
    parser.add_argument("--burn_in", type=int, default=10_000, help="Burn-in time steps before recording.")
    parser.add_argument("--thin", type=int, default=1, help="Record every 'thin' steps after burn-in.")
    parser.add_argument("--eps", type=float, default=1e-6, help="Epsilon for log(|M|+eps) return definition.")

    #topology selector + size controls added
    parser.add_argument("--topology", type=str, default="lattice",
                        help="Topology: lattice, BA, ER, or WS.")
    parser.add_argument("--L", type=int, default=32,
                        help="Lattice size (lattice) OR size proxy for networks (N=L*L if --n not given).")
    parser.add_argument("--n", type=int, default=None,
                        help="Network only: number of nodes. If omitted, uses N=L*L.")

    #network parameters added
    parser.add_argument("--m", type=int, default=2, help="BA only: edges per new node (avg degree ~ 2m).")
    parser.add_argument("--k", type=int, default=4, help="WS only: nearest neighbors (should be even).")
    parser.add_argument("--p", type=float, default=0.1, help="ER/WS: connection/rewiring probability.")

    parser.add_argument("--steps_2_5", type=int, default=50_000,
                        help="Total steps for the (T=1.5, alpha=4) run (Figs. 2 & 5).")
    parser.add_argument("--steps_3_4", type=int, default=1_000_000,
                        help="Total steps for the (T=1.0, alpha=8) run (Figs. 3 & 4).")

    args = parser.parse_args()
    out_dir = args.out_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(SCRIPT_DIR, out_dir)

    ensure_dir(out_dir)

    #collect network params in one dict
    net_params = dict(m=int(args.m), k=int(args.k), p=float(args.p))

    #Run A: parameters used for Fig. 1 caption and Fig. 2,  Fig. 5
    data_2_5 = run_and_collect(
        topology=args.topology,
        L=int(args.L),
        n=args.n,
        net_params=net_params,
        J=1.0,
        alpha=4.0,
        T=1.5,
        seed=args.seed,
        n_steps=args.steps_2_5,
        burn_in=min(args.burn_in, args.steps_2_5 - 1),
        thin=args.thin,
        eps=args.eps,
    )
    save_csv(os.path.join(out_dir, "data_fig_2_5.csv"), data_2_5)

    #Run B: Fig. 3 caption explicitly uses T=1.0, alpha=8, sampled over 10^6 sweeps
    data_3_4 = run_and_collect(
        topology=args.topology,
        L=int(args.L),
        n=args.n,
        net_params=net_params,
        J=1.0,
        alpha=8.0,
        T=1.0,
        seed=args.seed,
        n_steps=args.steps_3_4,
        burn_in=min(args.burn_in, args.steps_3_4 - 1),
        thin=args.thin,
        eps=args.eps,
    )
    save_csv(os.path.join(out_dir, "data_fig_3_4.csv"), data_3_4)


if __name__ == "__main__":
    main()
