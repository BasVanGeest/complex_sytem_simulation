import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation

from model_network import BornholdtNetwork
from networks import create_network


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize Bornholdt dynamics on a network (ER/BA/WS).")
    ap.add_argument("--topology", choices=["ER", "BA", "WS"], required=True,
                    help="Network topology to simulate.")
    ap.add_argument("--L", type=int, default=16, help="N = L*L nodes (default: 16 -> 256 nodes).")
    ap.add_argument("--seed", type=int, default=0)

    # Network parameters
    ap.add_argument("--m", type=int, default=2, help="BA: edges per new node.")
    ap.add_argument("--k", type=int, default=4, help="WS: nearest neighbors (even).")
    ap.add_argument("--p", type=float, default=0.01, help="ER: p, WS: rewiring p.")

    # Model parameters
    ap.add_argument("--J", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--T", type=float, default=1.5)

    # Animation parameters
    ap.add_argument("--steps-per-frame", type=int, default=1)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--interval-ms", type=int, default=60)
    ap.add_argument("--layout", choices=["spring", "circular", "kamada_kawai"], default="spring")
    ap.add_argument("--node-size", type=int, default=80)
    ap.add_argument("--edge-alpha", type=float, default=0.2)
    ap.add_argument("--show", choices=["S", "C"], default="S", help="Initial view: S or C.")
    args = ap.parse_args()

    topo = args.topology.upper()
    N = args.L * args.L

    G = create_network(net_type=topo, n=N, seed=args.seed, m=args.m, k=args.k, p=args.p)
    model = BornholdtNetwork(G, J=args.J, alpha=args.alpha, T=args.T, seed=args.seed)

    # Layout
    if args.layout == "circular":
        pos = nx.circular_layout(G)
    elif args.layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=args.seed)

    state = {"show": args.show}
    t = {"step": 0}

    fig, ax = plt.subplots()
    ax.axis("off")

    edges = nx.draw_networkx_edges(G, pos, ax=ax, alpha=args.edge_alpha)

    def colors():
        spins = model.S if state["show"] == "S" else model.C
        return np.where(spins == 1, "red", "blue")

    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_color=colors(),
        node_size=args.node_size,
        ax=ax,
    )

    txt = ax.text(
        0.02, 0.98, "", transform=ax.transAxes,
        va="top", ha="left", fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    def on_key(event):
        if event.key == "t":
            state["show"] = "C" if state["show"] == "S" else "S"
            nodes.set_facecolor(colors())
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_):
        for _ in range(args.steps_per_frame):
            model.time_step()  # one sweep
            t["step"] += 1

        nodes.set_facecolor(colors())

        txt.set_text(
            f"topology={topo}, N={model.N}\n"
            f"view={state['show']} (press 't' to toggle)\n"
            f"t={t['step']}, M={model.M():.3f}"
        )
        return nodes, txt, edges

    _ani = FuncAnimation(fig, update, frames=args.frames, interval=args.interval_ms, blit=False)
    plt.show()


if __name__ == "__main__":
    main()
