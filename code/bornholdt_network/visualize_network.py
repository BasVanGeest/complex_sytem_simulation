# visualize_network.py
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation

from model_network import BornholdtNetwork
from networks import create_network


def load_params(path: str = "params_network.json"):
    defaults = dict(
        net_type="WS",
        L=32,
        n=None,
        m=2,
        k=4,
        p=0.1,
        J=1.0,
        alpha=8.0,
        T=1.5,
        seed=0,
        steps_per_frame=1,
        n_frames=300,
        interval_ms=60,
        show="S",
        layout="spring",
        node_size=120,
        edge_alpha=0.25,
    )

    if not os.path.exists(path):
        return defaults

    with open(path, "r", encoding="utf-8") as f:
        user = json.load(f)

    out = defaults.copy()
    out.update({k: v for k, v in user.items() if k in out})
    return out


def main():
    parser = argparse.ArgumentParser(description="Visualize Bornholdt network dynamics")

    parser.add_argument("--topology", choices=["BA", "ER", "WS"], help="Network topology")
    parser.add_argument("--L", type=int, help="Linear size (N = L*L)")
    parser.add_argument("--n", type=int, help="Number of nodes (overrides L*L)")
    parser.add_argument("--m", type=int, help="BA parameter")
    parser.add_argument("--k", type=int, help="WS parameter")
    parser.add_argument("--p", type=float, help="ER / WS parameter")
    parser.add_argument("--seed", type=int, help="Random seed")

    args = parser.parse_args()

    # Load defaults / JSON
    p = load_params()

    # CLI overrides JSON/defaults
    if args.topology is not None:
        p["net_type"] = args.topology
    if args.L is not None:
        p["L"] = args.L
    if args.n is not None:
        p["n"] = args.n
    if args.m is not None:
        p["m"] = args.m
    if args.k is not None:
        p["k"] = args.k
    if args.p is not None:
        p["p"] = args.p
    if args.seed is not None:
        p["seed"] = args.seed

    net_type = p["net_type"].upper()
    seed = int(p["seed"])

    # N = n or L*L
    if p["n"] is None:
        n = int(p["L"]) ** 2
    else:
        n = int(p["n"])

    # Build network (same factory as run_network)
    G = create_network(
        net_type=net_type,
        n=n,
        seed=seed,
        m=int(p["m"]),
        k=int(p["k"]),
        p=float(p["p"]),
    )

    model = BornholdtNetwork(
        G,
        J=float(p["J"]),
        alpha=float(p["alpha"]),
        T=float(p["T"]),
        seed=seed,
    )

    steps_per_frame = int(p["steps_per_frame"])
    n_frames = int(p["n_frames"])
    interval_ms = int(p["interval_ms"])

    state = {"show": p["show"].upper()}
    t = {"step": 0}

    fig, ax = plt.subplots()
    ax.axis("off")

    if p["layout"] == "circular":
        pos = nx.circular_layout(G)
    elif p["layout"] == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=seed)

    def node_colors():
        spins = model.S if state["show"] == "S" else model.C
        return np.where(spins == 1, "red", "blue")

    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors(),
        node_size=int(p["node_size"]),
        ax=ax,
    )
    nx.draw_networkx_edges(G, pos, alpha=float(p["edge_alpha"]), ax=ax)

    txt = ax.text(
        0.02, 0.98, "", transform=ax.transAxes,
        va="top", ha="left", fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    def on_key(event):
        if event.key == "t":
            state["show"] = "C" if state["show"] == "S" else "S"
            nodes.set_facecolor(node_colors())
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_):
        for _ in range(steps_per_frame):
            model.time_step()
            t["step"] += 1

        nodes.set_facecolor(node_colors())

        M = model.magnetization_M()
        txt.set_text(
            f"t={t['step']}\n"
            f"net={net_type}, N={model.N}\n"
            f"M={M:.3f}"
        )
        return nodes, txt

    ani = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=interval_ms,
    )
    plt.show()


if __name__ == "__main__":
    main() 