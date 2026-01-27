import os
import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation

from model_network import BornholdtNetwork
from networks import create_network


def load_params(path="params_network.json"):
    # mirror lattice visualizer style so teammates have one mental model
    defaults = dict(
        net_type="WS",   # BA, ER, WS
        L=32,
        n=None,          # if None, uses N=L*L like the paper lattice size

        m=2,             # BA only
        k=4,             # WS only
        p=0.1,           # ER/WS

        J=1.0,
        alpha=4.0,
        T=1.5,
        seed=0,

        steps_per_frame=1, 
        n_frames=300,
        interval_ms=60,
        show="S",           # "S" or "C"
        layout="spring",    # spring / circular / kamada_kawai
    )

    if not os.path.exists(path):
        return defaults

    with open(path, "r", encoding="utf-8") as f:
        user = json.load(f)

    out = defaults.copy()
    for k, v in user.items():
        if k in out:
            out[k] = v
    return out


def main():
    p = load_params()

    net_type = str(p["net_type"]).upper().strip()
    seed = int(p["seed"])

    n = p.get("n", None)
    if n is None:
        L = int(p["L"])
        n = L * L
    else:
        n = int(n)

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

    state = {"show": p.get("show", "S")}
    t = {"step": 0}

    fig, ax = plt.subplots()
    ax.axis("off")

    layout = str(p.get("layout", "spring")).lower().strip()
    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=seed)

    def title():
        if state["show"] == "S":
            return "Bornholdt network: S (buy/sell)  [press 't' to toggle S/C]"
        return "Bornholdt network: C (strategy)  [press 't' to toggle S/C]"

    ax.set_title(title())

    def node_colors():
        spins = model.S if state["show"] == "S" else model.C
        return ["red" if s == 1 else "blue" for s in spins]

    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors(), node_size=120, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.25, ax=ax)

    txt = ax.text(
        0.02, 0.98, "", transform=ax.transAxes,
        va="top", ha="left", fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    def on_key(event):
        if event.key == "t":
            state["show"] = "C" if state["show"] == "S" else "S"
            ax.set_title(title())
            nodes.set_color(node_colors())
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_frame_idx):
        #use time_step(), not sweep(), so one step matches the runner. 
        for _ in range(steps_per_frame):
            model.time_step()
            t["step"] += 1

        nodes.set_color(node_colors())

        M = model.magnetization_M()
        frac_buy = float(np.mean(model.S == 1))
        frac_sell = float(np.mean(model.S == -1))
        frac_fund = float(np.mean(model.C == 1))
        frac_chart = float(np.mean(model.C == -1))

        txt.set_text(
            f"t (steps): {t['step']}\n"
            f"net={net_type}, N={model.N}\n"
            f"J={model.J:g}, alpha={model.alpha:g}, T={model.T:g} (beta={model.beta:.3f})\n"
            f"M: {M:.3f}\n"
            f"buy(+1): {frac_buy:.2f}   sell(-1): {frac_sell:.2f}\n"
            f"fund(C=+1): {frac_fund:.2f}   chart(C=-1): {frac_chart:.2f}"
        )
        return nodes, txt

    ani = FuncAnimation(fig, update, frames=n_frames, interval=interval_ms, blit=False)
    plt.show()


if __name__ == "__main__":
    main()
