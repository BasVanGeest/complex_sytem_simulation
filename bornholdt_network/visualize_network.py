import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
from model_network import BornholdtNetwork
from networks import create_network


def main():
    n = 50
    SEED = 42
    NETWORK_TYPE = "WS"  # "BA", "ER", or "WS"

    G = create_network(
        net_type=NETWORK_TYPE,
        n=n,
        seed=SEED,

        m=2,   # BA only 
        k=4,   # WS only
        p=0.2  # Used by ER or WS
    )

    J = 1.0
    alpha = 4.0
    T = 1.5

    model = BornholdtNetwork(G, J=J, alpha=alpha, T=T, seed=SEED)

    sweeps_per_frame = 1
    n_frames = 200
    interval_ms = 100

    fig, ax = plt.subplots()
    ax.set_title("Bornholdt model on network (red=+1 buy, blue=-1 sell)")
    ax.axis("off")


    pos = nx.spring_layout(G, seed=SEED)
    t = {"sweep": 0}

    def update(frame_idx):
        for _ in range(sweeps_per_frame):
            model.sweep()
            t["sweep"] += 1

        colors = ["red" if s == 1 else "blue" for s in model.S]

        ax.clear()
        ax.set_title("Bornholdt model on network (red=+1 buy, blue=-1 sell)")
        ax.axis("off")
        nx.draw(G, pos, node_color=colors, with_labels=True, node_size=300, ax=ax)

        M = model.magnetization()
        frac_buy = (model.S == 1).mean()
        frac_sell = (model.S == -1).mean()

        ax.text(
            0.02, 0.98,
            f"sweep: {t['sweep']}\nM: {M:.3f}\nbuy(+1): {frac_buy:.2f}  sell(-1): {frac_sell:.2f}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=10, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
        )

        return []

    ani = FuncAnimation(fig, update, frames=n_frames, interval=interval_ms, blit=False)
    plt.show()

if __name__ == "__main__":
    main()
