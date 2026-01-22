import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from model import Bornholdt2D


def main():
    # Bornholdt-like parameters (paper-style)
    L = 32
    J = 1.0
    alpha = 4.0
    T = 1.5
    seed = 0

    model = Bornholdt2D(L=L, J=J, alpha=alpha, T=T, seed=seed)

    sweeps_per_frame = 1      # increase to speed up dynamics (e.g., 5 or 10)
    n_frames = 300            # total frames
    interval_ms = 60          # delay between frames in milliseconds

    fig, ax = plt.subplots()
    ax.set_title("Bornholdt model: S lattice (red=+1 buy, blue=-1 sell)")

    im = ax.imshow(model.S, cmap="bwr", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])

    # Text overlay for time and magnetization
    txt = ax.text(
        0.02, 0.98, "", transform=ax.transAxes,
        va="top", ha="left", fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
    )

    t = {"sweep": 0}

    def update(frame_idx):
        # advance the model
        for _ in range(sweeps_per_frame):
            model.sweep()
            t["sweep"] += 1

        # update image + text
        im.set_data(model.S)
        M = model.magnetization()
        frac_buy = np.mean(model.S == 1)
        frac_sell = np.mean(model.S == -1)

        txt.set_text(
            f"sweep: {t['sweep']}\n"
            f"M: {M:.3f}\n"
            f"buy(+1): {frac_buy:.2f}   sell(-1): {frac_sell:.2f}"
        )
        return im, txt

    ani = FuncAnimation(fig, update, frames=n_frames, interval=interval_ms, blit=False)
    plt.show()


if __name__ == "__main__":
    main()
