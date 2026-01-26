import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from add_run_return import Bornholdt2D


def load_params(path="params.json"):
    defaults = dict(
        L=32,
        J=1.0,
        alpha=4.0,
        T=1.5,
        seed=0,
        sweeps_per_frame=1,
        n_frames=300,
        interval_ms=60,
    )

    if not os.path.exists(path):
        return defaults

    with open(path, "r", encoding="utf-8") as f:
        user = json.load(f)

    out = defaults.copy()
    # flat merge only (simple and safe)
    for k, v in user.items():
        if k in out:
            out[k] = v
    return out


def main():
    p = load_params("params.json")

    L = int(p["L"])
    J = float(p["J"])
    alpha = float(p["alpha"])
    T = float(p["T"])
    seed = int(p["seed"])

    sweeps_per_frame = int(p["sweeps_per_frame"])
    n_frames = int(p["n_frames"])
    interval_ms = int(p["interval_ms"])

    model = Bornholdt2D(L=L, J=J, alpha=alpha, T=T, seed=seed)

    # Toggle between showing S and C
    state = {"show": "S"}  # 'S' or 'C'
    t = {"sweep": 0}

    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])

    def grid():
        return model.S if state["show"] == "S" else model.C

    def title():
        if state["show"] == "S":
            return "Bornholdt lattice: S (buy/sell)  [press 't' to toggle S/C]"
        return "Bornholdt lattice: C (strategy)  [press 't' to toggle S/C]"

    ax.set_title(title())
    im = ax.imshow(grid(), cmap="bwr", vmin=-1, vmax=1, interpolation="nearest")

    txt = ax.text(
        0.02, 0.98, "", transform=ax.transAxes,
        va="top", ha="left", fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    def on_key(event):
        if event.key == "t":
            state["show"] = "C" if state["show"] == "S" else "S"
            ax.set_title(title())
            im.set_data(grid())
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_frame_idx):
        # advance the model
        for _ in range(sweeps_per_frame):
            model.sweep()
            t["sweep"] += 1

        # update image + text
        im.set_data(grid())
        M = model.magnetization()
        frac_buy = float(np.mean(model.S == 1))
        frac_sell = float(np.mean(model.S == -1))
        C_mean = float(np.mean(model.C))

        txt.set_text(
            f"sweep: {t['sweep']}\n"
            f"M: {M:.3f}\n"
            f"buy(+1): {frac_buy:.2f}   sell(-1): {frac_sell:.2f}\n"
            f"<C>: {C_mean:.3f}"
        )
        return im, txt

    # IMPORTANT: keep a reference so animation isn't garbage-collected
    ani = FuncAnimation(fig, update, frames=n_frames, interval=interval_ms, blit=False)

    plt.show()


if __name__ == "__main__":
    main()
