import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from bornholdt_model import Bornholdt2D


def load_params(path="params.json"):
    # Paper defaults (Bornholdt Fig. 1 caption)
    defaults = dict(
        L=32,
        J=1.0,
        alpha=4.0,
        T=1.5,          # paper uses T = 1/beta = 1.5  (beta = 1/T)
        seed=0,
        steps_per_frame=1,  # 1 "time step" = 1 sweep + synchronous C update
        n_frames=300,
        interval_ms=60,
        show="S",           # "S" or "C"
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
    p = load_params("params.json")

    # Force the paper parameters unless you intentionally override in params.json
    L = int(p["L"])
    J = float(p["J"])
    alpha = float(p["alpha"])
    T = float(p["T"])
    seed = int(p["seed"])

    steps_per_frame = int(p["steps_per_frame"])
    n_frames = int(p["n_frames"])
    interval_ms = int(p["interval_ms"])

    model = Bornholdt2D(L=L, J=J, alpha=alpha, T=T, seed=seed)

    # Toggle between showing S and C
    state = {"show": p.get("show", "S")}  # 'S' or 'C'
    t = {"step": 0}  # paper time index ~ sweeps

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
        # advance the model by Bornholdt time steps
        for _ in range(steps_per_frame):
            model.time_step()   # sweep + synchronous C update (paper t -> t+1)
            t["step"] += 1

        # update image + text
        im.set_data(grid())
        M = model.magnetization_M()

        frac_buy = float(np.mean(model.S == 1))
        frac_sell = float(np.mean(model.S == -1))
        C_mean = model.mean_strategy_C()

        txt.set_text(
            f"t (sweeps): {t['step']}\n"
            f"L={L}, J={J:g}, alpha={alpha:g}, T={T:g} (beta={1.0/T:.3f})\n"
            f"M: {M:.3f}\n"
            f"buy(+1): {frac_buy:.2f}   sell(-1): {frac_sell:.2f}\n"
            f"<C>: {C_mean:.3f}"
        )
        return im, txt

    # Keep a reference so animation isn't garbage-collected
    ani = FuncAnimation(fig, update, frames=n_frames, interval=interval_ms, blit=False)

    plt.show()


if __name__ == "__main__":
    main()
