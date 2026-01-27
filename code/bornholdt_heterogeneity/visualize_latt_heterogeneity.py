import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from bornholdt_heterogeneity import Bornholdt2D_heterogeneity


# ============================================================
# Default parameters (edit here if needed)
# ============================================================

L = 32
J = 1.0
T = 1.5
seed = 0

# --- heterogeneous alpha parameters ---
alpha_mean = 8.0
alpha_std = 2.0          # variance = 4.0 (moderate heterogeneity)
alpha_min = 1e-6         # enforce positivity

# --- animation parameters ---
steps_per_frame = 1      # sweeps per animation frame
n_frames = 300
interval_ms = 60

# initial view: "S", "C", or "A"
initial_show = "S"


# ============================================================
# Main visualization
# ============================================================

def main():
    model = Bornholdt2D_heterogeneity(
        L=L,
        J=J,
        T=T,
        seed=seed,
        alpha_mean=alpha_mean,
        alpha_std=alpha_std,
        alpha_min=alpha_min,
    )

    state = {"show": initial_show}
    t = {"step": 0}

    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])

    def grid():
        if state["show"] == "S":
            return model.S
        if state["show"] == "C":
            return model.C
        return model.alpha   # "A"

    def title():
        if state["show"] == "S":
            return "Heterogeneous-α Bornholdt: S (buy/sell)   [t: S/C, a: α]"
        if state["show"] == "C":
            return "Heterogeneous-α Bornholdt: C (strategy)   [t: S/C, a: α]"
        return "Heterogeneous-α Bornholdt: α-field (quenched)   [t: S/C, a: α]"

    ax.set_title(title())

    # initialize image
    if state["show"] in ("S", "C"):
        im = ax.imshow(grid(), cmap="bwr", vmin=-1, vmax=1, interpolation="nearest")
    else:
        im = ax.imshow(grid(), cmap="viridis", interpolation="nearest")
        a = model.alpha
        im.set_clim(a.min(), a.max())

    txt = ax.text(
        0.02, 0.98, "", transform=ax.transAxes,
        va="top", ha="left", fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    def refresh():
        im.set_data(grid())
        ax.set_title(title())
        if state["show"] in ("S", "C"):
            im.set_cmap("bwr")
            im.set_clim(-1, 1)
        else:
            im.set_cmap("viridis")
            a = model.alpha
            im.set_clim(a.min(), a.max())

    def on_key(event):
        if event.key == "t":
            state["show"] = "C" if state["show"] == "S" else "S"
            refresh()
            fig.canvas.draw_idle()
        elif event.key == "a":
            state["show"] = "A" if state["show"] != "A" else "S"
            refresh()
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_):
        for _ in range(steps_per_frame):
            model.sweep()
            t["step"] += 1

        refresh()

        M = model.M()
        frac_buy = np.mean(model.S == 1)
        frac_chartist = model.frac_chartist()
        frac_fund = 1.0 - frac_chartist
        C_mean = np.mean(model.C)

        a = model.alpha
        txt.set_text(
            f"t (sweeps): {t['step']}\n"
            f"L={L}, J={J}, T={T} (beta={1.0/T:.3f})\n"
            f"α_ij ~ Normal({alpha_mean}, {alpha_std}²), quenched\n"
            f"α stats: mean={a.mean():.2f}, std={a.std():.2f}, "
            f"min={a.min():.2f}, max={a.max():.2f}\n"
            f"M: {M:.3f}\n"
            f"buy(+1): {frac_buy:.2f}\n"
            f"fundamentalist: {frac_fund:.2f}   chartist: {frac_chartist:.2f}\n"
            f"<C>: {C_mean:.3f}\n"
            f"[keys] t: toggle S/C   a: toggle α"
        )
        return im, txt

    ani = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=interval_ms,
        blit=False,
    )

    plt.show()


if __name__ == "__main__":
    main()
