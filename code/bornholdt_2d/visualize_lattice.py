import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from bornholdt_model import Bornholdt2D


def main() -> None:
    """Animate Bornholdt2D lattice; press 't' to toggle S/C view."""
    ap = argparse.ArgumentParser(description="Visualize Bornholdt 2D lattice dynamics.")
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--J", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--T", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--steps-per-frame", type=int, default=1,
                    help="Sweeps advanced per animation frame.")
    ap.add_argument("--frames", type=int, default=300,
                    help="Number of animation frames.")
    ap.add_argument("--interval-ms", type=int, default=60,
                    help="Delay between frames in milliseconds.")
    ap.add_argument("--show", choices=["S", "C"], default="S",
                    help="Initial view: S (buy/sell) or C (strategy).")

    args = ap.parse_args()

    model = Bornholdt2D(L=args.L, J=args.J, alpha=args.alpha, T=args.T, seed=args.seed)

    state = {"show": args.show}
    t = {"step": 0}

    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])

    def grid():
        return model.S if state["show"] == "S" else model.C

    def title():
        return f"Bornholdt lattice: {state['show']}  (press 't' to toggle S/C)"

    im = ax.imshow(grid(), cmap="bwr", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(title())

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

    def update(_):
        for _ in range(args.steps_per_frame):
            model.sweep()
            t["step"] += 1

        im.set_data(grid())

        M = float(model.M())
        frac_buy = float(np.mean(model.S == 1))
        frac_sell = 1.0 - frac_buy
        chartist_frac = float(np.mean(model.C == -1))  # keep simplest: no hasattr
        fund_frac = 1.0 - chartist_frac
        C_mean = float(np.mean(model.C))

        txt.set_text(
            f"t (sweeps): {t['step']}\n"
            f"L={args.L}, J={args.J:g}, alpha={args.alpha:g}, T={args.T:g} (beta={1.0/args.T:.3f})\n"
            f"M: {M:.3f}\n"
            f"buy(+1): {frac_buy:.2f}  sell(-1): {frac_sell:.2f}\n"
            f"fund(C=+1): {fund_frac:.2f}  chart(C=-1): {chartist_frac:.2f}\n"
            f"<C>: {C_mean:.3f}"
        )
        return im, txt

    ani = FuncAnimation(fig, update, frames=args.frames, interval=args.interval_ms, blit=False)
    plt.show()


if __name__ == "__main__":
    main()