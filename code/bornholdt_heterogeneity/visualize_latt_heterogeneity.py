import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from bornholdt_heterogeneity import Bornholdt2D_heterogeneity


def main() -> None:
    """Animate heterogeneous-α Bornholdt lattice. Keys: t (S/C), a (α-field)."""
    ap = argparse.ArgumentParser(description="Visualize heterogeneous-α Bornholdt lattice.")
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--J", type=float, default=1.0)
    ap.add_argument("--T", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--alpha-mean", type=float, default=8.0)
    ap.add_argument("--alpha-std", type=float, default=2.0)
    ap.add_argument("--alpha-min", type=float, default=1e-6)

    ap.add_argument("--steps-per-frame", type=int, default=1)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--interval-ms", type=int, default=60)
    ap.add_argument("--show", choices=["S", "C", "A"], default="S",
                    help="Initial view: S (spins), C (strategy), A (alpha field).")
    args = ap.parse_args()

    model = Bornholdt2D_heterogeneity(
        L=args.L,
        J=args.J,
        T=args.T,
        seed=args.seed,
        alpha_mean=args.alpha_mean,
        alpha_std=args.alpha_std,
        alpha_min=args.alpha_min,
    )

    state = {"show": args.show}
    step = {"t": 0}

    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])

    def get_grid():
        if state["show"] == "S":
            return model.S
        if state["show"] == "C":
            return model.C
        return model.alpha  # A

    def set_mode(mode: str):
        state["show"] = mode
        ax.set_title(
            f"Heterogeneous-α Bornholdt: {mode}   [t: S/C, a: α]"
            if mode != "A"
            else "Heterogeneous-α Bornholdt: α-field (quenched)   [t: S/C, a: α]"
        )
        im.set_data(get_grid())
        if mode in ("S", "C"):
            im.set_cmap("bwr")
            im.set_clim(-1, 1)
        else:
            im.set_cmap("viridis")
            a = model.alpha
            im.set_clim(float(a.min()), float(a.max()))
        fig.canvas.draw_idle()

    # initialize image
    im = ax.imshow(get_grid(), interpolation="nearest")
    txt = ax.text(
        0.02, 0.98, "", transform=ax.transAxes,
        va="top", ha="left", fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )
    set_mode(state["show"])

    def on_key(event):
        if event.key == "t":
            set_mode("C" if state["show"] == "S" else "S")
        elif event.key == "a":
            set_mode("A" if state["show"] != "A" else "S")

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_):
        for _ in range(args.steps_per_frame):
            model.sweep()
            step["t"] += 1

        # update grid only (colormap handled in set_mode)
        im.set_data(get_grid())

        M = float(model.M())
        frac_buy = float(np.mean(model.S == 1))
        frac_chartist = float(model.frac_chartist())
        frac_fund = 1.0 - frac_chartist
        C_mean = float(np.mean(model.C))

        a = model.alpha
        txt.set_text(
            f"t (sweeps): {step['t']}\n"
            f"L={args.L}, J={args.J:g}, T={args.T:g} (beta={1.0/args.T:.3f})\n"
            f"α ~ Normal({args.alpha_mean:g}, {args.alpha_std:g}²), clipped ≥ {args.alpha_min:g}\n"
            f"α stats: mean={a.mean():.2f}, std={a.std():.2f}, min={a.min():.2f}, max={a.max():.2f}\n"
            f"M: {M:.3f}\n"
            f"buy(+1): {frac_buy:.2f}\n"
            f"fund: {frac_fund:.2f}   chart: {frac_chartist:.2f}\n"
            f"<C>: {C_mean:.3f}\n"
            f"[keys] t: S/C   a: α"
        )
        return im, txt

    _ani = FuncAnimation(fig, update, frames=args.frames, interval=args.interval_ms, blit=False)
    plt.show()


if __name__ == "__main__":
    main()
