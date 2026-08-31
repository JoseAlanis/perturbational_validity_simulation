"""Compare trajectory, structure, and perturbational-response accuracy."""

import importlib.util
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_spec = importlib.util.spec_from_file_location(
    "cell1_minimal", str(Path(__file__).with_name("cell1_minimal.py")))
c1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c1)

TRUTH = os.environ.get("TRUTH", "#2C3038")
# Match the passive-model colour used in the result figures.
MODEL = os.environ.get("MODEL", "#D55E00")
PULSE = os.environ.get("PULSE", "#E0A23C")

SIGMA = float(os.environ.get("SIGMA", "0.16"))
N_REP = int(os.environ.get("N_REP", "60"))
U_AMP = float(os.environ.get("U_AMP", "0.60"))
U_START, U_DUR = 4.0, 5.0
DURATION = float(os.environ.get("DURATION", "16.0"))
Z_STAR = -1.0                      # the attractor the model is built around
SLOPE = c1.ALPHA - 3 * c1.BETA * Z_STAR**2      # f'(z1) at that attractor


def drift(z1, z2, u, model):
    """True double-well drift, or its linearisation about the left attractor."""
    if model:
        return z2, -c1.DELTA * z2 + SLOPE * (z1 - Z_STAR) + u
    return z2, -c1.DELTA * z2 + c1.ALPHA * z1 - c1.BETA * z1**3 + u


def run(model, u_amp, seed, n_rep=1, dt=0.01):
    """Ensemble under either system, with a pulse of the given amplitude."""
    rng = np.random.default_rng(seed)
    n = int(DURATION / dt)
    u = np.zeros(n)
    u[int(U_START / dt):int((U_START + U_DUR) / dt)] = u_amp

    z1 = np.full(n_rep, Z_STAR)
    z2 = np.zeros(n_rep)
    out = np.empty((n, n_rep))
    for t in range(n):
        out[t] = z1
        d1, d2 = drift(z1, z2, u[t], model)
        z1 = z1 + dt * d1
        z2 = z2 + dt * d2 + SIGMA * np.sqrt(dt) * rng.standard_normal(n_rep)
    return out, u


def potential(z1, model):
    """Energy landscape of either system, offset to zero at its minimum."""
    if model:
        v = -0.5 * SLOPE * (z1 - Z_STAR) ** 2
    else:
        v = -c1.ALPHA * z1**2 / 2 + c1.BETA * z1**4 / 4
    return v - v.min()


def frame(ax, title):
    ax.set_facecolor(c1.BG)
    for spine in ax.spines.values():
        spine.set_color(c1.SPINE)
        spine.set_linewidth(0.9)
    ax.tick_params(colors="black", labelsize=9.5, width=1.0, length=3.0)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_color("black")
    ax.set_title(title, fontsize=9.2, fontweight="bold", color="black", pad=7)


def band(ax, time, ensemble, colour, ls, label):
    lo = np.percentile(ensemble, 5, axis=1, method="hazen")
    hi = np.percentile(ensemble, 95, axis=1, method="hazen")
    ax.fill_between(time, lo, hi, color=colour, alpha=0.18, lw=0)
    ax.plot(time, ensemble.mean(axis=1), color=colour, lw=1.6, ls=ls,
            label=label)


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(17.4 * c1.CM, 7.6 * c1.CM))
    width, y0, height = 0.235, 0.34, 0.51
    axes = [fig.add_axes([0.070 + k * 0.325, y0, width, height]) for k in range(3)]
    # Panel letters continue the sequence from cells A-C, which carry their own.
    for a, letter in zip(axes, "DEF"):
        box = a.get_position()
        fig.text(box.x0 - 0.052, box.y1 + 0.085, letter, ha="left", va="bottom",
                 fontsize=13, fontweight="bold", color="black")
    #: Per-panel verdict labels.
    verdicts = []
    dt = 0.01
    time = np.arange(int(DURATION / dt)) * dt

    # --- 1. time-series accuracy: the model passes -------------------------
    ax = axes[0]
    true_passive, _ = run(False, 0.0, seed=3)
    model_passive, _ = run(True, 0.0, seed=3)
    ax.plot(time, true_passive[:, 0], color=TRUTH, lw=1.4,
            label="Ground truth")
    ax.plot(time, model_passive[:, 0], color=MODEL, lw=1.4, ls=(0, (4, 2.5)),
            label="Passive model")
    ax.set_xlabel("$t$", fontsize=11, color="black", labelpad=1)
    ax.set_ylabel("$z_1$", fontsize=11, color="black", labelpad=2)
    frame(ax, "Time-series accuracy")
    ax.set_ylim(-1.3, -0.6)
    # Amber marks agreement limited to passive data.
    verdicts.append(("accurate on passive data", "#8A6A2F"))

    # --- 2. dynamical structure: the model fails ---------------------------
    ax = axes[1]
    grid = np.linspace(-1.9, 1.9, 400)
    ax.plot(grid, potential(grid, False), color=TRUTH, lw=1.6,
            label="Ground truth")
    ax.plot(grid, potential(grid, True), color=MODEL, lw=1.6, ls=(0, (4, 2.5)),
            label="Passive model")
    ax.scatter([-1, 1], [0, 0], s=26, facecolor="white", edgecolor=TRUTH,
               linewidth=1.4, zorder=6)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("$z_1$", fontsize=11, color="black", labelpad=1)
    ax.set_ylabel("$V(z_1)$", fontsize=11, color="black", labelpad=2)
    frame(ax, "Dynamical-structure similarity")
    verdicts.append(("structure not preserved", "#9A4B3F"))

    # --- 3. response to a held-out input: the model fails -------------------
    ax = axes[2]
    true_driven, u = run(False, U_AMP, seed=11, n_rep=N_REP)
    model_driven, _ = run(True, U_AMP, seed=11, n_rep=N_REP)
    band(ax, time, true_driven, TRUTH, "-", "Ground truth")
    band(ax, time, model_driven, MODEL, (0, (4, 2.5)), "Passive model")
    ax.axhline(1.0, color="#C9D0D8", lw=0.8, ls=(0, (4, 3)), zorder=0)
    ax.axhline(-1.0, color="#C9D0D8", lw=0.8, ls=(0, (4, 3)), zorder=0)

    base, tall = -2.5, 0.4
    ax.step(time, base + tall * (u / U_AMP), where="post", color=PULSE, lw=1.3)
    ax.text(U_START + U_DUR + 0.3, base + tall, " held-out $u(t)$", fontsize=7.5,
            color="#9A6B1E", va="center", ha="left")
    ax.set_ylim(-2.8, 2.8)
    ax.set_xlabel("$t$", fontsize=11, color="black", labelpad=1)
    ax.set_ylabel("$z_1$", fontsize=11, color="black", labelpad=2)
    frame(ax, "Perturbational-response similarity")
    verdicts.append(("response not preserved", "#9A4B3F"))

    # Use one legend for all three panels.
    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(handles, ["Ground truth", "Passive-data model"],
               loc="lower center", bbox_to_anchor=(0.5, 0.145), ncol=2,
               fontsize=8.2, frameon=False, labelcolor="black",
               handlelength=2.5, columnspacing=2.2)

    for a, (text, colour) in zip(axes, verdicts):
        a.text(0.5, 0.94, text, transform=a.transAxes, ha="center", va="top",
               fontsize=8.2, fontweight="bold", color=colour, zorder=20,
               bbox={"boxstyle": "round,pad=0.28", "facecolor": "white",
                     "edgecolor": colour, "linewidth": 0.8, "alpha": 0.96})

    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/cell4.png")
    fig.savefig(out, dpi=400)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
