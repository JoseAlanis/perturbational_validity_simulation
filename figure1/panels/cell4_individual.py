"""Render validation criteria as independent manuscript panels."""

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

_spec = importlib.util.spec_from_file_location(
    "cell4_minimal", str(Path(__file__).with_name("cell4_minimal.py")))
c4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c4)


CM = 1 / 2.54
FIGSIZE = (7.6, 6.8)
#: Left inset has to clear the widest y tick labels in the row (panel d's
#: "-1.3") and still leave room for the axis label outside them, or the
#: "z" of $z_1$ is clipped at the figure edge. The right edge stays at 0.85
#: so d, e and f keep identical plot areas.
AXIS_BOX = (0.215, 0.29, 0.635, 0.56)
FAIL = "#9A4B3F"
CAUTION = "#8A6A2F"


def canvas(letter, title):
    fig = plt.figure(figsize=(FIGSIZE[0] * CM, FIGSIZE[1] * CM))
    fig.text(0.025, 0.975, letter, ha="left", va="top", fontsize=13,
             fontweight="bold", color="black")
    ax = fig.add_axes(AXIS_BOX)
    c4.frame(ax, title)
    return fig, ax


def verdict(ax, text, colour):
    ax.text(0.5, 0.94, text, transform=ax.transAxes, ha="center", va="top",
            fontsize=8.2, fontweight="bold", color=colour, zorder=20,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white",
                  "edgecolor": colour, "linewidth": 0.8, "alpha": 0.96})


def save(fig, out):
    fig.savefig(out, dpi=400)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def trajectory_panel(out_dir):
    fig, ax = canvas("d", "Time-series accuracy")
    dt = 0.01
    time = np.arange(int(c4.DURATION / dt)) * dt
    truth, _ = c4.run(False, 0.0, seed=3)
    model, _ = c4.run(True, 0.0, seed=3)
    ax.plot(time, truth[:, 0], color=c4.TRUTH, lw=1.4, label="Ground truth")
    ax.plot(time, model[:, 0], color=c4.MODEL, lw=1.4, ls=(0, (4, 2.5)),
            label="Passive-data model")
    ax.set(xlim=(0, c4.DURATION), ylim=(-1.3, -0.5), xlabel="$t$",
           ylabel="$z_1$")
    ax.set_xticks(np.linspace(0, c4.DURATION, 5))
    ax.set_yticks(np.linspace(-1.3, -0.5, 5))
    verdict(ax, "accurate on passive data", CAUTION)
    save(fig, out_dir / "cell4D_time_series.png")


def structure_panel(out_dir):
    fig, ax = canvas("e", "Dynamical-structure similarity")
    grid = np.linspace(-1.9, 1.9, 400)
    ax.plot(grid, c4.potential(grid, False), color=c4.TRUTH, lw=1.6)
    ax.plot(grid, c4.potential(grid, True), color=c4.MODEL, lw=1.6,
            ls=(0, (4, 2.5)))
    ax.scatter([-1, 1], [0, 0], s=26, facecolor="white", edgecolor=c4.TRUTH,
               linewidth=1.4, zorder=6)
    ax.set(xlim=(-2.0, 2.0), ylim=(-0.05, 1.15), xlabel="$z_1$",
           ylabel="$V(z_1)$")
    ax.set_xticks(np.linspace(-2.0, 2.0, 5))
    ax.set_yticks(np.linspace(0.0, 1.0, 5))
    verdict(ax, "structure not preserved", FAIL)
    handles = [
        Line2D([0], [0], color=c4.TRUTH, lw=1.6, label="Ground truth"),
        Line2D([0], [0], color=c4.MODEL, lw=1.6, ls=(0, (4, 2.5)),
               label="Passive-data model"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
               ncol=2, fontsize=7.5, frameon=False, handlelength=2.1,
               columnspacing=1.2)
    save(fig, out_dir / "cell4E_structure.png")


def response_panel(out_dir):
    fig, ax = canvas("f", "Perturbational-response similarity")
    dt = 0.01
    time = np.arange(int(c4.DURATION / dt)) * dt
    truth, u = c4.run(False, c4.U_AMP, seed=11, n_rep=c4.N_REP)
    model, _ = c4.run(True, c4.U_AMP, seed=11, n_rep=c4.N_REP)
    c4.band(ax, time, truth, c4.TRUTH, "-", "Ground truth")
    c4.band(ax, time, model, c4.MODEL, (0, (4, 2.5)), "Passive-data model")
    for level in (-1.0, 1.0):
        ax.axhline(level, color="#C9D0D8", lw=0.8, ls=(0, (4, 3)), zorder=0)
    base, height = -2.5, 0.4
    ax.step(time, base + height * (u / c4.U_AMP), where="post",
            color=c4.PULSE, lw=1.3)
    ax.text(c4.U_START + c4.U_DUR + 0.3, base + height,
            "held-out $u(t)$", fontsize=7.5, color="#9A6B1E",
            va="center", ha="left")
    ax.set(xlim=(0, c4.DURATION), ylim=(-2.8, 2.8), xlabel="$t$",
           ylabel="$z_1$")
    ax.set_xticks(np.linspace(0, c4.DURATION, 5))
    ax.set_yticks(np.linspace(-2.0, 2.0, 5))
    verdict(ax, "response not preserved", FAIL)
    save(fig, out_dir / "cell4F_response.png")


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans",
                            "DejaVu Sans"],
        "pdf.fonttype": 42,
    })
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/figure1")
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectory_panel(out_dir)
    structure_panel(out_dir)
    response_panel(out_dir)
    print(f"wrote panels D--F to {out_dir}")


if __name__ == "__main__":
    main()
