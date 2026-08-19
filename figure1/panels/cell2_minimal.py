"""Render controlled transitions for one participant or a population."""

import importlib.util
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("MODE", "samples")
_spec = importlib.util.spec_from_file_location(
    "cell1_minimal", str(Path(__file__).with_name("cell1_minimal.py")))
c1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c1)

U_AMP = float(os.environ.get("U_AMP", "0.60"))
U_START = float(os.environ.get("U_START", "6.0"))
U_DUR = float(os.environ.get("U_DUR", "3.0"))
DURATION = float(os.environ.get("DURATION", str(c1.T_SHOWN)))
SIGMA = float(os.environ.get("SIGMA", "0.20"))
SEED = int(os.environ.get("SEED", "7"))
N_SAMPLES = int(os.environ.get("N_SAMPLES", "18"))
PULSE = os.environ.get("PULSE", "#E0A23C")     # accent for the driven stretch
OBS_NOISE = float(os.environ.get("OBS_NOISE", "0.08"))
SUBTITLE = os.environ.get("SUBTITLE", "Observations")
#: Coloured measurement-noise settings.
EEG_AMP = float(os.environ.get("EEG_AMP", "0.30"))
EEG_SLOPE = float(os.environ.get("EEG_SLOPE", "1.0"))
#: Population mode varies input gain across participants.
POPULATION = os.environ.get("POPULATION", "") not in ("", "0", "false")
Z_START = tuple(float(v) for v in
                os.environ.get("Z_START", "-1.0,0.0").split(","))
#: Population-mode initial states and passive colour.
Z_PASSIVE = tuple(float(v) for v in
                  os.environ.get("Z_PASSIVE", "-0.45,-1.30").split(","))
Z_DRIVEN = tuple(float(v) for v in
                 os.environ.get("Z_DRIVEN", "-0.85,0.00").split(","))
PASSIVE_COLOUR = os.environ.get("PASSIVE_COLOUR", "#94588D")
if POPULATION:
    U_DUR = float(os.environ.get("U_DUR", "5.0"))
TITLE = os.environ.get(
    "TITLE",
    "Shared Dynamical Model" if POPULATION
    else "Latent Dynamics Under Perturbation")
PANEL_LETTER = os.environ.get("PANEL_LETTER", "C" if POPULATION else "B")


def driven_run(dt=0.01, gain=1.0, seed=None, start=None):
    """Stochastic record with a pulse applied part way through."""
    rng = np.random.default_rng(SEED if seed is None else seed)
    n = int(DURATION / dt)
    u = np.zeros(n)
    u[int(U_START / dt):int((U_START + U_DUR) / dt)] = U_AMP

    z = np.array(Z_START if start is None else start, dtype=float)
    out = np.empty((n, 2))
    for t in range(n):
        out[t] = z
        d1, d2 = c1.field(z[0], z[1])
        z = z + dt * np.array([d1, d2 + gain * u[t]])
        z[1] += SIGMA * np.sqrt(dt) * rng.standard_normal()
    return out, u


def coloured_noise(n, slope, seed, dt=0.01):
    """Zero-mean 1/f^slope noise of unit standard deviation."""
    rng = np.random.default_rng(seed)
    freq = np.fft.rfftfreq(n, d=dt)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    scale = np.ones_like(freq)
    scale[1:] = freq[1:] ** (-slope / 2.0)
    out = np.fft.irfft(spectrum * scale, n=n)
    return out / out.std()


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "pdf.fonttype": 42,
    })
    # Same fixed rectangles as cell 1, so the phase planes coincide exactly.
    fig = plt.figure(figsize=(c1.FIGSIZE[0] * c1.CM, c1.FIGSIZE[1] * c1.CM))
    fig.text(0.025, 0.995 if c1.COMPACT else 0.975, PANEL_LETTER,
             ha="left", va="top", fontsize=15 if c1.COMPACT else 13,
             fontweight="bold", color="black")
    ax = fig.add_axes(c1.PANEL_BOX)
    ax_ts = fig.add_axes(c1.BAND_BOX)

    grid = np.linspace(-c1.LIM, c1.LIM, 90)
    g1, g2 = np.meshgrid(grid, grid)
    d1, d2 = c1.field(g1, g2)
    flow = ax.streamplot(g1, g2, d1, d2, color=c1.STREAM, linewidth=0.75,
                         density=1.1, arrowsize=0.8, arrowstyle="-|>", zorder=3)
    flow.lines.set_alpha(c1.STREAM_ALPHA)
    flow.arrows.set_alpha(c1.STREAM_ALPHA)
    for branch in c1.separatrix():
        ax.plot(branch[:, 0], branch[:, 1], color=c1.FRONTIER,
                lw=c1.FRONTIER_LW, ls=(0, (1.8, 2.4)), zorder=4)

    if POPULATION:
        # A zero gain makes the passive participant ignore the pulse.
        passive_run, _ = driven_run(gain=0.0, seed=SEED + 11, start=Z_PASSIVE)
        run, u = driven_run(gain=1.0, seed=SEED, start=Z_DRIVEN)
        runs = [(passive_run, False), (run, True)]
    else:
        run, u = driven_run()
    on = np.flatnonzero(u > 0)

    # Highlight the pulse-driven trajectory segment.
    if POPULATION:
        for r, driven in runs:
            if driven:
                ax.plot(r[on, 0], r[on, 1], color=PULSE, lw=6.0, alpha=0.50,
                        solid_capstyle="round", zorder=4.5)
                c1.plot_by_basin(ax, r[:, 0], r[:, 1], r[:, 0], lw=1.5,
                                 alpha=0.55, zorder=5)
                # Aim the final marker along the local tangent.
                arc = np.concatenate([[0.0], np.cumsum(
                    np.linalg.norm(np.diff(r, axis=0), axis=1))])
                back = min(int(np.searchsorted(arc, arc[-1] - 0.10)), len(r) - 2)
                dx, dy = r[-1] - r[back]
                ax.plot(r[-1, 0], r[-1, 1],
                        marker=(3, 0, np.degrees(np.arctan2(dy, dx)) - 90),
                        ms=8.0, color=c1.TRAJ_RIGHT if r[-1, 0] > 0
                        else c1.TRAJ_LEFT, markeredgewidth=0, zorder=8)
            else:
                ax.plot(r[:, 0], r[:, 1], color=PASSIVE_COLOUR, lw=1.5,
                        alpha=0.55, zorder=5)
            ax.scatter(r[0, 0], r[0, 1], s=26, color="black", zorder=7,
                       linewidths=0)
    else:
        # Draw the pulse halo behind the trajectory.
        ax.plot(run[on, 0], run[on, 1], color=PULSE, lw=6.0, alpha=0.50,
                solid_capstyle="round", zorder=4.5)
        c1.plot_by_basin(ax, run[:, 0], run[:, 1], run[:, 0], lw=1.5, alpha=0.55,
                         zorder=5)
        idx = np.linspace(0, len(run) - 1, N_SAMPLES).round().astype(int)
        for sign, colour in ((-1, c1.TRAJ_LEFT), (1, c1.TRAJ_RIGHT)):
            keep = np.sign(run[idx, 0]) == sign
            ax.scatter(run[idx][keep, 0], run[idx][keep, 1], s=26, color=colour,
                       zorder=6, linewidths=0)
        ax.scatter(run[0, 0], run[0, 1], s=26, color="black", zorder=7,
                   linewidths=0)

    ax.set_facecolor(c1.BG)
    for spine in ax.spines.values():
        spine.set_color(c1.SPINE)
        spine.set_linewidth(0.9)
    ax.set_xlim(-c1.LIM, c1.LIM)
    ax.set_ylim(-c1.LIM, c1.LIM)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_aspect("equal")
    ax.tick_params(colors="black", labelsize=10, width=1.0, length=3.5)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_color("black")
    ax.set_xlabel("$z_1$", fontsize=12, color="black", labelpad=2)
    ax.set_ylabel("$z_2$", fontsize=12, color="black", labelpad=2)
    ax.set_title(TITLE, fontsize=10.5, fontweight="bold", color="black", pad=7)
    # --- the same record as a recording-like observation --------------------
    time = np.arange(len(run)) * 0.01
    for level in (-1.0, 1.0):
        ax_ts.axhline(level, color="#C9D0D8", lw=0.8, ls=(0, (4, 3)), zorder=0)

    if POPULATION:
        for k, (r, driven) in enumerate(runs):
            sig = r[:, 0] + EEG_AMP * coloured_noise(len(r), EEG_SLOPE, seed=9 + k)
            if driven:
                c1.plot_by_basin(ax_ts, time, sig, r[:, 0], lw=0.7, alpha=0.75,
                                 zorder=4)
            else:
                ax_ts.plot(time, sig, color=PASSIVE_COLOUR, lw=0.7, alpha=0.75,
                           zorder=4)
    else:
        signal = run[:, 0] + EEG_AMP * coloured_noise(len(run), EEG_SLOPE, seed=9)
        c1.plot_by_basin(ax_ts, time, signal, run[:, 0], lw=0.7, alpha=0.75,
                         zorder=4)

    # u(t) as a step in the free band below, in the colour of the halo above.
    base, height = -2.55, 0.45
    ax_ts.step(time, base + height * (u / U_AMP), where="post", color=PULSE,
               lw=1.4, zorder=5)
    # Place the pulse label in the empty region.
    ax_ts.text(U_START + U_DUR + 0.35, base + height, "$u(t) > 0$",
               fontsize=8.5, color="#9A6B1E", va="center", ha="left")

    ax_ts.set_facecolor(c1.BG)
    for spine in ax_ts.spines.values():
        spine.set_color(c1.SPINE)
        spine.set_linewidth(0.9)
    ax_ts.set_xlim(0, time[-1])
    ax_ts.set_xticks(np.arange(0, time[-1] + 1, 5))
    ax_ts.set_ylim(-3.30, 2.65)
    ax_ts.set_yticks([-1.0, 1.0])
    ax_ts.set_yticklabels(["State 1", "State 2"], fontsize=8.0)
    ax_ts.tick_params(colors="black", labelsize=9.5, width=1.0, length=3.0)
    for tick in ax_ts.get_xticklabels() + ax_ts.get_yticklabels():
        tick.set_color("black")
    ax_ts.set_xlabel("$t$", fontsize=11.5, color="black", labelpad=1)
    ax_ts.set_title(SUBTITLE, fontsize=9.5, fontweight="bold", color="black",
                    pad=1 if c1.COMPACT else 6)

    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/cell2.png")
    fig.savefig(out, dpi=400)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out}  (ends in {'right' if run[-1, 0] > 0 else 'left'} well)")


if __name__ == "__main__":
    main()
