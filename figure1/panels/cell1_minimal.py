"""Render the bistable field, basin boundary, trajectories, and time series."""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize

BG = os.environ.get("BG", "#FCFDFE")   # panel fill, near white
STREAM = "#BBBBBB"
#: Basin colours use equal lightness and chroma.
STREAM_ALPHA = float(os.environ.get("STREAM_ALPHA", "0.6"))
STREAM_LEFT = os.environ.get("C_LEFT", "#A8B9CB")    # muted grey-blue
STREAM_RIGHT = os.environ.get("C_RIGHT", "#D9ADA5")  # muted brick
#: Foreground trajectory colours.
TRAJ_LEFT = "#37729D"
TRAJ_RIGHT = "#B04F43"
SPINE = os.environ.get("SPINE", "#8C949D")   # panel box
#: Basin-boundary style.
FRONTIER = os.environ.get("FRONTIER", "#3F454B")
FRONTIER_LW = float(os.environ.get("FRONTIER_LW", "0.85"))
BASIN_LEFT = "#DCE4EC"
BASIN_RIGHT = "#F3DFDC"

CM = 1 / 2.54
#: Shared phase-plane layout.
COMPACT = os.environ.get("COMPACT", "") not in ("", "0", "false")
FIGSIZE = (7.6, 10.2) if COMPACT else (7.6, 10.8)  # cm
#: Symmetric margins for outer state labels.
PANEL_BOX = ((0.17, 0.40, 0.66, 0.51) if COMPACT
             else (0.15, 0.40, 0.68, 0.53))
BAND_BOX = ((0.17, 0.09, 0.66, 0.15) if COMPACT
            else (0.15, 0.075, 0.68, 0.175))
#: Shared observation duration.
T_SHOWN = 16.0
ALPHA, BETA = 1.0, 1.0
DELTA = float(os.environ.get("DELTA", "0.65"))
FILL = os.environ.get("FILL", "") not in ("", "0", "false")
LIM = 2.1
TITLE = os.environ.get("TITLE", "Unperturbed latent dynamics")
PANEL_LETTER = os.environ.get("PANEL_LETTER", "A")
STYLE = os.environ.get("STYLE", "boundary")   # "boundary" | "basincolor"
#: Observation-band heading.
SUBTITLE = os.environ.get("SUBTITLE", "Passive observations")
#: Observation-band y label.
SIGNAL_LABEL = os.environ.get("SIGNAL_LABEL", "Observed\nsignal")

#: Figure variant.
MODE = os.environ.get("MODE", "field")
PASSIVE_SIGMA = float(os.environ.get("PASSIVE_SIGMA", "0.38"))
PASSIVE_SPAN = float(os.environ.get("PASSIVE_SPAN", "70.0"))   # window shown
PASSIVE_SEED = int(os.environ.get("PASSIVE_SEED", "11"))
#: Passive-record drawing style.
PASSIVE_STYLE = os.environ.get("PASSIVE_STYLE", "dots")
N_DOTS = int(os.environ.get("N_DOTS", "20"))
#: Time window for sparse samples.
DOT_SPAN = float(os.environ.get("DOT_SPAN", "10.0"))
DOT_FROM = float(os.environ.get("DOT_FROM", "0.0"))
CONNECT = os.environ.get("CONNECT", "1") not in ("0", "false", "")
#: Dense occupancy-dot style.
DOT_SIZE = float(os.environ.get("DOT_SIZE", "1.6"))
DOT_ALPHA = float(os.environ.get("DOT_ALPHA", "0.30"))
BRIDGE = os.environ.get("BRIDGE", "") not in ("", "0", "false")
#: Passive-record initial state and burn-in.
PASSIVE_START = tuple(float(v) for v in
                      os.environ.get("PASSIVE_START", "-1.85,0.30").split(","))
BURN_IN = float(os.environ.get("BURN_IN", "0.0"))
WELL_SPAN = float(os.environ.get("WELL_SPAN", "25.0"))
WELL_START = float(os.environ.get("WELL_START", "1.85"))
#: Sampling-orbit initial state.
N_SAMPLES = int(os.environ.get("N_SAMPLES", "10"))
SAMPLE_START = tuple(float(v) for v in
                     os.environ.get("SAMPLE_START", "-0.7,-1.5").split(","))
#: Sampling-path noise level.
SAMPLE_SIGMA = float(os.environ.get("SAMPLE_SIGMA", "0.0"))
SAMPLE_DURATION = float(os.environ.get("SAMPLE_DURATION", str(T_SHOWN)))
#: Optional path decimation and smoothing.
DECIMATE = int(os.environ.get("DECIMATE", "1"))
SMOOTH = int(os.environ.get("SMOOTH", "1"))


#: Initial states for basin trajectories.
STARTS = [(-0.40, -1.40), (0.40, 1.40)]
if os.environ.get("EXTRA_START", "") not in ("", "0", "false"):
    STARTS.append((-1.25, 0.75))

#: Observation-band settings.
STRIP = os.environ.get("STRIP", "0") not in ("0", "false", "")
OBS_NOISE = float(os.environ.get("OBS_NOISE", "0.055"))
#: Observation-band time window.
T_MAX = float(os.environ.get("T_MAX", "3.0"))


def field(z1, z2):
    return z2, -DELTA * z2 + ALPHA * z1 - BETA * z1**3


def basins(resolution=420, dt=0.01, n_steps=3000):
    """Integrate a grid of initial conditions and label by final attractor."""
    axis = np.linspace(-LIM, LIM, resolution)
    g1, g2 = np.meshgrid(axis, axis)
    z1, z2 = g1.copy(), g2.copy()

    for _ in range(n_steps):
        d1, d2 = field(z1, z2)
        z1 = z1 + dt * d1
        z2 = z2 + dt * d2

    return axis, np.where(z1 > 0.0, 1.0, 0.0)


def trajectory(z0, dt=0.01, max_steps=4000, tol=0.18):
    """Integrate a path until it reaches an attractor or ``max_steps``."""
    z = np.array(z0, dtype=float)
    path = [z.copy()]
    for _ in range(max_steps):
        d1, d2 = field(z[0], z[1])
        z = z + dt * np.array([d1, d2])
        path.append(z.copy())
        attractor = np.array([np.sign(z[0]) * np.sqrt(ALPHA / BETA), 0.0])
        if np.linalg.norm(z - attractor) < tol:
            break
    return np.array(path)


def plot_by_basin(ax, px, py, basin_of, lw, alpha=1.0, zorder=5):
    """Draw path segments colored by the sign of ``basin_of``."""
    side = np.sign(basin_of)
    side[side == 0] = 1.0
    breaks = [0, *(np.flatnonzero(np.diff(side)) + 1), basin_of.size]
    for start, stop in zip(breaks[:-1], breaks[1:]):
        lo, hi = start, min(stop + 1, basin_of.size)
        if hi - lo < 2:
            continue
        colour = TRAJ_LEFT if basin_of[lo] < 0 else TRAJ_RIGHT
        ax.plot(px[lo:hi], py[lo:hi], color=colour, lw=lw, alpha=alpha,
                solid_capstyle="round", zorder=zorder)


def sample_path(sign, seed, dt=0.01):
    """The path the samples are drawn from, deterministic or stochastic."""
    start = tuple(sign * abs(v) for v in SAMPLE_START)
    if SAMPLE_SIGMA <= 0:
        return trajectory(start, tol=0.06, max_steps=12000)

    rng = np.random.default_rng(seed)
    z = np.array(start, dtype=float)
    out = np.empty((int(SAMPLE_DURATION / dt), 2))
    for t in range(len(out)):
        out[t] = z
        d1, d2 = field(z[0], z[1])
        z = z + dt * np.array([d1, d2])
        z[1] += SAMPLE_SIGMA * np.sqrt(dt) * rng.standard_normal()
    return out


def separatrix(dt=0.002, n_steps=20000):
    """Trace the saddle's stable manifold backward in time."""
    jac = np.array([[0.0, 1.0], [ALPHA, -DELTA]])
    values, vectors = np.linalg.eig(jac)
    stable_direction = vectors[:, int(np.argmin(values.real))].real
    stable_direction /= np.linalg.norm(stable_direction)

    branches = []
    for sign in (1, -1):
        z = sign * 1e-6 * stable_direction
        path = [z.copy()]
        for _ in range(n_steps):
            d1, d2 = field(z[0], z[1])
            z = z - dt * np.array([d1, d2])          # backwards in time
            if np.max(np.abs(z)) > 1.8 * LIM:
                break
            path.append(z.copy())
        branches.append(np.array(path))
    return branches


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "pdf.fonttype": 42,
    })

    fig = plt.figure(figsize=(FIGSIZE[0] * CM, FIGSIZE[1] * CM))
    fig.text(0.025, 0.995 if COMPACT else 0.975, PANEL_LETTER,
             ha="left", va="top", fontsize=15 if COMPACT else 13,
             fontweight="bold", color="black")
    ax = fig.add_axes(PANEL_BOX)
    if not STRIP:
        strips = ()
    else:
        # The two panels sit almost flush; their labels go on the outer edges.
        gap = 0.045 * BAND_BOX[2]
        width = (BAND_BOX[2] - gap) / 2
        strips = tuple(
            fig.add_axes([BAND_BOX[0] + k * (width + gap), BAND_BOX[1], width,
                          BAND_BOX[3]]) for k in range(2))

    if FILL:
        axis, label = basins()
        ax.pcolormesh(axis, axis, label,
                      cmap=ListedColormap([BASIN_LEFT, BASIN_RIGHT]),
                      shading="nearest", zorder=1, rasterized=True)

    if STYLE == "basincolor":
        # Colour each streamline by its basin.
        axis, label = basins(resolution=200, dt=0.015, n_steps=2200)
        g1, g2 = np.meshgrid(axis, axis)
        d1, d2 = field(g1, g2)
        flow = ax.streamplot(g1, g2, d1, d2, color=label,
                             cmap=ListedColormap([STREAM_LEFT, STREAM_RIGHT]),
                             norm=Normalize(0, 1), linewidth=0.8, density=1.1,
                             arrowsize=0.8, arrowstyle="-|>", zorder=3)
        flow.lines.set_alpha(STREAM_ALPHA)
        flow.arrows.set_alpha(STREAM_ALPHA)
    else:
        g1, g2 = np.meshgrid(np.linspace(-LIM, LIM, 90), np.linspace(-LIM, LIM, 90))
        d1, d2 = field(g1, g2)
        flow = ax.streamplot(g1, g2, d1, d2, color=STREAM, linewidth=0.75,
                             density=1.1, arrowsize=0.8, arrowstyle="-|>",
                             zorder=3)
        flow.lines.set_alpha(STREAM_ALPHA)
        flow.arrows.set_alpha(STREAM_ALPHA)

        for branch in separatrix():
            ax.plot(branch[:, 0], branch[:, 1], color=FRONTIER,
                    lw=FRONTIER_LW, ls=(0, (1.8, 2.4)), zorder=4)

    if MODE == "field":
        pass
    elif MODE == "samples":
        # Draw one mirror-symmetric sampled orbit per well.
        sample_data = {}
        for sign, colour in ((-1, TRAJ_LEFT), (1, TRAJ_RIGHT)):
            # Reflect the left-well start into the requested basin.
            orbit = sample_path(sign, seed=61 + (sign > 0))
            # Space deterministic samples by arc length.
            if SAMPLE_SIGMA > 0:
                idx = np.linspace(0, len(orbit) - 1, N_SAMPLES).round().astype(int)
            else:
                arc = np.concatenate([[0.0], np.cumsum(
                    np.linalg.norm(np.diff(orbit, axis=0), axis=1))])
                idx = np.searchsorted(arc, np.linspace(0, arc[-1], N_SAMPLES))
                idx = idx.clip(0, len(orbit) - 1)
            sample_data[sign] = (orbit, idx)

            if CONNECT:
                # Draw the full orbit behind its samples.
                ax.plot(orbit[:, 0], orbit[:, 1], color=colour, lw=1.5,
                        alpha=0.55, zorder=5)
            ax.scatter(orbit[idx, 0], orbit[idx, 1], s=26, color=colour,
                       zorder=6, linewidths=0)
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.9)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_aspect("equal")
    ax.tick_params(colors="black", labelsize=10, width=1.0, length=3.5)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_color("black")

    ax.set_xlabel("$z_1$", fontsize=12, color="black", labelpad=2)
    ax.set_ylabel("$z_2$", fontsize=12, color="black", labelpad=2)
    ax.set_title(TITLE, fontsize=10.5, fontweight="bold", color="black", pad=7)

    if strips and MODE == "samples":
        dt = 0.01
        widest = max(np.ptp(o[i, 0]) for o, i in sample_data.values())
        for k, sign in enumerate((-1, 1)):
            orbit, idx = sample_data[sign]
            colour = TRAJ_LEFT if sign < 0 else TRAJ_RIGHT
            axs = strips[k]
            time = np.arange(len(orbit)) * dt
            axs.axhline(sign * 1.0, color="#C9D0D8", lw=0.8, ls=(0, (4, 3)),
                        zorder=0)
            # Plot the same noisy observation as a line and samples.
            signal = orbit[:, 0] + OBS_NOISE * np.random.default_rng(
                40 + k).standard_normal(len(orbit))
            axs.plot(time, signal, color=colour, lw=1.5, alpha=0.45, zorder=4)
            axs.scatter(time[idx], signal[idx], s=22, color=colour, zorder=6,
                        linewidths=0)
            axs.set_facecolor(BG)
            for spine in axs.spines.values():
                spine.set_color(SPINE)
                spine.set_linewidth(0.9)
            axs.set_xlim(0, time[-1])
            # Same tick spacing as cell 2, so the time axes read alike.
            axs.set_xticks(np.arange(0, time[-1] + 1, 5))
            # Centre each panel on its attractor.
            half = widest * 1.20
            axs.set_ylim(sign * 1.0 - half, sign * 1.0 + half)
            # Put each state label on the corresponding outer edge.
            axs.set_yticks([sign * 1.0])
            axs.set_yticklabels([f"State {2 if sign > 0 else 1}"], fontsize=8.0)
            if k == 1:
                axs.yaxis.tick_right()
            axs.tick_params(colors="black", labelsize=9.5, width=1.0, length=3.0)
            for tick in axs.get_xticklabels() + axs.get_yticklabels():
                tick.set_color("black")
            axs.set_xlabel("$t$", fontsize=11.5, color="black", labelpad=1)
    if strips:
        top = max(strip.get_position().y1 for strip in strips)
        heading_gap = 0.004 if COMPACT else 0.012
        fig.text(BAND_BOX[0] + BAND_BOX[2] / 2, top + heading_gap, SUBTITLE,
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                 color="black")

    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/cell1.png")
    fig.savefig(out, dpi=400)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out}  (delta={DELTA}, fill={FILL})")


if __name__ == "__main__":
    main()
