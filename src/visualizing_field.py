"""Play the tactical ATD coordinates in real time: players and ball as points on plain x-y axes,
each with an arrow showing its direction of movement, and the possession timeline underneath.

The velocities are computed exactly as the possession classifier does
(tracking_possession.possession.prepare: smoothed positions, centred difference, in metres), so the
arrows are what the classifier sees. A player arrow ends where it would be PLAYER_ARROW_SECONDS
ahead, the ball's BALL_ARROW_SECONDS ahead (much longer, only to be visible). Anything slower than
MIN_SPEED gets no arrow.

The timeline below shows one row per entry of ROWS -- "gt" is the public ground truth, anything else
a file in evaluation/results/ without .csv -- coloured H blue, A red, D grey, X white. It is paged
like test/visualizing_compare.py: a block of TIMELINE_SECONDS, the cursor walking across it.

    python src/visualizing_field.py

What to show is set at the bottom of this file, in the `if __name__ == '__main__':` block.
Moving through it:
  - click or drag on the timeline to jump to that moment of the page;
  - , and .  one frame back / forward;   a and d  one second;   [ and ]  one page;
  - space or p pauses / plays. At the end of the range it pauses.

The public ground truth is for looking, not for tuning (info_hackathon/RULES.md).
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, half_video_blocks, require  # noqa: E402
from visualizing_tracking import labels_path, read_atd, read_labels, slot_info  # noqa: E402
from tracking_possession.possession import PossessionConfig, prepare  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]
PITCH = np.array(GAMES[GAME]["pitch"])      # metres
TEAM_COLOURS = {"H": "tab:blue", "A": "tab:red"}
BALL = "goldenrod"
MIN_SPEED = 0.5                             # m/s: slower than this, no arrow
LABEL_CODES = {"H": 0, "A": 1, "D": 2, "X": 3}
LABEL_CMAP = ListedColormap(["tab:blue", "tab:red", "grey", "white"])

# Keys this viewer uses; matplotlib binds some of them by default (p = pan)
for _keymap in ("keymap.pan",):
    if "p" in plt.rcParams[_keymap]:
        plt.rcParams[_keymap].remove("p")


def arrows(velocity, seconds):
    """Arrow components in normalised units: the velocity (m/s) times `seconds`, 0 where too slow
    or unknown, so the arrow set keeps a fixed size from frame to frame."""
    speed = np.linalg.norm(velocity, axis=-1)
    keep = np.isfinite(speed) & (speed >= MIN_SPEED)
    out = np.where(keep[..., None], velocity / PITCH * seconds, 0.0)
    return out[..., 0], out[..., 1]


def row_name(name):
    return "GT" if name == "gt" else name.replace(".csv", "")


def run(start, seconds=60, speed=1.0, player_arrow_seconds=1.0, ball_arrow_seconds=4.0,
        config=None, rows=("gt",), timeline_seconds=20):
    """Play video frames [start, start + seconds) as points with direction arrows and a timeline.

    speed                 2.0 is twice real time
    player_arrow_seconds  player arrow = where it would be this many seconds ahead
    ball_arrow_seconds    the same for the ball, longer only to be visible
    config                smoothing / velocity window, as in the possession classifier
    rows                  timeline rows: "gt" or names of files in evaluation/results/
    timeline_seconds      length of one timeline page
    """
    config = config or PossessionConfig()
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    stop = start + int(round(seconds * FPS))

    # A margin on both sides, so smoothing and centred velocities exist at the range's edges
    margin = config.smooth_frames + config.velocity_frames
    first = max(0, start - margin)
    print("reading ATD frames {}..{} ...".format(first, stop + margin - 1))
    atd = read_atd(raw_path, first, stop + margin)
    colours = np.array([TEAM_COLOURS[team] for team, _ in slot_info(meta_path)])

    loaded = np.arange(first, first + len(atd))
    ball_m = np.array([atd[f][1][:2] for f in loaded], dtype=np.float64) * PITCH
    slots_m = np.array([[s[:2] for s in atd[f][0]] for f in loaded], dtype=np.float64) * PITCH
    _, slot_v, _, ball_v = prepare(slots_m, ball_m, config, FPS)   # velocities in m/s

    # Keep only the frames to play; positions back to normalised, as the axes are
    shown = slice(start - first, min(stop, loaded[-1] + 1) - first)
    frames = loaded[shown]
    ball, slots = ball_m[shown] / PITCH, slots_m[shown] / PITCH
    slot_v, ball_v = slot_v[shown], ball_v[shown]
    stop = frames[-1] + 1

    # Timeline rows: label codes over the played range
    names = [row_name(name) for name in rows]
    codes = np.array([[LABEL_CODES.get(labels.get(f, "X"), 3) for f in frames]
                      for labels in (read_labels(labels_path(name), start, stop) for name in rows)])
    window = int(round(timeline_seconds * FPS))

    fig = plt.figure(figsize=(12, 9))
    grid = fig.add_gridspec(2, 1, height_ratios=[8, 0.4 + 0.35 * len(rows)], hspace=0.25)
    ax = fig.add_subplot(grid[0])
    ax_t = fig.add_subplot(grid[1])

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(1.1, -0.1)          # y = 0 at the top, as in the video
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="black", linestyle="--", linewidth=1)  # pitch
    players = ax.scatter([], [], s=30)
    ball_dot = ax.scatter([], [], s=60, color=BALL, edgecolors="black", zorder=3)
    # One arrow per slot (zero-length when absent or still) and one for the ball. angles="xy" and
    # scale_units="xy" make the arrows live in data units, so they point the right way even with
    # the y axis inverted.
    n_slots = slots.shape[1]
    player_arrows = ax.quiver(np.zeros(n_slots), np.zeros(n_slots), np.zeros(n_slots),
                              np.zeros(n_slots), color=colours, angles="xy", scale_units="xy",
                              scale=1, width=0.002, zorder=2)
    ball_arrow = ax.quiver([0], [0], [0], [0], color=BALL, angles="xy", scale_units="xy", scale=1,
                           width=0.005, edgecolor="black", linewidth=0.5, zorder=4)

    # Timeline: one page at a time, x in video frames, so a click gives the frame directly
    strip = ax_t.imshow(codes[:, :window], aspect="auto", cmap=LABEL_CMAP, vmin=0, vmax=3,
                        interpolation="nearest")
    cursor = ax_t.axvline(start, color="gold", linewidth=2)
    ax_t.set_yticks(range(len(names)))
    ax_t.set_yticklabels(names)
    ax_t.set_xlabel("video frame  (click / drag to jump;  , .  a d  [ ]  to move;  space / p to pause)")

    state = {"i": 0, "playing": True, "page": None}

    def show_page(i):
        """Redraw the timeline strip when the frame falls on a new page."""
        page = i // window
        if page == state["page"]:
            return
        state["page"] = page
        a, b = page * window, min(len(frames), (page + 1) * window)
        strip.set_data(codes[:, a:b])
        strip.set_extent((frames[a] - 0.5, frames[a] + (b - a) - 0.5, len(names) - 0.5, -0.5))
        ax_t.set_xlim(frames[a] - 0.5, frames[a] + window - 0.5)

    def update(_):
        if state["playing"]:
            if state["i"] < len(frames) - 1:
                state["i"] += 1
            else:
                state["playing"] = False       # stay on the last frame instead of closing
        i = state["i"]

        present = ~np.isnan(slots[i, :, 0])
        players.set_offsets(slots[i, present])
        players.set_facecolor(colours[present])
        player_arrows.set_offsets(np.nan_to_num(slots[i]))
        player_arrows.set_UVC(*arrows(slot_v[i], player_arrow_seconds))

        bx, by = ball[i]
        ball_dot.set_offsets(np.empty((0, 2)) if np.isnan(bx) else [[bx, by]])
        ball_arrow.set_offsets([[np.nan_to_num(bx), np.nan_to_num(by)]])
        ball_arrow.set_UVC(*arrows(ball_v[i:i + 1], ball_arrow_seconds))

        show_page(i)
        cursor.set_xdata([frames[i], frames[i]])
        labels_now = "  ".join("{} {}".format(name, "HADX"[codes[r, i]])
                               for r, name in enumerate(names))
        ax.set_title("frame {}   ball x = {}   y = {}   |   {}".format(
            frames[i], "NaN" if np.isnan(bx) else "{:.3f}".format(bx),
            "NaN" if np.isnan(by) else "{:.3f}".format(by), labels_now), family="monospace")
        return []

    def jump(i):
        state["i"] = int(min(len(frames) - 1, max(0, i)))

    def on_key(event):
        steps = {",": -1, ".": 1, "a": -FPS, "d": FPS, "[": -window, "]": window}
        if event.key in (" ", "p"):
            state["playing"] = not state["playing"]
        elif event.key in steps:
            jump(state["i"] + steps[event.key])

    def on_mouse(event):
        dragging = event.name == "motion_notify_event" and event.button == 1
        if event.inaxes is ax_t and event.xdata is not None and \
                (event.name == "button_press_event" or dragging):
            jump(int(round(event.xdata)) - frames[0])

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_mouse)
    fig.canvas.mpl_connect("motion_notify_event", on_mouse)
    animation = FuncAnimation(fig, update, interval=max(1, int(1000 / (FPS * speed))),
                              cache_frame_data=False)
    plt.show()
    return animation


if __name__ == '__main__':
    first_half_kickoff = half_video_blocks(GAME)[0][1]     # 11185, first graded frame

    # -- what to show ------------------------------------------------------------------------
    START = first_half_kickoff      # video frame to start from (minute * 1500 + second * 25)
    SECONDS = 2000                  # range loaded and played
    SPEED = 1.0                     # 2.0 = twice as fast
    PLAYER_ARROW_SECONDS = 1.0      # player arrow = where it would be 1 s ahead
    BALL_ARROW_SECONDS = 4.0        # ball arrow, much longer to see its direction well
    ROWS = ["gt"]                   # timeline rows; add e.g. "vector_tracking_clasification"
    TIMELINE_SECONDS = 20           # length of one timeline page
    CONFIG = PossessionConfig(      # same velocity computation as the possession classifier
        smooth_frames=5,            # moving average on positions before differentiating
        velocity_frames=4,          # velocity over this many frames, centred
    )

    run(start=START, seconds=SECONDS, speed=SPEED, player_arrow_seconds=PLAYER_ARROW_SECONDS,
        ball_arrow_seconds=BALL_ARROW_SECONDS, config=CONFIG, rows=ROWS,
        timeline_seconds=TIMELINE_SECONDS)
