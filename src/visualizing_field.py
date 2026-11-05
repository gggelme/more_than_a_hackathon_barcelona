"""Play the tactical ATD coordinates in real time: players and ball as points on plain x-y axes.

    python src/visualizing_field.py

What to show is set at the bottom of this file, in the `if __name__ == '__main__':` block.
Keys: space = play/pause.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, half_video_blocks, require  # noqa: E402
from visualizing_tracking import read_atd, slot_info  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]
TEAM_COLOURS = {"H": "tab:blue", "A": "tab:red"}
BALL = "goldenrod"


def run(start, seconds=60, speed=1.0):
    """Play video frames [start, start + seconds) as points. speed 2.0 is twice real time."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    stop = start + int(round(seconds * FPS))

    print("reading ATD frames {}..{} ...".format(start, stop - 1))
    atd = read_atd(raw_path, start, stop)
    colours = np.array([TEAM_COLOURS[team] for team, _ in slot_info(meta_path)])

    frames = np.arange(start, stop)
    ball = np.array([atd[f][1][:2] for f in frames])                 # (n, 2)
    slots = np.array([[s[:2] for s in atd[f][0]] for f in frames])   # (n, 46, 2)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(1.1, -0.1)          # y = 0 at the top, as in the video
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="black", linestyle="--", linewidth=1)  # pitch
    players = ax.scatter([], [], s=30)
    ball_dot = ax.scatter([], [], s=60, color=BALL, edgecolors="black", zorder=3)

    state = {"i": 0, "playing": True}

    def update(_):
        if state["playing"] and state["i"] < len(frames) - 1:
            state["i"] += 1
        i = state["i"]

        present = ~np.isnan(slots[i, :, 0])
        players.set_offsets(slots[i, present])
        players.set_facecolor(colours[present])

        bx, by = ball[i]
        ball_dot.set_offsets(np.empty((0, 2)) if np.isnan(bx) else [[bx, by]])
        ax.set_title("frame {}   ball x = {}   y = {}".format(
            frames[i], "NaN" if np.isnan(bx) else "{:.3f}".format(bx),
            "NaN" if np.isnan(by) else "{:.3f}".format(by)), family="monospace")
        return []

    def on_key(event):
        if event.key == " ":
            state["playing"] = not state["playing"]

    fig.canvas.mpl_connect("key_press_event", on_key)
    animation = FuncAnimation(fig, update, interval=max(1, int(1000 / (FPS * speed))),
                              cache_frame_data=False)
    plt.show()
    return animation


if __name__ == '__main__':
    first_half_kickoff = half_video_blocks(GAME)[0][1]     # 11185, first graded frame

    # -- what to show ------------------------------------------------------------------------
    START = 16000      # video frame to start from (minute * 1500 + second * 25)
    SECONDS = 2000                  # range loaded and played
    SPEED = 1.0                     # 2.0 = twice as fast

    run(start=START, seconds=SECONDS, speed=SPEED)
