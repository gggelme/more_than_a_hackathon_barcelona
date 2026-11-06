"""Possession from the vector classifier (src/tracking_possession/possession.py) as a submission CSV.

Run from the repository root:
    python src/experiments/vector_tracking_clasification.py

What to run is set at the bottom of this file, in the `if __name__ == "__main__":` block. It writes
evaluation/results/vector_tracking_clasification.csv -- 199319 rows, frame,label, the submission
format. The name does not start with "baseline_" on purpose: .gitignore keeps every CSV in that
folder out of git except "baseline_*.csv", and a submission does not belong in a public repository.

From the classifier to labels:
  - outside the two halves (before kick-off, half-time, after the final whistle): X;
  - inside each half: the team in possession by the rules -- the last team the classifier gave an
    owner to. A free ball, a ball in flight and a frame with no ATD ball keep it (CHALLENGE.md:
    possession stays with the last team that had it);
  - at the start of a half, before the first owner: D;
  - NO_BALL = "D" turns the frames with no ATD ball into D instead, as starter/baseline.py does --
    to compare. Dead ball is not detected otherwise yet.
The classifier restarts at each half, so possession never carries over half-time.

Only the ATD is read: no video, no YOLO. No labels or ground truth are used for building or tuning.
"""

import os
import re
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))   # src/experiments
SRC = os.path.dirname(HERE)                          # src
ROOT = os.path.dirname(SRC)                          # repository root
sys.path.insert(0, ROOT)                             # games, starter
sys.path.insert(0, SRC)                              # visualizing_tracking, tracking_possession

from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from starter.baseline import write_submission  # noqa: E402
from visualizing_tracking import slot_info  # noqa: E402
from tracking_possession.possession import PossessionConfig, classify  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
OUTPUT_NAME = "vector_tracking_clasification.csv"


def read_positions(raw_path, start, stop, n_slots):
    """(n, slots, 2) players and (n, 2) ball in normalised pitch units for video frames
    [start, stop), NaN where missing. Straight into arrays: a dict of tuples for a whole half
    would cost far more memory."""
    players = np.full((stop - start, n_slots, 2), np.nan)
    ball = np.full((stop - start, 2), np.nan)
    with open(raw_path) as handle:
        for line_number, line in enumerate(handle):
            if line_number < start:
                continue
            if line_number >= stop:
                break
            i = line_number - start
            _, slots, ball_text = line.strip().split(":")
            for slot, group in enumerate(slots.split(";")):
                if group[0] != "N":
                    x, y, _ = group.split(",")
                    players[i, slot] = (float(x), float(y))
            if ball_text[0] != "N":
                x, y, _ = ball_text.split(",")
                ball[i] = (float(x), float(y))
    return players, ball


def run(start=None, seconds=None, config=None, no_ball="keep", out=RESULTS_DIR):
    """Classify every half (or the part of it inside [start, start + seconds)) and write the CSV.

    start / seconds  None / None = the whole match. Frames outside the processed range are X.
    no_ball          "keep" (frames with no ATD ball keep the team) or "D" (they become D)
    """
    if no_ball not in ("keep", "D"):
        raise ValueError("no_ball must be 'keep' or 'D'")
    config = config or PossessionConfig()
    fps = GAMES[GAME]["fps"]
    pitch = np.array(GAMES[GAME]["pitch"])
    n_frames = source_frames()
    first = 0 if start is None else start
    last = n_frames if seconds is None else min(n_frames, first + int(round(seconds * fps)))

    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    teams = [team for team, _ in slot_info(meta_path)]

    labels = np.full(n_frames, "X", dtype="<U1")
    states, changed_vs_nearest, compared = Counter(), 0, 0
    for half, half_first, half_last in half_video_blocks(GAME):
        a, b = max(first, half_first), min(last, half_last + 1)
        if a >= b:
            continue
        print("half {}: reading ATD frames {}..{} ...".format(half, a, b - 1))
        players, ball = read_positions(raw_path, a, b, len(teams))
        print("half {}: classifying ...".format(half))
        results, *_ = classify(players * pitch, ball * pitch, teams, config, fps)

        for i, result in enumerate(results):
            states[result.state] += 1
            if result.state == "no_ball" and no_ball == "D":
                labels[a + i] = "D"
            else:
                labels[a + i] = result.team or "D"     # D before the half's first owner
            if result.owner is not None and result.nearest is not None:
                compared += 1
                changed_vs_nearest += teams[result.owner] != teams[result.nearest]

    path = write_submission(os.path.join(out, OUTPUT_NAME), labels)
    summarise(labels, states, changed_vs_nearest, compared, fps)
    print("Submission written: {}".format(path))
    print("Check it:  python evaluation/validate_submission.py {}".format(
        os.path.relpath(path, ROOT)))


def summarise(labels, states, changed_vs_nearest, compared, fps):
    """Label counts, possessions reported, classifier states."""
    text = "".join(labels)
    spells = [m.group() for m in re.finditer(r"H+|A+", text)]
    print("\nlabels: {}".format({k: int(v) for k, v in Counter(labels).items()}))
    if spells:
        print("possessions reported: {}  (mean {:.1f} s)".format(
            len(spells), np.mean([len(s) for s in spells]) / fps))
    print("classifier states: {}".format(dict(states)))
    if compared:
        print("frames where the owner's team differs from the nearest player's team: "
              "{} of {} ({:.1f}%)".format(changed_vs_nearest, compared,
                                          100.0 * changed_vs_nearest / compared))


if __name__ == "__main__":
    # -- what to run -------------------------------------------------------------------------
    START = 11185                    # None = whole match; or a video frame (minute*1500 + second*25)
    SECONDS = 1000                  # None = to the end; frames outside the range are X
    NO_BALL = "keep"                # "keep": no ATD ball keeps the team; "D": becomes D (baseline)

    # -- the classifier (metres, m/s), same as src/visualizing_possession.py ----------------
    CONFIG = PossessionConfig(
        smooth_frames=5,            # moving average on positions before differentiating
        velocity_frames=4,          # velocity over this many frames, centred
        d_max=3.0,                  # candidates: players within this distance of the ball
        d0=1.5,                     # distance scale of the score
        v0=2.0,                     # relative-speed scale of the score
        w_dir=0.5,                  # weight of the direction alignment
        moving=0.5,                 # below this speed there is no direction
        stick_slot=1.0,             # hysteresis: same slot as the previous owner
        stick_team=0.5,             # hysteresis: same team as the previous owner
        min_score=-2.0,             # the best candidate must reach this to be the owner
        flight_speed=10.0,          # faster than this, moving with nobody: in flight
    )
    OUT = RESULTS_DIR               # writes OUT/vector_tracking_clasification.csv

    run(start=START, seconds=SECONDS, config=CONFIG, no_ball=NO_BALL, out=OUT)
