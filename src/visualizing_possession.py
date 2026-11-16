"""Compare the possession classifier (src/tracking_possession/possession.py) with its touch pass
(src/tracking_possession/possession_touch.py) on the tactical video, drawn as in
src/visualizing_owner.py.

possession_touch.py keeps possession.py's frame-by-frame owner and state and only replaces the TEAM
in possession, so both are drawn on one video. Display only: nothing is saved. Per frame:
  - the ATD as in src/visualizing_tracking.py: every tracked slot as a dot with its number, the
    ball, and the bottom strip;
  - the OWNER chosen by the classifier with a green ring and its slot number;
  - with SHOW_RELATIVE_SPEED, under every tracked player |v_player - v_ball| in m/s ("--" where a
    velocity is unknown);
  - with SHOW_TOUCHES, the touches possession_touch.py detects (a break in the ball's trajectory
    next to a player): an orange ring on the toucher for TOUCH_SHOW_S after it, with its team and
    velocity change -- "duel" when it was inside one (players of both teams within REACH);
  - top-left, the team in possession with possession_touch.py (blue H, red A), and under it the
    state and owner, the team with each module -- DIFFER when they disagree --, DUEL when both
    teams are at the ball, and possession_touch.py's last change with its reason;
  - below the video, a timeline per module and with SHOW_GT the public ground truth, paged as in
    src/visualizing_owner.py.

Both are run from the start of the half that holds START, so their memory (hysteresis, pending
changes) is as on the whole match. The range must lie inside one half.

    python src/visualizing_possession.py

Moving through the video, as in src/visualizing_owner.py:
  - click or drag on the timeline to jump to that moment of the page;
  - , and .  one frame back / forward;   a and d  one second;   [ and ]  one page;
  - space or p (or a click on the video) pauses / plays, q or Esc quits.

What to show and every weight are set at the bottom of this file, in the `if __name__ == '__main__':`
block -- keep CONFIG equal to the experiment's to see the same classifier.
"""

import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))   # vector_tracking_clasification
from games import GAME, GAMES, half_video_blocks, require  # noqa: E402
from starter.load_homography import load as load_homography  # noqa: E402
from visualizing_tracking import (FONT, draw_frame as draw_tracking, draw_label_box,  # noqa: E402
                                  labels_path, read_atd, read_labels, slot_info)
from visualizing_owner import (OWNER, draw_owner, draw_relative_speeds, pixel, play,  # noqa: E402
                               read_players, timeline)
from tracking_possession.possession import PossessionConfig, classify  # noqa: E402
from tracking_possession import possession_touch  # noqa: E402
from tracking_possession.possession_touch import TouchConfig  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]
PITCH = np.array(GAMES[GAME]["pitch"])      # metres
TOUCH = (0, 165, 255)                       # BGR: a detected touch, orange
DIFFER = (0, 0, 255)                        # red


def relative_speeds(player_v, ball_v):
    """(slots,) |v_player - v_ball| in m/s for one frame, NaN where a velocity is unknown."""
    out = np.full(len(player_v), np.nan)
    if np.all(np.isfinite(ball_v)):
        known = np.all(np.isfinite(player_v), axis=1)
        out[known] = np.linalg.norm(player_v[known] - ball_v, axis=1)
    return out


def draw_touches(image, matrix, i, touches, players, show_frames):
    """Orange ring on the toucher of every touch in the last `show_frames` frames."""
    height, width = image.shape[:2]
    for touch in touches:
        if not 0 <= i - touch.frame < show_frames:
            continue
        point = pixel(matrix, players[i, touch.slot] / PITCH, width, height)
        if point is None:
            continue
        cv2.circle(image, point, 28, TOUCH, 3, cv2.LINE_AA)
        text = "TOUCH {} {:.1f} m/s{}".format(touch.team, touch.dv, "  duel" if touch.in_duel else "")
        cv2.putText(image, text, (point[0] + 30, point[1] + 30), FONT, 0.6, (0, 0, 0), 5,
                    cv2.LINE_AA)
        cv2.putText(image, text, (point[0] + 30, point[1] + 30), FONT, 0.6, TOUCH, 2, cv2.LINE_AA)


def draw_state(image, result, team_base, duel, change):
    """Lines under the team box, in the style of visualizing_owner.draw_state."""
    lines = [
        ("state: {}   owner: {}".format(
            result.state, "--" if result.owner is None else "#{}".format(result.owner + 1)),
         OWNER),
        ("possession.py: {}   possession_touch.py: {}".format(team_base or "--",
                                                              result.team or "--"), OWNER),
    ]
    if team_base != result.team:
        lines.append(("DIFFER", DIFFER))
    if duel:
        lines.append(("DUEL: both teams at the ball", TOUCH))
    if change is not None:
        frame, team, reason = change
        lines.append(("last change: {} ({}) at {:.1f}s".format(team, reason, frame / FPS),
                      (235, 235, 235)))
    width = image.shape[1]
    bottom = 90 + 34 * len(lines) + 8
    cv2.rectangle(image, (15, 90), (min(width - 15, 820), bottom), (0, 0, 0), -1)
    for k, (text, colour) in enumerate(lines):
        cv2.putText(image, text, (25, 116 + 34 * k), FONT, 0.7, colour, 2, cv2.LINE_AA)


def run(start, seconds=60, config=None, touch_config=None, show_relative_speed=True,
        show_touches=True, touch_show_seconds=0.5, scale=0.75, show_gt=True, timeline_seconds=20,
        fix_keepers=True):
    """Play video frames [start, start + seconds) with both modules' decisions drawn on top and
    their timelines below. fix_keepers puts each keeper back in his own team's slot first
    (fix_goalkeepers.py), for both modules and the drawing."""
    config = config or PossessionConfig()
    touch_config = touch_config or TouchConfig()
    video_path, meta_path, raw_path = require(["video", "atd_meta", "atd_raw"])
    halves = [(hf, hl) for _, hf, hl in half_video_blocks(GAME) if hf <= start <= hl]
    if not halves:
        raise ValueError("START must be inside a half: {}".format(half_video_blocks(GAME)))
    half_first, half_last = halves[0]
    stop = min(start + int(round(seconds * FPS)), half_last + 1)

    print("reading homography ...")
    matrices, _, valid = load_homography()
    print("reading ATD frames {}..{} ...".format(start, stop - 1))
    atd = read_atd(raw_path, start, stop)
    info = slot_info(meta_path)
    teams = [team for team, _ in info]
    print("reading ATD frames {}..{} (from the half's start, to classify) ...".format(
        half_first, stop - 1))
    players, ball = read_players(raw_path, half_first, stop, info, fix_keepers, atd)

    print("classifying with possession.py ...")
    base, *_ = classify(players, ball, teams, config, FPS)
    print("classifying with possession_touch.py ...")
    results, players_s, player_v, _, ball_v, touch_info = possession_touch.classify(
        players, ball, teams, config, FPS, touch_config)
    possession_touch.summarise(touch_info, FPS, offset=half_first)
    offset = start - half_first
    differ = sum(base[i].team != results[i].team for i in range(offset, stop - half_first))
    print("frames where the team differs in the range shown: {} of {}".format(differ,
                                                                              stop - start))

    # The CSV's labels: the team in possession, D before a half's first owner
    rows = [("possession", {half_first + i: base[i].team or "D"
                            for i in range(offset, stop - half_first)}),
            ("possession_touch", {half_first + i: results[i].team or "D"
                                  for i in range(offset, stop - half_first)})]
    if show_gt:
        rows.append(("GT", read_labels(labels_path("gt"), start, stop)))
    window = int(round(timeline_seconds * FPS))
    show_frames = int(round(touch_show_seconds * FPS))
    change_frames = [frame for frame, _, _ in touch_info.changes]

    def compose(image, frame):
        """(window image, height of the video part) for one frame."""
        i = frame - half_first
        result = results[i]
        if valid[frame] and frame in atd:
            matrix = matrices[frame]
            slots, ball_atd = atd[frame]
            draw_tracking(image, matrix, slots, ball_atd, info, frame, FPS)
            if show_relative_speed:
                draw_relative_speeds(image, matrix, slots, result.owner,
                                     relative_speeds(player_v[i], ball_v[i]))
            if show_touches:
                draw_touches(image, matrix, i, touch_info.touches, players_s, show_frames)
            draw_owner(image, matrix, slots, result.owner)
        draw_label_box(image, "touch", result.team or "X")
        k = int(np.searchsorted(change_frames, i, side="right")) - 1
        change = None
        if k >= 0:
            change_frame, team, reason = touch_info.changes[k]
            change = (change_frame + half_first, team, reason)
        draw_state(image, result, base[i].team, bool(touch_info.duel[i]), change)
        top = cv2.resize(image, None, fx=scale, fy=scale)
        return np.vstack([top, timeline(rows, frame, start, stop, window, top.shape[1])]), \
            top.shape[0]

    play(video_path, compose, start, stop, window, "possession vs possession_touch")


if __name__ == '__main__':
    first_half_kickoff = half_video_blocks(GAME)[0][1]     # 11185, first graded frame

    # -- what to show ------------------------------------------------------------------------
    START = first_half_kickoff      # video frame to start from (minute * 1500 + second * 25)
    SECONDS = 300                   # how long to play
    SCALE = 0.75                    # window size relative to 1920x1080
    SHOW_RELATIVE_SPEED = True      # |v_player - v_ball| (m/s) under every tracked player
    SHOW_TOUCHES = True             # orange ring on the player of each detected touch
    TOUCH_SHOW_SECONDS = 0.5        # how long a touch stays marked
    SHOW_GT = True                  # public ground truth as a third timeline row
    TIMELINE_SECONDS = 20           # length of one timeline page

    # -- the classifier (metres, m/s), same as src/experiments/vector_tracking_clasification.py
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
    # -- the touch pass (metres, m/s, seconds), as in possession_touch.TouchConfig ------------
    TOUCH_CONFIG = TouchConfig(
        reach=1.5,                  # duel: a player of each team this close to the ball
        touch_reach=2.0,            # a touch needs a player this close to the break
        half_window=6,              # frames fitted on each side of a candidate touch (0.24 s)
        min_points=4,               # ball positions needed on EACH side to fit a line
        dv_min=4.0,                 # velocity change (m/s) friction and noise cannot explain
        max_link_s=3.0,             # a claim and its confirmation farther apart: not one play
        min_control_s=1.0,          # a rival control with no touch seen confirms after this long
        control_gap=5,              # frames of missing owner tolerated inside one run of control
    )

    FIX_GOALKEEPERS = True          # keepers back in their own team's slot (fix_goalkeepers.py)

    run(start=START, seconds=SECONDS, config=CONFIG, touch_config=TOUCH_CONFIG,
        show_relative_speed=SHOW_RELATIVE_SPEED, show_touches=SHOW_TOUCHES,
        touch_show_seconds=TOUCH_SHOW_SECONDS, scale=SCALE, show_gt=SHOW_GT,
        timeline_seconds=TIMELINE_SECONDS, fix_keepers=FIX_GOALKEEPERS)
