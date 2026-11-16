"""Watch the dead-ball stretches of src/tracking_possession/goals.py -- goals, disallowed goals,
cooling breaks anchored on the Smart Tagging -- on top of the possession, drawn as in
src/visualizing_owner.py.

The possession is a touch module's (TOUCH_MODULE, e.g. possession_touch_4, with the keepers fixed
and keeper catches); goals.py then writes D over every stoppage it confirms. Display only:
nothing is saved. Per frame:
  - the ATD as in src/visualizing_tracking.py (keepers put right) and the classifier's OWNER;
  - top-left, a compact see-through panel: the final label, and inside a stoppage how it started
    ("shot", "lost at goal", "stoppage"), how it ended (kick off / set piece, refined by the ball
    moving again), who scored and how long the ball was dead; the active GOALS / SET PIECES tags;
  - below the video, a timeline with the final labels (with D), the touch module's alone and,
    with SHOW_GT, the public ground truth; then the GOALS and SET PIECES tags in lanes (restarts
    outlined in yellow).
Every stoppage of the processed stretch is printed with its evidence, and why each anchor was
kept or rejected -- to jump straight to each one.

The stretch is classified from the start of the half that holds START, and goes on up to
MAX_EPISODE_S past the range shown, so a stoppage whose restart is after the range is still found.

    python src/visualizing_goals.py

Moving through the video, as in src/visualizing_owner.py:
  - click or drag on the timeline to jump to that moment of the page;
  - , and .  one frame back / forward;   a and d  one second;   [ and ]  one page;
  - space or p (or a click on the video) pauses / plays, q or Esc quits.

What to show and every threshold are set at the bottom of this file, in the
`if __name__ == '__main__':` block. Keep SECONDS moderate: the ATD of the range is read into memory.
"""

import importlib
import inspect
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import (GAME, GAMES, anchor_video_frame, half_of_video_frame,  # noqa: E402
                   half_video_blocks, require)
from starter.load_homography import load as load_homography  # noqa: E402
from starter.load_smart_tagging import load as load_smart_tagging  # noqa: E402
from visualizing_tracking import (COLOURS, FONT, draw_frame as draw_tracking,  # noqa: E402
                                  labels_path, read_atd, read_labels, slot_info)
from visualizing_owner import draw_owner, play, read_players, timeline  # noqa: E402
from visualizing_smart_tagging import lanes, tag_timeline  # noqa: E402
from tracking_possession.possession import PossessionConfig  # noqa: E402
from tracking_possession.fix_goalkeepers import goalkeeper_slots  # noqa: E402
from tracking_possession.goals import GoalConfig, dead_ball, print_stoppages  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]
PITCH = np.array(GAMES[GAME]["pitch"])      # metres
DEAD = (160, 160, 160)                      # BGR: a stoppage, grey
PANEL_ALPHA = 0.5                           # how much of the video shows through the panel
TAG_CODES = ("goals", "set pieces")         # Smart Tagging codes drawn in the lanes


def draw_panel(image, label, stoppage, tags_now, frame):
    """Compact see-through panel: the final label, the stoppage the frame is in, the active tags."""
    lines = [("final label: {}".format(label), COLOURS.get(label, DEAD))]
    if stoppage is not None:
        lines.append(("D  {} -> {}  scorer {}  dead {:.0f} s  ({:.1f} s in)".format(
            stoppage.start_case, stoppage.end_case, stoppage.scorer or "?", stoppage.dead_s,
            (frame - stoppage.start) / FPS), DEAD))
    for e in tags_now:
        lines.append(("{}  {}  {}".format(e["code"], e["tags"].get("Team", "N/A"),
                                          e["tags"].get("Type", "")), (0, 255, 255)))
    scale, step, x0, y0 = 0.55, 24, 12, 12
    text_width = max(cv2.getTextSize(text, FONT, scale, 1)[0][0] for text, _ in lines)
    x1 = min(image.shape[1] - 1, x0 + text_width + 16)
    y1 = y0 + step * len(lines) + 8
    image[y0:y1, x0:x1] = (image[y0:y1, x0:x1] * (1 - PANEL_ALPHA)).astype(np.uint8)
    for k, (text, colour) in enumerate(lines):
        y = y0 + 20 + step * k
        cv2.putText(image, text, (x0 + 8, y), FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, text, (x0 + 8, y), FONT, scale, colour, 1, cv2.LINE_AA)


def run(start, seconds=120, config=None, touch_module="possession_touch_4", touch_overrides=None,
        goal_config=None, fix_keepers=True, show_gt=True, scale=0.75, timeline_seconds=60):
    """Play video frames [start, start + seconds) with the possession, the stoppages and the tags."""
    config = config or PossessionConfig()
    goal_config = goal_config or GoalConfig()
    touch = importlib.import_module("tracking_possession." + touch_module)
    touch_config = touch.TouchConfig(**(touch_overrides or {}))
    video_path, meta_path, raw_path = require(["video", "atd_meta", "atd_raw"])
    halves = [(hf, hl) for _, hf, hl in half_video_blocks(GAME) if hf <= start <= hl]
    if not halves:
        raise ValueError("START must be inside a half: {}".format(half_video_blocks(GAME)))
    half_first, half_last = halves[0]
    # games.py's kick-off anchor of this half (goals.py only searches from it, it can be early)
    kickoff_hint = anchor_video_frame(
        GAMES[GAME]["kickoff" if half_of_video_frame(GAME, start) == 1 else "kickoff_2nd"])
    stop = min(start + int(round(seconds * FPS)), half_last + 1)
    # read on past the range, so a stoppage whose restart comes after it is still found
    read_stop = (half_last + 1 if goal_config.max_episode_s is None else
                 min(half_last + 1, stop + int(round(goal_config.max_episode_s * FPS))))

    print("reading homography ...")
    matrices, _, valid = load_homography()
    print("reading ATD frames {}..{} ...".format(start, stop - 1))
    atd = read_atd(raw_path, start, stop)
    info = slot_info(meta_path)
    teams = [team for team, _ in info]
    print("reading ATD frames {}..{} (from the half's start, to classify) ...".format(
        half_first, read_stop - 1))
    players, ball = read_players(raw_path, half_first, read_stop, info, fix_keepers, atd)

    print("classifying with {}.py ...".format(touch_module))
    extra = {}
    parameters = inspect.signature(touch.classify).parameters
    if fix_keepers and "keepers" in parameters:
        extra = dict(keepers=goalkeeper_slots(info), pitch_length=PITCH[0])
        if "pitch_width" in parameters:
            extra["pitch_width"] = PITCH[1]
        if "kickoff" in parameters:                 # the formation at the half's kick-off
            extra["kickoff"] = kickoff_hint - half_first
    results, _, _, _, _, touch_info = touch.classify(players, ball, teams, config, FPS,
                                                     touch_config, **extra)

    events = load_smart_tagging()               # for the tag lanes below; goals.py gets the same
    touch_labels = [r.team or "D" for r in results]
    stats = {}
    stoppages, final = dead_ball(touch_labels, touch, touch_info, touch_config, ball, half_first,
                                 PITCH[1], goal_config, FPS, events, stats, players=players,
                                 teams=teams, half_last=half_last, kickoff_hint=kickoff_hint)
    print()
    print_stoppages(stoppages, stats, half_first, FPS)
    offset, end = start - half_first, stop - half_first
    rows = [("with D", {half_first + i: final[i] for i in range(offset, end)}),
            (touch_module, {half_first + i: touch_labels[i] for i in range(offset, end)})]
    if show_gt:
        rows.append(("GT", read_labels(labels_path("gt"), start, stop)))

    shown_tags = sorted((e for e in events if (e["code"] or "").strip().lower() in TAG_CODES
                         and e["end_frame"] > start and e["start_frame"] < stop),
                        key=lambda e: e["start_frame"])
    lane_of = lanes(shown_tags)
    n_lanes = max(lane_of) + 1 if lane_of else 1
    window = int(round(timeline_seconds * FPS))

    def compose(image, frame):
        """(window image, height of the video part) for one frame."""
        i = frame - half_first
        if valid[frame] and frame in atd:
            slots, ball_atd = atd[frame]
            draw_tracking(image, matrices[frame], slots, ball_atd, info, frame, FPS)
            draw_owner(image, matrices[frame], slots, results[i].owner)
        stoppage = next((s for s in stoppages if s.start <= i < s.end), None)
        tags_now = [e for e in shown_tags if e["start_frame"] <= frame < e["end_frame"]]
        draw_panel(image, final[i], stoppage, tags_now, i)
        top = cv2.resize(image, None, fx=scale, fy=scale)
        width = top.shape[1]
        return np.vstack([top, timeline(rows, frame, start, stop, window, width),
                          tag_timeline(shown_tags, lane_of, n_lanes, frame, start, window,
                                       width)]), top.shape[0]

    play(video_path, compose, start, stop, window, "goals and stoppages: " + touch_module)


if __name__ == '__main__':
    first_half_kickoff = half_video_blocks(GAME)[0][1]     # 11185, first graded frame

    # -- what to show ------------------------------------------------------------------------
    START = 46000      # video frame to start from (minute * 1500 + second * 25)
    SECONDS = 120                   # how long; keep it moderate (the ATD is read into memory)
    SHOW_GT = True                  # public ground truth as a timeline row (first 5 min only)
    SCALE = 0.75                    # window size relative to 1920x1080
    TIMELINE_SECONDS = 60           # length of one timeline page
    FIX_GOALKEEPERS = True          # keepers back in their own team's slot (fix_goalkeepers.py)

    # -- the possession ----------------------------------------------------------------------
    TOUCH_MODULE = "possession_touch_4"     # any src/tracking_possession/possession_touch_*.py
    TOUCH_OVERRIDES = {}                    # fields changed on that module's TouchConfig defaults
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

    # -- the stoppages (metres, m/s, seconds) -------------------------------------------------
    GOAL_CONFIG = GoalConfig(
        max_episode_s=None,         # no time limit between a goal and its restart ...
        max_live_s=30.0,            # ... real play inside (ball live this long in a row) rejects it
        look_before_s=30.0,         # the stoppage may start this long before the GOALS tag
        min_stop_s=10.0,            # dead ball (missing or still) needed: rejects tags in live play
        still_speed=0.5,            # slower than this the ball is still
        blip_s=1.0,                 # a live ball shorter than this inside a stoppage is flicker
        restart_speed=2.0,          # the ball moving again faster than this is the restart ...
        restart_search_s=3.0,       # ... searched this long around the restart tag
        link_s=5.0,                 # a shot this close to the start of the stoppage caused it
        shot_speed=8.0,             # m/s: a kick this fast at the goal mouth was a shot
        goal_margin=2.0,            # the goal mouth (7.32 m) give or take this much
    )

    run(start=START, seconds=SECONDS, config=CONFIG, touch_module=TOUCH_MODULE,
        touch_overrides=TOUCH_OVERRIDES, goal_config=GOAL_CONFIG, fix_keepers=FIX_GOALKEEPERS,
        show_gt=SHOW_GT, scale=SCALE, timeline_seconds=TIMELINE_SECONDS)
