"""Watch who the vector classifier (src/tracking_possession/possession.py) gives the ball to, on the
tactical video.

Display only: nothing is saved. Per frame:
  - the ATD as in src/visualizing_tracking.py: every tracked slot as a dot with its number, the
    ball, and the bottom strip;
  - the OWNER chosen by classify() with a green ring and its slot number;
  - with SHOW_RELATIVE_SPEED, under every tracked player |v_player - v_ball| in m/s, from the SAME
    smoothed velocities classify() uses ("--" where a velocity is unknown -- nothing is made up).
    It is the relative speed, not the player's speed nor the score;
  - top-left, the team in possession by the rules (the last team that had an owner), coloured as
    in test/visualizing_compare.py -- blue for H, red for A -- and under it the state
    (possession / free / flight / no_ball), the owner and the team;
  - below the video, a timeline of the possession -- H blue, A red, D grey, X white -- labelled as
    the experiment's CSV (D before a half's first owner, X outside the halves), and with SHOW_GT
    the public ground truth under it. It is paged as in test/visualizing_compare.py: a fixed block
    of TIMELINE_SECONDS, the cursor walking across it; to its right is what comes next.

The classifier is run from the start of the half that holds START, not from START itself, so the
hysteresis (stick_slot, stick_team) is in the same state as in the CSV of
src/experiments/vector_tracking_clasification.py. Frames outside the halves have no decision.
With FIX_GOALKEEPERS each keeper is first put back in his own team's slot
(src/tracking_possession/fix_goalkeepers.py), for the classifier and for the drawing alike.

    python src/visualizing_owner.py

Moving through the video, as in test/visualizing_compare.py:
  - click or drag on the timeline to jump to that moment of the page;
  - , and .  one frame back / forward;   a and d  one second;   [ and ]  one page;
  - space or p (or a click on the video) pauses / plays, q or Esc quits. At the end of the range
    it pauses instead of closing. Keys only reach the window while it has the focus.

What to show and every weight of the classifier are set at the bottom of this file, in the
`if __name__ == '__main__':` block -- keep CONFIG equal to the experiment's to see the same
classifier.
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
from starter.load_homography import load as load_homography, project  # noqa: E402
from visualizing_tracking import (COLOURS, FONT, check_mirror,  # noqa: E402
                                  draw_frame as draw_tracking, draw_label_box, inside,
                                  keepers_into_atd, kickoff_frame, labels_path,
                                  print_goalkeepers, read_atd, read_labels, slot_info)
from tracking_possession.possession import PossessionConfig, classify  # noqa: E402
from tracking_possession.fix_goalkeepers import fix_goalkeepers, goalkeeper_slots  # noqa: E402
from vector_tracking_clasification import read_positions  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]
PITCH = np.array(GAMES[GAME]["pitch"])      # metres
OWNER = (0, 255, 110)                       # BGR: the classifier's owner, green
OTHERS = (235, 235, 235)                    # relative speed of the other players
TIMELINE_ROW = 26                           # pixel height of one timeline row
LABEL_BGR = {"H": COLOURS["H"], "A": COLOURS["A"], "D": COLOURS["D"], "X": (255, 255, 255)}


def read_players(raw_path, half_first, stop, info, fix_keepers, atd=None):
    """(players, ball) in metres for video frames [half_first, stop) of one half, with each keeper
    put back in his own team's slot (tracking_possession/fix_goalkeepers) when fix_keepers -- the
    goal each team defends measured on those frames. The fixed keepers are also written into `atd`
    (the frames drawn), so what is drawn is what is classified. Shared by the visualizers that
    classify."""
    players, ball = read_positions(raw_path, half_first, stop, len(info))
    players, ball = players * PITCH, ball * PITCH
    if fix_keepers:
        keepers = goalkeeper_slots(info)
        # the stretch starts at the half's first frame; its KICK-OFF (games.py's anchor, maybe
        # minutes later) says who defends which goal
        half = [h for h, hf, _ in half_video_blocks(GAME) if hf == half_first]
        kickoff = kickoff_frame(half[0]) - half_first if half else None
        players, report = fix_goalkeepers(players, keepers, PITCH[0],
                                          teams=[team for team, _ in info], kickoff=kickoff)
        print_goalkeepers([(half[0] if half else "?", report)])
        if half:
            check_mirror(raw_path, info, half[0], report.goals)
        if atd is not None:
            keepers_into_atd(atd, players, half_first, keepers)
    return players, ball


def decisions(raw_path, start, stop, info, config, fix_keepers=True, atd=None):
    """{video_frame: (FrameResult, relative speeds (slots,) in m/s)} for frames [start, stop).

    Each half is classified from its first frame, so the hysteresis is the experiment's; with
    fix_keepers the keepers are put right first (and in `atd`, for drawing).
    """
    teams = [team for team, _ in info]
    out = {}
    for half, half_first, half_last in half_video_blocks(GAME):
        a, b = max(start, half_first), min(stop, half_last + 1)
        if a >= b:
            continue
        print("classifying half {} from its start: frames {}..{} ...".format(half, half_first,
                                                                            b - 1))
        players, ball = read_players(raw_path, half_first, b, info, fix_keepers, atd)
        results, _, player_v, _, ball_v = classify(players, ball, teams, config, FPS)
        for frame in range(a, b):
            i = frame - half_first
            relative = np.full(len(teams), np.nan)
            if np.all(np.isfinite(ball_v[i])):
                known = np.all(np.isfinite(player_v[i]), axis=1)
                relative[known] = np.linalg.norm(player_v[i, known] - ball_v[i], axis=1)
            out[frame] = (results[i], relative)
    return out


def pixel(matrix, slot, width, height):
    """Pixel (u, v) of a slot (normalised x, y, ...) inside the image, or None."""
    x, y = slot[:2]
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    u, v = project(matrix, float(x), float(y))
    if not inside(u, v, width, height):
        return None
    return int(round(float(u))), int(round(float(v)))


def draw_relative_speeds(image, matrix, slots, owner, relative):
    """|v_player - v_ball| under every tracked player; the owner's in green."""
    height, width = image.shape[:2]
    for slot, xy in enumerate(slots):
        point = pixel(matrix, xy, width, height)
        if point is None:
            continue
        value = relative[slot]
        text = "vrel {:.1f} m/s".format(value) if np.isfinite(value) else "vrel --"
        (text_width, text_height), _ = cv2.getTextSize(text, FONT, 0.45, 1)
        tx = max(2, min(width - text_width - 5, point[0] + 10))
        ty = max(text_height + 5, min(height - 5, point[1] + 23))
        cv2.rectangle(image, (tx - 3, ty - text_height - 4), (tx + text_width + 3, ty + 4),
                      (15, 15, 15), -1)
        cv2.putText(image, text, (tx, ty), FONT, 0.45, OWNER if slot == owner else OTHERS, 1,
                    cv2.LINE_AA)


def draw_owner(image, matrix, slots, owner):
    """Green ring and label on the owner (the ATD dot and number stay visible inside)."""
    if owner is None or not 0 <= owner < len(slots):
        return
    height, width = image.shape[:2]
    point = pixel(matrix, slots[owner], width, height)
    if point is None:
        return
    cv2.circle(image, point, 20, (0, 0, 0), 7, cv2.LINE_AA)
    cv2.circle(image, point, 20, OWNER, 4, cv2.LINE_AA)
    text = "OWNER #{}".format(owner + 1)
    tx = max(5, min(point[0] + 24, width - 170))
    ty = max(25, min(point[1] - 18, height - 12))
    cv2.putText(image, text, (tx, ty), FONT, 0.7, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(image, text, (tx, ty), FONT, 0.7, OWNER, 2, cv2.LINE_AA)


def draw_state(image, result):
    """Line under the team box: state, owner and team."""
    width = image.shape[1]
    text = "state: {}   owner: {}   team: {}".format(
        result.state, "--" if result.owner is None else "#{}".format(result.owner + 1),
        result.team or "--")
    cv2.rectangle(image, (15, 90), (min(width - 15, 700), 127), (0, 0, 0), -1)
    cv2.putText(image, text, (25, 116), FONT, 0.7, OWNER, 2, cv2.LINE_AA)


def timeline(rows, frame, start, stop, window, width):
    """Image with one coloured row per (name, {video_frame: label}) over the page of `window`
    frames holding `frame`, with a cursor at `frame`. Pages are consecutive blocks of `window`
    frames from `start`; each pixel column is one slice of time."""
    first = start + ((frame - start) // window) * window
    last = min(first + window, stop) - 1
    frames = np.minimum(np.linspace(first, first + window - 1, width).astype(int), last)
    image = np.zeros((TIMELINE_ROW * len(rows) + 22, width, 3), np.uint8)
    for row, (name, labels) in enumerate(rows):
        y0 = row * TIMELINE_ROW
        for column, f in enumerate(frames):
            image[y0 + 2:y0 + TIMELINE_ROW - 2, column] = LABEL_BGR.get(labels.get(f, "X"),
                                                                        (60, 60, 60))
        cv2.putText(image, name, (6, y0 + TIMELINE_ROW - 8), FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, name, (6, y0 + TIMELINE_ROW - 8), FONT, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
    cursor = int((frame - first) * (width - 1) / max(1, window - 1))
    cv2.line(image, (cursor, 0), (cursor, TIMELINE_ROW * len(rows)), (0, 255, 255), 2)
    text = "page {}/{}  |  video {:.1f}s .. {:.1f}s ({:.0f} s per page)".format(
        (frame - start) // window + 1, (stop - start + window - 1) // window, first / FPS,
        last / FPS, window / FPS)
    cv2.putText(image, text, (6, image.shape[0] - 6), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return image


def run(start, seconds=60, config=None, show_relative_speed=True, scale=0.75, show_gt=True,
        timeline_seconds=20, fix_keepers=True):
    """Play video frames [start, start + seconds) with the classifier's owner drawn on top and
    the possession timeline below.

    show_gt           add the public ground truth as a second timeline row
    timeline_seconds  length of one timeline page
    """
    config = config or PossessionConfig()
    video_path, meta_path, raw_path = require(["video", "atd_meta", "atd_raw"])
    stop = start + int(round(seconds * FPS))

    print("reading homography ...")
    matrices, _, valid = load_homography()
    print("reading ATD frames {}..{} ...".format(start, stop - 1))
    atd = read_atd(raw_path, start, stop)
    info = slot_info(meta_path)
    decided = decisions(raw_path, start, stop, info, config, fix_keepers, atd)

    # The CSV's labels: the team in possession, D before a half's first owner, X outside halves
    rows = [("classifier", {f: result.team or "D" for f, (result, _) in decided.items()})]
    if show_gt:
        rows.append(("GT", read_labels(labels_path("gt"), start, stop)))
    window = int(round(timeline_seconds * FPS))

    def compose(image, frame):
        """(window image, height of the video part) for one frame."""
        decision = decided.get(frame)
        if valid[frame] and frame in atd:
            slots, ball = atd[frame]
            draw_tracking(image, matrices[frame], slots, ball, info, frame, FPS)
            if decision is not None:
                result, relative = decision
                if show_relative_speed:
                    draw_relative_speeds(image, matrices[frame], slots, result.owner, relative)
                draw_owner(image, matrices[frame], slots, result.owner)
        draw_label_box(image, "team", (decision[0].team or "X") if decision else "X")
        if decision is not None:
            draw_state(image, decision[0])
        top = cv2.resize(image, None, fx=scale, fy=scale)
        return np.vstack([top, timeline(rows, frame, start, stop, window, top.shape[1])]), \
            top.shape[0]

    play(video_path, compose, start, stop, window, "possession owner")


def play(video_path, compose, start, stop, window, window_name):
    """Play video frames [start, stop) through compose(image, frame) -> (window image, height of
    its video part), with the timeline controls of the module doc. `window` is the timeline page
    in frames (a click on the timeline maps to that page)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("could not open {}".format(video_path))
    delay_ms = max(1, int(1000 / FPS))
    cv2.namedWindow(window_name)

    # `frame` is the frame to show next; a click on the timeline sets `seek`, one on the video
    # sets `toggle` (pause / play)
    state = {"frame": start, "seek": None, "toggle": False, "top_height": None, "width": None}

    def on_mouse(event, x, y, flags, _):
        dragging = event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON
        if state["top_height"] is None:
            return
        if y < state["top_height"]:
            if event == cv2.EVENT_LBUTTONDOWN:
                state["toggle"] = True
            return
        if event == cv2.EVENT_LBUTTONDOWN or dragging:
            first = start + ((state["frame"] - start) // window) * window
            target = first + int(round(x * (window - 1) / max(1, state["width"] - 1)))
            state["seek"] = min(stop - 1, max(start, target))

    cv2.setMouseCallback(window_name, on_mouse)

    playing = True
    shown_frame, next_in_video, shown = None, None, None
    while True:
        target = state["frame"] if state["seek"] is None else state["seek"]
        state["seek"] = None
        if target != shown_frame:
            if target != next_in_video:   # a jump: reposition the video (sequential reads are free)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, image = cap.read()
            if not ok:
                break
            next_in_video = target + 1
            shown, state["top_height"] = compose(image, target)
            state["width"] = shown.shape[1]
            shown_frame = target
        state["frame"] = shown_frame

        cv2.imshow(window_name, shown)
        key = cv2.waitKey(delay_ms if playing else 30) & 0xFF
        if key in (ord("q"), 27):
            break
        steps = {ord(","): -1, ord("."): 1, ord("a"): -FPS, ord("d"): FPS,
                 ord("["): -window, ord("]"): window}
        if key in (ord(" "), ord("p")) or state["toggle"]:
            state["toggle"] = False
            playing = not playing
        elif key in steps:
            state["frame"] = min(stop - 1, max(start, shown_frame + steps[key]))
        elif playing and state["seek"] is None:
            if shown_frame >= stop - 1:
                playing = False            # stay on the last frame instead of closing
            else:
                state["frame"] = shown_frame + 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    first_half_kickoff = half_video_blocks(GAME)[0][1]     # 11185, first graded frame

    # -- what to show ------------------------------------------------------------------------
    START = first_half_kickoff      # video frame to start from (minute * 1500 + second * 25)
    SECONDS = 600                    # how long to play
    SCALE = 0.75                    # window size relative to 1920x1080
    SHOW_RELATIVE_SPEED = True      # |v_player - v_ball| (m/s) under every tracked player
    SHOW_GT = True                  # public ground truth as a second timeline row
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

    FIX_GOALKEEPERS = True          # keepers back in their own team's slot (fix_goalkeepers.py)

    run(start=START, seconds=SECONDS, config=CONFIG, show_relative_speed=SHOW_RELATIVE_SPEED,
        scale=SCALE, show_gt=SHOW_GT, timeline_seconds=TIMELINE_SECONDS,
        fix_keepers=FIX_GOALKEEPERS)
