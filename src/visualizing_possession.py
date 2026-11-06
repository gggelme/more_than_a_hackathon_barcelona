"""Watch the possession classifier (src/tracking_possession/possession.py) on the tactical video.

Display only: nothing is saved. Per frame:
  - every tracked slot as a dot in its team colour with its velocity arrow, and the ball with a much
    longer arrow (scaled only to be visible -- the classifier uses the real velocities);
  - a thin circle of radius D_MAX around the ball: only the players inside it are candidates, each
    labelled with its score;
  - the OWNER chosen by the classifier with a thick green ring, and the NEAREST player (the old
    method) with a white square. When they are different players, both are labelled, so you see
    where the new criterion disagrees with the old one;
  - top-left, the team in possession by the rules (the last team that had an owner) in a coloured
    box as in test/visualizing_compare.py -- blue for H, red for A -- and under it a panel with the
    state (possession / free / flight / no_ball), the owner, the nearest and whether they agree.

    python src/visualizing_possession.py

What to show and every weight of the classifier are set at the bottom of this file, in the
`if __name__ == '__main__':` block. Window keys: space pauses, q or Esc quits.
"""

import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, half_video_blocks, require  # noqa: E402
from starter.load_homography import load as load_homography, project  # noqa: E402
from visualizing_tracking import (COLOURS, FONT, draw_label_box, inside, read_atd,  # noqa: E402
                                  slot_info)
from visualizing_vectors import draw_arrow  # noqa: E402
from tracking_possession.possession import PossessionConfig, classify  # noqa: E402

# Params
PITCH = np.array(GAMES[GAME]["pitch"])      # metres
DOT_RADIUS = 5
STRIP_HEIGHT = 50
OWNER = (0, 255, 0)                         # BGR: the classifier's owner, green
NEAREST = (255, 255, 255)                   # the nearest player, white
STATE_COLOURS = {"possession": (0, 150, 0), "free": (0, 140, 200), "flight": (200, 80, 0),
                 "no_ball": (90, 90, 90)}


def to_metres(atd, first, stop):
    """(n, slots, 2) players and (n, 2) ball in metres from the ATD, NaN where missing."""
    frames = range(first, stop)
    players = np.array([[s[:2] for s in atd[f][0]] for f in frames], dtype=np.float64) * PITCH
    ball = np.array([atd[f][1][:2] for f in frames], dtype=np.float64) * PITCH
    return players, ball


def pixel(matrix, xy_m):
    """Pixel (int u, int v) of a point in metres, or None if not finite."""
    u, v = project(matrix, xy_m[0] / PITCH[0], xy_m[1] / PITCH[1])
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    return int(round(float(u))), int(round(float(v)))


def draw_candidate_circle(image, matrix, ball_m, radius):
    """The D_MAX circle around the ball, drawn on the grass."""
    angles = np.linspace(0, 2 * np.pi, 48)
    points = [pixel(matrix, ball_m + radius * np.array([np.cos(a), np.sin(a)])) for a in angles]
    if all(p is not None for p in points):
        cv2.polylines(image, [np.array(points, np.int32)], True, (200, 200, 200), 1, cv2.LINE_AA)


def draw_panel(image, result, teams, top=95):
    """Panel under the team box: state, owner, nearest and whether they agree."""
    def who(slot):
        return "-" if slot is None else "{} slot {}".format(teams[slot], slot + 1)

    owner_score = result.scores.get(result.owner)
    lines = [
        ("STATE: {}".format(result.state.upper()), (255, 255, 255)),
        ("owner (new): {}{}".format(who(result.owner), "" if owner_score is None
                                    else "  score {:.2f}".format(owner_score)), OWNER),
        ("nearest (old): {}{}".format(who(result.nearest), "" if result.nearest_distance is None
                                      else "  {:.1f} m".format(result.nearest_distance)), NEAREST),
    ]
    if result.owner is not None and result.nearest is not None:
        agree = result.owner == result.nearest
        lines.append(("AGREE" if agree else "DIFFER", (0, 255, 0) if agree else (0, 0, 255)))

    colour = STATE_COLOURS.get(result.state, (60, 60, 60))
    height = 20 + 32 * len(lines)
    cv2.rectangle(image, (15, top), (600, top + height), (30, 30, 30), -1)
    cv2.rectangle(image, (15, top), (600, top + height), colour, 3)
    for i, (text, text_colour) in enumerate(lines):
        cv2.putText(image, text, (30, top + 32 + 32 * i), FONT, 0.75, text_colour, 2,
                    cv2.LINE_AA)


def draw_frame(image, matrix, i, result, players, player_v, ball, ball_v, info, teams, config,
               player_seconds, ball_seconds, frame, fps):
    """Everything for one frame, onto the full-resolution BGR image in place."""
    height, width = image.shape[:2]

    if np.all(np.isfinite(ball[i])):
        draw_candidate_circle(image, matrix, ball[i], config.d_max)

    for slot, (team, is_goalkeeper) in enumerate(info):
        position = players[i, slot]
        if not np.all(np.isfinite(position)):
            continue
        point = pixel(matrix, position)
        if point is None or not inside(point[0], point[1], width, height):
            continue
        colour = COLOURS[(team, "gk")] if is_goalkeeper else COLOURS[team]
        velocity = player_v[i, slot]
        if np.all(np.isfinite(velocity)) and np.linalg.norm(velocity) >= config.moving:
            draw_arrow(image, matrix, position[0] / PITCH[0], position[1] / PITCH[1],
                       velocity / PITCH, player_seconds, colour, 2)
        cv2.circle(image, point, DOT_RADIUS, colour, -1)
        cv2.circle(image, point, DOT_RADIUS, (0, 0, 0), 1)
        if slot in result.scores:
            cv2.putText(image, "{:.1f}".format(result.scores[slot]), (point[0] + 8, point[1] + 20),
                        FONT, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        differ = result.owner is not None and result.nearest is not None \
            and result.owner != result.nearest
        if slot == result.nearest:
            cv2.drawMarker(image, point, NEAREST, cv2.MARKER_SQUARE, 26, 2)
            if differ:
                cv2.putText(image, "NEAREST", (point[0] - 35, point[1] - 22), FONT, 0.55, NEAREST,
                            2, cv2.LINE_AA)
        if slot == result.owner:
            cv2.circle(image, point, 16, OWNER, 3, cv2.LINE_AA)
            if differ:
                cv2.putText(image, "OWNER", (point[0] - 28, point[1] - 22), FONT, 0.55, OWNER, 2,
                            cv2.LINE_AA)

    if np.all(np.isfinite(ball[i])):
        point = pixel(matrix, ball[i])
        if point is not None and inside(point[0], point[1], width, height):
            if np.all(np.isfinite(ball_v[i])) and np.linalg.norm(ball_v[i]) >= config.moving:
                draw_arrow(image, matrix, ball[i][0] / PITCH[0], ball[i][1] / PITCH[1],
                           ball_v[i] / PITCH, ball_seconds, COLOURS["ball"], 4)
            cv2.circle(image, point, DOT_RADIUS + 4, COLOURS["ball"], 2)

    # Team in possession, coloured like the label boxes of visualizing_compare (H blue, A red)
    draw_label_box(image, "team", result.team or "X")
    draw_panel(image, result, teams)
    cv2.rectangle(image, (0, height - STRIP_HEIGHT), (width, height), (0, 0, 0), -1)
    seconds = frame / fps
    text = "frame {}  {:02d}:{:05.2f}   |   ball speed: {}   |   candidates: {}".format(
        frame, int(seconds // 60), seconds % 60,
        "-" if result.ball_speed is None else "{:.1f} m/s".format(result.ball_speed),
        len(result.scores))
    cv2.putText(image, text, (15, height - 17), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def run(start, seconds=30, config=None, player_arrow_seconds=1.0, ball_arrow_seconds=4.0,
        scale=0.75):
    """Play video frames [start, start + seconds) with the classifier's decisions drawn on top."""
    config = config or PossessionConfig()
    video_path, meta_path, raw_path = require(["video", "atd_meta", "atd_raw"])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("could not open {}".format(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    stop = start + int(round(seconds * fps))

    # A margin on both sides, so smoothing and centred velocities exist at the range's edges
    margin = config.smooth_frames + config.velocity_frames
    first, last = max(0, start - margin), stop + margin
    print("reading homography ...")
    matrices, _, valid = load_homography()
    print("reading ATD frames {}..{} ...".format(first, last - 1))
    atd = read_atd(raw_path, first, last)
    last = first + len(atd)                       # the file may end before the margin does
    info = slot_info(meta_path)
    teams = [team for team, _ in info]

    print("classifying ...")
    players, ball = to_metres(atd, first, last)
    results, players_s, player_v, ball_s, ball_v = classify(players, ball, teams, config, fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    delay_ms = max(1, int(1000 / fps))
    paused = False
    frame = start
    while frame < min(stop, last):
        if not paused:
            ok, image = cap.read()
            if not ok:
                break
            i = frame - first
            if valid[frame]:
                draw_frame(image, matrices[frame], i, results[i], players_s, player_v, ball_s,
                           ball_v, info, teams, config, player_arrow_seconds, ball_arrow_seconds,
                           frame, fps)
            shown = cv2.resize(image, None, fx=scale, fy=scale)
            frame += 1

        cv2.imshow("possession classifier", shown)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    first_half_kickoff = half_video_blocks(GAME)[0][1]     # 11185, first graded frame

    # -- what to show ------------------------------------------------------------------------
    START = first_half_kickoff      # video frame to start from (minute * 1500 + second * 25)
    SECONDS = 60                    # how long to play
    SCALE = 0.75                    # window size relative to 1920x1080
    PLAYER_ARROW_SECONDS = 1.0      # drawing only: player arrow = where it would be 1 s ahead
    BALL_ARROW_SECONDS = 4.0        # drawing only: ball arrow, much longer to be visible

    # -- the classifier (metres, m/s) --------------------------------------------------------
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

    run(start=START, seconds=SECONDS, config=CONFIG, player_arrow_seconds=PLAYER_ARROW_SECONDS,
        ball_arrow_seconds=BALL_ARROW_SECONDS, scale=SCALE)
