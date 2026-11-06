"""Draw the tactical ATD on top of the tactical video, frame by frame.

Each tracked slot is projected from the pitch (normalised 0-1, as the ATD stores it) to image
pixels with that frame's homography, and drawn as a dot with its slot number. The ball gets its own
marker. A strip along the bottom says, for the ball, whether x, y and speed are present (True) or
NaN (False) on that frame. A box in the top-left corner shows the label of a frame,label CSV --
the public ground truth or any submission in evaluation/results/ -- and False where it says X.

    python src/visualizing_tracking.py

What to show (start frame, duration, which CSV, ...) is set at the bottom of this file, in the
`if __name__ == '__main__':` block. Window keys: space pauses, q or Esc quits.

Three files are lined up by video frame and nothing else:
    video        frame v
    homography   row v            (frame column counts from 0)
    ATD .txt     line v           (frame_id = v + 1)
"""

import os
import sys
import xml.etree.ElementTree as ET
from itertools import islice

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks, require  # noqa: E402
from starter.load_homography import load as load_homography, project  # noqa: E402

# Params
FIELD_SEP, GROUP_SEP = ":", ";"     # "<frame_id>:<slot>;<slot>;...:<ball>", each "x,y,speed"
HOME, AWAY = "H", "A"

# BGR, because that is what OpenCV draws in
COLOURS = {
    HOME: (255, 120, 0),            # home outfield: blue
    AWAY: (40, 40, 230),            # away outfield: red
    (HOME, "gk"): (255, 255, 0),    # home goalkeeper slot: cyan
    (AWAY, "gk"): (255, 0, 255),    # away goalkeeper slot: magenta
    "ball": (0, 255, 255),          # yellow
    "D": (128, 128, 128),           # dead ball, in the ground-truth box
    "X": (60, 60, 60),              # no label here
}
# Public ground truth: real labels on the first 5 minutes of play only, X everywhere else.
GT_PATH = os.path.join(ROOT, "evaluation", "gt", "{}_possession_gt_public.csv".format(GAME))
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
DOT_RADIUS = 7
FONT = cv2.FONT_HERSHEY_SIMPLEX
STRIP_HEIGHT = 50                   # bottom strip with the ball's True/False flags


def labels_path(name):
    """Path of the CSV to show: the public ground truth for "gt", else results/<name>.csv."""
    if name == "gt":
        path = GT_PATH
    else:
        path = os.path.join(RESULTS_DIR, name if name.endswith(".csv") else name + ".csv")
    if not os.path.isfile(path):
        available = sorted(f[:-4] for f in os.listdir(RESULTS_DIR) if f.endswith(".csv"))
        raise IOError("no such labels file: {}\nchoose 'gt' or one of {}".format(path, available))
    return path


def slot_info(meta_path):
    """[(team, is_goalkeeper)] per slot, in file order -- which is the order of the .txt groups.

    The first <Team> in the metadata is the home side. A slot is flagged goalkeeper when its
    position_type says so; for BAR-ATH that is one slot per team.
    """
    root = ET.parse(meta_path).getroot()
    home_id = [team.get("id") for team in root.iter("Team")][0]
    info = []
    for player in root.iter("Player"):
        params = {p.findtext("Name"): p.findtext("Value") for p in player.iter("ProviderParameter")}
        info.append((HOME if player.get("teamId") == home_id else AWAY,
                     params.get("position_type") == "Goalkeeper"))
    return info


def parse_group(group):
    """'x,y,speed' -> three floats. 'NaN' parses to nan, so a missing object needs no special case."""
    x, y, speed = (float(value) for value in group.split(","))
    return x, y, speed


def read_atd(raw_path, start, stop):
    """{video_frame: (slots, ball)} for video frames [start, stop).

    slots is a list of (x, y, speed) per slot, ball is (x, y, speed). Line v of the file is video
    frame v, so islice can skip straight to it -- it still reads the earlier lines, just does not
    parse them.
    """
    out = {}
    with open(raw_path) as handle:
        for video_frame, line in enumerate(islice(handle, start, stop), start=start):
            frame_id, players, ball = line.strip().split(FIELD_SEP)
            if int(frame_id) - 1 != video_frame:
                raise ValueError("ATD line {} has frame_id {}, expected {}".format(
                    video_frame, frame_id, video_frame + 1))
            out[video_frame] = ([parse_group(g) for g in players.split(GROUP_SEP)],
                                parse_group(ball))
    return out


def read_labels(path, start, stop):
    """{video_frame: label} for video frames [start, stop) from a frame,label CSV.

    Row v of the CSV (after the header) is video frame v, the same index as everything else here.
    """
    out = {}
    with open(path) as handle:
        next(handle)  # header: frame,label
        for line in islice(handle, start, stop):
            frame, label = line.strip().split(",")
            out[int(frame)] = label.strip().upper()
    return out


def inside(u, v, width, height):
    return np.isfinite(u) and np.isfinite(v) and 0 <= u < width and 0 <= v < height


def draw_label_box(image, name, label):
    """Top-left box with the CSV's label for this frame, or False where it says X."""
    known = label in ("H", "A", "D")
    text = "{}: {}".format(name, label if known else False)
    colour = COLOURS[label] if known else COLOURS["X"]
    (text_width, _), _ = cv2.getTextSize(text, FONT, 1.1, 3)
    cv2.rectangle(image, (15, 15), (45 + text_width, 80), colour, -1)
    cv2.rectangle(image, (15, 15), (45 + text_width, 80), (255, 255, 255), 2)
    cv2.putText(image, text, (30, 60), FONT, 1.1, (255, 255, 255), 3, cv2.LINE_AA)


def draw_frame(image, matrix, slots, ball, info, video_frame, fps):
    """Draw slots, ball and the bottom strip onto `image` (full resolution, BGR) in place."""
    height, width = image.shape[:2]

    off_screen = 0
    for slot, ((x, y, _), (team, is_goalkeeper)) in enumerate(zip(slots, info), start=1):
        if np.isnan(x):
            continue  # nobody in this slot on this frame
        u, v = project(matrix, x, y)
        if not inside(u, v, width, height):
            off_screen += 1  # tracked, but the camera is not looking there
            continue
        colour = COLOURS[(team, "gk")] if is_goalkeeper else COLOURS[team]
        centre = (int(round(u)), int(round(v)))
        cv2.circle(image, centre, DOT_RADIUS, colour, -1)
        cv2.circle(image, centre, DOT_RADIUS, (0, 0, 0), 1)
        cv2.putText(image, str(slot), (centre[0] - 8, centre[1] - 12), FONT, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, str(slot), (centre[0] - 8, centre[1] - 12), FONT, 0.55,
                    colour, 1, cv2.LINE_AA)

    ball_x, ball_y, ball_speed = ball
    if not np.isnan(ball_x) and not np.isnan(ball_y):
        u, v = project(matrix, ball_x, ball_y)
        if inside(u, v, width, height):
            centre = (int(round(u)), int(round(v)))
            cv2.circle(image, centre, DOT_RADIUS + 5, COLOURS["ball"], 2)
            cv2.drawMarker(image, centre, COLOURS["ball"], cv2.MARKER_CROSS, 10, 2)

    # Bottom strip: frame, video time, and whether each ball field is present
    cv2.rectangle(image, (0, height - STRIP_HEIGHT), (width, height), (0, 0, 0), -1)
    seconds = video_frame / fps
    text = ("frame {}  {:02d}:{:05.2f}   |   ball  x: {}   y: {}   speed: {}   |   "
            "slots off-screen: {}").format(
        video_frame, int(seconds // 60), seconds % 60,
        not np.isnan(ball_x), not np.isnan(ball_y), not np.isnan(ball_speed), off_screen)
    cv2.putText(image, text, (15, height - 17), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def run(start, seconds=30, labels="gt", scale=0.75, out=None):
    """Play (or write) the video from frame `start` for `seconds`, with the ATD drawn on top.

    labels  "gt" for the public ground truth, or a file name in evaluation/results/ (no .csv)
    scale   window/output size relative to 1920x1080
    out     an .mp4 path to write instead of opening a window, or None
    """
    video_path, meta_path, raw_path = require(["video", "atd_meta", "atd_raw"])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("could not open {}".format(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    stop = start + int(round(seconds * fps))

    print("reading homography ...")
    matrices, _, valid = load_homography()
    print("reading ATD frames {}..{} ...".format(start, stop - 1))
    atd = read_atd(raw_path, start, stop)
    path = labels_path(labels)
    print("labels from {} ...".format(os.path.relpath(path, ROOT)))
    frame_labels = read_labels(path, start, stop)
    box_name = "GT" if labels == "gt" else labels.replace(".csv", "")
    info = slot_info(meta_path)

    writer = None
    if out:
        size = (int(1920 * scale), int(1080 * scale))
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    delay_ms = max(1, int(1000 / fps))
    paused = False
    video_frame = start
    while video_frame < stop:
        if not paused:
            ok, image = cap.read()
            if not ok:
                break
            if valid[video_frame]:
                slots, ball = atd[video_frame]
                draw_frame(image, matrices[video_frame], slots, ball, info, video_frame, fps)
            draw_label_box(image, box_name, frame_labels.get(video_frame, "X"))
            shown = cv2.resize(image, None, fx=scale, fy=scale)
            video_frame += 1

        if writer is not None:
            writer.write(shown)
            continue

        cv2.imshow("tactical + ATD", shown)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused

    cap.release()
    if writer is not None:
        writer.release()
        print("wrote {}".format(out))
    cv2.destroyAllWindows()


if __name__ == '__main__':
    first_half_kickoff = half_video_blocks(GAME)[0][1]     # 11185, first graded frame

    # -- what to show ------------------------------------------------------------------------
    START = first_half_kickoff     # video frame to start from (minute * 1500 + second * 25)
    SECONDS = 420                    # how long to play
    LABELS = "baseline_raw"                   #"gt" or a file in evaluation/results/, e.g. "baseline_smooth50"
    SCALE = 0.75                    # window size relative to 1920x1080
    OUT = None                      # e.g. "clip.mp4" to write a file instead of a window

    run(start=START, seconds=SECONDS, labels=LABELS, scale=SCALE, out=OUT)
