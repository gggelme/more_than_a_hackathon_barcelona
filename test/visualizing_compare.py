"""Compare two frame,label CSVs on the tactical video, side by side.

Top: the same video frame twice, each with the label one CSV gives it (left = LEFT, right = RIGHT).
A red frame around both panels marks the frames where the two CSVs disagree.

Bottom: a timeline for each CSV (and the public ground truth, where it has labels) -- H blue, A red,
D grey, X white -- so you see where one fragments a possession, catches a dead ball, or changes team
earlier than the other. It is paged: it shows a fixed block of TIMELINE_SECONDS, the cursor walks
across it from left to right, and when it reaches the end the next block is shown. The labels to
the right of the cursor are what is coming next.

Either side can be "gt" (the public ground truth) or the name of a file in evaluation/results/
without .csv, e.g. "baseline_smooth50" or "baseline_smooth50_ball_height".

    python test/visualizing_compare.py

With SHOW_TRACKING the ATD is drawn on the video as in src/visualizing_tracking.py: every tracked
slot as a dot with its number (projected with the homography), the ball, and a strip saying whether
the ball has x, y and speed -- to see which players are tracked and which are not.

Moving through the video:
  - click or drag on the timeline to jump to that moment of the page;
  - , and .  one frame back / forward;   a and d  one second;   [ and ]  one page;
  - space or p (or a click on the video) pauses / plays, q or Esc quits. At the end of the range
    it pauses instead of closing. Keys only reach the window while it has the focus.

What to show is set at the bottom of this file, in the `if __name__ == '__main__':` block.

The public ground truth is for looking and checking the pipeline, not for choosing between the two
CSVs -- see info_hackathon/RULES.md.
"""

import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))   # test
ROOT = os.path.dirname(HERE)                         # repository root
sys.path.insert(0, ROOT)                             # games
sys.path.insert(0, os.path.join(ROOT, "src"))        # visualizing_tracking
from games import GAME, GAMES, half_video_blocks, require  # noqa: E402
from starter.load_homography import load as load_homography  # noqa: E402
from visualizing_tracking import (COLOURS, FONT, draw_frame as draw_tracking,  # noqa: E402
                                  draw_label_box, labels_path, read_atd, read_labels, slot_info)

# Params
FPS = GAMES[GAME]["fps"]
TIMELINE_ROW = 26                   # pixel height of one timeline row
DISAGREE = (0, 0, 255)              # BGR: red border when the two CSVs differ
LABEL_BGR = {"H": COLOURS["H"], "A": COLOURS["A"], "D": COLOURS["D"], "X": (255, 255, 255)}


def name_of(labels):
    """Short name shown on screen for a labels choice."""
    return "GT" if labels == "gt" else labels.replace(".csv", "")


def timeline(rows, frame, start, stop, window, width):
    """Image with one coloured row per (name, labels) over the page of `window` frames holding
    `frame`, with a cursor at `frame`.

    rows is [(name, {video_frame: label})]. Pages are consecutive blocks of `window` frames from
    `start`, so the cursor moves from left to right and jumps back to the left on a new page. Each
    pixel column is one slice of time.
    """
    first = start + ((frame - start) // window) * window
    last = min(first + window, stop) - 1
    frames = np.linspace(first, first + window - 1, width).astype(int)   # a short last page is
    frames = np.minimum(frames, last)                                    # padded with its end
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
    page = (frame - start) // window + 1
    pages = (stop - start + window - 1) // window
    text = "page {}/{}  |  video {:.1f}s .. {:.1f}s ({:.0f} s per page)".format(
        page, pages, first / FPS, last / FPS, window / FPS)
    cv2.putText(image, text, (6, image.shape[0] - 6), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return image


def compose(image, frame, rows, left_labels, right_labels, start, stop, window, panel_scale,
            tracking):
    """The whole window for one frame: two labelled panels on top, the timeline below.

    tracking is None or (matrices, valid, atd, info): when given, the ATD is drawn on the
    full-resolution frame first, as in visualizing_tracking.py.
    """
    if tracking is not None:
        matrices, valid, atd, info = tracking
        if valid[frame]:
            slots, ball = atd[frame]
            draw_tracking(image, matrices[frame], slots, ball, info, frame, FPS)
    image = cv2.resize(image, None, fx=panel_scale, fy=panel_scale)

    panels = []
    for name, labels in rows[:2]:
        panel = image.copy()
        draw_label_box(panel, name, labels.get(frame, "X"))
        panels.append(panel)
    top = np.hstack(panels)
    if left_labels.get(frame) != right_labels.get(frame):
        cv2.rectangle(top, (0, 0), (top.shape[1] - 1, top.shape[0] - 1), DISAGREE, 6)
    if tracking is None:   # the tracking strip already shows frame and time
        seconds_now = frame / FPS
        cv2.putText(top, "frame {}  {:02d}:{:05.2f}".format(
            frame, int(seconds_now // 60), seconds_now % 60),
            (top.shape[1] // 2 - 110, top.shape[0] - 15), FONT, 0.7, (255, 255, 255), 2,
            cv2.LINE_AA)
    return np.vstack([top, timeline(rows, frame, start, stop, window, top.shape[1])]), top.shape[0]


def run(start, seconds=60, left="baseline_smooth50", right="baseline_smooth50_ball_height",
        show_gt=True, show_tracking=True, panel_scale=0.5, timeline_seconds=20, speed=1.0):
    """Play video frames [start, start + seconds) with LEFT's and RIGHT's labels side by side.

    show_tracking     draw the ATD players and ball on the video
    panel_scale       size of each video panel relative to 1920x1080 (0.5 -> two 960x540 panels)
    timeline_seconds  length of one timeline page
    speed             1.0 real time, 0.5 half speed, 2.0 twice as fast
    """
    video_path = require(["video"])[0]
    stop = start + int(round(seconds * FPS))

    left_labels = read_labels(labels_path(left), start, stop)
    right_labels = read_labels(labels_path(right), start, stop)
    rows = [(name_of(left), left_labels), (name_of(right), right_labels)]
    if show_gt and "gt" not in (left, right):
        rows.append(("GT", read_labels(labels_path("gt"), start, stop)))

    differ = sum(left_labels.get(f) != right_labels.get(f) for f in range(start, stop))
    print("{} vs {}: {} of {} frames differ ({:.1f}%)".format(
        name_of(left), name_of(right), differ, stop - start, 100.0 * differ / (stop - start)))

    tracking = None
    if show_tracking:
        meta_path, raw_path = require(["atd_meta", "atd_raw"])
        print("reading homography ...")
        matrices, _, valid = load_homography()
        print("reading ATD frames {}..{} ...".format(start, stop - 1))
        tracking = (matrices, valid, read_atd(raw_path, start, stop), slot_info(meta_path))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("could not open {}".format(video_path))

    window = int(round(timeline_seconds * FPS))
    delay_ms = max(1, int(1000 / (FPS * speed)))
    window_name = "compare results"
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
    shown_frame, next_in_video = None, None
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
            shown, top_height = compose(image, target, rows, left_labels, right_labels, start, stop,
                                        window, panel_scale, tracking)
            state["top_height"], state["width"] = top_height, shown.shape[1]
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
    SECONDS = 500                    # how long
    LEFT = "vector_tracking_clasification"      # "gt" or a file in evaluation/results/ (without .csv)
    RIGHT = "vector_tracking_deadball"
    SHOW_GT = True                  # add the public ground truth as a third timeline row
    SHOW_TRACKING = True            # draw the ATD players and ball on the video
    PANEL_SCALE = 0.5               # each panel relative to 1920x1080 (0.5 -> 960x540)
    TIMELINE_SECONDS = 20           # length of one timeline page
    SPEED = 1.0                     # 1.0 = real time, 0.5 = half speed

    run(start=START, seconds=SECONDS, left=LEFT, right=RIGHT, show_gt=SHOW_GT,
        show_tracking=SHOW_TRACKING, panel_scale=PANEL_SCALE, timeline_seconds=TIMELINE_SECONDS,
        speed=SPEED)
