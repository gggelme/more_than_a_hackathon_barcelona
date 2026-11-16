"""Watch the Smart Tagging on the tactical video, drawn as in src/visualizing_owner.py.

The Smart Tagging is Metrica's AUTOMATIC annotation of the tactical video (info_hackathon/DATA.md):
intervals with a code (SET PIECES, GOALS, BUILD UP, ...) and tags (Team, Type, Side, ...).
starter/load_smart_tagging.py reads it and turns its seconds into video frames. It is allowed and
meant to be used (info_hackathon/FAQ.md), but it is not ground truth: false positives and false
negatives, approximate windows (set pieces ~35 s, kick-offs ~5 s), and every throw-in has Team N/A.

Why look at it: every set piece is a RESTART, the end of a dead-ball stretch, and GOALS / "Kick Off
(After Goal)" mark the goals -- the raw material to label D.

Display only: nothing is saved. Per frame:
  - with SHOW_TRACKING, the ATD as in src/visualizing_tracking.py (keepers put right with
    FIX_GOALKEEPERS);
  - top-left, a compact see-through panel with every tag ACTIVE on the frame: code, team (blue
    Barcelona, red Athletic, grey N/A), Type / Side and the seconds it has left;
  - below the video, a paged timeline with every tag as a bar in its own lane (overlapping tags are
    stacked), coloured by team, labelled with its code; SET PIECES and GOALS outlined in yellow so
    the restarts stand out; with SHOW_GT the public ground truth under it.
CODES picks which codes to show (None = all of them).

    python src/visualizing_smart_tagging.py

Moving through the video, as in src/visualizing_owner.py:
  - click or drag on the timeline to jump to that moment of the page;
  - , and .  one frame back / forward;   a and d  one second;   [ and ]  one page;
  - space or p (or a click on the video) pauses / plays, q or Esc quits.

What to show is set at the bottom of this file, in the `if __name__ == '__main__':` block. Keep
SECONDS moderate with SHOW_TRACKING: the ATD of the range is read into memory.
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
from starter.load_homography import load as load_homography  # noqa: E402
from starter.load_smart_tagging import load as load_smart_tagging  # noqa: E402
from visualizing_tracking import (COLOURS, FONT, draw_frame as draw_tracking,  # noqa: E402
                                  fix_goalkeepers_in_atd, labels_path, print_goalkeepers,
                                  read_atd, read_labels, slot_info)
from visualizing_owner import TIMELINE_ROW, play, timeline  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]
LANE_HEIGHT = 20                            # pixel height of one tag lane
TEAM_BGR = {"Barcelona": COLOURS["H"], "Atletic Club": COLOURS["A"]}
NO_TEAM = (120, 120, 120)                   # N/A or no Team tag
RESTART_CODES = ("set pieces", "goals")     # outlined: restarts and goals, what D is built from
RESTART = (0, 255, 255)                     # yellow outline
PANEL_ALPHA = 0.5                           # how much of the video shows through the panel
SHOWN_TAGS = ("Type", "Side")               # tags written next to the code in the panel


def team_colour(event):
    return TEAM_BGR.get(event["tags"].get("Team"), NO_TEAM)


def is_restart(event):
    return (event["code"] or "").strip().lower() in RESTART_CODES


def lanes(events):
    """Lane index per event, greedy: the first lane whose last bar ended before it starts."""
    ends, out = [], []
    for event in events:
        for lane, end in enumerate(ends):
            if event["start_frame"] >= end:
                ends[lane] = event["end_frame"]
                out.append(lane)
                break
        else:
            ends.append(event["end_frame"])
            out.append(len(ends) - 1)
    return out


def draw_panel(image, events, frame):
    """Compact see-through panel with the tags active on `frame`."""
    now = [e for e in events if e["start_frame"] <= frame < e["end_frame"]]
    lines = [("smart tagging: {} active".format(len(now)), (235, 235, 235))]
    for e in now:
        tags = "  ".join(e["tags"][t] for t in SHOWN_TAGS if e["tags"].get(t))
        lines.append(("{}{}  {}  {}  ({:.1f} s left)".format(
            "> " if is_restart(e) else "", e["code"], e["tags"].get("Team", "N/A"), tags,
            (e["end_frame"] - frame) / FPS), RESTART if is_restart(e) else team_colour(e)))
    scale, step, x0, y0 = 0.55, 24, 12, 12
    text_width = max(cv2.getTextSize(text, FONT, scale, 1)[0][0] for text, _ in lines)
    x1 = min(image.shape[1] - 1, x0 + text_width + 16)
    y1 = y0 + step * len(lines) + 8
    image[y0:y1, x0:x1] = (image[y0:y1, x0:x1] * (1 - PANEL_ALPHA)).astype(np.uint8)
    for k, (text, colour) in enumerate(lines):
        y = y0 + 20 + step * k
        cv2.putText(image, text, (x0 + 8, y), FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, text, (x0 + 8, y), FONT, scale, colour, 1, cv2.LINE_AA)


def tag_timeline(events, lane_of, n_lanes, frame, start, window, width):
    """One lane per row of overlapping tags over the page of `window` frames holding `frame`,
    paged exactly like visualizing_owner.timeline (so a click maps to the same frame)."""
    first = start + ((frame - start) // window) * window
    image = np.zeros((LANE_HEIGHT * max(1, n_lanes) + 4, width, 3), np.uint8)

    def column(f):
        return int((f - first) * (width - 1) / max(1, window - 1))

    for event, lane in zip(events, lane_of):
        if event["end_frame"] <= first or event["start_frame"] >= first + window:
            continue
        x0 = max(0, column(event["start_frame"]))
        x1 = min(width - 1, column(event["end_frame"] - 1))
        y0, y1 = 2 + lane * LANE_HEIGHT, 2 + (lane + 1) * LANE_HEIGHT - 3
        cv2.rectangle(image, (x0, y0), (x1, y1), team_colour(event), -1)
        if is_restart(event):
            cv2.rectangle(image, (x0, y0), (x1, y1), RESTART, 2)
        label = event["code"].title() if not event["tags"].get("Type") else event["tags"]["Type"]
        if x1 - x0 > 40:
            cv2.putText(image, label, (x0 + 4, y1 - 5), FONT, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, label, (x0 + 4, y1 - 5), FONT, 0.42, (255, 255, 255), 1,
                        cv2.LINE_AA)
    x = column(frame)
    cv2.line(image, (x, 0), (x, image.shape[0]), (0, 255, 255), 2)
    return image


def run(start, seconds=120, codes=None, show_tracking=True, fix_keepers=True, show_gt=False,
        scale=0.75, timeline_seconds=60):
    """Play video frames [start, start + seconds) with the Smart Tagging drawn on top and below.

    codes          codes to show (case-insensitive), None = all
    show_tracking  draw the ATD players and ball
    fix_keepers    with show_tracking, put each keeper back in his own team's slot
    show_gt        add the public ground truth as a timeline row
    """
    video_path, meta_path, raw_path = require(["video", "atd_meta", "atd_raw"])
    stop = start + int(round(seconds * FPS))

    wanted = None if codes is None else {c.strip().lower() for c in codes}
    events = [e for e in load_smart_tagging()
              if (wanted is None or (e["code"] or "").strip().lower() in wanted)
              and e["end_frame"] > start and e["start_frame"] < stop]
    events.sort(key=lambda e: e["start_frame"])
    lane_of = lanes(events)
    n_lanes = max(lane_of) + 1 if lane_of else 1
    print("smart tagging in the range: {} tags".format(len(events)))
    for e in events:
        seconds_e = e["start_frame"] / FPS
        print("  {:02d}:{:05.2f}  {:>6} .. {:>6}  {:<30} {:<13} {}".format(
            int(seconds_e // 60), seconds_e % 60, e["start_frame"], e["end_frame"], e["code"],
            e["tags"].get("Team", "N/A"), e["tags"].get("Type", "")))

    tracking = None
    if show_tracking:
        print("reading homography ...")
        matrices, _, valid = load_homography()
        print("reading ATD frames {}..{} ...".format(start, stop - 1))
        atd = read_atd(raw_path, start, stop)
        info = slot_info(meta_path)
        if fix_keepers:
            print_goalkeepers(fix_goalkeepers_in_atd(atd, info, raw_path))
        tracking = (matrices, valid, atd, info)

    rows = [("GT", read_labels(labels_path("gt"), start, stop))] if show_gt else []
    window = int(round(timeline_seconds * FPS))

    def compose(image, frame):
        """(window image, height of the video part) for one frame."""
        if tracking is not None:
            matrices, valid, atd, info = tracking
            if valid[frame] and frame in atd:
                slots, ball = atd[frame]
                draw_tracking(image, matrices[frame], slots, ball, info, frame, FPS)
        draw_panel(image, events, frame)
        top = cv2.resize(image, None, fx=scale, fy=scale)
        width = top.shape[1]
        parts = [top, tag_timeline(events, lane_of, n_lanes, frame, start, window, width)]
        if rows:
            parts.append(timeline(rows, frame, start, stop, window, width))
        return np.vstack(parts), top.shape[0]

    play(video_path, compose, start, stop, window, "smart tagging")


if __name__ == '__main__':
    first_half_kickoff = half_video_blocks(GAME)[0][1]     # 11185, first graded frame

    # -- what to show ------------------------------------------------------------------------
    second_start, second_end = half_video_blocks(GAME)[1][1:]
    START = 23700      # video frame to start from (minute * 1500 + second * 25)
    SECONDS = second_end + 300                  # how long; keep it moderate with SHOW_TRACKING
    CODES = None                    # None = all; e.g. ["SET PIECES", "GOALS"] for the restarts
    SHOW_TRACKING = True            # draw the ATD players and ball
    FIX_GOALKEEPERS = True          # keepers back in their own team's slot (fix_goalkeepers.py)
    SHOW_GT = True                  # public ground truth as a timeline row (first 5 min only)
    SCALE = 0.75                    # window size relative to 1920x1080
    TIMELINE_SECONDS = 60           # length of one timeline page

    run(start=START, seconds=SECONDS, codes=CODES, show_tracking=SHOW_TRACKING,
        fix_keepers=FIX_GOALKEEPERS, show_gt=SHOW_GT, scale=SCALE,
        timeline_seconds=TIMELINE_SECONDS)
