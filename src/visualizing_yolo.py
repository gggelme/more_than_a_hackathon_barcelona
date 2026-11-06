"""Possession with the ball from the ATD AND YOLO together, on the tactical video.

Two passes over the chosen range:

  1. ANALYSIS. For every frame, YOLO (COCO, class 32, sports ball) looks for the ball in a CROP x CROP
     square at full resolution, centred on the ATD ball projected into the image -- or, when the ATD
     has no ball, on where the ball should be by now (last known ball + its velocity). With nothing
     to centre on, it searches the whole frame at IMGSZ (FALLBACK_FULL) or skips the frame. The box
     centres are taken to the pitch with the inverse homography, and tracking_possession/
     ball_fusion.py turns ATD + YOLO into one ball per frame:
        both      ATD and a YOLO box agree (within AGREE_M)       -> the ATD
        conflict  YOLO only sees things far from the ATD          -> the ATD, YOLO ignored
        atd       only the ATD                                     -> the ATD
        yolo      no ATD, a credible YOLO box near the prediction  -> that YOLO box
        none      nothing credible                                 -> no ball
  2. PLAYBACK. The possession classifier (tracking_possession/possession.py) runs on the fused ball
     -- and, for comparison, on the ATD ball alone -- and the video is played with everything drawn.

Display only: nothing is saved (unless OUT is set, which writes the video). On screen:
  - players as dots in team colours, the OWNER with a green ring, the NEAREST with a white square;
  - the ATD ball (yellow circle), every YOLO box (thin green; the one used, thick blue), the fused
    ball (magenta dot), the predicted ball when the ATD is missing (magenta cross), and where YOLO
    looked (white square);
  - a panel with the ball source, the classifier state, the owner, and the team in possession with
    the fused ball next to the team with the ATD ball alone -- DIFFER when YOLO changed the call.

    python src/visualizing_yolo.py

What to run is set at the bottom of this file, in the `if __name__ == '__main__':` block. The
analysis pass runs YOLO on every frame before anything is shown: keep SECONDS moderate.
Window keys: space pauses, q or Esc quits.

Runs locally only. The match video must not be sent to any hosted service (see RULES.md).
"""

import os
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, half_video_blocks, require  # noqa: E402
from starter.load_homography import load as load_homography, project, unproject  # noqa: E402
from visualizing_tracking import COLOURS, FONT, inside, read_atd, slot_info  # noqa: E402
from tracking_possession.possession import PossessionConfig, classify  # noqa: E402
from tracking_possession.ball_fusion import BallFusion, FusionConfig  # noqa: E402

# Params
SPORTS_BALL = 32                    # COCO class id
MODELS_DIR = os.path.join(ROOT, "models")
PITCH = np.array(GAMES[GAME]["pitch"])      # metres
STRIP_HEIGHT = 50
DOT_RADIUS = 5

# BGR
BOX = (0, 255, 0)                   # YOLO detections: green
USED = (255, 120, 0)                # the YOLO box the fusion used: blue
ATD_BALL = (0, 255, 255)            # ATD ball: yellow
FUSED = (255, 0, 255)               # fused ball / prediction: magenta
CROP_COLOUR = (255, 255, 255)       # where YOLO looked: white
OWNER = (0, 255, 0)                 # the classifier's owner: green ring
NEAREST = (255, 255, 255)           # the nearest player: white square
SOURCE_COLOURS = {"both": (0, 160, 0), "atd": (0, 160, 160), "conflict": (0, 100, 230),
                  "yolo": (230, 80, 0), "none": (90, 90, 90)}


# -- detection ---------------------------------------------------------------------------------

def detect(model, image, imgsz, conf, device):
    """[(x1, y1, x2, y2, confidence)] for every sports-ball box YOLO finds in one BGR image."""
    result = model.predict(image, imgsz=imgsz, conf=conf, classes=[SPORTS_BALL], device=device,
                           verbose=False)[0]
    boxes = result.boxes
    return [(*box, score) for box, score in zip(boxes.xyxy.cpu().numpy().tolist(),
                                               boxes.conf.cpu().numpy().tolist())]


def crop_around(cx, cy, size, width, height):
    """(x0, y0, w, h) of a size x size square centred on (cx, cy), kept inside the frame."""
    w, h = min(size, width), min(size, height)
    x0 = int(np.clip(cx - w / 2.0, 0, width - w))
    y0 = int(np.clip(cy - h / 2.0, 0, height - h))
    return x0, y0, w, h


def detect_crop(model, image, crop, conf, device):
    """detect() on image[y0:y0+h, x0:x0+w] at the crop's own size; boxes in full-frame pixels."""
    x0, y0, w, h = crop
    imgsz = int(np.ceil(max(w, h) / 32.0) * 32)        # a multiple of YOLO's stride
    return [(x1 + x0, y1 + y0, x2 + x0, y2 + y0, score) for x1, y1, x2, y2, score in
            detect(model, image[y0:y0 + h, x0:x0 + w], imgsz, conf, device)]


# -- pitch <-> pixels --------------------------------------------------------------------------

def to_pixel(matrix, xy_m):
    """Pixel (int u, int v) of a pitch point in metres, or None if not finite."""
    u, v = project(matrix, xy_m[0] / PITCH[0], xy_m[1] / PITCH[1])
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    return int(round(float(u))), int(round(float(v)))


def box_on_pitch(matrix, box):
    """Pitch position (metres) of a box centre, assuming the ball is on the grass."""
    x, y = unproject(matrix, (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
    return np.array([float(x), float(y)]) * PITCH


# -- pass 1: YOLO + fusion ---------------------------------------------------------------------

def analyse(cap, model, start, stop, matrices, valid, atd, fps, conf, device, imgsz, crop,
            fallback_full, fusion_config):
    """Per frame: dict with the ATD ball, where YOLO looked, its boxes and the fused ball."""
    fusion = BallFusion(PITCH, fps, fusion_config)
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    began_all = time.time()
    for frame in range(start, stop):
        ok, image = cap.read()
        if not ok:
            break
        height, width = image.shape[:2]
        matrix = matrices[frame] if valid[frame] else None

        ball_x, ball_y, _ = atd[frame][1]
        atd_m = np.array([ball_x, ball_y]) * PITCH        # NaN when the ATD has no ball
        predicted = fusion.predict(frame)

        # Where to look: the ATD ball if it is on screen, else the predicted ball
        centre = None
        if matrix is not None:
            for point in (atd_m, predicted):
                if point is not None and np.all(np.isfinite(point)):
                    pixel = to_pixel(matrix, point)
                    if pixel is not None and inside(pixel[0], pixel[1], width, height):
                        centre = pixel
                        break

        began = time.time()
        region, mode = None, "skipped"
        if centre is not None:
            region = crop_around(*centre, crop, width, height)
            mode = "crop on ATD" if np.all(np.isfinite(atd_m)) else "crop on prediction"
            boxes = detect_crop(model, image, region, conf, device)
        elif fallback_full:
            mode = "full frame {}px".format(imgsz)
            boxes = detect(model, image, imgsz, conf, device)
        else:
            boxes = []
        yolo_ms = 1000 * (time.time() - began)

        on_pitch = [(box_on_pitch(matrix, b) if matrix is not None else np.full(2, np.nan), b[4])
                    for b in boxes]
        fused = fusion.update(frame, atd_m, on_pitch)
        frames.append({"boxes": boxes, "region": region, "mode": mode, "fused": fused,
                       "yolo_ms": yolo_ms})

        done = frame - start + 1
        if done % 250 == 0:
            rate = done / (time.time() - began_all)
            print("  analysed {}/{} frames  ({:.1f} fps, ~{:.0f} s left)".format(
                done, stop - start, rate, (stop - start - done) / rate))
    return frames


# -- drawing -----------------------------------------------------------------------------------

def draw_panel(image, info, result, result_atd):
    """Top-left panel: ball source, state, owner, team with the fused ball vs the ATD alone."""
    teams = [team for team, _ in info]

    def who(slot):
        return "-" if slot is None else "{} slot {}".format(teams[slot], slot + 1)

    source = result["source"]
    lines = [
        ("ball: {}".format(source), SOURCE_COLOURS.get(source)),
        ("state: {}".format(result["fused"].state), (255, 255, 255)),
        ("owner: {}".format(who(result["fused"].owner)), OWNER),
        ("nearest: {}".format(who(result["fused"].nearest)), NEAREST),
        ("team (ATD + YOLO): {}".format(result["fused"].team or "-"), (0, 255, 255)),
        ("team (ATD only):   {}".format(result_atd.team or "-"), (200, 200, 200)),
    ]
    if result["fused"].team != result_atd.team:
        lines.append(("DIFFER: YOLO changed the call", (0, 0, 255)))
    height = 20 + 30 * len(lines)
    cv2.rectangle(image, (15, 15), (560, 15 + height), (30, 30, 30), -1)
    cv2.rectangle(image, (15, 15), (560, 15 + height), SOURCE_COLOURS.get(source), 3)
    for i, (text, colour) in enumerate(lines):
        cv2.putText(image, text, (30, 45 + 30 * i), FONT, 0.7, colour, 2, cv2.LINE_AA)


def draw_frame(image, matrix, frame, slots, info, analysis, result, result_atd, atd_ball, fps):
    """Everything for one frame, onto the full-resolution BGR image in place."""
    height, width = image.shape[:2]
    fused = analysis["fused"]

    if analysis["region"] is not None:
        x0, y0, w, h = analysis["region"]
        cv2.rectangle(image, (x0, y0), (x0 + w, y0 + h), CROP_COLOUR, 1)

    for slot, ((x, y, _), (team, is_goalkeeper)) in enumerate(zip(slots, info)):
        if np.isnan(x):
            continue
        u, v = project(matrix, x, y)
        if not inside(u, v, width, height):
            continue
        point = (int(round(float(u))), int(round(float(v))))
        colour = COLOURS[(team, "gk")] if is_goalkeeper else COLOURS[team]
        cv2.circle(image, point, DOT_RADIUS, colour, -1)
        cv2.circle(image, point, DOT_RADIUS, (0, 0, 0), 1)
        if slot == result.nearest:
            cv2.drawMarker(image, point, NEAREST, cv2.MARKER_SQUARE, 26, 2)
        if slot == result.owner:
            cv2.circle(image, point, 16, OWNER, 3, cv2.LINE_AA)

    for index, (x1, y1, x2, y2, score) in enumerate(analysis["boxes"]):
        used = index == fused.used
        colour = USED if used else BOX
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), colour, 3 if used else 1)
        cv2.putText(image, "{:.2f}".format(score), (int(x1), int(y1) - 6), FONT, 0.5, colour, 1,
                    cv2.LINE_AA)

    if np.all(np.isfinite(atd_ball)):
        point = to_pixel(matrix, atd_ball)
        if point is not None:
            cv2.circle(image, point, 14, ATD_BALL, 2, cv2.LINE_AA)
    if fused.predicted is not None and np.all(np.isfinite(fused.predicted)):
        point = to_pixel(matrix, fused.predicted)
        if point is not None:
            cv2.drawMarker(image, point, FUSED, cv2.MARKER_TILTED_CROSS, 18, 2)
    if np.all(np.isfinite(fused.ball)):
        point = to_pixel(matrix, fused.ball)
        if point is not None:
            cv2.circle(image, point, 6, FUSED, -1, cv2.LINE_AA)

    draw_panel(image, info, {"source": fused.source, "fused": result}, result_atd)
    cv2.rectangle(image, (0, height - STRIP_HEIGHT), (width, height), (0, 0, 0), -1)
    seconds = frame / fps
    text = "frame {}  {:02d}:{:05.2f}  |  {}  |  YOLO boxes: {}  |  YOLO {:.0f} ms".format(
        frame, int(seconds // 60), seconds % 60, analysis["mode"], len(analysis["boxes"]),
        analysis["yolo_ms"])
    cv2.putText(image, text, (15, height - 17), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


# -- run ---------------------------------------------------------------------------------------

def run(start, seconds=10, weights="yolo11s.pt", imgsz=1280, conf=0.1, device=0, scale=0.75,
        out=None, crop=640, fallback_full=True, fusion_config=None, possession_config=None):
    """Analyse video frames [start, start + seconds) with YOLO + ATD, then play the result.

    imgsz          whole-frame inference size (the fallback)
    conf           minimum confidence of a YOLO box at all (the fusion has its own, fill_conf)
    crop           side of the search square, in pixels of the original frame
    fallback_full  with nothing to centre on: search the whole frame (True) or skip it (False)
    """
    fusion_config = fusion_config or FusionConfig()
    possession_config = possession_config or PossessionConfig()
    video_path, meta_path, raw_path = require(["video", "atd_meta", "atd_raw"])
    os.makedirs(MODELS_DIR, exist_ok=True)
    model = YOLO(weights if os.path.isabs(weights) else os.path.join(MODELS_DIR, weights))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("could not open {}".format(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    stop = start + int(round(seconds * fps))

    # A margin around the range, for the classifier's smoothing and centred velocities
    margin = possession_config.smooth_frames + possession_config.velocity_frames
    first, last = max(0, start - margin), stop + margin
    print("reading homography ...")
    matrices, _, valid = load_homography()
    print("reading ATD frames {}..{} ...".format(first, last - 1))
    atd = read_atd(raw_path, first, last)
    last = first + len(atd)
    info = slot_info(meta_path)
    teams = [team for team, _ in info]

    print("pass 1: YOLO + fusion on {} frames ...".format(stop - start))
    analysis = analyse(cap, model, start, stop, matrices, valid, atd, fps, conf, device, imgsz,
                       crop, fallback_full, fusion_config)
    stop = start + len(analysis)

    # The classifier, on the fused ball and on the ATD ball alone (outside [start, stop) both are
    # the ATD, so the margin frames only serve the smoothing)
    loaded = range(first, last)
    players = np.array([[s[:2] for s in atd[f][0]] for f in loaded], dtype=np.float64) * PITCH
    atd_ball = np.array([atd[f][1][:2] for f in loaded], dtype=np.float64) * PITCH
    fused_ball = atd_ball.copy()
    for k, item in enumerate(analysis):
        fused_ball[start - first + k] = item["fused"].ball
    results, *_ = classify(players, fused_ball, teams, possession_config, fps)
    results_atd, *_ = classify(players, atd_ball, teams, possession_config, fps)
    sources = [item["fused"].source for item in analysis]
    print("ball sources: {}".format({s: sources.count(s) for s in sorted(set(sources))}))
    changed = sum(results[start - first + k].team != results_atd[start - first + k].team
                  for k in range(len(analysis)))
    print("frames where YOLO changed the team in possession: {} of {}".format(changed,
                                                                             len(analysis)))

    writer = None
    if out:
        out = os.path.abspath(out)
        os.makedirs(os.path.dirname(out), exist_ok=True)   # OpenCV does not create folders
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                 (int(1920 * scale), int(1080 * scale)))
        if not writer.isOpened():
            raise IOError("could not open {} for writing".format(out))
        print("writing {} ...".format(out))

    print("pass 2: playback")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    delay_ms = max(1, int(1000 / fps))
    paused = False
    frame = start
    while frame < stop:
        if not paused:
            ok, image = cap.read()
            if not ok:
                break
            i = frame - first
            if valid[frame]:
                draw_frame(image, matrices[frame], frame, atd[frame][0], info,
                           analysis[frame - start], results[i], results_atd[i], atd_ball[i], fps)
            shown = cv2.resize(image, None, fx=scale, fy=scale)
            frame += 1

        if writer is not None:
            writer.write(shown)
            continue

        cv2.imshow("possession: ATD + YOLO", shown)
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

    # -- what to run -------------------------------------------------------------------------
    START = 19000                   # video frame to start from (minute * 1500 + second * 25)
    SECONDS = 120                  # pass 1 runs YOLO on every frame first: this is slow
    WEIGHTS = "yolo11s.pt"          # yolo11n.pt is faster, yolo11m.pt finds more but is slower
    IMGSZ = 1280                    # whole-frame size (only for FALLBACK_FULL)
    CONF = 0.1                      # minimum confidence of a YOLO box at all
    DEVICE = 0                      # 0 = GPU, "cpu" = no GPU
    SCALE = 0.75                    # window size relative to 1920x1080
    OUT = None                      # e.g. "data/derived/yolo_possession.mp4" to write the video

    # -- where YOLO looks --------------------------------------------------------------------
    CROP = 600                      # search square in pixels, centred on the ATD / predicted ball
    FALLBACK_FULL = True            # nothing to centre on: whole frame (True) or skip (False)

    # -- ATD + YOLO fusion (metres, m/s, seconds) --------------------------------------------
    FUSION = FusionConfig(
        agree_m=2.0,                # a YOLO box this close to the ATD ball agrees with it
        fill_conf=0.2,              # minimum confidence to fill an ATD gap with YOLO
        max_gap_seconds=2.0,        # longer ATD gaps are not filled
        gate_m=2.0,                 # base radius around the predicted ball
        v_max=30.0,                 # the ball cannot move faster than this
        pitch_margin_m=3.0,         # how far outside the lines a filled ball may be
        flight_speed=10.0,          # a ball faster than this is in flight ...
        fill_in_flight=False,       # ... and its gaps are not filled with YOLO
    )

    # -- the possession classifier (metres, m/s), as in src/visualizing_possession.py -------
    POSSESSION = PossessionConfig(
        smooth_frames=5, velocity_frames=4, d_max=3.0, d0=1.5, v0=2.0, w_dir=0.5, moving=0.5,
        stick_slot=1.0, stick_team=0.5, min_score=-2.0, flight_speed=10.0,
    )

    run(start=START, seconds=SECONDS, weights=WEIGHTS, imgsz=IMGSZ, conf=CONF, device=DEVICE,
        scale=SCALE, out=OUT, crop=CROP, fallback_full=FALLBACK_FULL, fusion_config=FUSION,
        possession_config=POSSESSION)
