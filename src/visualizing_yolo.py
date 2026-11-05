"""First look at a ball detector on the tactical video: every YOLO 'sports ball' box, unfiltered.

A COCO-pretrained YOLO runs on each frame of a short clip and every detection of class 32 (sports
ball) is drawn as a green box with its confidence -- no tracking, no filtering, no choosing, so you
see exactly what the detector returns. For comparison the ATD ball is projected with the homography
and drawn as a yellow circle. The bottom strip gives, per frame, how many boxes YOLO found, the best
confidence, whether the ATD has a ball, and the pixel distance between the ATD ball and the closest
box.

    python src/visualizing_yolo.py

What to run is set at the bottom of this file, in the `if __name__ == '__main__':` block.
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
from games import GAME, half_video_blocks, require  # noqa: E402
from starter.load_homography import load as load_homography, project  # noqa: E402
from visualizing_tracking import FONT, inside, read_atd  # noqa: E402

# Params
SPORTS_BALL = 32                    # COCO class id
BOX = (0, 255, 0)                   # BGR: YOLO detections, green
ATD_BALL = (0, 255, 255)            # BGR: ATD ball, yellow
STRIP_HEIGHT = 50
MODELS_DIR = os.path.join(ROOT, "models")


def detect(model, image, imgsz, conf, device):
    """[(x1, y1, x2, y2, confidence)] for every sports-ball box YOLO finds in one BGR frame."""
    result = model.predict(image, imgsz=imgsz, conf=conf, classes=[SPORTS_BALL], device=device,
                           verbose=False)[0]
    boxes = result.boxes
    return [(*box, score) for box, score in zip(boxes.xyxy.cpu().numpy().tolist(),
                                               boxes.conf.cpu().numpy().tolist())]


def draw(image, detections, atd_pixel, video_frame, fps, infer_ms):
    """Boxes, ATD ball and the bottom strip, drawn onto `image` (full resolution) in place."""
    height, width = image.shape[:2]
    for x1, y1, x2, y2, score in detections:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), BOX, 2)
        cv2.putText(image, "{:.2f}".format(score), (int(x1), int(y1) - 6), FONT, 0.6, BOX, 2,
                    cv2.LINE_AA)

    distance = None
    if atd_pixel is not None:
        cv2.circle(image, atd_pixel, 14, ATD_BALL, 2, cv2.LINE_AA)
        if detections:
            centres = np.array([((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2, _ in detections])
            distance = float(np.min(np.hypot(*(centres - np.array(atd_pixel)).T)))

    cv2.rectangle(image, (0, height - STRIP_HEIGHT), (width, height), (0, 0, 0), -1)
    seconds = video_frame / fps
    text = ("frame {}  {:02d}:{:05.2f}  |  YOLO boxes: {}  best conf: {}  |  ATD ball: {}  |  "
            "ATD-to-closest-box: {}  |  {:.0f} ms").format(
        video_frame, int(seconds // 60), seconds % 60, len(detections),
        "{:.2f}".format(max(d[4] for d in detections)) if detections else "-",
        atd_pixel is not None, "{:.0f} px".format(distance) if distance is not None else "-",
        infer_ms)
    cv2.putText(image, text, (15, height - 17), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def run(start, seconds=10, weights="yolo11s.pt", imgsz=1280, conf=0.1, device=0, scale=0.75,
        out=None):
    """Run YOLO on video frames [start, start + seconds) and show (or write) every detection.

    weights  a COCO model name (downloaded once into models/) or a path to a .pt file
    imgsz    inference size: larger finds smaller balls, and is slower
    conf     minimum confidence kept; low on purpose, to see the weak detections too
    device   0 for the first GPU, "cpu" without one
    """
    video_path, raw_path = require(["video", "atd_raw"])
    os.makedirs(MODELS_DIR, exist_ok=True)
    model = YOLO(weights if os.path.isabs(weights) else os.path.join(MODELS_DIR, weights))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("could not open {}".format(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    stop = start + int(round(seconds * fps))

    print("reading homography ...")
    matrices, _, valid = load_homography()
    print("reading ATD frames {}..{} ...".format(start, stop - 1))
    atd = read_atd(raw_path, start, stop)

    writer = None
    if out:
        out = os.path.abspath(out)
        os.makedirs(os.path.dirname(out), exist_ok=True)   # OpenCV does not create folders
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                 (int(1920 * scale), int(1080 * scale)))
        if not writer.isOpened():
            raise IOError("could not open {} for writing".format(out))
        print("writing {} ...".format(out))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    paused = False
    video_frame = start
    while video_frame < stop:
        if not paused:
            ok, image = cap.read()
            if not ok:
                break
            began = time.time()
            detections = detect(model, image, imgsz, conf, device)
            infer_ms = 1000 * (time.time() - began)

            atd_pixel = None
            ball_x, ball_y, _ = atd[video_frame][1]
            if valid[video_frame] and not np.isnan(ball_x):
                u, v = project(matrices[video_frame], ball_x, ball_y)
                if inside(u, v, image.shape[1], image.shape[0]):
                    atd_pixel = (int(round(u)), int(round(v)))

            draw(image, detections, atd_pixel, video_frame, fps, infer_ms)
            shown = cv2.resize(image, None, fx=scale, fy=scale)
            video_frame += 1

        if writer is not None:
            writer.write(shown)
            continue

        cv2.imshow("tactical + YOLO", shown)
        key = cv2.waitKey(1) & 0xFF   # no pacing: YOLO is the bottleneck, not the frame rate
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
    START = first_half_kickoff      # video frame to start from (minute * 1500 + second * 25)
    SECONDS = 25                    # keep it short: YOLO runs on every frame
    WEIGHTS = "yolo11s.pt"          # yolo11n.pt is faster, yolo11m.pt finds more but is slower
    IMGSZ = 1280                    # 640 is faster but misses small balls
    CONF = 0.1                      # low on purpose, to see weak detections as well
    DEVICE = 0                      # 0 = GPU, "cpu" = no GPU
    SCALE = 0.75                    # window size relative to 1920x1080
    OUT = "data/derived/yolo_clip.mp4"                      # e.g. "yolo_clip.mp4" to write a file instead of a window

    run(start=START, seconds=SECONDS, weights=WEIGHTS, imgsz=IMGSZ, conf=CONF, device=DEVICE,
        scale=SCALE, out=OUT)
