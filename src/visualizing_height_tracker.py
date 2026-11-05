"""Watch the ball-height tracker on the tactical video. Display only: nothing is saved.

Per frame: the camera recovered from the homography, the ATD ball and the vertical above it, YOLO
on a crop around that vertical, every box tested against the vertical, and the chosen one with its
height. The pieces live in src/tracking_height/: camera_from_homography.py (the camera),
ball_height.py (height and state) and drawing.py.

    python src/visualizing_height_tracker.py

What to show is set at the bottom of this file, in the `if __name__ == '__main__':` block.
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
from starter.load_homography import load as load_homography  # noqa: E402
from visualizing_tracking import read_atd  # noqa: E402
from tracking_height.camera_from_homography import decompose  # noqa: E402
from tracking_height.ball_height import (HeightConfig, choose, classify, fit_boxes,  # noqa: E402
                                         search_region)
from tracking_height.drawing import draw_frame  # noqa: E402

# Params
SPORTS_BALL = 32                    # COCO class id
MODELS_DIR = os.path.join(ROOT, "models")


def detect(model, image, region, conf, device, max_imgsz):
    """[(x1, y1, x2, y2, confidence)] sports-ball boxes inside region, in full-frame pixels.

    The crop runs at its own size (so the ball is not shrunk) up to max_imgsz; a bigger crop is
    shrunk to max_imgsz, which is what keeps a 4 GB GPU from running out of memory.
    """
    x0, y0, (w, h) = region
    crop = image[y0:y0 + h, x0:x0 + w]
    imgsz = min(max_imgsz, int(np.ceil(max(w, h) / 32.0) * 32))   # multiple of YOLO's stride
    result = model.predict(crop, imgsz=imgsz, conf=conf, classes=[SPORTS_BALL], device=device,
                           verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy() + np.array([x0, y0, x0, y0])
    return [(*box, score) for box, score in zip(boxes.tolist(),
                                               result.boxes.conf.cpu().numpy().tolist())]


def run(start, seconds=10, weights="yolo11s.pt", conf=0.1, device=0, config=None, scale=0.75,
        max_imgsz=1088):
    """Show the tracker on video frames [start, start + seconds).

    max_imgsz  largest size YOLO runs at; lower it if the GPU runs out of memory
    """
    config = config or HeightConfig()
    video_path, raw_path = require(["video", "atd_raw"])
    pitch = GAMES[GAME]["pitch"]
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

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    paused = False
    frame = start
    while frame < stop:
        if not paused:
            ok, image = cap.read()
            if not ok:
                break
            began = time.time()

            camera = decompose(matrices[frame], pitch=pitch) if valid[frame] else None
            ball_x, ball_y, _ = atd[frame][1]
            fits, chosen, region, x_m, y_m = [], None, None, None, None
            if np.isnan(ball_x):
                state = "no_atd"
            elif camera is None:
                state = "no_camera"
            else:
                x_m, y_m = ball_x * pitch[0], ball_y * pitch[1]
                region = search_region(camera, x_m, y_m, config, (image.shape[1], image.shape[0]))
                boxes = (detect(model, image, region, conf, device, max_imgsz)
                         if region is not None else [])
                fits = fit_boxes(camera, x_m, y_m, boxes, config)
                chosen = choose(fits)
                state = classify(fits, chosen, config)

            draw_frame(image, frame, camera, x_m, y_m, fits, chosen, region, state, config.z_max,
                       fps, 1000 * (time.time() - began))
            shown = cv2.resize(image, None, fx=scale, fy=scale)
            frame += 1

        cv2.imshow("ball height tracker", shown)
        key = cv2.waitKey(1) & 0xFF   # no pacing: YOLO is the bottleneck, not the frame rate
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
    SECONDS = 5000                    # how long
    WEIGHTS = "yolo11s.pt"          # COCO model, as in visualizing_yolo.py
    CONF = 0.02                     # minimum YOLO confidence
    CONFIG = HeightConfig(          # thresholds in metres (see tracking_height/ball_height.py)
        max_residual_m=1.5,         # boxes farther than this from the vertical are not used
        ground_max=0.8,             # z up to this is "ground"
    )
    SCALE = 0.75                    # window size relative to 1920x1080
    MAX_IMGSZ = 960                # largest YOLO input; 960 or 832 if the GPU runs out of memory
    DEVICE = 0                      # 0 = GPU, "cpu" = no GPU

    run(start=START, seconds=SECONDS, weights=WEIGHTS, conf=CONF, device=DEVICE, config=CONFIG,
        scale=SCALE, max_imgsz=MAX_IMGSZ)
