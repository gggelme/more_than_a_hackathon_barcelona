"""Baseline smooth50 + correction of possession during ball-in-air intervals.

Run from the repository root:
    python src/baseline_smooth50_ball_height.py

This script does NOT save or upload video. It reads the tactical video locally for
YOLO inference and writes frame,label CSV submissions to evaluation/results/.

Algorithm:
  1. Produce baseline_smooth50 using starter.baseline.build/smooth (a centred
     majority vote over +/-50 frames, i.e. up to 101 frames total).
  2. Classify height per video frame using the existing homography/YOLO pipeline.
     Treat 'inconsistent' and 'no_detection' as 'ground' for voting.
     Smooth the ground/air classes with a trailing 15-frame vote.
  3. For every maximal consecutive interval classified 'air', copy the ORIGINAL
     baseline label from the frame immediately BEFORE the interval into ALL
     frames of that interval. Outside those intervals, preserve the baseline.

No labels/GT are used for model training or selection.
"""

import argparse
import os
import sys
from collections import Counter, deque

import cv2
import numpy as np
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from games import GAME, GAMES, require  # noqa: E402
from starter.baseline import build, smooth, write_submission  # noqa: E402
from starter.load_homography import load as load_homography  # noqa: E402
from visualizing_tracking import read_atd  # noqa: E402
from tracking_height.camera_from_homography import decompose  # noqa: E402
from tracking_height.ball_height import (  # noqa: E402
    HeightConfig, choose, classify, fit_boxes, search_region,
)

SPORTS_BALL = 32  # COCO sports ball category
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
MODELS_DIR = os.path.join(ROOT, "models")


def detect(model, image, region, conf, device, max_imgsz):
    """Return sports-ball boxes in full-frame coordinates, exactly as in the visualizer."""
    x0, y0, (width, height) = region
    crop = image[y0:y0 + height, x0:x0 + width]
    imgsz = min(max_imgsz, int(np.ceil(max(width, height) / 32.0) * 32))
    result = model.predict(
        crop, imgsz=imgsz, conf=conf, classes=[SPORTS_BALL],
        device=device, verbose=False,
    )[0]
    boxes = result.boxes.xyxy.cpu().numpy() + np.array([x0, y0, x0, y0])
    return [(*box, confidence) for box, confidence in zip(
        boxes.tolist(), result.boxes.conf.cpu().numpy().tolist()
    )]


def vote_height(history, raw_state):
    """Vote over recent frames; do not turn a missing ATD/camera into an air interval.

    'inconsistent' and 'no_detection' are explicitly counted as ground.
    'no_atd' and 'no_camera' do not cast a vote and BREAK any current air interval.
    On a tie, prefer the most recent valid observation.
    """
    if raw_state in ("ground", "inconsistent", "no_detection"):
        observation = "ground"
    elif raw_state == "air":
        observation = "air"
    else:
        observation = None

    history.append(observation)
    if observation is None:
        return "unknown"

    counts = Counter(state for state in history if state is not None)
    if counts["air"] > counts["ground"]:
        return "air"
    if counts["ground"] > counts["air"]:
        return "ground"
    return observation  # deterministic tie: latest valid state


def air_intervals(air_flags):
    """Yield half-open [start, stop) runs of air frames."""
    start = None
    for frame, air in enumerate(air_flags):
        if air and start is None:
            start = frame
        elif not air and start is not None:
            yield start, frame
            start = None
    if start is not None:
        yield start, len(air_flags)


def replace_air_with_previous_baseline(baseline_labels, air_flags):
    """Use immutable baseline labels as anchors, avoiding cascading edits."""
    corrected = baseline_labels.copy()
    intervals = 0
    replaced = 0
    for start, stop in air_intervals(air_flags):
        if start == 0:
            # There is no previous frame; retain the baseline in this interval.
            continue
        corrected[start:stop] = baseline_labels[start - 1]
        intervals += 1
        replaced += stop - start
    return corrected, intervals, replaced


def predict_air_frames(start, stop, weights, conf, device, config,
                       max_imgsz, voting_window, log_every=1000):
    """Classify only [start, stop), with 0-based video-frame IDs."""
    video_path, raw_path = require(["video", "atd_raw"])
    pitch = GAMES[GAME]["pitch"]
    model_path = weights if os.path.isabs(weights) else os.path.join(MODELS_DIR, weights)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"YOLO weights not found: {model_path}. Download the weights separately "
            "before running; the match video is never sent anywhere."
        )
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"could not open {video_path}")

    print("Loading homographies ...")
    matrices, _, valid = load_homography()
    print(f"Reading ATD frames {start}..{stop - 1} ...")
    atd = read_atd(raw_path, start, stop)

    flags = np.zeros(stop - start, dtype=bool)
    history = deque(maxlen=voting_window)
    states = Counter()
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    try:
        for frame in range(start, stop):
            ok, image = cap.read()
            if not ok:
                raise IOError(f"Video ended or frame could not be decoded at {frame}")

            slots, ball = atd[frame]
            ball_x, ball_y, _ = ball
            if not np.isfinite(ball_x) or not np.isfinite(ball_y):
                raw_state = "no_atd"
            elif not valid[frame]:
                raw_state = "no_camera"
            else:
                camera = decompose(matrices[frame], pitch=pitch)
                if camera is None:
                    raw_state = "no_camera"
                else:
                    x_m, y_m = ball_x * pitch[0], ball_y * pitch[1]
                    region = search_region(
                        camera, x_m, y_m, config,
                        (image.shape[1], image.shape[0]),
                    )
                    boxes = detect(model, image, region, conf, device, max_imgsz) if region is not None else []
                    fits = fit_boxes(camera, x_m, y_m, boxes, config)
                    chosen = choose(fits)
                    raw_state = classify(fits, chosen, config)

            states[raw_state] += 1
            flags[frame - start] = vote_height(history, raw_state) == "air"
            if log_every and (frame - start + 1) % log_every == 0:
                print(f"  height tracking: {frame - start + 1}/{stop - start} frames")
    finally:
        cap.release()

    print("Height states (raw):", dict(states))
    return flags


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", default="yolo11s.pt")
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--device", default="0", help="CUDA GPU index (0) or cpu")
    parser.add_argument("--max-imgsz", type=int, default=960)
    parser.add_argument("--voting-window", type=int, default=15)
    parser.add_argument("--max-residual-m", type=float, default=1.5)
    parser.add_argument("--ground-max", type=float, default=2.0)
    parser.add_argument("--start", type=int, default=0, help="First frame to process (default: 0)")
    parser.add_argument("--seconds", type=float, default=None,
                        help="Process only this many seconds for testing (default: full video)")
    parser.add_argument("--out", default=RESULTS_DIR, help="Output CSV folder")
    args = parser.parse_args()

    if args.voting_window < 1:
        parser.error("--voting-window must be >= 1")

    print("Building exact baseline_smooth50 ...")
    baseline_labels = smooth(build(), 50)
    n_frames = len(baseline_labels)
    if not 0 <= args.start < n_frames:
        parser.error(f"--start must be between 0 and {n_frames - 1}")

    baseline_path = write_submission(
        os.path.join(args.out, "baseline_smooth50.csv"), baseline_labels
    )
    print(f"Baseline written: {baseline_path}")

    # Video and ATD frame indices are aligned, beginning at 0.
    if args.seconds is None:
        stop = n_frames
    else:
        if args.seconds <= 0:
            parser.error("--seconds must be positive")
        video_path = require(["video"])[0]
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"could not open {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps <= 0:
            raise ValueError("Video FPS is not available")
        stop = min(n_frames, args.start + int(round(args.seconds * fps)))

    config = HeightConfig(
        max_residual_m=args.max_residual_m,
        ground_max=args.ground_max,
    )
    flags_partial = predict_air_frames(
        args.start, stop, args.weights, args.conf,
        int(args.device) if args.device.isdecimal() else args.device,
        config, args.max_imgsz, args.voting_window,
    )
    air_flags = np.zeros(n_frames, dtype=bool)
    air_flags[args.start:stop] = flags_partial

    # A partial run's first air interval can anchor on the preceding baseline
    # frame even though that preceding frame wasn't run through YOLO.
    corrected, intervals, air_frames = replace_air_with_previous_baseline(
        baseline_labels, air_flags
    )
    output_path = write_submission(
        os.path.join(args.out, "baseline_smooth50_ball_height.csv"), corrected
    )
    actual_changes = int(np.count_nonzero(corrected != baseline_labels))
    print(f"Air intervals replaced: {intervals}")
    print(f"Air frames covered: {air_frames}")
    print(f"Frames whose label actually changed: {actual_changes}")
    print(f"Submission written: {output_path}")
    if args.seconds is not None:
        print("NOTE: Frames outside the test range keep their original baseline labels.")


if __name__ == "__main__":
    main()
