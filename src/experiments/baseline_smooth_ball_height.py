"""A baseline + freezing possession while the ball is in the air.

Run from the repository root:
    python src/experiments/baseline_smooth_ball_height.py

What to run is set at the bottom of this file, in the `if __name__ == "__main__":` block, including
which baseline to start from: BASELINE = "raw", "smooth25", "smooth50", "smooth125" or any
"smoothN" (a centred majority vote over +/-N frames, as in starter/baseline.py).

This script does NOT save or upload video. It reads the tactical video locally for YOLO inference
and writes one frame,label CSV to evaluation/results/ball_height_<baseline>.csv. The name does not
start with "baseline_" on purpose: .gitignore keeps every CSV in that folder out of git except
"baseline_*.csv", and a submission does not belong in a public repository.

The idea. A lofted ball passes over players, so the nearest-player baseline hands it to whoever it
flies over. While the ball is in the air, possession should stay with the team that kicked it,
until somebody has it again (CHALLENGE.md: possession stays with the last team that had it). The
height tracker in src/tracking_height/ tells, per frame, whether the ball is on the ground or in
the air -- but it often loses an airborne ball against the stands or the background, so a "no
detection" must not be read as "back on the ground". Hence two states:

  NORMAL   the baseline label is kept. When AIR_CONFIRM 'air' frames are seen within AIR_WINDOW
           frames, the ball is in flight: possession is frozen to the baseline label LOOKBACK
           frames before the first of those air frames (the ball has already travelled a little by
           the time it is detected in the air, and the baseline may have switched by then).
  IN FLIGHT  every frame gets the frozen label. Losing the ball ('no_detection', 'inconsistent',
           no ATD ball) or seeing it in the air again keeps the freeze. It ends when:
             - STABLE: the ATD and YOLO agree that the ball is on the ground (a detection on the ATD
               vertical, low and close: state 'ground') on STABLE_FRAMES frames since the last air
               frame -- somebody has the ball again;
             - OUT: the ATD puts the ball outside the pitch (x or y outside 0..1);
             - TIMEOUT: MAX_FLIGHT_SECONDS have passed, a safety net so a flight that never lands
               cannot freeze possession for minutes.
           From then on the baseline label is used again.

Frames outside [START, START + SECONDS) keep the baseline labels. No labels or ground truth are used
for building or tuning any of this.
"""

import os
import re
import sys
from collections import Counter, deque

import cv2
import numpy as np
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))   # src/experiments
SRC = os.path.dirname(HERE)                          # src
ROOT = os.path.dirname(SRC)                          # repository root
sys.path.insert(0, ROOT)                             # games, starter
sys.path.insert(0, SRC)                              # visualizing_tracking, tracking_height

from games import GAME, GAMES, half_video_blocks, require  # noqa: E402
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


# -- baseline ----------------------------------------------------------------------------------

def baseline_window(name):
    """Half-width of the baseline's majority vote from its name: "raw" -> 0, "smooth50" -> 50."""
    if name == "raw":
        return 0
    match = re.fullmatch(r"smooth(\d+)", name)
    if not match:
        raise ValueError(f"baseline must be 'raw' or 'smoothN' (e.g. 'smooth50'), got {name!r}")
    return int(match.group(1))


# -- height per frame --------------------------------------------------------------------------

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


def height_state(model, image, matrix, is_valid, ball_x, ball_y, pitch, config, conf, device,
                 max_imgsz):
    """The height tracker's state for one frame: ground, air, inconsistent, no_detection,
    no_camera or no_atd (see tracking_height/ball_height.py)."""
    if not np.isfinite(ball_x) or not np.isfinite(ball_y):
        return "no_atd"
    camera = decompose(matrix, pitch=pitch) if is_valid else None
    if camera is None:
        return "no_camera"
    x_m, y_m = ball_x * pitch[0], ball_y * pitch[1]
    region = search_region(camera, x_m, y_m, config, (image.shape[1], image.shape[0]))
    boxes = detect(model, image, region, conf, device, max_imgsz) if region is not None else []
    fits = fit_boxes(camera, x_m, y_m, boxes, config)
    return classify(fits, choose(fits), config)


def outside_pitch(ball_x, ball_y):
    """True when the ATD puts the ball beyond a touchline or a goal line."""
    return bool(np.isfinite(ball_x) and np.isfinite(ball_y)
                and not (0.0 <= ball_x <= 1.0 and 0.0 <= ball_y <= 1.0))


# -- freezing possession in flight -------------------------------------------------------------

class FlightFreezer:
    """The NORMAL / IN FLIGHT state machine of the module docstring, fed one frame at a time.

    It writes, per frame, the label into `out` (a copy of the baseline): the baseline's, or the
    frozen one while in flight. When a flight is confirmed, the frames since its first air frame
    are frozen too, since by then it is known the flight started there. Finished flights are kept
    in `flights` as (first_frame, end_frame, frozen_label, reason).
    """

    def __init__(self, baseline_labels, out, air_confirm, air_window, lookback, stable_frames,
                 max_flight_frames):
        self.baseline = baseline_labels
        self.out = out
        self.air_confirm = air_confirm
        self.lookback = lookback
        self.stable_frames = stable_frames
        self.max_flight_frames = max_flight_frames
        self.recent_air = deque(maxlen=air_window)   # (frame, was_air) of the last few frames
        self.frozen = None                           # the frozen label while in flight, else None
        self.flight_start = None
        self.ground_since_air = 0
        self.flights = []

    def _end_flight(self, frame, reason):
        self.flights.append((self.flight_start, frame, self.frozen, reason))
        self.frozen, self.flight_start, self.ground_since_air = None, None, 0
        self.recent_air.clear()

    def step(self, frame, state, ball_x, ball_y):
        """Write the label of `frame`, given the height tracker's state and the ATD ball there."""
        self.out[frame] = self._label(frame, state, ball_x, ball_y)

    def _label(self, frame, state, ball_x, ball_y):
        air = state == "air"
        self.recent_air.append((frame, air))

        if self.frozen is None:
            air_frames = [f for f, was_air in self.recent_air if was_air]
            if len(air_frames) >= self.air_confirm:
                anchor = max(0, air_frames[0] - self.lookback)
                label = self.baseline[anchor]
                if label in ("H", "A"):           # nothing to keep if nobody had the ball
                    self.frozen, self.flight_start = label, air_frames[0]
                    self.ground_since_air = 0
                    self.out[air_frames[0]:frame] = label   # the flight began at the first air
            if self.frozen is None:
                return self.baseline[frame]

        # In flight
        if outside_pitch(ball_x, ball_y):
            self._end_flight(frame, "out")
            return self.baseline[frame]
        if frame - self.flight_start >= self.max_flight_frames:
            self._end_flight(frame, "timeout")
            return self.baseline[frame]
        if air:
            self.ground_since_air = 0
        elif state == "ground":                   # a detection on the vertical, low: ATD == YOLO
            self.ground_since_air += 1
            if self.ground_since_air >= self.stable_frames:
                self._end_flight(frame, "stable")
                return self.baseline[frame]
        return self.frozen

    def finish(self, frame):
        """Close a flight still open when the processed range ends."""
        if self.frozen is not None:
            self._end_flight(frame, "end_of_range")


# -- run ---------------------------------------------------------------------------------------

def run(start, seconds=None, baseline="smooth50", weights="yolo11s.pt", conf=0.01, device=0,
        max_imgsz=960, max_residual_m=1.5, ground_max=2.0, air_confirm=2, air_window=5,
        lookback=10, stable_frames=5, max_flight_seconds=5.0, out=RESULTS_DIR, log_every=1000):
    """Apply the in-flight freeze to `baseline` over [start, start + seconds) and write the CSV.

    seconds  None processes from `start` to the end of the video.
    """
    window = baseline_window(baseline)
    fps = GAMES[GAME]["fps"]
    pitch = GAMES[GAME]["pitch"]

    print(f"Building baseline '{baseline}' (+/-{window} frames) ...")
    baseline_labels = smooth(build(), window)
    n_frames = len(baseline_labels)
    if not 0 <= start < n_frames:
        raise ValueError(f"start must be between 0 and {n_frames - 1}")
    if seconds is not None and seconds <= 0:
        raise ValueError("seconds must be positive (or None for the whole video)")
    stop = n_frames if seconds is None else min(n_frames, start + int(round(seconds * fps)))

    video_path, raw_path = require(["video", "atd_raw"])
    model_path = weights if os.path.isabs(weights) else os.path.join(MODELS_DIR, weights)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"YOLO weights not found: {model_path}")
    model = YOLO(model_path)
    print("Loading homographies ...")
    matrices, _, valid = load_homography()
    print(f"Reading ATD frames {start}..{stop - 1} ...")
    atd = read_atd(raw_path, start, stop)

    config = HeightConfig(max_residual_m=max_residual_m, ground_max=ground_max)
    corrected = baseline_labels.copy()
    freezer = FlightFreezer(baseline_labels, corrected, air_confirm, air_window, lookback,
                            stable_frames, int(round(max_flight_seconds * fps)))
    states = Counter()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"could not open {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    try:
        for frame in range(start, stop):
            ok, image = cap.read()
            if not ok:
                raise IOError(f"Video ended or frame could not be decoded at {frame}")
            ball_x, ball_y, _ = atd[frame][1]
            state = height_state(model, image, matrices[frame], valid[frame], ball_x, ball_y,
                                 pitch, config, conf, device, max_imgsz)
            states[state] += 1
            freezer.step(frame, state, ball_x, ball_y)
            if log_every and (frame - start + 1) % log_every == 0:
                print(f"  {frame - start + 1}/{stop - start} frames")
    finally:
        cap.release()
    freezer.finish(stop)

    output_path = write_submission(os.path.join(out, f"ball_height_{baseline}.csv"), corrected)

    print("\nHeight states (raw):", dict(states))
    reasons = Counter(reason for *_, reason in freezer.flights)
    frozen = sum(end - first for first, end, *_ in freezer.flights)
    print(f"Flights: {len(freezer.flights)}  (ended by: {dict(reasons)})")
    print(f"Frames frozen: {frozen}")
    print(f"Frames whose label actually changed: "
          f"{int(np.count_nonzero(corrected != baseline_labels))}")
    print(f"Submission written: {output_path}")
    if seconds is not None:
        print("NOTE: frames outside the processed range keep the baseline labels.")
    return corrected, freezer.flights


if __name__ == "__main__":
    first_half_kickoff = half_video_blocks(GAME)[0][1]     # 11185, first graded frame

    # -- what to run -------------------------------------------------------------------------
    BASELINE = "smooth25"           # "raw", "smooth25", "smooth50", "smooth125" or any "smoothN"
    START = first_half_kickoff      # video frame to start from (minute * 1500 + second * 25)
    SECONDS = 500                   # how long to process; None = from START to the end

    # height tracker
    WEIGHTS = "yolo11s.pt"          # YOLO weights in models/
    CONF = 0.01                     # minimum YOLO confidence
    DEVICE = 0                      # 0 = GPU, "cpu" = no GPU
    MAX_IMGSZ = 960                 # largest YOLO input; lower it if the GPU runs out of memory
    MAX_RESIDUAL_M = 1.5            # boxes farther than this from the vertical are not used
    GROUND_MAX = 2.0                # z up to this is "ground"

    # in-flight freeze
    AIR_CONFIRM = 2                 # 'air' frames needed ...
    AIR_WINDOW = 5                  # ... within this many frames to start a flight
    LOOKBACK = 10                   # freeze the baseline label this many frames before take-off
    STABLE_FRAMES = 5               # 'ground' frames (ATD == YOLO) since the last air to land
    MAX_FLIGHT_SECONDS = 5.0        # safety net: a flight never lasts longer than this

    OUT = RESULTS_DIR               # writes OUT/ball_height_<BASELINE>.csv

    run(start=START, seconds=SECONDS, baseline=BASELINE, weights=WEIGHTS, conf=CONF,
        device=DEVICE, max_imgsz=MAX_IMGSZ, max_residual_m=MAX_RESIDUAL_M, ground_max=GROUND_MAX,
        air_confirm=AIR_CONFIRM, air_window=AIR_WINDOW, lookback=LOOKBACK,
        stable_frames=STABLE_FRAMES, max_flight_seconds=MAX_FLIGHT_SECONDS, out=OUT)
