"""visualizing_deadball's dead ball, frame by frame, from the video -- for the possession
(tracking_possession/goals.dead_ball, through tracking_possession/referee.py).

It runs the very detection and decision of src/visualizing_deadball.py, without its window: YOLO
finds the people, the homography + the ATD pick the referee, the stable states of his box and YOLO
Pose on his arms say when he gestures, and its DeadBallScore decides the dead ball -- the
gesture, the players' stillness, the ball out of the pitch, the hysteresis and its own restart
detection for the end. Per frame it keeps the referee's gesture score and the dead ball itself.

Video and YOLO are slow (GPU), so the results are cached in data/derived/referee/: the YOLO
caches of visualizing_deadball and a compact deadball_<start>_<stop>.npz. data/ is ignored by git:
these are frame-level derived data and must never be published (info_hackathon/RULES.md).

    python src/referee_signal.py

precomputes a range, set in the `if __name__ == '__main__':` block -- run it before a visualizer
that uses the referee, so the visualizer only reads the cache (YOLO and the viewer together may not
fit in memory).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, require, source_frames  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "data", "derived", "referee")
FIELDS = ("gesture", "dead")


def cache_path(start, stop, cache_dir=CACHE_DIR):
    return os.path.join(cache_dir, "deadball_{}_{}.npz".format(start, stop))


def _from_cache(start, stop, cache_dir):
    """The cached signals of any range covering [start, stop), cut to it; None if there is none."""
    if not os.path.isdir(cache_dir):
        return None
    for name in sorted(os.listdir(cache_dir)):
        parts = name[:-4].split("_") if name.startswith("deadball_") and \
            name.endswith(".npz") else []
        if len(parts) == 3 and int(parts[1]) <= start and int(parts[2]) >= stop:
            data = np.load(os.path.join(cache_dir, name))
            keep = (data["frames"] >= start) & (data["frames"] < stop)
            print("referee dead ball: cache {}".format(name))
            return dict(frames=data["frames"][keep],
                        **{field: data[field][keep] for field in FIELDS + ("box", "restart")
                           if field in data.files})
    return None


def deadball_signal(start, stop, args=None, cache_dir=CACHE_DIR, recompute=False):
    """{frames, gesture, dead, box} for video frames [start, stop), each (m,) -- box (m, 4):

      gesture   the referee's gesture score in [0, 1] (pose + box, visualizing_deadball's weights)
      dead      visualizing_deadball's OWN dead-ball decision (compute_scores / DeadBallScore:
                the referee's gesture, the players' stillness, the ball out of the pitch, the
                hysteresis, and its own restart detection for the end), True where D
      box       the box of whom it took for the referee, [x1, y1, x2, y2] in pixels, NaN where
                it found nobody (for drawing; caches written before it was added have none)
      restart   its own detected restarts (detect_ball_restarts: the ball leaves a player and
                moves away), True on the frame -- tracking_possession/referee.py ends the D at
                the first one after which the ball stays in play (older caches have none)

    Read from the cache when one covers the range; else visualizing_deadball's detection is run
    (YOLO Detect + Pose on the range plus its context seconds) and the result cached. `args` are
    visualizing_deadball's options (None = its defaults, build_parser().parse_args([])).
    """
    if not recompute:
        cached = _from_cache(start, stop, cache_dir)
        if cached is not None:
            return cached

    import cv2
    import visualizing_deadball as deadball
    from starter.load_homography import load as load_homography
    from visualizing_tracking import slot_info

    args = args or deadball.build_parser().parse_args([])
    os.makedirs(cache_dir, exist_ok=True)
    video_path, meta_path, atd_path = require(["video", "atd_meta", "atd_raw"])
    fps = GAMES[GAME]["fps"]
    pitch = np.asarray(GAMES[GAME]["pitch"], dtype=float)
    context = int(round(args.context_seconds * fps))
    first, end = max(0, start - context), min(source_frames(), stop + context)
    pre_atd = max(0, first - 30)                   # movement history, as visualizing_deadball

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("could not open {}".format(video_path))
    try:
        print("referee: homography, ATD {}..{} ...".format(pre_atd, end - 1))
        matrices, _, valid = load_homography()
        info = slot_info(meta_path)
        players_all, balls_all = deadball.load_players(atd_path, pre_atd, end, len(info))
        players = players_all[first - pre_atd:]
        rows, _ = deadball.detect_to_cache(args, cap, video_path, meta_path, atd_path, info,
                                           players, matrices, valid, first, end, fps, cache_dir)
        analysis = deadball.analyze_states(rows, fps, args, pitch)
        analysis = deadball.penalize_gestures_while_running(rows, analysis, fps, pitch, args)
        pose_rows = deadball.pose_cache_rows(args, cap, rows, cache_dir, fps)
        analysis = deadball.integrate_pose_signal(analysis, pose_rows, fps, args)
    finally:
        cap.release()

    pose = np.clip(np.nan_to_num(np.asarray(analysis["gesture_pose"], float)), 0, 1)
    bbox = np.clip(np.nan_to_num(np.asarray(analysis["gesture_bbox"], float)), 0, 1)
    gesture = 1.0 - (1.0 - args.pose_weight * pose) * (1.0 - args.bbox_weight * bbox)
    scoring = deadball.compute_scores(rows, analysis, players_all, balls_all, pre_atd, pitch, fps,
                                      args)

    frames = np.array([row["frame"] for row in rows])
    keep = (frames >= start) & (frames < stop)
    boxes = np.array([row["detections"][row["referee"]]["box"] if row.get("referee") is not None
                      else [np.nan] * 4 for row in rows], float)
    out = dict(frames=frames[keep], gesture=gesture[keep],
               dead=np.asarray(scoring["dead"], bool)[keep], box=boxes[keep],
               restart=np.asarray(scoring["restart_detected"], bool)[keep])
    np.savez_compressed(cache_path(start, stop, cache_dir), **out)
    print("referee dead ball: cached {}".format(
        os.path.relpath(cache_path(start, stop, cache_dir), ROOT)))
    return out


def dead_for_block(block_first, n, start, stop, **kwargs):
    """(n,) bool: visualizing_deadball's dead ball, aligned to a block whose index 0 is video
    frame block_first, on [start, stop) -- False elsewhere (no evidence, not "live")."""
    out = np.zeros(n, bool)
    a, b = max(start, block_first), min(stop, block_first + n)
    if a < b:
        signal = deadball_signal(a, b, **kwargs)
        out[signal["frames"] - block_first] = signal["dead"]
    return out


def _field_for_block(field, shape, block_first, n, start=None, stop=None, cache_dir=CACHE_DIR,
                     names=None, verbose=True):
    """(n, *shape) float: one cached field aligned to a block whose index 0 is video frame
    block_first, NaN where nothing was precomputed. Several cached ranges may overlap it: the one
    dead_for_block reads for [start, stop) (_from_cache's rule: the first covering it) comes
    FIRST, so the gesture, the box and the dead ball of a frame are from the same run; the others
    only fill the frames it leaves out (e.g. the two ends of a cooling break). `names`: read only
    these cache files (e.g. the chunks run.py computed), not every one in the folder."""
    out = np.full((n,) + shape, np.nan)
    if not os.path.isdir(cache_dir):
        return out
    ranges = []
    for name in sorted(os.listdir(cache_dir)):
        parts = name[:-4].split("_") if name.startswith("deadball_") and \
            name.endswith(".npz") else []
        if names is not None and name not in names:
            continue
        if len(parts) == 3 and int(parts[2]) > block_first and int(parts[1]) < block_first + n:
            ranges.append((int(parts[1]), int(parts[2]), name))
    if start is not None:
        covering = [r for r in ranges if r[0] <= start and r[1] >= stop]
        if covering:
            ranges.remove(covering[0])
            ranges.insert(0, covering[0])
    filled = np.zeros(n, bool)
    for _, _, name in ranges:
        data = np.load(os.path.join(cache_dir, name))
        if field not in data.files:
            print("referee {}: not in {} (cached before it was saved: recompute it)".format(
                field, name))
            continue
        index = data["frames"] - block_first
        keep = (index >= 0) & (index < n)
        keep &= ~filled[np.clip(index, 0, n - 1)]   # a frame already given by an earlier file
        out[index[keep]] = data[field][keep]
        filled[index[keep]] = True
        if verbose:
            print("referee {}: cache {}".format(field, name))
    return out


def gesture_for_block(block_first, n, start=None, stop=None, cache_dir=CACHE_DIR):
    """(n,) float: the referee's gesture score aligned to the block, from every cached range that
    overlaps it -- NaN where nothing was precomputed. Only reads the cache: never runs YOLO.
    [start, stop) = the range given to dead_for_block, so both read the same run there."""
    return _field_for_block("gesture", (), block_first, n, start, stop, cache_dir)


def chunks_for_block(block_first, n, names, cache_dir=CACHE_DIR):
    """{dead, restart (n,) bool, gesture (n,), box (n, 4)} aligned to a block, read from the cache
    files `names` only (run.py's chunks of a half) -- False / NaN where none covers a frame. Only
    reads the cache: never runs YOLO."""
    def field(name, shape=()):
        return _field_for_block(name, shape, block_first, n, cache_dir=cache_dir, names=names,
                                verbose=False)
    return dict(dead=field("dead") > 0.5, restart=field("restart") > 0.5,
                gesture=field("gesture"), box=field("box", (4,)))


def restart_for_block(block_first, n, start=None, stop=None, cache_dir=CACHE_DIR):
    """(n,) bool: visualizing_deadball's detected restarts aligned to the block, from the same run
    as the dead ball -- False where nothing was precomputed (or the cache is older than this
    field: recompute it). Only reads the cache: never runs YOLO."""
    return _field_for_block("restart", (), block_first, n, start, stop, cache_dir) > 0.5


def boxes_for_block(block_first, n, start=None, stop=None, cache_dir=CACHE_DIR):
    """(n, 4) float: the referee's box in the image [x1, y1, x2, y2] (pixels), from the same run
    as his gesture and the dead ball -- NaN where he was not found or nothing was precomputed.
    For drawing. Only reads the cache: never runs YOLO."""
    return _field_for_block("box", (4,), block_first, n, start, stop, cache_dir)


if __name__ == '__main__':
    # -- what to precompute -----------------------------------------------------------------
    START = 70150                   # video frame (minute * 1500 + second * 25)
    SECONDS = 200                   # how long; YOLO Detect + Pose run on every frame of it
    RECOMPUTE = True                # True: ignore the dead-ball cache (the YOLO caches stay) --
                                    # back to False once recomputed

    fps = GAMES[GAME]["fps"]
    s = deadball_signal(START, START + int(round(SECONDS * fps)), recompute=RECOMPUTE)
    print("frames {}..{}: referee gesture max {:.2f} | DEAD BALL on {} frames ({:.1f} s)".format(
        s["frames"][0], s["frames"][-1], float(s["gesture"].max()), int(s["dead"].sum()),
        s["dead"].sum() / fps))
