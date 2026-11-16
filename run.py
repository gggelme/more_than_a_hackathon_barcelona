"""End-to-end pipeline: the whole match, from the provided data to the submission and its reports.

    uv run python run.py

Everything it does is set in the `if __name__ == "__main__":` block at the bottom.

Needs: `uv sync` and the match data unpacked in data/ (gated, never in the repository). The YOLO
weights (Ultralytics' official yolo11s.pt and yolo11s-pose.pt) are downloaded into models/ on the
first run that needs them -- internet needed once.

STAGE 1 -- the referee (YOLO), optional (USE_REFEREE). The whole of each half, in chunks of
  CHUNK_SECONDS: src/referee_signal.py runs visualizing_deadball's detection on each (YOLO Detect +
  Pose on the referee, its dead-ball decision, its restarts) and caches it in data/derived/referee/
  (deadball_<a>_<b>.npz + the YOLO caches). A chunk already cached is skipped, so a run that is cut
  resumes where it stopped. Slow: hours on a GPU for the whole match. Done before the possession,
  so YOLO and a whole half of tracking are never in memory together.

STAGE 2 -- the possession, half by half, from the half's first frame:
  1. the ATD of the half, in metres;
  2. the keepers back in their own team's slot (fix_goalkeepers.py), the goal each team defends
     measured on the formation at the half's kick-off (games.py's anchor);
  3. possession_base.py: possession.py's frame-by-frame owner (the team of the last owner) plus the
     keeper catches (keeper_catch.py);
  4. goals.dead_ball: D over goals, cooling breaks, the referee's stoppages (fouls, offsides, ball
     out, VAR), the half's start before its kick-off and its end; whoever restarts gets the ball;
     spurious catches / touches of the conceding team right before a goal dropped.
  Outside the two halves (before the kick-off, half-time, after the final whistle): X.

STAGE 3 -- the outputs, in evaluation/:
  results/<OUTPUT_NAME>.csv                 the submission: frame,label for every video frame
                                            (validated with evaluation/validate_submission.py)
  reports/<name>_stoppages.csv              every stoppage: where, why it started / ended, who
                                            restarted
  reports/<name>_summary.json               counts: labels, keepers, catches, stoppages, timings
  reports/<name>.json, _transitions.csv     the score on the PUBLIC ground truth
                                            (evaluation/run_evaluation.py, RUN_EVALUATION) -- to
                                            look at, never to choose parameters (RULES.md)
  results/ and reports/ are ignored by git: frame-level derived data must not be published.

STAGE 4 -- CLEAN_REFEREE_CACHE: delete data/derived/referee/ (all the YOLO caches) at the very end.

Every threshold is the modules' own default, physical and reasoned about -- not tuned against the
public ground truth (info_hackathon/RULES.md).
"""

import csv
import gc
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
for path in (ROOT, SRC, os.path.join(SRC, "experiments")):
    if path not in sys.path:
        sys.path.insert(0, path)

from games import GAME, GAMES, anchor_video_frame, half_video_blocks, require, source_frames  # noqa: E402,E501
from starter.baseline import write_submission  # noqa: E402
from starter.load_smart_tagging import load as load_smart_tagging  # noqa: E402
from vector_tracking_clasification import read_positions  # noqa: E402
from visualizing_tracking import slot_info  # noqa: E402
from tracking_possession import goals as goals_module  # noqa: E402
from tracking_possession import possession_base  # noqa: E402
from tracking_possession.fix_cooling_break import CoolingConfig  # noqa: E402
from tracking_possession.fix_goalkeepers import (fix_goalkeepers, goalkeeper_slots,  # noqa: E402
                                                 mirrored)
from tracking_possession.goals import GoalConfig  # noqa: E402
from tracking_possession.possession import PossessionConfig  # noqa: E402
import referee_signal  # noqa: E402  (YOLO itself is only imported if a chunk must be computed)

FPS = GAMES[GAME]["fps"]
PITCH = np.array(GAMES[GAME]["pitch"], dtype=float)
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
REPORTS_DIR = os.path.join(ROOT, "evaluation", "reports")
T0 = time.time()


def log(message):
    """A progress line with the time since the start."""
    elapsed = time.time() - T0
    print("[{:02d}:{:02d}:{:02d}] {}".format(int(elapsed // 3600), int(elapsed % 3600 // 60),
                                           int(elapsed % 60), message), flush=True)


def clock(frame):
    """mm:ss.ss of a video frame, to find it in the video."""
    seconds = frame / FPS
    return "{:02d}:{:05.2f}".format(int(seconds // 60), seconds % 60)


def kickoff_frame(half):
    """Video frame of a half's kick-off: games.py's anchor (the half's video block may start
    minutes earlier, with the players walking on)."""
    return anchor_video_frame(GAMES[GAME]["kickoff" if half == 1 else "kickoff_2nd"])


# -- stage 1: the referee ------------------------------------------------------------------------

def referee_chunks(half_first, half_last, chunk_seconds):
    """[(a, b)] video frames covering the half, CHUNK_SECONDS each (the last one shorter)."""
    step = int(round(chunk_seconds * FPS))
    return [(a, min(a + step, half_last + 1)) for a in range(half_first, half_last + 1, step)]


def chunk_is_cached(a, b):
    """The chunk's cache exists and has every field the pipeline reads (older caches lack some)."""
    path = referee_signal.cache_path(a, b)
    if not os.path.isfile(path):
        return False
    with np.load(path) as data:
        return all(field in data.files for field in ("dead", "gesture", "restart", "box"))


def ensure_weights(names):
    """The YOLO weights in models/, downloaded there if missing. They are Ultralytics' OFFICIAL
    pretrained weights (YOLO11, COCO), public on Ultralytics' GitHub releases -- not ours, so they
    are not in the repository (models/ is ignored by git): the first run downloads them (internet
    needed once)."""
    from ultralytics.utils.downloads import attempt_download_asset
    folder = os.path.join(ROOT, "models")
    os.makedirs(folder, exist_ok=True)
    for name in names:
        target = os.path.join(folder, name)
        if os.path.isfile(target):
            log("weights: models/{} found".format(name))
            continue
        log("weights: models/{} missing -- downloading it from Ultralytics ...".format(name))
        got = attempt_download_asset(target)
        # Ultralytics may answer with a copy it already has in its own weights folder
        if os.path.isfile(got) and os.path.abspath(got) != os.path.abspath(target):
            shutil.copyfile(got, target)
        if not os.path.isfile(target):
            raise FileNotFoundError("could not get the YOLO weights {}: download them from "
                                    "Ultralytics into models/".format(name))
        log("weights: models/{} ready".format(name))


def run_referee(halves, chunk_seconds, device):
    """Stage 1: visualizing_deadball's detection on every chunk of every half not cached yet."""
    todo =[(half, a, b) for half, first, last in halves
            for a, b in referee_chunks(first, last, chunk_seconds) if not chunk_is_cached(a, b)]
    total = sum(len(referee_chunks(first, last, chunk_seconds)) for _, first, last in halves)
    log("STAGE 1 referee: {} chunks of {:.0f} s, {} cached, {} to compute".format(
        total, chunk_seconds, total - len(todo), len(todo)))
    if not todo:
        return
    import visualizing_deadball as deadball
    args = deadball.build_parser().parse_args(["--device", str(device)])
    # the weights visualizing_deadball reads from models/ (Detect) and models/ or here (Pose)
    ensure_weights([args.weights, args.pose_weights])
    started = time.time()
    for k, (half, a, b) in enumerate(todo, 1):
        log("referee chunk {}/{} -- half {}, frames {}..{} ({} -> {})".format(
            k, len(todo), half, a, b - 1, clock(a), clock(b)))
        # recompute: a stale cache of this exact range (an older format) is rebuilt; the YOLO
        # caches of the range are reused when they match
        signal = referee_signal.deadball_signal(a, b, args=args, recompute=True)
        done = time.time() - started
        eta = done / k * (len(todo) - k)
        log("  chunk done: referee gesture max {:.2f}, D on {:.0f} s, {} restarts | "
            "ETA stage 1: {:.0f} min".format(float(np.max(signal["gesture"])),
                                             signal["dead"].sum() / FPS,
                                             int(signal["restart"].sum()), eta / 60))
        del signal
        gc.collect()


# -- stage 2: the possession ---------------------------------------------------------------------

def classify_half(half, half_first, half_last, raw_path, info, events, configs, use_referee,
                  chunk_seconds):
    """(labels of the half, stoppages, summary dict) -- everything stage 2 does for one half."""
    possession_config, touch_config, goal_config, cooling_config = configs
    teams = [team for team, _ in info]
    keepers = goalkeeper_slots(info)
    n = half_last + 1 - half_first
    kickoff = kickoff_frame(half)
    summary = {"frames": n, "first_frame": half_first, "last_frame": half_last,
               "kickoff_frame": kickoff}

    log("half {}: reading the ATD, frames {}..{} ({} -> {}) ...".format(
        half, half_first, half_last, clock(half_first), clock(half_last)))
    players, ball = read_positions(raw_path, half_first, half_last + 1, len(info))
    players, ball = players * PITCH, ball * PITCH

    log("half {}: keepers -- goals from the formation at the kick-off ({}) ...".format(
        half, clock(kickoff)))
    players, report = fix_goalkeepers(players, keepers, PITCH[0], teams=teams,
                                      kickoff=kickoff - half_first)
    log("  goals {} (from {}), frames swapped {}, disabled {}, moved {}".format(
        report.goals, report.source, report.swapped, report.disabled, report.moved))
    summary["goalkeepers"] = dict(goals=report.goals, source=report.source,
                                  swapped=report.swapped, disabled=report.disabled,
                                  moved=report.moved)

    log("half {}: possession_base (possession.py + keeper catches) on {} frames ...".format(
        half, n))
    results, _, _, _, _, info_base = possession_base.classify(
        players, ball, teams, possession_config, FPS, touch_config, keepers=keepers,
        pitch_length=PITCH[0], pitch_width=PITCH[1], kickoff=kickoff - half_first)
    possession_base.summarise(info_base, FPS)
    summary["changes"] = len(info_base.changes)
    summary["catches"] = len(info_base.catches)
    summary["catch_stats"] = info_base.catch_stats

    referee = {}
    if use_referee:
        names = {os.path.basename(referee_signal.cache_path(a, b))
                 for a, b in referee_chunks(half_first, half_last, chunk_seconds)}
        referee = referee_signal.chunks_for_block(half_first, n, names)
        known = int(np.isfinite(referee["gesture"]).sum())
        log("half {}: referee from {} chunks -- known on {}/{} frames, D on {:.0f} s, "
            "{} restarts".format(half, len(names), known, n, referee["dead"].sum() / FPS,
                                 int(referee["restart"].sum())))
        summary["referee_known_frames"] = known

    log("half {}: dead ball (goals, cooling breaks, referee, half edges, restarts) ...".format(
        half))
    stats = {}
    stoppages, labels = goals_module.dead_ball(
        [r.team or "D" for r in results], possession_base, info_base, touch_config, ball,
        half_first, PITCH[1], goal_config, FPS, events=events, stats=stats, players=players,
        teams=teams, cooling_config=cooling_config, pitch_length=PITCH[0], half_last=half_last,
        kickoff_hint=kickoff, referee_dead=referee.get("dead"),
        referee_gesture=referee.get("gesture"), referee_restart=referee.get("restart"))
    goals_module.print_stoppages(stoppages, stats, half_first, FPS)
    summary["dead_ball"] = stats
    summary["stoppages"] = len(stoppages)
    return labels, stoppages, summary


# -- stage 3: the outputs -------------------------------------------------------------------------

def write_stoppages(path, rows):
    """One row per stoppage, in video frames and video time."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["half", "start_frame", "end_frame", "start_time", "end_time", "seconds",
                         "dead_s", "start_case", "end_case", "scorer", "restarter",
                         "restart_how"])
        for half, first, s in rows:
            a, b = s.start + first, s.end + first
            writer.writerow([half, a, b, clock(a), clock(b), round((b - a) / FPS, 2),
                             round(s.dead_s, 1), s.start_case, s.end_case, s.scorer or "",
                             s.restarter or "", s.restart_how or ""])


def run_script(script, *arguments):
    """Run one of evaluation/'s scripts with this interpreter; True if it succeeded."""
    command = [sys.executable, os.path.join(ROOT, "evaluation", script)] + list(arguments)
    log("running {} ...".format(" ".join(os.path.relpath(c, ROOT) if os.path.isabs(c) else c
                                         for c in command[1:])))
    return subprocess.run(command, cwd=ROOT).returncode == 0


# -- the pipeline ---------------------------------------------------------------------------------

def run(output_name="possession_base", possession_config=None, touch_config=None,
        goal_config=None, cooling_config=None, use_referee=True, chunk_seconds=300.0,
        referee_device="auto", run_evaluation=True, clean_referee_cache=False):
    """The whole pipeline (see the module doc). Returns the summary dict it also writes."""
    configs = (possession_config or PossessionConfig(),
               touch_config or possession_base.TouchConfig(),
               goal_config or GoalConfig(), cooling_config or CoolingConfig())
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    halves = half_video_blocks(GAME)
    n_frames = source_frames()
    log("pipeline: game {}, {} video frames, halves {}".format(GAME, n_frames, halves))
    summary = {"game": GAME, "output": output_name, "use_referee": use_referee, "halves": {}}

    # 1. the referee (YOLO), cached chunk by chunk
    if use_referee:
        run_referee(halves, chunk_seconds, referee_device)
    else:
        log("STAGE 1 referee: skipped (USE_REFEREE = False)")
    summary["stage1_s"] = round(time.time() - T0, 1)

    # 2. the possession, half by half
    log("STAGE 2 possession: possession_base + every fix")
    info = slot_info(meta_path)
    events = load_smart_tagging()
    log("Smart Tagging: {} events".format(len(events)))
    labels = np.full(n_frames, "X", dtype="<U1")         # outside the halves: X
    stoppage_rows = []
    for half, half_first, half_last in halves:
        half_labels, stoppages, half_summary = classify_half(
            half, half_first, half_last, raw_path, info, events, configs, use_referee,
            chunk_seconds)
        labels[half_first:half_last + 1] = half_labels
        stoppage_rows += [(half, half_first, s) for s in stoppages]
        summary["halves"][half] = half_summary
        gc.collect()
    goals_by_half = [summary["halves"][h]["goalkeepers"]["goals"] for h, _, _ in halves]
    if len(goals_by_half) == 2:
        ok = mirrored(*goals_by_half)
        summary["halves_mirrored"] = ok
        if ok is False:
            log("WARNING goalkeepers: the halves' goals {} are NOT mirrored -- the teams change "
                "ends at half time, one kick-off formation is wrong".format(goals_by_half))
    summary["stage2_s"] = round(time.time() - T0 - summary["stage1_s"], 1)

    # 3. the outputs
    log("STAGE 3 outputs")
    counts = {label: int((labels == label).sum()) for label in ("H", "A", "D", "X")}
    summary["labels"] = {label: {"frames": c, "share": round(c / n_frames, 4)}
                         for label, c in counts.items()}
    log("labels: " + "  ".join("{} {:.1f}%".format(k, 100 * v / n_frames)
                               for k, v in counts.items()))
    submission = write_submission(os.path.join(RESULTS_DIR, output_name + ".csv"), labels)
    log("submission: {}".format(os.path.relpath(submission, ROOT)))
    summary["valid"] = run_script("validate_submission.py", submission)
    if not summary["valid"]:
        log("WARNING: the submission did NOT validate")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    stoppages_path = os.path.join(REPORTS_DIR, output_name + "_stoppages.csv")
    write_stoppages(stoppages_path, stoppage_rows)
    log("stoppages: {} -> {}".format(len(stoppage_rows), os.path.relpath(stoppages_path, ROOT)))
    if run_evaluation:
        summary["evaluated"] = run_script("run_evaluation.py")
    summary["total_s"] = round(time.time() - T0, 1)
    summary_path = os.path.join(REPORTS_DIR, output_name + "_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    log("summary: {}".format(os.path.relpath(summary_path, ROOT)))

    # 4. clean up the referee's caches, at the very end
    if clean_referee_cache and os.path.isdir(referee_signal.CACHE_DIR):
        shutil.rmtree(referee_signal.CACHE_DIR)
        log("deleted {} (CLEAN_REFEREE_CACHE)".format(
            os.path.relpath(referee_signal.CACHE_DIR, ROOT)))
    log("done")
    return summary


if __name__ == "__main__":
    # -- what to produce ---------------------------------------------------------------------
    OUTPUT_NAME = "enlisaderos"    # evaluation/results/<name>.csv + evaluation/reports/
    RUN_EVALUATION = True                   # score every CSV in evaluation/results on the public GT

    # -- the referee (YOLO): stage 1 ---------------------------------------------------------
    USE_REFEREE = True                      # False: no YOLO, no referee stoppages
    CHUNK_SECONDS = 300                     # one cached chunk = 5 minutes of video (+ 12 s of
                                            # context on each side, YOLO'd too)
    REFEREE_DEVICE = "auto"                 # "auto", "cpu", "0" ... (YOLO)
    CLEAN_REFEREE_CACHE = True             # True: delete data/derived/referee/ at the very end
                                            # (for the delivery; the next run recomputes YOLO)

    # -- the classifier (metres, m/s), same as the visualizers ------------------------------
    POSSESSION_CONFIG = PossessionConfig(
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
    TOUCH_CONFIG = possession_base.TouchConfig(     # collisions: only for the dead ball
        min_control_frames=0,       # a rival control shorter than this does not change possession
                                    # (0 = every control counts; 12 = possession_base.csv)
    )
    GOAL_CONFIG = GoalConfig()
    COOLING_CONFIG = CoolingConfig()

    run(output_name=OUTPUT_NAME, possession_config=POSSESSION_CONFIG, touch_config=TOUCH_CONFIG,
        goal_config=GOAL_CONFIG, cooling_config=COOLING_CONFIG, use_referee=USE_REFEREE,
        chunk_seconds=CHUNK_SECONDS, referee_device=REFEREE_DEVICE,
        run_evaluation=RUN_EVALUATION, clean_referee_cache=CLEAN_REFEREE_CACHE)
