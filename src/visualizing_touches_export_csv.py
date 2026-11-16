"""Watch what a collision pass (src/tracking_possession/possession_touch_*.py) sees and decides, on
the tactical video, drawn as in src/visualizing_owner.py.

GENERIC: TOUCH_MODULE in the `if __name__ == '__main__':` block picks the variant, e.g.
"possession_touch_2". Any module with the same interface works: classify(players, ball, teams,
config, fps, touch_config) -> (results, ..., info with collisions / runs / changes), summarise(info,
fps), _line_fits(ball, offsets, fps) and a TouchConfig dataclass. Its TouchConfig is built with its
OWN defaults, changed only by the fields in TOUCH_OVERRIDES -- so a variant with other fields needs
no edit here. The effective config is printed. `skip` is honoured if the variant has it.

The variants use two pieces of evidence: a COLLISION (the ball's trajectory breaks) says WHEN, a
run of CONTROL (possession.py names an owner) says WHO. Exports a new full-match CSV before the viewer opens. Per frame:
  - the ATD as in src/visualizing_tracking.py: every tracked slot as a dot with its number, the
    ball, and the bottom strip; the classifier's OWNER with a green ring;
  - every COLLISION, for COLLISION_SHOW_S after it: the two straight lines the module fits to the
    ball before and after the break -- leaving SKIP frames out on each side, as it does -- as
    arrows from the break point (BEFORE cyan, AFTER magenta, length = ARROW_SECONDS of travel),
    the TOUCH_REACH circle around it (a collision needs some player inside), and its velocity
    change dv and nearest player's distance;
  - top-left, the team in possession with the touch module (blue H, red A), and under it the
    state and owner, the team with possession.py alone -- DIFFER when they disagree --, the run of
    control the frame is in (team and length so far), and the last change with its reason;
  - below the video, the possession timeline of both modules (and with SHOW_GT the ground truth),
    and an EVENTS strip on the same page: the runs of control (in the team's colour), the
    collisions (white ticks) and the changes the touch module made (tall yellow ticks).

Both are run from the start of the half that holds START, so their memory is as on the whole
match. The range must lie inside one half. The collisions and changes of the range are printed.

    python src/visualizing_touches.py

Moving through the video, as in src/visualizing_owner.py:
  - click or drag on the timeline to jump to that moment of the page;
  - , and .  one frame back / forward;   a and d  one second;   [ and ]  one page;
  - space or p (or a click on the video) pauses / plays, q or Esc quits.

What to show and every weight are set at the bottom of this file, in the `if __name__ == '__main__':`
block -- keep CONFIG and TOUCH_OVERRIDES equal to the ones you evaluate.
"""

import importlib
import os
import sys
from collections import Counter

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))   # vector_tracking_clasification
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from starter.load_homography import load as load_homography, project  # noqa: E402
from visualizing_tracking import (COLOURS, FONT, draw_frame as draw_tracking,  # noqa: E402
                                  draw_label_box, labels_path, read_atd, read_labels, slot_info)
from visualizing_owner import (OWNER, TIMELINE_ROW, draw_owner, play, read_players,  # noqa: E402
                               timeline)
from tracking_possession.possession import PossessionConfig, classify  # noqa: E402
from starter.baseline import write_submission  # noqa: E402
from vector_tracking_clasification import read_positions  # noqa: E402
from tracking_possession.fix_goalkeepers import goalkeeper_slots  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]
PITCH = np.array(GAMES[GAME]["pitch"])      # metres
COLLISION = (255, 255, 255)                 # BGR: collisions, white
REACH = (0, 165, 255)                       # the touch_reach circle, orange
BEFORE = (255, 255, 0)                      # line fitted before the break, cyan
AFTER = (255, 0, 255)                       # line fitted after the break, magenta
CHANGE = (0, 255, 255)                      # a change of possession, yellow
DIFFER = (0, 0, 255)                        # red


# Archivo que lee el comparador --vector vector_tracking_touch_2.
TOUCH_CSV_NAME = "vector_tracking_touch_2.csv"


def export_touch_predictions(touch_module, config, touch_overrides=None,
                             fix_keepers=True, output_name=TOUCH_CSV_NAME):
    """Genera las predicciones renovadas para AMBAS mitades del partido.

    El CSV tiene una fila por frame, como los archivos de submission; los
    frames fuera de las mitades son X. No lee ground truth ni detector D.
    """
    touch = importlib.import_module("tracking_possession." + touch_module)
    touch_cfg = touch.TouchConfig(**(touch_overrides or {}))
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    info = slot_info(meta_path)
    teams = [team for team, _ in info]
    keepers = goalkeeper_slots(info) if fix_keepers else None
    count = int(source_frames())
    labels = np.full(count, "X", dtype="<U1")

    for half, first, last_inclusive in half_video_blocks(GAME):
        stop = min(count, last_inclusive + 1)
        if first >= stop:
            continue
        print(f"Mitad {half}: procesando frames {first}..{stop - 1}", flush=True)
        players, ball = read_positions(raw_path, first, stop, len(info))
        # El nuevo clasificador permite reparar slots de arqueros antes del
        # analisis, sin volver a usar las posiciones normalizadas del ATD.
        result = touch.classify(
            players * PITCH, ball * PITCH, teams, config, FPS, touch_cfg,
            keepers=keepers, pitch_length=float(PITCH[0]) if keepers else None
        )
        decisions = result[0]
        if len(decisions) != stop - first:
            raise RuntimeError("Numero incorrecto de resultados de posesion")
        for i, decision in enumerate(decisions):
            label = decision.team or "D"
            if label not in ("H", "A", "D"):
                raise ValueError(f"Etiqueta invalida {label!r} en frame {first + i}")
            labels[first + i] = label
        touch.summarise(result[-1], FPS)

    out_dir = os.path.join(ROOT, "evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)
    path = write_submission(os.path.join(out_dir, output_name), labels)
    print(f"CSV de posesion renovada: {path}")
    print(f"Filas: {len(labels)}  Distribucion: {dict(Counter(labels))}")
    print("Para comparar con detector D:")
    print(f"  python test/visualizing_compare_detector_d.py --vector {os.path.splitext(output_name)[0]}")
    return path


def to_pixel(matrix, xy_m):
    """Pixel (int u, int v) of a pitch point in metres, or None."""
    if not np.all(np.isfinite(xy_m)):
        return None
    u, v = project(matrix, xy_m[0] / PITCH[0], xy_m[1] / PITCH[1])
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    return int(round(float(u))), int(round(float(v)))


def draw_circle_m(image, matrix, centre_m, radius, colour, thickness):
    """A circle of `radius` metres around a pitch point, drawn on the grass."""
    angles = np.linspace(0, 2 * np.pi, 48)
    points = [to_pixel(matrix, centre_m + radius * np.array([np.cos(a), np.sin(a)]))
              for a in angles]
    if all(p is not None for p in points):
        cv2.polylines(image, [np.array(points, np.int32)], True, colour, thickness, cv2.LINE_AA)


def draw_collision(image, matrix, collision, fits, reach, arrow_seconds):
    """The two fitted lines from the break point, the reach circle and the collision's numbers."""
    v_before, p_before, v_after, p_after = (array[collision.frame] for array in fits)
    where = (p_before + p_after) / 2.0
    start = to_pixel(matrix, where)
    if start is None:
        return
    draw_circle_m(image, matrix, where, reach, REACH, 2)
    for velocity, colour in ((v_before, BEFORE), (v_after, AFTER)):
        end = to_pixel(matrix, where + velocity * arrow_seconds)
        if end is not None:
            cv2.arrowedLine(image, start, end, colour, 3, cv2.LINE_AA, tipLength=0.15)
    cv2.circle(image, start, 6, COLLISION, -1, cv2.LINE_AA)
    text = "COLLISION dv {:.1f} m/s  nearest {:.1f} m".format(collision.dv, collision.distance)
    cv2.putText(image, text, (start[0] + 14, start[1] - 14), FONT, 0.6, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(image, text, (start[0] + 14, start[1] - 14), FONT, 0.6, COLLISION, 2, cv2.LINE_AA)


def draw_state(image, result, team_base, run, i, change, name):
    """Lines under the team box, in the style of visualizing_owner.draw_state. `name` is the
    touch module's."""
    lines = [
        ("state: {}   owner: {}".format(
            result.state, "--" if result.owner is None else "#{}".format(result.owner + 1)),
         OWNER),
        ("{}: {}   possession: {}".format(name, result.team or "--", team_base or "--"), OWNER),
    ]
    if team_base != result.team:
        lines.append(("DIFFER", DIFFER))
    if run is not None:
        team, start, end = run
        lines.append(("control: {} for {:.1f} s (run of {:.1f} s)".format(
            team, (i - start + 1) / FPS, (end - start + 1) / FPS), COLOURS[team]))
    if change is not None:
        frame, team, reason = change
        lines.append(("last change: {} ({}) at {:.1f}s".format(team, reason, frame / FPS),
                      (235, 235, 235)))
    width = image.shape[1]
    bottom = 90 + 34 * len(lines) + 8
    cv2.rectangle(image, (15, 90), (min(width - 15, 900), bottom), (0, 0, 0), -1)
    for k, (text, colour) in enumerate(lines):
        cv2.putText(image, text, (25, 116 + 34 * k), FONT, 0.7, colour, 2, cv2.LINE_AA)


def events_strip(frame, start, window, width, runs, collisions, changes):
    """Two rows on the timeline's page: runs of control, and collisions / changes as ticks.

    All in video frames: runs [(team, start, end)], collisions [frame], changes [frame].
    """
    first = start + ((frame - start) // window) * window
    image = np.zeros((2 * TIMELINE_ROW, width, 3), np.uint8)

    def column(f):
        return int((f - first) * (width - 1) / max(1, window - 1))

    for team, run_start, run_end in runs:
        if run_end < first or run_start >= first + window:
            continue
        x0, x1 = column(max(run_start, first)), column(min(run_end, first + window - 1))
        image[2:TIMELINE_ROW - 2, x0:x1 + 1] = COLOURS[team]
    for f in collisions:
        if first <= f < first + window:
            x = column(f)
            image[TIMELINE_ROW + 4:2 * TIMELINE_ROW - 4, max(0, x - 1):x + 2] = COLLISION
    for f in changes:
        if first <= f < first + window:
            x = column(f)
            image[:, max(0, x - 1):x + 2] = CHANGE
    for row, name in enumerate(("control", "collisions")):
        y = row * TIMELINE_ROW + TIMELINE_ROW - 8
        cv2.putText(image, name, (6, y), FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, name, (6, y), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    x = column(frame)
    cv2.line(image, (x, 0), (x, image.shape[0]), (0, 255, 255), 2)
    return image


def run(start, seconds=60, config=None, touch_module="possession_touch_2", touch_overrides=None,
        collision_show_seconds=0.6, arrow_seconds=0.5, scale=0.75, show_gt=True,
        timeline_seconds=20, fix_keepers=True):
    """Play video frames [start, start + seconds) with the collision pass drawn on top.

    touch_module     name of the variant in src/tracking_possession/, e.g. "possession_touch_2"
    touch_overrides  {field: value} changed on that variant's own TouchConfig defaults
    fix_keepers      put each keeper back in his own team's slot first (fix_goalkeepers.py), for
                     both classifiers and the drawing -- done here, so it works with any variant
    """
    config = config or PossessionConfig()
    touch = importlib.import_module("tracking_possession." + touch_module)
    touch_config = touch.TouchConfig(**(touch_overrides or {}))
    print("touch module: {}\n  {}".format(touch_module, touch_config))
    video_path, meta_path, raw_path = require(["video", "atd_meta", "atd_raw"])
    halves = [(hf, hl) for _, hf, hl in half_video_blocks(GAME) if hf <= start <= hl]
    if not halves:
        raise ValueError("START must be inside a half: {}".format(half_video_blocks(GAME)))
    half_first, half_last = halves[0]
    stop = min(start + int(round(seconds * FPS)), half_last + 1)

    print("reading homography ...")
    matrices, _, valid = load_homography()
    print("reading ATD frames {}..{} ...".format(start, stop - 1))
    atd = read_atd(raw_path, start, stop)
    info = slot_info(meta_path)
    teams = [team for team, _ in info]
    print("reading ATD frames {}..{} (from the half's start, to classify) ...".format(
        half_first, stop - 1))
    players, ball = read_players(raw_path, half_first, stop, info, fix_keepers, atd)

    print("classifying with possession.py and {}.py ...".format(touch_module))
    base, *_ = classify(players, ball, teams, config, FPS)
    results, _, _, _, _, touch_info = touch.classify(players, ball, teams, config, FPS,
                                                     touch_config)
    touch.summarise(touch_info, FPS)

    # The lines the collision pass fits on each side of every frame (the raw ball, as it does,
    # leaving `skip` frames out on each side if the variant has that field)
    h, g = touch_config.half_window, getattr(touch_config, "skip", 0)
    v_before, p_before, _ = touch._line_fits(ball, list(range(-g - h, -g + 1)), FPS)
    v_after, p_after, _ = touch._line_fits(ball, list(range(g, g + h + 1)), FPS)
    fits = (v_before, p_before, v_after, p_after)

    offset, end = start - half_first, stop - half_first
    shown = [c for c in touch_info.collisions if offset <= c.frame < end]
    in_range = [c for c in touch_info.changes if offset <= c[0] < end]
    print("\ncollisions in the range: {}".format(len(shown)))
    for c in shown:
        seconds_c = (c.frame + half_first) / FPS
        print("  {:02d}:{:05.2f}  dv {:5.1f} m/s  nearest player {:.1f} m".format(
            int(seconds_c // 60), seconds_c % 60, c.dv, c.distance))
    print("changes of possession in the range: {}".format(len(in_range)))
    for frame_c, team, reason in in_range:
        seconds_c = (frame_c + half_first) / FPS
        print("  {:02d}:{:05.2f}  -> {}  ({})".format(int(seconds_c // 60), seconds_c % 60, team,
                                                    reason))

    rows = [(touch_module, {half_first + i: results[i].team or "D" for i in range(offset, end)}),
            ("possession", {half_first + i: base[i].team or "D" for i in range(offset, end)})]
    if show_gt:
        rows.append(("GT", read_labels(labels_path("gt"), start, stop)))
    runs_video = [(team, s + half_first, e + half_first) for team, s, e in touch_info.runs]
    collision_marks = [c.frame + half_first for c in shown]
    change_marks = [c[0] + half_first for c in in_range]
    change_frames = [c[0] for c in touch_info.changes]
    run_starts = [s for _, s, _ in touch_info.runs]
    window = int(round(timeline_seconds * FPS))
    show_frames = int(round(collision_show_seconds * FPS))

    def compose(image, frame):
        """(window image, height of the video part) for one frame."""
        i = frame - half_first
        result = results[i]
        if valid[frame] and frame in atd:
            matrix = matrices[frame]
            slots, ball_atd = atd[frame]
            draw_tracking(image, matrix, slots, ball_atd, info, frame, FPS)
            for collision in touch_info.collisions:
                if 0 <= i - collision.frame < show_frames:
                    draw_collision(image, matrix, collision, fits, touch_config.touch_reach,
                                   arrow_seconds)
            draw_owner(image, matrix, slots, result.owner)
        draw_label_box(image, "touch", result.team or "X")
        k = int(np.searchsorted(run_starts, i, side="right")) - 1
        current_run = touch_info.runs[k] if k >= 0 and touch_info.runs[k][2] >= i else None
        k = int(np.searchsorted(change_frames, i, side="right")) - 1
        change = None
        if k >= 0:
            change_frame, team, reason = touch_info.changes[k]
            change = (change_frame + half_first, team, reason)
        draw_state(image, result, base[i].team, current_run, i, change, touch_module)
        top = cv2.resize(image, None, fx=scale, fy=scale)
        width = top.shape[1]
        bottom = np.vstack([timeline(rows, frame, start, stop, window, width),
                            events_strip(frame, start, window, width, runs_video,
                                         collision_marks, change_marks)])
        return np.vstack([top, bottom]), top.shape[0]

    play(video_path, compose, start, stop, window, "collisions and control: " + touch_module)


if __name__ == '__main__':
    first_half_kickoff = half_video_blocks(GAME)[0][1]     # 11185, first graded frame

    # -- what to show ------------------------------------------------------------------------
    START = first_half_kickoff      # video frame to start from (minute * 1500 + second * 25)
    SECONDS = 3000                   # how long to play
    SCALE = 0.75                    # window size relative to 1920x1080
    COLLISION_SHOW_SECONDS = 0.6    # how long a collision stays drawn
    ARROW_SECONDS = 0.5             # drawing only: arrow = where the fitted line goes in 0.5 s
    SHOW_GT = True                  # public ground truth as a timeline row
    TIMELINE_SECONDS = 20           # length of one timeline page

    # -- the classifier (metres, m/s), same as src/experiments/vector_tracking_clasification.py
    CONFIG = PossessionConfig(
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
    # -- the collision pass ------------------------------------------------------------------
    TOUCH_MODULE = "possession_touch_2"     # any src/tracking_possession/possession_touch_*.py
    TOUCH_OVERRIDES = {                     # fields changed on that module's TouchConfig defaults;
        # "dv_min": 4.0,                    # empty = the module's own defaults (printed at start)
        # "skip": 3,
    }

    FIX_GOALKEEPERS = True                 # correccion de slots de arqueros
    EXPORT_CSV = True                      # guarda TODAS las predicciones
    EXPORT_ONLY = False                    # True = solo CSV, sin reproduccion
    OUTPUT_CSV = TOUCH_CSV_NAME            # deja intacto el CSV original

    if EXPORT_CSV:
        export_touch_predictions(TOUCH_MODULE, CONFIG, TOUCH_OVERRIDES,
                                 fix_keepers=FIX_GOALKEEPERS,
                                 output_name=OUTPUT_CSV)
    if not EXPORT_ONLY:
        run(start=START, seconds=SECONDS, config=CONFIG, touch_module=TOUCH_MODULE,
            touch_overrides=TOUCH_OVERRIDES, collision_show_seconds=COLLISION_SHOW_SECONDS,
            arrow_seconds=ARROW_SECONDS, scale=SCALE, show_gt=SHOW_GT,
            timeline_seconds=TIMELINE_SECONDS, fix_keepers=FIX_GOALKEEPERS)
