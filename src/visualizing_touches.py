"""Watch what a collision pass (src/tracking_possession/possession_touch_*.py) sees and decides, on
the tactical video, drawn as in src/visualizing_owner.py.

GENERIC: TOUCH_MODULE in the `if __name__ == '__main__':` block picks the variant, e.g.
"possession_touch_2". Any module with the same interface works: classify(players, ball, teams,
config, fps, touch_config) -> (results, ..., info with collisions / runs / changes), summarise(info,
fps), _line_fits(ball, offsets, fps) and a TouchConfig dataclass. Its TouchConfig is built with its
OWN defaults, changed only by the fields in TOUCH_OVERRIDES -- so a variant with other fields needs
no edit here. The effective config is printed. `skip` is honoured if the variant has it.

The variants use two pieces of evidence: a COLLISION (the ball's trajectory breaks) says WHEN, a
run of CONTROL (possession.py names an owner) says WHO. Display only: nothing is saved. Per frame:
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
import inspect
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))   # vector_tracking_clasification
from games import (GAME, GAMES, anchor_video_frame, half_video_blocks,  # noqa: E402
                   require, source_frames)
from starter.load_homography import load as load_homography, project  # noqa: E402
from visualizing_tracking import (COLOURS, FONT, draw_frame as draw_tracking,  # noqa: E402
                                  labels_path, read_atd, read_labels, slot_info)
from visualizing_owner import (OWNER, TIMELINE_ROW, draw_owner, pixel, play,  # noqa: E402
                               read_players, timeline)
from tracking_possession.possession import PossessionConfig  # noqa: E402
from tracking_possession import possession_base  # noqa: E402
from tracking_possession.fix_goalkeepers import goalkeeper_slots  # noqa: E402
from tracking_possession import goals as goals_module  # noqa: E402
from tracking_possession.goals import GoalConfig  # noqa: E402
from tracking_possession.fix_cooling_break import CoolingConfig, referee_signals  # noqa: E402
import referee_signal  # noqa: E402  (YOLO is only loaded if a gesture has to be computed)

# Params
FPS = GAMES[GAME]["fps"]
PITCH = np.array(GAMES[GAME]["pitch"])      # metres
COLLISION = (255, 255, 255)                 # BGR: collisions, white
REACH = (0, 165, 255)                       # the touch_reach circle, orange
BEFORE = (255, 255, 0)                      # line fitted before the break, cyan
AFTER = (255, 0, 255)                       # line fitted after the break, magenta
CHANGE = (0, 255, 255)                      # a change of possession, yellow
DIFFER = (0, 0, 255)                        # red
CATCH = (180, 105, 255)                     # a keeper catch, pink
REFEREE = (0, 255, 0)                       # the referee while he signals, green
PANEL_ALPHA = 0.5                           # how much of the video shows through the panel


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


def draw_catch(image, matrix, slots, catch):
    """Pink ring on the keeper while he holds the ball (hidden), with how it was released."""
    height, width = image.shape[:2]
    if not 0 <= catch.slot < len(slots):
        return
    point = pixel(matrix, slots[catch.slot], width, height)
    if point is None:
        return
    cv2.circle(image, point, 32, CATCH, 3, cv2.LINE_AA)
    text = "CATCH {} #{}".format(catch.team, catch.slot + 1)
    cv2.putText(image, text, (point[0] + 34, point[1] - 10), FONT, 0.6, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(image, text, (point[0] + 34, point[1] - 10), FONT, 0.6, CATCH, 2, cv2.LINE_AA)


def draw_referee(image, box, score, signalling):
    """A circle around the referee (his YOLO box, pixels): thin grey while he is seen, thick green
    with his gesture score while he SIGNALS."""
    x1, y1, x2, y2 = box
    centre = (int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2)))
    radius = int(round(0.7 * max(x2 - x1, y2 - y1)))
    if not signalling:
        cv2.circle(image, centre, radius, (200, 200, 200), 1, cv2.LINE_AA)
        return
    cv2.circle(image, centre, radius, REFEREE, 3, cv2.LINE_AA)
    text = "REFEREE SIGNAL {:.2f}".format(score)
    where = (centre[0] + radius + 6, centre[1])
    cv2.putText(image, text, where, FONT, 0.6, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(image, text, where, FONT, 0.6, REFEREE, 2, cv2.LINE_AA)


def draw_state(image, result, team_base, run, i, change, name, catch, label=None, stoppage=None,
               referee=None):
    """A compact, see-through panel at the top left: the team in possession and why. `label` is
    the final label (with goals.py's D), `stoppage` the goals.py stoppage the frame is in,
    `referee` (gesture G or NaN, visualizing_deadball's D) when the referee is used."""
    label = label or result.team or "X"
    lines = [
        ("{}: {}".format(name, label), COLOURS.get(label, (235, 235, 235))),
        ("state: {}   owner: {}   possession_base: {}".format(
            result.state, "--" if result.owner is None else "#{}".format(result.owner + 1),
            team_base or "--"), OWNER),
    ]
    if team_base != label:
        lines.append(("DIFFER", DIFFER))
    if run is not None:
        team, start, end = run
        lines.append(("control: {} for {:.1f} s (run of {:.1f} s)".format(
            team, (i - start + 1) / FPS, (end - start + 1) / FPS), COLOURS[team]))
    if catch is not None:
        lines.append(("CATCH: {} keeper #{} ({}, {})".format(
            catch.team, catch.slot + 1, getattr(catch, "case", "?"), catch.release), CATCH))
    if stoppage is not None:
        lines.append(("D (goals.py): {} -> {}  scorer {}  dead {:.0f} s".format(
            stoppage.start_case, stoppage.end_case, stoppage.scorer or "?", stoppage.dead_s),
            COLOURS["D"]))
    if change is not None:
        frame, team, reason = change
        lines.append(("last change: {} ({}) at {:.1f}s".format(team, reason, frame / FPS),
                      (235, 235, 235)))
    if referee is not None:
        g, dead = referee
        lines.append(("referee: not precomputed here" if not np.isfinite(g) else
                      "referee: G {:.2f}   visualizing_deadball: {}".format(
                          g, "D" if dead else "live"),
                      REFEREE if np.isfinite(g) and dead else (235, 235, 235)))
    scale, step, x0, y0 = 0.55, 24, 12, 12
    text_width = max(cv2.getTextSize(text, FONT, scale, 1)[0][0] for text, _ in lines)
    x1 = min(image.shape[1] - 1, x0 + text_width + 16)
    y1 = y0 + step * len(lines) + 8
    image[y0:y1, x0:x1] = (image[y0:y1, x0:x1] * (1 - PANEL_ALPHA)).astype(np.uint8)
    for k, (text, colour) in enumerate(lines):
        y = y0 + 20 + step * k
        cv2.putText(image, text, (x0 + 8, y), FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, text, (x0 + 8, y), FONT, scale, colour, 1, cv2.LINE_AA)


def events_strip(frame, start, window, width, runs, catches, collisions, changes):
    """Three rows on the timeline's page: runs of control, keeper catches, and collisions /
    changes as ticks.

    All in video frames: runs and catches [(team, start, end)], collisions [frame], changes
    [frame].
    """
    first = start + ((frame - start) // window) * window
    image = np.zeros((3 * TIMELINE_ROW, width, 3), np.uint8)

    def column(f):
        return int((f - first) * (width - 1) / max(1, window - 1))

    def band(spans, row, colour_of):
        for team, span_start, span_end in spans:
            if span_end < first or span_start >= first + window:
                continue
            x0, x1 = column(max(span_start, first)), column(min(span_end, first + window - 1))
            image[row * TIMELINE_ROW + 2:(row + 1) * TIMELINE_ROW - 2, x0:x1 + 1] = colour_of(team)

    band(runs, 0, lambda team: COLOURS[team])
    band(catches, 1, lambda team: CATCH)
    for f in collisions:
        if first <= f < first + window:
            x = column(f)
            image[2 * TIMELINE_ROW + 4:3 * TIMELINE_ROW - 4, max(0, x - 1):x + 2] = COLLISION
    for f in changes:
        if first <= f < first + window:
            x = column(f)
            image[:, max(0, x - 1):x + 2] = CHANGE
    for row, name in enumerate(("control", "catches", "collisions")):
        y = row * TIMELINE_ROW + TIMELINE_ROW - 8
        cv2.putText(image, name, (6, y), FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, name, (6, y), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    x = column(frame)
    cv2.line(image, (x, 0), (x, image.shape[0]), (0, 255, 255), 2)
    return image


def run(start, seconds=60, config=None, touch_module="possession_touch_2", touch_overrides=None,
        collision_show_seconds=0.6, arrow_seconds=0.5, scale=0.75, show_gt=True,
        timeline_seconds=20, fix_keepers=True, apply_goals=True, goal_config=None,
        use_referee=False):
    """Play video frames [start, start + seconds) with the collision pass drawn on top.

    touch_module     name of the variant in src/tracking_possession/, e.g. "possession_touch_2"
    touch_overrides  {field: value} changed on that variant's own TouchConfig defaults
    fix_keepers      put each keeper back in his own team's slot first (fix_goalkeepers.py), for
                     both classifiers and the drawing -- done here, so it works with any variant
    apply_goals      write goals.py's dead-ball stoppages (goals, cooling breaks) as D; all the
                     logic is in goals.dead_ball, this only calls it
    use_referee      add visualizing_deadball's own dead ball (fouls, offsides, ball out: the
                     referee's gesture, the ball out, its restart) on the range shown, where
                     no goal or half's edge already is -- read from src/referee_signal.py's cache,
                     or computed with YOLO the first time (slow; better precompute it)
    """
    goal_config = goal_config or GoalConfig()
    config = config or PossessionConfig()
    touch = importlib.import_module("tracking_possession." + touch_module)
    touch_config = touch.TouchConfig(**(touch_overrides or {}))
    print("touch module: {}\n  {}".format(touch_module, touch_config))
    video_path, meta_path, raw_path = require(["video", "atd_meta", "atd_raw"])
    # START and SECONDS decide what is shown, across halves if they say so: EVERY half the range
    # touches is classified on its own, from its first frame (half-time has no possession)
    stop = min(start + int(round(seconds * FPS)), source_frames())

    print("reading homography ...")
    matrices, _, valid = load_homography()
    print("reading ATD frames {}..{} ...".format(start, stop - 1))
    atd = read_atd(raw_path, start, stop)
    info = slot_info(meta_path)
    teams = [team for team, _ in info]
    # A variant that takes the keeper slots (e.g. to detect keeper catches) gets them; the keepers
    # are already fixed here, so fixing them again inside it changes nothing
    parameters = inspect.signature(touch.classify).parameters
    extra = {}
    if fix_keepers and "keepers" in parameters:
        extra = dict(keepers=goalkeeper_slots(info), pitch_length=PITCH[0])
        if "pitch_width" in parameters:
            extra["pitch_width"] = PITCH[1]
    h, g = touch_config.half_window, getattr(touch_config, "skip", 0)

    def classify_half(half, half_first, half_last):
        """Everything one half needs to be drawn, classified from its first frame."""
        print("\n== half {} ({}..{}) ==".format(half, half_first, half_last))
        # games.py's kick-off anchor of this half (goals.py only searches from it)
        kickoff_hint = anchor_video_frame(GAMES[GAME]["kickoff" if half == 1 else "kickoff_2nd"])
        classified_stop = min(stop, half_last + 1)
        # with the goals, read on past the range: a stoppage whose restart is after it is still
        # found (with no time limit on a stoppage, to the end of the half -- arrays only, light)
        read_stop = classified_stop
        if apply_goals:
            read_stop = (half_last + 1 if goal_config.max_episode_s is None else
                         min(half_last + 1, classified_stop +
                             int(round(goal_config.max_episode_s * FPS))))
        print("reading ATD frames {}..{} (from the half's start, to classify) ...".format(
            half_first, read_stop - 1))
        players, ball = read_players(raw_path, half_first, read_stop, info, fix_keepers, atd)

        print("classifying with {}.py and possession_base.py ...".format(touch_module))
        # the goal each team defends: from the formation at the half's KICK-OFF, not its first
        # frames (the second half's block starts 2 minutes before the kick-off)
        kw = dict(extra, kickoff=kickoff_hint - half_first) if "kickoff" in parameters else extra
        results, _, _, _, _, touch_info = touch.classify(players, ball, teams, config, FPS,
                                                         touch_config, **kw)
        touch.summarise(touch_info, FPS)
        # The base possession (possession.py, no collisions) with the very same fixes -- keepers,
        # catches and, below, the same dead ball -- to compare the two methods on equal terms.
        # Same collision thresholds as the touch module when it shares its TouchConfig, so both
        # get the same stoppages
        base_kw = (dict(keepers=goalkeeper_slots(info), pitch_length=PITCH[0],
                        pitch_width=PITCH[1], kickoff=kickoff_hint - half_first)
                   if fix_keepers else {})
        base_config = (touch_config if isinstance(touch_config, possession_base.TouchConfig)
                       else possession_base.TouchConfig())
        base, _, _, _, _, base_info = possession_base.classify(players, ball, teams, config, FPS,
                                                               base_config, **base_kw)
        possession_base.summarise(base_info, FPS)

        # The referee: visualizing_deadball's own decision FIRST (it computes and caches the range
        # if no cache covers it), then his gesture -- read from the cache it has just written --
        # and his SIGNALS (the cooling break's own rule)
        gesture = signal = referee_dead = boxes = referee_restart = None
        if use_referee:
            a, b = max(start, half_first), min(stop, half_last + 1)
            referee_dead = referee_signal.dead_for_block(half_first, len(ball), a, b)
            referee_restart = referee_signal.restart_for_block(half_first, len(ball), a, b)
            gesture = referee_signal.gesture_for_block(half_first, len(ball), a, b)
            boxes = referee_signal.boxes_for_block(half_first, len(ball), a, b)
            signal = referee_signals(gesture, CoolingConfig())
            g_range = gesture[a - half_first:b - half_first]
            known = np.isfinite(g_range)
            print("referee: gesture known on {}/{} frames of the range, max G {:.2f}, "
                  "signal on {} frames | visualizing_deadball D on {} frames".format(
                      int(known.sum()), len(g_range),
                      float(np.nanmax(g_range)) if known.any() else float("nan"),
                      int(signal[a - half_first:b - half_first].sum()),
                      int(referee_dead[a - half_first:b - half_first].sum())))

        # The dead-ball stoppages: one call, all the logic in goals.py
        final, stoppages = [r.team or "D" for r in results], []
        base_final = [r.team or "D" for r in base]
        if apply_goals:
            goal_stats = {}
            stoppages, final = goals_module.dead_ball(
                final, touch, touch_info, touch_config, ball, half_first, PITCH[1], goal_config,
                FPS, stats=goal_stats, players=players, teams=teams, half_last=half_last,
                kickoff_hint=kickoff_hint, referee_dead=referee_dead, referee_gesture=gesture,
                referee_restart=referee_restart)
            print()
            goals_module.print_stoppages(stoppages, goal_stats, half_first, FPS)
            # the same call for the base (its stoppages are the same; only who restarts and the
            # dropped catches / controls act on its own runs)
            base_stats = {}
            _, base_final = goals_module.dead_ball(
                base_final, possession_base, base_info, base_config, ball, half_first, PITCH[1],
                goal_config, FPS, stats=base_stats, players=players, teams=teams,
                half_last=half_last, kickoff_hint=kickoff_hint, referee_dead=referee_dead,
                referee_gesture=gesture, referee_restart=referee_restart)
            print("base possession, dead ball: {}".format(
                {k: v for k, v in base_stats.items() if "restart" in k or "dropped" in k}))

        # The lines the collision pass fits on each side of every frame (the raw ball, as it
        # does, leaving `skip` frames out on each side if the variant has that field)
        v_before, p_before, _ = touch._line_fits(ball, list(range(-g - h, -g + 1)), FPS)
        v_after, p_after, _ = touch._line_fits(ball, list(range(g, g + h + 1)), FPS)

        offset, end = max(0, start - half_first), classified_stop - half_first
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
            print("  {:02d}:{:05.2f}  -> {}  ({})".format(int(seconds_c // 60), seconds_c % 60,
                                                        team, reason))
        catches = list(getattr(touch_info, "catches", []))
        print("keeper catches in the range: {}".format(
            sum(offset <= c.start < end for c in catches)))
        return SimpleNamespace(
            first=half_first, last=half_last, results=results, base_final=base_final,
            final=final,
            touch_info=touch_info, catches=catches, stoppages=stoppages,
            gesture=gesture, signal=signal, referee_dead=referee_dead, boxes=boxes,
            fits=(v_before, p_before, v_after, p_after), offset=offset, end=end,
            shown=shown, in_range=in_range,
            run_starts=[s for _, s, _ in touch_info.runs],
            change_frames=[c[0] for c in touch_info.changes])

    halves = [classify_half(half, hf, hl) for half, hf, hl in half_video_blocks(GAME)
              if hf < stop and hl >= start]
    if use_referee:
        print("referee box known on {} frames of the range".format(sum(
            int(np.isfinite(hd.boxes[hd.offset:hd.end, 0]).sum()) for hd in halves)))

    # a catch is drawn from `since` -- when the ball became the keeper's team's -- if the variant
    # gives it, else from the start of the gap
    def since(c):
        return c.start if getattr(c, "since", None) is None else c.since

    # The timeline and the events strip, in video frames, all halves together
    final_row, base_row = {}, {}
    runs_video, catches_video, collision_marks, change_marks = [], [], [], []
    for hd in halves:
        for i in range(hd.offset, hd.end):
            final_row[hd.first + i] = hd.final[i]
            base_row[hd.first + i] = hd.base_final[i]
        runs_video += [(team, s + hd.first, e + hd.first) for team, s, e in hd.touch_info.runs]
        catches_video += [(c.team, since(c) + hd.first, c.end + hd.first) for c in hd.catches]
        collision_marks += [c.frame + hd.first for c in hd.shown]
        change_marks += [c[0] + hd.first for c in hd.in_range]
    rows = [(touch_module + (" + D" if apply_goals else ""), final_row),
            ("possession_base" + (" + D" if apply_goals else ""), base_row)]
    if show_gt:
        rows.append(("GT", read_labels(labels_path("gt"), start, stop)))
    window = int(round(timeline_seconds * FPS))
    show_frames = int(round(collision_show_seconds * FPS))

    def bottom_of(frame, width):
        return np.vstack([timeline(rows, frame, start, stop, window, width),
                          events_strip(frame, start, window, width, runs_video, catches_video,
                                       collision_marks, change_marks)])

    def compose(image, frame):
        """(window image, height of the video part) for one frame."""
        hd = next((hd for hd in halves if hd.first <= frame <= hd.last and
                   frame - hd.first < len(hd.results)), None)
        if hd is None:                              # half-time / outside the match: video + ATD
            if valid[frame] and frame in atd:
                slots, ball_atd = atd[frame]
                draw_tracking(image, matrices[frame], slots, ball_atd, info, frame, FPS)
            text = "outside the halves (frame {}): no possession".format(frame)
            cv2.putText(image, text, (20, 40), FONT, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image, text, (20, 40), FONT, 0.8, (235, 235, 235), 2, cv2.LINE_AA)
            top = cv2.resize(image, None, fx=scale, fy=scale)
            return np.vstack([top, bottom_of(frame, top.shape[1])]), top.shape[0]
        i = frame - hd.first
        result = hd.results[i]
        if valid[frame] and frame in atd:
            matrix = matrices[frame]
            slots, ball_atd = atd[frame]
            draw_tracking(image, matrix, slots, ball_atd, info, frame, FPS)
            for collision in hd.touch_info.collisions:
                if 0 <= i - collision.frame < show_frames:
                    draw_collision(image, matrix, collision, hd.fits, touch_config.touch_reach,
                                   arrow_seconds)
            draw_owner(image, matrix, slots, result.owner)
        if hd.boxes is not None and np.isfinite(hd.boxes[i]).all():
            draw_referee(image, hd.boxes[i], hd.gesture[i], bool(hd.signal[i]))
        catch = next((c for c in hd.catches if since(c) <= i <= c.end), None)
        if catch is not None and valid[frame] and frame in atd:
            draw_catch(image, matrices[frame], atd[frame][0], catch)
        runs = hd.touch_info.runs
        k = int(np.searchsorted(hd.run_starts, i, side="right")) - 1
        current_run = runs[k] if k >= 0 and runs[k][2] >= i else None
        k = int(np.searchsorted(hd.change_frames, i, side="right")) - 1
        change = None
        if k >= 0:
            change_frame, team, reason = hd.touch_info.changes[k]
            change = (change_frame + hd.first, team, reason)
        stoppage = next((s for s in hd.stoppages if s.start <= i < s.end), None)
        referee = (None if hd.gesture is None else
                   (float(hd.gesture[i]), bool(hd.referee_dead[i])))
        draw_state(image, result, hd.base_final[i], current_run, i, change, touch_module, catch,
                   hd.final[i], stoppage, referee)
        top = cv2.resize(image, None, fx=scale, fy=scale)
        return np.vstack([top, bottom_of(frame, top.shape[1])]), top.shape[0]

    play(video_path, compose, start, stop, window, "collisions and control: " + touch_module)


if __name__ == '__main__':
    first_half_kickoff = half_video_blocks(GAME)[0][1]     # 11185, first graded frame

    # -- what to show ------------------------------------------------------------------------
    #second_start, second_end = half_video_blocks(GAME)[1][1:]
    START = 67200    # video frame to start from (minute * 1500 + second * 25)
    #SECONDS = (second_end + 1 - second_start) / FPS
    SECONDS = 50                              # how long to play
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
    TOUCH_MODULE = "possession_touch_4"     # any src/tracking_possession/possession_touch_*.py
    TOUCH_OVERRIDES = {                     # fields changed on that module's TouchConfig defaults;
        # "dv_min": 4.0,                    # empty = the module's own defaults (printed at start)
        # "skip": 3,
    }

    FIX_GOALKEEPERS = True                  # keepers back in their own team's slot first
    APPLY_GOALS = True                      # goals.py: D over goals and long stoppages
    USE_REFEREE = False                     # + the referee's stoppages (fouls, offsides) on the
                                            # range: run src/referee_signal.py first (YOLO)

    run(start=START, seconds=SECONDS, config=CONFIG, touch_module=TOUCH_MODULE,
        touch_overrides=TOUCH_OVERRIDES, collision_show_seconds=COLLISION_SHOW_SECONDS,
        arrow_seconds=ARROW_SECONDS, scale=SCALE, show_gt=SHOW_GT,
        timeline_seconds=TIMELINE_SECONDS, fix_keepers=FIX_GOALKEEPERS,
        apply_goals=APPLY_GOALS, use_referee=USE_REFEREE)
