"""The BASE possession (possession.py, frame by frame, no collisions) with every fix the touch pass
has -- to compare the two methods on equal terms. possession.py itself is left untouched.

possession.py says: the team in possession is the team of the LAST OWNER (a frame in state
"possession"); flights, free balls and missing balls inherit it. That rule is kept as it is. The
fixes only ADD owners or write D, exactly as for possession_touch_4:

  keepers     each keeper back in his own team's slot first (fix_goalkeepers.py), the goals each
              team defends measured on the formation at the half's kick-off (`kickoff`);
  catches     a keeper catch (keeper_catch.py) makes the keeper's team the owner from `since` --
              how the ball vanished decides it -- to the end of the gap: the tracking loses the
              ball in his hands and possession.py never names him;
  dead ball   goals.dead_ball, unchanged, with this module as its `touch` module: goals, cooling
              breaks, the referee, the half's edges as D; whoever RESTARTS becomes the owner at the
              restart; a catch inside a stoppage, or a catch / short control by the conceding team
              right before a goal, is dropped (its frames go back to inheriting).

So with no catch and no stoppage the labels are possession.py's, frame for frame.

Collisions are still detected -- ONLY because goals.dead_ball needs them (restarts are the first
collision, a shot is a kick at the goal mouth), so both methods get the very same stoppages. They
never decide the possession here.

Same interface as possession_touch_4 (TouchConfig, classify, resolve, summarise, _line_fits), so
visualizing_touches and goals.dead_ball take it as any touch module: TOUCH_MODULE =
"possession_base". Thresholds of the collisions are possession_touch_4's TouchConfig; only the
fields that find collisions (touch_reach, half_window, skip, min_points, dv_min) and control_gap
matter here.
"""

from dataclasses import replace

import numpy as np

from tracking_possession.possession import PossessionConfig, classify as classify_base
from tracking_possession.fix_goalkeepers import fix_goalkeepers
from tracking_possession.keeper_catch import detect_catches
from tracking_possession.possession_touch_4 import (TouchConfig, TouchInfo,  # noqa: F401
                                                    _line_fits, control_runs, detect_collisions)

# anchors painted later win where they overlap: a catch over the shooter's last owner frames, a
# restart over anything
_PRIORITY = {"control": 0, "catch": 1, "restart": 2}
_REASON = {"control": "owner (possession.py)", "catch": "keeper catch",
           "restart": "restart after a dead ball"}


def resolve(runs, collision_frames, n, cfg=None, fps=25):
    """Team in possession per frame: the team of the LAST anchor, possession.py's rule.

    runs   (team, start, end) or (team, start, end, kind): every frame of a run is an anchor of its
           team; kind "control" (possession.py's owner), "catch" or "restart" -- later kinds win
           where runs overlap. Between anchors the last team is inherited; before the first, None.
    collision_frames, cfg, fps are not used: the same signature as possession_touch_4.resolve, so
           goals.dead_ball can call either.
    Returns (team per frame as a list of "H" / "A" / None, changes, rejected = 0).
    """
    runs = sorted(((tuple(r) + ("control",))[:4] for r in runs),
                  key=lambda r: (_PRIORITY.get(r[3], 0), r[1]))
    anchor = [None] * n
    kind_of = [None] * n
    for team, start, end, kind in runs:
        for t in range(max(0, start), min(n, end + 1)):
            anchor[t], kind_of[t] = team, kind
    labels, changes, team = [None] * n, [], None
    for t in range(n):
        if anchor[t] is not None and anchor[t] != team:
            changes.append((t, anchor[t], "first owner" if team is None else _REASON.get(
                kind_of[t], kind_of[t])))
            team = anchor[t]
        labels[t] = team
    return labels, changes, 0


def classify(players, ball, teams, config=None, fps=25, touch_config=None, keepers=None,
             pitch_length=None, goalkeeper_config=None, catch_config=None, pitch_width=68.0,
             kickoff=None):
    """possession.py's possession with the keepers fixed and the keeper catches added; the same
    arguments and returns as possession_touch_4.classify (see it). `results` are possession.py's
    FrameResults with only `team` replaced (state / owner are possession.py's)."""
    config = config or PossessionConfig()
    cfg = touch_config or TouchConfig()
    goalkeepers = None
    if keepers:
        if pitch_length is None:
            raise ValueError("pitch_length is needed to fix the goalkeepers")
        players, goalkeepers = fix_goalkeepers(players, keepers, pitch_length, goalkeeper_config,
                                               teams=teams, kickoff=kickoff)
    results, players_s, player_v, ball_s, ball_v = classify_base(players, ball, teams, config, fps)

    # collisions: only for goals.dead_ball (restarts, shots) and keeper_catch (shots lost in
    # flight) -- not for the possession
    collisions = detect_collisions(players_s, ball, cfg, fps)
    h, g = cfg.half_window, cfg.skip
    v_after, p_after, _ = _line_fits(ball, list(range(g, g + h + 1)), fps)
    _, p_before, _ = _line_fits(ball, list(range(-g - h, -g + 1)), fps)
    kicks = [(c.frame, (p_before[c.frame] + p_after[c.frame]) / 2.0, v_after[c.frame])
             for c in collisions]
    catch_stats = {}
    catches = detect_catches(players_s, ball, teams, keepers, catch_config, fps,
                             goals=goalkeepers.goals if goalkeepers else None,
                             pitch_width=pitch_width, stats=catch_stats,
                             kicks=kicks) if keepers else []

    runs = [run + ("control",) for run in control_runs(results, teams, cfg.control_gap)]
    runs += [(c.team, c.start if c.since is None else c.since, c.end, "catch") for c in catches]
    labels, changes, rejected = resolve(runs, [], len(results))

    results = [replace(r, team=team) for r, team in zip(results, labels)]
    all_runs = sorted(runs, key=lambda r: r[1])
    info = TouchInfo(collisions=collisions, runs=[r[:3] for r in all_runs], changes=changes,
                     rejected=rejected, goalkeepers=goalkeepers, catches=catches,
                     catch_stats=catch_stats, all_runs=all_runs)
    return results, players_s, player_v, ball_s, ball_v, info


def summarise(info, fps):
    """Print what the base possession with fixes found."""
    reasons = {}
    for _, _, reason in info.changes:
        reasons[reason] = reasons.get(reason, 0) + 1
    print("base possession: changes {}  {}  | collisions (dead ball only, not possession): "
          "{}".format(len(info.changes), reasons, len(info.collisions)))
    if info.catches:
        by_how = {}
        for c in info.catches:
            by_how[c.release] = by_how.get(c.release, 0) + 1
        print("keeper catches: {}  {}  (mean hidden {:.1f} s)".format(
            len(info.catches), by_how, np.mean([c.end - c.start for c in info.catches]) / fps))
    gk = info.goalkeepers
    if gk is not None:
        print("goalkeepers: goals {} (from {})  frames swapped {}, wrong-side slot disabled {}, "
              "moved {}".format(gk.goals, gk.source, gk.swapped, gk.disabled, gk.moved))
