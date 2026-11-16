"""Stoppages from the referee's gestures -- fouls, offsides -- confirmed by the ball. Pure logic:
no video, no files (the gesture score comes from src/referee_signal.py).

Why. A foul or an offside stops the game with no tag and no players walking off: the referee says
it, with his arms. But a gesture alone is not a stoppage: with ADVANTAGE the referee signals and
play goes on (info_hackathon/CHALLENGE.md: the ball is dead only when the game stops). So:

  GESTURE   the gesture score (src/referee_signal.py) at GESTURE_SCORE or more for GESTURE_FRAMES
            in a row -- visualizing_deadball's own entry rule.
  CONFIRM   the BALL dies around it: MIN_DEAD_S of dead ball in a row (missing, still, or outside
            the pitch) starting at most BEFORE_S before the gesture (the whistle comes first, the
            arm right after) and at most CONFIRM_S after it. Play going on -- the ball seen and
            moving -- is advantage: nothing is marked. Not the players' speed: slow play is not
            dead.
  START     where that dead ball begins.
  END       the restart: the first COLLISION once the ball has been dead MIN_DEAD_S, within
            MAX_STOP_S (fix_cooling_break.restart_after, the rule of goals and cooling breaks);
            else the ball moving again; nothing found: no stoppage.

The result is a list of goals.Stoppage (start_case "referee gesture"); goals.dead_ball adds them
only where no goal, cooling break or half's edge already is -- those come first.

Thresholds are physical guesses, to be checked on the video -- not tuned against the public ground
truth (info_hackathon/RULES.md).
"""

from dataclasses import dataclass

import numpy as np

from tracking_possession.goals import GoalConfig, Stoppage, ball_speed, dead_frames


@dataclass
class RefereeConfig:
    """Thresholds, in seconds and frames (gesture score in [0, 1])."""
    gesture_score: float = 0.65     # visualizing_deadball's enter score ...
    gesture_frames: int = 8         # ... held this many frames in a row
    before_s: float = 2.0           # the ball may have died this long before the gesture ...
    confirm_s: float = 4.0          # ... or must die within this long after it; else advantage
    min_dead_s: float = 2.0         # dead ball in a row that confirms the stoppage
    max_stop_s: float = 120.0       # the restart is looked for at most this long after
    out_margin: float = 0.5         # metres: the ball beyond a line by this much is out
    # The END of visualizing_deadball's D: its first restart after which the ball stays in play
    restart_ahead_frames: int = 7   # the ball is checked inside the pitch this many frames after
                                    # the restart (visualizing_deadball's 0.28 s look-ahead)
    live_check_s: float = 4.0       # after the restart, the ball must not die again within this
    dead_again_s: float = 1.5       # long for this long in a row (still or missing)


def deadball_stoppages(dead, fps=25, min_s=1.0, stats=None, restart=None, ball=None,
                       pitch_length=None, pitch_width=None, cfg=None):
    """[goals.Stoppage] from visualizing_deadball's OWN dead-ball decision (src/referee_signal.py:
    the referee's gesture, the players' stillness, the ball out, its hysteresis and its own
    restart detection): every run of D of at least MIN_S. goals.dead_ball adds them where no goal
    or half's edge already is.

    The END. visualizing_deadball's own end comes LATE after a slow restart (a short throw-in, a
    goal kick played short): its restart only unlocks the exit, which then needs its score under
    the exit threshold -- and the players' stillness keeps it up while they walk. So, with
    `restart` (its detected restarts, (n,) bool) and `ball` (metres) and the pitch, the D ends at
    the FIRST of its restarts inside the run after which the ball STAYS IN PLAY:
      - the ball is inside the pitch RESTART_AHEAD_FRAMES later (not the ball tossed to the taker
        from outside);
      - within LIVE_CHECK_S after it, the ball is never dead (still or missing) DEAD_AGAIN_S in a
        row (not the pass to the free-kick taker, who places it; not the keeper setting it down).
        Slow play passes: a slow ball is moving, not still.
    No such restart: its own end, as before.
    """
    from tracking_possession.goals import GoalConfig
    cfg = cfg or RefereeConfig()
    stats = {} if stats is None else stats
    usable = (restart is not None and ball is not None and pitch_length is not None and
              pitch_width is not None)
    if usable:
        restart = np.asarray(restart, bool)
        still = dead_frames(ball, GoalConfig(), fps)
        live_check = int(round(cfg.live_check_s * fps))
        dead_again = int(round(cfg.dead_again_s * fps))
        n = len(ball)

    def stays_live(r):
        end = r + cfg.restart_ahead_frames
        if end >= n or not np.all(np.isfinite(ball[end])):
            return False
        x, y = ball[end]
        if not (0 <= x <= pitch_length and 0 <= y <= pitch_width):
            return False
        run = 0
        for t in range(r + 1, min(n, r + 1 + live_check)):
            run = run + 1 if still[t] else 0
            if run >= dead_again:
                return False
        return True

    found = []
    for a, b in _runs(np.asarray(dead, bool)):
        if (b - a) / fps < min_s:
            continue
        end, how = int(b), "deadball restart"
        if usable:
            first = next((int(r) for r in np.flatnonzero(restart[a:b]) + a
                          if (r - a) / fps >= min_s and stays_live(int(r))), None)
            if first is not None:
                end, how = first, "restart + ball in play"
                stats["deadball ended at a restart with the ball in play"] = stats.get(
                    "deadball ended at a restart with the ball in play", 0) + 1
        found.append(Stoppage(int(a), end, "deadball (referee / ball out)", how))
        stats["deadball stoppages"] = stats.get("deadball stoppages", 0) + 1
    return found


def cut_before(ref, stoppages, fps=25, min_s=1.0, stats=None):
    """The referee's stoppages `ref`, fitted around the `stoppages` already found (goals, cooling
    breaks, half edges), which come first and are never changed:

      touches none          kept whole -- its start, its end, its restart
      starts inside one     dropped: that stoppage already covers its start
      runs into one         cut where the first one it meets begins: the referee's START is his
                            own evidence (the gesture), his END is weaker (deadball's restart
                            detection comes late) and that stoppage has its own. The piece has NO
                            restart of its own: it flows into that stoppage (end_case "into ...",
                            restarts.py skips it). Shorter than MIN_S: dropped.
    """
    stats = {} if stats is None else stats
    kept = []
    for r in ref:
        hit = [s for s in stoppages if s.start < r.end and r.start < s.end]
        if not hit:
            kept.append(r)
            continue
        first = min(hit, key=lambda s: s.start)
        if first.start <= r.start or (first.start - r.start) / fps < min_s:
            stats["deadball stoppages inside a goal / break / half edge (dropped)"] = stats.get(
                "deadball stoppages inside a goal / break / half edge (dropped)", 0) + 1
            continue
        kept.append(Stoppage(int(r.start), int(first.start), r.start_case,
                             "into the next stoppage ({})".format(first.start_case)))
        stats["deadball stoppages cut where a goal / break / half edge begins"] = stats.get(
            "deadball stoppages cut where a goal / break / half edge begins", 0) + 1
    return kept


def _runs(mask):
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


def referee_stoppages(gesture, ball, kicks, pitch_length, pitch_width, cfg=None, goal_config=None,
                      fps=25, stats=None):
    """[goals.Stoppage] from the referee's gestures in one block.

    gesture  (n,) gesture score per frame (0 where unknown), aligned to the block
    ball     (n, 2) raw ATD ball in metres, NaN where missing
    kicks    [(frame, point, velocity)] of the touch pass's collisions (for the restart)
    """
    from tracking_possession.fix_cooling_break import restart_after
    cfg = cfg or RefereeConfig()
    goal_config = goal_config or GoalConfig()
    stats = {} if stats is None else stats
    n = len(ball)

    def count(key):
        stats[key] = stats.get(key, 0) + 1

    m = cfg.out_margin
    out = (np.isfinite(ball[:, 0]) & ((ball[:, 0] < -m) | (ball[:, 0] > pitch_length + m) |
                                      (ball[:, 1] < -m) | (ball[:, 1] > pitch_width + m)))
    dead = dead_frames(ball, goal_config, fps) | out
    speed = ball_speed(ball, fps)
    moving = np.isfinite(speed) & (np.nan_to_num(speed) >= goal_config.restart_speed)
    min_dead = int(round(cfg.min_dead_s * fps))
    before, after = int(round(cfg.before_s * fps)), int(round(cfg.confirm_s * fps))

    found = []
    for a, b in _runs(np.asarray(gesture) >= cfg.gesture_score):
        if b - a < cfg.gesture_frames:
            continue
        count("referee gestures")
        if found and a < found[-1].end:
            continue                                # inside the stoppage it already opened
        # the ball dies around the gesture: MIN_DEAD_S in a row starting in the window
        start = next((t for t in range(max(0, a - before), min(n - min_dead, a + after))
                      if dead[t:t + min_dead].all()), None)
        if start is None:
            count("rejected: play went on (advantage?)")
            continue
        end, how = restart_after(start + min_dead, min(n, start + int(round(
            cfg.max_stop_s * fps))), kicks, moving)
        if end is None:
            count("rejected: no restart found")
            continue
        found.append(Stoppage(int(start), int(end), "referee gesture", "restart: " + how,
                              dead_s=float(dead[start:end].sum()) / fps))
        count("referee stoppages")
    return found
