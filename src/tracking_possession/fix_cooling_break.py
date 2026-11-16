"""Cooling breaks (hydration breaks) as dead ball: many players leave the pitch towards one place.
Pure logic on positions in METRES, indices of the processed block (one half).

Why. In a cooling break play stops and most players walk off the pitch, to the benches, for a
couple of minutes. The ball may be missing or lying still, the players walk slowly -- and slow play
looks the same to anything based on speed. What does not happen in play is MANY players off the
pitch at once: a throw-in or a corner puts one or two outside the lines, a substitution two.

  AWAY     a player is away when he is tracked beyond the lines by more than OUTSIDE_MARGIN, or --
           in case the tracking loses players who walk out of the image -- when he vanished within
           EDGE_M of a line and has not been seen since (for at most VANISH_MEMORY_S).
  BREAK    MIN_AWAY players or more away for MIN_AWAY_S in a row.
  WHERE    they gather is not fixed (the benches can be on either side): it is the centre of the
           away players' positions -- tracked ones outside, vanished ones where they were last seen.
  START    walking back from the moment they are away, while the players as a whole keep closing
           in on that point (their mean distance to it shrinks at APPROACH_SPEED or more): the
           general flow towards it. DIRECTION, not speed: in slow play they do not all converge on
           a point off the pitch. Then, if the ball was already dead there, back to where the dead
           ball began (the whistle), as goals.py does.
  END      the players are back: the first frame where the players tracked INSIDE the pitch are
           at least BACK_RATIO of their usual number (the median over the half -- measured, not
           22). Counting who is on the pitch, not who is still away: slots are tracklets, a
           player who walked out of the image often comes back in a NEW slot while the old one
           stays lost, and the tracking may keep slots on the bench. Then the RESTART is the first
           COLLISION (a kick, from the touch pass) after that -- not the ball merely moving: the
           keeper bouncing or rolling it while the others walk back is still the break
           (restart_after, which goals.py reuses for its kick-offs). No collision: the ball moving
           again, said so. Nothing within MAX_BREAK_S: no break is marked -- better than D to the
           end of the half.

The result is a list of goals.Stoppage (start_case "cooling break"), so D is written and the restart
is given to its team exactly as for a goal (goals.dead_ball merges them).

Thresholds are physical guesses, to be checked on the video -- not tuned against the public ground
truth (info_hackathon/RULES.md).
"""

from dataclasses import dataclass

import numpy as np

from tracking_possession.goals import (GoalConfig, Stoppage, ball_speed, dead_frames,
                                       stoppage_start)


@dataclass
class CoolingConfig:
    """Thresholds, in metres, metres per second and seconds."""
    outside_margin: float = 2.0     # beyond a line by more than this: outside the pitch
    edge_m: float = 3.0             # a player lost this close to a line walked out of the image ...
    vanish_memory_s: float = 180.0  # ... and counts as away for at most this long
    min_away: int = 8               # this many players away ...
    min_away_s: float = 20.0        # ... this long in a row: a cooling break
    approach_speed: float = 0.3     # m/s: the players' mean distance to the point shrinking this
                                    # fast is "flowing towards it"
    flow_gap_s: float = 2.0         # the flow may pause this long and still be the same walk off
    max_back_s: float = 120.0       # the start is looked for at most this long before
    back_ratio: float = 0.7         # the game is back when the players tracked ON the pitch are
                                    # this share of their usual number (the half's median) ...
    max_break_s: float = 360.0      # ... and the ball moves again, within this long of the break;
                                    # not found: no break is marked (better than D to the end)
    # The referee (src/referee_signal.py's gesture score, where it was precomputed)
    referee_score: float = 0.5      # a referee SIGNAL: his gesture score at this or more ...
    referee_frames: int = 4         # ... this many frames in a row
    referee_before_back_s: float = 10.0  # his restart signal may come this long before the players
                                    # are counted back on the pitch


def away_players(players, pitch_length, pitch_width, cfg, fps):
    """((n,) count of players away, (n, slots) bool) -- tracked outside, or lost at an edge."""
    n, slots = players.shape[:2]
    x, y = players[..., 0], players[..., 1]
    seen = np.isfinite(x)
    m = cfg.outside_margin
    outside = seen & ((x < -m) | (x > pitch_length + m) | (y < -m) | (y > pitch_width + m))

    lost = np.zeros((n, slots), bool)
    memory = int(round(cfg.vanish_memory_s * fps))
    for s in range(slots):
        t = 0
        while t < n:
            if seen[t, s]:
                t += 1
                continue
            gap_start = t
            while t < n and not seen[t, s]:
                t += 1
            if gap_start == 0:
                continue
            px, py = x[gap_start - 1, s], y[gap_start - 1, s]
            near_edge = (px < cfg.edge_m or px > pitch_length - cfg.edge_m or
                         py < cfg.edge_m or py > pitch_width - cfg.edge_m)
            if near_edge:
                lost[gap_start:min(t, gap_start + memory), s] = True
    away = outside | lost
    return away.sum(axis=1), away


def _runs(mask):
    """[(start, stop)] of the True runs of a boolean array."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


def gathering_point(players, away, a, b):
    """Centre of where the away players are (or were last seen) in [a, b)."""
    points = []
    for s in range(players.shape[1]):
        frames = np.flatnonzero(away[a:b, s]) + a
        if not len(frames):
            continue
        seen = [f for f in frames if np.isfinite(players[f, s, 0])]
        if seen:
            points.append(players[seen, s].mean(axis=0))
        else:                                       # vanished: where he was last seen
            before = np.flatnonzero(np.isfinite(players[:frames[0], s, 0]))
            if len(before):
                points.append(players[before[-1], s])
    return np.mean(points, axis=0) if points else None


def flow_start(players, point, a, cfg, fps):
    """First frame of the general walk towards `point` that ends at `a` (see the module doc)."""
    lo = max(0, a - int(round(cfg.max_back_s * fps)))
    d = np.linalg.norm(players[lo:a + 1] - point, axis=2)           # (frames, slots)
    mean = np.array([np.nanmean(row) if np.isfinite(row).any() else np.nan for row in d])
    k = max(1, fps // 2)
    mean = np.convolve(np.nan_to_num(mean, nan=np.nanmean(mean)), np.ones(k) / k, mode="same")
    closing = -np.gradient(mean) * fps >= cfg.approach_speed          # shrinking fast enough
    gap = int(round(cfg.flow_gap_s * fps))
    t, pause = len(closing) - 1, 0
    while t > 0:
        pause = 0 if closing[t] else pause + 1
        if pause > gap:
            return lo + t + pause
        t -= 1
    return lo


def referee_signals(gesture, cfg):
    """(n,) bool: frames inside a referee SIGNAL -- his gesture score at REFEREE_SCORE or more for
    REFEREE_FRAMES in a row. Unknown frames (NaN: not precomputed) are no signal."""
    raised = np.nan_to_num(np.asarray(gesture, float), nan=0.0) >= cfg.referee_score
    signal = np.zeros(len(raised), bool)
    for a, b in _runs(raised):
        if b - a >= cfg.referee_frames:
            signal[a:b] = True
    return signal


def restart_after(after, limit, kicks, moving=None):
    """(frame, how) of the restart in [after, limit): the first COLLISION there (a kick -- the
    restart itself, not the ball merely rolling or bounced in a keeper's hands), else the first
    frame the ball moves (`moving`, (n,) bool), else (None, None). Shared with goals.py."""
    frames = sorted(k[0] for k in (kicks or []) if after <= k[0] < limit)
    if frames:
        return int(frames[0]), "first collision"
    if moving is not None:
        t = next((t for t in range(max(0, after), min(limit, len(moving))) if moving[t]), None)
        if t is not None:
            return int(t), "ball moving (no collision seen)"
    return None, None


def cooling_breaks(players, ball, pitch_length, pitch_width, cfg=None, goal_config=None, fps=25,
                   stats=None, kicks=None, ends_half=False, referee_gesture=None):
    """[goals.Stoppage] of the cooling breaks in one half.

    players  (n, slots, 2) in metres (keepers fixed or not, it does not matter here)
    ball     (n, 2) raw ATD ball in metres, NaN where missing
    kicks    [(frame, point, velocity)] of the touch pass's collisions (goals.kicks_from_
             collisions): the restart is the first one once the players are back
    ends_half  the arrays reach the half's last frame: players who walk off and are not back
             before it is the END OF THE HALF -- D from when it began to the last frame
    referee_gesture  (n,) the referee's gesture score (src/referee_signal.gesture_for_block, NaN
             where not precomputed). With it:
               START  the onset of the LAST referee signal before the walk-off is sure (within
                      MAX_BACK_S): the whistle that stopped the game for the break;
               END    the first referee signal once the players are (nearly) back, then the ball
                      starting to move after it (restart_after: first collision, else moving).
             No signal there (or no gesture): the walk-off flow / players-back rules above.
    """
    cfg = cfg or CoolingConfig()
    goal_config = goal_config or GoalConfig()
    stats = {} if stats is None else stats
    n = len(ball)
    count, away = away_players(players, pitch_length, pitch_width, cfg, fps)
    dead = dead_frames(ball, goal_config, fps)
    speed = ball_speed(ball, fps)
    min_len = int(round(cfg.min_away_s * fps))
    x, y = players[..., 0], players[..., 1]
    on_pitch = (np.isfinite(x) & (x >= 0) & (x <= pitch_length) & (y >= 0) &
                (y <= pitch_width)).sum(axis=1)
    usual = float(np.median(on_pitch))              # most of a half is play
    back_needed = cfg.back_ratio * usual
    moving = np.isfinite(speed) & (np.nan_to_num(speed) >= goal_config.restart_speed)
    signal = (referee_signals(referee_gesture, cfg) if referee_gesture is not None
              else np.zeros(n, bool))
    onsets = np.flatnonzero(signal & ~np.concatenate([[False], signal[:-1]]))
    max_back = int(round(cfg.max_back_s * fps))

    def tally(key):
        stats[key] = stats.get(key, 0) + 1

    breaks = []
    for a, b in _runs(count >= cfg.min_away):
        if b - a < min_len:
            continue
        tally("candidates")
        point = gathering_point(players, away, a, b)
        if point is None:
            continue
        # start: the referee's last signal before the walk-off, else the walk-off flow
        earlier = onsets[(onsets >= max(0, a - max_back)) & (onsets <= a)]
        if len(earlier):
            start, start_case = int(earlier[-1]), "cooling break (referee signal)"
            tally("start: the referee's last signal")
        else:
            start, start_case = flow_start(players, point, a, cfg, fps), "cooling break"
            if dead[start]:                         # the ball already dead there: the whistle
                start = stoppage_start(dead, start + 1, max(0, start - max_back), goal_config,
                                       fps)
            tally("start: walk-off flow (no referee signal)")
        # end: the players back ON the pitch and the ball moving again -- looked for from the
        # moment the break is sure, not from when nobody is "away" any more: lost slots and bench
        # ghosts keep that count up long after the game has restarted
        limit = min(n, a + int(round(cfg.max_break_s * fps)))
        back = next((t for t in range(a + min_len, limit) if on_pitch[t] >= back_needed), None)
        end, restart_how = None, None
        if back is not None:
            # the referee signals the restart (players nearly back), then the ball moves
            lo = max(a + min_len, back - int(round(cfg.referee_before_back_s * fps)))
            given = next((t for t in range(lo, limit) if signal[t]), None)
            if given is not None:
                end, restart_how = restart_after(given, limit, kicks, moving)
                if end is not None:
                    restart_how = "referee signal + " + restart_how
                    tally("end: referee signal + ball moving")
            if end is None:
                end, restart_how = restart_after(back, limit, kicks, moving)
                if end is not None:
                    tally("end: players back (no referee signal)")
        if end is None and ends_half and back is None and limit == n:
            # they walked off and the half ended before they came back: the end of the half
            breaks.append(Stoppage(int(start), n, "players walked off", "end of the half",
                                   dead_s=float(dead[start:n].sum()) / fps))
            stats["end of the half (players walked off)"] = stats.get(
                "end of the half (players walked off)", 0) + 1
            continue
        if end is None:
            stats["rejected: the game never came back within max_break_s"] = stats.get(
                "rejected: the game never came back within max_break_s", 0) + 1
            continue
        how = "players back ({}/{:.0f} on the pitch) + {}".format(on_pitch[back], usual,
                                                                 restart_how)
        if breaks and start <= breaks[-1].end:      # the same break seen twice: merge
            breaks[-1].end = max(breaks[-1].end, end)
            continue
        breaks.append(Stoppage(int(start), int(end), start_case, how,
                               dead_s=float(dead[start:end].sum()) / fps))
        stats["cooling breaks"] = stats.get("cooling breaks", 0) + 1
    return breaks


def merge(stoppages, breaks):
    """Stoppages and cooling breaks as one list: overlapping ones become one stoppage (the goal's
    attributes kept, the span the union of both)."""
    out = sorted(stoppages, key=lambda s: s.start)
    for c in breaks:
        hit = next((s for s in out if c.start < s.end and s.start < c.end), None)
        if hit is None:
            out.append(c)
            continue
        hit.start, hit.end = min(hit.start, c.start), max(hit.end, c.end)
        hit.start_case = hit.start_case + " + cooling break"
    return sorted(out, key=lambda s: s.start)
