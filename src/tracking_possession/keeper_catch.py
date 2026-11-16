"""Keeper catches the tracking cannot see: the ball vanishes at the keeper and comes back from him.

Pure logic on positions in METRES: no possession rules here. possession_touch_4.py turns every catch
into a run of control of the keeper's team.

Why. When a keeper catches the ball it is in his hands, hidden by his body: the ATD loses it right
there, often for seconds, until he rolls, throws or kicks it out. possession.py never sees a ball
moving with him, so it never names him the owner -- and the attacking team keeps a possession it
lost. What does show is the shape of the gap:

  release  what makes it a catch is where the ball COMES BACK after a gap of at least MIN_GAP_S:
           within RELEASE_REACH of the keeper (he rolls it, drops it, walks out with it), or its
           flight extended back up to BACK_S passes within RELEASE_REACH of him (a throw or a kick
           picked up only in the air), or -- with the goals known -- inside his PENALTY AREA (plus
           AREA_MARGIN) with the player nearest to it of his team (a rebound an attacker picks up
           in the area is not a catch). A ball that comes back elsewhere is not a catch: a goal
           comes back at the centre spot for the kick-off, a corner at the corner flag (outside the
           area's width). A goal kick does come back in the area: taken as a catch -- the right team
           afterwards, though those frames are really a dead ball (another problem).

  since    WHEN the keeper got it is not seen -- the ball was hidden -- so it is set by how the
           ball VANISHED, and possession is the keeper's team's only from then on:
             "at the keeper"  it vanished within CATCH_REACH of him: hidden in his hands right
                              there -> from the last frame it was seen;
             "shot"           it was SHOT at his area: the last KICK (a collision, with the line
                              fitted after it) within SHOT_WINDOW_S before it vanished is fast
                              (SHOT_SPEED or more) on a line that enters his penalty area at the
                              goal line -> from when it reached him, kick + distance to him /
                              speed. The kick's line is fitted beyond the part the ATD rounds off,
                              so it gives the real shot; the last frames before the ball vanishes
                              do not -- the smoothed ATD slows the ball down BEFORE the keeper
                              takes it. Without kicks, those last frames are used instead. Until
                              then it is still the shooting team's ball, as the rules say;
             "unseen"         it vanished anywhere else (a dribble down the wing, a cross never
                              seen): when he got it cannot be told -> from the frame it comes
                              back, the only moment that is sure. Never backdated.
Keeper slots must already be right (fix_goalkeepers.py): the catch is credited to the keeper's slot.

Every threshold is physical (metres, seconds), to be reasoned about and checked on the video -- not
tuned against the public ground truth (info_hackathon/RULES.md).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CatchConfig:
    """Thresholds of the catch detector, in metres and seconds."""
    min_gap_s: float = 0.4      # a shorter gap is tracking flicker, not a ball held
    catch_reach: float = 3.0    # keeper - last ball seen: arms, a dive, and the error of a ball in
                                # the air projected on the grass
    shot_speed: float = 8.0     # m/s: a kick this fast, heading into his area, was a shot (a
                                # dribble runs at 5-7 m/s; a soft shot still above this)
    shot_window_s: float = 1.5  # ... if it came at most this long before the ball vanished
    release_reach: float = 3.0  # keeper - ball as it comes back (or its flight extended back)
    area_depth: float = 16.5    # the penalty area (laws of the game): the ball back in it ...
    area_half_width: float = 20.16  # ... 40.32 m wide, centred on the goal
    area_margin: float = 3.0    # ... give or take this much (tracking error, the ball rolling)
    back_s: float = 1.0         # how far back the returning flight is extended
    fit_points: int = 6         # ball positions after the gap used to extend its flight back
    search: int = 5             # frames to look around for a keeper position when one is missing


@dataclass
class Catch:
    team: str
    slot: int                   # the keeper's slot
    start: int                  # last frame the ball was seen, index inside the processed block
    end: int                    # frame the ball comes back
    vanish_distance: float      # keeper - ball as it vanished, m
    release_distance: float     # keeper - ball (or its extended flight) as it came back, m
    release: str                # "from his hands", "thrown / kicked" or "back in his area"
    case: str = "at the keeper"  # how it vanished: "at the keeper", "shot" or "unseen"
    since: int = None           # frame from which the ball is the keeper's team's (see module doc)


def ball_gaps(ball, min_len):
    """[(last_seen, back)] for every gap of at least `min_len` missing frames inside the block."""
    present = np.isfinite(ball[:, 0])
    gaps, i, n = [], 0, len(ball)
    while i < n:
        if present[i]:
            i += 1
            continue
        j = i
        while j < n and not present[j]:
            j += 1
        if i > 0 and j < n and j - i >= min_len:
            gaps.append((i - 1, j))
        i = j
    return gaps


def _position(track, t, search):
    """track[t], or the nearest present frame within `search` frames; None if there is none."""
    for d in range(search + 1):
        for u in (t - d, t + d):
            if 0 <= u < len(track) and np.isfinite(track[u, 0]):
                return track[u]
    return None


def _flight_back(ball, back, keeper_track, cfg, fps):
    """Smallest distance between the keeper and the returning ball's flight extended back."""
    idx = [t for t in range(back, min(len(ball), back + cfg.fit_points * 2))
           if np.isfinite(ball[t, 0])][:cfg.fit_points]
    if len(idx) < 3:
        return np.inf
    t = np.asarray(idx, float)
    vx, x0 = np.polyfit(t - back, ball[idx, 0], 1)
    vy, y0 = np.polyfit(t - back, ball[idx, 1], 1)
    best = np.inf
    for tau in range(0, int(round(cfg.back_s * fps)) + 1):
        keeper = _position(keeper_track, back - tau, cfg.search)
        if keeper is not None:
            point = np.array([x0 - vx * tau, y0 - vy * tau])
            best = min(best, float(np.linalg.norm(point - keeper)))
    return best


def _flight_out(ball, last_seen, cfg, fps):
    """(velocity in m/s per frame-time, position) of the ball's last straight flight before it
    vanished, fitted on its last `fit_points` positions; None if too few."""
    idx = [t for t in range(max(0, last_seen - cfg.fit_points * 2), last_seen + 1)
           if np.isfinite(ball[t, 0])][-cfg.fit_points:]
    if len(idx) < 3:
        return None
    t = np.asarray(idx, float) - last_seen
    vx, x0 = np.polyfit(t, ball[idx, 0], 1)
    vy, y0 = np.polyfit(t, ball[idx, 1], 1)
    return np.array([vx, vy]) * fps, np.array([x0, y0])


def _shot_into_area(velocity, where, goal_x, pitch_width, cfg):
    """The ball, moving at `velocity` from `where`, crosses the goal line at x = goal_x inside the
    width of the penalty area."""
    if abs(velocity[0]) < 1e-6:
        return False
    t = (goal_x - where[0]) / velocity[0]
    if t <= 0:
        return False                                # moving away from that goal
    y = where[1] + velocity[1] * t
    return abs(y - pitch_width / 2.0) <= cfg.area_half_width + cfg.area_margin


def _in_area(point, goal_x, pitch_width, cfg):
    """The point is inside the penalty area of the goal at x = goal_x, give or take the margin."""
    return (abs(point[0] - goal_x) <= cfg.area_depth + cfg.area_margin and
            abs(point[1] - pitch_width / 2.0) <= cfg.area_half_width + cfg.area_margin)


def _nearest_team(players_s, t, teams, point):
    """Team of the player nearest to `point` at frame t, or None."""
    present = np.isfinite(players_s[t, :, 0])
    if not present.any():
        return None
    d = np.linalg.norm(players_s[t, present] - point, axis=1)
    return teams[np.flatnonzero(present)[int(np.argmin(d))]]


def _shot(kicks, out, last_seen, where, goal_x, keeper_at, pitch_width, cfg, fps):
    """Frame the shot reached the keeper, or None if the ball was not shot at his area.

    kicks      [(frame, point, velocity)] of the collisions, velocity in m/s from the line fitted
               after each one; the latest within shot_window_s before last_seen is the shot
    out        the fit of the ball's last frames (_flight_out), used only without kicks
    keeper_at  function frame -> keeper position (or None)
    """
    window = int(round(cfg.shot_window_s * fps))
    candidates = [k for k in (kicks or []) if last_seen - window <= k[0] <= last_seen]
    if candidates:
        frame, point, velocity = max(candidates, key=lambda k: k[0])
    elif kicks is None and out is not None:
        frame, (velocity, point) = last_seen, out
    else:
        return None
    speed = float(np.linalg.norm(velocity))
    if not (np.isfinite(speed) and speed >= cfg.shot_speed and
            _shot_into_area(velocity, point, goal_x, pitch_width, cfg)):
        return None
    keeper = keeper_at(frame)
    if keeper is None:
        return None
    return frame + max(1, int(round(float(np.linalg.norm(keeper - point)) / speed * fps)))


def detect_catches(players_s, ball, teams, keepers, cfg=None, fps=25, goals=None,
                   pitch_width=68.0, stats=None, kicks=None):
    """[Catch] in one block.

    players_s    (n, slots, 2) players in metres, keepers already fixed
    ball         (n, 2) raw ball in metres, NaN where missing
    teams        team ("H" / "A") of every slot
    keepers      {team: keeper slot index}
    goals        {team: x of the goal it defends} (fix_goalkeepers' report); without it only the
                 distances to the keeper are used
    pitch_width  metres along y (games.GAMES[GAME]["pitch"][1]), to place the penalty area
    stats        a dict to count, per gap, why it was or was not a catch (for the diagnostic)
    kicks        [(frame, point, velocity m/s)] of the collisions, velocity from the line fitted
                 after each (the touch pass has them): how a shot is recognised. None = use the
                 ball's last frames before it vanished instead
    """
    cfg = cfg or CatchConfig()
    teams = np.asarray(teams)
    goals = goals or {}
    stats = {} if stats is None else stats

    def count(key):
        stats[key] = stats.get(key, 0) + 1

    catches = []
    for last_seen, back in ball_gaps(ball, int(round(cfg.min_gap_s * fps))):
        count("gaps")
        where, again = ball[last_seen], ball[back]
        out = _flight_out(ball, last_seen, cfg, fps)
        reason = "came back away from the keepers and their areas"
        for team, slot in keepers.items():
            goal_x = goals.get(team)
            # 1. where it comes back: from the keeper, or in his area with his team on it
            keeper_back = _position(players_s[:, slot], back, cfg.search)
            near = (np.inf if keeper_back is None
                    else float(np.linalg.norm(keeper_back - again)))
            if near <= cfg.release_reach:
                release, distance = "from his hands", near
            else:
                flight = _flight_back(ball, back, players_s[:, slot], cfg, fps)
                if flight <= cfg.release_reach:
                    release, distance = "thrown / kicked", flight
                elif goal_x is not None and _in_area(again, goal_x, pitch_width, cfg):
                    if _nearest_team(players_s, back, teams, again) != team:
                        reason = "came back in his area to a rival"
                        continue
                    release, distance = "back in his area", near
                else:
                    continue
            # 2. how it vanished decides since when it is his
            keeper = _position(players_s[:, slot], last_seen, cfg.search)
            vanish = np.inf if keeper is None else float(np.linalg.norm(keeper - where))
            arrived = None if goal_x is None else _shot(
                kicks, out, last_seen, where, goal_x,
                lambda f: _position(players_s[:, slot], f, cfg.search), pitch_width, cfg, fps)
            if vanish <= cfg.catch_reach:
                case, since = "at the keeper", last_seen
            elif arrived is not None:
                # never before the ball was last seen, never after it comes back
                case, since = "shot", min(back, max(last_seen, arrived))
            else:
                case, since = "unseen", back
            catches.append(Catch(team, slot, last_seen, back, vanish, distance, release, case,
                                 since))
            break
        else:
            count(reason)
            continue
        count("catch ({}): {}".format(catches[-1].case, catches[-1].release))
    return catches
