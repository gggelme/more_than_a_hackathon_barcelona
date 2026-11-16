"""Long stoppages anchored on the Smart Tagging -- goals, disallowed goals, cooling breaks -- as
dead-ball (D) intervals.

ONE CALL does it all, after a touch module's classify() on one half (from its first frame):

    stoppages, labels = goals.dead_ball(labels, touch, touch_info, touch_config, ball,
                                        block_first, pitch_width)

It takes the kicks from the touch pass's collisions, the goals each team defends from its keeper
fix, the Smart Tagging from starter/load_smart_tagging.py (the only file read here, and only if
`events` is not given), finds the stoppages and writes D over them. Callers -- the visualizers,
the experiment -- have no logic of their own.

Why. After a goal the ball is dead from the moment it crosses the line until the kick-off: a minute
or two of celebration, replays, maybe VAR, labelled today with the team that shot. The referee
makes no gesture for a goal, and the players' speed cannot tell a stoppage from slow play. What can:

  ANCHORS, from the Smart Tagging (automatic, allowed, NOT ground truth -- info_hackathon/FAQ.md):
    a GOALS tag and the Kick Off that follows it; a Kick Off without a GOALS before it (the tag was
    missed); a GOALS without a Kick Off (a disallowed goal: play restarts with a free kick or a goal
    kick, the next set piece). The kick-offs that start each half are left out. The tags are not
    trusted on their own: a cooling break was tagged GOALS + Kick Off -- which is fine, it is a
    dead ball too -- and a Kick Off can be tagged where there is none. Hence:

  THE BALL decides, never the players' speed:
    dead    a frame where the ATD ball is missing or still (slower than STILL_SPEED). A stoppage
            is the stretch of dead frames that runs into the restart, ignoring live blips shorter
            than BLIP_S (tracking flicker). It must add up to MIN_STOP_S of dead ball: a tag in the
            middle of live play -- the ball seen and moving -- is rejected.

  END     the restart: the anchor's Kick Off (or, for a disallowed goal, the next set piece) and,
          inside its window, the first frame the ball is seen moving again (faster than
          RESTART_SPEED) after being dead. Not "leaves the centre spot": after a cooling break play
          may restart anywhere. No restart found: nothing is marked -- better than inventing.

  STRONG  a GOALS tag with its team, followed by a Kick Off taken by the other team: a goal. The
          start is searched AROUND THE GOALS TAG (LOOK_BEFORE_S before it to GOAL_AFTER_S after
          it), in the goal being attacked: the last shot at the goal mouth -> when it reached the
          line; else the last frame the ball was seen in that penalty area; else the tag's start.
          From there to the kick-off it is all D, even while the ball MOVES in between -- taken
          out of the net and carried to the centre spot, the ATD sees it moving, but the game is
          stopped. The dead-ball filter only asks for MIN_STOP_S of dead ball in total.

  START   of the WEAK ones -- a lone Kick Off, a GOALS without one, a team N/A (a cooling break):
          how the stoppage that runs into the restart began, in the goal being attacked (the
          GOALS tag's team attacks the other team's goal; else the team taking the kick-off
          conceded, so it is its own goal):
            "shot"           a kick (a collision, with the line fitted after it) at most LINK_S
                             from the start of the stoppage, fast (SHOT_SPEED or more) on a line
                             into the goal mouth (GOAL_HALF_WIDTH + GOAL_MARGIN) -> from when it
                             reached the goal line: kick + distance / speed;
            "lost at goal"   the ball was last seen alive inside that penalty area -> from then;
            "stoppage"       anything else (the cooling break) -> from the start of the stoppage.
          Before the start the possession is untouched: it stays with the team that shot.

Positions in METRES, frames are indices of the processed block (one half, from its first frame).
Thresholds are physical guesses, to be checked on the video -- not tuned against the public ground
truth (info_hackathon/RULES.md).
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GoalConfig:
    """Every threshold, in metres, metres per second and seconds."""
    max_episode_s: float = None     # optional time limit between a goal and its restart; None =
                                    # none (a goal and then a cooling break can last minutes) ...
    max_live_s: float = 30.0        # ... what tells one stoppage from two is REAL PLAY inside: the
                                    # ball seen and moving longer than this, in a row
    look_before_s: float = 30.0     # the stoppage may start this long before the GOALS tag ...
    goal_after_s: float = 10.0      # ... or, for a goal, this long after the tag ends
    min_stop_s: float = 10.0        # dead ball needed between start and restart
    still_speed: float = 0.5        # slower than this the ball is still (placed, held, dead)
    blip_s: float = 1.0             # a live ball shorter than this inside a stoppage is flicker
    restart_speed: float = 2.0      # the ball moving again faster than this is the restart ...
    restart_search_s: float = 3.0   # ... searched this long around the restart tag
    link_s: float = 5.0             # a shot this close to the start of the stoppage caused it
    shot_speed: float = 8.0         # m/s: a kick this fast at the goal mouth was a shot
    goal_half_width: float = 3.66   # the goal is 7.32 m wide (laws of the game) ...
    goal_margin: float = 2.0        # ... give or take this much (tracking, the posts)
    area_depth: float = 16.5        # the penalty area (laws of the game)
    area_half_width: float = 20.16
    start_guard_s: float = 30.0     # a kick-off this close to a half's first frame is its start
    catch_before_goal_s: float = 15.0  # a "catch" by the conceding team ending this soon before
                                       # a goal stoppage is not one: the ball off the net ...
    short_control_before_goal_s: float = 2.0  # ... nor is a control of theirs shorter than this
                                              # (a touch on the way in)
    kickoff_search_s: float = 15.0  # a half's kick-off kick is looked for this long after its
                                    # anchor (tag / games.py), which can be a little early
    kickoff_at_start_s: float = 5.0  # an anchor this close to the block's first frame: the block
                                     # starts with the kick-off (as the first half does), and its
                                     # kick cannot be seen as a collision -- no D before it
    team_names: dict = field(default_factory=lambda: {"barcelona": "H", "atletic club": "A"})


@dataclass
class Stoppage:
    """One dead-ball stretch, in indices of the processed block (end exclusive)."""
    start: int
    end: int                        # the restart: first frame of live play again
    start_case: str                 # "shot", "lost at goal" or "stoppage"
    end_case: str                   # "kick off" or "set piece", "+ ball moving" when refined
    scorer: str = None              # "H" / "A" when the tags or the kick-off say who scored
    tags: list = field(default_factory=list)    # the Smart Tagging tags it was built from
    dead_s: float = 0.0             # seconds of dead ball inside it
    longest_live_s: float = 0.0     # longest stretch of live ball inside (must stay <= max_live_s)
    restarter: str = None           # "H" / "A": who restarts (restarts.py), set by dead_ball()
    restart_how: str = None         # "kick-off rule" or "nearest player"


# -- the tags ----------------------------------------------------------------------------------

def _low(text):
    return (text or "").strip().lower()


def _is_kickoff(event):
    return _low(event["code"]) == "set pieces" and _low(event["tags"].get("Type")).startswith(
        "kick off")


def _team(event, cfg):
    return cfg.team_names.get(_low(event["tags"].get("Team")))


def kicks_from_collisions(collisions, ball, touch_config, fps, line_fits):
    """[(frame, break point, velocity m/s)] of every collision, velocity from the line fitted
    after it -- as possession_touch_4 computes them for the keeper catches.

    line_fits is possession_touch_4._line_fits (passed in, so this module stays independent)."""
    h, g = touch_config.half_window, getattr(touch_config, "skip", 0)
    v_after, p_after, _ = line_fits(ball, list(range(g, g + h + 1)), fps)
    _, p_before, _ = line_fits(ball, list(range(-g - h, -g + 1)), fps)
    return [(c.frame, (p_before[c.frame] + p_after[c.frame]) / 2.0, v_after[c.frame])
            for c in collisions]


# -- the ball ----------------------------------------------------------------------------------

def ball_speed(ball, fps):
    """(n,) centred speed of the ball in m/s, NaN where a neighbour is missing."""
    speed = np.full(len(ball), np.nan)
    speed[1:-1] = np.linalg.norm(ball[2:] - ball[:-2], axis=1) * fps / 2.0
    return speed


def dead_frames(ball, cfg, fps):
    """(n,) bool: the ball missing, or seen but still (no measurable speed counts as still)."""
    speed = ball_speed(ball, fps)
    moving = np.isfinite(speed) & (np.nan_to_num(speed) >= cfg.still_speed)
    return ~moving


def stoppage_start(dead, restart, floor, cfg, fps):
    """First frame of the stretch of dead ball that runs into `restart`, ignoring live blips
    shorter than blip_s, not before `floor`."""
    blip = int(round(cfg.blip_s * fps))
    live, t = 0, restart - 1
    while t >= floor:
        live = 0 if dead[t] else live + 1
        if live > blip:
            return t + live                         # first frame after that live stretch
        t -= 1
    return floor


def refine_restart(dead, ball, tag_start, tag_end, cfg, fps):
    """First frame, around the restart tag, where the ball moves again after being dead; None."""
    search = int(round(cfg.restart_search_s * fps))
    lo, hi = max(1, tag_start - search), min(len(ball) - 1, tag_end + search)
    speed = ball_speed(ball, fps)
    for t in range(lo, hi):
        if np.isfinite(speed[t]) and speed[t] >= cfg.restart_speed and dead[max(0, t - 10):t].any():
            return t
    return None


def shot_reaches_line(kick, goal_x, pitch_width, cfg, fps):
    """Frame the kick reached the goal line at x = goal_x, or None if it was not a shot at that
    goal mouth (fast enough, heading into it, crossing the line inside the goal's width)."""
    frame, point, velocity = kick
    speed = float(np.linalg.norm(velocity))
    if not (np.isfinite(speed) and speed >= cfg.shot_speed) or abs(velocity[0]) < 1e-6:
        return None
    seconds = (goal_x - point[0]) / velocity[0]
    if seconds <= 0:
        return None                                 # moving away from that goal
    y = point[1] + velocity[1] * seconds
    if abs(y - pitch_width / 2.0) > cfg.goal_half_width + cfg.goal_margin:
        return None
    return frame + max(1, int(round(seconds * fps)))


# -- detection ---------------------------------------------------------------------------------

def detect_stoppages(events, block_first, ball, kicks, goals, pitch_width=68.0, cfg=None, fps=25,
                     stats=None, ends_half=False, kickoff_hint=None):
    """[Stoppage] in one half.

    events       Smart Tagging, starter.load_smart_tagging.load() (video frames)
    block_first  video frame of index 0 of the arrays (the half's first frame)
    ball         (n, 2) raw ATD ball in metres, NaN where missing
    kicks        [(frame, point, velocity)] in block indices (kicks_from_collisions)
    goals        {team: x of the goal it defends}, from fix_goalkeepers' report
    stats        a dict to count what happened to every anchor
    ends_half    the arrays reach the half's last frame: look for the final whistle too
    kickoff_hint block index where the half's kick-off should be (games.py's anchor), or None
    The half's start, up to its kick-off, is always looked for (half_edges).
    """
    cfg = cfg or GoalConfig()
    stats = {} if stats is None else stats
    n = len(ball)

    def count(key):
        stats[key] = stats.get(key, 0) + 1

    def local(e):
        return dict(e, start=e["start_frame"] - block_first, end=e["end_frame"] - block_first)

    inside = [local(e) for e in events if e["end_frame"] > block_first and
              e["start_frame"] < block_first + n]
    guard = int(round(cfg.start_guard_s * fps))
    half_starts = sorted((e for e in inside if _is_kickoff(e) and
                          _low(e["tags"].get("Type")) == "kick off (start)"), key=lambda e: e["start"])
    kickoffs = sorted((e for e in inside if _is_kickoff(e) and e["start"] > guard and
                       _low(e["tags"].get("Type")) != "kick off (start)"), key=lambda e: e["start"])
    goal_tags = sorted((e for e in inside if _low(e["code"]) == "goals"), key=lambda e: e["start"])
    set_pieces = sorted((e for e in inside if _low(e["code"]) == "set pieces" and not _is_kickoff(e)),
                        key=lambda e: e["start"])
    # no time limit by default: what tells one stoppage from two is real play inside (below)
    max_episode = n if cfg.max_episode_s is None else int(round(cfg.max_episode_s * fps))
    max_live = int(round(cfg.max_live_s * fps))

    # Episodes: (GOALS tag or None, restart tag, kind of restart)
    episodes, used, windows = [], set(), []
    for g in goal_tags:
        if any(a <= g["start"] <= b for a, b in windows):
            episodes.append((g, None, "inside"))   # e.g. a cooling break tagged GOALS after a goal:
            continue                                # it must not take the next goal's kick-off
        k = next((k for k in kickoffs if g["start"] - 5 * fps <= k["start"] <= g["end"] + max_episode
                  and id(k) not in used), None)
        if k is not None:
            used.add(id(k))
            episodes.append((g, k, "kick off"))
            windows.append((g["start"], k["end"]))
            continue
        s = next((s for s in set_pieces if g["start"] <= s["start"] <= g["end"] + max_episode), None)
        episodes.append((g, s, "set piece") if s is not None else (g, None, None))
    episodes += [(None, k, "kick off") for k in kickoffs if id(k) not in used]

    dead = dead_frames(ball, cfg, fps)
    stoppages = []
    for g, r, end_kind in sorted(episodes, key=lambda e: (e[1] or e[0])["start"]):
        count("anchors")
        if end_kind == "inside" or g is not None and any(s.start <= g["start"] < s.end
                                                          for s in stoppages):
            count("skipped: GOALS inside a stoppage already found (e.g. a cooling break after "
                  "a goal)")
            continue
        if r is None:
            count("rejected: no restart after the GOALS tag")
            continue
        # the restart: the first COLLISION around the restart tag (the kick-off kick itself),
        # the same rule as after a cooling break (fix_cooling_break.restart_after); else the
        # ball moving again after being dead; else the tag
        from tracking_possession.fix_cooling_break import restart_after
        search = int(round(cfg.restart_search_s * fps))
        refined, how = restart_after(max(0, r["start"] - search), min(n, r["end"] + search),
                                     kicks)
        if refined is None:
            refined = refine_restart(dead, ball, max(0, r["start"]), min(n, r["end"]), cfg, fps)
            how = "ball moving" if refined is not None else None
        restart = refined if refined is not None else max(0, r["start"])
        end_case = end_kind + (" + " + how if how else " (tag)")
        if not 0 < restart <= n:
            count("rejected: restart outside the half")
            continue

        # who scored, and so which goal was attacked
        tagged_scorer = _team(g, cfg) if g is not None else None
        kicker = _team(r, cfg) if end_kind == "kick off" else None
        scorer = tagged_scorer
        if scorer is None and kicker is not None:
            scorer = "A" if kicker == "H" else "H"
        attacked = None
        if scorer is not None:
            attacked = goals.get("A" if scorer == "H" else "H")

        # STRONG: a GOALS tag with its team, its Kick Off after it, the kicker the other team. The
        # goal is searched around the GOALS tag; the ball may move in between (taken out of the
        # net, carried to the centre spot) -- it is all dead ball until the kick-off
        strong = (tagged_scorer is not None and end_kind == "kick off" and attacked is not None
                  and kicker != tagged_scorer)
        if strong:
            start, start_case = goal_moment(g, restart, ball, kicks, attacked, pitch_width, cfg,
                                            fps)
            dead_s = float(dead[start:restart].sum()) / fps
            if dead_s < cfg.min_stop_s:
                count("rejected (goal + kick off): the ball was not dead long enough")
                continue
            live = longest_live(dead, start, restart)
            if live > max_live:
                count("rejected (goal + kick off): real play inside")
                continue
            stoppages.append(Stoppage(start, restart, start_case, end_case, scorer,
                                      [g, r], dead_s, longest_live_s=live / fps))
            count("stoppage (goal + kick off): {} -> {}".format(start_case, end_case))
            continue

        # WEAK (a lone Kick Off, a GOALS without one, a team N/A -- a cooling break): the stretch
        # of dead ball that runs into the restart, with the strict filter
        floor = max(0, (g["start"] - int(round(cfg.look_before_s * fps))) if g is not None
                    else restart - max_episode)
        start = stoppage_start(dead, restart, floor, cfg, fps)
        dead_s = float(dead[start:restart].sum()) / fps
        if dead_s < cfg.min_stop_s:
            count("rejected: the ball was not dead long enough")
            continue

        start_case = "stoppage"
        if attacked is not None:
            link = int(round(cfg.link_s * fps))
            shots = [shot_reaches_line(kick, attacked, pitch_width, cfg, fps) for kick in kicks
                     if start - link <= kick[0] <= start + link]
            shots = [s for s in shots if s is not None]
            last_live = max(0, start - 1)
            if shots:
                start_case, start = "shot", int(min(max(shots), restart - 1))
            elif _in_area(ball[last_live], attacked, pitch_width, cfg):
                start_case = "lost at goal"

        live = longest_live(dead, start, restart)
        if live > max_live:
            count("rejected: real play inside")
            continue
        stoppages.append(Stoppage(start, restart, start_case, end_case, scorer,
                                  [t for t in (g, r) if t is not None],
                                  float(dead[start:restart].sum()) / fps,
                                  longest_live_s=live / fps))
        count("stoppage: {} -> {}".format(start_case, end_case))

    stoppages += half_edges(half_starts, kickoff_hint, dead, ball, kicks, n, cfg, fps, count,
                            ends_half)
    return sorted(stoppages, key=lambda s: s.start)


def half_edges(half_starts, kickoff_hint, dead, ball, kicks, n, cfg, fps, count, ends_half=False):
    """The dead ball at the edges of a half (block indices, the block starting with the half):

      START  from the half's first frame to its kick-off. The anchor is the "Kick Off (Start)" tag
             and/or `kickoff_hint` (games.py's kick-off anchor, which can be a little EARLY): the
             earlier of the two, if in the first minutes of the block. The kick-off is the first
             COLLISION from there on, within KICKOFF_SEARCH_S (the kick itself -- whoever stood
             with the ball on the centre spot before it, an "owner" for the classifier, is still
             dead ball); else the ball moving again; else the anchor. The second half's block
             begins about two minutes before its kick-off.
      END    with ends_half (the arrays reach the half's last frame): the stretch of dead ball that
             runs into that last frame -- the final whistle -- if it lasts MIN_STOP_S. Players
             walking off for good is fix_cooling_break's, merged with it."""
    from tracking_possession.fix_cooling_break import restart_after
    out = []
    early = int(round(5 * 60 * fps))                # a half's kick-off is within its first minutes
    anchors = [k["start"] for k in half_starts[:1]] + ([kickoff_hint] if kickoff_hint is not None
                                                       else [])
    anchors = [a for a in anchors if 0 <= a < early]
    if anchors and min(anchors) <= int(round(cfg.kickoff_at_start_s * fps)):
        anchors = []                    # the block starts with the kick-off (the first half): no
        count("half start: the block starts at the kick-off, nothing before it")  # dead ball
    if anchors:
        anchor = min(anchors)
        search = int(round(cfg.restart_search_s * fps))
        limit = min(n, anchor + int(round(cfg.kickoff_search_s * fps)))
        kick, how = restart_after(max(0, anchor - search), limit, kicks)
        if kick is None:
            kick = refine_restart(dead, ball, max(0, anchor), limit, cfg, fps)
            how = "ball moving" if kick is not None else None
        kickoff = kick if kick is not None else anchor
        if kickoff > 0:
            out.append(Stoppage(0, int(kickoff), "half start", "kick off (start) + " + (
                how or "anchor"), tags=list(half_starts[:1]),
                dead_s=float(dead[:kickoff].sum()) / fps))
            count("stoppage: half start -> kick off")
    if ends_half:
        start = stoppage_start(dead, n, max(0, n - int(round(10 * 60 * fps))), cfg, fps)
        if (n - start) / fps >= cfg.min_stop_s:
            out.append(Stoppage(int(start), n, "final whistle", "end of the half",
                                dead_s=float(dead[start:n].sum()) / fps))
            count("stoppage: end of the half (dead ball to the last frame)")
    return out


def longest_live(dead, start, stop):
    """Longest run of consecutive live frames (ball seen and moving) in [start, stop)."""
    best = run = 0
    for is_dead in dead[start:stop]:
        run = 0 if is_dead else run + 1
        best = max(best, run)
    return best


def _in_area(point, goal_x, pitch_width, cfg):
    """The point is seen and inside the penalty area of the goal at x = goal_x."""
    return bool(np.all(np.isfinite(point)) and abs(point[0] - goal_x) <= cfg.area_depth and
                abs(point[1] - pitch_width / 2.0) <= cfg.area_half_width)


def goal_moment(g, restart, ball, kicks, attacked, pitch_width, cfg, fps):
    """(frame, case) of the goal of a strong episode, searched around its GOALS tag -- from
    LOOK_BEFORE_S before it to GOAL_AFTER_S after it ends, never past the restart:
      "shot"          the LAST kick at the goal mouth in that window: when it reached the line;
      "lost at goal"  else the last frame the ball was seen inside that penalty area;
      "goal tag"      else the start of the GOALS tag (the tag may or may not hold the goal)."""
    lo = max(0, g["start"] - int(round(cfg.look_before_s * fps)))
    hi = min(restart - 1, g["end"] + int(round(cfg.goal_after_s * fps)))
    shots = [(kick[0], shot_reaches_line(kick, attacked, pitch_width, cfg, fps))
             for kick in kicks if lo <= kick[0] <= hi]
    shots = [(frame, line) for frame, line in shots if line is not None]
    if shots:
        return int(min(max(shots)[1], restart - 1)), "shot"
    for t in range(hi, lo - 1, -1):
        if _in_area(ball[t], attacked, pitch_width, cfg):
            return t, "lost at goal"
    return int(min(max(0, g["start"]), restart - 1)), "goal tag"


def dead_ball(labels, touch, touch_info, touch_config, ball, block_first, pitch_width=68.0,
              cfg=None, fps=25, events=None, stats=None, players=None, teams=None,
              restart_config=None, cooling=True, cooling_config=None, pitch_length=None,
              half_last=None, kickoff_hint=None, referee_gesture=None, referee_dead=None,
              referee_restart=None):
    """(stoppages, labels with D) for one half -- everything goals.py does, in one call.

    labels        the team per frame of the touch module ("H" / "A" / "D" ...), a list; a copy
                  with D written over the stoppages is returned
    players       (n, slots, 2) players in metres (keepers fixed) and `teams` the team of every
                  slot: with them, whoever RESTARTS after each stoppage gets the ball from the
                  restart (tracking_possession/restarts.py) -- the touch pass resolves the
                  possession again with those restarts as runs of control (it needs resolve() and
                  info.all_runs, as possession_touch_4 has); without them only D is written
    cooling       with players, also find the cooling breaks (fix_cooling_break.py: many players
                  off the pitch towards one place) and merge them with the stoppages
    pitch_length  metres along x; None = taken from the goals the teams defend (0 and length)
    half_last     video frame of the half's last frame (games.half_video_blocks): when the arrays
                  reach it, the end of the half is D too (the final whistle, players walking off)
    kickoff_hint  video frame of the half's kick-off from games.py (anchor_video_frame of
                  "kickoff" / "kickoff_2nd"), helping the "Kick Off (Start)" tag; may be early
    referee_gesture  (n,) the referee's gesture score per frame (src/referee_signal.
                  gesture_for_block, NaN where not precomputed): a cooling break starts at his
                  last signal before the walk-off and ends when the ball moves after his restart
                  signal (fix_cooling_break.py)
    referee_dead  (n,) bool: visualizing_deadball's own dead-ball decision (src/referee_signal.py
                  .dead_for_block: referee, ball out, its restart) -- its stretches of D are
                  added as they are where nothing else is; one that runs into a goal / break /
                  half edge is cut where that begins (referee.cut_before)
    referee_restart  (n,) bool: visualizing_deadball's detected restarts (src/referee_signal.
                  restart_for_block): its D ends at the first one after which the ball stays in
                  play (referee.deadball_stoppages), not when its score drops
    touch         the touch module itself (e.g. tracking_possession.possession_touch_4): its
                  _line_fits gives each collision's line after it
    touch_info    the info its classify() returned: collisions, and goalkeepers (the goal each
                  team defends -- classify must have been given the keepers)
    touch_config  its TouchConfig (half_window, skip)
    ball          (n, 2) raw ATD ball in metres, the same array classify() was given
    block_first   video frame of index 0 (the half's first frame)
    events        the Smart Tagging; None = read it with starter.load_smart_tagging
    stats         a dict to count what happened to every anchor
    """
    if events is None:
        from starter.load_smart_tagging import load as load_smart_tagging
        events = load_smart_tagging()
    kicks = kicks_from_collisions(touch_info.collisions, ball, touch_config, fps,
                                  touch._line_fits)
    keepers = getattr(touch_info, "goalkeepers", None)
    goals = keepers.goals if keepers is not None else {}
    ends_half = half_last is not None and block_first + len(ball) - 1 >= half_last
    hint = None if kickoff_hint is None else kickoff_hint - block_first
    stoppages = detect_stoppages(events, block_first, ball, kicks, goals, pitch_width, cfg, fps,
                                 stats, ends_half, hint)

    # Cooling breaks: many players off the pitch at once (fix_cooling_break.py), merged in
    if cooling and players is not None:
        length = pitch_length if pitch_length is not None else (max(goals.values()) if goals
                                                                else None)
        if length:
            from tracking_possession.fix_cooling_break import cooling_breaks, merge
            breaks = cooling_breaks(players, ball, length, pitch_width, cooling_config, cfg, fps,
                                    stats, kicks, ends_half, referee_gesture)
            stoppages = merge(stoppages, breaks)

    # visualizing_deadball's own decision (fouls, offsides, ball out), only where nothing else is
    if referee_dead is not None:
        from tracking_possession.referee import cut_before, deadball_stoppages
        length = pitch_length if pitch_length is not None else (max(goals.values()) if goals
                                                                else None)
        ref = deadball_stoppages(referee_dead, fps, stats=stats, restart=referee_restart,
                                 ball=ball, pitch_length=length, pitch_width=pitch_width)
        stoppages = sorted(stoppages + cut_before(ref, stoppages, fps, stats=stats),
                           key=lambda s: s.start)

    # Resolve the possession again (when the touch pass allows it), with two corrections:
    #  - a keeper catch that overlaps a stoppage is dropped: the ball was dead, there was no catch
    #    (a goal whose ball came back off the net into the area looks like one);
    #  - whoever restarts gets the ball: the restarts become runs of control (restarts.py)
    labels = list(labels)
    if hasattr(touch, "resolve") and getattr(touch_info, "all_runs", None):
        runs = list(touch_info.all_runs)
        cfg = cfg or GoalConfig()
        link = int(round(cfg.catch_before_goal_s * fps))

        short = int(round(cfg.short_control_before_goal_s * fps))

        def spurious(run):
            """What the conceding team cannot have had, or the dead ball cannot hold:
              - a catch inside a stoppage, or by the conceding team ending at most
                catch_before_goal_s before a goal: had the keeper caught it, no goal;
              - a SHORT control (under short_control_before_goal_s) by the conceding team ending
                that soon before a goal: a touch on the way in, not a control -- had they
                controlled it, no goal. Longer controls, and everything else, are left alone."""
            if len(run) < 4 or run[3] not in ("catch", "control"):
                return False
            kind, length = run[3], run[2] - run[1] + 1
            for s in stoppages:
                if kind == "catch" and run[1] < s.end and run[2] >= s.start:
                    return True
                if s.scorer is not None and run[0] != s.scorer and \
                        s.start - link <= run[2] < s.start and \
                        (kind == "catch" or length < short):
                    return True
            return False

        dropped = [r for r in runs if spurious(r)]
        runs = [r for r in runs if r not in dropped]
        found = []
        if players is not None and teams is not None:
            from tracking_possession.restarts import restart_runs, restarts
            found = restarts(stoppages, players, ball, teams, restart_config)
            for s in stoppages:
                r = next((r for r in found if r.frame == s.end), None)
                s.restarter, s.restart_how = (r.team, r.how) if r is not None else (None, None)
            runs += restart_runs(found)
        if found or dropped:
            resolved, _, _ = touch.resolve(runs, [c.frame for c in touch_info.collisions],
                                           len(labels), touch_config, fps)
            labels = [team or "D" for team in resolved]
        if stats is not None:
            stats["restarts given to their team"] = len(found)
            stats["keeper catches dropped (dead ball, or a goal right after)"] = sum(
                r[3] == "catch" for r in dropped)
            stats["short controls of the conceding team dropped (a goal right after)"] = sum(
                r[3] == "control" for r in dropped)
    return stoppages, apply_dead_ball(labels, stoppages)


def print_stoppages(stoppages, stats, block_first, fps=25):
    """What happened to every anchor, and each stoppage in video time, to jump to it."""
    print("goals.py anchors: {}".format(stats))
    for s in stoppages:
        a, b = (s.start + block_first) / fps, (s.end + block_first) / fps
        print("  D {:02d}:{:05.2f} -> {:02d}:{:05.2f}  ({:.0f} s, dead {:.0f} s, longest live "
              "{:.0f} s)  start: {:<13} end: {:<26} scorer {}  restart by {}{}".format(
                  int(a // 60), a % 60, int(b // 60), b % 60, b - a, s.dead_s, s.longest_live_s,
                  s.start_case,
                  s.end_case, s.scorer or "?", s.restarter or "?",
                  " ({})".format(s.restart_how) if s.restart_how else ""))


def apply_dead_ball(labels, stoppages, offset=0):
    """Write D over labels[offset + start : offset + end] for every stoppage, in place."""
    for s in stoppages:
        labels[offset + s.start:offset + s.end] = ["D"] * (s.end - s.start) if isinstance(
            labels, list) else "D"
    return labels
