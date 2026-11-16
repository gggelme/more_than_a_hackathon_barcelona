"""Possession from collisions and control: a collision says WHEN, the control says WHO.

Second pass on top of possession.py, which is left untouched. Same inputs (ATD positions in METRES)
and the same output as possession.classify(), so swapping one import is enough to compare.

Why. Deciding frame by frame who the ball belongs to fails exactly where possession is decided:
  - in a duel shoulder to shoulder, both players have the same distance, speed and direction, so
    possession.py's score cannot tell them apart and noise picks the winner;
  - who touched the ball is just as arbitrary: "the nearest player" flips with half a metre of
    tracking error.
So nobody is credited with a touch. Two pieces of evidence, each used for what it can tell:

  collision  the ball's trajectory breaks. A free ball goes straight and slows down gently
             (friction), so a straight line fitted to the frames before a moment and another to
             the frames after must agree; when their velocities differ by more than DV_MIN,
             something hit it. The ATD rounds breaks off over a few frames, so SKIP frames on each
             side are left out and the two lines compare the ball before and after the curve. Only WHEN is used, not who. A break with nobody within TOUCH_REACH
             is a bounce or tracking noise and is ignored. Fitting lines over several frames,
             not comparing frame to frame, is what keeps tracking noise from looking like contact.
  control    possession.py names an owner (the ball moves with a player). Runs of control by the
             same team, with gaps up to CONTROL_GAP frames, are the evidence of WHO.

The rule (CHALLENGE.md: possession changes when the other team gains CONTROLLED possession; loose
and contested balls stay with the last team):
  - the team in possession changes when the rival CONTROLS the ball before the team in possession
    controls it again, and that control is real, which is either:
      * PLAYED: the rival's control ends in a collision -- a pass, a clearance -- and the team in
        possession does not control the ball within QUICK_S after it. Receiving and passing in half
        a second is common, so the length of the control alone cannot be the test; or
      * KEPT: the rival controls it for MIN_CONTROL_S in total (adding up its runs, so a team passing
        one-touch still counts), for when the collisions are not visible.
    A defender running next to a pass is neither: he never alters the ball, and when his "control"
    ends the other team traps it right there;
  - once confirmed, the change starts at a CHAIN OF COLLISIONS that leads to the rival's control:
    the last collision at the start of that control (the rival's trap), then back from collision
    to collision while each is less than CHAIN_GAP_S from the next -- a one-touch interception
    followed by the reception is one chain. A longer gap cuts the chain: what lies before it is
    another moment of the play, often a control of the team in possession that possession.py
    did not see (a player off camera, a hidden ball), and jumping over it would move the change
    seconds too early. The first link must itself be within CHAIN_GAP_S of the rival's control;
  - the chain never reaches the team's own kick (collisions within HALF_WINDOW + SKIP frames of
    the end of its last control) nor goes back more than MAX_LINK_S. No chain: the change starts
    with the rival's control.

So: interception -> change from the interception; deflection the team gets back -> no change; duel
then the rival keeps it -> change from the contact that starts the chain into the rival's control;
pass running next to a defender who never keeps it -> no change; the team receives a long pass
unseen and loses it seconds later -> the change starts where the rival's chain starts, not at the
team's unseen reception.

Keeper catches (keeper_catch.py). When the keeper catches the ball the tracking loses it in his
hands, often for seconds, and possession.py never names him the owner. When classify() is given the
keeper slots, every catch -- the ball vanishes at a keeper and comes back from him -- becomes a run
of control of the keeper's team over the whole gap, confirmed on its own (a ball that comes back
from the keeper's hands needs no second piece of evidence). Then the same rules apply: the change
starts at the chain of collisions into it, if any, or where the ball vanished.

Before anything, if classify() is given the keeper slots (`keepers`, from the metadata) and the
pitch length, each keeper is put back in his own team's slot (tracking_possession/fix_goalkeepers.py:
swapped, duplicated or crossed keeper slots), so a keeper's catch is credited to the right team.
Call it once per half FROM THE HALF'S FIRST FRAME: the kick-off there (everyone in his own half)
says which goal each team defends, so keeper slots crossed for the whole half are still caught.

Every threshold is physical (metres, m/s, seconds), to be reasoned about and checked on the video
-- not tuned against the public ground truth (info_hackathon/RULES.md).
"""

from dataclasses import dataclass, field, replace

import numpy as np

from tracking_possession.possession import PossessionConfig, classify as classify_base
from tracking_possession.fix_goalkeepers import fix_goalkeepers
from tracking_possession.keeper_catch import CatchConfig, detect_catches


@dataclass
class TouchConfig:
    """Thresholds of the collision pass, in metres, metres per second and seconds."""
    touch_reach: float = 2.0    # a collision needs some player this close to the break
    half_window: int = 6        # frames fitted on each side of a candidate collision (0.24 s)
    skip: int = 3               # frames left out on each side of the candidate: the ATD rounds a
                                # break off over a few frames, so the lines are fitted beyond the
                                # curve and compare the whole change, not a slice of it
    min_points: int = 4         # ball positions needed on EACH side to fit a line
    dv_min: float = 4.0         # velocity change (m/s) that friction and tracking noise cannot explain
    min_control_s: float = 1.0  # rival control, in total, that changes possession
                                # (CHALLENGE.md proxy: "a player keeps the ball for a few seconds")
    control_gap: int = 5        # frames without owner tolerated inside one run of control
    max_link_s: float = 3.0     # a collision older than this before the rival's control: another play
    quick_s: float = 0.4        # the team in possession controlling it this soon after a collision:
                                # the collision was its own trap, not a pass by the rival
    chain_gap_s: float = 1.0    # collisions closer than this are one chain of touches (a pass
                                # between two touches of a one-touch move takes less than a second)
    min_control_frames: int = 12   # a rival run of control shorter than this is not a control: it
                                  # neither changes possession ("played") nor adds up ("kept").
                                  # 0 = every run counts, as the first version: a receive-and-pass
                                  # is short and does change possession. (The short touch of the
                                  # conceding team right before a goal is dropped in goals.py.)
                                  # A keeper catch is exempt: it has its own evidence


@dataclass
class Collision:
    frame: int                  # index inside the processed block
    dv: float                   # |v_after - v_before|, m/s
    distance: float             # nearest player - ball, m (only to check it on the video)


@dataclass
class TouchInfo:
    """What the collision pass saw and decided, for debugging and the summary."""
    collisions: list = field(default_factory=list)    # [Collision]
    runs: list = field(default_factory=list)          # [(team, start, end)] runs of control,
                                                      # keeper catches included
    changes: list = field(default_factory=list)       # [(frame, team, reason)]
    rejected: int = 0                                 # rival control too short to change anything
    goalkeepers: object = None                        # GoalkeeperReport, when keepers were fixed
    catches: list = field(default_factory=list)       # [keeper_catch.Catch]
    catch_stats: dict = field(default_factory=dict)   # per ball gap: why it was / was not a catch
    all_runs: list = field(default_factory=list)      # [(team, start, end, kind)] as resolve()
                                                      # took them -- to resolve again with more
                                                      # runs (restarts.py)


# -- collisions --------------------------------------------------------------------------------

def _line_fits(ball, offsets, fps):
    """Least-squares straight line through ball[t + o] for o in offsets, for every t at once.

    Returns velocity (n, 2) in m/s, position at t (n, 2) and the number of points used (n,).
    NaN where fewer than two points exist. Missing positions are simply left out of the fit.
    """
    n = len(ball)
    shifted = np.full((len(offsets), n, 2), np.nan)
    for k, o in enumerate(offsets):
        if o >= 0:
            shifted[k, :n - o] = ball[o:]
        else:
            shifted[k, -o:] = ball[:n + o]
    present = np.isfinite(shifted[..., 0])
    tau = (np.asarray(offsets, float) / fps)[:, None] * present
    count = present.sum(axis=0)
    safe = np.maximum(count, 1)
    tau_mean = tau.sum(axis=0) / safe
    values = np.where(present[..., None], shifted, 0.0)
    x_mean = values.sum(axis=0) / safe[:, None]
    dt = np.where(present, tau - tau_mean, 0.0)
    sxx = (dt ** 2).sum(axis=0)
    sxy = (dt[..., None] * np.where(present[..., None], shifted - x_mean, 0.0)).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        velocity = sxy / sxx[:, None]
    velocity[(count < 2) | (sxx <= 0)] = np.nan
    position = x_mean - velocity * tau_mean[:, None]
    return velocity, position, count


def detect_collisions(players_s, ball, cfg, fps):
    """[Collision]: breaks in the ball's trajectory with some player near them.

    `ball` is the RAW ball in metres (smoothing would round the break off); players_s the smoothed
    players from possession.prepare().
    """
    h, g = cfg.half_window, cfg.skip
    v_before, p_before, n_before = _line_fits(ball, list(range(-g - h, -g + 1)), fps)
    v_after, p_after, n_after = _line_fits(ball, list(range(g, g + h + 1)), fps)
    dv = np.linalg.norm(v_after - v_before, axis=1)
    usable = (n_before >= cfg.min_points) & (n_after >= cfg.min_points) & np.isfinite(dv)
    dv = np.where(usable, dv, 0.0)

    collisions = []
    w = h + g                                           # one break spans this many frames each side
    for t in np.flatnonzero(dv >= cfg.dv_min):
        lo, hi = max(0, t - w), min(len(dv), t + w + 1)
        if dv[t] < dv[lo:hi].max() or (dv[lo:t] == dv[t]).any():
            continue                                    # one collision per break: the largest
        where = (p_before[t] + p_after[t]) / 2          # both lines extended to the break
        present = np.isfinite(players_s[t, :, 0])
        if not present.any():
            continue
        nearest = float(np.min(np.linalg.norm(players_s[t, present] - where, axis=1)))
        if nearest <= cfg.touch_reach:
            collisions.append(Collision(int(t), float(dv[t]), nearest))
    return collisions


# -- control -----------------------------------------------------------------------------------

def control_runs(results, teams, gap):
    """[(team, start, end)] runs of possession.py's owner by one team, end inclusive.

    A run survives up to `gap` frames without owner (free, flight, no ball) and ends as soon as the
    other team owns it.
    """
    runs = []
    for i, r in enumerate(results):
        if r.state != "possession" or teams[r.owner] not in ("H", "A"):
            continue
        team = teams[r.owner]
        if runs and runs[-1][0] == team and i - runs[-1][2] <= gap + 1:
            runs[-1][2] = i
        else:
            runs.append([team, i, i])
    return [tuple(run) for run in runs]


# -- resolution --------------------------------------------------------------------------------

def resolve(runs, collision_frames, n, cfg, fps):
    """Team in possession per frame from runs of control and collision frames.

    runs are (team, start, end) or (team, start, end, kind); kind "catch" (a keeper catch) is
    confirmed on its own, anything else needs to be played or kept.
    Returns (team per frame as a list of "H" / "A" / None, changes, rejected).
    """
    runs = sorted(((tuple(r) + ("control",))[:4] for r in runs), key=lambda r: r[1])
    min_control = int(round(cfg.min_control_s * fps))
    max_link = int(round(cfg.max_link_s * fps))
    quick = int(round(cfg.quick_s * fps))
    w = cfg.half_window + cfg.skip                  # how far a collision can sit from what caused it
    collision_frames = np.sort(np.asarray(collision_frames, dtype=int))

    def played(k, team):
        """Run k (the rival's) ends in a collision and `team` does not take the ball right after."""
        _, start, end, _ = runs[k]
        near_end = collision_frames[(collision_frames >= max(start, end - w)) &
                                    (collision_frames <= end + w)]
        if not len(near_end):
            return False
        hit = int(near_end[-1])
        for who, s, _, _ in runs[k + 1:]:
            if s > hit + quick:
                break
            if who == team:
                return False                        # the team trapped it: that was its collision
        return True

    chain_gap = int(round(cfg.chain_gap_s * fps))

    def change_start(last_end, rival_start):
        """Start of the chain of collisions that leads into the rival's control."""
        lo = max(last_end + w + 1, rival_start - max_link)   # never the team's own kick
        hi = rival_start + w                        # the rival's trap can land a bit later
        inside = collision_frames[(collision_frames >= lo) & (collision_frames <= hi)]
        if not len(inside) or rival_start - inside[-1] > chain_gap:
            return rival_start, "no collision seen"
        k = len(inside) - 1                         # the collision at the rival's control
        while k > 0 and inside[k] - inside[k - 1] <= chain_gap:
            k -= 1                                  # one more touch of the same chain
        links = len(inside) - k
        return int(min(inside[k], rival_start)), "collision ({} in chain)".format(links)

    team, last_end, changes, rejected = None, None, [], 0
    rival_first, rival_total = None, 0              # rival runs since the team last controlled it
    for k, (who, start, end, kind) in enumerate(runs):
        if team is None:                            # start of the half: first owner, as before
            team, last_end = who, end
            changes.append((start, who, "first owner"))
            continue
        if who == team:
            if rival_first is not None:
                rejected += 1                       # the rival never kept it: nothing changed
            last_end, rival_first, rival_total = end, None, 0
            continue
        if kind not in ("catch", "restart") and end - start + 1 < cfg.min_control_frames:
            continue                                # too short to be a control (a touch, a blip)
        if rival_first is None:
            rival_first = start
        rival_total += end - start + 1
        why = ("keeper catch" if kind == "catch" else
               "restart after a dead ball" if kind == "restart" else
               "played" if played(k, team) else
               "kept" if rival_total >= min_control else None)
        if why:
            if kind in ("catch", "restart") and rival_first == start:
                # a catch carries its own estimate of when the keeper got it (keeper_catch's
                # `since`), a restart is the restart: no chain of collisions before it -- that
                # chain is the OTHER team's
                frame, how = start, "its own frame"
            else:
                frame, how = change_start(last_end, rival_first)
            changes.append((frame, who, "rival control {}, start at {}".format(why, how)))
            team, last_end, rival_first, rival_total = who, end, None, 0

    labels = [None] * n
    for k, (start, who, _) in enumerate(changes):
        stop = changes[k + 1][0] if k + 1 < len(changes) else n
        labels[start:stop] = [who] * (stop - start)
    return labels, changes, rejected


# -- entry point -------------------------------------------------------------------------------

def classify(players, ball, teams, config=None, fps=25, touch_config=None, keepers=None,
             pitch_length=None, goalkeeper_config=None, catch_config=None, pitch_width=68.0,
             kickoff=None):
    """Same as possession.classify(), with `team` decided by collisions and control.

    keepers            {team: keeper slot index} (fix_goalkeepers.goalkeeper_slots(slot_info(...)))
                       to put the keepers right first; None leaves the players as given
    pitch_length       metres along x (games.GAMES[GAME]["pitch"][0]), needed with keepers
    goalkeeper_config  fix_goalkeepers.GoalkeeperConfig, or None for its defaults
    catch_config       keeper_catch.CatchConfig, or None for its defaults (used with keepers)
    pitch_width        metres along y (games.GAMES[GAME]["pitch"][1]): where the penalty areas are
    kickoff            index of the half's kick-off in the arrays (games.py's anchor): the goal
                       each team defends is measured on the formation there (fix_goalkeepers)

    Returns (results, players_s, player_v, ball_s, ball_v, info). `results` are possession.py's
    FrameResults with only `team` replaced, so state / owner / nearest still say what the frame-by-
    frame classifier saw. players_s are smoothed from the FIXED keepers; info.goalkeepers says
    what was fixed.
    """
    config = config or PossessionConfig()
    cfg = touch_config or TouchConfig()
    goalkeepers = None
    if keepers:
        if pitch_length is None:
            raise ValueError("pitch_length is needed to fix the goalkeepers")
        # one half from its first frame: its kick-off says which goal each team defends
        players, goalkeepers = fix_goalkeepers(players, keepers, pitch_length, goalkeeper_config,
                                               teams=teams, kickoff=kickoff)
    results, players_s, player_v, ball_s, ball_v = classify_base(players, ball, teams, config, fps)

    collisions = detect_collisions(players_s, ball, cfg, fps)
    runs = [run + ("control",) for run in control_runs(results, teams, cfg.control_gap)]
    # every collision as a kick: where, and the velocity of the line fitted after it (beyond the
    # part the ATD rounds off) -- how keeper_catch recognises a shot lost in flight
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
    # a catch is the keeper's team's control from `since` (how the ball vanished decides it)
    runs += [(c.team, c.start if c.since is None else c.since, c.end, "catch") for c in catches]
    labels, changes, rejected = resolve(runs, [c.frame for c in collisions], len(results), cfg, fps)

    results = [replace(r, team=team) for r, team in zip(results, labels)]
    all_runs = sorted(runs, key=lambda r: r[1])                     # with their kind
    runs = [run[:3] for run in all_runs]                            # (team, start, end), catches too
    info = TouchInfo(collisions=collisions, runs=runs, changes=changes, rejected=rejected,
                     goalkeepers=goalkeepers, catches=catches, catch_stats=catch_stats,
                     all_runs=all_runs)
    return results, players_s, player_v, ball_s, ball_v, info


def summarise(info, fps):
    """Print what the collision pass found."""
    reasons = {}
    for _, _, reason in info.changes:
        reasons[reason] = reasons.get(reason, 0) + 1
    print("collisions: {}   runs of control: {}   possession changes: {}  {}".format(
        len(info.collisions), len(info.runs), len(info.changes), reasons))
    print("rival control too short to change possession: {}".format(info.rejected))
    if info.catch_stats:
        print("ball gaps checked for keeper catches: {}".format(info.catch_stats))
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


def write_debug_csv(info, path, offset=0):
    """One row per collision, run of control and change, in video frames, to check on the video."""
    rows = [(c.frame, "collision", "", "", "{:.2f}".format(c.dv), "{:.2f}".format(c.distance), "")
            for c in info.collisions]
    catch_runs = {(c.team, c.start, c.end): c for c in info.catches}
    rows += [(s, "catch" if (team, s, e) in catch_runs else "control", team, e + offset, "", "",
              catch_runs[(team, s, e)].release if (team, s, e) in catch_runs else "")
             for team, s, e in info.runs]
    rows += [(f, "change", team, "", "", "", reason) for f, team, reason in info.changes]
    with open(path, "w") as handle:
        handle.write("frame,kind,team,end_frame,dv,distance,reason\n")
        for row in sorted(rows, key=lambda r: (r[0], r[1])):
            handle.write(",".join(str(v) for v in (row[0] + offset,) + row[1:]) + "\n")
    return path