"""Put each goalkeeper back in his own team's slot before classifying possession. Pure logic.

Why. The ATD has one slot per team flagged as the goalkeeper (position_type "Goalkeeper" in the
metadata). Slots are tracklets, not people (info_hackathon/DATA.md), and the two keepers'
slots sometimes swap or both lock onto the same keeper. A home slot standing on the away keeper
makes the classifier give the ball to the home team when the away keeper catches it.

Which slots are the keepers is NOT hard-coded: the caller passes the slots the metadata flags
(goalkeeper_slots(slot_info(...))). Which goal each team defends is not either: it is measured,
per half (the teams change ends at half time), from the data -- and NOT from the keeper slots
themselves, which are exactly what may be wrong (swapped for a whole half, they would vouch for
the swap):

  DEFENDED GOALS, from the KICK-OFF. At a kick-off every player stands in his own half (Laws of
  the Game, law 8), and each half starts with one. Over KICKOFF_FRAMES centred on the kick-off
  (`kickoff`: games.py's anchor, "kickoff" / "kickoff_2nd"), the team whose OUTFIELD players
  (keeper slots left out) have the smaller mean x defends the goal at x = 0, the other the goal
  at x = pitch length. Not the half's first frames: its video block (games.half_video_blocks) may
  start minutes before the kick-off, with the players still walking on -- the second half's starts
  2 minutes early, and the formation measured there came out reversed (38 minutes of keepers
  "swapped" into the wrong goals). Without `kickoff`, the stretch's first frames are used.
  The teams change ends at half time, so the two halves must be mirrored (mirrored()).
  FALLBACK, only if the kick-off is not tracked: the keeper slot with the smaller median x over
  the stretch defends x = 0 -- right as long as the slots are right most of the time.

Then, frame by frame, a keeper slot is on the WRONG SIDE when it stands in the OTHER team's half
of the pitch (within WRONG_SIDE_M of the goal the other team defends; None = half the pitch). A
keeper does not cross the halfway line, so a keeper slot there is tracking someone else -- most
likely the other keeper:
  1. SWAP     both keeper slots on the wrong side: their positions are swapped;
  2. DISABLE  one on the wrong side while the other team's keeper slot is tracked on its own side
              -- both channels tracked, at any distance (on the same keeper, or one on a player):
              the wrong one is set to missing. Where its own keeper is, is not known, and nothing
              is made up;
  3. MOVE     one on the wrong side and the other team's keeper slot missing: it was tracking the
              other keeper, so the position moves to that slot.
The price: a keeper really going up for a last-minute corner is disabled for those frames.

Use it per half (games.half_video_blocks) on the positions in METRES, before possession.classify()
or any variant such as possession_touch_2.classify(). With `teams` and a stretch that starts at
the half's first frame, the goals come from its kick-off; otherwise compute them with
kickoff_goals() on the half's first frames and pass them as `goals`:

    keepers = goalkeeper_slots(slot_info(meta_path))            # from the metadata, per team
    players, report = fix_goalkeepers(players, keepers, GAMES[GAME]["pitch"][0], teams=teams,
                                      kickoff=kickoff_video_frame - half_first)

Every threshold is physical (metres), not tuned against the public ground truth
(info_hackathon/RULES.md).
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GoalkeeperConfig:
    """Thresholds, in metres and frames."""
    wrong_side_m: float = None      # a keeper slot this close to the OTHER team's goal is on the
                                    # wrong side; None = half the pitch (the other team's half)
    kickoff_frames: int = 25        # one second around the kick-off: everyone in his own half
    min_kickoff_points: int = 25    # outfield positions of EACH team needed to trust the kick-off


@dataclass
class GoalkeeperReport:
    """What fix_goalkeepers found and changed in one stretch."""
    goals: dict = field(default_factory=dict)       # {team: x of the goal it defends, metres}
    source: str = None                              # "kick-off", "given" or "keeper medians"
    medians: dict = field(default_factory=dict)     # {team: median x of its keeper slot}
    swapped: int = 0                                # frames where the two keeper slots were swapped
    disabled: int = 0                               # frames where a wrong-side keeper slot was
                                                    # disabled (the other one tracked at home)
    moved: int = 0                                  # frames where a keeper moved to the other slot


def goalkeeper_slots(info):
    """{team: slot index} of the keeper slots, from slot_info()'s [(team, is_goalkeeper)].

    One slot per team is expected; if the metadata flags several, the first one is used.
    """
    slots = {}
    for slot, (team, is_goalkeeper) in enumerate(info):
        if is_goalkeeper and team not in slots:
            slots[team] = slot
    return slots


def kickoff_goals(players, teams, keepers, pitch_length, min_points=25):
    """{team: goal x} from a kick-off formation, or {} if it is not tracked well enough.

    players are the first frames of a half, (n, slots, 2) in metres; teams the team of every slot.
    Only OUTFIELD slots count (the keeper slots are what may be wrong); the team whose players
    have the smaller mean x defends x = 0. Each team needs min_points tracked positions.
    """
    teams = np.asarray(teams)
    outfield = np.ones(len(teams), bool)
    outfield[list(keepers.values())] = False
    means = {}
    for team in keepers:
        x = players[:, (teams == team) & outfield, 0]
        x = x[np.isfinite(x)]
        if x.size < min_points:
            return {}
        means[team] = float(np.mean(x))
    if len(means) != 2:
        return {}
    low, high = sorted(means, key=means.get)
    return {low: 0.0, high: float(pitch_length)}


def mirrored(goals_a, goals_b):
    """True if two halves' {team: goal x} are each other's mirror (the teams change ends at half
    time); None if either is incomplete."""
    if len(goals_a) != 2 or len(goals_b) != 2 or set(goals_a) != set(goals_b):
        return None
    return all(goals_a[team] != goals_b[team] for team in goals_a)


def defended_goals(players, keepers, pitch_length):
    """({team: goal x}, {team: median x}) for one stretch (one half), in metres -- the FALLBACK
    from the keeper slots' medians, for when the kick-off is not available.

    players  (n, slots, 2) in metres, NaN where missing
    keepers  {team: keeper slot index}, from goalkeeper_slots()
    """
    medians = {}
    for team, slot in keepers.items():
        x = players[:, slot, 0]
        x = x[np.isfinite(x)]
        if len(x):
            medians[team] = float(np.median(x))
    teams = list(keepers)
    if len(medians) == 2:
        low, high = sorted(medians, key=medians.get)
        goals = {low: 0.0, high: float(pitch_length)}
    elif len(medians) == 1:
        (team, median), = medians.items()
        goal = 0.0 if median < pitch_length / 2.0 else float(pitch_length)
        goals = {team: goal}
        goals.update({other: float(pitch_length) - goal for other in teams if other != team})
    else:
        goals = {}
    return goals, medians


def fix_goalkeepers(players, keepers, pitch_length, config=None, teams=None, goals=None,
                    kickoff=None):
    """(players with the keepers put right, GoalkeeperReport) for one stretch (one half).

    players       (n, slots, 2) in metres, NaN where missing -- not modified, a copy is returned
    keepers       {team: keeper slot index}, from goalkeeper_slots(); needs both teams
    pitch_length  metres along x (games.GAMES[GAME]["pitch"][0])
    teams         the team of every slot: the goals are taken from the kick-off formation
    kickoff       index in the stretch of the half's kick-off (games.py's anchor): the formation
                  is measured on KICKOFF_FRAMES centred on it; None (or outside the stretch) = the
                  stretch's first frames -- then it must start AT the kick-off
    goals         {team: goal x} already known (e.g. kickoff_goals() on the kick-off frames)
    Order: `goals` if given, else the kick-off (with `teams`), else the keeper slots' medians.
    """
    config = config or GoalkeeperConfig()
    fixed = players.copy()
    fallback, medians = defended_goals(players, keepers, pitch_length)
    if goals:
        source = "given"
    else:
        lo = 0
        if kickoff is not None and 0 <= kickoff < len(players):
            lo = max(0, int(kickoff) - config.kickoff_frames // 2)
        goals = {} if teams is None else kickoff_goals(
            players[lo:lo + config.kickoff_frames], teams, keepers, pitch_length,
            config.min_kickoff_points)
        source = "kick-off" if goals else "keeper medians"
        goals = goals or fallback
    report = GoalkeeperReport(goals=dict(goals), source=source, medians=medians)
    if len(keepers) != 2 or len(goals) != 2:
        return fixed, report                    # nothing to compare against

    (team_a, slot_a), (team_b, slot_b) = keepers.items()
    a, b = players[:, slot_a], players[:, slot_b]
    seen_a, seen_b = np.all(np.isfinite(a), axis=1), np.all(np.isfinite(b), axis=1)
    # on the wrong side: within wrong_side_m of the goal the OTHER team defends
    zone = pitch_length / 2.0 if config.wrong_side_m is None else config.wrong_side_m
    a_wrong = seen_a & (np.abs(a[:, 0] - goals[team_b]) < zone)
    b_wrong = seen_b & (np.abs(b[:, 0] - goals[team_a]) < zone)

    # 1. both on the wrong side: crossed
    swap = a_wrong & b_wrong
    fixed[swap, slot_a], fixed[swap, slot_b] = b[swap], a[swap]

    # 2. one on the wrong side, the other tracked at home: two channels, the wrong one goes
    disable_a = a_wrong & seen_b & ~b_wrong
    disable_b = b_wrong & seen_a & ~a_wrong
    fixed[disable_a, slot_a] = np.nan
    fixed[disable_b, slot_b] = np.nan

    # 3. one on the wrong side, the other team's keeper slot empty: it was following him
    move_a = a_wrong & ~seen_b
    move_b = b_wrong & ~seen_a
    fixed[move_a, slot_b], fixed[move_a, slot_a] = a[move_a], np.nan
    fixed[move_b, slot_a], fixed[move_b, slot_b] = b[move_b], np.nan

    report.swapped = int(swap.sum())
    report.disabled = int((disable_a | disable_b).sum())
    report.moved = int((move_a | move_b).sum())
    return fixed, report
