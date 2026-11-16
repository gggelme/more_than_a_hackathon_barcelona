"""Who restarts the game after a dead ball: the team in possession from the restart on.

Pure logic on positions in METRES. CHALLENGE.md: "Whenever play restarts after a stoppage, there
is a clear possessor." The touch pass does not know it: after a goal it keeps the scoring team
until the other one controls the ball for a while. Here every stoppage's restart becomes a run of
control of the restarting team, confirmed on its own (like a keeper catch), so the touch pass gives
it the ball from the restart -- and from there the usual rules apply: if the other team wins it
back and controls it, possession changes again.

Who restarts:
  "kick-off rule"   a kick-off after a goal is taken by the team that CONCEDED (the laws of the
                    game): the other team than the scorer, when the stoppage knows who scored;
  "nearest player"  otherwise (a free kick, a throw-in, a goal kick, a cooling break): the player
                    nearest to the ball when it starts moving again, if within KICKER_REACH --
                    the one who takes it. The Smart Tagging often has no team for set pieces
                    (every throw-in is N/A), the tracking does;
  nobody            no rule and no player close enough: no run is added -- better than guessing.

Any dead-ball interval works, not only goals.py's: an object with `end` (the restart, the first
frame of live play, index of the processed block) and optionally `scorer` and `end_case`.

Thresholds are physical guesses, not tuned against the public ground truth (info_hackathon/RULES.md).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class RestartConfig:
    """Thresholds, in metres and frames."""
    kicker_reach: float = 3.0       # the player taking the restart is this close to the ball
    search_frames: int = 12         # looked for this many frames around the restart (~0.5 s)


@dataclass
class Restart:
    frame: int                      # the restart, index of the processed block
    team: str                       # "H" / "A"
    how: str                        # "kick-off rule" or "nearest player"


def restarter(stoppage, players, ball, teams, cfg=None):
    """(team, how) of whoever restarts after `stoppage`, or (None, None)."""
    cfg = cfg or RestartConfig()
    if stoppage.end >= len(ball):
        return None, None                           # it runs to the end of the half: no restart
    scorer = getattr(stoppage, "scorer", None)
    end_case = getattr(stoppage, "end_case", "") or ""
    if end_case.startswith("into "):
        return None, None                           # flows into the next stoppage: its restart
    if end_case.startswith("kick off") and scorer in ("H", "A"):
        return ("A" if scorer == "H" else "H"), "kick-off rule"
    teams = np.asarray(teams)
    best = None
    for t in range(max(0, stoppage.end - cfg.search_frames),
                   min(len(ball), stoppage.end + cfg.search_frames + 1)):
        if not np.all(np.isfinite(ball[t])):
            continue
        present = np.isfinite(players[t, :, 0])
        if not present.any():
            continue
        d = np.linalg.norm(players[t, present] - ball[t], axis=1)
        k = int(np.argmin(d))
        if d[k] <= cfg.kicker_reach and (best is None or d[k] < best[0]):
            best = (float(d[k]), teams[np.flatnonzero(present)[k]])
    return (best[1], "nearest player") if best is not None else (None, None)


def restarts(stoppages, players, ball, teams, cfg=None):
    """[Restart] for the stoppages whose restarter is known."""
    out = []
    for s in stoppages:
        team, how = restarter(s, players, ball, teams, cfg)
        if team is not None:
            out.append(Restart(int(s.end), team, how))
    return out


def restart_runs(found):
    """Runs of control for a touch pass's resolve(): (team, frame, frame, "restart")."""
    return [(r.team, r.frame, r.frame, "restart") for r in found]
