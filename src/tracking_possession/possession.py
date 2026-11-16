"""Who has the ball, frame by frame, from distance, relative velocity and direction -- not distance
alone. Pure logic: no video, no files.

Inputs are the ATD positions in METRES (pitch coordinates: the ATD's normalised 0-1 times the pitch
size), so perspective plays no part here -- the homography is only needed to draw.

Per frame, every slot within D_MAX of the ball gets a score

    s = -(d / d0)^2  -  (v_rel / v0)^2  +  w_dir * cos(theta)  +  stick

  d       distance player - ball (also the filter: farther than D_MAX is not a candidate);
  v_rel   |v_ball - v_player|: small when the ball moves WITH the player (dribbling);
          left out when either velocity is unknown;
  theta   angle between the two velocities; counted only when both move (a still player or ball
          has no direction);
  stick   hysteresis: STICK_SLOT for the slot that had the ball on the previous frame, STICK_TEAM
          for any other slot of that team. The team bonus matters because a slot is a tracklet, not
          a person: an identity switch hands the same player a new slot.

The owner is the best-scoring candidate if it reaches MIN_SCORE. Otherwise the ball is:
  "flight"  moving faster than FLIGHT_SPEED with nobody moving with it (v_rel < v0) -- a pass or
            a shot in the air or along the ground;
  "free"    nobody close or convincing enough;
  "no_ball" the ATD has no ball this frame: the previous owner is kept, nothing can be measured.

`team` is who is in possession for the challenge's rules: the last team that had an owner. A free
ball or a ball in flight stays with it (CHALLENGE.md: possession stays with the last team that had
it until the other team controls it). It is never a dead ball -- that is a different problem.

The old method is kept alongside: `nearest` is simply the closest slot to the ball.

Weights and thresholds are physical guesses (metres, m/s), to be reasoned about and checked on the
video -- not tuned against the public ground truth (info_hackathon/RULES.md).
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PossessionConfig:
    """Every weight and threshold, in metres and metres per second."""
    smooth_frames: int = 5          # centred moving average on positions before differentiating
    velocity_frames: int = 4        # velocity = change over this many frames, centred
    d_max: float = 3.0              # farther than this from the ball: not a candidate
    d0: float = 1.5                 # distance scale of the score
    v0: float = 2.0                 # relative-speed scale of the score
    w_dir: float = 0.5              # weight of the direction alignment
    moving: float = 0.5             # below this speed a velocity has no direction
    stick_slot: float = 1.0         # bonus for the slot that had the ball on the previous frame
    stick_team: float = 0.5         # bonus for the other slots of that team
    min_score: float = -2.0         # the best candidate must reach this to be the owner
    flight_speed: float = 10.0      # a ball faster than this, moving with nobody, is in flight


@dataclass
class FrameResult:
    """What the classifier decided on one frame."""
    state: str                      # "possession", "free", "flight" or "no_ball"
    owner: int = None               # slot index (0-based) of the owner, or None
    team: str = None                # team in possession for the rules ("H", "A" or None yet)
    nearest: int = None             # slot index of the closest player (the old method)
    nearest_distance: float = None  # metres
    ball_speed: float = None        # m/s
    best_score: float = None
    scores: dict = field(default_factory=dict)   # {slot index: score} of the candidates


# -- preprocessing -----------------------------------------------------------------------------

def smooth(track, window):
    """Centred moving average of a (n, ..., 2) track, ignoring NaN; a missing frame stays missing.

    Positions are only averaged with neighbours that exist, so a gap is never filled in -- the ball
    is missing on 41% of graded frames, and inventing it would invent velocities too.
    """
    if window <= 1:
        return track.copy()
    half = window // 2
    present = np.isfinite(track[..., 0])
    values = np.where(present[..., None], track, 0.0)
    padded = np.pad(values, [(half, half)] + [(0, 0)] * (track.ndim - 1))
    counts = np.pad(present.astype(float), [(half, half)] + [(0, 0)] * (present.ndim - 1))
    total = sum(padded[i:i + len(track)] for i in range(window))
    n = sum(counts[i:i + len(track)] for i in range(window))
    out = total / np.maximum(n, 1)[..., None]
    out[~present] = np.nan
    return out


def velocities(track, frames, fps):
    """Centred velocity (m/s) over `frames` frames: (p[t + h] - p[t - h]) / (2 h / fps)."""
    half = max(1, frames // 2)
    out = np.full_like(track, np.nan)
    out[half:-half] = (track[2 * half:] - track[:-2 * half]) * fps / (2 * half)
    return out


def prepare(players, ball, config, fps):
    """Smoothed positions and velocities of every slot and the ball, all in metres and m/s.

    players is (n, slots, 2), ball is (n, 2), both in metres with NaN where missing.
    """
    players_s = smooth(players, config.smooth_frames)
    ball_s = smooth(ball, config.smooth_frames)
    return (players_s, velocities(players_s, config.velocity_frames, fps),
            ball_s, velocities(ball_s, config.velocity_frames, fps))


# -- classification ----------------------------------------------------------------------------

class PossessionTracker:
    """Frame-by-frame owner with hysteresis. Feed frames in order with step()."""

    def __init__(self, teams, config):
        self.teams = teams              # team ("H" / "A") of every slot index
        self.config = config
        self.owner = None               # slot that had the ball on the previous frame
        self.team = None                # team in possession for the rules

    def step(self, players, player_v, ball, ball_v):
        """FrameResult for one frame. players / player_v are (slots, 2), ball / ball_v are (2,)."""
        c = self.config
        if not np.all(np.isfinite(ball)):
            return FrameResult("no_ball", owner=self.owner, team=self.team)

        present = np.isfinite(players[:, 0])
        distances = np.full(len(players), np.inf)
        distances[present] = np.linalg.norm(players[present] - ball, axis=1)
        nearest = int(np.argmin(distances)) if present.any() else None

        ball_moves = np.all(np.isfinite(ball_v))
        ball_speed = float(np.linalg.norm(ball_v)) if ball_moves else None

        scores, slow_relative = {}, False
        for slot in np.flatnonzero(present & (distances <= c.d_max)):
            score = -(distances[slot] / c.d0) ** 2
            v_player = player_v[slot]
            if ball_moves and np.all(np.isfinite(v_player)):
                v_rel = float(np.linalg.norm(ball_v - v_player))
                score -= (v_rel / c.v0) ** 2
                slow_relative |= v_rel < c.v0
                speed_player = float(np.linalg.norm(v_player))
                if ball_speed > c.moving and speed_player > c.moving:
                    score += c.w_dir * float(ball_v @ v_player) / (ball_speed * speed_player)
            if slot == self.owner:
                score += c.stick_slot
            elif self.teams[slot] == self.team:
                score += c.stick_team
            scores[int(slot)] = float(score)

        best = max(scores, key=scores.get) if scores else None
        best_score = scores[best] if best is not None else None
        result = FrameResult("free", nearest=nearest, ball_speed=ball_speed, best_score=best_score,
                             scores=scores,
                             nearest_distance=None if nearest is None else float(distances[nearest]))

        if ball_speed is not None and ball_speed > c.flight_speed and not slow_relative:
            result.state, self.owner = "flight", None
        elif best is not None and best_score >= c.min_score:
            result.state, self.owner, self.team = "possession", best, self.teams[best]
        else:
            self.owner = None
        result.owner, result.team = self.owner, self.team
        return result


def classify(players, ball, teams, config, fps):
    """[FrameResult] for every frame of (n, slots, 2) players and (n, 2) ball, in metres."""
    players_s, player_v, ball_s, ball_v = prepare(players, ball, config, fps)
    tracker = PossessionTracker(teams, config)
    return [tracker.step(players_s[i], player_v[i], ball_s[i], ball_v[i])
            for i in range(len(players))], players_s, player_v, ball_s, ball_v
