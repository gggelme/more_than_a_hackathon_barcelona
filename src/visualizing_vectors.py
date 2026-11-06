"""Draw the direction of movement of every tracked player and of the ball, on the tactical video.

For each tracked slot and for the ball, the velocity is the change of position over the last
VELOCITY_FRAMES frames, in pitch coordinates (the ATD's normalised 0-1). The arrow starts at the
current position and ends where that velocity would take it PLAYER_ARROW_SECONDS ahead (the ball:
BALL_ARROW_SECONDS, much longer, so its direction is easy to see). Both ends are projected with the
frame's homography, so the arrows follow the perspective of the pitch like the dots do.

A slot or the ball only gets an arrow when it is present on both frames being compared. A slot is a
tracklet, not a person (see info_hackathon/DATA.md), so an identity switch can show up as a sudden
jump.

    python src/visualizing_vectors.py

What to show is set at the bottom of this file, in the `if __name__ == '__main__':` block.
Window keys: space pauses, q or Esc quits. With OUT set, the video is written to that .mp4 instead of
shown (e.g. data/derived/vectors.mp4 -- data/ is gitignored, and clips of the match must not be
published).
"""

import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, half_video_blocks, require  # noqa: E402
from starter.load_homography import load as load_homography, project  # noqa: E402
from visualizing_tracking import COLOURS, FONT, inside, read_atd, slot_info  # noqa: E402

# Params
PITCH_LENGTH, PITCH_WIDTH = GAMES[GAME]["pitch"]   # metres, only for the speeds printed
DOT_RADIUS = 5
STRIP_HEIGHT = 50
MIN_SPEED = 0.2                     # m/s: below this nothing is drawn but the dot (standing still)


def velocity(now, before, frames, fps):
    """(vx, vy) in normalised pitch units per second, or None if either position is missing."""
    x, y = now[:2]
    px, py = before[:2]
    if not all(np.isfinite(v) for v in (x, y, px, py)):
        return None
    return (x - px) * fps / frames, (y - py) * fps / frames


def speed_ms(v):
    """Speed in m/s of a normalised velocity."""
    return float(np.hypot(v[0] * PITCH_LENGTH, v[1] * PITCH_WIDTH))


def draw_arrow(image, matrix, x, y, v, seconds, colour, thickness):
    """Arrow from (x, y) to where velocity v takes it in `seconds`, projected onto the image."""
    height, width = image.shape[:2]
    u0, v0 = project(matrix, x, y)
    if not inside(u0, v0, width, height):
        return
    u1, v1 = project(matrix, x + v[0] * seconds, y + v[1] * seconds)
    if not (np.isfinite(u1) and np.isfinite(v1)):
        return
    start, end = (int(round(u0)), int(round(v0))), (int(round(u1)), int(round(v1)))
    cv2.arrowedLine(image, start, end, (0, 0, 0), thickness + 2, cv2.LINE_AA, tipLength=0.25)
    cv2.arrowedLine(image, start, end, colour, thickness, cv2.LINE_AA, tipLength=0.25)


def draw_frame(image, matrix, atd, frame, info, velocity_frames, player_seconds, ball_seconds,
               fps):
    """Dots and direction arrows of every slot and the ball, plus the bottom strip."""
    height, width = image.shape[:2]
    slots, ball = atd[frame]
    past_slots, past_ball = atd.get(frame - velocity_frames, (None, None))

    for index, ((x, y, _), (team, is_goalkeeper)) in enumerate(zip(slots, info)):
        if np.isnan(x):
            continue
        colour = COLOURS[(team, "gk")] if is_goalkeeper else COLOURS[team]
        u, v = project(matrix, x, y)
        if not inside(u, v, width, height):
            continue
        v_slot = velocity(slots[index], past_slots[index], velocity_frames, fps) \
            if past_slots is not None else None
        if v_slot is not None and speed_ms(v_slot) >= MIN_SPEED:
            draw_arrow(image, matrix, x, y, v_slot, player_seconds, colour, 2)
        cv2.circle(image, (int(round(u)), int(round(v))), DOT_RADIUS, colour, -1)
        cv2.circle(image, (int(round(u)), int(round(v))), DOT_RADIUS, (0, 0, 0), 1)

    ball_speed = None
    bx, by, _ = ball
    if not np.isnan(bx):
        v_ball = velocity(ball, past_ball, velocity_frames, fps) if past_ball is not None else None
        if v_ball is not None:
            ball_speed = speed_ms(v_ball)
            if ball_speed >= MIN_SPEED:
                draw_arrow(image, matrix, bx, by, v_ball, ball_seconds, COLOURS["ball"], 4)
        u, v = project(matrix, bx, by)
        if inside(u, v, width, height):
            cv2.circle(image, (int(round(u)), int(round(v))), DOT_RADIUS + 4, COLOURS["ball"], 2)

    cv2.rectangle(image, (0, height - STRIP_HEIGHT), (width, height), (0, 0, 0), -1)
    seconds = frame / fps
    text = ("frame {}  {:02d}:{:05.2f}   |   ball speed: {}   |   velocity over {} frames   |   "
            "arrows: players {:g} s ahead, ball {:g} s ahead").format(
        frame, int(seconds // 60), seconds % 60,
        "-" if ball_speed is None else "{:.1f} m/s".format(ball_speed), velocity_frames,
        player_seconds, ball_seconds)
    cv2.putText(image, text, (15, height - 17), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def run(start, seconds=30, velocity_frames=5, player_arrow_seconds=1.0, ball_arrow_seconds=4.0,
        scale=0.75, out=None):
    """Play (or write) video frames [start, start + seconds) with the movement arrows drawn on top.

    velocity_frames       compare each position with the one this many frames earlier
                          (1 = frame to frame, noisier; 5 = 0.2 s, steadier)
    player_arrow_seconds  arrow length of a player: where it would be this many seconds ahead
    ball_arrow_seconds    the same for the ball, much longer so it stands out
    out                   an .mp4 path to write instead of opening a window, or None
    """
    video_path, meta_path, raw_path = require(["video", "atd_meta", "atd_raw"])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("could not open {}".format(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    stop = start + int(round(seconds * fps))

    print("reading homography ...")
    matrices, _, valid = load_homography()
    first = max(0, start - velocity_frames)   # earlier frames, for the first velocities
    print("reading ATD frames {}..{} ...".format(first, stop - 1))
    atd = read_atd(raw_path, first, stop)
    info = slot_info(meta_path)

    writer = None
    if out:
        out = os.path.abspath(out)
        os.makedirs(os.path.dirname(out), exist_ok=True)   # OpenCV does not create folders
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                 (int(1920 * scale), int(1080 * scale)))
        if not writer.isOpened():
            raise IOError("could not open {} for writing".format(out))
        print("writing {} ...".format(out))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    delay_ms = max(1, int(1000 / fps))
    paused = False
    frame = start
    while frame < stop:
        if not paused:
            ok, image = cap.read()
            if not ok:
                break
            if valid[frame]:
                draw_frame(image, matrices[frame], atd, frame, info, velocity_frames,
                           player_arrow_seconds, ball_arrow_seconds, fps)
            shown = cv2.resize(image, None, fx=scale, fy=scale)
            frame += 1

        if writer is not None:
            writer.write(shown)
            continue

        cv2.imshow("movement vectors", shown)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused

    cap.release()
    if writer is not None:
        writer.release()
        print("wrote {}".format(out))
    cv2.destroyAllWindows()


if __name__ == '__main__':
    first_half_kickoff = half_video_blocks(GAME)[0][1]     # 11185, first graded frame

    # -- what to show ------------------------------------------------------------------------
    START = first_half_kickoff      # video frame to start from (minute * 1500 + second * 25)
    SECONDS = 30                    # how long to play
    VELOCITY_FRAMES = 5             # velocity over the last N frames (5 = 0.2 s)
    PLAYER_ARROW_SECONDS = 1.0      # player arrow = where it would be 1 s ahead
    BALL_ARROW_SECONDS = 4.0        # ball arrow, much longer to see its direction well
    SCALE = 0.75                    # window / video size relative to 1920x1080
    OUT = None                      # e.g. "data/derived/vectors.mp4" to write instead of showing

    run(start=START, seconds=SECONDS, velocity_frames=VELOCITY_FRAMES,
        player_arrow_seconds=PLAYER_ARROW_SECONDS, ball_arrow_seconds=BALL_ARROW_SECONDS,
        scale=SCALE, out=OUT)
