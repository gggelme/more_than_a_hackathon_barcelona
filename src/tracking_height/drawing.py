"""Drawing for the ball-height visualizer. Only drawing -- the logic is in ball_height.py.

Everything is drawn onto a full-resolution BGR frame in place:
  - the two goals in 3D (posts and the 2.44 m crossbar), to check the recovered camera by eye:
    they must sit on the real goals whenever the camera points at one;
  - the ATD ball as a yellow circle on the grass, and the vertical rising from it with a tick every
    metre and labelled heights -- an airborne ball should sit on this line;
  - every YOLO box tested against the vertical (green if consistent, grey if not), labelled with its
    height and residual; the chosen one in blue, with a magenta dot where it meets the vertical;
  - the frame's state in a coloured box, and a bottom strip with the numbers.
"""

import cv2
import numpy as np

from .ball_height import vertical
from .camera_from_homography import project

# Params
FONT = cv2.FONT_HERSHEY_SIMPLEX
STRIP_HEIGHT = 50
LABELLED_HEIGHTS = (1, 2, 3, 5, 10, 20, 30)
GOAL_WIDTH, CROSSBAR = 7.32, 2.44   # metres

# BGR
ATD_BALL = (0, 255, 255)            # yellow
VERTICAL = (0, 200, 255)            # orange
GOAL = (255, 255, 255)              # white
FIT_OK = (0, 255, 0)                # green: consistent with the vertical
FIT_BAD = (150, 150, 150)           # grey: not consistent
CHOSEN = (255, 120, 0)              # blue
MEETING = (255, 0, 255)             # magenta: where the chosen detection meets the vertical
STATE_COLOURS = {"ground": (0, 160, 0), "air": (200, 80, 0), "inconsistent": (0, 0, 200),
                 "no_detection": (90, 90, 90), "no_camera": (60, 60, 60), "no_atd": (60, 60, 60)}


def _point(u, v):
    return int(round(float(u))), int(round(float(v)))


def _visible(u, v, width, height, margin=2000):
    """Finite and not absurdly far off screen (cv2 cannot draw near-infinite coordinates)."""
    return (np.isfinite(u) and np.isfinite(v)
            and -margin < u < width + margin and -margin < v < height + margin)


def draw_goals(image, camera):
    """Posts and crossbar of both goals, projected with the full camera."""
    height, width = image.shape[:2]
    length, pitch_width = camera.pitch
    for goal_x in (0.0, length):
        left, right = pitch_width / 2.0 - GOAL_WIDTH / 2.0, pitch_width / 2.0 + GOAL_WIDTH / 2.0
        corners = [(left, 0.0), (left, CROSSBAR), (right, CROSSBAR), (right, 0.0)]
        points = [project(camera, goal_x, y, z) for y, z in corners]
        if all(_visible(u, v, width, height) for u, v in points):
            pixels = np.array([_point(u, v) for u, v in points], np.int32)
            cv2.polylines(image, [pixels], False, GOAL, 2, cv2.LINE_AA)


def draw_vertical(image, camera, x_m, y_m, z_max):
    """The ATD circle on the grass and the vertical above it, ticked every metre."""
    height, width = image.shape[:2]
    us, vs = vertical(camera, x_m, y_m, np.arange(0.0, z_max + 0.01, 1.0))
    if not all(_visible(u, v, width, height) for u, v in zip(us, vs)):
        return
    pixels = np.array([_point(u, v) for u, v in zip(us, vs)], np.int32)
    cv2.polylines(image, [pixels], False, VERTICAL, 2, cv2.LINE_AA)
    for metres, point in enumerate(pixels):
        cv2.circle(image, tuple(point), 3, VERTICAL, -1, cv2.LINE_AA)
        if metres in LABELLED_HEIGHTS:
            cv2.putText(image, "{} m".format(metres), (point[0] + 8, point[1] + 5), FONT, 0.5,
                        VERTICAL, 1, cv2.LINE_AA)
    cv2.circle(image, tuple(pixels[0]), 16, ATD_BALL, 2, cv2.LINE_AA)


def draw_fits(image, camera, x_m, y_m, fits, chosen):
    """Every box tested against the vertical, labelled 'z / residual'; the chosen one stands out."""
    for fit in fits:
        x1, y1, x2, y2 = (int(round(v)) for v in fit.box[:4])
        colour = CHOSEN if fit is chosen else (FIT_OK if fit.consistent else FIT_BAD)
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 3 if fit is chosen else 2)
        cv2.putText(image, "z {:.1f} / {:.1f} m".format(fit.z, fit.residual_m), (x1, y1 - 6),
                    FONT, 0.55, colour, 2, cv2.LINE_AA)
    if chosen is not None:
        u, v = project(camera, x_m, y_m, chosen.z)
        if _visible(u, v, image.shape[1], image.shape[0]):
            cv2.circle(image, _point(u, v), 6, MEETING, -1, cv2.LINE_AA)
            cv2.line(image, _point(chosen.u, chosen.v), _point(u, v), MEETING, 1, cv2.LINE_AA)


def draw_region(image, region):
    """Where YOLO looked this frame."""
    if region is None:
        return
    x0, y0, (w, h) = region
    cv2.rectangle(image, (x0, y0), (x0 + w, y0 + h), (255, 255, 255), 1)


def draw_state(image, state, chosen):
    """Top-left box: the frame's state, and the height when there is one."""
    text = state.upper() if chosen is None else "{}  z = {:.1f} m".format(state.upper(), chosen.z)
    (text_width, _), _ = cv2.getTextSize(text, FONT, 1.1, 3)
    colour = STATE_COLOURS.get(state, (60, 60, 60))
    cv2.rectangle(image, (15, 15), (45 + text_width, 80), colour, -1)
    cv2.rectangle(image, (15, 15), (45 + text_width, 80), (255, 255, 255), 2)
    cv2.putText(image, text, (30, 60), FONT, 1.1, (255, 255, 255), 3, cv2.LINE_AA)


def draw_strip(image, frame, chosen, camera, fps, infer_ms):
    """Bottom strip: frame, video time, height and residual of the chosen box, camera, timing."""
    height, width = image.shape[:2]
    cv2.rectangle(image, (0, height - STRIP_HEIGHT), (width, height), (0, 0, 0), -1)
    seconds = frame / fps
    fit = ("z {:.1f} m  residual {:.2f} m ({:.0f} px)".format(chosen.z, chosen.residual_m,
                                                             chosen.residual_px)
           if chosen is not None else "z -  residual -")
    cam = "f {:.0f}  cam h {:.1f} m".format(camera.f, camera.height) if camera else "f -  cam h -"
    text = "frame {}  {:02d}:{:05.2f}  |  {}  |  {}  |  {:.0f} ms".format(
        frame, int(seconds // 60), seconds % 60, fit, cam, infer_ms)
    cv2.putText(image, text, (15, height - 17), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def draw_frame(image, frame, camera, x_m, y_m, fits, chosen, region, state, z_max, fps, infer_ms):
    """Everything for one frame. camera / x_m may be None (no homography / no ATD ball)."""
    if camera is not None:
        draw_goals(image, camera)
        if x_m is not None:
            draw_region(image, region)
            draw_vertical(image, camera, x_m, y_m, z_max)
            draw_fits(image, camera, x_m, y_m, fits, chosen)
    draw_state(image, state, chosen)
    draw_strip(image, frame, chosen, camera, fps, infer_ms)
