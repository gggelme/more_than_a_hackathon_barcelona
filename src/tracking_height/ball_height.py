"""Ball height from a detection, and the frame's state. Pure logic -- no video, no files.

Given, for one frame, the Camera (camera_from_homography.decompose), the ATD ball (x, y) in METRES
-- approximately the point of the grass under the ball -- and YOLO boxes in full-resolution pixels:

The ATD (x, y) defines a vertical line rising from the grass; the camera turns it into a segment in
the image. A detection (u, v) is a ray from the camera. Where the ray meets the vertical is the ball,
and the height of that meeting point is z. In the image: slide a point up the projected vertical
until it is as close as possible to (u, v). What is left over -- the RESIDUAL, the distance from the
detection to the vertical -- says whether the detection fits at all:
  - small residual, z near 0   -> the ball is on the ground;
  - small residual, z above it -> the ball is in the air, at about z;
  - large residual             -> the detection is not this ball, so no z is taken from it.
Residuals and thresholds are in METRES at the ball: a pixel is a different distance near the camera
than on the far side of the pitch.
"""

from dataclasses import dataclass

import numpy as np

from .camera_from_homography import project


@dataclass
class HeightConfig:
    """Thresholds, in metres. Rough defaults: the height error is ~0.5-1 m."""
    max_residual_m: float = 1.5     # a detection farther than this from the vertical is not used
    z_min: float = -1.0             # heights accepted for a detection; a little below 0 is noise
    z_max: float = 30.0
    ground_max: float = 0.8         # z up to this is "ground"


@dataclass
class Fit:
    """One YOLO box tested against the vertical of this frame."""
    box: tuple                      # (x1, y1, x2, y2, confidence)
    u: float                        # detection centre, pixels
    v: float
    z: float                        # best height on the vertical, metres above the grass
    residual_px: float              # distance from the detection to the vertical at z
    residual_m: float               # the same, in metres at the ball
    consistent: bool                # residual and z within the config limits


def pixels_per_metre(camera, x_m, y_m, z_m):
    """How many pixels one metre on the grass spans at (x, y, z): the local scale of the image."""
    u0, v0 = project(camera, x_m, y_m, z_m)
    u1, v1 = project(camera, x_m + 1.0, y_m, z_m)
    u2, v2 = project(camera, x_m, y_m + 1.0, z_m)
    return float((np.hypot(u1 - u0, v1 - v0) + np.hypot(u2 - u0, v2 - v0)) / 2.0)


def vertical(camera, x_m, y_m, heights):
    """Pixels (u, v) of the points at `heights` metres above the grass point (x, y)."""
    return project(camera, x_m, y_m, np.asarray(heights, dtype=np.float64))


def estimate_height(camera, x_m, y_m, u, v):
    """(z, residual_px, residual_m) of a detection at pixel (u, v) against the vertical at (x, y).

    A point at height z projects to (a + z d) / (a3 + z d3) with a = P (x, y, 0, 1) and
    d = P (0, 0, up, 0). Asking it to equal (u, v) gives two equations linear in z:
        z (u d3 - d1) = a1 - u a3
        z (v d3 - d2) = a2 - v a3
    solved together by least squares.
    """
    a = camera.P @ np.array([x_m, y_m, 0.0, 1.0])
    d = camera.P[:, 2] * camera.up
    c = np.array([u * d[2] - d[0], v * d[2] - d[1]])
    r = np.array([a[0] - u * a[2], a[1] - v * a[2]])
    z = float(c @ r / (c @ c)) if np.any(c) else 0.0

    pu, pv = project(camera, x_m, y_m, z)
    residual_px = float(np.hypot(u - pu, v - pv))
    return z, residual_px, residual_px / pixels_per_metre(camera, x_m, y_m, z)


def fit_boxes(camera, x_m, y_m, boxes, config):
    """[Fit] for every box, testing its centre against the vertical at (x, y)."""
    fits = []
    for box in boxes:
        u, v = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        z, residual_px, residual_m = estimate_height(camera, x_m, y_m, u, v)
        consistent = residual_m <= config.max_residual_m and config.z_min <= z <= config.z_max
        fits.append(Fit(tuple(box), u, v, z, residual_px, residual_m, consistent))
    return fits


def choose(fits):
    """The consistent fit closest to the vertical (confidence breaks ties), or None."""
    consistent = [fit for fit in fits if fit.consistent]
    if not consistent:
        return None
    return min(consistent, key=lambda fit: (round(fit.residual_m, 2), -fit.box[4]))


def classify(fits, chosen, config):
    """State of a frame that has a camera and an ATD ball: ground, air, inconsistent, no_detection."""
    if not fits:
        return "no_detection"
    if chosen is None:
        return "inconsistent"
    return "ground" if chosen.z <= config.ground_max else "air"


def search_region(camera, x_m, y_m, config, image_size, margin=80, min_size=320):
    """(x0, y0, (w, h)) crop around the projected vertical from z_min to z_max, plus a margin.

    The ball has to be on (or near) the vertical, so YOLO only needs to look there -- at full
    resolution, which also makes the ball bigger for it. None when the vertical is off screen.
    """
    width, height = image_size
    us, vs = vertical(camera, x_m, y_m, np.linspace(max(0.0, config.z_min), config.z_max, 31))
    if not (np.all(np.isfinite(us)) and np.all(np.isfinite(vs))):
        return None
    x0, x1 = us.min() - margin, us.max() + margin
    y0, y1 = vs.min() - margin, vs.max() + margin
    if x1 < 0 or y1 < 0 or x0 >= width or y0 >= height:
        return None
    # at least min_size on each side, centred on the vertical, inside the frame
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    w = int(min(width, max(min_size, x1 - x0)))
    h = int(min(height, max(min_size, y1 - y0)))
    left = int(np.clip(cx - w / 2.0, 0, width - w))
    top = int(np.clip(cy - h / 2.0, 0, height - h))
    return left, top, (w, h)
