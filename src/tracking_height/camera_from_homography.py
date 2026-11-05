"""A full camera P = K [R | t] recovered from one frame's homography.

The homography H only knows the pitch plane (z = 0): it maps a point on the grass to a pixel. To put
a point at some height above the grass into the image, the camera itself is required. With two mild
assumptions it can be read off H:

  - square pixels and the principal point at the centre of the image, so K = [[f, 0, cx],
    [0, f, cy], [0, 0, 1]] has a single unknown, the focal length f;
  - H, once expressed in metres and centred, is lambda * K [r1 r2 t] with r1, r2 the first two
    columns of a rotation. Their being perpendicular and of equal length gives two equations in f
    (Zhang's constraints), solved together by least squares.

Then r1, r2, t follow, r3 = r1 x r2, R is re-orthonormalised and P = K [R | t]. f is solved per frame
because the tactical camera zooms.

Units: pitch in METRES, x along the length (0 = left goal line), y along the width (0 = far
touchline) -- the ATD axes times the pitch size. The homography file maps NORMALISED (0-1)
coordinates, so it is rescaled before decomposing. Camera.up records which sign of world z is above
the grass, so project(..., height) always takes metres above the grass, positive upwards.
"""

from dataclasses import dataclass

import numpy as np

# Params
IMAGE_SIZE = (1920, 1080)           # tactical video, the frame the homography maps into
PITCH = (105.3, 68.0)               # metres, games.GAMES[GAME]["pitch"]


@dataclass
class Camera:
    """A pinhole camera recovered from one frame's homography."""
    f: float                        # focal length in pixels
    P: np.ndarray                   # 3x4 projection, world metres -> homogeneous pixels
    centre: np.ndarray              # 3, camera position in world metres
    up: float                       # +1 or -1: sign of world z that is above the grass
    pitch: tuple                    # (length, width) in metres

    @property
    def height(self):
        """Camera height above the grass, in metres."""
        return float(self.up * self.centre[2])


def solve_focal(G):
    """Focal length (px) from a centred metric homography G, or None when the geometry gives none.

    With K = diag(f, f, 1), the columns g1, g2 of K^-1 G must be perpendicular and equally long.
    Writing w = 1 / f^2, each condition is linear in w:
        perpendicular:  (g11 g12 + g21 g22) w + g31 g32                   = 0
        equal length:   (g11^2 + g21^2 - g12^2 - g22^2) w + g31^2 - g32^2 = 0
    Both are solved together by least squares.
    """
    g1, g2 = G[:, 0], G[:, 1]
    a = np.array([g1[0] * g2[0] + g1[1] * g2[1],
                  g1[0] ** 2 + g1[1] ** 2 - g2[0] ** 2 - g2[1] ** 2])
    b = np.array([g1[2] * g2[2],
                  g1[2] ** 2 - g2[2] ** 2])
    if not np.any(a):
        return None
    w = -float(a @ b) / float(a @ a)
    if w <= 0:
        return None  # no real focal length: the homography is (close to) affine, or inconsistent
    return 1.0 / np.sqrt(w)


def decompose(H_norm, image_size=IMAGE_SIZE, pitch=PITCH):
    """Camera from one frame's homography (normalised pitch -> pixels), or None if it cannot be done."""
    if not np.all(np.isfinite(H_norm)):
        return None
    cx, cy = image_size[0] / 2.0, image_size[1] / 2.0
    to_centred = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    to_metres = np.diag([1.0 / pitch[0], 1.0 / pitch[1], 1.0])
    G = to_centred @ H_norm @ to_metres

    f = solve_focal(G)
    if f is None:
        return None
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    M = np.diag([1.0 / f, 1.0 / f, 1.0]) @ G          # = lambda [r1 r2 t]

    scale = (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1])) / 2.0
    r1, r2, t = M[:, 0] / scale, M[:, 1] / scale, M[:, 2] / scale

    # lambda's sign is unknown: choose the one that puts the pitch in front of the camera
    if (r1 * pitch[0] / 2.0 + r2 * pitch[1] / 2.0 + t)[2] < 0:
        r1, r2, t = -r1, -r2, -t

    # r1, r2 come from noisy data: snap [r1 r2 r1xr2] to the nearest true rotation
    U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    centre = -R.T @ t
    return Camera(f=f, P=K @ np.column_stack([R, t]), centre=centre,
                  up=1.0 if centre[2] >= 0 else -1.0, pitch=tuple(pitch))


def project(camera, x_m, y_m, height_m=0.0):
    """Pitch metres (x, y) at `height_m` metres above the grass -> pixels (u, v). Arrays welcome."""
    x_m, y_m, height_m = np.broadcast_arrays(np.asarray(x_m, dtype=np.float64),
                                             np.asarray(y_m, dtype=np.float64),
                                             np.asarray(height_m, dtype=np.float64))
    world = np.stack([x_m, y_m, camera.up * height_m, np.ones_like(x_m)])
    pixels = camera.P @ world.reshape(4, -1)
    u, v = pixels[0] / pixels[2], pixels[1] / pixels[2]
    return u.reshape(x_m.shape), v.reshape(x_m.shape)
