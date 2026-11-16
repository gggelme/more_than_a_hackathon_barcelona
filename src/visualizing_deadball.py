"""Deteccion retrospectiva de gestos del arbitro a partir de estados estables YOLO.

Ejecutar desde la raiz del repositorio:
  python src/visualizing_referee_stable_states.py
  python src/visualizing_referee_stable_states.py --start 12675 --seconds 20
  python src/visualizing_referee_stable_states.py --start 12675 --seconds 20 --device cpu

DOS PASADAS: (1) YOLO Detect clase person en cada frame, arbitro central elegido
con homografia, ATD y paleta cromatica SOLO de ATD; cache de detecciones; (2)
segmentacion temporal retrospectiva (mediana, estados estables de ancho/alto,
A->B->C y verificacion de retorno). La reproduccion con barra clicable usa el
cache. YOLO Pose y BBox aportan evidencia independiente al score D. Modificar umbrales temporales NO requiere ejecutar YOLO nuevamente.

El score D fusiona Pose, BBox y quietud de jugadores ATD sin pesos que sumen 1. No usa movimiento de pelota
ni etiquetas publicas de posesion. Heuristica experimental: no es un detector
entrenado de gestos. No transmite ningun video.
"""

from __future__ import annotations

import argparse
import os
import sys
import math
import csv
import json
from pathlib import Path
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import islice

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from starter.load_homography import load as load_homography, project  # noqa: E402
from visualizing_tracking import FONT, COLOURS, slot_info  # noqa: E402

WINDOW = 'Arbitro + pelota ATD | D hasta saque'
PANEL_ALPHA = 0.5                   # how much the top-left panel darkens the video (1 = solid black)
YELLOW = (0, 255, 255)
PERSON_COLOR = (255, 180, 50)       # cian: YOLO asoc. a ATD
OUTSIDE_COLOR = (200, 90, 200)     # violeta: fuera del campo
UNMATCHED_COLOR = (40, 150, 255)   # naranja: dentro, sin match ATD
ATD_OUTLINE = (0, 0, 0)


def load_players(path, start, stop, n_slots):
    """Posiciones normalizadas ATD para identificar jugadores, sin cargar todo el partido."""
    data = np.full((stop-start, n_slots, 2), np.nan, dtype=np.float32)
    balls = np.full((stop-start, 2), np.nan, dtype=np.float32)
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(islice(f, start, stop)):
            parts = line.strip().split(':')
            if len(parts) != 3:
                raise ValueError(f'ATD no valido en frame {start+i}')
            entries = parts[1].split(';')
            if len(entries) != n_slots:
                raise ValueError(f'ATD: se esperaban {n_slots} slots, hay {len(entries)}')
            for j, entry in enumerate(entries):
                x, y, _ = entry.split(',')
                data[i, j] = float(x), float(y)
            ball_fields = parts[2].split(',')
            if len(ball_fields) >= 2:
                balls[i] = float(ball_fields[0]), float(ball_fields[1])
        else:
            if i + 1 != stop-start:
                raise IOError('Faltan datos ATD para el intervalo solicitado')
    return data, balls


def jersey_lab(image, box):
    x1, y1, x2, y2 = map(float, box)
    bh, bw = y2-y1, x2-x1
    a = max(0, int(x1+.30*bw)); b = min(image.shape[1], int(x1+.70*bw))
    c = max(0, int(y1+.16*bh)); d = min(image.shape[0], int(y1+.47*bh))
    if b-a < 3 or d-c < 3:
        return None
    roi = image[c:d, a:b]
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    return np.median(lab.reshape(-1, 3).astype(np.float32), axis=0)



def atd_jersey_lab(image, foot_uv, estimated_height=48., torso_width=12.):
    """Lee color en un parche de camiseta ANCLADO en la posicion ATD del jugador.

    No usa cajas ni detecciones YOLO. El ATD aporta la posicion de los pies;
    la camiseta se aproxima con un rectangulo situado mas arriba.
    Las dimensiones son estimaciones en pixeles, ajustables por CLI.
    """
    cx, fy = map(float, foot_uv)
    height = float(estimated_height)
    width = float(torso_width)
    if not np.isfinite([cx, fy, height, width]).all() or height <= 0 or width <= 0:
        return None
    im_h, im_w = image.shape[:2]
    x1 = max(0, int(round(cx - .5 * width)))
    x2 = min(im_w, int(round(cx + .5 * width)))
    # Banda correspondiente aproximadamente a la camiseta (desde los pies).
    y1 = max(0, int(round(fy - .83 * height)))
    y2 = min(im_h, int(round(fy - .55 * height)))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    patch = image[y1:y2, x1:x2]
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    return np.median(lab.reshape(-1, 3).astype(np.float32), axis=0)


def iou(a, b):
    x1 = max(a[0],b[0]); y1 = max(a[1],b[1])
    x2 = min(a[2],b[2]); y2 = min(a[3],b[3])
    inter = max(0., x2-x1)*max(0., y2-y1)
    a_area = max(0.,a[2]-a[0])*max(0.,a[3]-a[1])
    b_area = max(0.,b[2]-b[0])*max(0.,b[3]-b[1])
    return float(inter/(a_area+b_area-inter+1e-6))


@dataclass
class Detection:
    box: np.ndarray
    confidence: float
    color: np.ndarray | None
    foot: np.ndarray
    on_pitch: bool = False
    role: str | None = None
    slot_index: int | None = None
    novelty: float | None = None
    proximity: float = float('inf')


class RefereeDetector:
    """Detecciones YOLO independientes por frame; asociacion ligera para identidad.

    Nunca predice ni traslada un bounding box entre frames. Si YOLO falla se muestra
    SIN DETECCION, incluso si el arbitro estaba localizado en el frame anterior.
    """
    def __init__(self, model, device, imgsz=1280, conf=0.18, novelty=27.,
                 atd_height=48., atd_width=12., palette_every=5, max_match_dist=65.):
        self.model = model
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.novelty_threshold = novelty
        self.atd_height = atd_height
        self.atd_width = atd_width
        self.palette_every = palette_every
        self.max_match_dist = max_match_dist
        self.palettes = defaultdict(lambda: deque(maxlen=100))
        self.last_box = None
        self.last_color = None
        self.last_frame = None
        self.streak = 0

    def reset(self):
        self.palettes.clear()
        self.last_box = None
        self.last_color = None
        self.last_frame = None
        self.streak = 0

    def _new_detection(self, image, box, confidence):
        box = np.asarray(box, dtype=np.float32)
        foot = np.array([(box[0]+box[2])/2, box[3]], dtype=np.float32)
        return Detection(box, float(confidence), jersey_lab(image, box), foot)

    def update_palette_from_atd(self, frame, image, matrix, slots, info):
        """Actualiza exclusivamente desde los slots de ATD, sin consultar detecciones YOLO."""
        if frame % self.palette_every != 0:
            return
        margin = .02
        polygon = np.asarray([project(matrix, x, y) for x, y in
                              ((margin, margin), (1-margin, margin),
                               (1-margin, 1-margin), (margin, 1-margin))], dtype=np.float32)
        if not np.isfinite(polygon).all() or abs(cv2.contourArea(polygon)) < 100:
            return
        projections = []
        for slot, (x, y) in enumerate(slots):
            if not np.isfinite([x, y]).all():
                continue
            uv = np.asarray(project(matrix, float(x), float(y)), dtype=np.float32)
            if np.isfinite(uv).all():
                projections.append((uv, slot))
        # SOLO posiciones ATD, incluso cuando YOLO no encuentre a nadie.
        for uv, slot in projections:
            if cv2.pointPolygonTest(polygon, tuple(map(float, uv)), False) < 0:
                continue
            # Evitar muestrear dos camisetas solapadas o puntos ATD dudosos.
            if any(other_slot != slot and np.linalg.norm(other_uv - uv) < self.atd_width * 1.5
                   for other_uv, other_slot in projections):
                continue
            sample = atd_jersey_lab(image, uv, self.atd_height, self.atd_width)
            if sample is not None:
                team, goalkeeper = info[slot]
                role = team + ('_gk' if goalkeeper else '')
                self.palettes[role].append(sample)


    def detect(self, frame, image, matrix, slots, info, valid_h):
        if valid_h:
            self.update_palette_from_atd(frame, image, matrix, slots, info)
        # IMPORTANTE: se invoca YOLO para TODOS los frames, incluso si ya hay arbitro.
        result = self.model.predict(image, classes=[0], imgsz=self.imgsz,
                                    conf=self.conf, device=self.device, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            self.streak = 0
            return None, [], 'YOLO sin personas'
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        candidates = [self._new_detection(image, box, confidence)
                      for box, confidence in zip(boxes, confidences)
                      if box[3]-box[1] >= 22]
        if not valid_h:
            return None, candidates, 'Homografia no disponible'

        margin = .02
        polygon = np.asarray([project(matrix, x, y) for x,y in
                              ((margin,margin),(1-margin,margin),
                               (1-margin,1-margin),(margin,1-margin))], dtype=np.float32)
        if not np.isfinite(polygon).all() or abs(cv2.contourArea(polygon)) < 100:
            return None, candidates, 'Poligono de cancha invalido'
        projections = []
        for j, (x,y) in enumerate(slots):
            if not np.isfinite([x,y]).all():
                continue
            u,v = project(matrix, float(x), float(y))
            if np.isfinite([u,v]).all():
                projections.append((np.asarray([u,v], dtype=np.float32), j))

        for det in candidates:
            det.on_pitch = cv2.pointPolygonTest(polygon, tuple(map(float,det.foot)), False) >= 0
            if projections:
                det.proximity = min(float(np.linalg.norm(det.foot-uv)) for uv,_ in projections)

        # Matching 1:1 a jugadores ATD utilizando sus pies; los excluye del arbitraje.
        pairs = []
        for i, det in enumerate(candidates):
            if not det.on_pitch:
                continue
            height = float(det.box[3]-det.box[1])
            gate = float(np.clip(height * .55, 18., 75.))
            for uv, slot in projections:
                dist = float(np.linalg.norm(det.foot-uv))
                if dist <= gate:
                    pairs.append((dist/gate, i, slot))
        used_people, used_slots = set(), set()
        for _, i, slot in sorted(pairs):
            if i in used_people or slot in used_slots:
                continue
            used_people.add(i); used_slots.add(slot)
            team, goalkeeper = info[slot]
            role = team + ('_gk' if goalkeeper else '')
            candidates[i].role = role
            candidates[i].slot_index = slot

        palette_ready = len(self.palettes['H']) >= 3 and len(self.palettes['A']) >= 3
        all_colors = (np.vstack([np.asarray(p) for p in self.palettes.values() if p])
                      if palette_ready else None)
        ranked = []
        for det in candidates:
            if not det.on_pitch or det.role is not None or det.box[3]-det.box[1] < 22:
                continue
            novelty = None
            if det.color is not None and all_colors is not None:
                novelty = float(np.percentile(np.linalg.norm(all_colors-det.color,axis=1),5))
            det.novelty = novelty
            if novelty is None or novelty < self.novelty_threshold:
                continue
            nearby = float(np.exp(-det.proximity/150.0)) if np.isfinite(det.proximity) else 0.
            continuity = 0.
            if self.last_box is not None and self.last_frame is not None and frame-self.last_frame == 1:
                dist = np.linalg.norm((det.box[:2]+det.box[2:])/2 -
                                      (self.last_box[:2]+self.last_box[2:])/2)
                # Bonus to retain referee identity, WITHOUT moving boxes or predicting locations.
                continuity = .5*iou(det.box,self.last_box) + .35*np.exp(-dist/35.)
                if self.last_color is not None and det.color is not None:
                    continuity += .15*np.exp(-np.linalg.norm(det.color-self.last_color)/30.)
            rank = novelty/50 + .28*nearby + continuity
            ranked.append((rank,det))

        # Continuidad fuerte: en el frame siguiente, la deteccion elegible mas
        # cercana a la caja YOLO previa gana independientemente del ranking LAB.
        # Solo entre personas DENTRO de la cancha y NO asociadas al ATD.
        chosen = None
        status = None
        gap = frame - self.last_frame if self.last_frame is not None else 10**9
        if self.last_box is not None and 0 < gap <= 3:
            prev_center = (self.last_box[:2] + self.last_box[2:]) / 2
            prev_h = max(1., float(self.last_box[3]-self.last_box[1]))
            nearest = []
            for det in candidates:
                if not det.on_pitch or det.role is not None:
                    continue
                current_center = (det.box[:2] + det.box[2:]) / 2
                dist = float(np.linalg.norm(current_center-prev_center))
                # Puerta de movimiento: evita saltar a una persona distante.
                gate = min(self.max_match_dist * gap, max(28., prev_h * .85) * gap)
                # Una caja de otra escala es sospechosa (oclusiones / cambio de persona).
                cur_h = max(1., float(det.box[3]-det.box[1]))
                size_ratio = cur_h / prev_h
                if dist <= gate and .60 <= size_ratio <= 1.65:
                    nearest.append((dist, det))
            if nearest:
                chosen = min(nearest, key=lambda x:x[0])[1]
                status = 'ARBITRO (YOLO cercano al frame anterior)'

        # Si se perdio la continuidad, reidentificar con color ATD / cercania.
        if chosen is None:
            ranked.sort(key=lambda x:x[0],reverse=True)
            if not ranked or (len(ranked)>1 and ranked[0][0]-ranked[1][0]<.07):
                self.streak = 0
                status = 'Sin candidato' if not ranked else 'Candidatos ambiguos'
                # Se retiene la referencia hasta 3 frames para reacquisicion.
                if gap > 3:
                    self.last_box = None
                return None, candidates, status
            chosen = ranked[0][1]
            status = 'ARBITRO (reidentificado por color ATD)'

        self.last_box = chosen.box.copy()  # SIEMPRE caja YOLO del frame actual
        self.last_color = chosen.color
        self.last_frame = frame
        self.streak += 1
        return chosen, candidates, status





def compute_player_movement(players, pitch, fps, window=25):
    """Media de velocidades de jugadores ATD, sin usar la pelota para el score."""
    pts = players * np.asarray(pitch, dtype=np.float32)
    good = np.isfinite(pts[1:]).all(axis=-1) & np.isfinite(pts[:-1]).all(axis=-1)
    dist = np.linalg.norm(pts[1:] - pts[:-1], axis=-1)
    per_sum = np.where(good, dist * fps, 0.).sum(axis=1)
    per_count = good.sum(axis=1)
    cs = np.r_[0., np.cumsum(per_sum, dtype=float)]
    cc = np.r_[0, np.cumsum(per_count, dtype=np.int64)]
    speed = np.full(len(players), np.nan, dtype=float)
    for i in range(1, len(players)):
        a = max(0, i-window)
        n = cc[i] - cc[a]
        if n:
            speed[i] = (cs[i] - cs[a]) / n
    return speed


def quiet(value,low,high):
    if value is None or not np.isfinite(value):
        return None
    return float(np.clip((high-value)/(high-low),0,1))


def detect_ball_restarts(players, balls, pitch, fps, args):
    """Encuentra saques offline usando ATD (coordenadas normalizadas).

    La pelota comienza cerca de un jugador y luego avanza alejándose de EL,
    sin utilizar la velocidad de los jugadores como sustituto de la pelota.
    Los saltos excesivos de tracking no se consideran un saque.
    """
    n = min(len(players), len(balls))
    pitch = np.asarray(pitch, dtype=float)
    player_m = np.asarray(players[:n], dtype=float) * pitch
    ball_m = np.asarray(balls[:n], dtype=float) * pitch
    valid_ball = np.isfinite(ball_m).all(axis=1)
    valid_players = np.isfinite(player_m).all(axis=2)
    nearest = np.full(n, np.nan, dtype=float)
    near_slot = np.full(n, -1, dtype=int)
    for i in range(n):
        if valid_ball[i] and valid_players[i].any():
            dist = np.linalg.norm(player_m[i, valid_players[i]] - ball_m[i], axis=1)
            which = np.flatnonzero(valid_players[i])
            j = int(np.argmin(dist))
            nearest[i] = float(dist[j])
            near_slot[i] = int(which[j])

    release = np.zeros(n, dtype=bool)
    speed = np.full(n, np.nan, dtype=float)
    ahead = max(2, round(args.restart_lookahead_seconds * fps))
    # Detectar que el balón estuvo controlado antes de despegar; exigir que no
    # sea una única medición aislada (y evitar teletransportes de tracking).
    for t in range(1, n - ahead):
        if not valid_ball[t-1] or not valid_ball[t] or not np.isfinite(nearest[t-1]):
            continue
        player_id = near_slot[t-1]
        if player_id < 0:
            continue
        nearby = 0
        for j in range(max(0,t-round(.48*fps)), t):
            if (valid_ball[j] and valid_players[j,player_id] and
                    np.linalg.norm(ball_m[j]-player_m[j,player_id]) <= args.restart_near_m):
                nearby += 1
        if nearby < args.restart_near_frames:
            continue
        endpoint = t + ahead
        if not valid_ball[endpoint] or not valid_players[endpoint,player_id]:
            continue
        # Evitar saltos en el ATD: exigir continuidad suficiente durante trayecto.
        sub = ball_m[t-1:endpoint+1]
        if not np.isfinite(sub).all():
            continue
        jumps = np.linalg.norm(np.diff(sub,axis=0),axis=1)*fps
        if np.any(jumps > args.restart_max_speed_mps):
            continue
        displacement = float(np.linalg.norm(ball_m[endpoint]-ball_m[t-1]))
        v = displacement*fps/(ahead+1)
        speed[t] = v
        start_dist = float(np.linalg.norm(ball_m[t-1]-player_m[t-1,player_id]))
        end_dist = float(np.linalg.norm(ball_m[endpoint]-player_m[endpoint,player_id]))
        if (start_dist <= args.restart_near_m and
                end_dist >= args.restart_far_m and
                end_dist-start_dist >= args.restart_separation_m and
                displacement >= args.restart_displacement_m and
                v >= args.restart_min_speed_mps):
            release[t] = True
    # Un mismo saque puede cumplir la condición en varios frames; dejar solo
    # el primer evento de cada grupo.
    cooldown = max(1,round(args.restart_event_gap_seconds*fps))
    last = -cooldown
    for i in np.flatnonzero(release):
        if i-last < cooldown:
            release[i] = False
        else:
            last = i
    return dict(nearest=nearest, release=release, ball_speed=speed)


class BallBoundaryTracker:
    """Clasifica pelota dentro/fuera con tolerancia e histeresis temporal.

    Las coordenadas normalizadas levemente fuera de [0,1] no activan D.
    Una vez FUERA, el estado persiste en la banda de tolerancia y se
    desactiva solo tras un regreso claramente dentro, sostenido.
    """
    def __init__(self, margin=0.005, out_frames=3, in_frames=3):
        self.margin = float(margin)
        self.out_frames = int(out_frames)
        self.in_frames = int(in_frames)
        self.outside = False
        self.out_streak = 0
        self.in_streak = 0

    def update(self, ball):
        if ball is None:
            self.out_streak = self.in_streak = 0
            return float('nan'), float('nan'), 'DESCONOCIDA'
        xy = np.asarray(ball, dtype=float).reshape(-1)
        if xy.size < 2 or not np.isfinite(xy[:2]).all():
            self.out_streak = self.in_streak = 0
            return float('nan'), float('nan'), 'DESCONOCIDA'
        x, y = float(xy[0]), float(xy[1])
        m = self.margin
        # FUERA claro: el centro esta mas alla de la banda exterior.
        clearly_out = x < -m or x > 1+m or y < -m or y > 1+m
        # DENTRO claro: ambos ejes se alejaron de la linea hacia adentro.
        clearly_in = m <= x <= 1-m and m <= y <= 1-m
        if self.outside:
            self.in_streak = self.in_streak + 1 if clearly_in else 0
            if self.in_streak >= self.in_frames:
                self.outside = False
                self.out_streak = 0
                self.in_streak = 0
        else:
            self.out_streak = self.out_streak + 1 if clearly_out else 0
            if self.out_streak >= self.out_frames:
                self.outside = True
                self.in_streak = 0
                self.out_streak = 0
        if self.outside:
            return x, y, 'FUERA'
        if clearly_out or not clearly_in:
            return x, y, 'BORDE'
        return x, y, 'DENTRO'


class DeadBallScore:
    """Fusion de evidencias independientes: Pose, BBox y quietud de jugadores; pelota fuera favorece D.

    S_D = 1 - (1 - wP*P) * (1 - wB*B) * (1 - wQ*Q)

    Todos los scores y pesos son [0,1]. Los pesos NO deben sumar 1.
    Si pelota x/y queda fuera de [0,1], se agrega evidencia positiva
    con peso ball_out_weight. Una pelota ausente no agrega evidencia.
    El único criterio de entrada es superar el umbral durante N frames.
    """
    def __init__(self, enter=.65, exit=.35, enter_frames=8, exit_frames=6,
                 pose_weight=.9, bbox_weight=.2, players_weight=.5, ball_out_weight=.8,
                 hold_until_restart=True, restart_timeout_frames=875,
                 restart_cooldown_frames=25,
                 ball_out_margin=0.005, ball_out_frames=3, ball_in_frames=3):
        self.enter = enter
        self.exit = exit
        self.enter_frames = enter_frames
        self.exit_frames = exit_frames
        self.pose_weight = pose_weight
        self.bbox_weight = bbox_weight
        self.players_weight = players_weight
        self.ball_out_weight = ball_out_weight
        self.ball_boundary = BallBoundaryTracker(ball_out_margin, ball_out_frames, ball_in_frames)
        self.hold_until_restart = hold_until_restart
        self.restart_timeout_frames = restart_timeout_frames
        self.restart_cooldown_frames = restart_cooldown_frames
        self.dead_frames = 0
        self.cooldown = 0
        self.last_reason = 'NO_D'
        self.restart_unlocked = False
        self.dead = False
        self.rising = 0
        self.falling = 0

    def step(self, player_speed, pose, bbox, ball=None, *, restart=False):
        q = quiet(player_speed, .7, 2.7)
        pose_value = float(np.clip(pose, 0, 1)) if pose is not None and np.isfinite(pose) else 0.
        bbox_value = float(np.clip(bbox, 0, 1)) if bbox is not None and np.isfinite(bbox) else 0.
        quiet_value = float(np.clip(q, 0, 1)) if q is not None else 0.
        evidence_pose = self.pose_weight * pose_value
        evidence_bbox = self.bbox_weight * bbox_value
        evidence_players = self.players_weight * quiet_value
        gesture = 1 - (1 - evidence_pose) * (1 - evidence_bbox)
        score = 1 - (1 - gesture) * (1 - evidence_players)
        # Aportes marginales exactos en este orden: Pose, BBox, jugadores.
        contribution_pose = evidence_pose
        contribution_bbox = (1 - evidence_pose) * evidence_bbox
        contribution_players = (1 - gesture) * evidence_players

        # Evidencia POSITIVA de pelota FUERA confirmada temporalmente.
        # La banda de tolerancia evita falsos D por jitter cerca de la linea.
        ball_x, ball_y, ball_status = self.ball_boundary.update(ball)
        evidence_ball = self.ball_out_weight if ball_status == 'FUERA' else 0.
        score_before_ball = score
        score = float(np.clip(1. - (1. - score_before_ball) * (1. - evidence_ball), 0., 1.))
        contribution_ball = score - score_before_ball  # aporte positivo o cero

        if self.cooldown > 0:
            self.cooldown -= 1
        if not self.dead:
            # Fuera de D se necesita superar el umbral ALTO.
            self.restart_unlocked = False
            self.last_reason = 'NO_D' if not self.cooldown else 'POST_SAQUE'
            self.rising = self.rising + 1 if score >= self.enter and self.cooldown == 0 else 0
            if self.rising >= self.enter_frames:
                self.dead = True
                self.dead_frames = 0
                self.rising = 0
                self.falling = 0
                self.restart_unlocked = False
                self.last_reason = 'D_POR_SCORE'
        elif self.hold_until_restart:
            # Modo combinado: la salida por score esta BLOQUEADA hasta que haya saque.
            # El saque habilita la histeresis pero NO termina D por si mismo.
            self.dead_frames += 1
            if restart:
                self.restart_unlocked = True
                self.last_reason = 'SAQUE_HABILITA_SALIDA'
            if not self.restart_unlocked:
                self.falling = 0
                self.last_reason = 'ESPERANDO_SAQUE'
            else:
                self.falling = self.falling + 1 if score < self.exit else 0
                if self.falling >= self.exit_frames:
                    self.dead = False
                    self.dead_frames = 0
                    self.restart_unlocked = False
                    self.falling = 0
                    self.rising = 0
                    self.cooldown = self.restart_cooldown_frames
                    self.last_reason = 'SAQUE_Y_SCORE_BAJO'
                elif restart:
                    self.last_reason = 'SAQUE_ESPERANDO_SCORE_BAJO'
                else:
                    self.last_reason = 'POST_SAQUE_UMBRAL_BAJO'
            # Timeout opcional (por defecto 0, DESACTIVADO): seguridad ante
            # fallas persistentes de tracking sin saque detectado.
            if (self.dead and self.restart_timeout_frames > 0 and
                    self.dead_frames >= self.restart_timeout_frames):
                self.dead = False
                self.dead_frames = 0
                self.restart_unlocked = False
                self.falling = 0
                self.rising = 0
                self.cooldown = self.restart_cooldown_frames
                self.last_reason = 'TIMEOUT_SIN_SAQUE'
        else:
            # Modo opcional de sola histeresis, sin condicion de saque.
            self.restart_unlocked = False
            self.last_reason = 'D_UMBRAL_BAJO'
            self.falling = self.falling + 1 if score < self.exit else 0
            if self.falling >= self.exit_frames:
                self.dead = False
                self.falling = 0
                self.rising = 0
                self.last_reason = 'SCORE_BAJO'
        return (self.dead, score, q, (contribution_pose, contribution_bbox, contribution_players),
                (ball_x, ball_y, ball_status, evidence_ball, score_before_ball, contribution_ball))


def fmt_num(value):
    return '--' if value is None or not np.isfinite(value) else f'{value:.3f}'


def draw_atd_overlay(image, matrix, slots, ball, info):
    """Mostrar los mismos puntos/ids del ATD que usa visualizing_tracking.py."""
    h, w = image.shape[:2]
    for i, ((x, y), (team, goalkeeper)) in enumerate(zip(slots, info)):
        if not np.isfinite([x, y]).all():
            continue
        u, v = project(matrix, float(x), float(y))
        if not np.isfinite([u, v]).all():
            continue
        px, py = int(round(u)), int(round(v))
        if not (0 <= px < w and 0 <= py < h):
            continue
        color = COLOURS[(team, 'gk')] if goalkeeper else COLOURS[team]
        cv2.circle(image, (px, py), 6, ATD_OUTLINE, -1)
        cv2.circle(image, (px, py), 4, color, -1)
        cv2.putText(image, str(i+1), (px-6, py-10), FONT, .42,
                    ATD_OUTLINE, 3, cv2.LINE_AA)
        cv2.putText(image, str(i+1), (px-6, py-10), FONT, .42,
                    color, 1, cv2.LINE_AA)
    if np.isfinite(ball).all():
        u, v = project(matrix, float(ball[0]), float(ball[1]))
        if np.isfinite([u, v]).all() and 0 <= u < w and 0 <= v < h:
            cv2.circle(image, (int(round(u)), int(round(v))), 10, COLOURS['ball'], 2)


def draw_yolo_detections(image, candidates, selected):
    """Todas las cajas son del frame actual: nunca predichas por tracker."""
    for i, det in enumerate(candidates):
        is_ref = det is selected
        if is_ref:
            color = YELLOW
            label = f'ARBITRO {det.confidence:.2f}'
        elif det.role is not None:
            color = PERSON_COLOR
            label = f'YOLO #{i+1} ATD:{det.slot_index+1} {det.role} {det.confidence:.2f}'
        elif not det.on_pitch:
            color = OUTSIDE_COLOR
            label = f'YOLO #{i+1} FUERA {det.confidence:.2f}'
        else:
            color = UNMATCHED_COLOR
            novelty = f'{det.novelty:.0f}' if det.novelty is not None else '--'
            label = f'YOLO #{i+1} CAND. {det.confidence:.2f} LAB:{novelty}'
        x1, y1, x2, y2 = np.rint(det.box).astype(int)
        cv2.rectangle(image, (x1,y1), (x2,y2), color, 3 if is_ref else 1)
        ytext = max(12, y1 - 5)
        cv2.putText(image, label, (max(0,x1), ytext), FONT, .42,
                    (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(image, label, (max(0,x1), ytext), FONT, .42,
                    color, 1, cv2.LINE_AA)




# -----------------------------------------------------------------------------
# Primera pasada: guardar todas las detecciones para que la busqueda de estados
# y la reproduccion no ejecuten YOLO otra vez.
# -----------------------------------------------------------------------------

def file_signature(path):
    st = os.stat(path)
    return [os.path.abspath(path), st.st_size, st.st_mtime_ns]


def detection_cache_metadata(args, video_path, meta_path, atd_path, weights,
                             first, end, fps):
    return dict(schema=2, video=file_signature(video_path),
                meta=file_signature(meta_path), atd=file_signature(atd_path),
                weights=file_signature(weights), first=first, end=end,
                fps=round(float(fps), 5), game=str(GAME),
                detect=dict(imgsz=args.imgsz, conf=args.conf,
                            novelty=args.novelty, atd_height=args.atd_height,
                            atd_width=args.atd_width,
                            palette_every=args.palette_every,
                            max_match_dist=args.max_match_dist))


def selected_to_dict(frame, candidates, selected, status, matrix=None):
    ref_index = next((i for i, d in enumerate(candidates) if d is selected), None)
    payload = []
    for d in candidates:
        payload.append(dict(box=[float(x) for x in d.box], conf=float(d.confidence),
                            on_pitch=bool(d.on_pitch), role=d.role,
                            slot=d.slot_index, novelty=(float(d.novelty)
                            if d.novelty is not None else None)))
    world = None
    if selected is not None and matrix is not None:
        try:
            inv = np.linalg.inv(np.asarray(matrix, dtype=np.float64))
            foot = np.array([float(selected.foot[0]), float(selected.foot[1]), 1.0])
            projection = inv @ foot
            if abs(projection[2]) > 1e-9:
                xy = projection[:2] / projection[2]
                if np.isfinite(xy).all() and np.all(xy >= -.1) and np.all(xy <= 1.1):
                    world = [float(v) for v in xy]
        except (np.linalg.LinAlgError, ValueError):
            pass
    return dict(frame=int(frame), detections=payload, referee=ref_index,
                status=status, world=world)


def load_cache(path, expected):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding='utf-8') as handle:
            header = json.loads(next(handle))
            if header.get('metadata') != expected:
                print('Cache incompatible con el video o configuracion YOLO: se recalculara.')
                return None
            rows = [json.loads(line) for line in handle]
        if len(rows) != expected['end'] - expected['first'] or any(
                r['frame'] != expected['first'] + i for i, r in enumerate(rows)):
            print('Cache incompleto: se recalculara.')
            return None
        print(f'Cache YOLO cargado: {len(rows)} frames ({path})')
        return rows
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        print(f'Cache ilegible ({exc}): se recalculara.')
        return None


def detect_to_cache(args, cap, video_path, meta_path, atd_path, info,
                    players, matrices, valid, first, end, fps, out_dir):
    weights = (args.weights if os.path.isabs(args.weights) else
               os.path.join(ROOT, 'models', args.weights))
    if not os.path.isfile(weights):
        raise FileNotFoundError(f'Faltan los pesos YOLO: {weights}')
    metadata = detection_cache_metadata(args, video_path, meta_path, atd_path,
                                        weights, first, end, fps)
    cache_name = f'referee_yolo_{first}_{end}_cache.jsonl'
    path = os.path.join(out_dir, cache_name)
    rows = None if args.recompute else load_cache(path, metadata)
    if rows is not None:
        return rows, path

    from ultralytics import YOLO  # no se necesita al reutilizar el cache
    device = int(args.device) if str(args.device).isdecimal() else (
        None if str(args.device).lower() == 'auto' else args.device)
    detector = RefereeDetector(YOLO(weights), device, args.imgsz, args.conf,
                               args.novelty, atd_height=args.atd_height,
                               atd_width=args.atd_width,
                               palette_every=args.palette_every,
                               max_match_dist=args.max_match_dist)
    print(f'Procesando YOLO en {end-first} frames ({first}..{end-1}) ...')
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    temp = path + '.partial'
    rows = []
    try:
        with open(temp, 'w', encoding='utf-8') as handle:
            handle.write(json.dumps({'metadata': metadata}, allow_nan=False) + '\n')
            for frame in range(first, end):
                ok, image = cap.read()
                if not ok:
                    raise IOError(f'No se pudo leer el frame {frame} durante YOLO')
                good = frame < len(valid) and bool(valid[frame])
                matrix = matrices[frame] if good else None
                selected, candidates, status = detector.detect(
                    frame, image, matrix, players[frame-first], info, good)
                row = selected_to_dict(frame, candidates, selected, status, matrix)
                handle.write(json.dumps(row, allow_nan=False) + '\n')
                rows.append(row)
                if (frame-first+1) % max(1, round(fps)) == 0:
                    print(f'  YOLO: {frame-first+1}/{end-first} frames', flush=True)
        os.replace(temp, path)
    except BaseException:
        if os.path.isfile(temp):
            os.remove(temp)
        raise
    print(f'Detecciones YOLO guardadas: {path}')
    return rows, path


# -----------------------------------------------------------------------------
# Segunda pasada: dos ejes, mediana -> estados estables -> A/B/C con futuro.
# -----------------------------------------------------------------------------

@dataclass
class StableState:
    index: int
    start: int                       # indices relativos al inicio del analisis
    end: int                         # exclusivo
    width: float
    height: float
    x: float
    y: float
    world_x: float | None
    world_y: float | None
    cv_width: float
    cv_height: float

    @property
    def duration(self):
        return self.end - self.start


def ranges_of_true(mask):
    """Intervalos semiabiertos de True en un vector booleano."""
    mask = np.asarray(mask, dtype=bool)
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def rolling_median_valid(values, size, breaks):
    """Mediana por subtramos continuos; nunca cruza perdida/reidentificacion."""
    from numpy.lib.stride_tricks import sliding_window_view
    values = np.asarray(values, dtype=np.float64)
    result = np.full(len(values), np.nan, dtype=np.float64)
    good = np.isfinite(values)
    good[np.asarray(breaks, dtype=bool)] = False
    for a, b in ranges_of_true(good):
        run = values[a:b]
        half = size // 2
        if len(run) == 1:
            result[a:b] = run
        else:
            padded = np.pad(run, (half, half), mode='edge')
            result[a:b] = np.median(sliding_window_view(padded, size), axis=1)
    return result


def interpolate_short_gaps(width, height, cx, cy, world, breaks, max_gap):
    """Interpolacion SOLO para analisis temporal; no inventa cajas de YOLO."""
    missing = ~np.isfinite(width) | ~np.isfinite(height)
    for a, b in ranges_of_true(missing):
        if (b-a > max_gap or a == 0 or b == len(width) or
                np.any(breaks[a:b+1]) or
                not np.isfinite(width[a-1]) or not np.isfinite(width[b])):
            continue
        for data in (width, height, cx, cy, world[:, 0], world[:, 1]):
            if np.isfinite(data[a-1]) and np.isfinite(data[b]):
                data[a:b] = np.linspace(data[a-1], data[b], b-a+2)[1:-1]


def state_from_run(index, a, b, width, height, cx, cy, world):
    def med(x):
        z = np.asarray(x[a:b], dtype=float)
        z = z[np.isfinite(z)]
        return float(np.median(z)) if len(z) else float('nan')
    w, h = med(width), med(height)
    cv_w = float(np.std(width[a:b]) / max(w, 1e-6))
    cv_h = float(np.std(height[a:b]) / max(h, 1e-6))
    wx, wy = med(world[:, 0]), med(world[:, 1])
    return StableState(index, a, b, w, h, med(cx), med(cy),
                       wx if np.isfinite(wx) else None,
                       wy if np.isfinite(wy) else None, cv_w, cv_h)


def state_distance_m(a, b, pitch):
    if None in (a.world_x, a.world_y, b.world_x, b.world_y):
        return None
    return float(np.linalg.norm((np.array([a.world_x, a.world_y]) -
                                 np.array([b.world_x, b.world_y])) * pitch))


def compare_triplet(a, b, c, fps, args, pitch):
    """Evaluar estable normal -> estable diferente -> retorno normal.

    Se usan frames futuros. Devuelve (score, causa, metricas).
    """
    metrics = {}
    gap_ab = b.start - a.end
    gap_bc = c.start - b.end
    if gap_ab > round(args.max_transition_seconds * fps) or gap_bc > round(args.return_seconds * fps):
        return 0., 'TRANSICION_LARGA', metrics
    if b.duration > round(args.max_gesture_seconds * fps):
        return 0., 'MESETA_DEMASIADO_LARGA', metrics
    if min(a.width, a.height, b.width, b.height, c.width, c.height) <= 0:
        return 0., 'DIMENSIONES_INVALIDAS', metrics
    dw = math.log(b.width / a.width)
    dh = math.log(b.height / a.height)
    magnitude = max(dw, dh)
    anisotropy = abs(dw - dh)
    back_w = math.log(c.width / a.width)
    back_h = math.log(c.height / a.height)
    return_gap = max(abs(back_w), abs(back_h))
    metrics = {'dw': dw, 'dh': dh, 'anisotropy': anisotropy,
               'return_gap': return_gap, 'gap_ab': gap_ab, 'gap_bc': gap_bc}
    if magnitude < args.gesture_change:
        return 0., 'CAMBIO_INSUFICIENTE', metrics
    if anisotropy < args.min_anisotropy:
        return 0., 'ESCALA_PROPORCIONAL', metrics
    if return_gap > args.return_tolerance:
        return 0., 'SIN_RETORNO_DIMENSIONES', metrics
    dist_ac = state_distance_m(a, c, pitch)
    dist_ab = state_distance_m(a, b, pitch)
    dist_bc = state_distance_m(b, c, pitch)
    metrics['distance_ac_m'] = dist_ac
    metrics['distance_ab_m'] = dist_ab
    metrics['distance_bc_m'] = dist_bc
    known = [x for x in (dist_ac, dist_ab, dist_bc) if x is not None]
    if known:
        if any(x > args.max_position_m for x in known):
            return 0., 'CAMBIO_DE_POSICION', metrics
    else:
        # Respaldo si la homografia no permite desproyectar los pies.
        max_dist_px = args.max_position_body * max(a.height, b.height, c.height)
        if any(math.hypot(p.x - q.x, p.y - q.y) > max_dist_px
               for p, q in ((a, b), (b, c), (a, c))):
            return 0., 'DESPLAZAMIENTO_EN_IMAGEN', metrics
    score = float(np.clip(0.5 + 0.5 * (magnitude-args.gesture_change) /
                          max(1e-6, args.gesture_full-args.gesture_change), 0.5, 1.))
    return score, 'GESTO_CONFIRMADO_POR_RETORNO', metrics


def axis_segments(stable, values, min_state, join_gap, breaks):
    """Mesetas independientes: la estabilidad del otro eje NO interviene."""
    candidates = ranges_of_true(stable)
    combined = []
    for a, b in candidates:
        if combined:
            x, y = combined[-1]
            gap = a - y
            left, right = np.median(values[x:y]), np.median(values[a:b])
            if (gap <= join_gap and not breaks[y:a+1].any()
                    and np.isfinite(left) and np.isfinite(right)
                    and abs(math.log(right / left)) <= .07):
                combined[-1] = (x, b)
                continue
        combined.append((a, b))
    return [(a, b) for a, b in combined if b-a >= min_state]


def compare_axis_triplet(a, b, c, axis, fps, args, pitch):
    """Busca meseta ampliada en UN eje con retorno en ese eje.

    El otro eje no tiene que ser estable: se usa para descartar zoom
    aproximadamente proporcional, nunca como condicion de segmentacion.
    """
    axis_name = 'width' if axis == 'W' else 'height'
    other_name = 'height' if axis == 'W' else 'width'
    av, bv, cv = (getattr(x, axis_name) for x in (a, b, c))
    ao, bo, co = (getattr(x, other_name) for x in (a, b, c))
    metrics = {'axis': axis}
    if min(av, bv, cv, ao, bo, co) <= 0:
        return 0., 'DIMENSIONES_INVALIDAS', metrics
    gap_ab = b.start-a.end
    gap_bc = c.start-b.end
    metrics.update(gap_ab=gap_ab, gap_bc=gap_bc)
    if gap_ab > round(args.max_transition_seconds*fps) or gap_bc > round(args.return_seconds*fps):
        return 0., 'TRANSICION_LARGA', metrics
    # No limitar la duracion de una postura sostenida; solo mirar el retorno
    # dentro del horizonte posterior al FINAL de la meseta.
    change = math.log(bv/av)
    other_change = math.log(bo/ao)
    back = math.log(cv/av)
    metrics.update(dw=math.log(b.width/a.width), dh=math.log(b.height/a.height),
                   change=change, other_change=other_change,
                   anisotropy=abs(change-other_change), return_gap=abs(back))
    if change < args.gesture_change:
        return 0., 'CAMBIO_INSUFICIENTE', metrics
    if change-other_change < args.min_anisotropy:
        return 0., 'ESCALA_PROPORCIONAL', metrics
    if abs(back) > args.return_tolerance:
        return 0., 'SIN_RETORNO_EJE', metrics
    known = [state_distance_m(p,q,pitch) for p,q in ((a,b),(b,c),(a,c))]
    metrics['distance_ac_m'] = known[2]
    known = [v for v in known if v is not None]
    if known:
        if any(d > args.max_position_m for d in known):
            return 0., 'CAMBIO_DE_POSICION', metrics
    elif any(math.hypot(p.x-q.x,p.y-q.y) > args.max_position_body * max(p.height,q.height)
             for p,q in ((a,b),(b,c),(a,c))):
        return 0., 'DESPLAZAMIENTO_EN_IMAGEN', metrics
    amp = float(np.clip((change-args.gesture_change) /
                        (args.gesture_full-args.gesture_change), 0., 1.))
    return 0.5 + 0.5*amp, 'GESTO_CONFIRMADO_POR_RETORNO', metrics


def detect_local_plateaus(smoothed_w, smoothed_h, epochs, fps, args):
    """Cambios sostenidos por eje, INCLUSO dentro de un estado estable largo.

    Contrasta una referencia anterior con una meseta posterior (uso offline de
    frames futuros). Las dos deben ser de baja dispersion. Se comprueba que
    el cambio no sea simplemente zoom proporcional. No depende de A-B-A.
    """
    n = len(smoothed_w)
    signals = {'W': np.zeros(n, np.float32), 'H': np.zeros(n, np.float32)}
    events = []
    before = max(12, round(.8 * fps))
    after = max(14, round(1.0 * fps))
    offset = max(6, round(.36 * fps))
    # Centrar referencia ANTES de la transicion y muestra POSTERIOR a ella.
    min_change = args.plateau_change
    for axis, values, other in (('W', smoothed_w, smoothed_h),
                                ('H', smoothed_h, smoothed_w)):
        last_end = 0
        for t in range(before + offset, n - after - offset):
            if t < last_end or epochs[t] < 0:
                continue
            l0, l1 = t-before-offset, t-offset
            r0, r1 = t+offset, t+offset+after
            # No mezclar identidades ni observaciones invalidas.
            if not np.all(epochs[l0:r1] == epochs[t]):
                continue
            prev = values[l0:l1]
            next_ = values[r0:r1]
            prev_other = other[l0:l1]
            next_other = other[r0:r1]
            if (min(np.min(prev), np.min(next_), np.min(prev_other),
                    np.min(next_other)) <= 0):
                continue
            logprev, lognext = np.log(prev), np.log(next_)
            variation_prev = float(np.std(logprev))
            variation_next = float(np.std(lognext))
            if variation_prev > args.plateau_std or variation_next > args.plateau_std:
                continue
            base = float(np.median(prev))
            high = float(np.median(next_))
            increase = math.log(high/base)
            if increase < min_change:
                continue
            other_increase = math.log(float(np.median(next_other)) / float(np.median(prev_other)))
            if increase - other_increase < args.plateau_anisotropy:
                continue
            # Primera diferencia de nivel: evita marcar meseta ya empezada.
            early = values[max(l1, t-round(.2*fps)):t]
            if len(early) and float(np.median(early)) > base*math.exp(increase*.75):
                continue
            # Confirmacion de meseta disponible en el futuro (offline).
            threshold = base * math.exp(max(min_change * .58, increase * .45))
            begin = t
            # Dentro de los siguientes instantes, buscar inicio real de subida.
            for j in range(max(l1,t-round(.35*fps)), min(r0+1,n)):
                if values[j] >= threshold:
                    begin = j
                    break
            end = r1
            # Seguir la meseta mientras mantenga el nuevo nivel, incluso si se
            # relaja lentamente. Una excursion breve no la termina.
            below = 0
            for j in range(r1, min(n, r1+round(args.plateau_max_seconds*fps))):
                if epochs[j] != epochs[t]:
                    break
                if values[j] < threshold:
                    below += 1
                else:
                    below = 0
                if below >= max(5, round(.3*fps)):
                    end = j-below+1
                    break
                end = j+1
            if end-begin < round(.6*fps):
                continue
            # Que un escalon claro no tenga score cero por estar cerca del umbral.
            strength = float(np.clip(increase/max(min_change, args.plateau_full), 0., 1.))
            signals[axis][begin:end] = np.maximum(signals[axis][begin:end], strength)
            events.append(dict(axis=axis, start=begin, end=end, change_log=increase,
                               baseline=base, plateau=high, std_pre=variation_prev,
                               std_post=variation_next, other_change_log=other_increase,
                               score=strength))
            last_end = end
    return signals['W'], signals['H'], events


def penalize_gestures_while_running(rows, analysis, fps, pitch, args):
    """Descuenta mesetas detectadas durante desplazamiento continuo del arbitro.

    Las coordenadas de sus pies se recuperaron con la homografia, en unidades
    normalizadas de cancha. El procesamiento es offline, sin datos de posesion.
    Se aplica una mediana temporal a las posiciones y a la rapidez para reducir
    el jitter de las cajas YOLO. Cuando faltan posiciones, NO se presume carrera.

    Ademas, mide oscilacion de anchura de alta frecuencia (piernas al correr)
    respecto de una mediana ya calculada; sirve de penalizacion secundaria.
    """
    n = len(rows)
    loc = np.full((n, 2), np.nan, dtype=float)
    for i, row in enumerate(rows):
        xy = row.get('world')
        if xy is not None and len(xy) == 2 and np.isfinite(xy).all():
            loc[i] = xy
    # Mediana espacial de 7 frames, sin rellenar huecos largos.
    loc_med = loc.copy()
    for i in range(n):
        a, b = max(0, i-3), min(n, i+4)
        points = loc[a:b]
        valid = np.isfinite(points).all(axis=1)
        if valid.sum() >= 4:
            loc_med[i] = np.median(points[valid], axis=0)
    metres = loc_med * np.asarray(pitch, dtype=float)
    speed = np.full(n, np.nan, dtype=float)
    step = max(2, round(.20 * fps))
    for i in range(step, n-step):
        if np.isfinite(metres[i-step]).all() and np.isfinite(metres[i+step]).all():
            speed[i] = np.linalg.norm(metres[i+step] - metres[i-step]) * fps / (2*step)
    # El filtro temporal de velocidades atenúa fluctuaciones de YOLO.
    smooth_speed = np.full(n, np.nan, dtype=float)
    radius = max(2, round(.24 * fps))
    for i in range(n):
        a, b = max(0, i-radius), min(n, i+radius+1)
        valid = speed[a:b][np.isfinite(speed[a:b])]
        if len(valid) >= max(3, (b-a)//2):
            smooth_speed[i] = np.median(valid)

    # Oscilacion residual: si una caja cambia de anchura frecuentemente,
    # hay menos evidencia de gesto a partir de una meseta de bounding boxes.
    raw_w, med_w = analysis['width'], analysis['smoothed_w']
    residual = np.full(n, np.nan, dtype=float)
    ok = np.isfinite(raw_w) & np.isfinite(med_w) & (med_w > 1)
    residual[ok] = (raw_w[ok] - med_w[ok]) / med_w[ok]
    oscillation = np.full(n, np.nan, dtype=float)
    win = max(5, round(.8*fps))
    for i in range(n):
        a, b = max(0, i-win+1), i+1
        seg = residual[a:b]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= max(5, (b-a)*.7):
            oscillation[i] = np.std(seg)

    # Penalizacion suave: no elimina por completo señales durante movimiento.
    penalty_speed = np.ones(n, dtype=float)
    valid_s = np.isfinite(smooth_speed)
    penalty_speed[valid_s] = 1-np.clip(
        (smooth_speed[valid_s]-args.ref_run_low)/max(1e-6,args.ref_run_high-args.ref_run_low),0,1
    ) * args.ref_run_penalty
    penalty_osc = np.ones(n, dtype=float)
    valid_o = np.isfinite(oscillation)
    penalty_osc[valid_o] = 1-np.clip(
        (oscillation[valid_o]-args.ref_osc_low)/max(1e-6,args.ref_osc_high-args.ref_osc_low),0,1
    ) * args.ref_osc_penalty
    reliability = np.clip(penalty_speed*penalty_osc,0,1)
    analysis['gesture_unfiltered'] = analysis['gesture'].copy()
    analysis['gesture_w'] = (analysis['gesture_w'] * reliability).astype(np.float32)
    analysis['gesture_h'] = (analysis['gesture_h'] * reliability).astype(np.float32)
    analysis['gesture'] = np.maximum(analysis['gesture_w'],analysis['gesture_h'])
    analysis['referee_speed'] = smooth_speed
    analysis['referee_oscillation'] = oscillation
    analysis['gesture_reliability'] = reliability
    analysis['labels'] = analysis['labels'].copy()
    # No mostrar GESTO si se descartó la evidencia por carrera.
    suppressed = (analysis['gesture_unfiltered'] > 0) & (analysis['gesture'] < .20)
    analysis['labels'][suppressed] = 'CANDIDATO_DESCARTADO_CARRERA'
    return analysis


def analyze_states(rows, fps, args, pitch):
    """Estados de ANCHO y ALTO totalmente independientes, con futuro disponible."""
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(rows)
    frame_values = np.array([r['frame'] for r in rows], dtype=int)
    width = np.full(n, np.nan)
    height = np.full(n, np.nan)
    cx = np.full(n, np.nan)
    cy = np.full(n, np.nan)
    world = np.full((n,2), np.nan)
    breaks = np.zeros(n, dtype=bool)
    for i,row in enumerate(rows):
        ref = row.get('referee')
        if ref is None or ref >= len(row['detections']):
            continue
        bb = row['detections'][ref]['box']
        width[i],height[i] = bb[2]-bb[0], bb[3]-bb[1]
        cx[i],cy[i] = (bb[0]+bb[2])/2, (bb[1]+bb[3])/2
        if row.get('world') is not None:
            world[i] = row['world']
        if i and 'reidentificado' in row['status'] and np.isfinite(width[i-1]):
            breaks[i] = True

    aw,ah,ax,ay,loc = (width.copy(),height.copy(),cx.copy(),cy.copy(),world.copy())
    interpolate_short_gaps(aw,ah,ax,ay,loc,breaks,args.fill_gap)
    mw = rolling_median_valid(aw,args.median_frames,breaks)
    mh = rolling_median_valid(ah,args.median_frames,breaks)
    # IDs de secuencias de identidad: los huecos largos y las reidentificaciones
    # no pueden vincular gestos de personas diferentes.
    good = np.isfinite(mw) & np.isfinite(mh) & ~breaks
    epochs = np.full(n,-1,dtype=int)
    for eid,(a,b) in enumerate(ranges_of_true(good)):
        epochs[a:b] = eid

    local_n = max(3,round(args.stability_seconds*fps))
    if local_n%2==0:
        local_n+=1
    half=local_n//2
    third=max(2,local_n//3)
    cvw,cvh,dw,dh = [np.full(n,np.nan) for _ in range(4)]
    stable_w=np.zeros(n,bool)
    stable_h=np.zeros(n,bool)
    for value,sigma,drift,stable in ((mw,cvw,dw,stable_w),(mh,cvh,dh,stable_h)):
        finite=np.isfinite(value)&(value>0)&~breaks
        for a,b in ranges_of_true(finite):
            if b-a<local_n:
                continue
            windows=sliding_window_view(np.log(value[a:b]),local_n)
            sl=slice(a+half,b-half)
            sigma[sl]=windows.std(axis=1)
            drift[sl]=np.abs(np.median(windows[:,-third:],axis=1)-
                             np.median(windows[:,:third],axis=1))
            stable[sl]=(sigma[sl] <= args.stable_cv)&(drift[sl]<=args.stable_drift)

    min_state=max(3,round(args.min_state_seconds*fps))
    states=[]
    states_axis={'W':[],'H':[]}
    ids_w=np.full(n,-1,dtype=int)
    ids_h=np.full(n,-1,dtype=int)
    labels=np.full(n,'INDETERMINADO',dtype=object)
    labels[good]='TRANSICION_O_MOVIMIENTO'
    for axis,stable,value,ids in (('W',stable_w,mw,ids_w),('H',stable_h,mh,ids_h)):
        segments=axis_segments(stable,value,min_state,args.join_gap,breaks)
        for a,b in segments:
            # IDs globales para que CSV de transiciones identifique el estado.
            st=state_from_run(len(states),a,b,mw,mh,ax,ay,loc)
            states.append(st)
            states_axis[axis].append(st)
            ids[a:b]=st.index
    labels[(ids_w>=0)|(ids_h>=0)]='ESTABLE'

    gesture_w=np.zeros(n,dtype=np.float32)
    gesture_h=np.zeros(n,dtype=np.float32)
    assessments=[]
    for axis, axis_states, target_score, smoothed in (
            ('W',states_axis['W'],gesture_w,mw),
            ('H',states_axis['H'],gesture_h,mh)):
        for j,b in enumerate(axis_states[1:],1):
            # Comparar con varios antecedentes cercanos, porque una pequena
            # meseta espuria no deberia ocultar una referencia previa valida.
            previous=[a for a in axis_states[max(0,j-5):j]
                      if b.start-a.end <= round(args.max_transition_seconds*fps)]
            future=[c for c in axis_states[j+1:]
                    if c.start-b.end <= round(args.return_seconds*fps)]
            if not previous or not future:
                continue
            accepted=None
            rejected=None
            for a in reversed(previous):
                for c in future:
                    if min(epochs[a.end-1],epochs[b.start],epochs[c.start])<0 or not (
                            epochs[a.end-1]==epochs[b.start]==epochs[c.start]):
                        score,reason,metrics=0.,'IDENTIDAD_INTERRUMPIDA',{'axis':axis}
                    else:
                        score,reason,metrics=compare_axis_triplet(a,b,c,axis,fps,args,pitch)
                    item=dict(axis=axis,a=a.index,b=b.index,c=c.index,
                              status=reason,score=score,metrics=metrics)
                    if rejected is None:
                        rejected=item
                    if score>0 and (accepted is None or score>accepted['score']):
                        accepted=item
            item=accepted or rejected
            if item is not None:
                assessments.append(item)
            if accepted:
                target_score[b.start:b.end]=np.maximum(
                    target_score[b.start:b.end],accepted['score'])
                # Extender a zonas breves entre mesetas que mantengan el nivel
                # de B; nunca atravesar un tramo de identidad distinta.
                a=states[accepted['a']]
                c=states[accepted['c']]
                axis_value='width' if axis=='W' else 'height'
                reference=getattr(a,axis_value)
                required=math.log(getattr(b,axis_value)/reference)
                for k in range(max(a.end,b.start-round(.5*fps)),min(n,c.start)):
                    if epochs[k] != epochs[b.start] or not np.isfinite(smoothed[k]):
                        continue
                    if math.log(smoothed[k]/reference)>=max(args.gesture_change*.8,required*.60):
                        target_score[k]=max(target_score[k],accepted['score'])

    # Complemento: una meseta puede estar dentro de un unico estado largo,
    # y no generar tres estados A-B-C. Detectarla por contraste local offline.
    local_w, local_h, local_events = detect_local_plateaus(mw, mh, epochs, fps, args)
    gesture_w = np.maximum(gesture_w, local_w)
    gesture_h = np.maximum(gesture_h, local_h)
    gesture=np.maximum(gesture_w,gesture_h)
    labels[gesture_w>0]='GESTO_W'
    labels[gesture_h>0]='GESTO_H'
    labels[(gesture_w>0)&(gesture_h>0)]='GESTO_WH'
    ids=np.where(ids_w>=0,ids_w,ids_h)
    return dict(frame=frame_values,width=width,height=height,
                smoothed_w=mw,smoothed_h=mh,cvw=cvw,cvh=cvh,
                drift_w=dw,drift_h=dh,stable_w=stable_w,stable_h=stable_h,
                stable=stable_w|stable_h,state_id=ids,state_id_w=ids_w,state_id_h=ids_h,
                labels=labels,gesture=gesture,gesture_w=gesture_w,gesture_h=gesture_h,
                states=states,states_axis=states_axis,assessments=assessments,
                plateau_events=local_events)

# -----------------------------------------------------------------------------
# Score final y archivos de diagnostico.
# -----------------------------------------------------------------------------

def compute_scores(rows, analysis, players_all, balls_all, atd_first, pitch, fps, args):
    movement = compute_player_movement(players_all, pitch, fps, window=25)
    classifier = DeadBallScore(enter=args.enter_score, exit=args.exit_score,
                               enter_frames=args.enter_frames,
                               exit_frames=args.exit_frames,
                               pose_weight=args.pose_weight,
                               bbox_weight=args.bbox_weight,
                               players_weight=args.players_weight,
                               ball_out_weight=args.ball_out_weight,
                               ball_out_margin=args.ball_out_margin,
                               ball_out_frames=args.ball_out_frames,
                               ball_in_frames=args.ball_in_frames,
                               hold_until_restart=args.hold_until_restart,
                               restart_timeout_frames=round(args.restart_timeout_seconds*fps),
                               restart_cooldown_frames=round(args.restart_cooldown_seconds*fps))
    restart_data = detect_ball_restarts(players_all, balls_all, pitch, fps, args)
    n = len(rows)
    player_speed = np.full(n, np.nan)
    player_quiet = np.full(n, np.nan)
    score = np.zeros(n, dtype=float)
    dead = np.zeros(n, dtype=bool)
    contribution_pose = np.zeros(n, dtype=float)
    contribution_bbox = np.zeros(n, dtype=float)
    contribution_players = np.zeros(n, dtype=float)
    ball_x = np.full(n, np.nan)
    ball_y = np.full(n, np.nan)
    ball_status = np.full(n, 'DESCONOCIDA', dtype=object)
    ball_factor = np.zeros(n, dtype=float)
    score_before_ball = np.zeros(n, dtype=float)
    contribution_ball = np.zeros(n, dtype=float)
    nearest_player_m = np.full(n, np.nan)
    restart_detected = np.zeros(n, dtype=bool)
    state_reason = np.full(n, '', dtype=object)
    exit_unlocked = np.zeros(n, dtype=bool)
    for i, row in enumerate(rows):
        idx = row['frame'] - atd_first
        p = float(movement[idx]) if 0 <= idx < len(movement) and np.isfinite(movement[idx]) else None
        ball = balls_all[idx] if 0 <= idx < len(balls_all) else None
        restart = bool(restart_data['release'][idx]) if 0 <= idx < len(restart_data['release']) else False
        if 0 <= idx < len(restart_data['nearest']):
            nearest_player_m[i] = restart_data['nearest'][idx]
        d, sc, pq, contributions, ball_data = classifier.step(
            p, float(analysis['gesture_pose'][i]), float(analysis['gesture_bbox'][i]), ball,
            restart=restart)
        restart_detected[i] = restart
        state_reason[i] = classifier.last_reason
        exit_unlocked[i] = classifier.restart_unlocked
        dead[i] = d
        score[i] = sc
        contribution_pose[i], contribution_bbox[i], contribution_players[i] = contributions
        (ball_x[i], ball_y[i], ball_status[i], ball_factor[i],
         score_before_ball[i], contribution_ball[i]) = ball_data
        if p is not None:
            player_speed[i] = p
        if pq is not None:
            player_quiet[i] = pq
    return dict(speed=player_speed, quiet=player_quiet, score=score, dead=dead,
                contribution_pose=contribution_pose,
                contribution_bbox=contribution_bbox,
                contribution_players=contribution_players,
                ball_x=ball_x, ball_y=ball_y, ball_status=ball_status,
                ball_factor=ball_factor, score_before_ball=score_before_ball,
                contribution_ball=contribution_ball,
                nearest_player_m=nearest_player_m,
                restart_detected=restart_detected,
                state_reason=state_reason, exit_unlocked=exit_unlocked)


def write_csv_results(out_dir, start, stop, fps, rows, analysis, scoring):
    prefix = f'referee_estados_{start}_{stop}'
    csv_path = os.path.join(out_dir, prefix + '.csv')
    states_path = os.path.join(out_dir, prefix + '_estados.csv')
    transitions_path = os.path.join(out_dir, prefix + '_transiciones.csv')
    first = rows[0]['frame']

    def numeric(value):
        if value is None:
            return ''
        try:
            v = float(value)
            return round(v, 6) if np.isfinite(v) else ''
        except (TypeError, ValueError):
            return ''

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'segundo', 'arbitro_detectado', 'bbox_ancho', 'bbox_alto',
                         'bbox_area', 'ancho_mediana', 'alto_mediana',
                         'cv_log_ancho', 'cv_log_alto', 'estable_local',
                         'estado_id', 'estado_temporal', 'score_gesto', 'estable_ancho', 'estable_alto', 'estado_id_ancho', 'estado_id_alto', 'score_gesto_ancho', 'score_gesto_alto',
                         'velocidad_jugadores', 'quietud_jugadores', 'score_D',
                         'prediccion_D', 'estado_detector', 'mundo_x', 'mundo_y', 'score_gesto_bruto', 'velocidad_arbitro_m_s', 'oscilacion_ancho', 'fiabilidad_gesto', 'score_gesto_bbox', 'score_gesto_pose', 'pose_bruto', 'pose_brazo_arriba', 'pose_brazo_lateral', 'pose_articulaciones_visibles', 'aporte_pose_fusion', 'aporte_bbox_G', 'aporte_pose_G', 'pose_extension_codo', 'pose_orientacion_brazo', 'pose_respaldo_codo', 'pose_angulo_codo_izq', 'pose_angulo_codo_der', 'pose_direccion_izq', 'pose_direccion_der', 'pose_score_mediana', 'pose_factor_carrera', 'pose_hold_activo', 'pose_up_raw', 'pose_up_fast', 'pose_up_meseta', 'pose_up_combinado', 'aporte_pose_D', 'aporte_bbox_D', 'aporte_jugadores_D', 'pelota_x', 'pelota_y', 'pelota_estado', 'pelota_evidencia', 'score_D_sin_pelota', 'aporte_pelota_D', 'distancia_pelota_jugador_m', 'saque_detectado', 'motivo_estado_D', 'salida_habilitada_por_saque'])
        for frame in range(start, stop):
            i = frame - first
            row = rows[i]
            w = analysis['width'][i]
            h = analysis['height'][i]
            world = row.get('world') or [None, None]
            writer.writerow([frame, round(frame/fps, 4), int(row['referee'] is not None),
                             numeric(w), numeric(h), numeric(w*h),
                             numeric(analysis['smoothed_w'][i]),
                             numeric(analysis['smoothed_h'][i]),
                             numeric(analysis['cvw'][i]),
                             numeric(analysis['cvh'][i]), int(analysis['stable'][i]),
                             int(analysis['state_id'][i]), analysis['labels'][i],
                             numeric(analysis['gesture'][i]),
                             int(analysis['stable_w'][i]),int(analysis['stable_h'][i]),
                             int(analysis['state_id_w'][i]),int(analysis['state_id_h'][i]),
                             numeric(analysis['gesture_w'][i]),numeric(analysis['gesture_h'][i]),
                             numeric(scoring['speed'][i]), numeric(scoring['quiet'][i]),
                             numeric(scoring['score'][i]), int(scoring['dead'][i]),
                             row['status'], numeric(world[0]), numeric(world[1]),
                             numeric(analysis['gesture_unfiltered'][i]), numeric(analysis['referee_speed'][i]),
                             numeric(analysis['referee_oscillation'][i]), numeric(analysis['gesture_reliability'][i]), numeric(analysis['gesture_bbox'][i]), numeric(analysis['gesture_pose'][i]), numeric(analysis['pose_raw'][i]), numeric(analysis['pose_up'][i]), numeric(analysis['pose_side'][i]), numeric(analysis['pose_coverage'][i]), numeric(analysis['gesture_pose_contribution'][i]), numeric(analysis['gesture_bbox_contribution'][i]), numeric(analysis['gesture_pose_contribution'][i]), numeric(analysis['pose_extension'][i]), numeric(analysis['pose_orientation'][i]), numeric(analysis['pose_elbow_fallback'][i]), numeric(analysis['pose_angle_left'][i]), numeric(analysis['pose_angle_right'][i]), numeric(analysis['pose_direction_left'][i]), numeric(analysis['pose_direction_right'][i]), numeric(analysis['pose_filtered'][i]), numeric(analysis['pose_run_factor'][i]), int(analysis['pose_hold_active'][i]), numeric(analysis['pose_up_raw'][i]), numeric(analysis['pose_up_fast'][i]), numeric(analysis['pose_up_plateau'][i]), numeric(analysis['pose_up_combined'][i]), numeric(scoring['contribution_pose'][i]), numeric(scoring['contribution_bbox'][i]), numeric(scoring['contribution_players'][i]), numeric(scoring['ball_x'][i]), numeric(scoring['ball_y'][i]), scoring['ball_status'][i], numeric(scoring['ball_factor'][i]), numeric(scoring['score_before_ball'][i]), numeric(scoring['contribution_ball'][i]), numeric(scoring['nearest_player_m'][i]), int(scoring['restart_detected'][i]), scoring['state_reason'][i], int(scoring['exit_unlocked'][i])])

    with open(states_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['id', 'eje', 'frame_inicio', 'frame_fin_inclusivo', 'duracion_s',
                    'ancho_mediano_px', 'alto_mediano_px',
                    'cv_ancho', 'cv_alto', 'posicion_campo_x', 'posicion_campo_y'])
        for s in analysis['states']:
            w.writerow([s.index, 'W' if any(x is s for x in analysis['states_axis']['W']) else 'H', first+s.start, first+s.end-1,
                        round(s.duration/fps, 3), round(s.width, 3),
                        round(s.height, 3), round(s.cv_width, 4),
                        round(s.cv_height, 4), s.world_x, s.world_y])

    with open(transitions_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['eje', 'estado_anterior', 'estado_candidato', 'estado_posterior',
                    'frame_inicio_candidato', 'frame_fin_candidato',
                    'diagnostico', 'score_gesto', 'delta_log_ancho', 'delta_log_alto',
                    'anisotropia', 'diferencia_retorno', 'distancia_A_C_m'])
        for item in analysis['assessments']:
            m = item['metrics']
            b = analysis['states'][item['b']]
            w.writerow([item['axis'],item['a'], item['b'], item['c'], first+b.start,
                        first+b.end-1, item['status'], round(item['score'], 4),
                        m.get('dw'), m.get('dh'), m.get('anisotropy'),
                        m.get('return_gap'), m.get('distance_ac_m')])
    print('CSV por frame:     ', csv_path)
    print('CSV estados:       ', states_path)
    print('CSV transiciones:  ', transitions_path)


# -----------------------------------------------------------------------------
# Reproduccion: todas las cajas YOLO y puntos ATD, estados, score y timeline.
# -----------------------------------------------------------------------------

def cached_detections(row):
    """Convertir cache a objetos solo para dibujar, sin pasar nuevamente por YOLO."""
    ds = []
    for item in row['detections']:
        box = np.asarray(item['box'], dtype=np.float32)
        det = Detection(box=box, confidence=item['conf'], color=None,
                        foot=np.array([(box[0]+box[2])/2, box[3]], dtype=np.float32),
                        on_pitch=item['on_pitch'], role=item['role'],
                        slot_index=item['slot'], novelty=item['novelty'])
        ds.append(det)
    idx = row.get('referee')
    selected = ds[idx] if idx is not None and idx < len(ds) else None
    return ds, selected


def fmt(x, decimals=2):
    if x is None or not np.isfinite(x):
        return '--'
    return f'{x:.{decimals}f}'


def draw_timeline(current, start, stop, width, fps, page_frames,
                  rows, analysis, scoring):
    h = 154
    img = np.full((h, width, 3), (25, 25, 28), np.uint8)
    first = rows[0]['frame']
    page = start + ((current-start)//page_frames)*page_frames
    page_stop = min(page+page_frames, stop)
    denom = max(1, width-17)
    for x in range(8, width-8):
        f = min(page_stop-1, page + round((x-8)*(page_frames-1)/denom))
        i = f-first
        if str(analysis['labels'][i]).startswith('GESTO'):
            img[6:26, x] = (0, 200, 245)    # naranja gesto
        elif analysis['state_id'][i] >= 0:
            img[6:26, x] = (85, 170, 80)    # estable
        elif np.isfinite(analysis['smoothed_w'][i]):
            img[6:26, x] = (65, 90, 155)    # transicion/inestable
        else:
            img[6:26, x] = (75, 75, 75)
        img[32:48, x] = (72, 70, 227) if scoring['dead'][i] else (67, 168, 83)
    cv2.putText(img, 'Estado: naranja gesto | verde estable | azul inestable',
                (9, 64), FONT, .43, (215, 215, 215), 1, cv2.LINE_AA)
    last_score = None
    last_gesture = None
    for x in range(8, width-8):
        f = min(page_stop-1, page + round((x-8)*(page_frames-1)/denom))
        i = f-first
        score = scoring['score'][i]
        gesture = analysis['gesture'][i]
        if np.isfinite(score):
            pt = (x, 122-round(float(score)*45))
            if last_score is not None:
                cv2.line(img, last_score, pt, (45, 215, 255), 1, cv2.LINE_AA)
            last_score = pt
        else:
            last_score = None
        gp = (x, 122-round(float(gesture)*45))
        if last_gesture is not None:
            cv2.line(img, last_gesture, gp, (230, 85, 185), 1, cv2.LINE_AA)
        last_gesture = gp
    cursor = 8 + int((current-page)*denom/max(1,page_frames-1))
    cv2.line(img, (cursor, 4), (cursor, 126), YELLOW, 2)
    pages = (stop-start+page_frames-1)//page_frames
    number = (current-start)//page_frames+1
    cv2.putText(img, f'Score D amarillo | Gesto magenta | Pagina {number}/{pages} '
                f'({page/fps:.2f}s - {(page_stop-1)/fps:.2f}s)',
                (9, 139), FONT, .42, (225, 225, 225), 1, cv2.LINE_AA)
    cv2.putText(img, 'Clic/arrastre barra: buscar  ,/. frame  A/D: 1 s  [/] pagina  Espacio: pausa',
                (9, 151), FONT, .33, (190, 190, 190), 1, cv2.LINE_AA)
    return img



# -----------------------------------------------------------------------------
# YOLO Pose: articulaciones y analisis de brazos; cache Detect independiente.
# -----------------------------------------------------------------------------
POSE_NAMES = ('nariz','ojo izq','ojo der','oreja izq','oreja der',
              'hombro izq','hombro der','codo izq','codo der',
              'muneca izq','muneca der','cadera izq','cadera der',
              'rodilla izq','rodilla der','tobillo izq','tobillo der')
POSE_LIMBS = ((5,7),(7,9),(6,8),(8,10),
              (5,6),(5,11),(6,12),(11,12),
              (11,13),(13,15),(12,14),(14,16))
POSE_PALETTE = {'brazos': (0,210,255), 'caderas': (255,110,230),
                'piernas': (90,245,110), 'torso': (255,225,70)}


def joint_group(j):
    if j in (7,8,9,10): return 'brazos'
    if j in (11,12): return 'caderas'
    if j in (13,14,15,16): return 'piernas'
    return 'torso'


class RefereePoseViewer:
    """Pose sobre recorte de la caja Detect previamente seleccionada.

    Conserva detecciones Pose en RAM; el analisis se integra posteriormente.
    Se guardan en RAM por frame para facilitar el scrubbing de la timeline.
    """
    def __init__(self, args):
        from ultralytics import YOLO
        path = args.pose_weights
        if not os.path.isabs(path):
            local = os.path.join(ROOT, 'models', path)
            if os.path.isfile(local):
                path = local
        print(f'Inicializando YOLO Pose: {path} (solo para visualizacion)')
        self.model = YOLO(path)
        self.device = (int(args.device) if str(args.device).isdecimal() else
                       None if args.device == 'auto' else args.device)
        self.args = args
        self.cache = {}

    def get(self, frame_idx, image, ref_box):
        if frame_idx in self.cache:
            return self.cache[frame_idx]
        if ref_box is None:
            self.cache[frame_idx] = None
            return None
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [float(x) for x in ref_box]
        bw, bh = x2-x1, y2-y1
        if bw < 3 or bh < 16:
            return None
        padx = max(20., bw*self.args.pose_padding)
        pady = max(16., bh*self.args.pose_padding)
        left = max(0, int(x1-padx)); right = min(w, int(x2+padx))
        top = max(0, int(y1-pady)); bottom = min(h, int(y2+pady))
        crop = image[top:bottom,left:right]
        if crop.size == 0:
            return None
        # Amplificacion del recorte: el arbitro mide ~50 px en el video tactico.
        factor = max(1., self.args.pose_min_person_pixels / max(1.,bh))
        factor = min(factor, self.args.pose_max_upscale)
        enlarged = cv2.resize(crop, None, fx=factor, fy=factor,
                              interpolation=cv2.INTER_CUBIC if factor>1 else cv2.INTER_LINEAR)
        result = self.model.predict(enlarged, imgsz=self.args.pose_imgsz,
                                    conf=self.args.pose_conf, classes=[0],
                                    device=self.device, verbose=False)[0]
        if (result.boxes is None or len(result.boxes)==0 or
            result.keypoints is None or result.keypoints.xy is None):
            out = {'points': None, 'conf': None, 'crop': (left,top,right,bottom),
                   'status': 'Pose: sin personas'}
            self.cache[frame_idx] = out
            return out
        boxes = result.boxes.xyxy.cpu().numpy()
        centers = .5*(boxes[:,:2]+boxes[:,2:])
        expected = np.array([((x1+x2)/2-left)*factor, ((y1+y2)/2-top)*factor])
        # Elegir la persona cuya caja corresponda al arbitro, no otra del crop.
        selected = int(np.argmin(np.linalg.norm(centers-expected,axis=1)))
        pxy = result.keypoints.xy[selected].cpu().numpy()
        pc = (result.keypoints.conf[selected].cpu().numpy()
              if result.keypoints.conf is not None else np.ones(len(pxy),np.float32))
        coords = pxy/factor + np.array([left,top])
        reliable = (np.isfinite(coords).all(axis=1) & (pc>=self.args.pose_kpt_conf) &
                    np.any(pxy>0,axis=1))
        coords[~reliable] = np.nan
        out = {'points': coords, 'conf': pc,
               'crop': (left,top,right,bottom),
               'status': f'Pose: {int(reliable.sum())}/17 articulaciones'}
        self.cache[frame_idx] = out
        return out



# -----------------------------------------------------------------------------
# YOLO Pose: inferencia offline y señal de gesto con brazos (no usa etiquetas GT).
# -----------------------------------------------------------------------------
def pose_cache_rows(args, cap, rows, out_dir, fps):
    """Reutiliza Pose por frame; el cache Detect sigue siendo independiente.

    El cache queda asociado al video y a la caja YOLO seleccionada en cada frame.
    """
    import hashlib
    first, stop = rows[0]['frame'], rows[-1]['frame'] + 1
    weight = args.pose_weights
    if not os.path.isabs(weight):
        local = os.path.join(ROOT, 'models', weight)
        if os.path.isfile(local):
            weight = local
    source_hash = hashlib.sha256(json.dumps([
        (r['frame'], r['detections'][r['referee']]['box'] if r.get('referee') is not None else None)
        for r in rows], separators=(',', ':')).encode()).hexdigest()
    weight_info = (file_signature(weight) if os.path.isfile(weight) else str(weight))
    metadata = dict(schema=1, game=str(GAME), first=first, stop=stop,
                    source_hash=source_hash, weights=weight_info,
                    pose_conf=args.pose_conf, pose_imgsz=args.pose_imgsz,
                    pose_kpt_conf=args.pose_kpt_conf, padding=args.pose_padding,
                    person_pixels=args.pose_min_person_pixels,
                    upscale=args.pose_max_upscale)
    path = os.path.join(out_dir, f'referee_pose_{first}_{stop}_cache.jsonl')
    if not args.pose_recompute and os.path.isfile(path):
        try:
            with open(path, encoding='utf-8') as fd:
                header = json.loads(next(fd))
                data = [json.loads(line) for line in fd]
            if header.get('metadata') == metadata and len(data) == len(rows) and all(
                    x.get('frame') == r['frame'] for x,r in zip(data,rows)):
                print(f'Cache Pose reutilizado: {path}')
                return data
        except (ValueError, OSError, StopIteration, KeyError):
            pass
    print(f'Calculando YOLO Pose en {len(rows)} frames (cache separado de Detect)...')
    viewer = RefereePoseViewer(args)
    output = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    tmp = path+'.partial'
    try:
        with open(tmp, 'w', encoding='utf-8') as fd:
            fd.write(json.dumps({'metadata': metadata})+'\n')
            for i, r in enumerate(rows):
                success, image = cap.read()
                if not success:
                    raise IOError(f'No se pudo leer video en frame {r["frame"]}')
                idx = r.get('referee')
                box = r['detections'][idx]['box'] if idx is not None else None
                pose = viewer.get(r['frame'], image, box)
                points = pose['points'] if pose is not None else None
                conf = pose.get('conf') if pose is not None else None
                serial = dict(frame=r['frame'],
                              points=[[float(v) if np.isfinite(v) else None for v in point]
                                      for point in points] if points is not None else None,
                              conf=[float(v) for v in conf] if conf is not None else None,
                              crop=list(pose['crop']) if pose is not None and pose.get('crop') else None,
                              status=pose['status'] if pose is not None else 'Pose: sin arbitro')
                fd.write(json.dumps(serial, allow_nan=False)+'\n')
                output.append(serial)
                if (i+1) % max(1,round(fps)*5) == 0:
                    print(f'  Pose: {i+1}/{len(rows)}')
        os.replace(tmp,path)
    except BaseException:
        if os.path.exists(tmp): os.remove(tmp)
        raise
    print(f'Cache Pose guardado: {path}')
    return output


def _pose_arm_details(record, args):
    """Medidas geometricas independientes de la direccion del gesto.

    Para cada brazo: angulo interior del codo (hombro-codo-muneca),
    separacion angular del brazo respecto al eje del torso hacia abajo,
    y respaldo parcial por un codo claramente elevado si falta la muneca.
    No inventa puntos ausentes y no presupone que derecha/izquierda en
    imagen correspondan a derecha/izquierda anatomicas.
    """
    details = {'strength': 0., 'extension': 0., 'orientation': 0.,
               'elbow_fallback': 0., 'up': 0., 'lateral': 0.,
               'coverage': 0., 'up_strength': 0., 'angle_left': float('nan'),
               'angle_right': float('nan'), 'direction_left': float('nan'),
               'direction_right': float('nan')}
    points = record.get('points')
    if points is None or len(points) != 17:
        return details
    p = np.asarray([[np.nan if x is None else float(x) for x in point]
                    for point in points], dtype=float)
    conf = record.get('conf')
    conf = np.asarray(conf if conf is not None else np.ones(17), dtype=float)
    if conf.shape != (17,):
        return details
    valid = np.isfinite(p).all(axis=1) & (conf >= args.pose_kpt_conf)

    # Referencia anatomica hacia los pies; permite inclinacion del torso.
    if valid[5] and valid[6] and valid[11] and valid[12]:
        torso = (p[11] + p[12] - p[5] - p[6]) / 2.
        torso_quality = 1.
    elif valid[5] and valid[11]:
        torso = p[11]-p[5]
        torso_quality = .9
    elif valid[6] and valid[12]:
        torso = p[12]-p[6]
        torso_quality = .9
    else:
        # No usar una orientacion corporal inventada como evidencia fuerte.
        torso = np.array([0., 1.])
        torso_quality = .70
    norm = np.linalg.norm(torso)
    if norm < 1e-5:
        return details
    down = torso/norm
    torso_len = max(norm, 3.)

    observed = []
    for side, (sh, el, wr) in zip(('left','right'), ((5,7,9),(6,8,10))):
        if not (valid[sh] and valid[el]):
            continue
        upper = p[el]-p[sh]
        upper_norm = np.linalg.norm(upper)
        if upper_norm < 2.:
            continue
        elbow_dir = upper/upper_norm
        elbow_direction = float(np.degrees(np.arccos(np.clip(np.dot(elbow_dir,down),-1,1))))
        details['direction_'+side] = elbow_direction
        if valid[wr]:
            fore = p[wr]-p[el]
            fore_norm = np.linalg.norm(fore)
            if fore_norm >= 2.:
                v1 = -upper / upper_norm
                v2 = fore / fore_norm
                angle = float(np.degrees(np.arccos(np.clip(np.dot(v1,v2),-1,1))))
                details['angle_'+side] = angle
                # Extension: no confundir un codo muy doblado con un brazo recto.
                extension = float(np.clip((angle-115.)/(165.-115.),0.,1.))
                # Medimos la direccion general hombro->muneca, no el signo X/Y.
                arm = p[wr]-p[sh]
                arm_norm = np.linalg.norm(arm)
                direction = float(np.degrees(np.arccos(np.clip(np.dot(arm/arm_norm,down),-1,1)))) if arm_norm > 1 else 0.
                # Vertical arriba ~180, lateral ~90, diagonal ~45-135.
                orientation = float(np.clip((direction-30.)/(85.-30.),0.,1.))
                extent = float(np.clip((arm_norm/max(torso_len,upper_norm)-.40)/.50,0.,1.))
                score = extension*orientation*(.65+.35*extent)*torso_quality
                details['extension'] = max(details['extension'],extension)
                details['orientation'] = max(details['orientation'],orientation)
                details['up'] = max(details['up'],float(np.clip((direction-110.)/55.,0,1)))
                # Canal especial: brazo apuntando hacia ARRIBA. No exige codo
                # casi recto: en las tomas lejanas el codo puede ser impreciso.
                # Solo consideramos la geometria observada en este frame.
                above = float(np.clip((direction-args.pose_up_angle_start) /
                                      max(1., args.pose_up_angle_full-args.pose_up_angle_start),0,1))
                above *= (.60 + .40*extension) * torso_quality
                details['up_strength'] = max(details['up_strength'], above)
                details['lateral'] = max(details['lateral'],float(np.clip(1.-abs(direction-90.)/55.,0,1)))
                observed.append(('full',score))
                continue
        # Respaldo cuando NO se ve la muneca: evidencia parcial por codo elevado
        # o muy separado del torso. Nunca alcanza por si sola el score completo.
        lifted = float(np.clip((elbow_direction-55.)/(115.-55.),0.,1.))
        score = .67*lifted*torso_quality
        details['elbow_fallback'] = max(details['elbow_fallback'],score)
        # Si falta la muneca, el codo alto sigue siendo una pista de señal
        # vertical. Su fuerza es limitada porque no vemos el antebrazo.
        up_elbow = float(np.clip((elbow_direction-args.pose_up_angle_start-10.) /
                                 max(1.,args.pose_up_angle_full-args.pose_up_angle_start),0,1))
        details['up_strength'] = max(details['up_strength'], .82*up_elbow*torso_quality)
        details['orientation'] = max(details['orientation'],lifted)
        observed.append(('elbow',score))
    if observed:
        details['strength'] = max(score for _,score in observed)
        details['coverage'] = min(1.,sum(1. if kind=='full' else .5 for kind,_ in observed)/2.)
    return details


def score_pose_frame(record, args):
    """Compatibilidad: (gesto, arriba, lateral, cobertura)."""
    d = _pose_arm_details(record,args)
    return d['strength'],d['up'],d['lateral'],d['coverage']


def fuse_gesture_scores(bbox_score, pose_score, pose_weight=.9, bbox_weight=.2):
    """Fusion independiente, sin reparto fijo de pesos.

    G = 1 - (1 - wP*P)*(1 - wB*B).
    El aporte Pose es wP*P y el aporte BBox es (1-wP*P)*wB*B.
    Ambos suman exactamente G. Los pesos individuales estan en [0,1].
    """
    bbox = np.nan_to_num(np.clip(np.asarray(bbox_score, dtype=float), 0., 1.), nan=0.)
    pose = np.nan_to_num(np.clip(np.asarray(pose_score, dtype=float), 0., 1.), nan=0.)
    contribution_pose = float(pose_weight)*pose
    contribution_bbox = (1.-contribution_pose)*float(bbox_weight)*bbox
    fused = contribution_pose+contribution_bbox
    return (fused.astype(np.float32), contribution_bbox.astype(np.float32),
            contribution_pose.astype(np.float32))


def reconstruct_upward_pose_plateaus(up_raw, fps, args):
    """Agrupa observaciones verticales dispersas en mesetas retrospectivas.

    Solo afecta el canal UP. No rellena los extremos ni modifica Pose lateral.
    Exige varias detecciones reales y un tramo temporal minimo; entre dos
    observaciones cercanas el score se sostiene como evidencia inferida.
    """
    values = np.nan_to_num(np.clip(np.asarray(up_raw, dtype=float), 0., 1.), nan=0.)
    result = np.zeros_like(values)
    n = len(values)
    if not n:
        return result
    hits = np.flatnonzero(values >= args.pose_up_min_score)
    if not len(hits):
        return result
    max_missing = max(0, round(args.pose_up_plateau_gap_seconds * fps))
    min_duration = max(1, round(args.pose_up_plateau_min_seconds * fps))
    groups = []
    group = [int(hits[0])]
    for idx in hits[1:]:
        idx = int(idx)
        # Cuantos frames no detectados hay ENTRE las observaciones.
        if idx - group[-1] - 1 <= max_missing:
            group.append(idx)
        else:
            groups.append(group)
            group = [idx]
    groups.append(group)

    for group in groups:
        first, last = group[0], group[-1]
        if (len(group) < args.pose_up_plateau_min_hits or
                last-first+1 < min_duration):
            continue
        # Se reconstruye solo el interior respaldado por observaciones reales.
        # El suelo alto es intencional: es una postura vertical reconocida.
        plateau = float(np.clip(max(args.pose_up_score_floor,
                                    np.percentile(values[group], 75)), 0., 1.))
        result[first:last+1] = plateau
    return result


def integrate_pose_signal(analysis, pose_rows, fps, args):
    """Evalua extension/orientacion, permanencia, carrera y fusiona con bbox."""
    n=len(pose_rows)
    assert len(analysis['gesture']) == n
    details=[_pose_arm_details(r,args) for r in pose_rows]
    raw=np.array([d['strength'] for d in details],dtype=float)
    up=np.array([d['up'] for d in details],dtype=float)
    side=np.array([d['lateral'] for d in details],dtype=float)
    coverage=np.array([d['coverage'] for d in details],dtype=float)
    up_raw=np.array([d['up_strength'] for d in details],dtype=float)
    # Mediana temporal: un brazo no debe cambiar de estado cada frame.
    from numpy.lib.stride_tricks import sliding_window_view
    k=max(1,args.pose_median_frames)
    filtered=np.median(sliding_window_view(np.pad(raw,(k//2,k//2),mode='edge'),k),axis=1)
    active=filtered>=args.pose_arm_threshold
    max_gap=max(0,round(args.pose_bridge_seconds*fps))
    if max_gap:
        rr=ranges_of_true(active)
        for (a,b),(c,d) in zip(rr,rr[1:]):
            # Permitir huecos cortos por oclusion, incluso si no hay articulaciones.
            if 0<c-b<=max_gap:
                active[b:c]=True
    minimum=max(1,round(args.pose_min_hold_seconds*fps))
    sustained=np.zeros(n,dtype=float)
    for a,b in ranges_of_true(active):
        if b-a>=minimum:
            sustained[a:b]=np.maximum(args.pose_arm_threshold,filtered[a:b])
    # Un arbitro puede senalar MIENTRAS CORRE: no descartar gestos por velocidad.
    # Penalizacion minima y configurable para Pose (10% maximo por defecto).
    # La caja completa sigue teniendo su propia penalizacion por carrera.
    speed=analysis.get('referee_speed',np.full(n,np.nan))
    run_penalty=np.ones(n,dtype=float)
    valid=np.isfinite(speed)
    run_penalty[valid]=1.-args.pose_run_penalty*np.clip(
        (speed[valid]-args.ref_run_low)/max(1e-5,args.ref_run_high-args.ref_run_low),0,1)
    # Vertical: aceptar senales de apenas 2-3 frames, sin suavizado que
    # borraria las observaciones aisladas. Exigimos frames consecutivos con
    # mediciones reales (no inventamos articulaciones).
    up_active=up_raw >= args.pose_up_min_score
    up_sustained=np.zeros(n,dtype=float)
    for a,b in ranges_of_true(up_active):
        if b-a >= args.pose_up_min_frames:
            up_sustained[a:b]=np.maximum(args.pose_up_score_floor, up_raw[a:b])
    # Reconstruir una unica meseta si el gesto vertical fue observado varias
    # veces con huecos breves. Los gestos laterales siguen sin este tratamiento.
    up_plateau = reconstruct_upward_pose_plateaus(up_raw, fps, args)
    up_combined = np.maximum(up_sustained, up_plateau)
    # La carrera NO penaliza un brazo levantado. Solo el canal de gestos
    # generales conserva la penalizacion previa.
    pose_score=np.maximum(sustained*run_penalty, up_combined).clip(0,1).astype(np.float32)
    baseline=np.asarray(analysis['gesture'],dtype=np.float32).copy()
    combined, bbox_contribution, pose_contribution = fuse_gesture_scores(
        baseline,pose_score,args.pose_weight,args.bbox_weight)
    analysis['gesture_bbox']=baseline
    analysis['gesture_pose']=pose_score
    analysis['pose_filtered']=filtered.astype(np.float32)
    analysis['pose_run_factor']=run_penalty.astype(np.float32)
    analysis['pose_hold_active']=active.astype(bool) & (sustained>0)
    analysis['gesture_bbox_contribution']=bbox_contribution
    analysis['gesture_pose_contribution']=pose_contribution
    analysis['pose_raw']=raw
    analysis['pose_up_raw']=up_raw.astype(np.float32)
    analysis['pose_up_fast']=up_sustained.astype(np.float32)
    analysis['pose_up_plateau']=up_plateau.astype(np.float32)
    analysis['pose_up_combined']=up_combined.astype(np.float32)
    analysis['pose_up']=up
    analysis['pose_side']=side
    analysis['pose_coverage']=coverage
    analysis['pose_extension']=np.array([d['extension'] for d in details])
    analysis['pose_orientation']=np.array([d['orientation'] for d in details])
    analysis['pose_elbow_fallback']=np.array([d['elbow_fallback'] for d in details])
    for key in ('angle_left','angle_right','direction_left','direction_right'):
        analysis['pose_'+key]=np.array([d[key] for d in details])
    analysis['gesture']=combined
    new=pose_contribution>1e-4
    analysis['labels'][new & (baseline>0)]='GESTO_BBOX_POSE'
    analysis['labels'][new & (baseline<=0)]='GESTO_POSE'
    return analysis


def pose_from_record(row):
    if row is None:
        return None
    if row.get('points') is None:
        return {'points':None,'conf':None,'crop':row.get('crop'), 'status':row.get('status','Pose: sin datos')}
    pts=np.array([[float('nan') if v is None else float(v) for v in pt] for pt in row['points']])
    return {'points':pts,'conf':np.asarray(row.get('conf') or np.ones(len(pts))),
            'crop':row.get('crop'),'status':row.get('status','Pose')}


def draw_pose_skeleton(image, pose, offset=(0.,0.), scale=1., thick=2, names=False):
    if pose is None or pose.get('points') is None:
        return
    pts = pose['points']
    def coord(j):
        return tuple(np.rint((pts[j]-offset)*scale).astype(int))
    for a,b in POSE_LIMBS:
        if np.isfinite(pts[[a,b]]).all():
            group = 'piernas' if a>=13 or b>=13 else (
                'brazos' if a in (7,8,9,10) or b in (7,8,9,10) else 'torso')
            cv2.line(image, coord(a), coord(b), POSE_PALETTE[group], thick, cv2.LINE_AA)
    for j in range(5, min(len(pts),17)):
        if not np.isfinite(pts[j]).all():
            continue
        p = coord(j)
        color = POSE_PALETTE[joint_group(j)]
        cv2.circle(image,p,max(3,thick+1),(0,0,0),-1,cv2.LINE_AA)
        cv2.circle(image,p,max(2,thick),color,-1,cv2.LINE_AA)
        if names:
            cv2.putText(image,str(j),(p[0]+4,p[1]-4),FONT,.40,color,1,cv2.LINE_AA)


def draw_pose_zoom(image, pose, ref_box, viewer_args):
    """Inset inferior derecho que permite inspeccionar articulaciones subpixel."""
    if pose is None or ref_box is None:
        return
    h,w=image.shape[:2]
    left,top,right,bottom=pose['crop']
    crop = image[top:bottom,left:right].copy()
    if crop.size == 0:
        return
    target_h = min(viewer_args.pose_zoom_height, max(130,h-170))
    factor = target_h/max(1,crop.shape[0])
    target_w = max(1,int(round(crop.shape[1]*factor)))
    if target_w > w//3:
        factor = (w//3)/max(1,crop.shape[1])
        target_w=int(round(crop.shape[1]*factor))
        target_h=int(round(crop.shape[0]*factor))
    zoom=cv2.resize(crop,(target_w,target_h),interpolation=cv2.INTER_NEAREST)
    draw_pose_skeleton(zoom,pose,offset=np.array([left,top]),scale=factor,
                       thick=3,names=True)
    x0=w-target_w-15; y0=h-target_h-34
    cv2.rectangle(image,(x0-3,y0-24),(x0+target_w+3,y0+target_h+3),(0,0,0),-1)
    cv2.putText(image,pose['status'],(x0,y0-8),FONT,.50,(255,255,255),1,cv2.LINE_AA)
    image[y0:y0+target_h,x0:x0+target_w]=zoom
    cv2.rectangle(image,(x0,y0),(x0+target_w,y0+target_h),(255,255,255),2)
    # Cobertura desglosada por region, para diagnosticar si sirve el modelo.
    pts = pose.get('points')
    if pts is not None:
        counts = [sum(np.isfinite(pts[j]).all() for j in group) for group in
                  ((5,6,7,8,9,10),(11,12),(13,14,15,16))]
        printline=f'Brazos {counts[0]}/6  Caderas {counts[1]}/2  Piernas {counts[2]}/4'
        cv2.putText(image,printline,(max(8,x0-20),min(h-10,y0+target_h+21)),
                    FONT,.46,(240,240,240),1,cv2.LINE_AA)


def replay(args, cap, fps, start, stop, rows, analysis, scoring,
           matrices, valid, players, balls, info, pose_rows):
    first = rows[0]['frame']
    pose_viewer = None  # Pose ya calculado en pasada offline
    pose_enabled = args.show_pose
    pose_zoom_enabled = args.pose_zoom
    window_frames = max(1, round(args.timeline_seconds*fps))
    view = dict(frame=start, seek=None, toggle=False, top_height=0, width=0)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    def mouse(event, x, y, flags, _):
        if not view['top_height']:
            return
        dragging = event == cv2.EVENT_MOUSEMOVE and bool(flags & cv2.EVENT_FLAG_LBUTTON)
        if y >= view['top_height'] and (event == cv2.EVENT_LBUTTONDOWN or dragging):
            page = start + ((view['frame']-start)//window_frames)*window_frames
            fraction = (x-8) / max(1, view['width']-17)
            view['seek'] = int(np.clip(page + round(fraction*(window_frames-1)), start, stop-1))
        elif y < view['top_height'] and event == cv2.EVENT_LBUTTONDOWN:
            view['toggle'] = True

    cv2.setMouseCallback(WINDOW, mouse)
    shown_frame = None
    next_video = None
    shown = None
    playing = True
    while True:
        target = view['seek'] if view['seek'] is not None else view['frame']
        view['seek'] = None
        if target != shown_frame:
            if target != next_video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, image = cap.read()
            if not ok:
                raise IOError(f'No se puede reproducir el frame {target}')
            next_video = target+1
            i = target-first
            row = rows[i]
            detections, referee = cached_detections(row)
            # Ejecutar Pose sobre pixels ORIGINALES, antes de dibujar anotaciones.
            pose = pose_from_record(pose_rows[i]) if pose_enabled else None
            if args.show_boxes:
                draw_yolo_detections(image, detections, referee)
            if pose_enabled:
                draw_pose_skeleton(image, pose, thick=2)
                if pose_zoom_enabled:
                    draw_pose_zoom(image, pose, referee.box if referee is not None else None, args)
            if args.show_atd and target < len(valid) and bool(valid[target]):
                draw_atd_overlay(image, matrices[target], players[i], balls[i], info)
            # see-through panel: the video darkened, not covered (PANEL_ALPHA of it hidden)
            x1, y1 = min(image.shape[1]-10, 1370), 245
            image[7:y1, 7:x1] = (image[7:y1, 7:x1] * (1 - PANEL_ALPHA)).astype(np.uint8)
            color = (0, 215, 245) if analysis['gesture'][i] >= .5 else YELLOW
            cv2.putText(image, f'YOLO frame {target}  |  {row["status"]}',
                        (16, 34), FONT, .65, color, 2, cv2.LINE_AA)
            cv2.putText(image,
                        f'Caja W:{fmt(analysis["width"][i])} H:{fmt(analysis["height"][i])}  '
                        f'Mediana W:{fmt(analysis["smoothed_w"][i])} H:{fmt(analysis["smoothed_h"][i])} '
                        f'CV W:{fmt(analysis["cvw"][i],3)} CV H:{fmt(analysis["cvh"][i],3)}',
                        (16, 65), FONT, .53, (230,230,230), 1, cv2.LINE_AA)
            state_id = analysis['state_id'][i]
            cv2.putText(image,
                        f'ESTADO: {analysis["labels"][i]}  |  BBOX W:{fmt(analysis["gesture_w"][i])} '
                        f'H:{fmt(analysis["gesture_h"][i])}  |  Arbitro:{fmt(analysis["referee_speed"][i])} m/s '
                        f'| Jugadores:{fmt(scoring["speed"][i])} m/s',
                        (16, 95), FONT, .53, (240,240,240), 1, cv2.LINE_AA)
            cv2.putText(image,
                        f'B={fmt(analysis["gesture_bbox"][i])}  P={fmt(analysis["gesture_pose"][i])}  '
                        f'| aporte B a G={fmt(analysis["gesture_bbox_contribution"][i])}  '
                        f'aporte P a G={fmt(analysis["gesture_pose_contribution"][i])}  '
                        f'| G={fmt(analysis["gesture"][i])}',
                        (16, 126), FONT, .57, (0,225,240), 2, cv2.LINE_AA)
            cv2.putText(image,
                        f'CODOS izq:{fmt(analysis["pose_angle_left"][i],0)} deg  '
                        f'der:{fmt(analysis["pose_angle_right"][i],0)} deg  '
                        f'| Ext:{fmt(analysis["pose_extension"][i])} '
                        f'Orient:{fmt(analysis["pose_orientation"][i])} '
                        f'Respaldo codo:{fmt(analysis["pose_elbow_fallback"][i])}',
                        (16, 154), FONT, .51, (215,215,215), 1, cv2.LINE_AA)
            cv2.putText(image,
                        f'POSE raw:{fmt(analysis["pose_raw"][i])} '
                        f'mediana:{fmt(analysis["pose_filtered"][i])} '
                        f'hold:{int(analysis["pose_hold_active"][i])} '
                        f'factor carrera:{fmt(analysis["pose_run_factor"][i])} '
                        f'cobertura:{fmt(analysis["pose_coverage"][i])}  '
                        f'UP raw:{fmt(analysis["pose_up_raw"][i])} fast:{fmt(analysis["pose_up_fast"][i])} meseta:{fmt(analysis["pose_up_plateau"][i])}',
                        (16, 180), FONT, .52, (200, 255, 220), 1, cv2.LINE_AA)
            score = scoring['score'][i]
            is_dead = scoring['dead'][i]
            cv2.putText(image,
                        f'SCORE D:{fmt(score)} | P:{fmt(scoring["contribution_pose"][i])} '
                        f'B:{fmt(scoring["contribution_bbox"][i])} '
                        f'Jug:{fmt(scoring["contribution_players"][i])} '
                        f'| {"D" if is_dead else "NO D"} '
                        f'| umbral:{(args.exit_score if (not args.hold_until_restart or scoring["exit_unlocked"][i]) else args.enter_score) if is_dead else args.enter_score:.2f}'
                        f' | {"SAQUE PENDIENTE" if (is_dead and args.hold_until_restart and not scoring["exit_unlocked"][i]) else "SALIDA HABILITADA" if (is_dead and scoring["exit_unlocked"][i]) else ""}',
                        (16, 207), FONT, .58,
                        (72,70,227) if is_dead else (67,168,83), 2, cv2.LINE_AA)
            status_ball = scoring['ball_status'][i]
            cv2.putText(image,
                        f'Pelota ({fmt(scoring["ball_x"][i])}, {fmt(scoring["ball_y"][i])}) '
                        f'{status_ball} | Evidencia:{fmt(scoring["ball_factor"][i])} '
                        f'| D previo:{fmt(scoring["score_before_ball"][i])} '
                        f'Aporte pelota:{fmt(scoring["contribution_ball"][i])}',
                        (16, 235), FONT, .52,
                        (70, 90, 255) if status_ball == 'FUERA' else (190, 220, 220),
                        1, cv2.LINE_AA)
            cv2.putText(image,
                        f'Estado D: {scoring["state_reason"][i]} '
                        f'| distancia pelota-jugador: {fmt(scoring["nearest_player_m"][i])} m '
                        f'| SAQUE: {int(scoring["restart_detected"][i])}',
                        (16, 263), FONT, .52,
                        (80, 230, 255) if scoring['restart_detected'][i] else (195, 215, 200),
                        1, cv2.LINE_AA)
            top = cv2.resize(image, None, fx=args.scale, fy=args.scale)
            timeline = draw_timeline(target, start, stop, top.shape[1], fps,
                                     window_frames, rows, analysis, scoring)
            shown = np.vstack((top, timeline))
            view['top_height'], view['width'] = top.shape[0], shown.shape[1]
            shown_frame = target
        view['frame'] = shown_frame
        cv2.imshow(WINDOW, shown)
        delay = max(1, round(1000/(fps*args.speed))) if playing else 30
        key = cv2.waitKey(delay) & 0xff
        if key in (ord('q'), 27):
            break
        if key == ord('k'):
            pose_enabled = not pose_enabled
            shown_frame = None
            continue
        if key == ord('z'):
            pose_zoom_enabled = not pose_zoom_enabled
            shown_frame = None
            continue
        if view['toggle'] or key in (ord(' '), ord('p')):
            view['toggle'] = False
            playing = not playing
        jumps = {ord(','):-1, ord('.'):1, ord('a'):-round(fps),
                 ord('d'):round(fps), ord('['):-window_frames, ord(']'):window_frames}
        if key in jumps:
            view['seek'] = int(np.clip(shown_frame+jumps[key], start, stop-1))
        elif playing and view['seek'] is None:
            if shown_frame >= stop-1:
                playing = False
            else:
                view['frame'] = shown_frame+1


def run(args):
    video_path, meta_path, atd_path = require(['video', 'atd_meta', 'atd_raw'])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f'No se pudo abrir el video: {video_path}')
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or GAMES[GAME]['fps']
        end_video = min(source_frames(), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        start = (args.start if args.start is not None else half_video_blocks(GAME)[0][1])
        stop = (min(end_video, start+round(args.seconds*fps)) if args.seconds is not None
                else end_video)
        if not 0 <= start < stop:
            raise ValueError('Intervalo de video invalido')
        context = round(args.context_seconds*fps)
        first = max(0, start-context)
        end = min(end_video, stop+context)
        pre_atd = max(0, first-30)  # historial de movimiento para el score
        print('Cargando homografias ...')
        matrices, _, valid = load_homography()
        info = slot_info(meta_path)
        print(f'Cargando ATD {pre_atd}..{end-1} ...')
        players_all, balls_all = load_players(atd_path, pre_atd, end, len(info))
        players = players_all[first-pre_atd:]
        balls = balls_all[first-pre_atd:]
        # The same folder as src/referee_signal.py, so both share the YOLO caches. Inside data/,
        # which git ignores: these are frame-level derived data and must never be published.
        out_dir = os.path.join(ROOT, 'data', 'derived', 'referee')
        os.makedirs(out_dir, exist_ok=True)
        rows, cache_path = detect_to_cache(args, cap, video_path, meta_path, atd_path,
                                           info, players, matrices, valid, first, end,
                                           fps, out_dir)
        pitch = np.asarray(GAMES[GAME]['pitch'], dtype=float)
        analysis = analyze_states(rows, fps, args, pitch)
        analysis = penalize_gestures_while_running(rows, analysis, fps, pitch, args)
        pose_rows = pose_cache_rows(args, cap, rows, out_dir, fps)
        analysis = integrate_pose_signal(analysis, pose_rows, fps, args)
        scoring = compute_scores(rows, analysis, players_all, balls_all, pre_atd, pitch, fps, args)
        n_gestures = sum(a['status'] == 'GESTO_CONFIRMADO_POR_RETORNO'
                         for a in analysis['assessments'])
        print(f'Mesetas locales: {len(analysis["plateau_events"])} candidatos (sin exigir retorno).')
        print(f'Analisis: {len(analysis["states"])} estados estables, '
              f'{n_gestures} gestos retrospectivos confirmados.')
        if not n_gestures:
            print('No hubo gestos confirmados con los umbrales actuales. '
                  'Podes revisar los CSV y cambiar umbrales sin recalcular YOLO.')
        write_csv_results(out_dir, start, stop, fps, rows, analysis, scoring)
        # Eventos para revisar facilmente qué detectó cada eje.
        import csv
        with open(os.path.join(out_dir, f'referee_mesetas_{start}_{stop}.csv'),
                  'w', newline='', encoding='utf-8') as fp:
            writer = csv.DictWriter(fp, fieldnames=['axis','start','end','change_log',
                'baseline','plateau','std_pre','std_post','other_change_log','score'])
            writer.writeheader()
            for event in analysis['plateau_events']:
                writer.writerow(dict(event, start=first+event['start'],
                                     end=first+event['end']-1))
        if not args.no_gui:
            replay(args, cap, fps, start, stop, rows, analysis, scoring,
                   matrices, valid, players, balls, info, pose_rows)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def build_parser():
    """Every option of this detector with its default -- also used by src/referee_signal.py,
    which runs the same detection without the window (parse_args([]) gives the defaults)."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--start', type=int, default=None, help='Frame de inicio')
    p.add_argument('--seconds', type=float, default=500., help='Segundos de reproduccion')
    p.add_argument('--context-seconds', type=float, default=12.,
                   help='Segundos extra antes y despues, para analizar pasado/futuro')
    # Parametros YOLO / identificacion del arbitro (misma logica de script anterior)
    p.add_argument('--weights', default='yolo11s.pt')
    p.add_argument('--device', default='auto', help='auto, cpu, 0, 1 ...')
    p.add_argument('--imgsz', type=int, default=1280)
    p.add_argument('--conf', type=float, default=.1)
    p.add_argument('--novelty', type=float, default=27.)
    p.add_argument('--atd-height', type=float, default=48.)
    p.add_argument('--atd-width', type=float, default=12.)
    p.add_argument('--palette-every', type=int, default=5)
    p.add_argument('--max-match-dist', type=float, default=65.)
    p.add_argument('--recompute', action='store_true', help='Ignorar cache YOLO')
    # Estados estables: parametros no modifican el cache de YOLO
    p.add_argument('--median-frames', type=int, default=7,
                   help='Mediana de ancho/alto: impar')
    p.add_argument('--fill-gap', type=int, default=2,
                   help='Interpolar detecciones perdidas cortas SOLO en analisis')
    p.add_argument('--stability-seconds', type=float, default=.44,
                   help='Ventana local para estimar estabilidad')
    p.add_argument('--stable-cv', type=float, default=.09,
                   help='Desviacion log maxima en CADA eje')
    p.add_argument('--stable-drift', type=float, default=.10,
                   help='Tendencia maxima dentro de ventana de estabilidad')
    p.add_argument('--min-state-seconds', type=float, default=.28,
                   help='Minimo de estado estable')
    p.add_argument('--join-gap', type=int, default=3)
    p.add_argument('--max-transition-seconds', type=float, default=3.0,
                   help='Separacion maxima entre estados adyacentes')
    p.add_argument('--return-seconds', type=float, default=10.,
                   help='Horizonte de busqueda de retorno con frames futuros')
    p.add_argument('--max-gesture-seconds', type=float, default=12.,
                   help='Duracion maxima del estado candidato (compatibilidad)')
    p.add_argument('--plateau-change', type=float, default=.115,
                   help='Salto minimo logaritmico para reconocer meseta local')
    p.add_argument('--plateau-std', type=float, default=.048,
                   help='Dispersion maxima de ventanas anterior y posterior')
    p.add_argument('--plateau-anisotropy', type=float, default=.055,
                   help='Diferencia de cambio entre eje de gesto y eje opuesto')
    p.add_argument('--plateau-full', type=float, default=.155,
                   help='Cambio logaritmico que da score 1.0')
    p.add_argument('--plateau-max-seconds', type=float, default=10.0,
                   help='Tiempo maximo para seguir una meseta elevada')
    p.add_argument('--gesture-change', type=float, default=.12,
                   help='Aumento minimo log(W) o log(H)')
    p.add_argument('--gesture-full', type=float, default=.30,
                   help='Amplitud de gesto de score maximo')
    p.add_argument('--min-anisotropy', type=float, default=.08,
                   help='Diferencia minima entre variacion de W y de H')
    p.add_argument('--return-tolerance', type=float, default=.10,
                   help='Maximo cambio logaritmico de dimensiones A vs C')
    p.add_argument('--max-position-m', type=float, default=5.,
                   help='Desplazamiento maximo del arbitro entre A-B-C en metros')
    p.add_argument('--max-position-body', type=float, default=2.,
                   help='Respaldo de desplazamiento en alturas corporales si no hay homografia')
    # Antifalsos positivos: carrera del arbitro + oscilacion de piernas.
    p.add_argument('--ref-run-low', type=float, default=1.1,
                   help='Velocidad del arbitro (m/s) desde donde comienza penalizacion')
    p.add_argument('--ref-run-high', type=float, default=2.8,
                   help='Velocidad del arbitro (m/s) con penalizacion maxima')
    p.add_argument('--ref-run-penalty', type=float, default=.85,
                   help='Penalizacion maxima por correr (0..1)')
    p.add_argument('--ref-osc-low', type=float, default=.035,
                   help='Oscilacion residual de ancho desde la que se penaliza')
    p.add_argument('--ref-osc-high', type=float, default=.10,
                   help='Oscilacion residual de ancho con penalizacion maxima')
    p.add_argument('--ref-osc-penalty', type=float, default=.45,
                   help='Penalizacion maxima por oscilacion (0..1)')
    # Score: SOLO gesto + quietud de jugadores, nunca pelota.
    p.add_argument('--enter-score', type=float, default=.65,
                   help='Umbral alto para INICIAR D cuando no esta activa')
    p.add_argument('--exit-score', type=float, default=.35,
                   help='Umbral bajo: D termina si el score cae por debajo durante exit-frames')
    p.add_argument('--enter-frames', type=int, default=8)
    p.add_argument('--exit-frames', type=int, default=6)
    # Reproduccion (no recalcula YOLO)
    p.add_argument('--scale', type=float, default=.75)
    p.add_argument('--speed', type=float, default=1.)
    p.add_argument('--timeline-seconds', type=float, default=20.)
    p.add_argument('--show-atd', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--show-boxes', action=argparse.BooleanOptionalAction, default=True)
    # Inspeccion de articulaciones: Pose solo en recorte del arbitro ya detectado.
    # Estas opciones no alteran el detector de D ni el cache de Detect.
    p.add_argument('--show-pose', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--pose-zoom', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--pose-weights', default='yolo11s-pose.pt')
    p.add_argument('--pose-imgsz', type=int, default=640)
    p.add_argument('--pose-conf', type=float, default=.12)
    p.add_argument('--pose-kpt-conf', type=float, default=.20)
    p.add_argument('--pose-padding', type=float, default=.65)
    p.add_argument('--pose-min-person-pixels', type=float, default=320.)
    p.add_argument('--pose-max-upscale', type=float, default=8.)
    p.add_argument('--pose-zoom-height', type=int, default=350)
    p.add_argument('--pose-recompute', action='store_true', help='Recalcular solo Pose (no Detect)')
    p.add_argument('--pose-arm-threshold', type=float, default=.48,
                   help='Umbral de extension de codo + orientacion respecto del torso')
    p.add_argument('--pose-min-hold-seconds', type=float, default=.44,
                   help='Permanencia minima del gesto de brazo')
    p.add_argument('--pose-bridge-seconds', type=float, default=.12,
                   help='Hueco breve permitido por perdida de articulaciones')
    p.add_argument('--pose-median-frames', type=int, default=5,
                   help='Mediana temporal del score de brazos, impar')
    p.add_argument('--pose-up-angle-start', type=float, default=105.,
                   help='Grados respecto a torso hacia abajo desde los que empieza evidencia vertical')
    p.add_argument('--pose-up-angle-full', type=float, default=145.,
                   help='Angulo que da evidencia vertical maxima')
    p.add_argument('--pose-up-min-score', type=float, default=.35,
                   help='Evidencia minima por frame para canal vertical rapido')
    p.add_argument('--pose-up-min-frames', type=int, default=2,
                   help='Frames consecutivos requeridos para brazo levantado')
    p.add_argument('--pose-up-plateau-gap-seconds', type=float, default=.45,
                   help='Mayor hueco entre detecciones UP para unirlas en meseta (s)')
    p.add_argument('--pose-up-plateau-min-hits', type=int, default=3,
                   help='Cantidad minima de detecciones UP reales para confirmar meseta')
    p.add_argument('--pose-up-plateau-min-seconds', type=float, default=.16,
                   help='Duracion minima entre el primer y ultimo pico de meseta (s)')
    p.add_argument('--pose-up-score-floor', type=float, default=.80,
                   help='Score minimo asignado a un gesto vertical rapido confirmado')
    p.add_argument('--pose-run-penalty', type=float, default=.10,
                   help='Reduccion maxima de Pose al correr (0.10; 0 para desactivarla)')
    p.add_argument('--pose-weight', type=float, default=.95,
                   help='Influencia independiente de Pose en D (0..1; defecto 0.90)')
    p.add_argument('--bbox-weight', type=float, default=.30,
                   help='Influencia independiente de BBox en D (0..1; defecto 0.20)')
    p.add_argument('--players-weight', type=float, default=.60,
                   help='Influencia independiente de quietud de jugadores en D (0..1; defecto 0.50)')
    p.add_argument('--ball-out-weight', '--ball-out-penalty', dest='ball_out_weight',
                   type=float, default=.80,
                   help='Evidencia POSITIVA de D cuando pelota x/y fuera de [0,1] (0..1)')
    p.add_argument('--ball-out-margin', type=float, default=0.01,
                   help='Tolerancia NORMALIZADA fuera y dentro de la linea (0.005 ~= 0.5 m)')
    p.add_argument('--ball-out-frames', type=int, default=3,
                   help='Frames consecutivos claramente fuera para activar evidencia D')
    p.add_argument('--ball-in-frames', type=int, default=3,
                   help='Frames consecutivos claramente dentro para quitar evidencia D')
    # Por defecto: esperar saque; solo despues permitir salida con umbral bajo.
    p.add_argument('--hold-until-restart', action=argparse.BooleanOptionalAction, default=True,
                   help='Exigir saque ANTES de permitir salida por exit-score; --no-hold-until-restart desactiva el requisito')
    p.add_argument('--restart-near-m', type=float, default=2.0,
                   help='Distancia maxima de pelota a jugador antes del saque (m)')
    p.add_argument('--restart-near-frames', type=int, default=3,
                   help='Frames de cercania previa requeridos')
    p.add_argument('--restart-far-m', type=float, default=3.0,
                   help='Distancia minima al jugador despues del saque (m)')
    p.add_argument('--restart-separation-m', type=float, default=1.25,
                   help='Aumento minimo de separacion respecto del jugador (m)')
    p.add_argument('--restart-displacement-m', type=float, default=2.25,
                   help='Desplazamiento minimo de la pelota para confirmar saque (m)')
    p.add_argument('--restart-min-speed-mps', type=float, default=3.0,
                   help='Velocidad media minima de la pelota en el saque (m/s)')
    p.add_argument('--restart-max-speed-mps', type=float, default=40.,
                   help='Rechazar saltos de tracking mas rapidos que esto (m/s)')
    p.add_argument('--restart-lookahead-seconds', type=float, default=.28,
                   help='Tiempo futuro utilizado para confirmar que se aleja')
    p.add_argument('--restart-event-gap-seconds', type=float, default=.7)
    p.add_argument('--restart-cooldown-seconds', type=float, default=1.0,
                   help='Evita reentrar inmediatamente en D tras el saque')
    p.add_argument('--restart-timeout-seconds', type=float, default=0.,
                   help='Seguridad: salida de D si no se confirma un saque; 0 desactiva')
    p.add_argument('--no-gui', action='store_true', help='Solo guardar CSVs')
    return p


def main():
    p = build_parser()
    args = p.parse_args()
    if (args.start is not None and args.start < 0 or
        args.seconds is not None and args.seconds <= 0 or
        args.context_seconds < 0 or args.imgsz < 64 or not 0 < args.conf <= 1 or
        args.novelty < 0 or args.atd_height <= 0 or args.atd_width <= 0 or
        args.palette_every < 1 or args.max_match_dist <= 0 or
        args.median_frames < 1 or args.median_frames % 2 != 1 or
        args.fill_gap < 0 or args.stability_seconds <= 0 or
        args.stable_cv <= 0 or args.stable_drift <= 0 or
        args.min_state_seconds <= 0 or args.join_gap < 0 or
        args.max_transition_seconds <= 0 or args.return_seconds <= 0 or args.max_gesture_seconds <= 0 or
        args.gesture_change <= 0 or args.gesture_full <= args.gesture_change or
        args.min_anisotropy < 0 or args.return_tolerance <= 0 or
        args.max_position_m <= 0 or args.max_position_body <= 0 or
        not 0<=args.ref_run_low<args.ref_run_high or
        not 0<=args.ref_run_penalty<=1 or
        not 0<=args.ref_osc_low<args.ref_osc_high or
        not 0<=args.ref_osc_penalty<=1 or
        not 0 <= args.exit_score < args.enter_score <= 1 or
        args.enter_frames < 1 or args.exit_frames < 1 or
        args.pose_imgsz < 64 or not 0 < args.pose_conf <= 1 or
        not 0 <= args.pose_kpt_conf <= 1 or args.pose_padding < 0 or
        args.pose_min_person_pixels <= 0 or args.pose_max_upscale < 1 or
        args.pose_zoom_height < 80 or
        not 0 < args.pose_arm_threshold <= 1 or
        args.pose_min_hold_seconds <= 0 or args.pose_bridge_seconds < 0 or
        args.pose_median_frames < 1 or args.pose_median_frames % 2 != 1 or
        not 0 <= args.pose_up_angle_start < args.pose_up_angle_full <= 180 or
        not 0 < args.pose_up_min_score <= 1 or args.pose_up_min_frames < 1 or
        not 0 <= args.pose_up_score_floor <= 1 or
        args.pose_up_plateau_gap_seconds < 0 or
        args.pose_up_plateau_min_hits < 2 or args.pose_up_plateau_min_seconds < 0 or
        not 0 <= args.pose_run_penalty <= 1 or not 0 <= args.pose_weight <= 1 or
        not 0 <= args.bbox_weight <= 1 or not 0 <= args.players_weight <= 1 or
        not 0 <= args.ball_out_weight <= 1 or
        not 0 <= args.ball_out_margin < 0.5 or
        args.ball_out_frames < 1 or args.ball_in_frames < 1 or
        args.restart_near_m <= 0 or args.restart_near_frames < 1 or
        args.restart_far_m <= args.restart_near_m or
        args.restart_separation_m <= 0 or args.restart_displacement_m <= 0 or
        args.restart_min_speed_mps <= 0 or args.restart_max_speed_mps <= args.restart_min_speed_mps or
        args.restart_lookahead_seconds <= 0 or args.restart_event_gap_seconds < 0 or
        args.restart_cooldown_seconds < 0 or args.restart_timeout_seconds < 0 or
        not 0 < args.scale <= 2 or args.speed <= 0 or args.timeline_seconds <= 0):
        p.error('Parametros invalidos')
    run(args)


if __name__ == '__main__':
    main()
