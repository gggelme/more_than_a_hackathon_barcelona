"""Comparador visual: vector_tracking original vs vector_tracking con D del arbitro.

Lee evaluation/results/vector_tracking_clasification.csv y el CSV por frame que
exporta visualizing_referee_stable_states_antirun.py (columna prediccion_D).
NO ejecuta YOLO ni entrena ni cambia las predicciones del detector.

Izquierda: clasificacion vectorial. Derecha: misma clasificacion, reemplazando
por D exclusivamente donde prediccion_D=1. Tercera barra: detector D (rojo
para D, verde para NO D, gris para fuera de cobertura).
Controles: click/arrastre timeline, espacio/p pausa, ,/. frame, a/d segundo,
[/] pagina, q/Esc salir. GT opcional y desactivado por defecto.
"""

import os
import sys
import csv
import glob
import argparse

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))   # test
ROOT = os.path.dirname(HERE)                         # repository root
sys.path.insert(0, ROOT)                             # games
sys.path.insert(0, os.path.join(ROOT, "src"))        # visualizing_tracking
from games import GAME, GAMES, half_video_blocks, require  # noqa: E402
from starter.load_homography import load as load_homography  # noqa: E402
from visualizing_tracking import (COLOURS, FONT, draw_frame as draw_tracking,  # noqa: E402
                                  draw_label_box, labels_path, read_atd, read_labels, slot_info)

# Params
FPS = GAMES[GAME]["fps"]
TIMELINE_ROW = 26                   # pixel height of one timeline row
DISAGREE = (0, 0, 255)              # BGR: red border when the two CSVs differ
LABEL_BGR = {"H": COLOURS["H"], "A": COLOURS["A"], "D": COLOURS["D"],
             "X": (255, 255, 255), "N": (67, 168, 83), "?": (65, 65, 65)}


def name_of(labels):
    """Short name shown on screen for a labels choice."""
    return "GT" if labels == "gt" else labels.replace(".csv", "")


def timeline(rows, frame, start, stop, window, width):
    """Image with one coloured row per (name, labels) over the page of `window` frames holding
    `frame`, with a cursor at `frame`.

    rows is [(name, {video_frame: label})]. Pages are consecutive blocks of `window` frames from
    `start`, so the cursor moves from left to right and jumps back to the left on a new page. Each
    pixel column is one slice of time.
    """
    first = start + ((frame - start) // window) * window
    last = min(first + window, stop) - 1
    frames = np.linspace(first, first + window - 1, width).astype(int)   # a short last page is
    frames = np.minimum(frames, last)                                    # padded with its end
    image = np.zeros((TIMELINE_ROW * len(rows) + 22, width, 3), np.uint8)
    for row, (name, labels) in enumerate(rows):
        y0 = row * TIMELINE_ROW
        for column, f in enumerate(frames):
            image[y0 + 2:y0 + TIMELINE_ROW - 2, column] = LABEL_BGR.get(labels.get(f, "X"),
                                                                        (60, 60, 60))
        cv2.putText(image, name, (6, y0 + TIMELINE_ROW - 8), FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, name, (6, y0 + TIMELINE_ROW - 8), FONT, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
    cursor = int((frame - first) * (width - 1) / max(1, window - 1))
    cv2.line(image, (cursor, 0), (cursor, TIMELINE_ROW * len(rows)), (0, 255, 255), 2)
    page = (frame - start) // window + 1
    pages = (stop - start + window - 1) // window
    text = "page {}/{}  |  video {:.1f}s .. {:.1f}s ({:.0f} s per page)".format(
        page, pages, first / FPS, last / FPS, window / FPS)
    cv2.putText(image, text, (6, image.shape[0] - 6), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return image


def compose(image, frame, rows, left_labels, right_labels, start, stop, window, panel_scale,
            tracking):
    """The whole window for one frame: two labelled panels on top, the timeline below.

    tracking is None or (matrices, valid, atd, info): when given, the ATD is drawn on the
    full-resolution frame first, as in visualizing_tracking.py.
    """
    if tracking is not None:
        matrices, valid, atd, info = tracking
        if valid[frame]:
            slots, ball = atd[frame]
            draw_tracking(image, matrices[frame], slots, ball, info, frame, FPS)
    image = cv2.resize(image, None, fx=panel_scale, fy=panel_scale)

    panels = []
    for name, labels in rows[:2]:
        panel = image.copy()
        draw_label_box(panel, name, labels.get(frame, "X"))
        panels.append(panel)
    top = np.hstack(panels)
    if (frame in left_labels and frame in right_labels and
            left_labels.get(frame) != right_labels.get(frame)):
        cv2.rectangle(top, (0, 0), (top.shape[1] - 1, top.shape[0] - 1), DISAGREE, 6)
    if tracking is None:   # the tracking strip already shows frame and time
        seconds_now = frame / FPS
        cv2.putText(top, "frame {}  {:02d}:{:05.2f}".format(
            frame, int(seconds_now // 60), seconds_now % 60),
            (top.shape[1] // 2 - 110, top.shape[0] - 15), FONT, 0.7, (255, 255, 255), 2,
            cv2.LINE_AA)
    return np.vstack([top, timeline(rows, frame, start, stop, window, top.shape[1])]), top.shape[0]




def resolve_detector_csv(path, start, stop):
    """Find a referee per-frame CSV; reject ambiguous automatic choices."""
    results_dir = os.path.join(ROOT, "evaluation", "results")
    if path and path.lower() != "auto":
        candidate = path if os.path.isabs(path) else os.path.join(results_dir, path)
        if not os.path.isfile(candidate):
            raise FileNotFoundError(f"CSV del detector no encontrado: {candidate}")
        return candidate
    matches = []
    for candidate in glob.glob(os.path.join(results_dir, "referee_estados_*.csv")):
        filename = os.path.basename(candidate)
        if filename.endswith(("_estados.csv", "_transiciones.csv")):
            continue
        bits = filename.removesuffix(".csv").split("_")
        try:
            a, b = int(bits[-2]), int(bits[-1])
        except (ValueError, IndexError):
            continue
        if a <= start and b >= stop:
            matches.append(candidate)
    if not matches:
        raise FileNotFoundError(
            "No encontre un CSV del detector que cubra todo el intervalo "
            f"[{start}, {stop}). Ejecuta primero "
            "src/visualizing_referee_stable_states_antirun.py con --no-gui, "
            "o especifica --detector-csv nombre.csv."
        )
    # Same range can be regenerated; newest is usually the one user wants.
    chosen = max(matches, key=os.path.getmtime)
    if len(matches) > 1:
        print('Aviso: hay varios CSV que cubren el rango. Elegido:', chosen)
        print('Para elegir otro, pasa --detector-csv nombre.csv')
    return chosen


def read_referee_predictions(path, start, stop):
    """Read the detector's *final* D decision, not its raw gesture score."""
    predictions, scores = {}, {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        needed = {"frame", "prediccion_D"}
        if not needed.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV {path}: faltan columnas {needed - set(reader.fieldnames or [])}")
        for row in reader:
            f = int(row["frame"])
            if not start <= f < stop:
                continue
            raw = row["prediccion_D"].strip().lower()
            if raw in ("1", "true", "d"):
                predictions[f] = True
            elif raw in ("0", "false", "no d", "n"):
                predictions[f] = False
            else:
                continue   # missing is unknown, never guess 'NO D'
            try:
                scores[f] = float(row.get("score_D", ""))
            except (ValueError, TypeError):
                pass
    return predictions, scores


def run(start, seconds=60, vector="vector_tracking_clasification",
        detector_csv="auto", show_gt=False, show_tracking=True,
        panel_scale=0.5, timeline_seconds=20, speed=1.0):
    """Left: vector prediction. Right: vector overwritten by the referee's D."""
    video_path = require(["video"])[0]
    stop = start + int(round(seconds * FPS))
    if seconds <= 0 or panel_scale <= 0 or timeline_seconds <= 0 or speed <= 0:
        raise ValueError('seconds, panel_scale, timeline_seconds, speed deben ser positivos')

    base = read_labels(labels_path(vector), start, stop)
    detector_path = resolve_detector_csv(detector_csv, start, stop)
    predicted_dead, scores = read_referee_predictions(detector_path, start, stop)
    covered = sum(f in predicted_dead for f in range(start, stop))
    positive = sum(predicted_dead.values())
    print('Prediccion vectorial:', labels_path(vector))
    print('CSV detector D:', detector_path)
    print(f'Cobertura detector: {covered}/{stop-start} frames, {positive} positivos D')
    if covered < stop-start:
        print('Aviso: frames no cubiertos NO se sobreescriben; se muestran como ? en la tercera barra.')

    # Never discard original vector labels: only overlay the detector's positive D.
    combined = {f: ('D' if predicted_dead.get(f) is True else base.get(f, 'X'))
                for f in range(start, stop)}
    detector_track = {f: ('D' if predicted_dead[f] else 'N')
                      for f in predicted_dead}
    rows = [('VECTOR', base), ('VECTOR + D', combined), ('DETECTOR D', detector_track)]
    if show_gt:
        print('Aviso: GT solo para inspeccion; no usar para calibrar hiperparametros.')
        rows.append(('GT', read_labels(labels_path('gt'), start, stop)))

    tracking = None
    if show_tracking:
        meta_path, raw_path = require(['atd_meta', 'atd_raw'])
        print('Leyendo homografias y ATD...')
        matrices, _, valid = load_homography()
        tracking = (matrices, valid, read_atd(raw_path, start, stop), slot_info(meta_path))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f'No se pudo abrir {video_path}')
    window = max(1, int(round(timeline_seconds * FPS)))
    delay_ms = max(1, int(1000 / (FPS * speed)))
    name = 'Comparar Vector Tracking vs Vector + Detector D'
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    state = {'frame': start, 'seek': None, 'toggle': False,
             'top_height': None, 'width': None}

    def on_mouse(event, x, y, flags, _):
        dragging = event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON
        if state['top_height'] is None:
            return
        if y < state['top_height']:
            if event == cv2.EVENT_LBUTTONDOWN:
                state['toggle'] = True
            return
        if event == cv2.EVENT_LBUTTONDOWN or dragging:
            page = start + ((state['frame']-start)//window)*window
            dest = page + round(x * (window-1)/max(1, state['width']-1))
            state['seek'] = min(stop-1, max(start, dest))

    cv2.setMouseCallback(name, on_mouse)
    playing = True
    shown_frame = next_in_video = None
    shown = None
    try:
        while True:
            target = state['frame'] if state['seek'] is None else state['seek']
            state['seek'] = None
            if target != shown_frame:
                if target != next_in_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ok, img = cap.read()
                if not ok:
                    print('No se pudo leer frame', target)
                    break
                next_in_video = target + 1
                shown, h = compose(img, target, rows, base, combined, start, stop,
                                   window, panel_scale, tracking)
                state['top_height'], state['width'] = h, shown.shape[1]
                # Difference explanation on second panel and detector score.
                msg = 'D: SI' if predicted_dead.get(target) is True else (
                      'D: NO' if target in predicted_dead else 'D: SIN DATOS')
                if target in scores:
                    msg += f' | SCORE: {scores[target]:.3f}'
                x0 = shown.shape[1]//2 + 12
                y0 = h-20
                cv2.putText(shown, msg, (x0+1,y0+1), FONT, .53,
                            (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(shown, msg, (x0,y0), FONT, .53,
                            (245,245,245), 1, cv2.LINE_AA)
                shown_frame = target
            state['frame'] = shown_frame
            cv2.imshow(name, shown)
            key = cv2.waitKey(delay_ms if playing else 30) & 0xFF
            if key in (ord('q'), 27):
                break
            jump = {ord(','):-1, ord('.'):1, ord('a'):-FPS, ord('d'):FPS,
                    ord('['):-window, ord(']'):window}
            if key in (ord(' '),ord('p')) or state['toggle']:
                state['toggle'] = False
                playing = not playing
            elif key in jump:
                state['seek'] = min(stop-1, max(start, shown_frame+jump[key]))
            elif playing and state['seek'] is None:
                if shown_frame >= stop-1:
                    playing = False
                else:
                    state['frame'] = shown_frame+1
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--start', type=int, default=half_video_blocks(GAME)[0][1])
    parser.add_argument('--seconds', type=float, default=5000.)
    parser.add_argument('--vector', default='vector_tracking_touch_2')
    parser.add_argument('--detector-csv', default='auto',
                        help='Nombre dentro de evaluation/results/, ruta absoluta, o auto')
    parser.add_argument('--show-gt', action='store_true', default=True)
    parser.add_argument('--no-tracking', action='store_true')
    parser.add_argument('--panel-scale', type=float, default=.5)
    parser.add_argument('--timeline-seconds', type=float, default=20.)
    parser.add_argument('--speed', type=float, default=1.)
    args = parser.parse_args()
    run(start=args.start, seconds=args.seconds, vector=args.vector,
        detector_csv=args.detector_csv, show_gt=args.show_gt,
        show_tracking=not args.no_tracking, panel_scale=args.panel_scale,
        timeline_seconds=args.timeline_seconds, speed=args.speed)


if __name__ == '__main__':
    main()
