# More Than A Hack 2027 — possession

Per-frame possession labels (`H` / `A` / `D` / `X`) for the 199,319 frames of the tactical video.

## Run

```bash
uv sync
uv run python run.py
```

That is all. `run.py` is the whole pipeline, from the provided data to the submission file:

- the match data must be unpacked in `data/` as described in `info_hackathon/DATA.md` (it is not in
  this repository);
- the YOLO weights (Ultralytics' official `yolo11s.pt` and `yolo11s-pose.pt`) are downloaded into
  `models/` on the first run;
- the referee stage runs YOLO on the whole match (hours on a GPU); it is cached in chunks, so an
  interrupted run resumes where it stopped;
- the result is `evaluation/results/<OUTPUT_NAME>.csv`, already checked with
  `evaluation/validate_submission.py`.

Every option (output name, referee on/off, chunk size, device, thresholds) is in the
`if __name__ == "__main__":` block at the bottom of `run.py`.

## The possession files — `src/tracking_possession/`

Pure logic on the ATD positions in metres: no video, no files. `run.py` uses them in this order, one
half at a time.

| file | what it does |
|---|---|
| `fix_goalkeepers.py` | puts each goalkeeper back in his own team's slot (the ATD slots swap), and finds which goal each team defends from the kick-off formation |
| `possession.py` | the base classifier: who owns the ball each frame, from distance, relative velocity and direction; the team in possession is the team of the last owner |
| `keeper_catch.py` | keeper catches the tracking cannot see: the ball vanishes at the keeper and comes back from him |
| `possession_base.py` | **the method used**: `possession.py` plus the keeper catches and the restarts, resolved as "the last anchor wins" |
| `goals.py` | writes `D` over every stoppage (goals, cooling breaks, the referee's stoppages, half edges) and re-resolves the possession around them |
| `fix_cooling_break.py` | cooling breaks: many players leave the pitch towards one place |
| `referee.py` | the referee's dead balls (fouls, offsides, ball out, VAR), ended at the restart once the ball is back in play |
| `restarts.py` | who restarts after a stoppage (kick-off rule, or the player nearest the ball) — that team has the ball from then on |
| `possession_touch_4.py` | the alternative we compared and dropped: possession from collisions + control |

The referee signal itself comes from the video:

| file | what it does |
|---|---|
| `src/referee_signal.py` | runs the referee detection on a range of frames and caches it (`data/derived/referee/`) |
| `src/visualizing_deadball.py` | the detection: YOLO finds the referee, YOLO Pose reads his arms, and its dead-ball score decides the stoppage |

`games.py` holds the per-game constants (frame alignment, halves, pitch size).

## Development tools — `src/visualizing_*.py`

Not part of the pipeline. These are the viewers we used to build and check the possession files
above: each draws the tracking and a module's decisions on the tactical video, frame by frame, so
every rule could be checked by eye.

| file | shows |
|---|---|
| `visualizing_tracking.py` | the ATD projected on the video |
| `visualizing_vectors.py`, `visualizing_field.py` | the movement vectors the classifier sees |
| `visualizing_owner.py` | the base classifier's owner per frame |
| `visualizing_touches.py` | the full possession (base vs. touch pass) with stoppages and referee |
| `visualizing_goals.py` | the dead-ball stretches of `goals.py` |
| `visualizing_smart_tagging.py` | Metrica's Smart Tagging |
| `visualizing_deadball.py` | the referee detection (also used by the pipeline) |

`src/experiments/` holds earlier submission scripts, kept for reference.
