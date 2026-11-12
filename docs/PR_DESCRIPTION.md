# feat: Reasoning, Decision and Action (Role A) with the shared CogState pipeline

## Summary

Adds the three Role A cognitive functions and the interface the other roles build on.

- **Reasoning:** recency-weighted smoothing over Attention-accepted frames, R2 on the smoothed probability, R1/R3/R4 labels, trend (`stable`, `changing`, `insufficient_history`) and visible `reasoning_steps`.
- **Decision:** 5 alternatives × 4 criteria (fit, reliability, disruption, time cost), ranked with `dss.model.rank()` using the default weights (0.35, 0.25, 0.25, 0.15).
- **Action:** tutor recommendation text with the top criterion contributions, a decision record in PostgreSQL (no images) and a session report.
- **Core:** `CogState`, a 7-stage `run_cycle()`, stubs for `dss/model.py` and `learning.py` (owned by Roles C and B), 15 unit tests.

The CNN, its weights and its architecture are unchanged. The tool classifies facial expressions, not internal states, and the tutor makes the final decision.

## Changes to shared files

| File | Change | Reason |
|---|---|---|
| `src/emotion_system/config.py` | DSS settings (matrix M, criteria, weights, costs, trend and R2/R3 constants); `CONFIDENCE_THRESHOLD` default 0.25 → 0.55 | Spec says thresholds, weights and M come from config; 0.55 is the specified Attention threshold |
| `src/emotion_system/memory.py` | In-memory fallback when PostgreSQL is unreachable; `decisions` table and `save_decision()`; `utility` attribute (matrix M) | Clean-machine `streamlit run` must start without a database; Action needs a DB record; Learning needs a place to update M |
| `src/emotion_system/pipeline.py` | `run_cycle()` returns `CogState`; `process()` keeps its old dict keys, adds `"state"`; R1/R3 keep action `request_more_data` | One pipeline for all 7 functions without breaking `app.py` |
| `docker-compose.yml` | `CONFIDENCE_THRESHOLD` 0.25 → 0.55 | Keep Compose consistent with config |
| `requirements.txt` | Add `pyarrow==18.1.0` | Unpinned pyarrow needs NumPy 2 and breaks Streamlit with `numpy==1.26.4` (host and Docker) |
| `scripts/run_demo.py` | Prints reasoning, DSS ranking, action, DB status; uses `examples/open_source/smiling-face.jpg` when `data/processed/index.csv` is missing | Demo runs on a clean machine and shows the new fields |

New files: `src/emotion_system/{state,reasoning,decision,action,learning}.py`, `dss/model.py` (stub), `tests/`, `docs/`.

## How to verify

```bash
git checkout feat/reasoning-decision-action
pip install -r requirements.txt pytest
python -m pytest -q                      # expect 15 passed
PYTHONPATH=. python scripts/run_demo.py  # three scenarios; db_written = False without PostgreSQL

docker compose up -d db                  # then re-run the demo: db_written = True
docker compose exec -T db psql -U emotion -d emotiondb -c "select * from decisions order by id desc limit 3"

streamlit run app.py                     # Upload tab -> demo image -> Analyze photo
docker compose build app && docker compose up -d app   # http://localhost:8501 returns 200
docker compose exec -T app python -c "import pyarrow"
```

Verified so far: tests (Python 3.11 venv and Docker container), demo with and without PostgreSQL (host and container), Upload tab on host and Docker page load. Not verified: live camera tab, the new fields in the UI (`app.py` unchanged), TOPSIS and Learning (not implemented), Python versions other than 3.11.

Demo outcome (default weights): clear and mild frames recommend `continue`. The heavily degraded frame is rejected (R1); `ask_directly` rises from 4th to 2nd but `continue` still wins because of earlier happy frames. With no history, the same frame ranks `ask_directly` first.

## Risks

- **Lower acceptance rate:** threshold 0.55 rejects more live frames than 0.25.
- **`process()` consumers:** `app.py` depends on the dict shape; it is preserved, but any new UI work should read `result["state"]`.
- **Fallback hides DB problems:** with PostgreSQL down, the app runs with in-memory data and a log warning; nothing is persisted.
- **Default ranking:** `continue` wins for most emotions (including angry and sad at moderate confidence), which may look wrong in a live demo.
- **Shared-file edits:** `memory.py` and `pipeline.py` are also touched by Role B and C work; expect merge conflicts there.
- **Unused code:** `Memory.smooth_prediction` is left in place but no longer called.

## Decisions needed from the team

- [ ] Keep the confidence threshold at 0.55 (config and Compose), or go back to 0.25?
- [ ] Change the default weights or matrix M so that angry, sad and fear do not default to `continue`?
- [ ] Keep the in-memory fallback when PostgreSQL is unreachable?
- [ ] Keep the reliability rule for rejected frames (min of smoothed and current confidence × current quality), or force `ask_directly` higher when a frame is rejected?
- [ ] Update `README_RU.md` and `examples/open_source/README.md`, which still say confidence ≥ 0.25 (Role C)?
- [ ] Remove the unused `Memory.smooth_prediction`?
