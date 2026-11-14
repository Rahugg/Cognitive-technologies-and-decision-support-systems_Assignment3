# Role A handoff: Reasoning, Decision, Action

Branch: `feat/reasoning-decision-action` (branch from it, not from `main`). Pipeline: Perception → Attention → Memory → Reasoning → Decision → Action → Learning, run by `CognitiveEmotionSystem.run_cycle()` in `src/emotion_system/pipeline.py`. Every module exposes `run(state, memory) -> state` and appends one line to `state.explanation`.

## 1. What was built

| Commit | Content |
|---|---|
| `feat(config)` | DSS settings in `config.py` (matrix M, criteria, default weights, disruption/time costs, trend and R2/R3 constants); confidence threshold default 0.25 → 0.55 (config and `docker-compose.yml`). |
| DB fallback (same commit) | `Memory` uses in-memory storage when PostgreSQL is unreachable; new `decisions` table and `save_decision()`; `Memory.utility` holds M. |
| `feat(core)` | `CogState` (`state.py`), 7-stage `run_cycle()`, `process()` kept with the old dict shape plus `"state"`, stubs for `dss/model.py` and `learning.py`. |
| `feat(reasoning)` | Recency-weighted smoothing over accepted frames only, R2 (+0.05) on smoothed probs, R1/R3/R4 labels, trend, `reasoning_steps`. |
| `feat(decision)` | 5×4 criteria matrix, `rank()` with default weights, `ranking`, `recommended`. |
| `feat(action)` | Tutor recommendation text, decision record to PostgreSQL (no images), session report. |
| `test(...)` | 15 pytest tests (synthetic `CogState`, no database needed). |
| `feat(demo)`, `fix(core)`, `fix(deps)` | `run_demo.py` prints new fields; R1/R3 keep action `request_more_data` in `process()`; `pyarrow==18.1.0` pinned. |

## 2. Contract

### `CogState` (`src/emotion_system/state.py`)

| Field | Type | Set by |
|---|---|---|
| session_id | str | caller |
| image | Any (transient, cleared after the cycle) | caller |
| emotion, confidence, quality, brightness, sharpness | str / float | Perception |
| probs | dict[str, float] | Perception |
| face_found | bool | Perception |
| accepted, relevance | bool, float | Attention (face rule applied in pipeline) |
| reject_reason | str ("" if accepted) | Attention |
| previous_emotion | str or None | Memory |
| smoothed_probs, smoothed_confidence | dict, float | Reasoning |
| trend | `stable`, `changing` or `insufficient_history` | Reasoning |
| reasoning_steps, rules | list[str] | Reasoning |
| label | `insufficient_data` (R1), `uncertain` (R3) or emotion (R4) | Reasoning |
| alternatives | list[str] | Decision |
| criteria_matrix | dict: criteria, types, weights, reliability, values {alt: [C1..C4]} | Decision |
| ranking | list[dict]: alternative, rank, score, contributions {criterion: value}; best first | Decision |
| recommended | str | Decision |
| action | str (recommendation text) | Action |
| report | dict or None (record fields, text, rules, steps, explanation, db_written, limitation) | Action |
| explanation | list[str] | every module |

### `rank()` (`dss/model.py`, currently a stub)

| Item | Contract |
|---|---|
| Signature | `rank(matrix, weights, types, method="wsm") -> dict` |
| `matrix` | alternatives × criteria raw values (list of rows) |
| `weights` | one weight per criterion (normalised inside) |
| `types` | `"benefit"` or `"cost"` per criterion |
| Returns | `scores` (per alternative), `ranks` (1 = best, per alternative), `order` (indices best → worst), `contributions` (weight × normalised value, alternatives × criteria), `method` |

## 3. How to run

| What | Command | Status |
|---|---|---|
| Tests | `pip install -r requirements.txt pytest`, then `python -m pytest -q` | 15 passed in a Python 3.11 venv and inside the Docker app container |
| Demo, no PostgreSQL | `PYTHONPATH=. python scripts/run_demo.py` | Verified; prints `db_written = False` |
| Demo, with PostgreSQL | `docker compose up -d db`, then the same command | Verified on host and in container; rows written to `decisions` and `predictions` |
| Streamlit | `streamlit run app.py` | Verified on host (Python 3.11): page loads, Upload tab with a demo photo analysed without errors |
| Docker | `docker compose up -d app`, then http://localhost:8501 | Verified: image rebuilt with pyarrow 18.1.0, HTTP 200, page renders without errors, tests and demo pass in the container |

**Not verified:** the live camera tab (needs a webcam and browser permission); the new fields in the UI (`app.py` does not show them yet); TOPSIS and Learning behaviour (not implemented); Python 3.13/3.14 (only 3.11 tested); the `Makefile` / `.command` one-click route.

## 4. Role B: Learning and evaluation

- Replace the stub `src/emotion_system/learning.py` (keep `run(state, memory) -> state` and the `explanation` line).
- Read and write matrix M through `memory.utility` (`dict[emotion] -> [continue, re_explain, pause, easier_task, ask_directly]`). Decision reads it on every call, so updates apply immediately. Initial values live in `config.UTILITY`.
- Feedback table: new table in PostgreSQL (create it in `Memory._initialize` and keep the in-memory fallback working). Link rows to `decisions`:

| `decisions` column | Type |
|---|---|
| id | BIGSERIAL |
| session_id | TEXT |
| recommended | TEXT |
| emotion | TEXT (top smoothed emotion) |
| reliability | DOUBLE PRECISION |
| trend | TEXT |
| accepted | BOOLEAN |
| ranking | JSONB (`[{alternative, score}]`) |
| created_at | TIMESTAMPTZ |

- Show before/after: run the same frame sequence before feedback and after M changes, and print the ranking and `recommended` for both.
- Evaluation: accuracy, macro-F1 and confusion matrix for the CNN. Do not retrain or change weights or architecture.

## 5. Role C: DSS and UI

- Replace `dss/model.py` with a real WSM (optional TOPSIS via `method="topsis"`). Keep the signature and return keys above, including per-criterion `contributions`.
- In `app.py`, show all 7 stages and the new state fields: `reasoning_steps`, `ranking`, `recommended`, `action`, `report`. `process()` returns the state under `result["state"]`; `CognitiveEmotionSystem.run_cycle()` returns it directly.
- Add weight sliders and what-if: call `rank(list(state.criteria_matrix["values"].values()), weights, state.criteria_matrix["types"])` with the user's weights, no restart.
- Update README and slides. `pyarrow==18.1.0` is already pinned in `requirements.txt`.

## 6. Known issues and open decisions

| Topic | Detail |
|---|---|
| Confidence threshold | Changed 0.25 → 0.55 (`config.py`, `docker-compose.yml`). Fewer frames are accepted in the live demo. |
| Default weights and M | `continue` ranks first for most emotions, including angry (0.65 confidence) and sad, because it has zero disruption and zero time cost. Fear ties with `easier_task`. Weights or M need a team decision. |
| In-memory fallback | If PostgreSQL is unreachable, `Memory` logs a warning and keeps data in memory only (`db_ok = False`, `db_written = False`). Nothing is persisted in that mode. |
| Reliability for rejected frames | r = min(smoothed, current confidence) × current quality; a `changing` trend multiplies r by 0.8. With history, a rejected frame lifts `ask_directly` (4th → 2nd in the demo) but `continue` can still win. Without history, probabilities are uniform and `ask_directly` wins. |
| Stale docs | `README_RU.md` and `examples/open_source/README.md` still say confidence ≥ 0.25. |
| Local test server | A Streamlit server on port 8599 started during testing may still be running. |
| `Memory.smooth_prediction` | Still in `memory.py` but no longer used by the pipeline. |

## 7. Git

- `git fetch && git checkout -b feat/<your-part> feat/reasoning-decision-action`. Do not branch from or push to `main`.
- Small commits, Conventional Commits, tests passing before each commit.

| Role | Files |
|---|---|
| A | `reasoning.py`, `decision.py`, `action.py`, `state.py`, their tests, `docs/ROLE_A_HANDOFF.md`, `docs/PR_DESCRIPTION.md` |
| B | `learning.py`, feedback table, evaluation scripts, `models/*.metrics.json` |
| C | `dss/model.py`, `app.py`, `requirements.txt`, README, slides |
| Shared (coordinate) | `config.py`, `memory.py`, `pipeline.py`, `docker-compose.yml`, `scripts/run_demo.py` |
