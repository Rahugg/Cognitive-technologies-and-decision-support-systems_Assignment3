# Tutor Decision Support

A Streamlit app for an online tutor deciding whether to continue, explain differently, pause, offer an easier task, or ask the student directly. A facial-expression classifier supplies one signal. The tutor makes and records the decision.

## Run locally

Use Python 3.11. From the repository or unzipped submission directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

On Windows, create the environment with `py -3.11 -m venv .venv` and activate it with `.venv\Scripts\Activate.ps1` in PowerShell. Open the local URL printed by Streamlit. The bundled checkpoint is loaded from `models/emotion_cnn.pt`. Installation needs network access for the Python packages. Running the demo needs no training, dataset download, camera, or database.

Run `DATABASE_URL= streamlit run app.py` to explicitly use in-memory storage. Session data and feedback are lost when the server stops in this mode. For persistence, set `DATABASE_URL` to a PostgreSQL database that the app can write to. For example, `docker compose up -d db` starts the optional local database on port 5433, which matches the default connection setting. `docker compose up --build app` is the full Docker alternative.

## Use the app

1. In **Analyze a frame**, select a bundled image or upload a JPG/PNG, then select **Analyze frame**. The **Live camera** tab can capture a result with **Inspect latest frame**.
2. Inspect the **DSS** recommendation and its score margin against the runner-up. Change any weight to rerank instantly. The sensitivity chart sweeps one criterion while holding the others in proportion. The score curves show where recommendations change.
3. Open the seven stage tabs to see CNN probabilities, the attention gate, stored context, inference steps, the default ranking, the actionable report, and learning from feedback.
4. In **Learning**, choose the action you took and confirm the expression if it looked accurate. The app compares fit, score and rank before and after the update for the same frame. New analyses use the revised utility matrix. Rejected frames can record a decision without updating that matrix.

Use the [timed demo script](docs/DEMO.md) for the two-minute defense. [Demo image provenance](examples/open_source/README.md) covers the bundled fixtures.

## How the ranking works

Each frame runs through Perception, Attention, Memory, Reasoning, Decision, Action and Learning. Reasoning weights recent accepted frames more heavily and uses uniform expression probabilities when no accepted frame is available. The Decision stage and the Streamlit what-if controls call the same `dss.model.rank` function.

| Criterion | Type | Default | Raw value |
| --- | --- | ---: | --- |
| Expression fit | Benefit | 35% | Expected utility over the smoothed expression probabilities. The starting matrix is an expert assumption. |
| Signal reliability | Benefit | 25% | Smoothed confidence × frame quality, reduced for changing trends and capped by current confidence on rejection. Asking directly receives `1 − reliability`. |
| Lesson disruption | Cost | 25% | Expert-assigned disruption, from 0 to 0.7. |
| Time cost | Cost | 15% | Estimated minutes, from 0 to 5. |

WSM puts the criteria on a common 0–1 scale and reverses the cost scales so lower costs contribute more. It normalizes weights to sum to 1 and adds the weighted values. This makes both the score and the winner's margin traceable to specific criteria. A constant criterion contributes equally to all alternatives. Equal scores keep the input order. What-if weights only explore the selected frame. The Action report records the default analysis, and tutor feedback records the recommendation the tutor saw alongside their actual choice.

## Evaluation and limits

The [team’s held-out evaluation](models/emotion_cnn.evaluation.metrics.json) reports 67.0% accuracy and 61.3% macro-F1 across 5,325 images. Fear recall is 31.4%. Those figures describe CNN classification on the test split. Pedagogical effectiveness was not measured. The [learning demo](output/learning_demo.json) uses simulated tutor feedback.

Expressions are ambiguous and class performance varies. Ask the student when evidence is weak. Use consented images. Decision and feedback records contain metadata only. Frame pixels are processed in memory.

```bash
python -m pip install pytest
python -m pytest -q
```
