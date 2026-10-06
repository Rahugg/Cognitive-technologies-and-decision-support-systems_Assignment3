# Assignment 3 deliverables

## Project files

- [x] Python source: Streamlit app, four cognitive modules, CNN, data/training/demo scripts.
- [x] Dockerfile and Docker Compose profiles for PostgreSQL, app, data preparation, training and notebook.
- [x] Assignment notebook with reproducible pipeline and three demo cases.
- [x] Updated technical report PDF: `Assignment3_Technical_Report_FINAL.pdf`.
- [x] Updated six-slide presentation: `output/Assignment3_Presentation_FINAL.pptx` (within the required 5-7 slides).

## Required run

```bash
cp .env.example .env
docker compose --profile data run --rm dataset-downloader
docker compose --profile train run --rm trainer
docker compose up --build
docker compose exec app python scripts/run_demo.py
```

Open `http://localhost:8501`. For Jupyter use `docker compose --profile notebook up jupyter db` and open `http://localhost:8888`.

## Submission checks

- The older `Assignment3_Technical_Report.pdf` and `Assignment3_Presentation.pptx` are source drafts; submit the files marked `FINAL`.
- Run data download and training; retain generated metrics JSON with the trained model.
- Run all three demonstrations and capture their actual outputs for the report/presentation.
- Replace the team-contribution placeholder with the students' factual contribution split.
- Confirm each dataset's current access and license terms. Do not include raw datasets or Kaggle credentials in the submission archive.
- Do not state that the low-quality example was rejected unless its printed Attention result confirms rejection.
