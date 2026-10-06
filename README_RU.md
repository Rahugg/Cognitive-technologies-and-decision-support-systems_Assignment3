# Assignment 3: Emotion Recognition System

Учебный прототип продолжает тему Assignment 2 и оценивает выражение лица по одному изображению. Он не определяет внутреннее эмоциональное состояние человека. Видеопоток и анализ аудио в эту версию не входят.

## Состав проекта

- `app.py` — Streamlit-интерфейс.
- `src/emotion_system/` — Perception, Attention, Memory, Knowledge Representation и CNN.
- `scripts/download_datasets.py` и `scripts/prepare_dataset.py` — загрузка и подготовка данных.
- `scripts/train.py` — обучение и оценка модели; реальные метрики записываются в `models/emotion_cnn.metrics.json`.
- `scripts/run_demo.py` — три сценария демонстрации.
- `docker-compose.yml` — приложение, PostgreSQL, профили `data`, `train`, `notebook`.

## Требования

Нужен Docker Desktop или Docker Engine с Compose v2. Загрузка KaggleHub требует доступа к сети; некоторые датасеты могут потребовать Kaggle credentials или принятия условий. У каждого набора данных свои лицензия и правила использования.

## Запуск

Из каталога `ass3`:

```bash
cp .env.example .env
docker compose --profile data run --rm dataset-downloader
docker compose --profile train run --rm trainer
docker compose up --build
```

Откройте `http://localhost:8501`. Для трёх воспроизводимых примеров:

```bash
docker compose exec app python scripts/run_demo.py
```

При необходимости авторизации Kaggle укажите доступный Kaggle credential в `.env`, затем повторите загрузку. Не добавляйте токен в репозиторий.

Дополнительный JupyterLab:

```bash
docker compose --profile notebook up jupyter db
```

Откройте `http://localhost:8888` и запустите `Assignment3_Emotion_Recognition_FINAL.ipynb`.

Чтобы уменьшить время учебного запуска, задайте `MAX_PER_CLASS_SOURCE` и `EPOCHS` в `.env`. Ограничение по классу и источнику по умолчанию равно 1500, число эпох — 5. Эти значения не являются заявлением о достигнутом качестве. Смотрите фактические метрики в `models/emotion_cnn.metrics.json`.

Остановка и удаление контейнеров:

```bash
docker compose down
```

Том PostgreSQL сохраняется при `down`; для его удаления используйте `docker compose down -v`.

## Архитектура

`Input → Perception → Attention → Memory → Knowledge Representation → Output`

- **Perception:** OpenCV обнаруживает лицо, а для уже обрезанных изображений использует центральный crop; CNN выдаёт вероятности семи классов и оценку качества.
- **Attention:** принимает вход при `quality ≥ 0.45` и `confidence ≥ 0.55`. Релевантность: `0.45 × quality + 0.55 × confidence`.
- **Memory:** deque хранит до пяти результатов текущей сессии; PostgreSQL сохраняет метаданные прогнозов. Сырые изображения в базу не пишутся.
- **Knowledge Representation:** NetworkX-граф и четыре правила формируют действие либо `insufficient_data`/`uncertain`.
- **Output:** интерфейс показывает класс, уверенность, релевантность, историю памяти и правила.

Краткая память в этой реализации хранит недавние результаты, но не усредняет вероятности CNN. Долговременная память влияет на повторный запрос через правило R2 при совпадении с предыдущим принятым классом.

## Данные

Downloader запрашивает FER-2013 (`msambare/fer2013`), CK+ (`shawon10/ckplus`) и RAF-DB (`shuvoalok/raf-db-dataset`). Подготовка распознаёт изображения в папках классов и FER CSV. Другие раскладки источников могут потребовать отдельной настройки парсера. CK+ `contempt` пропускается, потому что в модели семь общих классов. Split детерминированный, seed по умолчанию `42`.

См. [`DATASETS.md`](DATASETS.md) и [`DELIVERABLES.md`](DELIVERABLES.md). Данные, модельные веса, метрики и credentials не включены в репозиторий.

Подготовленные материалы для сдачи: [`Assignment3_Technical_Report_FINAL.pdf`](Assignment3_Technical_Report_FINAL.pdf) и [`output/Assignment3_Presentation_FINAL.pptx`](output/Assignment3_Presentation_FINAL.pptx). В отчёте явно отмечены незапущенные проверки и отсутствующие фактические вклады участников; заполните эти пункты перед сдачей.

## Ограничения и приватность

Качество и точность зависят от данных, разметки, освещения, позы, окклюзий и согласованности доменов. Вероятности модели не доказывают внутреннее состояние человека. Не применяйте прототип для найма, оценки личности, наблюдения или иных значимых решений. Не храните изображения лица без отдельного согласия.
