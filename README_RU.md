# Assignment 3: Emotion Recognition System

Учебный прототип продолжает тему Assignment 2 и оценивает видимое выражение лица в живом видеопотоке с камеры. Анализ выполняется примерно раз в секунду, а краткая память сглаживает до пяти принятых кадров. Также можно загрузить отдельное изображение. Система не определяет внутреннее эмоциональное состояние человека.

## Состав проекта

- `app.py` — Streamlit-интерфейс с live webcam через WebRTC и загрузкой изображения.
- `src/emotion_system/` — Perception, Attention, Memory, Knowledge Representation и CNN.
- `scripts/download_datasets.py` и `scripts/prepare_dataset.py` — загрузка и подготовка данных.
- `scripts/train.py` — обучение и оценка модели; реальные метрики записываются в `models/emotion_cnn.metrics.json`.
- `scripts/run_demo.py` — три сценария демонстрации.
- `docker-compose.yml` — приложение, PostgreSQL, профили `data`, `train`, `notebook`. JupyterLab устанавливается только в notebook-образ.

## Требования

Нужен Docker Desktop или Docker Engine с Compose v2. Загрузка KaggleHub требует доступа к сети; некоторые датасеты могут потребовать Kaggle credentials или принятия условий. У каждого набора данных свои лицензия и правила использования.

## Запуск

Для запуска одним кликом дважды нажмите **`Run Emotion Recognition.command`** в Finder. На первом запуске файл автоматически загрузит данные и обучит модель, если подготовленный индекс или веса отсутствуют. После запуска откроется `http://localhost:8501`.

Из терминала используйте `make run` из каталога проекта. Нужен запущенный Docker Desktop.

Из каталога `ass3`:

```bash
cp .env.example .env
docker compose --profile data run --rm dataset-downloader
docker compose --profile train run --rm trainer
docker compose up --build
```

Откройте `http://localhost:8501`, нажмите **START** во вкладке **Live camera** и разрешите доступ к камере. Для трёх воспроизводимых примеров:

```bash
docker compose exec app python scripts/run_demo.py
```

При необходимости авторизации Kaggle укажите доступный Kaggle credential в `.env`, затем повторите загрузку. Не добавляйте токен в репозиторий.

Дополнительный JupyterLab:

```bash
docker compose --profile notebook up jupyter db
```

Откройте `http://localhost:8888` и запустите `Assignment3_Emotion_Recognition_FINAL.ipynb`.

По умолчанию подготовка использует все доступные изображения, а обучение запускает до 8 эпох с early stopping. Полный прогон занимает заметное время на CPU; можно задать `MAX_PER_CLASS_SOURCE` и `EPOCHS` в `.env` для более короткого запуска. Реальные результаты этой конфигурации: 53 080 изображений, test accuracy 67,74%, macro-F1 62,19% после дообучения checkpoint на 8 эпохах CPU (улучшение относительно предыдущей модели: 66,18% и 60,26%). Смотрите per-class и per-dataset метрики в `models/emotion_cnn.metrics.json`.

Остановка и удаление контейнеров:

```bash
docker compose down
```

Том PostgreSQL сохраняется при `down`; для его удаления используйте `docker compose down -v`.

## Архитектура

`Input → Perception → Attention → Memory → Knowledge Representation → Output`

- **Input / Perception:** WebRTC передаёт кадры камеры локальному приложению. OpenCV обнаруживает лицо, а для уже обрезанных изображений использует центральный crop; CNN выдаёт вероятности семи классов и оценку качества. Видеокадры не сохраняются.
- **Attention:** принимает вход при `quality ≥ 0.45` и `confidence ≥ 0.25`. Правило знаний помечает прогнозы с низкой уверенностью как `uncertain`. Релевантность: `0.45 × quality + 0.55 × confidence`.
- **Memory:** deque хранит до пяти результатов текущей сессии; PostgreSQL сохраняет метаданные прогнозов. Сырые изображения в базу не пишутся.
- **Knowledge Representation:** NetworkX-граф и четыре правила формируют действие либо `insufficient_data`/`uncertain`.
- **Output:** интерфейс показывает класс, уверенность, релевантность, историю памяти и правила.

Краткая память хранит вероятности последних пяти принятых кадров в RAM и усредняет их для более стабильного live-результата; PostgreSQL хранит метаданные каждого обработанного кадра. Долговременная память влияет на повторный запрос через правило R2 при совпадении с предыдущим принятым классом.

Assignment 2 также рассматривает микрофон и объединение лица с голосом. Эта версия реализует live-распознавание по камере; анализ речи и слияние аудио/видео не реализованы.

## Данные

Downloader запрашивает FER-2013 (`msambare/fer2013`), CK+ (`shawon10/ckplus`) и RAF-DB (`shuvoalok/raf-db-dataset`). Подготовка распознаёт изображения в папках классов и FER CSV. Другие раскладки источников могут потребовать отдельной настройки парсера. CK+ `contempt` пропускается, потому что в модели семь общих классов. Split детерминированный, seed по умолчанию `42`.

См. [`DATASETS.md`](DATASETS.md) и [`DELIVERABLES.md`](DELIVERABLES.md). Данные, модельные веса, метрики и credentials не включены в репозиторий.

Подготовленные материалы для сдачи: [`Assignment3_Technical_Report_FINAL.pdf`](Assignment3_Technical_Report_FINAL.pdf) и [`output/Assignment3_Presentation_FINAL.pptx`](output/Assignment3_Presentation_FINAL.pptx). В отчёте явно отмечены незапущенные проверки и отсутствующие фактические вклады участников; заполните эти пункты перед сдачей.

## Ограничения и приватность

Качество и точность зависят от данных, разметки, освещения, позы, окклюзий и согласованности доменов. После fine-tuning test recall для fear вырос с 28,0% до 32,1%, но остаётся слабым. На отдельных примерах fear путается с surprise. Смех не является отдельным классом в датасете: он маппится в happy, но тестовый laughing-face был классифицирован как angry. Для surprise общий test recall составил 84,2%, хотя отдельные фото могут ошибаться. Вероятности модели не доказывают внутреннее состояние человека. Не применяйте прототип для найма, оценки личности, наблюдения или иных значимых решений. Не храните изображения лица без отдельного согласия.
