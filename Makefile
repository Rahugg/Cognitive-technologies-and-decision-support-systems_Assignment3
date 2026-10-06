COMPOSE := docker compose
APP_URL := http://localhost:8501

.PHONY: run stop logs demo retrain

# One command for first setup and later launches.
run:
	@docker info >/dev/null 2>&1 || { echo "Docker is not running. Start Docker Desktop and try again."; exit 1; }
	@if [ ! -s data/processed/index.csv ]; then \
		echo "Dataset index missing; downloading and preparing FER-2013, CK+, and RAF-DB..."; \
		$(COMPOSE) --profile data run --rm dataset-downloader; \
	fi
	@if [ ! -s models/emotion_cnn.pt ]; then \
		echo "Trained model missing; training the CNN..."; \
		$(COMPOSE) --profile train run --rm trainer; \
	fi
	$(COMPOSE) up --build -d app
	@attempt=0; until curl --silent --fail --output /dev/null $(APP_URL); do \
		attempt=$$((attempt + 1)); \
		if [ $$attempt -ge 60 ]; then echo "App did not become ready. Run 'make logs' for details."; exit 1; fi; \
		sleep 2; \
	done
	@echo "Emotion Recognition System is ready at $(APP_URL)"
	@open $(APP_URL) 2>/dev/null || echo "Open $(APP_URL) in your browser."

stop:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs --follow app db

demo:
	$(COMPOSE) exec app python scripts/run_demo.py

# Force a fresh training run with the current prepared dataset.
retrain:
	$(COMPOSE) --profile train run --rm trainer
