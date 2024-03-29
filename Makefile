# Path: Makefile
training:
	docker compose  -f "docker-compose.yml" up -d --build training

streamlit:
	docker compose  -f "docker-compose.yml" up -d --build streamlit

stop:
	docker-compose down

logs:
	docker-compose logs -f

setup:
	conda create -n churn-prediction python=3.9
	pip install -r src/requirements.txt
	pip install -r app/requirements.txt
	pip install pytest pylint black isort

test:
	pytest training/tests/

quality_checks:
	isort .
	black .
	pylint . --recursive=y --fail-under=9