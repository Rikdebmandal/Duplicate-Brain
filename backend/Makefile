.PHONY: help install dev test lint migrate seed evaluate clean

help:
	@echo "install   install dependencies into the active environment"
	@echo "dev       run the API with autoreload"
	@echo "test      run the test suite with coverage"
	@echo "lint      run ruff"
	@echo "migrate   apply database migrations"
	@echo "seed      load the demo decision history"
	@echo "evaluate  compare prediction approaches on a time-based split"

install:
	pip install -r requirements-dev.txt

dev:
	uvicorn app.main:app --reload --port 8000

test:
	pytest --cov=app --cov-report=term-missing

lint:
	ruff check app tests seeds scripts

migrate:
	alembic upgrade head

seed:
	python -m seeds.seed_data --reset

evaluate:
	python -m scripts.evaluate --email demo@example.com

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
