.PHONY: help install test lint format clean run-backend run-frontend run

help:
	@echo "Options Visualizer - Development Commands"
	@echo ""
	@echo "Usage:"
	@echo "  make install        Install dependencies"
	@echo "  make test           Run tests"
	@echo "  make lint           Run linting checks"
	@echo "  make format         Format code with Black and isort"
	@echo "  make clean          Clean up build artifacts"
	@echo "  make run-backend    Run the backend server"
	@echo "  make run-frontend   Run the frontend server"
	@echo "  make run            Run both servers"
	@echo ""

install:
	pip install -r requirements.txt
	pip install -e .

test:
	python -m pytest tests/

lint:
	flake8 backend
	mypy backend

format:
	black backend tests
	isort backend tests

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

run-backend:
	python -m backend.app

run-frontend:
	python -m options_visualizer_web.app

run:
	python main.py 