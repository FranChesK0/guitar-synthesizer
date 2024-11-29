.PHONY: help clean format lint test check pre-commit-install pre-commit

.DEFAULT_GOAL := help

help:
	@echo "Available targets:"
	@echo "  help               - Display this message"
	@echo "  clean              - Remove all build artifacts and cache files"
	@echo "  format             - Format code with isort and black"
	@echo "  lint               - Lint code with flake8 and mypy"
	@echo "  test               - Run tests with pytest"
	@echo "  check              - Run format, lint and tests"
	@echo "  pre-commit-install - Install pre-commit hooks"
	@echo "  pre-commit         - Run pre-commit hooks"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".tox" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +

format:
	isort .
	black .

lint:
	flake8 .
	mypy .

test:
	pytest

check: format lint test

pre-commit-install:
	pre-commit install

pre-commit:
	pre-commit run --all --hook-stage pre-push
