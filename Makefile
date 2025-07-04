# ReasonIt - Advanced LLM Reasoning Architecture
# Development and deployment utilities

.PHONY: help install dev-install test test-verbose test-coverage lint format type-check clean docs serve benchmark examples run cli

# Default target
help:
	@echo "ReasonIt Development Commands"
	@echo "============================"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  install      Install ReasonIt in current environment"
	@echo "  dev-install  Install ReasonIt with development dependencies"
	@echo "  clean        Clean build artifacts and cache files"
	@echo ""
	@echo "Development & Testing:"
	@echo "  test         Run test suite"
	@echo "  test-verbose Run tests with verbose output"
	@echo "  test-coverage Run tests with coverage report"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with black and isort"
	@echo "  type-check   Run type checking with mypy"
	@echo ""
	@echo "Usage & Examples:"
	@echo "  run          Start ReasonIt CLI interface"
	@echo "  cli          Start ReasonIt CLI interface (alias)"
	@echo "  serve        Start ReasonIt API server"
	@echo "  examples     Run example scripts"
	@echo "  benchmark    Run benchmark suite"
	@echo ""
	@echo "Documentation:"
	@echo "  docs         Generate documentation"
	@echo ""
	@echo "Environment Variables:"
	@echo "  OPENAI_API_KEY       OpenAI API key for GPT models"
	@echo "  REASONIT_LOG_LEVEL   Logging level (DEBUG, INFO, WARNING, ERROR)"
	@echo "  REASONIT_MODEL       Default model (default: gpt-4o-mini)"

# Installation
install:
	pip install -e .

dev-install:
	pip install -e ".[dev,docs,benchmarks]"
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-verbose:
	pytest tests/ -v -s --tb=long

test-coverage:
	pytest tests/ --cov=. --cov-report=html --cov-report=term-missing

# Code Quality
lint:
	flake8 --max-line-length=120 --extend-ignore=E203,W503 .
	black --check --diff .
	isort --check-only --diff .

format:
	black .
	isort .

type-check:
	mypy --ignore-missing-imports --no-strict-optional .

# Usage
run:
	python reasonit.py

cli:
	python reasonit.py

serve:
	python -m reasonit serve

examples:
	@echo "Running example scripts..."
	@for example in examples/*.py; do \
		echo "Running $$example..."; \
		python "$$example" || true; \
	done

benchmark:
	python -m pytest benchmarks/ -v --tb=short

# Documentation
docs:
	@echo "Documentation generation not yet implemented"
	@echo "Will use Sphinx to generate docs from docstrings"

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + || true
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf sessions/
	rm -rf checkpoints/
	rm -f reasonit.log

# Advanced targets
validate: lint type-check test
	@echo "All validation checks passed!"

quick-test:
	pytest tests/test_models.py tests/test_agents.py -v

integration-test:
	pytest tests/test_integration.py -v

performance-test:
	pytest benchmarks/ -v --benchmark-only

# Docker targets (for future use)
docker-build:
	@echo "Docker support not yet implemented"

docker-run:
	@echo "Docker support not yet implemented"

# Release targets (for future use)
build:
	python -m build

upload-test:
	python -m twine upload --repository testpypi dist/*

upload:
	python -m twine upload dist/*

# Check environment
check-env:
	@echo "Checking environment setup..."
	@python -c "import sys; print(f'Python version: {sys.version}')"
	@python -c "import openai; print('OpenAI library available')" || echo "OpenAI library not installed"
	@python -c "import os; print(f'API key configured: {bool(os.getenv(\"OPENAI_API_KEY\"))}')"
	@echo "Environment check complete"

# Development server with auto-reload
dev-serve:
	@echo "Development server not yet implemented"
	@echo "Will use uvicorn with auto-reload for API server"

# Performance profiling
profile:
	python -m cProfile -o profile.stats -m reasonit query "What is machine learning?"
	python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

# Memory profiling
memory-profile:
	python -m memory_profiler reasonit.py query "What is artificial intelligence?"