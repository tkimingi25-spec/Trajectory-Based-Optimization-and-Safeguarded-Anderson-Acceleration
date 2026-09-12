.PHONY: help install install-dev install-lock test test-full test-cov lint format reproduce smoke report clean

PYTHON ?= python

help:
	@echo "Anderson Acceleration Research Repository"
	@echo "Available commands:"
	@echo "  make install       Install core dependencies"
	@echo "  make install-dev   Install development and test dependencies"
	@echo "  make install-lock  Install exact pinned dependencies from lockfile"
	@echo "  make test          Run fast unit tests (-m 'not slow')"
	@echo "  make test-full     Run full test suite including smoke tests"
	@echo "  make test-cov      Run tests with coverage report"
	@echo "  make lint          Run ruff and black check"
	@echo "  make format        Run ruff and black auto-formatting"
	@echo "  make smoke         Run quick smoke test suite (--quick)"
	@echo "  make reproduce     Run full 17-experiment reproduction suite"
	@echo "  make report        Compile publication-grade PDF research report"
	@echo "  make clean         Clean bytecode and temporary test files"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -r requirements-dev.txt

install-lock:
	$(PYTHON) -m pip install -r requirements-lock.txt

test:
	pytest -q -m "not slow"

test-full:
	pytest -q

test-cov:
	pytest --cov=src --cov-report=term-missing

lint:
	ruff check .
	black --check .

format:
	ruff check --fix .
	black .

smoke:
	$(PYTHON) experiments/run_all.py --quick

reproduce:
	$(PYTHON) experiments/run_all.py

report:
	$(PYTHON) build_pdf_report.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov .ruff_cache 2>/dev/null || true
