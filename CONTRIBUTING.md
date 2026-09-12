# Contributing to Anderson Acceleration Research

Thank you for your interest in contributing to this project. This document provides development setup instructions, guidelines for code quality, testing procedures, and our contribution workflow.

---

## Development Setup

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Git
- Recommended: Virtual environment (`venv` or `conda`)

### 2. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/tkimingi25-spec/Trajectory-Based-Optimization-and-Safeguarded-Anderson-Acceleration.git
cd Trajectory-Based-Optimization-and-Safeguarded-Anderson-Acceleration

# Create and activate virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Install core and development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

To reproduce the exact verified environment:
```bash
pip install -r requirements-lock.txt
```

---

## Running Tests

We use `pytest` for all unit, statistical, and smoke tests.

```bash
# Run all fast tests (recommended during development)
pytest -q -m "not slow"

# Run complete test suite including smoke benchmarks
pytest -q

# Run with test coverage report
pytest --cov=src --cov-report=term-missing
```

---

## Code Quality and Formatting

All code must pass linting and formatting checks before submission:

```bash
# Check code style with Ruff
ruff check .

# Automatically fix safe lint issues
ruff check --fix .

# Check formatting with Black
black --check .

# Auto-format files with Black
black .
```

### Code Style Guidelines
- Line length: 120 characters
- Docstrings: Google style or standard NumPy style on all public classes, functions, and modules
- Modularity: Keep modules focused and under 250–300 lines of code
- Avoid hardcoded machine-specific absolute paths; use `os.path` and relative references

---

## Git Workflow & Conventional Commits

We follow [Conventional Commits](https://www.conventionalcommits.org/) for clear, structured history:

- `feat:` New features, algorithms, or optimizers
- `fix:` Bug fixes
- `test:` Adding or updating tests
- `refactor:` Code restructuring without behavior changes
- `docs:` Documentation improvements
- `style:` Formatting or lint fixes
- `chore:` Maintenance, dependencies, or configuration

### Pull Request Process
1. Create a feature branch: `git checkout -b feat/your-feature-name`
2. Implement your changes with corresponding tests
3. Ensure `pytest -q`, `ruff check .`, and `black --check .` all pass
4. Commit with descriptive conventional commit messages
5. Open a Pull Request describing your motivation and results

---

## License Notice

Source code in this repository is licensed under the PolyForm Noncommercial License 1.0.0. Please review `LICENSE` before submitting contributions.
