# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2026-09-12

### Added
- **Test Suite**: Comprehensive unit tests covering Anderson extrapolation, diagnostics, models, paired statistics, and training routines.
- **CI/CD Pipeline**: GitHub Actions continuous integration workflow testing on Python 3.10 and 3.11 with automated linting and coverage checks.
- **Dependency Management**: Pinned lockfile (`requirements-lock.txt`), development dependencies (`requirements-dev.txt`), and automated Dependabot configuration (`.github/dependabot.yml`).
- **Code Quality**: Ruff linter configuration and Black formatter integration in `pyproject.toml`.
- **Documentation**: Developer contribution guidelines (`CONTRIBUTING.md`), environment configuration example (`.env.example`), and project change history (`CHANGELOG.md`).
- **Reproducibility**: Configurable YAML experiment execution and structured logging infrastructure.

### Changed
- **Report Architecture**: Modularized 650-line `build_pdf_report.py` into dedicated `report` package (`styles.py`, `sections.py`, `build.py`), keeping all individual modules under 250 LOC.
- **Path Portability**: Replaced machine-specific hardcoded absolute paths with dynamic repository-relative paths across all experiment scripts.
- **Repository Hygiene**: Added comprehensive `.gitignore` and untracked compiled bytecode and caches from version control.

---

## [1.0.0] - 2026-08-30

### Added
- Initial open-source release of the Anderson Acceleration & Optimization Phase-Space Telemetry research codebase.
- Implementations of Classical Type-II Anderson acceleration and Ito & Xue snapshot combination with strict-descent safeguards.
- 17 empirical benchmark experiments spanning toy quadratic ravines, Baldi-Hornik linear autoencoder strict saddles, and real convolutional networks.
- Two-stage shifted power iteration for matrix-free Hessian minimum eigenvalue estimation ($\lambda_{\min}$).
- Publication-grade PDF research report generator using ReportLab.
