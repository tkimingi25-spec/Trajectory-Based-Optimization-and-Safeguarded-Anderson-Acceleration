"""Publication-grade PDF report compiler.

Thin backwards-compatible wrapper around the modular `report` package.
"""

from report.build import build_pdf, main

__all__ = ["build_pdf", "main"]

if __name__ == "__main__":
    main()
