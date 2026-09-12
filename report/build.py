"""Main orchestrator for compiling publication-grade PDF research reports."""

import argparse
import os

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate

from report.sections import (
    build_anderson_section,
    build_benchmark_table_section,
    build_curvature_section,
    build_header_and_summary,
    build_industry_section,
    build_phase_space_section,
    build_reproduction_and_conclusion,
    build_results_section,
)
from report.styles import NumberedCanvas, get_report_styles

DEFAULT_PDF_NAME = "Anderson_Acceleration_Comprehensive_Research_Report.pdf"


def build_pdf(filename=None):
    """Compile the complete multi-page publication-grade PDF research report."""
    if filename is None:
        filename = os.path.join(os.getcwd(), DEFAULT_PDF_NAME)

    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = get_report_styles()
    story = []

    # Assemble all report sections
    build_header_and_summary(story, styles)
    build_phase_space_section(story, styles)
    build_curvature_section(story, styles)
    build_anderson_section(story, styles)
    build_benchmark_table_section(story, styles)
    build_results_section(story, styles)
    build_industry_section(story, styles)
    build_reproduction_and_conclusion(story, styles)

    # Build Document with two-pass canvas for dynamic page numbers
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully built: {filename}")
    return filename


def main():
    """CLI entry point for PDF generation."""
    parser = argparse.ArgumentParser(description="Generate comprehensive PDF research report")
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_PDF_NAME,
        help="Path for generated PDF file",
    )
    args = parser.parse_args()
    build_pdf(args.output)


if __name__ == "__main__":
    main()
