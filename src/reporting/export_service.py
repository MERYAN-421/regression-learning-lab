"""
Export Service for Academic Reports.
Provides utilities for generating downloadable files:
- Markdown document (.md)
- Standalone HTML document (.html)
- Audit manifest (.json)
- Reproducibility ZIP package (.zip containing report.md, report.html, manifest.json, reproduce.py, README, requirements.txt)
"""
import io
import json
import zipfile
from typing import Optional

from src.reporting.result_models import AcademicReportResult
from src.reporting.source_resolver import ResolvedReportSources
from src.reporting.figure_registry import ReportFigureRegistry
from src.reporting.markdown_renderer import render_report_markdown
from src.reporting.html_renderer import render_report_html
from src.reporting.reproduction_script import (
    generate_reproduction_script,
    generate_reproduction_readme,
    generate_requirements_txt
)


class ReportExportService:
    """
    Assembles exportable assets from an AcademicReportResult.
    """

    def __init__(
        self,
        report: AcademicReportResult,
        sources: Optional[ResolvedReportSources] = None,
        figure_registry: Optional[ReportFigureRegistry] = None
    ):
        self.report = report
        self.sources = sources
        self.figure_registry = figure_registry

    def export_markdown(self) -> str:
        """Generates standard academic markdown text."""
        return render_report_markdown(self.report)

    def export_html(self) -> str:
        """Generates standalone HTML text."""
        return render_report_html(self.report, figure_registry=self.figure_registry)

    def export_manifest_json(self, indent: int = 2) -> str:
        """Generates the verified evidence manifest as formatted JSON."""
        if self.report.evidence_manifest is None:
            return "{}"
        return self.report.evidence_manifest.to_json(indent=indent)

    def export_reproduction_script(self) -> str:
        """Generates the standalone Python reproduction script."""
        if self.sources is None:
            return "# Reproduction sources not provided to export service.\n"
        return generate_reproduction_script(self.report, self.sources)

    def export_reproduction_zip(self) -> bytes:
        """
        Creates a complete in-memory ZIP archive containing all report assets.
        """
        zip_buffer = io.BytesIO()

        md_content = self.export_markdown()
        html_content = self.export_html()
        json_content = self.export_manifest_json()
        py_content = self.export_reproduction_script()
        readme_content = generate_reproduction_readme(self.report)
        reqs_content = generate_requirements_txt()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("report.md", md_content.encode("utf-8"))
            zf.writestr("report.html", html_content.encode("utf-8"))
            zf.writestr("manifest.json", json_content.encode("utf-8"))
            zf.writestr("reproduce.py", py_content.encode("utf-8"))
            zf.writestr("README_REPRODUCTION.md", readme_content.encode("utf-8"))
            zf.writestr("requirements.txt", reqs_content.encode("utf-8"))

        zip_buffer.seek(0)
        return zip_buffer.getvalue()
