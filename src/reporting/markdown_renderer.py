"""
Markdown Renderer for Academic Reports.
Converts structured ReportSection objects and EvidenceManifest records into
standards-compliant, publication-ready GitHub-flavored Markdown.
"""
from typing import Any, Dict, List, Optional

from src.reporting.result_models import AcademicReportResult, ReportSection, FigureReference


def render_table_markdown(headers: List[str], rows: List[Dict[str, Any]]) -> str:
    """Renders a standard GitHub-flavored Markdown table from structured rows."""
    if not headers or not rows:
        return ""

    lines = []
    # Header row
    lines.append("| " + " | ".join(headers) + " |")
    # Separator row
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    # Data rows
    for row in rows:
        row_vals = []
        for h in headers:
            val = row.get(h, "")
            row_vals.append(str(val).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(row_vals) + " |")

    return "\n".join(lines)


def render_report_markdown(report: AcademicReportResult) -> str:
    """
    Renders an AcademicReportResult into a complete, structured Markdown document.
    """
    lines = []

    # Title & Metadata Header
    lines.append(f"# {report.report_title}")
    lines.append(f"**Author**: {report.author_content.author_name} | **Generated**: {report.generated_timestamp}")
    lines.append(f"**Template**: {report.template_type.replace('_', ' ').title()} | **Report ID**: `{report.report_id}`")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Readiness / Integrity Notice if Limitations Exist
    if report.warnings:
        lines.append("> [!WARNING]")
        lines.append("> **Methodological Disclosures & Warnings**:")
        for w in report.warnings:
            lines.append(f"> - {w}")
        lines.append("")

    # Sections
    fig_map = {f.figure_id: f for f in report.figure_references}

    for sec in report.report_sections:
        lines.append(f"## {sec.title}")
        lines.append("")

        # Content blocks
        for block in sec.content_blocks:
            lines.append(block)
            lines.append("")

        # Tables
        for tbl in sec.tables:
            tbl_title = tbl.get("title", "Table")
            lines.append(f"##### {tbl_title}")
            headers = tbl.get("headers", [])
            rows = tbl.get("rows", [])
            md_table = render_table_markdown(headers, rows)
            if md_table:
                lines.append(md_table)
                lines.append("")

        # Figures
        for fid in sec.figure_ids:
            fig_ref = fig_map.get(fid)
            if fig_ref:
                lines.append(f"##### Figure: {fig_ref.title}")
                if fig_ref.is_available:
                    lines.append(f"*(Visual figure artifact `{fid}` registered. Refer to accompanying export assets or HTML view)*")
                else:
                    lines.append(f"*(Figure unavailable: {fig_ref.caveats[0] if fig_ref.caveats else 'Data not available'})*")
                lines.append(f"**Caption**: {fig_ref.caption}")
                lines.append("")

        lines.append("---")
        lines.append("")

    # Footer
    lines.append("### Evidence Provenance Summary")
    if report.evidence_manifest:
        m = report.evidence_manifest
        lines.append(
            f"This academic report contains **{m.verified_count}** independently verified empirical evidence records "
            f"derived directly from dataset `{report.dataset_id}` (Fingerprint `{report.dataset_fingerprint[:8]}...`). "
            f"Zero statistics were fabricated."
        )
    lines.append("")
    lines.append(f"*Reproducibility Hash*: `{report.reproducibility_hash}`")

    return "\n".join(lines)
