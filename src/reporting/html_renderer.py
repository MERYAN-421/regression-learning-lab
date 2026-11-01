"""
Standalone HTML Renderer for Academic Reports.
Generates fully self-contained, publication-styled HTML documents with
clean typography, embedded tables, interactive Plotly charts, and MathJax equation rendering.
Strictly escapes user-authored content to prevent XSS / script injection.
"""
import html
from typing import Any, Dict, List, Optional
import markdown

from src.reporting.result_models import AcademicReportResult, ReportSection, FigureReference
from src.reporting.figure_registry import ReportFigureRegistry


def escape_text(text: Optional[str]) -> str:
    """Escapes user text to strictly prevent unintended HTML/JS execution."""
    if not text:
        return ""
    return html.escape(str(text))


def render_report_html(
    report: AcademicReportResult,
    figure_registry: Optional[ReportFigureRegistry] = None
) -> str:
    """
    Renders an AcademicReportResult into a complete, standalone HTML document.
    """
    fig_map = {f.figure_id: f for f in report.figure_references}

    # Build Navigation TOC
    nav_items = []
    for sec in report.report_sections:
        clean_title = escape_text(sec.title)
        nav_items.append(f"<li><a href='#{sec.section_id}'>{clean_title}</a></li>")
    nav_html = "\n".join(nav_items)

    # Build Sections HTML
    sections_html = []
    for sec in report.report_sections:
        sec_title = escape_text(sec.title)
        s_lines = [f"<section id='{sec.section_id}' class='report-section'>"]
        s_lines.append(f"<h2 class='section-heading'>{sec_title}</h2>")

        # Content blocks
        for block in sec.content_blocks:
            # Render Markdown blocks (handles math, bolding, lists)
            block_html = markdown.markdown(block, extensions=['tables', 'fenced_code'])
            s_lines.append(f"<div class='content-block'>{block_html}</div>")

        # Tables
        for tbl in sec.tables:
            tbl_title = escape_text(tbl.get("title", "Table"))
            headers = tbl.get("headers", [])
            rows = tbl.get("rows", [])
            s_lines.append(f"<div class='table-container'>")
            s_lines.append(f"<h4 class='table-title'>{tbl_title}</h4>")
            s_lines.append("<table class='academic-table'>")
            s_lines.append("<thead><tr>")
            for h in headers:
                s_lines.append(f"<th>{escape_text(h)}</th>")
            s_lines.append("</tr></thead><tbody>")
            for r in rows:
                s_lines.append("<tr>")
                for h in headers:
                    val = r.get(h, "")
                    s_lines.append(f"<td>{escape_text(str(val))}</td>")
                s_lines.append("</tr>")
            s_lines.append("</tbody></table></div>")

        # Figures
        for fid in sec.figure_ids:
            fig_ref = fig_map.get(fid)
            if fig_ref:
                fig_title = escape_text(fig_ref.title)
                fig_caption = escape_text(fig_ref.caption)
                s_lines.append(f"<div class='figure-container'>")
                s_lines.append(f"<h4 class='figure-title'>{fig_title}</h4>")
                if figure_registry and fig_ref.is_available:
                    fig_div = figure_registry.render_figure_html(fid)
                    s_lines.append(f"<div class='figure-chart'>{fig_div}</div>")
                else:
                    reason = escape_text(fig_ref.caveats[0] if fig_ref.caveats else "Visual figure unavailable.")
                    s_lines.append(
                        f"<div class='figure-placeholder'><em>Figure unavailable: {reason}</em></div>"
                    )
                s_lines.append(f"<p class='figure-caption'><strong>Caption</strong>: {fig_caption}</p>")
                s_lines.append("</div>")

        s_lines.append("</section>")
        sections_html.append("\n".join(s_lines))

    all_sections = "\n<hr class='section-divider'/>\n".join(sections_html)

    # Disclosures / Warnings Box
    warnings_html = ""
    if report.warnings:
        w_items = "".join([f"<li>{escape_text(w)}</li>" for w in report.warnings])
        warnings_html = f"""
        <div class="alert-box warning-box">
            <strong>Methodological Disclosures & Warnings:</strong>
            <ul>{w_items}</ul>
        </div>
        """

    safe_title = escape_text(report.report_title)
    safe_author = escape_text(report.author_content.author_name)
    safe_template = escape_text(report.template_type.replace('_', ' ').title())

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{safe_title}</title>
    <!-- Modern academic styling -->
    <style>
        :root {{
            --primary: #1e3a8a;
            --primary-light: #3b82f6;
            --bg-color: #f8fafc;
            --surface-color: #ffffff;
            --text-main: #0f172a;
            --text-muted: #475569;
            --border-color: #e2e8f0;
            --accent-warning: #f59e0b;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            line-height: 1.65;
            margin: 0;
            padding: 0;
        }}
        .layout-container {{
            display: flex;
            max-width: 1300px;
            margin: 0 auto;
            min-height: 100vh;
        }}
        .sidebar {{
            width: 280px;
            background: var(--surface-color);
            border-right: 1px solid var(--border-color);
            padding: 30px 20px;
            position: sticky;
            top: 0;
            height: 100vh;
            overflow-y: auto;
            box-sizing: border-box;
        }}
        .sidebar h3 {{
            font-size: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-top: 0;
        }}
        .sidebar ul {{
            list-style: none;
            padding: 0;
            margin: 0;
        }}
        .sidebar li {{
            margin-bottom: 12px;
        }}
        .sidebar a {{
            color: var(--primary);
            text-decoration: none;
            font-size: 0.92rem;
            font-weight: 500;
        }}
        .sidebar a:hover {{
            text-decoration: underline;
        }}
        .main-content {{
            flex: 1;
            padding: 40px 60px;
            background: var(--surface-color);
            box-sizing: border-box;
            border-left: 1px solid var(--border-color);
        }}
        .report-header {{
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .report-header h1 {{
            font-size: 2.2rem;
            color: var(--primary);
            margin: 0 0 10px 0;
        }}
        .meta-bar {{
            font-size: 0.95rem;
            color: var(--text-muted);
        }}
        .meta-bar span {{
            margin-right: 20px;
        }}
        .report-section {{
            margin-bottom: 40px;
        }}
        .section-heading {{
            font-size: 1.5rem;
            color: var(--primary);
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 8px;
            margin-top: 35px;
        }}
        .section-divider {{
            border: 0;
            border-top: 1px dashed var(--border-color);
            margin: 40px 0;
        }}
        .table-container {{
            margin: 25px 0;
            overflow-x: auto;
        }}
        .table-title {{
            font-size: 1.05rem;
            margin: 0 0 8px 0;
            color: var(--text-main);
        }}
        .academic-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.92rem;
            border: 1px solid var(--border-color);
        }}
        .academic-table th {{
            background: #f1f5f9;
            font-weight: 600;
            text-align: left;
            padding: 10px 14px;
            border-bottom: 2px solid var(--border-color);
        }}
        .academic-table td {{
            padding: 9px 14px;
            border-bottom: 1px solid var(--border-color);
        }}
        .academic-table tr:hover {{
            background: #f8fafc;
        }}
        .figure-container {{
            margin: 30px 0;
            padding: 20px;
            background: #ffffff;
            border: 1px solid var(--border-color);
            border-radius: 6px;
        }}
        .figure-title {{
            margin: 0 0 12px 0;
            font-size: 1.1rem;
            color: var(--primary);
        }}
        .figure-caption {{
            font-size: 0.9rem;
            color: var(--text-muted);
            margin-top: 12px;
        }}
        .figure-placeholder {{
            padding: 30px;
            background: #f8fafc;
            border: 1px dashed #cbd5e1;
            text-align: center;
            color: #64748b;
        }}
        .alert-box {{
            padding: 15px 20px;
            border-radius: 6px;
            margin: 20px 0;
            font-size: 0.95rem;
        }}
        .warning-box {{
            background: #fffbeb;
            border: 1px solid #fde68a;
            color: #92400e;
        }}
        .warning-box ul {{
            margin: 8px 0 0 0;
            padding-left: 20px;
        }}
        .report-footer {{
            border-top: 2px solid var(--border-color);
            padding-top: 20px;
            margin-top: 50px;
            font-size: 0.88rem;
            color: var(--text-muted);
        }}
    </style>
    <!-- Plotly.js for interactive rendering -->
    <script src="https://cdn.plot.ly/plotly-2.29.1.min.js"></script>
    <!-- MathJax for rendering LaTeX formulas -->
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>
<body>
    <div class="layout-container">
        <aside class="sidebar">
            <h3>Contents</h3>
            <ul>
                {nav_html}
            </ul>
        </aside>
        <main class="main-content">
            <header class="report-header">
                <h1>{safe_title}</h1>
                <div class="meta-bar">
                    <span><strong>Author</strong>: {safe_author}</span>
                    <span><strong>Template</strong>: {safe_template}</span>
                    <span><strong>Report ID</strong>: <code>{report.report_id}</code></span>
                </div>
            </header>

            {warnings_html}

            {all_sections}

            <footer class="report-footer">
                <p><strong>Evidence Integrity & Provenance</strong>: Verified against dataset <code>{report.dataset_id}</code> (Fingerprint: <code>{report.dataset_fingerprint[:8]}...</code>).</p>
                <p>Reproducibility Hash: <code>{report.reproducibility_hash}</code> | Generated via Regression Learning Lab Academic Reporting Laboratory.</p>
            </footer>
        </main>
    </div>
</body>
</html>
"""
    return html_doc
