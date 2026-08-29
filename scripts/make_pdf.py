"""Render the Phase 2 analysis markdown into a PDF (offline, pure-Python).

Pandoc is present but has no PDF engine installed (no LaTeX/wkhtmltopdf), so we
render docs/failure_analysis.md (+ architecture.md + citations.md) to a real PDF
with matplotlib's PdfPages. This is a minimal but faithful citable artifact for
the carried Generator & Analysis point.

Usage:
    uv run python scripts/make_pdf.py
"""

import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
OUT = DOCS / "phase2_analysis.pdf"
SOURCES = ["failure_analysis.md", "architecture.md", "citations.md"]

PAGE_W_IN, PAGE_H_IN = 8.27, 11.69
LEFT, RIGHT, TOP, BOTTOM = 0.08, 0.95, 0.94, 0.06
WRAP = 100
BASE = 9  # pt


def _style(line):
    if re.match(r"^#{1,3}\s", line):
        return BASE + 3, "bold"
    if re.match(r"^\s*\|", line) or re.match(r"^---", line):
        return BASE - 1, "normal"
    if line.strip().startswith("- ") or re.match(r"^\s*\d+\.", line):
        return BASE, "normal"
    return BASE, "normal"


def _wrap(line):
    if line.strip() == "":
        return [""]
    return textwrap.wrap(line, WRAP) or [""]


def main():
    pdf = PdfPages(str(OUT))
    fig = plt.figure(figsize=(PAGE_W_IN, PAGE_H_IN))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    y = TOP
    line_h = (BASE / 72.0) / PAGE_H_IN * 1.35

    def new_page():
        nonlocal fig, ax, y
        pdf.savefig(fig)
        plt.close(fig)
        fig = plt.figure(figsize=(PAGE_W_IN, PAGE_H_IN))
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        y = TOP

    for src in SOURCES:
        path = DOCS / src
        if not path.exists():
            continue
        ax.text(LEFT, y, f"\n### source: {src}\n", fontsize=BASE + 2,
                fontweight="bold", transform=ax.transAxes); y -= line_h * 2
        for raw in path.read_text().splitlines():
            for seg in _wrap(raw):
                if y < BOTTOM:
                    new_page()
                size, weight = _style(seg)
                ax.text(LEFT, y, seg, fontsize=size, fontweight=weight,
                        family="DejaVu Sans", transform=ax.transAxes)
                y -= line_h
        y -= line_h * 2
    pdf.savefig(fig)
    plt.close(fig)
    pdf.close()
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
