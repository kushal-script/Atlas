"""Render the two page failure analysis the addendum requires.

The deliverable is a pdf of at most two pages. It was previously a binary
committed without a builder, which let the pdf and the markdown beside it drift
apart until they disagreed on a headline number. This script makes the pdf a
build product of `docs/phase2_failure_analysis.md`, so correcting the prose is
the only way to change the deliverable, and refuses to write a file that runs
over the limit.

    .venv/bin/python scripts/build_failure_analysis.py
"""

import argparse
import re
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer)

REPO = Path(__file__).resolve().parent.parent
PAGE_LIMIT = 2


def _inline(text):
    """Markdown emphasis and code to reportlab markup, escaping the rest."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Courier" size="7.2">\1</font>', text)
    return text


def build(src, out):
    body = ParagraphStyle("body", fontName="Helvetica", fontSize=7.7, leading=9.5,
                          alignment=TA_JUSTIFY, spaceAfter=4.2)
    h1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=12.5, leading=14,
                        spaceAfter=3, textColor=HexColor("#111111"))
    h2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=8.9, leading=11,
                        spaceBefore=5, spaceAfter=2.5, textColor=HexColor("#7a2b0a"))

    flow = []
    for block in [b.strip() for b in src.read_text().split("\n\n") if b.strip()]:
        one = " ".join(line.strip() for line in block.splitlines())
        if one.startswith("## "):
            flow.append(Paragraph(_inline(one[3:]), h2))
        elif one.startswith("# "):
            flow.append(Paragraph(_inline(one[2:]), h1))
        else:
            flow.append(Paragraph(_inline(one), body))

    doc = BaseDocTemplate(str(out), pagesize=A4,
                          leftMargin=11 * mm, rightMargin=11 * mm,
                          topMargin=10 * mm, bottomMargin=10 * mm,
                          title="Drift Sense, Phase 2 failure analysis",
                          author="Team Atlas")
    gap, n = 6 * mm, 2
    col = (doc.width - gap * (n - 1)) / n
    frames = [Frame(doc.leftMargin + i * (col + gap), doc.bottomMargin, col,
                    doc.height, leftPadding=0, rightPadding=0,
                    topPadding=0, bottomPadding=0) for i in range(n)]
    doc.addPageTemplates([PageTemplate(id="two", frames=frames)])
    doc.build(flow)

    pages = len(re.findall(rb"/Type\s*/Page[^s]", out.read_bytes()))
    if pages > PAGE_LIMIT:
        raise SystemExit(f"failure analysis rendered {pages} pages, the limit is {PAGE_LIMIT}")
    return pages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=REPO / "docs/phase2_failure_analysis.md")
    ap.add_argument("--out", type=Path, default=REPO / "submission/failure_analysis.pdf")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pages = build(args.source, args.out)
    print(f"  wrote {args.out.relative_to(REPO)}, {pages} pages, "
          f"{args.out.stat().st_size / 1024:.0f} KB, from {args.source.name}")


if __name__ == "__main__":
    main()
