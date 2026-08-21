"""PDF input path — same sentence/section/page model as the paste-text path,
but page numbers come from the actual PDF instead of a word-count estimate,
and section headers are detected from numbered headings ("4.3 Attention
mechanism") in addition to the plain-word headings parse.py already knows.
"""

from __future__ import annotations

import re

from ..models import Sentence
from .parse import KNOWN_HEADINGS

NUMBERED_HEADING = re.compile(r"^\d+(\.\d+)*\.?\s+\S")
TOP_LEVEL_HEADING = re.compile(r"^\d+\.\s+\S")  # "4. Results" — not "4.1 Classification performance"


def _is_pdf_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 60:
        return False
    if line.endswith((".", ",", ";")):
        return False
    if len(line.split()) > 8:
        return False
    if NUMBERED_HEADING.match(line):
        return True
    return line.lower().strip(" :") in KNOWN_HEADINGS


def extract_pdf_pages(file) -> list[tuple[int, str]]:
    """file: a path, an open binary file, or a file-like object (e.g. Streamlit's UploadedFile)."""
    import pdfplumber

    pages = []
    with pdfplumber.open(file) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            pages.append((i, page.extract_text() or ""))
    return pages


def parse_pdf_pages(pages: list[tuple[int, str]]) -> list[Sentence]:
    sentences: list[Sentence] = []
    section = "Body"
    top_section = "Body"
    sid = 0
    buffer: list[str] = []
    current_page = pages[0][0] if pages else 1

    def flush():
        nonlocal sid
        if not buffer:
            return
        paragraph = " ".join(buffer).strip()
        buffer.clear()
        if not paragraph:
            return
        for raw in re.split(r"(?<=[.!?])\s+", paragraph):
            text = raw.strip()
            if not text:
                continue
            sid += 1
            sentences.append(Sentence(
                id=f"S{sid}", text=text, section=section, page=current_page, top_section=top_section,
            ))

    for page_num, text in pages:
        flush()  # leftover from the previous page belongs to the previous page number
        current_page = page_num
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                flush()
                continue
            if _is_pdf_heading(stripped):
                flush()
                section = stripped.strip(" :").rstrip(".").title()
                if TOP_LEVEL_HEADING.match(stripped) or stripped.lower().strip(" :") in KNOWN_HEADINGS:
                    top_section = re.sub(r"^\d+\.\s*", "", section).strip()
                continue
            buffer.append(stripped)
    flush()
    return sentences
