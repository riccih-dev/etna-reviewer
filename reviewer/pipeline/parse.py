"""Stage 1 — turn pasted manuscript text into sentences with provenance.

Real PDF ingestion would give you true page numbers; here we simulate
pagination from word count (~180 words/page) so every claim and every
piece of evidence still carries a citable location.
"""

from __future__ import annotations

import re

from ..models import Sentence

WORDS_PER_PAGE = 180

KNOWN_HEADINGS = {
    "abstract", "introduction", "related work", "background", "methods",
    "method", "methodology", "experiments", "experimental setup", "results",
    "discussion", "conclusion", "conclusions", "limitations", "appendix",
    "references", "acknowledgments", "acknowledgements",
}


def _strip_markdown_hashes(line: str) -> str:
    return re.sub(r"^#+\s*", "", line)


def _is_heading(line: str) -> bool:
    line = _strip_markdown_hashes(line.strip())
    if not line or len(line) > 40:
        return False
    if line.endswith((".", ",", ";")):
        return False
    words = line.split()
    if len(words) > 4:
        return False
    return line.lower().strip(" :") in KNOWN_HEADINGS


def parse_manuscript(raw_text: str) -> list[Sentence]:
    sentences: list[Sentence] = []
    section = "Body"
    word_count = 0
    sid = 0
    buffer: list[str] = []

    def flush_paragraph():
        nonlocal word_count, sid
        if not buffer:
            return
        paragraph = " ".join(buffer).strip()
        buffer.clear()
        if not paragraph:
            return
        for raw_sentence in re.split(r"(?<=[.!?])\s+", paragraph):
            text = raw_sentence.strip()
            if not text:
                continue
            sid += 1
            word_count += len(text.split())
            page = 1 + word_count // WORDS_PER_PAGE
            sentences.append(Sentence(id=f"S{sid}", text=text, section=section, page=page))

    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            continue
        if _is_heading(stripped):
            flush_paragraph()
            section = _strip_markdown_hashes(stripped).strip(" :").title()
            continue
        buffer.append(stripped)

    flush_paragraph()
    return sentences
