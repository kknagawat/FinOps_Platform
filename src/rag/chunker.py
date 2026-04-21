"""
src/rag/chunker.py
==================
Step 5 — Document Ingestion and Chunking

Chunking strategy decision
--------------------------
Documents: 4 markdown files, each ~2,000–2,900 chars, 9–10 headings.

WHY heading-aware splitting (not fixed-size):
  Policy documents have natural semantic units — one rule per heading section
  (e.g. "### Enterprise Plans", "### Resolution SLA"). Splitting at headings
  ensures each chunk contains a complete, self-contained policy rule rather
  than cutting mid-sentence in the middle of a refund window definition.

WHY 400-char target chunk size:
  Each policy section averages ~200–350 characters. A 400-char target keeps
  most sections as a single chunk. Larger sizes (e.g. 800 chars) would merge
  unrelated sections; smaller sizes would split bullet lists mid-point.

WHY 50-char overlap between adjacent chunks:
  When a section boundary falls mid-sentence (e.g. a bullet list that spans
  two logical groups), a 50-char overlap ensures the next chunk has enough
  context to be understood standalone. For ~400-char chunks this is ~12%
  overlap — enough for continuity, not so much that it creates near-duplicate
  retrieval results.

WHY preserve parent heading as chunk metadata:
  When the agent answers "What does the SLA say about critical tickets?",
  the retrieved chunk must carry its heading ("### First Response SLA") so
  the citation reads: [Source: sla_policy.md — First Response SLA] rather
  than just [Source: sla_policy.md].
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

DOCS_DIR     = Path(__file__).resolve().parents[2] / "docs"
CHUNK_SIZE   = 400   # target characters per chunk
OVERLAP      = 50    # overlap characters between adjacent same-section chunks


@dataclass
class Chunk:
    """
    A single text chunk ready for embedding.

    Attributes
    ----------
    text        : the actual text content to embed and retrieve
    source      : filename (e.g. "refund_policy.md")
    doc_title   : top-level H1 heading of the document
    section     : nearest H2/H3/H4 heading above this chunk
    chunk_id    : unique identifier "<source>::<section>::<index>"
    char_count  : length of text in characters
    """
    text:       str
    source:     str
    doc_title:  str
    section:    str
    chunk_id:   str
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)

    def to_dict(self) -> dict:
        return {
            "text":      self.text,
            "source":    self.source,
            "doc_title": self.doc_title,
            "section":   self.section,
            "chunk_id":  self.chunk_id,
        }


def _split_section_into_chunks(
    text:      str,
    source:    str,
    doc_title: str,
    section:   str,
    start_idx: int,
) -> list[Chunk]:
    """
    Split a single section's text into ≤CHUNK_SIZE character chunks
    with OVERLAP characters of context carried forward.

    If the section text fits in one chunk, it is returned as-is.
    For longer sections (e.g. the pricing table in pricing_tiers.md),
    we split on paragraph boundaries first, then on sentence boundaries,
    never cutting mid-word.
    """
    text = text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []

    # If the whole section fits → single chunk
    if len(text) <= CHUNK_SIZE:
        chunks.append(Chunk(
            text=text,
            source=source,
            doc_title=doc_title,
            section=section,
            chunk_id=f"{source}::{section}::{start_idx}",
        ))
        return chunks

    # Split on paragraph breaks first (double newline)
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    buffer     = ""
    idx        = start_idx

    for para in paragraphs:
        # If adding this paragraph would overflow, flush the buffer
        if buffer and len(buffer) + len(para) + 2 > CHUNK_SIZE:
            chunks.append(Chunk(
                text=buffer.strip(),
                source=source,
                doc_title=doc_title,
                section=section,
                chunk_id=f"{source}::{section}::{idx}",
            ))
            idx += 1
            # Carry overlap: last OVERLAP chars of the flushed buffer
            buffer = buffer[-OVERLAP:].strip() + "\n\n" + para
        else:
            buffer = (buffer + "\n\n" + para).strip() if buffer else para

    # Flush remaining buffer
    if buffer.strip():
        chunks.append(Chunk(
            text=buffer.strip(),
            source=source,
            doc_title=doc_title,
            section=section,
            chunk_id=f"{source}::{section}::{idx}",
        ))

    return chunks


def chunk_document(md_path: Path) -> list[Chunk]:
    """
    Parse a single markdown file and return a list of Chunk objects.

    Algorithm
    ---------
    1. Read the file and split on heading markers (# / ## / ### / ####).
    2. Track the current doc_title (H1) and section (H2/H3/H4).
    3. Accumulate body text between headings.
    4. When a new heading is encountered, flush the accumulated text
       through _split_section_into_chunks().
    5. Each chunk carries the full heading lineage as its section.

    The heading pattern `^(#{1,4} .+)$` matches any markdown heading.
    re.MULTILINE ensures ^ matches at start-of-line, not start-of-string.
    """
    text     = md_path.read_text(encoding="utf-8")
    source   = md_path.name
    all_chunks: list[Chunk] = []

    # Split into alternating [body, heading, body, heading, ...] segments
    heading_pattern = re.compile(r"^(#{1,4} .+)$", re.MULTILINE)
    parts           = heading_pattern.split(text)

    doc_title      = source.replace(".md", "").replace("_", " ").title()
    current_section = "Overview"
    chunk_counter   = 0
    pending_text    = ""

    for part in parts:
        part_stripped = part.strip()
        if not part_stripped:
            continue

        if heading_pattern.match(part_stripped):
            # Flush any pending text before switching section
            if pending_text.strip():
                new_chunks = _split_section_into_chunks(
                    pending_text, source, doc_title, current_section, chunk_counter
                )
                all_chunks.extend(new_chunks)
                chunk_counter += len(new_chunks)
                pending_text = ""

            # Update heading context
            level   = len(part_stripped) - len(part_stripped.lstrip("#"))
            heading = part_stripped.lstrip("#").strip()

            if level == 1:
                doc_title       = heading
                current_section = "Overview"
            else:
                current_section = heading
        else:
            # Accumulate body text
            pending_text = (pending_text + "\n\n" + part_stripped).strip()

    # Flush any remaining text after the last heading
    if pending_text.strip():
        new_chunks = _split_section_into_chunks(
            pending_text, source, doc_title, current_section, chunk_counter
        )
        all_chunks.extend(new_chunks)

    return all_chunks


def load_all_documents(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    """
    Load and chunk all markdown files in the docs directory.

    Returns
    -------
    list[Chunk] — all chunks from all documents, in document order.
    """
    md_files = sorted(docs_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md files found in {docs_dir}")

    all_chunks: list[Chunk] = []
    for md_file in md_files:
        doc_chunks = chunk_document(md_file)
        all_chunks.extend(doc_chunks)

    return all_chunks


if __name__ == "__main__":
    chunks = load_all_documents()
    print(f"Total chunks: {len(chunks)}\n")
    for c in chunks:
        print(f"[{c.source} — {c.section}]  {c.char_count} chars")
        print(f"  {c.text[:80]}...")
        print()