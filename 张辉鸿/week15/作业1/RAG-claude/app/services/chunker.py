"""Markdown chunking strategies for parsed PDF content."""

import re


def chunk_markdown(
    markdown_text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[dict]:
    """Split markdown text into overlapping chunks with page metadata.

    Each chunk dict: {text, page_start, page_end, chunk_index}
    """
    pages = _split_by_page(markdown_text)
    chunks = []
    idx = 0

    for page_num, page_text in pages:
        page_chunks = _split_page_text(page_text, chunk_size, chunk_overlap)
        for ch_text in page_chunks:
            chunks.append({
                "text": ch_text.strip(),
                "page_start": page_num,
                "page_end": page_num,
                "chunk_index": idx,
            })
            idx += 1

    # Merge consecutive small chunks from the same page if under half chunk_size
    chunks = _merge_small_chunks(chunks, chunk_size)

    return chunks


def _split_by_page(markdown_text: str) -> list[tuple[int, str]]:
    """Split markdown by page markers (MinerU uses <!-- PAGE X --> or similar).

    If no page markers found, treat entire text as page 1.
    """
    # Common page markers: <!-- PAGE X -->, [PAGE X], --- Page X ---
    parts = re.split(
        r"(?:<!--\s*PAGE\s*(\d+)\s*-->|\[PAGE\s*(\d+)\]|---\s*Page\s*(\d+)\s*---)",
        markdown_text,
    )
    pages = []
    current_page = 1
    current_text: list[str] = []

    for i, part in enumerate(parts):
        if part is None:
            continue
        # Check if this is a page number capture group
        page_match = re.match(r"^\d+$", part.strip())
        if page_match:
            if current_text:
                text = "".join(current_text).strip()
                if text:
                    pages.append((current_page, text))
            current_page = int(part.strip())
            current_text = []
        else:
            current_text.append(part)

    # Don't forget the last page
    if current_text:
        text = "".join(current_text).strip()
        if text:
            pages.append((current_page, text))

    if not pages:
        pages = [(1, markdown_text)]

    return pages


def _split_page_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split a single page's text into overlapping chunks, respecting paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_len = len(para)
        if current_len + para_len > chunk_size and current:
            chunks.append(current)
            # Keep overlap: take last `overlap` chars from current as prefix
            if overlap > 0 and len(current) > overlap:
                current = current[-overlap:] + "\n\n" + para
                current_len = len(current)
            else:
                current = para
                current_len = para_len
        else:
            if current:
                current += "\n\n" + para
            else:
                current = para
            current_len += para_len

    if current.strip():
        chunks.append(current)

    return chunks


def _merge_small_chunks(chunks: list[dict], chunk_size: int) -> list[dict]:
    """Merge consecutive chunks from same page that are under half chunk_size."""
    if len(chunks) <= 1:
        return chunks

    merged = []
    buffer = None

    for ch in chunks:
        if buffer is None:
            buffer = ch.copy()
            continue

        same_page = buffer["page_start"] == ch["page_start"]
        total_len = len(buffer["text"]) + len(ch["text"])

        if same_page and total_len < chunk_size + (chunk_size // 2):
            buffer["text"] += "\n\n" + ch["text"]
            buffer["page_end"] = ch["page_end"]
        else:
            merged.append(buffer)
            buffer = ch.copy()

    if buffer is not None:
        merged.append(buffer)

    return merged
