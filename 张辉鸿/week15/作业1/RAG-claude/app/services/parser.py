"""PDF parsing via MinerU. Calls the `magic-pdf` CLI tool."""

import json
import os
import subprocess
from pathlib import Path

from app.config import settings


class ParseResult:
    """Holds the result of parsing a PDF."""

    def __init__(self, doc_id: str, output_dir: str):
        self.doc_id = doc_id
        self.output_dir = Path(output_dir)
        self.markdown_files: list[Path] = []
        self.image_files: list[dict] = []  # [{path, page_num}]

    @property
    def full_markdown(self) -> str:
        texts = []
        for mf in sorted(self.markdown_files, key=lambda p: p.name):
            texts.append(mf.read_text(encoding="utf-8"))
        return "\n\n".join(texts)


def parse_pdf(doc_id: str, pdf_path: str) -> ParseResult:
    """Run MinerU magic-pdf to parse a PDF into markdown + images.

    Output structure (MinerU default):
        output_dir/
        └── {pdf_name}/
            ├── {pdf_name}.md            # full markdown
            ├── auto/                    # auto-detected layout
            │   └── {pdf_name}.md
            ├── images/                  # extracted images
            │   ├── page_1_img_1.png
            │   └── ...
            └── ...
    """
    output_dir = Path(settings.parsed_dir) / doc_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Call MinerU CLI
    cmd = [
        "magic-pdf",
        "-p", str(pdf_path),
        "-o", str(output_dir),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MinerU parsing failed:\n{e.stderr}")

    result = ParseResult(doc_id, str(output_dir))
    _collect_output_files(result)
    return result


def _collect_output_files(result: ParseResult):
    """Walk the output directory and collect markdown + image files."""
    for root, _dirs, files in os.walk(result.output_dir):
        root_path = Path(root)
        # Skip duplicate auto/ layout folder — prefer root-level files
        rel = root_path.relative_to(result.output_dir)

        for fname in files:
            fpath = root_path / fname
            suffix = fpath.suffix.lower()

            if suffix in (".md",):
                result.markdown_files.append(fpath)
            elif suffix in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
                # Try to extract page number from filename (e.g. page_3_img_1.png)
                page_num = _extract_page_num(fname)
                result.image_files.append({
                    "path": str(fpath),
                    "page_num": page_num,
                })


def _extract_page_num(filename: str) -> int:
    """Try to extract page number from MinerU image filename."""
    import re
    m = re.search(r"page[_-]?(\d+)", filename, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Fallback: try any leading number
    m = re.search(r"^(\d+)", filename)
    return int(m.group(1)) if m else 0
