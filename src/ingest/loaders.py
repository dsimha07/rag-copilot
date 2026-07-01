"""Load markdown docs from the corpus. (FR-3 — markdown only for the skeleton;
html/pdf/txt loaders are added during hardening.)
"""
from __future__ import annotations
from pathlib import Path


def load_markdown(corpus_dir: Path) -> list[dict]:
    docs = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        docs.append({
            "doc_id": path.stem,
            "source_name": path.name,
            "source_path": str(path),
            "text": path.read_text(encoding="utf-8"),
        })
    return docs
