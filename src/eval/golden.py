"""Load the golden Q&A eval set from evals/golden_qa.jsonl. (FR-17)"""
from __future__ import annotations
import json
from pathlib import Path
from pydantic import BaseModel

GOLDEN_PATH = Path(__file__).resolve().parents[2] / "evals" / "golden_qa.jsonl"


class GoldenEntry(BaseModel):
    id: str
    category: str
    difficulty: str
    question: str
    expected_answer_contains: list[str]
    expected_chunks: list[str]
    no_answer: bool


def load_golden(path: Path | None = None) -> list[GoldenEntry]:
    """Load all entries from the golden Q&A JSONL file."""
    p = path or GOLDEN_PATH
    entries = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(GoldenEntry(**json.loads(line)))
    return entries