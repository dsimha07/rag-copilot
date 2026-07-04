"""Generate a markdown eval report. (FR-19)"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parents[2] / "evals" / "reports"


def generate_report(
    results: list[dict],
    aggregated: dict,
    strategy: str,
) -> Path:
    """Write a markdown report and return the path."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"eval_{strategy}_{timestamp}.md"

    ret = aggregated["retrieval"]
    ans = aggregated["answer"]

    lines = [
        f"# RAG Copilot Eval Report",
        f"",
        f"## Summary",
        f"| | |",
        f"|---|---|",
        f"| strategy | {strategy} |",
        f"| total questions | {aggregated['total']} |",
        f"| date | {datetime.now().strftime('%Y-%m-%d %H:%M')} |",
        f"",
        f"## Retrieval Metrics",
        f"| Metric | Value |",
        f"|---|---|",
        f"| correct source retrieved | {ret['correct_source_retrieved']}/{aggregated['total']} ({ret['correct_source_pct']}%) |",
        f"| mean reciprocal rank | {ret['mean_reciprocal_rank']} |",
        f"",
        f"## Answer Metrics",
        f"| Metric | Value |",
        f"|---|---|",
        f"| answers correct | {ans['correct_answers']}/{ans['total_answerable']} ({ans['correct_answer_pct']}%) |",
        f"| correct refusals | {ans['correct_refusals']}/{ans['total_no_answer']} ({ans['correct_refusal_pct']}%) |",
        f"| avg citation support rate | {ans['avg_citation_support_rate']} |",
        f"| avg confidence | {ans['avg_confidence']} |",
        f"",
        f"## By Category",
        f"| Category | Correct | Total | % |",
        f"|---|---|---|---|",
    ]

    for cat, data in aggregated["by_category"].items():
        pct = round(data["correct"] / data["total"] * 100, 1) if data["total"] else 0
        lines.append(f"| {cat} | {data['correct']} | {data['total']} | {pct}% |")

    lines += [
        f"",
        f"## Per-Question Results",
        f"| ID | Category | Difficulty | Correct Source | Answer Correct | Refusal | Confidence |",
        f"|---|---|---|---|---|---|---|",
    ]

    for r in results:
        source  = "✅" if r["correct_source"] else "❌"
        correct = "✅" if r["answer_correct"] else ("n/a" if r["correct_refusal"] is not None else "❌")
        refusal = "✅" if r["correct_refusal"] else ("❌" if r["correct_refusal"] is False else "-")
        conf    = f"{r['confidence']:.2f}" if r["confidence"] else "n/a"
        lines.append(
            f"| {r['id']} | {r['category']} | {r['difficulty']} "
            f"| {source} | {correct} | {refusal} | {conf} |"
        )

    lines += [
        f"",
        f"## Failed Cases",
    ]

    failed = [r for r in results if not r["answer_correct"] and r["correct_refusal"] is None]
    if failed:
        for r in failed:
            lines += [
                f"",
                f"### {r['id']} — {r['question']}",
                f"**Retrieved:** {', '.join(r['retrieved_chunks'])}",
                f"**Answer:** {r['answer_text']}",
            ]
    else:
        lines.append("None — all answerable questions answered correctly.")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path