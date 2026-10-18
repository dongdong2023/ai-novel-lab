from __future__ import annotations

import json
from pathlib import Path

from novel_lab.understanding.pipeline import (
    UnderstandingPipeline,
)

DEFAULT_GOLD = (
    Path(__file__).resolve().parents[1] / "eval" / "gold.jsonl"
)


def load_gold(path: Path | None = None) -> list[dict]:
    path = path or DEFAULT_GOLD
    cases = []
    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        cases.append(json.loads(line))
    return cases


def evaluate(
    gold: list[dict],
    conn,
    client=None,
    *,
    mock: bool = False,
) -> dict:
    """AIHOT SelectBench 思想的简化版：

    - coverage：成功产出可解析结果的样本比例（失败/格式错都算没完成）
    - prefilter_acc：gold 标了 prefilter 的样本上的准确率
    - doc_kind_acc：gold 标了 doc_kind 且 PASS 的样本上的准确率
    """

    pipeline = UnderstandingPipeline(conn, client)

    total = len(gold)
    done = 0
    prefilter_hits = prefilter_total = 0
    kind_hits = kind_total = 0
    failures = []

    for index, case in enumerate(gold):
        doc = dict(case["document"])
        doc.setdefault("id", -(index + 1))

        result = pipeline.process(
            doc, doc.get("content", ""), mock=mock
        )

        ok = result["prefilter"] in {
            "PASS",
            "BLOCK",
            "UNKNOWN",
        } and (
            result["prefilter"] != "PASS"
            or isinstance(result["structure"], dict)
        )

        structure = result.get("structure")
        if isinstance(structure, dict) and structure.get(
            "parse_error"
        ):
            ok = False

        if not ok:
            failures.append(case.get("caseId", f"case{index}"))
            continue

        done += 1

        gold_pf = case.get("gold", {}).get("prefilter")
        if gold_pf:
            prefilter_total += 1
            if result["prefilter"] == gold_pf:
                prefilter_hits += 1

        gold_kind = case.get("gold", {}).get("doc_kind")
        if (
            gold_kind
            and result["prefilter"] == "PASS"
            and isinstance(structure, dict)
            and not structure.get("parse_error")
        ):
            kind_total += 1
            if structure.get("doc_kind") == gold_kind:
                kind_hits += 1

    def ratio(hits, total_n):
        return (
            round(hits / total_n, 4) if total_n else None
        )

    return {
        "total": total,
        "coverage": ratio(done, total),
        "prefilter_acc": ratio(
            prefilter_hits, prefilter_total
        ),
        "prefilter_n": prefilter_total,
        "doc_kind_acc": ratio(kind_hits, kind_total),
        "doc_kind_n": kind_total,
        "failures": failures,
    }


def format_report(report: dict) -> str:
    lines = [
        f"样本: {report['total']}",
        f"coverage(有效完成率): {report['coverage']}",
        f"prefilter准确率: {report['prefilter_acc']}"
        f" (n={report['prefilter_n']})",
        f"doc_kind准确率: {report['doc_kind_acc']}"
        f" (n={report['doc_kind_n']})",
    ]
    if report["failures"]:
        lines.append(
            "失败样本: " + ", ".join(report["failures"])
        )
    return "\n".join(lines)
