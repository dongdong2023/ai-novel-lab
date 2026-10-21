from __future__ import annotations

import json
import random
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

from novel_lab.dataset_factory import (
    _read_doc,
    load_novel_knowledge,
    trainable_documents,
)
from novel_lab.knowledge_extract import STATS_EXTRACTOR
from novel_lab.settings import DATA_DIR

# 质量门禁（对应用户总规划第八阶段）：
#   1 格式检查   字段完整、长度合理
#   2 重复检查   完全重复与近重复（同题同答头）
#   3 事实一致性 答案必须能溯源：引用型样本的 quote 必须在 output 里；
#               出场次数必须与知识库一致；写作切片必须与原文段落连续
#   4 来源检查   source_doc 必须仍是可训练 rights 的文档
#   5 质量评分   硬失败即弃；薄答案扣分
#   6 切分       按类别分层 8:1:1，固定种子可复现

HARD_FAIL = "hard_fail"
THIN_OUTPUT = "thin_output"


def _paragraphs(text: str) -> list[str]:
    return [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]


def check_format(sample: dict) -> str | None:
    instruction = (sample.get("instruction") or "").strip()
    output = (sample.get("output") or "").strip()
    if not instruction or not output:
        return "字段缺失"
    if not isinstance(sample.get("input", ""), str):
        return "input类型错误"
    if len(output) < 10:
        return "output过短"
    if len(output) > 3000:
        return "output超长"
    return None


def check_fact(
    sample: dict,
    conn: sqlite3.Connection,
    novel_title: str | None = None,
) -> str | None:
    """事实一致性：答案必须能在知识库或原文里找到。

    novel_title 给出时，原文查询按书名+文档名联合过滤，
    避免两本书有同名文档时串书（P1-2.3）；不给时退回旧的
    按文档名查询（兼容直接调用）。
    """

    category = sample.get("category")
    output = sample.get("output", "")

    # 引用型样本：knowledge 的 quote 必须出现在 output
    kid = sample.get("knowledge_id")
    if kid:
        row = conn.execute(
            "SELECT payload FROM knowledge WHERE id=?",
            (kid,),
        ).fetchone()
        if row is None:
            return "知识条目已消失"
        payload = json.loads(row["payload"])
        quote = (payload.get("quote") or "").strip()
        if quote and quote not in output:
            return "引用与知识库不一致"

    # 出场次数问答：数字必须与知识库重算一致
    if category == "qa" and "共提及" in output:
        m_qa = re.search(
            r"《(.+?)》前\d+章中，(.+?)主要出现",
            sample["instruction"],
        )
        m_num = re.search(r"共提及(\d+)次", output)
        if m_qa and m_num:
            book_title = novel_title or m_qa.group(1)
            name = m_qa.group(2)
            claimed = int(m_num.group(1))
            real = conn.execute(
                "SELECT COALESCE(SUM(json_extract(payload,"
                " '$.mentions')),0) AS n FROM knowledge k"
                " JOIN documents d ON d.id=k.document_id"
                " JOIN novels nv ON nv.id=k.novel_rowid"
                " WHERE nv.title=? AND k.kind='character'"
                " AND k.extractor=?"
                " AND json_extract(payload,'$.name')=?"
                " AND d.rights IN ('authorized',"
                " 'user_provided','open_license')",
                (book_title, STATS_EXTRACTOR, name),
            ).fetchone()["n"]
            if real != claimed:
                return f"提及次数不符(claim={claimed},real={real})"

    # 写作样本：按 subtype 分策略校验（P0-1.4）——v1 无
    # subtype 的续写切片与 continuation_v2/dialogue_tension
    # 校验段落连续性；beat_to_prose 的 input 是节拍规格不是
    # 正文，只能校验 output 是原文真实段落；cliffhanger_hook
    # 校验 output 段落在 input 末段之后且不与 input 重叠。
    if category == "writing":
        if novel_title:
            doc_row = conn.execute(
                "SELECT d.raw_path FROM documents d"
                " JOIN novels nv ON nv.id = d.novel_rowid"
                " WHERE nv.title=? AND d.title=?",
                (
                    novel_title,
                    sample.get("source_doc", ""),
                ),
            ).fetchone()
        else:
            doc_row = conn.execute(
                "SELECT raw_path FROM documents"
                " WHERE title=?",
                (sample.get("source_doc", ""),),
            ).fetchone()
        if doc_row is None:
            return "来源文档不存在"
        text = _read_doc(doc_row["raw_path"])
        paras = _paragraphs(text)
        in_tail = _paragraphs(sample.get("input", ""))
        out_head = _paragraphs(output)
        if not out_head:
            return "切片为空"

        subtype = sample.get("subtype")

        if subtype == "beat_to_prose":
            # output 必须逐段是原文真实段落
            for p in out_head:
                if p not in paras:
                    return "output段落不在原文"
            return None

        if not in_tail:
            return "切片为空"

        if subtype == "cliffhanger_hook":
            # input 末段定位后，output 首段必须在其后出现
            try:
                i = paras.index(in_tail[-1])
            except ValueError:
                return "切片段落不在原文"
            rest = paras[i + 1:]
            if out_head[0] not in rest:
                return "钩子段不在原文input之后"
            return None

        try:
            i = paras.index(in_tail[-1])
            j = paras.index(out_head[0])
        except ValueError:
            return "切片段落不在原文"
        if j != i + 1:
            return "切片不连续"

    return None


def check_source(
    sample: dict, trainable_titles: set[str]
) -> str | None:
    doc = sample.get("source_doc", "")
    if doc == "aggregate":
        return None
    if doc not in trainable_titles:
        return "来源不可训练或不存在"
    return None


def run_gate(
    conn: sqlite3.Connection,
    samples: list[dict],
    novel_title: str,
) -> tuple[list[dict], list[dict], dict]:

    trainable_titles = {
        d["title"] for d in trainable_documents(conn, novel_title)
    }

    seen_exact: set[tuple] = set()
    seen_head: set[tuple] = set()

    kept: list[dict] = []
    dropped: list[dict] = []
    reasons: dict[str, int] = defaultdict(int)

    for sample in samples:

        reason = None

        if reason is None:
            fmt = check_format(sample)
            if fmt:
                reason = f"格式:{fmt}"

        if reason is None:
            key = (
                sample.get("category"),
                sample.get("instruction", "").strip(),
                sample.get("input", "").strip(),
                sample.get("output", "").strip()[:200],
            )
            if key in seen_exact:
                reason = "重复:完全重复"
            else:
                seen_exact.add(key)
                head_key = key[:3] + (
                    sample["output"].strip()[:60],
                )
                if head_key in seen_head:
                    reason = "重复:近重复"
                else:
                    seen_head.add(head_key)

        if reason is None:
            fact = check_fact(sample, conn, novel_title)
            if fact:
                reason = f"事实:{fact}"

        if reason is None:
            src = check_source(sample, trainable_titles)
            if src:
                reason = f"来源:{src}"

        score = 1.0
        if reason is None and len(sample["output"]) < 30:
            score = 0.8
            reasons[THIN_OUTPUT] += 1

        record = dict(sample)
        record["quality_score"] = score
        if reason:
            record["drop_reason"] = reason
            reasons[reason] += 1
            dropped.append(record)
        else:
            kept.append(record)

    report = {
        "total": len(samples),
        "kept": len(kept),
        "dropped": len(dropped),
        "by_reason": dict(reasons),
        "by_category": {
            cat: {
                "kept": sum(
                    1 for s in kept if s["category"] == cat
                ),
                "dropped": sum(
                    1
                    for s in dropped
                    if s["category"] == cat
                ),
            }
            for cat in {
                s["category"] for s in samples
            }
        },
    }
    return kept, dropped, report


def split_dataset(
    kept: list[dict],
    *,
    seed: int = 42,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, list[dict]]:

    rng = random.Random(seed)

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for s in kept:
        by_cat[s["category"]].append(s)

    splits: dict[str, list[dict]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for cat, items in by_cat.items():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, round(n * test_ratio)) if n >= 3 else 0
        n_val = max(1, round(n * val_ratio)) if n >= 3 else 0
        splits["test"].extend(items[:n_test])
        splits["validation"].extend(items[n_test : n_test + n_val])
        splits["train"].extend(items[n_test + n_val :])

    return splits


def write_split(
    splits: dict[str, list[dict]],
    report: dict,
    out_dir: Path | None = None,
) -> Path:

    if out_dir is None:
        out_dir = DATA_DIR / "datasets" / "v1"

    out_dir.mkdir(parents=True, exist_ok=True)

    for name, items in splits.items():
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for s in items:
                f.write(
                    json.dumps(s, ensure_ascii=False) + "\n"
                )

    (out_dir / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return out_dir
