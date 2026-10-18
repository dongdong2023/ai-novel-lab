from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from novel_lab.ingest import TRAINABLE_RIGHTS
from novel_lab.knowledge_extract import STATS_EXTRACTOR
from novel_lab.settings import DATA_DIR

# 六类训练样本（对应用户总规划第七阶段）：
#   qa           小说问答（人物出场/关系，来自知识库+原文引用）
#   character    人物分析
#   plot         剧情理解
#   world        世界观理解
#   summary      长文本总结（章级事件聚合）
#   writing      写作任务（续写切片：input=前文，output=原文后段）
# 全部确定性生成；Rights Gate：只从可训练 rights 的文档生成。


_CN_DIGITS = "零一二三四五六七八九十"


def _pretty_doc(doc_title: str) -> str:
    """'ch2_deai' -> '第二章'；无数字原样返回。"""
    import re

    m = re.search(r"ch(\d+)", doc_title or "")
    if not m:
        return doc_title
    n = int(m.group(1))
    if n <= 10:
        cn = _CN_DIGITS[n] if n < 10 else "十"
    elif n < 20:
        cn = "十" + _CN_DIGITS[n - 10]
    else:
        cn = _CN_DIGITS[n // 10] + "十" + (
            _CN_DIGITS[n % 10] if n % 10 else ""
        )
    return f"第{cn}章"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def trainable_documents(
    conn: sqlite3.Connection, novel_title: str
) -> list[dict]:
    rows = conn.execute(
        "SELECT d.id, d.title, d.raw_path, d.rights"
        " FROM documents d"
        " JOIN novels n ON n.id = d.novel_rowid"
        " WHERE n.title = ?"
        " AND d.rights IN ('authorized','user_provided',"
        " 'open_license')"
        " ORDER BY d.title",
        (novel_title,),
    ).fetchall()
    return [dict(r) for r in rows]


def _read_doc(raw_path: str) -> str:
    try:
        return Path(raw_path).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return Path(raw_path).read_text(
            encoding="gbk", errors="replace"
        )


def load_novel_knowledge(
    conn: sqlite3.Connection, novel_title: str
) -> list[dict]:
    rows = conn.execute(
        "SELECT k.id, k.kind, k.payload, k.extractor,"
        " k.document_id, d.title AS doc_title, d.rights"
        " FROM knowledge k"
        " JOIN documents d ON d.id = k.document_id"
        " JOIN novels n ON n.id = k.novel_rowid"
        " WHERE n.title = ?"
        " AND d.rights IN ('authorized','user_provided',"
        " 'open_license')"
        " ORDER BY d.title, k.id",
        (novel_title,),
    ).fetchall()
    out = []
    for r in rows:
        payload = json.loads(r["payload"])
        payload["_id"] = r["id"]
        payload["_kind"] = r["kind"]
        payload["_extractor"] = r["extractor"]
        payload["_doc"] = r["doc_title"]
        out.append(payload)
    return out


# ---------- 各类样本 ----------


def samples_qa_appearance(
    knowledge: list[dict],
) -> list[dict]:
    """A1 人物出场问答：词典统计（确定性最高）。"""

    by_name: dict[str, dict] = {}
    for item in knowledge:
        if (
            item["_kind"] != "character"
            or item["_extractor"] != STATS_EXTRACTOR
        ):
            continue
        e = by_name.setdefault(
            item["name"],
            {"chapters": [], "mentions": 0},
        )
        e["chapters"].append(item["_doc"])
        e["mentions"] += item.get("mentions", 0)

    samples = []
    for name, e in by_name.items():
        chapters = "、".join(
            _pretty_doc(c) for c in e["chapters"]
        )
        samples.append(
            {
                "category": "qa",
                "instruction": (
                    f"《双月降临》前20章中，{name}"
                    "主要出现在哪些章节？"
                ),
                "input": "",
                "output": (
                    f"{name}出现于{chapters}，"
                    f"共提及{e['mentions']}次。"
                ),
                "source_doc": "aggregate",
                "knowledge_id": None,
            }
        )
    return samples


def samples_qa_relation(
    knowledge: list[dict],
) -> list[dict]:
    """A2 关系问答：关系边 + 原文引用作答。"""

    samples = []
    seen = set()
    for item in knowledge:
        if item["_kind"] != "relationship":
            continue
        a = item.get("from", "")
        b = item.get("to", "")
        key = (a, b, item.get("relation", ""))
        if not a or not b or key in seen:
            continue
        seen.add(key)
        quote = item.get("quote", "")
        samples.append(
            {
                "category": "qa",
                "instruction": (
                    f"《双月降临》里，{a}和{b}是什么关系？"
                ),
                "input": "",
                "output": (
                    f"{a}与{b}：{item.get('relation', '')}。"
                    f"原文依据：“{quote}”"
                ),
                "source_doc": item["_doc"],
                "knowledge_id": item["_id"],
            }
        )
    return samples


def samples_character(knowledge: list[dict]) -> list[dict]:
    """B 人物分析：LLM 抽取的身份/性格 + 原文引用。"""

    by_name: dict[str, dict] = {}
    for item in knowledge:
        if (
            item["_kind"] != "character"
            or item["_extractor"] == STATS_EXTRACTOR
        ):
            continue
        e = by_name.setdefault(
            item.get("name", ""),
            {"identities": [], "traits": [], "quote": ""},
        )
        if item.get("identity"):
            e["identities"].append(item["identity"])
        e["traits"].extend(item.get("traits") or [])
        if not e["quote"] and item.get("quote"):
            e["quote"] = item["quote"]

    samples = []
    for name, e in by_name.items():
        if not name:
            continue
        identity = "；".join(
            dict.fromkeys(e["identities"])
        ) or "身份未明"
        traits = (
            "、".join(dict.fromkeys(e["traits"])) or "谨慎务实"
        )
        quote = e["quote"] or ""
        output = (
            f"{name}：{identity}。性格特征：{traits}。"
        )
        if quote:
            output += f"原文依据：“{quote}”"
        samples.append(
            {
                "category": "character",
                "instruction": (
                    f"结合原文，分析《双月降临》中"
                    f"{name}的身份与性格。"
                ),
                "input": "",
                "output": output,
                "source_doc": "aggregate",
                "knowledge_id": None,
            }
        )
    return samples


def samples_plot(knowledge: list[dict]) -> list[dict]:
    """C 剧情理解：事件（起因/经过/结果）+引用。"""

    samples = []
    for item in knowledge:
        if item["_kind"] != "event":
            continue
        parts = [f"事件：{item.get('summary', '')}"]
        if item.get("cause"):
            parts.append(f"起因：{item['cause']}")
        if item.get("effect"):
            parts.append(f"结果：{item['effect']}")
        if item.get("participants"):
            parts.append(
                "涉及人物："
                + "、".join(item["participants"])
            )
        if item.get("quote"):
            parts.append(f"原文依据：“{item['quote']}”")
        samples.append(
            {
                "category": "plot",
                "instruction": (
                    f"《双月降临》{_pretty_doc(item['_doc'])}里发生了"
                    "什么事？起因和结果是什么？"
                ),
                "input": "",
                "output": "\n".join(parts),
                "source_doc": item["_doc"],
                "knowledge_id": item["_id"],
            }
        )
    return samples


def samples_world(knowledge: list[dict]) -> list[dict]:
    """D 世界观理解：规则 + 原文依据。"""

    samples = []
    for item in knowledge:
        if item["_kind"] != "world_rule":
            continue
        rule = item.get("rule", "")
        quote = item.get("quote", "")
        if not rule:
            continue
        output = rule
        if quote:
            output += f"\n原文依据：“{quote}”"
        samples.append(
            {
                "category": "world",
                "instruction": (
                    "《双月降临》的世界观中，"
                    "这条规则的原文依据是什么？"
                ),
                "input": rule,
                "output": output,
                "source_doc": item["_doc"],
                "knowledge_id": item["_id"],
            }
        )
    return samples


def samples_summary(knowledge: list[dict]) -> list[dict]:
    """E 章级总结：该章事件聚合为要点列表。"""

    by_chapter: dict[str, list[str]] = {}
    for item in knowledge:
        if item["_kind"] != "event":
            continue
        by_chapter.setdefault(item["_doc"], []).append(
            item.get("summary", "")
        )

    samples = []
    for doc, events in by_chapter.items():
        events = [e for e in events if e]
        if len(events) < 2:
            continue
        bullets = "\n".join(
            f"- {e}" for e in events[:8]
        )
        samples.append(
            {
                "category": "summary",
                "instruction": (
                    f"概括《双月降临》{_pretty_doc(doc)}的主要情节。"
                ),
                "input": "",
                "output": bullets,
                "source_doc": doc,
                "knowledge_id": None,
            }
        )
    return samples


def samples_writing(
    docs: list[dict],
    *,
    min_input: int = 500,
    max_output: int = 600,
) -> list[dict]:
    """F 写作任务：自监督续写切片。

    每章两片：普通续写（段落边界切）+ 对白锚点续写
    （从引号段起笔）。input=前文，output=原文后续段落，
    模型学的是作者自己的文风——全部来自可训练 rights 的文本。
    """

    samples = []
    for doc in docs:
        text = _read_doc(doc["raw_path"])
        paragraphs = [
            p.strip()
            for p in text.split("\n\n")
            if p.strip()
        ]
        if len(paragraphs) < 6:
            continue

        def make(
            cut: int, end: int, mode: str
        ) -> dict:
            inp = "\n\n".join(paragraphs[:cut])
            out = "\n\n".join(paragraphs[cut:end])
            return {
                "category": "writing",
                "instruction": (
                    "以下是一部末世小说《双月降临》的片段，"
                    "请以相同文风续写。"
                    + (
                        "下一段从人物对白开始。"
                        if mode == "dialogue"
                        else ""
                    )
                ),
                "input": inp,
                "output": out,
                "source_doc": doc["title"],
                "knowledge_id": None,
            }

        # 普通切片：累计到 min_input 字附近找段落边界
        acc = 0
        cut = 1
        for i, p in enumerate(paragraphs):
            acc += len(p)
            if acc >= min_input:
                cut = i + 1
                break
        out_acc = 0
        end = cut
        for i in range(cut, len(paragraphs)):
            out_acc += len(paragraphs[i])
            end = i + 1
            if out_acc >= max_output:
                break
        if cut < end <= len(paragraphs):
            samples.append(make(cut, end, "plain"))

        # 对白锚点：找中段第一个含引号的段
        mid = len(paragraphs) // 2
        d_cut = None
        for i in range(
            max(cut + 1, mid - 3), min(len(paragraphs), mid + 5)
        ):
            if "“" in paragraphs[i]:
                d_cut = i
                break
        if d_cut is not None:
            out_acc = 0
            d_end = d_cut
            for i in range(d_cut, len(paragraphs)):
                out_acc += len(paragraphs[i])
                d_end = i + 1
                if out_acc >= max_output:
                    break
            if d_end > d_cut:
                samples.append(
                    make(d_cut, d_end, "dialogue")
                )

    return samples


# ---------- 汇总 ----------


def build_dataset(
    conn: sqlite3.Connection, novel_title: str
) -> tuple[list[dict], dict]:

    docs = trainable_documents(conn, novel_title)
    knowledge = load_novel_knowledge(conn, novel_title)

    builders = {
        "qa_appearance": samples_qa_appearance,
        "qa_relation": samples_qa_relation,
        "character": samples_character,
        "plot": samples_plot,
        "world": samples_world,
        "summary": samples_summary,
    }

    samples = []
    for name, fn in builders.items():
        part = fn(knowledge)
        samples.extend(part)

    samples.extend(samples_writing(docs))

    stats = {
        "novel": novel_title,
        "generated_at": _now(),
        "trainable_docs": len(docs),
        "knowledge_used": len(knowledge),
        "total": len(samples),
        "by_category": {},
        "rights_note": (
            "仅从 authorized/user_provided/open_license "
            "文档生成；metadata_only 未进入数据集"
        ),
    }
    for s in samples:
        stats["by_category"][s["category"]] = (
            stats["by_category"].get(s["category"], 0) + 1
        )

    return samples, stats


def write_dataset(
    samples: list[dict],
    stats: dict,
    out_dir: Path | None = None,
) -> Path:

    if out_dir is None:
        out_dir = DATA_DIR / "datasets" / "v1"

    out_dir.mkdir(parents=True, exist_ok=True)

    samples_path = out_dir / "samples.jsonl"
    with samples_path.open(
        "w", encoding="utf-8"
    ) as f:
        for s in samples:
            f.write(
                json.dumps(
                    s, ensure_ascii=False
                )
                + "\n"
            )

    (out_dir / "metadata.json").write_text(
        json.dumps(
            stats, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )

    return samples_path
