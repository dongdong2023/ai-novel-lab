from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from novel_lab.style_stats import analyze_style
from novel_lab.understanding.pipeline import extract_json
from novel_lab.understanding.prompts import (
    prompt_version,
    render,
)

# 确定性抽取（无需 LLM）版本：改了算法就 bump 这个串
STATS_EXTRACTOR = "deterministic-v1"

CHARACTER_DICT_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "character_dicts"
)

EXTRACT_SYSTEM = (
    "你是网文研究系统的章节知识抽取器，只输出 JSON。"
    "每条知识必须带原文 quote，抽不出给空数组，禁止编造。"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_character_dict(novel_title: str) -> list[str]:
    path = CHARACTER_DICT_DIR / f"{novel_title}.txt"
    if not path.is_file():
        return []
    names = [
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return names


def passed_documents(
    conn: sqlite3.Connection,
) -> list[dict]:
    """P4 预筛通过（最新一次判定）的文档。"""

    rows = conn.execute(
        """
        SELECT d.*, s.prefilter, n.title AS novel_title
        FROM documents d
        JOIN doc_structures s ON s.document_id = d.id
        LEFT JOIN novels n ON n.id = d.novel_rowid
        WHERE s.id = (
            SELECT s2.id FROM doc_structures s2
            WHERE s2.document_id = d.id
            ORDER BY s2.id DESC LIMIT 1
        )
        AND s.prefilter = 'PASS'
        ORDER BY d.id
        """
    ).fetchall()
    return [dict(r) for r in rows]


def write_knowledge(
    conn: sqlite3.Connection,
    *,
    novel_rowid: int,
    document_id: int,
    items: list[tuple[str, dict, float]],
    extractor: str,
    version: str,
) -> int:
    """幂等写入：同 (document, extractor, version) 先删后插。"""

    conn.execute(
        "DELETE FROM knowledge WHERE document_id=? "
        "AND extractor=? AND prompt_version=?",
        (document_id, extractor, version),
    )

    for kind, payload, confidence in items:
        conn.execute(
            "INSERT INTO knowledge (novel_rowid,"
            " document_id, kind, payload, confidence,"
            " extractor, prompt_version, created_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (
                novel_rowid,
                document_id,
                kind,
                json.dumps(
                    payload, ensure_ascii=False
                ),
                confidence,
                extractor,
                version,
                _now(),
            ),
        )

    conn.commit()
    return len(items)


# ---------- 确定性抽取器 ----------


def extract_profile_facts(
    doc: dict, content: str
) -> list[tuple[str, dict, float]]:
    """book_profile JSON → 市场/题材事实。结构化数据直接映射，零幻觉。"""

    try:
        profile = json.loads(content)
    except json.JSONDecodeError:
        return []

    items: list[tuple[str, dict, float]] = []

    title = profile.get("title") or doc["title"]

    genre = {
        "subject": title,
        "genre": profile.get("category") or "",
        "author": profile.get("author") or "",
        "platform": profile.get("platform") or "",
        "tags": profile.get("tags") or [],
    }
    items.append(("genre_fact", genre, 1.0))

    for field, label in (
        ("read_count", "在读数"),
        ("word_count", "字数"),
        ("month_ticket", "月票"),
        ("collect", "收藏"),
        ("total_recommend", "总推荐"),
        ("chapter_count", "章节数"),
    ):
        value = profile.get(field)
        if value is not None:
            items.append(
                (
                    "market_fact",
                    {
                        "subject": title,
                        "metric": label,
                        "value": value,
                    },
                    1.0,
                )
            )

    return items


def extract_chapter_stats(
    doc: dict,
    content: str,
    character_dict: list[str],
) -> list[tuple[str, dict, float]]:
    """章节文本 → 风格统计 + 人物出场（词典匹配）。确定性，零幻觉。"""

    items: list[tuple[str, dict, float]] = []

    style = analyze_style(content)
    items.append(
        (
            "style_fact",
            {
                "subject": doc["title"],
                **style,
            },
            1.0,
        )
    )

    for name in character_dict:
        mentions = content.count(name)
        if mentions > 0:
            items.append(
                (
                    "character",
                    {
                        "name": name,
                        "chapter": doc["title"],
                        "mentions": mentions,
                    },
                    1.0,
                )
            )

    return items


# ---------- LLM 抽取器（需 key） ----------


def extract_chapter_llm(
    doc: dict, content: str, client
) -> list[tuple[str, dict, float]]:
    version = prompt_version("extract_chapter")
    user = render(
        "extract_chapter",
        title=doc["title"],
        content=content,
    )
    from novel_lab.llm.client import LLMError

    raw = None
    for attempt in (1, 2):
        try:
            raw = client.chat(
                EXTRACT_SYSTEM, user, purpose="extract_chapter"
            )
            break
        except LLMError:
            if attempt == 2:
                return []
    if raw is None:
        return []
    data = extract_json(raw)
    if data is None:
        return []

    items: list[tuple[str, dict, float]] = []
    kind_map = {
        "characters": "character",
        "relationships": "relationship",
        "events": "event",
        "world_rules": "world_rule",
        "style_notes": "style_note",
    }

    for key, kind in kind_map.items():
        entries = data.get(key) or []
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            quote = (entry.get("quote") or "").strip()
            if not quote:
                # 无原文依据的知识一律丢弃（防幻觉硬门禁）
                continue
            if quote not in content:
                # 引用必须逐字来自原文：拦住"拼接转述"
                # （抽检发现 27B 偶尔把两处原文缝合成一句）
                continue
            items.append((kind, entry, 0.7))

    return items


# ---------- 批处理 ----------


def run_extract(
    conn: sqlite3.Connection,
    client=None,
    *,
    use_llm: bool = False,
    limit: int | None = None,
) -> dict:

    stats = {
        "docs": 0,
        "knowledge": 0,
        "skipped_no_novel": 0,
        "llm_items": 0,
    }

    docs = passed_documents(conn)
    if limit:
        docs = docs[:limit]

    for doc in docs:

        if doc["novel_rowid"] is None:
            stats["skipped_no_novel"] += 1
            continue

        raw_path = doc["raw_path"] or ""
        if not raw_path or not Path(raw_path).is_file():
            continue
        try:
            content = Path(raw_path).read_text(
                encoding="utf-8"
            )
        except UnicodeDecodeError:
            content = Path(raw_path).read_text(
                encoding="gbk", errors="replace"
            )
        # 章节文本给 LLM 的截断上限放大一些
        content_llm = content[:3000]

        stats["docs"] += 1

        if doc["source_type"] == "book_profile":
            items = extract_profile_facts(doc, content)
            written = write_knowledge(
                conn,
                novel_rowid=doc["novel_rowid"],
                document_id=doc["id"],
                items=items,
                extractor=STATS_EXTRACTOR,
                version=STATS_EXTRACTOR,
            )
            stats["knowledge"] += written

        else:
            character_dict = load_character_dict(
                doc["novel_title"] or ""
            )
            items = extract_chapter_stats(
                doc, content, character_dict
            )

            # 确定性知识先落库：LLM 失败也不丢这部分
            stats["knowledge"] += write_knowledge(
                conn,
                novel_rowid=doc["novel_rowid"],
                document_id=doc["id"],
                items=items,
                extractor=STATS_EXTRACTOR,
                version=STATS_EXTRACTOR,
            )

            llm_items = []
            if use_llm and client is not None:
                try:
                    llm_items = extract_chapter_llm(
                        doc, content_llm, client
                    )
                    stats["llm_items"] += len(llm_items)
                except Exception as exc:
                    # 单章失败不拖垮整批，记录后继续
                    stats["llm_failed"] = (
                        stats.get("llm_failed", 0) + 1
                    )
                    print(
                        f"[WARN] {doc['title']} LLM抽取失败: {exc}"
                    )

            if llm_items:
                stats["knowledge"] += write_knowledge(
                    conn,
                    novel_rowid=doc["novel_rowid"],
                    document_id=doc["id"],
                    items=llm_items,
                    extractor="llm-v1",
                    version=prompt_version("extract_chapter"),
                )

    return stats


def knowledge_summary(
    conn: sqlite3.Connection,
    novel_title: str | None = None,
) -> dict:

    query = (
        "SELECT k.kind, COUNT(*) AS n FROM knowledge k "
        "JOIN novels n ON n.id = k.novel_rowid"
    )
    params: tuple = ()
    if novel_title:
        query += " WHERE n.title = ?"
        params = (novel_title,)
    query += " GROUP BY k.kind"

    rows = conn.execute(query, params).fetchall()
    return {r["kind"]: r["n"] for r in rows}


def character_totals(
    conn: sqlite3.Connection, novel_title: str
) -> list[tuple[str, int, int]]:
    """(人物名, 出现章数, 总提及次数)，按提及排序。"""

    rows = conn.execute(
        """
        SELECT json_extract(k.payload, '$.name') AS name,
               COUNT(DISTINCT k.document_id) AS chapters,
               SUM(json_extract(k.payload, '$.mentions')) AS total
        FROM knowledge k
        JOIN novels n ON n.id = k.novel_rowid
        WHERE n.title = ? AND k.kind = 'character'
        GROUP BY name
        ORDER BY total DESC
        """,
        (novel_title,),
    ).fetchall()
    return [
        (r["name"], r["chapters"], r["total"])
        for r in rows
    ]
