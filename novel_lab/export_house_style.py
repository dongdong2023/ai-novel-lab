"""NovelPack 导出：Gemini 评审 P1-1 改进。

诊断：ai-novel-lab 学到的东西与 webnovel-house-style（写作
Skill）是割裂的——知识库躺在 SQLite 里，Skill 用不上。

NovelPack 是跨仓桥梁：把一本书学到的知识打包成 4 个 JSON，
house-style Skill 可直接消费：

  knowledge_graph.json  人物/关系/世界观结构化知识
  tropes_and_beats.json 套路与叙事节拍（narrative_dynamics）
  style_profile.json    文风画像（句长/对话比/节奏）
  manifest.json         包元信息 + 各文件 SHA256 校验

CLI: novel-lab export-pack --novel-title <书名>
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from novel_lab import db
from novel_lab.settings import DATA_DIR

PACK_VERSION = "1.0"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_knowledge_graph_pack(
    conn: sqlite3.Connection, novel_title: str
) -> dict:
    """人物/关系/世界观 → knowledge_graph.json 的内容。"""
    from novel_lab.knowledge_graph import (
        character_index,
        load_knowledge,
        relationship_edges,
    )

    knowledge = load_knowledge(conn, novel_title)
    characters = character_index(knowledge)
    edges = relationship_edges(knowledge, characters)

    # character_index 的 chapters/sources 是 set，JSON 序列化前
    # 转成排序 list；identities/traits 去重保序（P0-1.2）
    serializable_chars = {
        name: {
            **info,
            "chapters": sorted(info["chapters"]),
            "sources": sorted(info["sources"]),
            "identities": list(
                dict.fromkeys(info["identities"])
            ),
            "traits": list(dict.fromkeys(info["traits"])),
        }
        for name, info in characters.items()
    }

    world_rules = [
        {
            "rule": k.get("rule", ""),
            "quote": k.get("quote", ""),
            "doc": k.get("_doc", ""),
        }
        for k in knowledge
        if k.get("_kind") == "world_rule"
    ]
    events = [
        {
            "summary": k.get("summary", ""),
            "cause": k.get("cause", ""),
            "effect": k.get("effect", ""),
            "participants": k.get("participants", []),
            "doc": k.get("_doc", ""),
        }
        for k in knowledge
        if k.get("_kind") == "event"
    ]

    return {
        "novel": novel_title,
        "characters": serializable_chars,
        "relationships": edges,
        "world_rules": world_rules,
        "events": events,
        "counts": {
            "characters": len(characters),
            "relationships": len(edges),
            "world_rules": len(world_rules),
            "events": len(events),
        },
    }


def build_tropes_and_beats_pack(
    conn: sqlite3.Connection, novel_title: str
) -> dict:
    """套路与叙事节拍 → tropes_and_beats.json 的内容。

    叙事节拍来自 knowledge 表里的 narrative_dynamics 条目
    （extract_chapter_stats 已挂载，P0-2）。
    """
    rows = conn.execute(
        "SELECT k.payload, d.title AS doc_title"
        " FROM knowledge k"
        " JOIN documents d ON d.id = k.document_id"
        " JOIN novels n ON n.id = k.novel_rowid"
        " WHERE n.title = ? AND k.kind = ?",
        (novel_title, "narrative_dynamics"),
    ).fetchall()
    # 按章节号数值排序，避免 ch10 排到 ch2 前面
    from novel_lab.knowledge_graph import _chapter_no

    rows = sorted(
        rows, key=lambda r: _chapter_no(r["doc_title"])
    )

    chapters = []
    hook_counter: dict[str, int] = {}
    cliff_counter: dict[str, int] = {}
    for r in rows:
        payload = json.loads(r["payload"])
        chapters.append(payload)
        for h in payload.get("hooks", []):
            bt = h.get("beat_type", "unknown")
            hook_counter[bt] = hook_counter.get(bt, 0) + 1
        ch = payload.get("cliffhanger")
        if ch:
            bt = ch.get("beat_type", "unknown")
            cliff_counter[bt] = cliff_counter.get(bt, 0) + 1

    total = len(chapters) or 1
    return {
        "novel": novel_title,
        "chapters": chapters,
        "tropes": {
            "hook_distribution": {
                k: round(v / total, 3)
                for k, v in sorted(
                    hook_counter.items(), key=lambda x: -x[1]
                )
            },
            "cliffhanger_distribution": {
                k: round(v / total, 3)
                for k, v in sorted(
                    cliff_counter.items(), key=lambda x: -x[1]
                )
            },
            "cliffhanger_rate": round(
                sum(cliff_counter.values()) / total, 3
            ),
        },
    }


def build_style_profile_pack(
    conn: sqlite3.Connection, novel_title: str
) -> dict:
    """文风画像 → style_profile.json 的内容。

    聚合各章 style_fact 条目：句长/段长/对话比取均值，
    供 house-style Skill 做文风对齐。
    """
    rows = conn.execute(
        "SELECT k.payload"
        " FROM knowledge k"
        " JOIN novels n ON n.id = k.novel_rowid"
        " WHERE n.title = ? AND k.kind = ?",
        (novel_title, "style_fact"),
    ).fetchall()

    facts = [json.loads(r["payload"]) for r in rows]
    if not facts:
        return {
            "novel": novel_title,
            "chapters_covered": 0,
            "style": {},
            # 跨仓兼容字段（P1-2.6）：无数据时给 0，
            # 避免消费方读到缺失键
            "dialogue_ratio": 0,
            "sentence_length_mean_of_files": 0,
            "paragraph_length_median_of_files": 0,
        }

    def _mean(key: str) -> float:
        vals = [
            f.get(key, 0)
            for f in facts
            if isinstance(f.get(key), (int, float))
        ]
        return round(sum(vals) / len(vals), 2) if vals else 0

    def _median(key: str) -> float:
        import statistics

        vals = sorted(
            f.get(key, 0)
            for f in facts
            if isinstance(f.get(key), (int, float))
        )
        return (
            round(statistics.median(vals), 2) if vals else 0
        )

    style = {
        "sentence_length_mean": _mean(
            "sentence_length_mean"
        ),
        "paragraph_length_mean": _mean(
            "paragraph_length_mean"
        ),
        "dialogue_ratio_mean": _mean("dialogue_ratio"),
        "chinese_chars_total": sum(
            f.get("chinese_chars", 0) for f in facts
        ),
    }

    return {
        "novel": novel_title,
        "chapters_covered": len(facts),
        "style": style,
        # 跨仓兼容字段（P1-2.6）：webnovel-house-style 的
        # compare_style_profile.py 读顶层字段，与本仓的
        # sentence_length_mean_of_files 等键名对齐。
        # paragraph 中位数口径：各章 paragraph_length_mean
        # 的中位数（style_fact 只存了章级均值，这是最接近
        # 的等价口径）。
        "dialogue_ratio": style["dialogue_ratio_mean"],
        "sentence_length_mean_of_files": style[
            "sentence_length_mean"
        ],
        "paragraph_length_median_of_files": _median(
            "paragraph_length_mean"
        ),
    }


def export_pack(novel_title: str) -> dict:
    """导出 NovelPack 到 data/packs/<书名>/，返回 manifest 信息。"""
    conn = db.connect()
    try:
        kg = build_knowledge_graph_pack(conn, novel_title)
        tb = build_tropes_and_beats_pack(conn, novel_title)
        sp = build_style_profile_pack(conn, novel_title)
    finally:
        conn.close()

    out_dir = DATA_DIR / "packs" / novel_title
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "knowledge_graph.json": kg,
        "tropes_and_beats.json": tb,
        "style_profile.json": sp,
    }
    checksums = {}
    for name, payload in files.items():
        path = out_dir / name
        path.write_text(
            json.dumps(
                payload, ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
        checksums[name] = _sha256_file(path)

    manifest = {
        "pack": "NovelPack",
        "pack_version": PACK_VERSION,
        "novel": novel_title,
        "exported_at": _now(),
        "files": {
            name: {
                "sha256": sha,
                "size": (out_dir / name).stat().st_size,
            }
            for name, sha in checksums.items()
        },
        "consumer": "webnovel-house-style skill",
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "out_dir": str(out_dir),
        "files": list(files) + ["manifest.json"],
        "manifest": manifest,
    }


def cmd_export_pack(novel_title: str) -> None:
    info = export_pack(novel_title)
    print(f"[PASS] NovelPack: {info['out_dir']}")
    for name in info["files"]:
        sha = info["manifest"]["files"].get(name, {}).get(
            "sha256", ""
        )[:12]
        print(f"  {name}  sha256:{sha}")
