from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from novel_lab.understanding.prompts import (
    prompt_version,
    render,
)

MAX_CONTENT_CHARS = 1500

VALID_PREFILTER = {"PASS", "BLOCK", "UNKNOWN"}

_KIND_BY_SOURCE = {
    "book_profile": "book_profile",
    "user_text": "chapter_text",
    "review": "review",
    "wiki": "wiki",
    "interview": "interview",
    "discussion": "discussion",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_json(text: str) -> dict | None:
    """从模型输出里抠出第一个 JSON 对象。
    兼容 markdown 代码块、前后闲话、Qwen3 思考块、不完整截断。"""
    if not text:
        return None
    # 剥离 Qwen3 的 <think>…</think>（未闭合时整段丢弃）
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    elif "<think>" in text:
        text = text.split("<think>", 1)[0]
    cleaned = re.sub(
        r"```(?:json)?", "", text
    )
    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(
                        cleaned[start : i + 1]
                    )
                    return (
                        value
                        if isinstance(value, dict)
                        else None
                    )
                except json.JSONDecodeError:
                    return None
    return None


def read_document_content(
    doc_row, max_chars: int = MAX_CONTENT_CHARS
) -> str:
    raw_path = doc_row["raw_path"] or ""
    text = ""
    if raw_path and Path(raw_path).is_file():
        try:
            text = Path(raw_path).read_text(
                encoding="utf-8"
            )
        except UnicodeDecodeError:
            text = Path(raw_path).read_text(
                encoding="gbk", errors="replace"
            )
    return text[:max_chars]


class UnderstandingPipeline:
    """文档级理解：预筛 → 结构化 → 落 doc_structures。

    mock=True 用确定性假分析跑通全链路（零 LLM 花费，结果标
    is_mock=1）；真跑需要 llm.enabled=true 且配好 key。
    改提示词 → prompt_versions 变 → 已分析文档自动视为待重跑。
    """

    def __init__(self, conn: sqlite3.Connection, client=None):
        self.conn = conn
        self.client = client
        self.versions = {
            "prefilter": prompt_version("prefilter"),
            "structure": prompt_version("structure"),
        }
        self.versions_key = json.dumps(
            self.versions, ensure_ascii=False
        )

    # ---------- 单文档处理 ----------

    def process(
        self,
        doc: dict,
        content: str,
        *,
        mock: bool = False,
    ) -> dict:

        if mock:
            prefilter, reason = self._mock_prefilter(
                doc, content
            )
        else:
            prefilter, reason = self._prefilter(
                doc, content
            )

        structure = None
        if prefilter == "PASS":
            if mock:
                structure = self._mock_structure(
                    doc, content
                )
            else:
                structure = self._structure(doc, content)

        return {
            "document_id": doc["id"],
            "prefilter": prefilter,
            "prefilter_reason": reason,
            "structure": structure,
        }

    def _prefilter(self, doc, content):
        user = render(
            "prefilter",
            title=doc["title"],
            source_type=doc["source_type"] or "",
            content=content,
        )
        raw = self.client.chat(
            PREFILTER_SYSTEM, user, purpose="prefilter"
        )
        data = extract_json(raw) or {}
        status = str(data.get("status", "")).upper()
        if status not in VALID_PREFILTER:
            return "UNKNOWN", "模型输出status无效"
        return (
            status,
            str(data.get("reason", ""))[:200],
        )

    def _structure(self, doc, content):
        user = render(
            "structure",
            title=doc["title"],
            source_type=doc["source_type"] or "",
            content=content,
        )
        raw = self.client.chat(
            STRUCTURE_SYSTEM, user, purpose="structure"
        )
        data = extract_json(raw)
        if data is None:
            return {
                "parse_error": True,
                "raw": str(raw)[:300],
            }
        return data

    # ---------- mock 干跑 ----------

    def _mock_prefilter(self, doc, content):
        if not content.strip():
            return "UNKNOWN", "mock: 内容为空"
        if (doc["source_type"] or "other") in (
            _KIND_BY_SOURCE
        ):
            return "PASS", "mock: 已知资料类型"
        return "BLOCK", "mock: 未知类型"

    def _mock_structure(self, doc, content):
        kind = _KIND_BY_SOURCE.get(
            doc["source_type"] or "", "other"
        )
        aspects = (
            ["题材", "热度数据", "卖点标签"]
            if kind == "book_profile"
            else ["章节信息"]
        )
        return {
            "doc_kind": kind,
            "about_novel": "",
            "aspects": aspects,
            "facts": [
                {
                    "subject": doc["title"],
                    "predicate": "is_document",
                    "object": kind,
                    "source_quote": content[:10],
                }
            ],
            "notes": "mock 结构化（无LLM）",
        }

    # ---------- 批处理 ----------

    def pending_documents(self, limit: int | None = None):
        # mock 行是干跑占位，不算已分析：真跑会自动接手重做
        rows = self.conn.execute(
            "SELECT d.* FROM documents d WHERE NOT EXISTS ("
            " SELECT 1 FROM doc_structures s "
            " WHERE s.document_id=d.id "
            " AND s.prompt_versions=? AND s.is_mock=0)"
            " ORDER BY d.id",
            (self.versions_key,),
        ).fetchall()
        rows = [dict(r) for r in rows]
        return rows[:limit] if limit else rows

    def run(
        self, limit: int | None = None, *, mock: bool = False
    ) -> dict:

        stats = {
            "processed": 0,
            "pass": 0,
            "block": 0,
            "unknown": 0,
            "parse_errors": 0,
        }

        for doc in self.pending_documents(limit):
            content = read_document_content(doc)
            result = self.process(doc, content, mock=mock)

            self.conn.execute(
                "INSERT OR REPLACE INTO doc_structures "
                "(document_id, prefilter, prefilter_reason,"
                " structure, model, prompt_versions, is_mock,"
                " created_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    result["document_id"],
                    result["prefilter"],
                    result["prefilter_reason"],
                    (
                        json.dumps(
                            result["structure"],
                            ensure_ascii=False,
                        )
                        if result["structure"] is not None
                        else None
                    ),
                    "mock" if mock else "live",
                    self.versions_key,
                    1 if mock else 0,
                    _now(),
                ),
            )
            self.conn.commit()

            stats["processed"] += 1
            stats[result["prefilter"].lower()] += 1
            structure = result.get("structure")
            if (
                isinstance(structure, dict)
                and structure.get("parse_error")
            ):
                stats["parse_errors"] += 1

        return stats


PREFILTER_SYSTEM = (
    "你是网文研究系统的资料预筛器，只输出 JSON。"
)

STRUCTURE_SYSTEM = (
    "你是网文研究系统的资料结构化器，只输出 JSON。"
    "事实必须来自原文，抽不出就给空数组，禁止编造。"
)
