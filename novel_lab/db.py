from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from novel_lab.models import NovelRecord
from novel_lab.settings import db_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS novels (
    id INTEGER PRIMARY KEY,
    platform TEXT NOT NULL,
    book_id  TEXT NOT NULL,
    title    TEXT NOT NULL,
    author   TEXT DEFAULT '',
    url      TEXT DEFAULT '',
    first_seen_at TEXT,
    last_seen_at  TEXT,
    UNIQUE(platform, book_id)
);

CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY,
    novel_rowid INTEGER NOT NULL REFERENCES novels(id),
    board TEXT NOT NULL,
    gender TEXT, rank_type TEXT, category TEXT,
    rank INTEGER, read_count REAL,
    word_count INTEGER, chapter_count INTEGER,
    creation_status INTEGER,
    updated_at TEXT, observed_at TEXT NOT NULL,
    UNIQUE(novel_rowid, board, observed_at)
);

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    novel_rowid INTEGER REFERENCES novels(id),
    source_url TEXT NOT NULL UNIQUE,
    source_type TEXT,
    rights TEXT NOT NULL DEFAULT 'metadata_only',
    title TEXT, published_at TEXT, crawl_time TEXT,
    content_hash TEXT, raw_path TEXT, clean_path TEXT
);

CREATE TABLE IF NOT EXISTS knowledge (
    id INTEGER PRIMARY KEY,
    novel_rowid INTEGER NOT NULL REFERENCES novels(id),
    document_id INTEGER NOT NULL REFERENCES documents(id),
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    confidence REAL, extractor TEXT, prompt_version TEXT,
    human_verified INTEGER DEFAULT 0,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS doc_structures (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    prefilter TEXT NOT NULL,
    prefilter_reason TEXT DEFAULT '',
    structure TEXT,
    model TEXT,
    prompt_versions TEXT,
    is_mock INTEGER DEFAULT 0,
    created_at TEXT,
    UNIQUE(document_id, prompt_versions)
);

CREATE TABLE IF NOT EXISTS llm_receipts (
    id INTEGER PRIMARY KEY,
    purpose TEXT, model TEXT, prompt_hash TEXT,
    status TEXT, response TEXT, created_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_obs_novel
    ON observations(novel_rowid);
CREATE INDEX IF NOT EXISTS idx_obs_novel_board
    ON observations(novel_rowid, board);
CREATE INDEX IF NOT EXISTS idx_receipt_lookup
    ON llm_receipts(purpose, prompt_hash, model, status);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    if path is None:
        path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def upsert_novel(
    conn: sqlite3.Connection,
    record: NovelRecord,
) -> int:
    """一书一行；重复 discover 幂等，只刷新 last_seen_at。"""

    now = _now()

    row = conn.execute(
        "SELECT id FROM novels WHERE platform=? AND book_id=?",
        (record.platform, record.book_id),
    ).fetchone()

    if row:
        conn.execute(
            "UPDATE novels SET last_seen_at=?, author=?, url=? "
            "WHERE id=?",
            (
                now,
                record.author or "",
                record.url,
                row["id"],
            ),
        )
        return row["id"]

    cursor = conn.execute(
        "INSERT INTO novels (platform, book_id, title, author, url,"
        " first_seen_at, last_seen_at) VALUES (?,?,?,?,?,?,?)",
        (
            record.platform,
            record.book_id,
            record.title,
            record.author or "",
            record.url,
            now,
            now,
        ),
    )
    return cursor.lastrowid


def insert_observation(
    conn: sqlite3.Connection,
    novel_rowid: int,
    record: NovelRecord,
    observed_at: str | None = None,
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO observations "
        "(novel_rowid, board, gender, rank_type, category, rank,"
        " read_count, word_count, chapter_count, creation_status,"
        " updated_at, observed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            novel_rowid,
            record.board,
            record.gender,
            record.rank_type,
            record.category_name,
            record.rank,
            record.read_count,
            record.word_count,
            record.chapter_count,
            record.creation_status,
            record.updated_at,
            observed_at or record.discovered_at,
        ),
    )


def save_discovery(
    conn: sqlite3.Connection,
    records: list[NovelRecord],
) -> int:
    """落一轮 discover：upsert 书 + 插观察。返回涉及的书数。"""

    seen: set[int] = set()

    for record in records:
        if not record.book_id:
            continue
        rowid = upsert_novel(conn, record)
        insert_observation(conn, rowid, record)
        seen.add(rowid)

    conn.commit()
    return len(seen)


def latest_growth(
    conn: sqlite3.Connection,
) -> dict[int, float]:
    """每本书按 (板块) 取最近两期观察，算在读相对增长，书内取最大值。

    只有 ≥2 期观察的板块才有增长率；单期数据返回空。
    """

    rows = conn.execute(
        "SELECT novel_rowid, board, read_count, observed_at "
        "FROM observations "
        "WHERE read_count IS NOT NULL AND read_count > 0 "
        "ORDER BY novel_rowid, board, observed_at"
    ).fetchall()

    series: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        series.setdefault(
            (row["novel_rowid"], row["board"]), []
        ).append(row["read_count"])

    growth: dict[int, float] = {}

    for (novel_rowid, _board), counts in series.items():
        if len(counts) < 2:
            continue
        prev, cur = counts[-2], counts[-1]
        if prev <= 0:
            continue
        rate = (cur - prev) / prev
        if rate > growth.get(novel_rowid, -1e9):
            growth[novel_rowid] = rate

    return growth


def candidate_rows(conn: sqlite3.Connection) -> list[dict]:
    """跨来源聚合出候选池：每本书的最高在读/最好排名/板块数/期数。"""

    rows = conn.execute(
        """
        SELECT n.id AS novel_rowid, n.title, n.platform, n.book_id,
               MAX(o.read_count) AS max_read,
               MIN(o.rank) AS best_rank,
               COUNT(DISTINCT o.board) AS board_count,
               COUNT(o.id) AS observation_count,
               MAX(o.observed_at) AS last_observed_at,
               MAX(o.word_count) AS word_count,
               GROUP_CONCAT(DISTINCT o.board) AS board,
               GROUP_CONCAT(DISTINCT o.rank_type) AS rank_type
        FROM novels n
        JOIN observations o ON o.novel_rowid = n.id
        GROUP BY n.id
        ORDER BY max_read DESC
        """
    ).fetchall()

    return [dict(row) for row in rows]


def top_novels(
    conn: sqlite3.Connection,
    scored: list[dict],
    limit: int = 10,
) -> list[dict]:
    """按 popularity_score 取 TopN，附上该书所有板块观察。"""

    top = scored[:limit]
    rowids = [row["novel_rowid"] for row in top]

    if not rowids:
        return []

    placeholders = ",".join("?" * len(rowids))

    boards = conn.execute(
        f"""
        SELECT novel_rowid, board, gender, rank_type, category,
               rank, read_count, observed_at
        FROM observations
        WHERE novel_rowid IN ({placeholders})
        ORDER BY novel_rowid, rank
        """,
        rowids,
    ).fetchall()

    by_novel: dict[int, list[dict]] = {}
    for b in boards:
        by_novel.setdefault(
            b["novel_rowid"], []
        ).append(dict(b))

    for row in top:
        row["boards"] = by_novel.get(row["novel_rowid"], [])

    return top
