from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# Rights Gate（架构文档第 5 节白名单）
# metadata_only/platform_free：可进知识库（P5 起抽事实，短引用+溯源），
#                             不可进训练集
# authorized/user_provided/open_license：可进训练集
ALLOWED_RIGHTS = {
    "metadata_only",
    "platform_free",
    "authorized",
    "user_provided",
    "open_license",
}
TRAINABLE_RIGHTS = {
    "authorized",
    "user_provided",
    "open_license",
}

_TITLE_NOISE_RE = re.compile(
    r"[\s，。！？?!：:、·\-—|《》「」“”\"']+"
)


def normalize_title(title: str) -> str:
    return _TITLE_NOISE_RE.sub("", title or "")


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def register_local_novel(
    conn: sqlite3.Connection, title: str
) -> int:
    """把用户自己的书登记进 novels（platform=local），
    不参与热度候选（没有 observations 就不会进 candidate_rows）。"""

    row = conn.execute(
        "SELECT id FROM novels WHERE platform='local' "
        "AND book_id=?",
        (title,),
    ).fetchone()

    if row:
        return row["id"]

    cursor = conn.execute(
        "INSERT INTO novels (platform, book_id, title,"
        " first_seen_at, last_seen_at) VALUES (?,?,?,?,?)",
        ("local", title, title, _now(), _now()),
    )
    conn.commit()
    return cursor.lastrowid


def novel_rowid(
    conn: sqlite3.Connection,
    platform: str,
    book_id: str,
) -> int | None:
    row = conn.execute(
        "SELECT id FROM novels WHERE platform=? AND "
        "book_id=?",
        (platform, book_id),
    ).fetchone()
    return row["id"] if row else None


def ingest_document(
    conn: sqlite3.Connection,
    *,
    novel_rowid: int | None,
    source_url: str,
    source_type: str,
    rights: str,
    title: str,
    published_at: str = "",
    content_hash: str = "",
    raw_path: str = "",
) -> tuple[int | None, str]:
    """三级判重入库：URL → 内容哈希 → 同书标题归一。

    返回 (doc_id, status)，status ∈
    {created, duplicate_url, duplicate_content, duplicate_title}。
    rights 不在白名单直接拒绝。
    """

    if rights not in ALLOWED_RIGHTS:
        raise ValueError(
            f"未知 rights: {rights}，"
            f"允许值: {sorted(ALLOWED_RIGHTS)}"
        )

    row = conn.execute(
        "SELECT id FROM documents WHERE source_url=?",
        (source_url,),
    ).fetchone()
    if row:
        return row["id"], "duplicate_url"

    if content_hash:
        row = conn.execute(
            "SELECT id FROM documents WHERE "
            "content_hash=? AND novel_rowid IS ?",
            (content_hash, novel_rowid),
        ).fetchone()
        if row:
            return row["id"], "duplicate_content"

    if novel_rowid is not None and title:
        norm = normalize_title(title)
        for existing in conn.execute(
            "SELECT id, title FROM documents "
            "WHERE novel_rowid IS ?",
            (novel_rowid,),
        ):
            if (
                normalize_title(existing["title"] or "")
                == norm
            ):
                return (
                    existing["id"],
                    "duplicate_title",
                )

    cursor = conn.execute(
        "INSERT INTO documents (novel_rowid, source_url,"
        " source_type, rights, title, published_at,"
        " crawl_time, content_hash, raw_path)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (
            novel_rowid,
            source_url,
            source_type,
            rights,
            title,
            published_at,
            _now(),
            content_hash,
            raw_path,
        ),
    )
    conn.commit()
    return cursor.lastrowid, "created"


def ingest_profile(
    conn: sqlite3.Connection,
    profile: dict,
    store_dir: Path,
) -> tuple[int | None, str]:
    """书页元数据资料落盘 + 入库（metadata_only）。"""

    store_dir.mkdir(parents=True, exist_ok=True)
    path = (
        store_dir
        / profile["platform"]
        / f"{profile['book_id']}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        profile, ensure_ascii=False, indent=2
    )
    path.write_text(payload, encoding="utf-8")

    return ingest_document(
        conn,
        novel_rowid=novel_rowid(
            conn, profile["platform"], profile["book_id"]
        ),
        source_url=profile["source_url"],
        source_type="book_profile",
        rights="metadata_only",
        title=f"{profile['title']}·书页资料",
        published_at=profile.get("last_publish_at", ""),
        content_hash=sha256_text(payload),
        raw_path=str(path),
    )


def ingest_directory(
    conn: sqlite3.Connection,
    directory: Path,
    *,
    rights: str = "user_provided",
    source_type: str = "user_text",
    novel_title: str = "",
) -> dict:
    """把一个目录的 txt/md 登记为文档（默认用户自有语料）。

    只登记元数据与哈希，不复制文件；raw_path 指向原文件。
    """

    if rights not in ALLOWED_RIGHTS:
        raise ValueError(f"未知 rights: {rights}")

    target_novel = (
        register_local_novel(conn, novel_title)
        if novel_title
        else None
    )

    stats = {"created": 0, "duplicates": 0}

    for path in sorted(
        directory.rglob("*")
    ):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".md"}:
            continue

        text = _read_best(path)

        _, status = ingest_document(
            conn,
            novel_rowid=target_novel,
            source_url=path.resolve().as_uri(),
            source_type=source_type,
            rights=rights,
            title=path.stem,
            content_hash=sha256_text(text),
            raw_path=str(path),
        )

        if status == "created":
            stats["created"] += 1
        else:
            stats["duplicates"] += 1

    return stats


def _read_best(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(
            encoding="gbk", errors="replace"
        )


def training_ready(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM documents "
        "WHERE rights IN "
        "('authorized','user_provided','open_license')"
    ).fetchone()
    return row["n"]


def document_stats(conn: sqlite3.Connection) -> dict:
    rows = conn.execute(
        "SELECT rights, source_type, COUNT(*) AS n "
        "FROM documents GROUP BY rights, source_type"
    ).fetchall()
    return {
        f"{r['rights']}/{r['source_type']}": r["n"]
        for r in rows
    }
