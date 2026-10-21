from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from novel_lab.ingest import training_ready
from novel_lab.settings import DATA_DIR


def _count(conn, sql, params=()):
    return conn.execute(sql, params).fetchone()[0]


def pipeline_status(conn: sqlite3.Connection) -> str:
    """P10：一屏验明整条流水线的产物链。"""

    lines = ["《AI 小说自动学习系统》流水线状态", ""]

    # P1/P2 发现层
    novels = _count(conn, "SELECT COUNT(*) FROM novels")
    observations = _count(
        conn, "SELECT COUNT(*) FROM observations"
    )
    platforms = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT platform FROM novels"
        )
    ]
    lines.append(
        f"[P1-P2 发现] {novels} 本书 / "
        f"{observations} 次榜单观察 / 平台: {'、'.join(platforms)}"
    )

    # P3 资料层
    docs_by_rights = {
        f"{r['rights']}/{r['source_type']}": r["n"]
        for r in conn.execute(
            "SELECT rights, source_type, COUNT(*) AS n"
            " FROM documents GROUP BY rights, source_type"
        )
    }
    trainable = training_ready(conn)
    lines.append(
        f"[P3 资料] {sum(docs_by_rights.values())} 份文档 "
        f"{docs_by_rights} | 可训练 {trainable}"
    )

    # P4 理解层
    live = _count(
        conn,
        "SELECT COUNT(*) FROM doc_structures"
        " WHERE is_mock=0",
    )
    mock = _count(
        conn,
        "SELECT COUNT(*) FROM doc_structures"
        " WHERE is_mock=1",
    )
    lines.append(f"[P4 理解] 结构化: live={live} mock={mock}")

    # P5 知识层
    knowledge = _count(conn, "SELECT COUNT(*) FROM knowledge")
    by_kind = {
        r["kind"]: r["n"]
        for r in conn.execute(
            "SELECT kind, COUNT(*) AS n FROM knowledge"
            " GROUP BY kind"
        )
    }
    # 引用逐字复检（抽样全量太慢时也够快，直接全查）
    rows = conn.execute(
        "SELECT k.payload, d.raw_path FROM knowledge k"
        " JOIN documents d ON d.id=k.document_id"
        " WHERE k.extractor='llm-v1'"
    ).fetchall()
    bad_quotes = 0
    for r in rows:
        payload = json.loads(r["payload"])
        quote = (payload.get("quote") or "").strip()
        if not quote:
            continue
        try:
            text = Path(r["raw_path"]).read_text(
                encoding="utf-8"
            )
        except UnicodeDecodeError:
            text = Path(r["raw_path"]).read_text(
                encoding="gbk", errors="replace"
            )
        if quote not in text[:3000]:
            bad_quotes += 1
    lines.append(
        f"[P5 知识] {knowledge} 条 {by_kind} | "
        f"引用非逐字残留: {bad_quotes}"
    )

    # P7/P8 数据集（默认 v2，与 cli dataset/quality-gate 对齐）
    ds_dir = DATA_DIR / "datasets" / "v2"
    samples_path = ds_dir / "samples.jsonl"
    if samples_path.is_file():
        samples = [
            json.loads(l)
            for l in samples_path.read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        report_path = ds_dir / "quality_report.json"
        gate = (
            json.loads(report_path.read_text(encoding="utf-8"))
            if report_path.is_file()
            else {}
        )
        splits = {
            name: len(
                (ds_dir / f"{name}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            for name in ("train", "validation", "test")
            if (ds_dir / f"{name}.jsonl").is_file()
        }
        lines.append(
            f"[P7 数据] {len(samples)} 条样本 | "
            f"[P8 质检] 保留 {gate.get('kept')}/"
            f"{gate.get('total')} | 切分 {splits}"
        )
    else:
        lines.append("[P7-P8 数据] 未生成")

    # P9 训练包
    lf_dir = DATA_DIR / "llama_factory" / "dataset"
    lf = {
        p.stem: len(
            json.loads(
                "[" + ",".join(
                    l
                    for l in p.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if l.strip()
                )
                + "]"
            )
        )
        for p in sorted(lf_dir.glob("sy_*.json"))
    } if lf_dir.is_dir() else {}
    if lf:
        lines.append(f"[P9 训练包] LLaMA-Factory alpaca: {lf}")
    else:
        lines.append("[P9 训练包] 未导出")

    return "\n".join(lines)
