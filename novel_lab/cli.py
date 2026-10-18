from __future__ import annotations

import argparse
import time
from pathlib import Path

from novel_lab import db, popularity
from novel_lab.collectors import fetch_profile
from novel_lab.discovery.fanqie import discover_all as discover_fanqie
from novel_lab.discovery.qidian import discover_qidian
from novel_lab.ingest import (
    document_stats,
    ingest_directory,
    ingest_profile,
    training_ready,
)
from novel_lab.settings import DATA_DIR, load_config


def _scored_rows(conn, config):
    rows = db.candidate_rows(conn)
    growth = db.latest_growth(conn)
    for row in rows:
        row["growth_rate"] = growth.get(row["novel_rowid"])
    return popularity.popularity_scores(
        rows,
        board_weight=config["popularity"]["board_weight"],
        read_weight=config["popularity"]["read_weight"],
        rank_weight=config["popularity"]["rank_weight"],
        growth_weight=config["popularity"].get(
            "growth_weight", 0.0
        ),
    )


def cmd_discover(config: dict) -> None:

    discovery_cfg = config["discovery"]

    records, errors = discover_fanqie(
        genders=discovery_cfg.get("genders"),
        rank_types=discovery_cfg.get("rank_types"),
        timeout=config["research"]["request_timeout"],
        user_agent=config["research"]["user_agent"],
        delay_seconds=discovery_cfg.get(
            "request_delay_seconds", 0.8
        ),
        top_per_board=discovery_cfg.get("top_per_board", 10),
    )

    platform_counts = {"fanqie": len(records)}
    fanqie_records = records

    if discovery_cfg.get("qidian_enabled", True):
        qd_records, qd_errors = discover_qidian(
            timeout=config["research"]["request_timeout"],
            user_agent=config["research"]["user_agent"],
        )
        records = fanqie_records + qd_records
        errors = errors + qd_errors
        platform_counts["qidian"] = len(qd_records)

    conn = db.connect()

    novels_count = db.save_discovery(conn, records)

    print(
        f"[PASS] records={len(records)} "
        f"novels_total={novels_count} "
        f"by_platform={platform_counts}"
    )
    if errors:
        print(f"[WARN] failed boards: {len(errors)}")
        for code, msg in errors:
            print(f"  {code}: {msg}")

    growth = db.latest_growth(conn)
    print(f"[PASS] 增长率可算的书: {len(growth)}")

    scored = _scored_rows(conn, config)
    print("\nPopularity Top10：")
    for row in scored[:10]:
        growth_str = (
            f"+{row['growth_rate']:.4f}"
            if row.get("growth_rate") is not None
            else "-"
        )
        print(
            f"  {row['popularity_score']:.4f}  "
            f"[{row['platform']}] {row['title']}  "
            f"read={int(row['max_read'] or 0):,}  "
            f"boards={row['board_count']}  "
            f"growth={growth_str}"
        )

    conn.close()


def cmd_top(config: dict, limit: int) -> None:

    conn = db.connect()

    for row in db.top_novels(
        conn, _scored_rows(conn, config), limit
    ):
        growth_str = (
            f"+{row['growth_rate']:.4f}"
            if row.get("growth_rate") is not None
            else "-"
        )
        print(
            f"\n{row['popularity_score']:.4f}  "
            f"[{row['platform']}] {row['title']}"
            f"  read={int(row['max_read'] or 0):,}"
            f"  growth={growth_str}"
        )
        for b in row["boards"]:
            print(
                f"    {b['board']} "
                f"{b['gender']}/{b['rank_type']}/{b['category']}"
                f"  rank={b['rank']}  "
                f"read={int(b['read_count'] or 0):,}"
            )

    conn.close()


def cmd_collect(config: dict, top: int, platform: str | None) -> None:
    """给 Popularity TopN 的书抓书页元数据，作为资料入库。"""

    conn = db.connect()

    scored = _scored_rows(conn, config)
    if platform:
        scored = [
            r for r in scored if r["platform"] == platform
        ]

    store_dir = DATA_DIR / "documents"

    created = dup = failed = 0

    for row in scored[:top]:
        try:
            profile = fetch_profile(
                row["platform"],
                row["book_id"],
                timeout=config["research"][
                    "request_timeout"
                ],
                user_agent=config["research"]["user_agent"],
            )
            _, status = ingest_profile(
                conn, profile, store_dir
            )
            if status == "created":
                created += 1
                print(
                    f"[NEW ] [{row['platform']}] "
                    f"{profile['title']} ({status})"
                )
            else:
                dup += 1
        except Exception as exc:
            failed += 1
            print(
                f"[FAIL] [{row['platform']}] "
                f"{row['title']}: {exc}"
            )
        time.sleep(1.0)

    conn.close()

    print(
        f"\n[PASS] created={created} duplicate={dup} "
        f"failed={failed}"
    )


def cmd_ingest(
    config: dict,
    directory: str,
    rights: str,
    novel_title: str,
) -> None:

    conn = db.connect()

    stats = ingest_directory(
        conn,
        Path(directory),
        rights=rights,
        novel_title=novel_title,
    )

    print(
        f"[PASS] created={stats['created']} "
        f"duplicates={stats['duplicates']}"
    )
    print(
        f"documents: {document_stats(conn)} | "
        f"可训练语料: {training_ready(conn)}"
    )
    conn.close()


def cmd_docs(config: dict, limit: int) -> None:

    conn = db.connect()

    print(
        f"documents: {document_stats(conn)} | "
        f"可训练语料: {training_ready(conn)}"
    )

    rows = conn.execute(
        "SELECT d.id, d.source_type, d.rights, d.title,"
        " n.title AS novel FROM documents d"
        " LEFT JOIN novels n ON n.id=d.novel_rowid"
        " ORDER BY d.id DESC LIMIT ?",
        (limit,),
    ).fetchall()

    for r in rows:
        print(
            f"  #{r['id']} [{r['rights']}|"
            f"{r['source_type']}] {r['title']}"
            f" <- {r['novel'] or '-'}"
        )

    conn.close()


def cmd_understand(config: dict, limit: int | None, mock: bool) -> None:
    """预筛+结构化跑待处理文档。mock=确定性假分析，零 LLM 花费。"""

    from novel_lab.llm.client import make_client
    from novel_lab.understanding.pipeline import (
        UnderstandingPipeline,
    )

    conn = db.connect()

    client = None
    if not mock:
        client = make_client(conn, config)
        # 提前触发禁用检查，给出清晰报错
        client.chat("ping", "ping", purpose="health")

    pipeline = UnderstandingPipeline(conn, client)
    stats = pipeline.run(limit, mock=mock)

    conn.close()

    mode = "mock(零LLM)" if mock else "live"
    print(
        f"[PASS] mode={mode} processed={stats['processed']} "
        f"pass={stats['pass']} block={stats['block']} "
        f"unknown={stats['unknown']} "
        f"parse_errors={stats['parse_errors']}"
    )


def cmd_eval_gold(config: dict, mock: bool) -> None:

    from novel_lab import eval_gold
    from novel_lab.llm.client import make_client

    gold = eval_gold.load_gold()

    conn = db.connect()
    client = None if mock else make_client(conn, config)

    report = eval_gold.evaluate(
        gold, conn, client, mock=mock
    )
    conn.close()

    mode = "mock" if mock else "live"
    print(f"=== 校准评测 ({mode}) ===")
    print(eval_gold.format_report(report))


def cmd_extract(config: dict, mock: bool, limit: int | None) -> None:
    """知识抽取：确定性部分总是跑；LLM 部分需要 key（非 mock）。"""

    from novel_lab.knowledge_extract import (
        knowledge_summary,
        run_extract,
    )
    from novel_lab.llm.client import make_client

    conn = db.connect()

    client = None
    use_llm = not mock
    if use_llm:
        client = make_client(conn, config)
        client.chat("ping", "ping", purpose="health")

    stats = run_extract(
        conn, client, use_llm=use_llm, limit=limit
    )

    summary = knowledge_summary(conn)
    conn.close()

    mode = (
        "确定性(零LLM)"
        if mock
        else "确定性+LLM"
    )
    print(
        f"[PASS] mode={mode} docs={stats['docs']} "
        f"knowledge={stats['knowledge']} "
        f"llm_items={stats['llm_items']} "
        f"llm_failed={stats.get('llm_failed', 0)} "
        f"skipped_no_novel={stats['skipped_no_novel']}"
    )
    print(f"知识分布: {summary}")


def cmd_knowledge(
    config: dict, novel_title: str | None
) -> None:

    from novel_lab.knowledge_extract import (
        character_totals,
        knowledge_summary,
    )

    conn = db.connect()

    summary = knowledge_summary(conn, novel_title)

    print(
        f"知识分布"
        f"{'（' + novel_title + '）' if novel_title else ''}"
        f": {summary}"
    )

    if novel_title:
        print("\n人物出场（名/章数/总提及）：")
        for name, chapters, total in character_totals(
            conn, novel_title
        ):
            print(f"  {name}: {chapters}章 / {total}次")

    conn.close()


def cmd_graph(config: dict, novel_title: str) -> None:

    from novel_lab.knowledge_graph import (
        character_index,
        graph_summary,
        load_knowledge,
        relationship_edges,
    )

    conn = db.connect()
    knowledge = load_knowledge(conn, novel_title)
    characters = character_index(knowledge)
    edges = relationship_edges(knowledge, characters)
    total = max(
        (item["_ch"] for item in knowledge), default=0
    )
    conn.close()

    print(
        graph_summary(characters, edges, total)
    )


def cmd_gaps(
    config: dict, novel_title: str, use_llm: bool
) -> None:

    from novel_lab.knowledge_graph import (
        character_index,
        detect_gaps,
        load_knowledge,
        rule_conflicts,
    )

    conn = db.connect()
    knowledge = load_knowledge(conn, novel_title)
    characters = character_index(knowledge)
    total = max(
        (item["_ch"] for item in knowledge), default=0
    )

    findings = detect_gaps(
        knowledge, characters, total
    )

    if use_llm:
        from novel_lab.llm.client import make_client

        client = make_client(conn, config)
        conflicts = rule_conflicts(
            knowledge, client
        )
        for c in conflicts:
            findings.append(
                {
                    "rule": "R5规则冲突(LLM)",
                    "who": c["pair"],
                    "detail": c["reason"],
                }
            )

    conn.close()

    if not findings:
        print("[PASS] 未发现知识缺口")
        return

    print(f"知识体检：{len(findings)} 项发现\n")
    for f in findings:
        print(f"[{f['rule']}] {f['who']}：{f['detail']}")


def cmd_dataset(config: dict, novel_title: str) -> None:

    from novel_lab.dataset_factory import (
        build_dataset,
        write_dataset,
    )

    conn = db.connect()
    samples, stats = build_dataset(conn, novel_title)
    conn.close()

    path = write_dataset(samples, stats)

    print(
        f"[PASS] 样本 {stats['total']} 条 "
        f"(可训练文档 {stats['trainable_docs']} 份/"
        f"知识 {stats['knowledge_used']} 条)"
    )
    for cat, n in sorted(
        stats["by_category"].items(), key=lambda x: -x[1]
    ):
        print(f"  {cat}: {n}")
    print(path)


def cmd_quality_gate(config: dict, novel_title: str) -> None:

    import json

    from novel_lab.quality_gate import (
        run_gate,
        split_dataset,
        write_split,
    )
    from novel_lab.settings import DATA_DIR

    samples_path = DATA_DIR / "datasets" / "v1" / "samples.jsonl"
    samples = [
        json.loads(line)
        for line in samples_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    conn = db.connect()
    kept, dropped, report = run_gate(
        conn, samples, novel_title
    )
    conn.close()

    splits = split_dataset(kept)
    out_dir = write_split(splits, report)

    print(
        f"[PASS] 总数 {report['total']} | 保留 {report['kept']}"
        f" | 淘汰 {report['dropped']}"
    )
    if report["by_reason"]:
        print("淘汰原因:", report["by_reason"])
    for name, items in splits.items():
        print(f"  {name}: {len(items)}")
    print(out_dir)


def main() -> None:

    config = load_config()

    parser = argparse.ArgumentParser(
        description=(
            "AI 小说自动学习系统"
            "（P3：多平台发现→资料采集→判重→Rights Gate）"
        )
    )

    sub = parser.add_subparsers(
        dest="command", required=True
    )

    sub.add_parser(
        "discover", help="轮询番茄+起点榜单并落库"
    )

    top = sub.add_parser(
        "top", help="按 PopularityScore 查询 TopN"
    )
    top.add_argument(
        "--limit", type=int, default=10
    )

    collect = sub.add_parser(
        "collect",
        help="抓 TopN 书的公开书页元数据入库",
    )
    collect.add_argument("--top", type=int, default=5)
    collect.add_argument(
        "--platform", choices=["fanqie", "qidian"]
    )

    ingest = sub.add_parser(
        "ingest", help="登记本地语料目录（rights 门禁）"
    )
    ingest.add_argument("directory")
    ingest.add_argument(
        "--rights", default="user_provided"
    )
    ingest.add_argument("--novel-title", default="")

    docs = sub.add_parser(
        "docs", help="查看 documents 概况"
    )
    docs.add_argument("--limit", type=int, default=20)

    understand = sub.add_parser(
        "understand",
        help="预筛+结构化待处理文档",
    )
    understand.add_argument(
        "--limit", type=int, default=None
    )
    understand.add_argument(
        "--mock",
        action="store_true",
        help="确定性假分析干跑，零 LLM 花费",
    )

    eval_gold_parser = sub.add_parser(
        "eval-gold",
        help="用 gold.jsonl 校准集评测预筛/结构化",
    )
    eval_gold_parser.add_argument(
        "--mock",
        action="store_true",
        help="mock 模式评测（无 LLM）",
    )

    extract = sub.add_parser(
        "extract",
        help="知识抽取（确定性总是跑，LLM 需 key）",
    )
    extract.add_argument("--limit", type=int, default=None)
    extract.add_argument(
        "--mock",
        action="store_true",
        help="只跑确定性抽取，零 LLM 花费",
    )

    knowledge = sub.add_parser(
        "knowledge", help="查询知识库"
    )
    knowledge.add_argument("--novel-title")

    graph = sub.add_parser(
        "graph", help="知识图谱视图（人物/关系）"
    )
    graph.add_argument("--novel-title", required=True)

    gaps = sub.add_parser(
        "gaps", help="知识缺口体检"
    )
    gaps.add_argument("--novel-title", required=True)
    gaps.add_argument(
        "--llm",
        action="store_true",
        help="附加 LLM 世界观规则冲突扫描（需隧道+enabled）",
    )

    dataset = sub.add_parser(
        "dataset", help="生成训练数据集（六类样本）"
    )
    dataset.add_argument("--novel-title", required=True)

    gate = sub.add_parser(
        "quality-gate", help="数据集质检+切分 train/val/test"
    )
    gate.add_argument("--novel-title", required=True)

    export_lf = sub.add_parser(
        "export-lf", help="导出 LLaMA-Factory 训练包"
    )

    status = sub.add_parser(
        "status", help="P10：流水线全景状态"
    )

    args = parser.parse_args()

    if args.command == "discover":
        cmd_discover(config)
    elif args.command == "top":
        cmd_top(config, args.limit)
    elif args.command == "collect":
        cmd_collect(config, args.top, args.platform)
    elif args.command == "ingest":
        cmd_ingest(
            config,
            args.directory,
            args.rights,
            args.novel_title,
        )
    elif args.command == "docs":
        cmd_docs(config, args.limit)
    elif args.command == "understand":
        cmd_understand(config, args.limit, args.mock)
    elif args.command == "eval-gold":
        cmd_eval_gold(config, args.mock)
    elif args.command == "extract":
        cmd_extract(config, args.mock, args.limit)
    elif args.command == "knowledge":
        cmd_knowledge(config, args.novel_title)
    elif args.command == "graph":
        cmd_graph(config, args.novel_title)
    elif args.command == "gaps":
        cmd_gaps(config, args.novel_title, args.llm)
    elif args.command == "dataset":
        cmd_dataset(config, args.novel_title)
    elif args.command == "quality-gate":
        cmd_quality_gate(config, args.novel_title)
    elif args.command == "export-lf":
        from novel_lab.export_llamafactory import (
            export_llamafactory,
        )

        info = export_llamafactory()
        print(
            f"[PASS] LLaMA-Factory 训练包: {info['counts']}"
        )
        print(info["out_dir"])
    elif args.command == "status":
        from novel_lab.pipeline_status import (
            pipeline_status,
        )

        conn = db.connect()
        print(pipeline_status(conn))
        conn.close()


if __name__ == "__main__":
    main()
