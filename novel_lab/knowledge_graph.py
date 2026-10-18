from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict

from novel_lab.knowledge_extract import (
    STATS_EXTRACTOR,
)

_CH_NUM_RE = re.compile(r"(\d+)")


def _chapter_no(doc_title: str) -> int:
    match = _CH_NUM_RE.search(doc_title or "")
    return int(match.group(1)) if match else 0


def _base_name(name: str) -> str:
    """去掉括注变体：'袁经理（袁立军）' -> '袁立军'；
    无括注时原样返回。"""
    name = (name or "").strip()
    m = re.search(r"[（(]([^（）()]+)[）)]$", name)
    if m:
        inner = m.group(1).strip()
        if inner:
            return inner
    return name


def load_knowledge(
    conn: sqlite3.Connection, novel_title: str
) -> list[dict]:
    rows = conn.execute(
        "SELECT k.id, k.kind, k.payload, k.extractor,"
        " d.title AS doc_title, d.id AS doc_id"
        " FROM knowledge k"
        " JOIN novels n ON n.id = k.novel_rowid"
        " JOIN documents d ON d.id = k.document_id"
        " WHERE n.title = ?"
        " ORDER BY d.title",
        (novel_title,),
    ).fetchall()
    out = []
    for r in rows:
        payload = json.loads(r["payload"])
        payload["_id"] = r["id"]
        payload["_kind"] = r["kind"]
        payload["_extractor"] = r["extractor"]
        payload["_ch"] = _chapter_no(r["doc_title"])
        payload["_doc"] = r["doc_title"]
        out.append(payload)
    return out


def character_index(
    knowledge: list[dict],
) -> dict[str, dict]:
    """人物归一索引：词典统计（有提及数）与 LLM 抽取（有身份/性格）
    合并到同一实体；括注变体归到括注内的本名。"""

    index: dict[str, dict] = {}

    def entry(name: str) -> dict:
        return index.setdefault(
            name,
            {
                "name": name,
                "chapters": set(),
                "mentions": 0,
                "identities": [],
                "traits": [],
                "sources": set(),
            },
        )

    for item in knowledge:
        if item["_kind"] != "character":
            continue
        name = _base_name(item.get("name", ""))
        if not name:
            continue
        e = entry(name)
        e["chapters"].add(item["_ch"])
        e["sources"].add(item["_extractor"])
        if item["_extractor"] == STATS_EXTRACTOR:
            e["mentions"] += item.get("mentions", 0)
        else:
            identity = item.get("identity")
            if identity:
                e["identities"].append(identity)
            e["traits"].extend(item.get("traits") or [])

    # 名字包含归并：'袁经理' 若存在且 '袁立军' 也存在，
    # 把短的并进词典确认过的本名（保守：仅当短名是长名的前缀）
    for short in list(index):
        for long_ in list(index):
            if short != long_ and long_.startswith(short):
                tgt = index[long_]
                src = index[short]
                if STATS_EXTRACTOR in tgt["sources"]:
                    tgt["chapters"] |= src["chapters"]
                    tgt["mentions"] += src["mentions"]
                    tgt["identities"].extend(src["identities"])
                    del index[short]
                    break

    return index


def relationship_edges(
    knowledge: list[dict],
    characters: dict[str, dict],
) -> list[dict]:
    """关系边聚合：(甲, 乙) 去重合并，记录关系类型与出现章。"""

    edges: dict[tuple[str, str], dict] = {}

    for item in knowledge:
        if item["_kind"] != "relationship":
            continue
        a = _base_name(item.get("from", ""))
        b = _base_name(item.get("to", ""))
        if not a or not b:
            continue
        key = (a, b) if a <= b else (b, a)
        e = edges.setdefault(
            key,
            {
                "from": key[0],
                "to": key[1],
                "relations": defaultdict(int),
                "chapters": set(),
            },
        )
        e["relations"][item.get("relation", "?")] += 1
        e["chapters"].add(item["_ch"])

    result = []
    for e in edges.values():
        e["chapters"] = sorted(e["chapters"])
        e["relations"] = dict(e["relations"])
        e["dangling"] = (
            e["from"] not in characters
            or e["to"] not in characters
        )
        result.append(e)

    return sorted(
        result,
        key=lambda x: -len(x["chapters"]),
    )


def detect_gaps(
    knowledge: list[dict],
    characters: dict[str, dict],
    total_chapters: int,
) -> list[dict]:
    """缺口规则（全部确定性）：

    R1 主角团断档：高出场角色连续 >=3 章完全缺席
    R2 早期沉底：前5章出现后再未出现（伏线沉底候选）
    R3 悬空关系：关系端点不在人物索引
    R4 泛称幽灵：LLM 抽出的单章泛称人物（噪声候选）
    """

    findings: list[dict] = []

    for name, e in characters.items():
        chapters = sorted(e["chapters"])
        if not chapters or chapters[0] == 0:
            continue

        # R1 连续缺席
        if e["mentions"] >= 30 and len(chapters) >= 2:
            present = set(chapters)
            missing = [
                c
                for c in range(
                    min(chapters), max(chapters) + 1
                )
                if c not in present
            ]
            run, runs = [], []
            for c in missing:
                if run and c == run[-1] + 1:
                    run.append(c)
                else:
                    if run:
                        runs.append(run)
                    run = [c]
            if run:
                runs.append(run)
            for run in runs:
                if len(run) >= 3:
                    findings.append(
                        {
                            "rule": "R1主角断档",
                            "who": name,
                            "detail": (
                                f"第{run[0]}-{run[-1]}章"
                                f"连续{len(run)}章未出现"
                            ),
                        }
                    )

        # R2 早期沉底
        first, last = chapters[0], chapters[-1]
        if first <= 5 and last <= 8 and total_chapters >= 15:
            findings.append(
                {
                    "rule": "R2早期沉底",
                    "who": name,
                    "detail": (
                        f"首次第{first}章、最后第{last}章，"
                        f"此后{total_chapters - last}章未再出现"
                    ),
                }
            )

    edges = relationship_edges(knowledge, characters)
    for e in edges:
        if e["dangling"]:
            findings.append(
                {
                    "rule": "R3悬空关系",
                    "who": f"{e['from']}→{e['to']}",
                    "detail": (
                        "关系端点无人物条目（"
                        + "、".join(e["relations"])
                        + "）"
                    ),
                }
            )

    ghost_words = ("未知", "的人", "老头", "老太太", "店长")
    for item in knowledge:
        if (
            item["_kind"] == "character"
            and item["_extractor"] != STATS_EXTRACTOR
        ):
            name = item.get("name", "")
            if any(w in name for w in ghost_words):
                findings.append(
                    {
                        "rule": "R4泛称幽灵",
                        "who": name,
                        "detail": (
                            f"LLM单章泛称（第{item['_ch']}章），"
                            "合并或清理候选"
                        ),
                    }
                )
                break  # 只报一次，避免刷屏

    return findings


def graph_summary(
    characters: dict[str, dict],
    edges: list[dict],
    total_chapters: int,
) -> str:
    lines = [
        f"## 人物图（覆盖 {total_chapters} 章）",
        "",
        "| 人物 | 出现章数 | 词典提及 | 身份（LLM） |",
        "|---|---|---|---|",
    ]

    ordered = sorted(
        characters.items(),
        key=lambda kv: (-kv[1]["mentions"], -len(kv[1]["chapters"])),
    )
    for name, e in ordered[:15]:
        identity = (
            "；".join(dict.fromkeys(e["identities"]))[:30]
            or "-"
        )
        lines.append(
            f"| {name} | {len(e['chapters'])} | "
            f"{e['mentions'] or '-'} | {identity} |"
        )

    lines.append("\n## 关系边（按覆盖章数排序，前 12）\n")
    lines.append("| 甲 | 乙 | 关系 | 覆盖章 |")
    lines.append("|---|---|---|---|")
    for e in edges[:12]:
        rel = "、".join(
            f"{k}×{v}"
            for k, v in sorted(
                e["relations"].items(),
                key=lambda kv: -kv[1],
            )[:3]
        )
        span = (
            f"{e['chapters'][0]}-{e['chapters'][-1]}"
            if len(e["chapters"]) > 1
            else str(e["chapters"][0])
        )
        mark = " ⚠悬空" if e["dangling"] else ""
        lines.append(
            f"| {e['from']} | {e['to']} | {rel} | {span}{mark} |"
        )

    return "\n".join(lines)


RULE_CONFLICT_SYSTEM = (
    "你是小说设定一致性审查器，只输出 JSON。"
    "只报告互相矛盾或前后冲突的规则对，没有冲突就给空数组，禁止硬凑。"
)


def rule_conflicts(
    knowledge: list[dict], client, chunk_size: int = 25
) -> list[dict]:
    """LLM 辅助：世界观规则互斥扫描（R5）。

    98 条规则两两比较对 27B 太重（单次生成可超 600s），
    分块扫描：每块 25 条、限报 5 对、每对一句话。
    跨块冲突不覆盖——按主题分块时可接受，记录在案。
    """

    rules = [
        item
        for item in knowledge
        if item["_kind"] == "world_rule"
        and item.get("rule")
    ]
    if not rules:
        return []

    from novel_lab.understanding.pipeline import (
        extract_json,
    )

    conflicts: list[dict] = []

    for start in range(0, len(rules), chunk_size):
        chunk = rules[start : start + chunk_size]
        listing = "\n".join(
            f"R{i}: {item['rule']}"
            for i, item in enumerate(chunk, 1)
        )
        user = (
            "以下是一部小说抽取的世界观规则（R编号: 内容）。\n"
            "找出**互相矛盾**的规则对（数值/因果/机制冲突），"
            "最多报 5 对，每对一句话理由。\n"
            "没有冲突就给空数组，禁止把互补当矛盾。\n\n"
            f"{listing}\n\n"
            '输出 JSON：{"conflicts": [{"pair": "R3×R17", '
            '"reason": "一句话"}]}'
            "\n没有冲突输出 {\"conflicts\": []}"
        )
        raw = client.chat(
            RULE_CONFLICT_SYSTEM,
            user,
            purpose=f"rule_conflicts_c{start // chunk_size}",
        )
        data = extract_json(raw) or {}
        chunk_conflicts = data.get("conflicts") or []

        if not isinstance(chunk_conflicts, list):
            continue
        for c in chunk_conflicts:
            if not isinstance(c, dict):
                continue
            pair = str(c.get("pair", ""))
            reason = str(c.get("reason", ""))[:120]
            if not re.match(r"^R\d+×R\d+$", pair):
                continue
            if not reason:
                continue
            # 局部编号 -> 全局行号，便于回查
            a, b = pair.split("×")
            ga = start + int(a[1:])
            gb = start + int(b[1:])
            conflicts.append(
                {
                    "pair": f"R{ga}×R{gb}",
                    "reason": reason,
                }
            )

    return conflicts
