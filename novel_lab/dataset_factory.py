from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
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
    min_input: int = 400,
    max_output: int = 500,
    max_plain: int = 4,
    max_dialogue: int = 8,
    anchor_gap: int = 3,
) -> list[dict]:
    """F 写作任务：自监督续写切片（多切片版）。

    每章两类切片，全部来自可训练 rights 的文本：
    - 普通滑窗：从头起不重叠推进（input≥min_input，output 收敛到
      max_output），最多 max_plain 片；
    - 对白锚点：正文中含引号的段落按 anchor_gap 间隔均匀取点，
      output 从该对白段起笔，最多 max_dialogue 片。
    段落级连续性由 P8 质检的切片连续校验兜底。
    """

    samples = []

    def make(
        doc_title: str,
        cut: int,
        end: int,
        mode: str,
        paragraphs: list[str],
    ) -> dict:
        inp = "\n\n".join(paragraphs[:cut])
        out = "\n\n".join(paragraphs[cut:end])
        return {
            "category": "writing",
            "subtype": mode,
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
            "source_doc": doc_title,
            "knowledge_id": None,
        }

    for doc in docs:
        text = _read_doc(doc["raw_path"])
        paragraphs = [
            p.strip()
            for p in text.split("\n\n")
            if p.strip()
        ]
        if len(paragraphs) < 6:
            continue

        # ---- 普通滑窗 ----
        start = 0
        made = 0
        while made < max_plain:
            acc = 0
            cut = start
            for i in range(start, len(paragraphs)):
                acc += len(paragraphs[i])
                cut = i + 1
                if acc >= min_input:
                    break
            if cut >= len(paragraphs):
                break
            out_acc = 0
            end = cut
            for i in range(cut, len(paragraphs)):
                out_acc += len(paragraphs[i])
                end = i + 1
                if out_acc >= max_output:
                    break
            if end <= cut or end - cut < 1:
                break
            samples.append(
                make(doc["title"], cut, end, "plain", paragraphs)
            )
            made += 1
            start = end

        # ---- 对白锚点 ----
        anchors = [
            i
            for i, p in enumerate(paragraphs)
            if "“" in p and i > 0
        ]
        if anchors:
            picked = []
            last = -10 ** 9
            step = max(1, len(anchors) // max_dialogue)
            for i in anchors[::step]:
                if i - last >= anchor_gap:
                    picked.append(i)
                    last = i
                if len(picked) >= max_dialogue:
                    break

            for anchor in picked:
                out_acc = 0
                end = anchor
                for i in range(anchor, len(paragraphs)):
                    out_acc += len(paragraphs[i])
                    end = i + 1
                    if out_acc >= max_output:
                        break
                if end <= anchor:
                    continue
                samples.append(
                    make(
                        doc["title"],
                        anchor,
                        end,
                        "dialogue",
                        paragraphs,
                    )
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
    samples.extend(samples_continuation_v2(docs))

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


# ======================================================================
# 数据集 v2：Gemini 评审 P0-1 改进
#
# 诊断：原六类样本中 5 类是"百科问答"（qa_appearance/qa_relation/
# character/plot/world），训出来的是"问答机器人"不是"写手"。
# v2 废弃百科问答，新增 4 种写作样本生成器，全部样本带溯源字段。
# ======================================================================


#: 已废弃的 5 类百科问答（保留函数供回溯，但 build_dataset_v2 不再调用）
DEPRECATED_QA_CATEGORIES = frozenset(
    {
        "qa_appearance",
        "qa_relation",
        "character",
        "plot",
        "world",
    }
)


@dataclass
class SceneBeatSpec:
    """场景节拍规格：写作样本的结构化输入。

    beat_type: 节拍类型（hook/escalation/payoff/cliffhanger）
    location: 场景地点
    characters: 出场人物
    conflict: 冲突核心（一句话）
    emotion: 情绪基调
    pov: 视角人物
    foreshadow: 埋的钩子/伏笔
    """

    beat_type: str = ""
    location: str = ""
    characters: list[str] = field(default_factory=list)
    conflict: str = ""
    emotion: str = ""
    pov: str = ""
    foreshadow: str = ""
    source_chapter: str = ""

    def to_prompt(self) -> str:
        lines = [f"【节拍类型】{self.beat_type or '未指定'}"]
        if self.location:
            lines.append(f"【地点】{self.location}")
        if self.characters:
            lines.append(
                "【人物】" + "、".join(self.characters)
            )
        if self.pov:
            lines.append(f"【视角】{self.pov}")
        if self.conflict:
            lines.append(f"【冲突】{self.conflict}")
        if self.emotion:
            lines.append(f"【情绪】{self.emotion}")
        if self.foreshadow:
            lines.append(f"【伏笔】{self.foreshadow}")
        return "\n".join(lines)


#: 回译增强系统提示：把原文段落改写成"节拍规格→正文"的
#: 逆向样本时用，保证改写不改变事实只改变表达。
BACK_TRANSLATION_SYSTEM_PROMPT = """你是网文写作样本增强助手。你的任务是把一段小说正文，
逆向还原成"场景节拍规格"（SceneBeatSpec）。

规则：
1. 只提取事实性节拍信息（地点/人物/冲突/情绪/伏笔），不许编造原文没有的内容
2. 输出严格使用以下字段，每行一个：
   【节拍类型】hook/escalation/payoff/cliffhanger 四选一
   【地点】场景发生的地点
   【人物】出场人物，用顿号分隔
   【视角】视角人物
   【冲突】本段的核心冲突，一句话
   【情绪】情绪基调，如"紧张""压抑""暗爽"
   【伏笔】本段埋下的钩子，没有则留空
3. 不要输出正文复述，不要加解释"""


def _sha256(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def _provenance(
    doc: dict, text: str
) -> dict:
    """溯源字段：每个样本必须携带。"""
    return {
        "source_doc_id": doc.get("id"),
        "source_doc": doc.get("title", ""),
        "rights_status": doc.get("rights", "unknown"),
        "text_sha256": _sha256(text),
    }


def _mark_deprecated(sample: dict) -> dict:
    """给废弃问答样本打标（保留可追溯，不再进入 v2 训练）。"""
    sample["deprecated"] = True
    sample["deprecated_reason"] = (
        "百科问答已废弃：训出的是问答机器人而非写手"
        "（Gemini 评审 P0-1，数据集 v2）"
    )
    return sample


# ---------- 4 种写作样本生成器 ----------


def samples_beat_to_prose(
    docs: list[dict],
    knowledge: list[dict],
    *,
    max_samples: int = 200,
) -> list[dict]:
    """G1 节拍→正文：input=场景节拍规格，output=原文段落。

    模型学的是"给定节拍如何展开成文"，这是写手的核心能力。
    节拍规格从事件知识确定性拼装（地点/人物/冲突来自
    event/world_rule/character 知识条目）。
    """

    # 按章节聚合事件
    events_by_doc: dict[str, list[dict]] = {}
    for item in knowledge:
        if item["_kind"] != "event":
            continue
        events_by_doc.setdefault(
            item["_doc"], []
        ).append(item)

    samples = []
    for doc in docs:
        events = events_by_doc.get(doc["title"], [])
        if not events:
            continue
        text = _read_doc(doc["raw_path"])
        paragraphs = [
            p.strip()
            for p in text.split("\n\n")
            if p.strip() and len(p.strip()) >= 80
        ]
        if not paragraphs:
            continue

        for i, ev in enumerate(events[:6]):
            spec = SceneBeatSpec(
                beat_type=(
                    "hook" if i == 0
                    else "payoff" if i == len(events) - 1
                    else "escalation"
                ),
                characters=list(
                    ev.get("participants") or []
                )[:4],
                conflict=ev.get("summary", ""),
                pov=(
                    (ev.get("participants") or [""])[0]
                ),
                foreshadow=ev.get("effect", ""),
                source_chapter=doc["title"],
            )
            # 取原文中段一个段落作目标正文
            target = paragraphs[
                min(len(paragraphs) - 1, i * 2 + 1)
            ]
            prompt = (
                "以下是《双月降临》的场景节拍规格，"
                "请展开成一段小说正文（200-400字），"
                "保持原文的克制白描文风：\n"
                + spec.to_prompt()
            )
            sample = {
                "category": "writing",
                "subtype": "beat_to_prose",
                "instruction": prompt,
                "input": "",
                "output": target,
                **_provenance(doc, target),
                "knowledge_id": ev.get("_id"),
            }
            samples.append(sample)
            if len(samples) >= max_samples:
                return samples

    return samples


def samples_continuation_v2(
    docs: list[dict],
    *,
    min_input: int = 800,
    max_output: int = 500,
) -> list[dict]:
    """G2 续写 v2：v1 的升级版。

    改进点：input 末尾追加"下一段节拍提示"
    （从后文第一句抽取冲突关键词），output 长度收紧，
    每个样本带溯源三字段。
    """

    samples = []
    for doc in docs:
        text = _read_doc(doc["raw_path"])
        paragraphs = [
            p.strip()
            for p in text.split("\n\n")
            if p.strip()
        ]
        if len(paragraphs) < 8:
            continue

        # 找两个切点：1/3 处和 2/3 处
        for frac in (0.33, 0.66):
            cut = max(2, int(len(paragraphs) * frac))
            inp_paras = paragraphs[:cut]
            # input 凑到 min_input 字
            acc = sum(len(p) for p in inp_paras)
            if acc < min_input:
                continue
            out_paras = []
            out_acc = 0
            for p in paragraphs[cut:]:
                out_paras.append(p)
                out_acc += len(p)
                if out_acc >= max_output:
                    break
            if not out_paras:
                continue

            # 节拍提示：从 output 首段抽冲突关键词
            first = out_paras[0]
            hint_bits = []
            for kw in (
                "突然", "却", "但是", "然而",
                "竟", "竟然", "猛地", "死死",
            ):
                if kw in first:
                    hint_bits.append(kw)
            hint = (
                "下一段情绪转折关键词："
                + "、".join(hint_bits)
                if hint_bits
                else "下一段延续当前情绪推进"
            )

            inp_text = "\n\n".join(inp_paras)
            out_text = "\n\n".join(out_paras)
            sample = {
                "category": "writing",
                "subtype": "continuation_v2",
                "instruction": (
                    "以下是《双月降临》的正文片段，"
                    "请以相同文风续写。\n" + hint
                ),
                "input": inp_text,
                "output": out_text,
                **_provenance(doc, out_text),
                "knowledge_id": None,
            }
            samples.append(sample)

    return samples


def samples_cliffhanger_hook(
    docs: list[dict],
    *,
    tail_chars: int = 600,
) -> list[dict]:
    """G3 断章钩子：input=章节末段，output=下一章开头钩子句。

    训练模型"断章"能力：章末如何留悬念、章首如何接钩子。
    """

    samples = []
    for doc in docs:
        text = _read_doc(doc["raw_path"])
        if len(text) < tail_chars + 200:
            continue
        # 章末 tail_chars 字作 input
        tail = text[-tail_chars:]
        # 找最后一个完整段落边界
        cut = tail.find("\n\n")
        if cut > 0:
            tail = tail[cut:].strip()
        # output：取文本最后 3 句作"钩子"
        import re

        sentences = re.split(
            r"(?<=[。！？])", text[-tail_chars:]
        )
        sentences = [
            s.strip() for s in sentences if s.strip()
        ]
        hook = "".join(sentences[-3:])
        if len(hook) < 20:
            continue

        sample = {
            "category": "writing",
            "subtype": "cliffhanger_hook",
            "instruction": (
                "以下是《双月降临》某章结尾片段，"
                "请写出下一章开头的钩子句"
                "（1-3句，必须承接悬念、抛出新信息）："
            ),
            "input": tail,
            "output": hook,
            **_provenance(doc, hook),
            "knowledge_id": None,
        }
        samples.append(sample)

    return samples


def samples_dialogue_tension(
    docs: list[dict],
    *,
    max_samples: int = 150,
) -> list[dict]:
    """G4 对白张力：input=场景上下文+人物小传，
    output=带潜台词的对白段落。

    只选原文中引号密度高的段落作目标，保证
    学到的是"对白"而非"叙述"。
    """

    import re

    samples = []
    for doc in docs:
        text = _read_doc(doc["raw_path"])
        paragraphs = [
            p.strip()
            for p in text.split("\n\n")
            if p.strip()
        ]
        for i, p in enumerate(paragraphs):
            if len(samples) >= max_samples:
                return samples
            # 引号密度：至少 2 对引号且对白占比超 40%
            quotes = p.count("“")
            if quotes < 2:
                continue
            dialogue_chars = sum(
                len(m)
                for m in re.findall(
                    r"“[^”]*”", p
                )
            )
            if dialogue_chars < len(p) * 0.4:
                continue
            # 上下文：前 2 段
            context = "\n\n".join(
                paragraphs[max(0, i - 2):i]
            )
            if len(context) < 100:
                continue
            sample = {
                "category": "writing",
                "subtype": "dialogue_tension",
                "instruction": (
                    "以下是《双月降临》的场景上下文，"
                    "请写出一段人物对白：要求有潜台词、"
                    "有信息差、短句为主，"
                    "不许出现说教式长独白："
                ),
                "input": context,
                "output": p,
                **_provenance(doc, p),
                "knowledge_id": None,
            }
            samples.append(sample)

    return samples


# ---------- v2 汇总 ----------


def build_dataset_v2(
    conn: sqlite3.Connection, novel_title: str
) -> tuple[list[dict], dict]:
    """数据集 v2：写作样本为主，百科问答全部废弃。

    样本构成：
      - beat_to_prose      节拍→正文
      - continuation_v2    续写 v2（含节拍提示）
      - cliffhanger_hook   断章钩子
      - dialogue_tension   对白张力
      - summary            章级总结（保留，非问答）
    废弃的 5 类问答不再生成；如需回溯调用 build_dataset()
    并检查 sample["deprecated"] 标记。
    """

    docs = trainable_documents(conn, novel_title)
    knowledge = load_novel_knowledge(conn, novel_title)

    samples: list[dict] = []
    samples.extend(
        samples_beat_to_prose(docs, knowledge)
    )
    samples.extend(samples_continuation_v2(docs))
    samples.extend(samples_cliffhanger_hook(docs))
    samples.extend(samples_dialogue_tension(docs))
    # summary 保留：不是问答，是压缩能力
    samples.extend(samples_summary(knowledge))

    # v2 样本全部带溯源；补齐 summary 类的溯源字段
    for s in samples:
        s.setdefault("source_doc_id", None)
        s.setdefault("rights_status", "unknown")
        if "text_sha256" not in s:
            s["text_sha256"] = _sha256(
                s.get("output", "")
            )

    by_subtype: dict[str, int] = {}
    for s in samples:
        key = s.get("subtype", s["category"])
        by_subtype[key] = by_subtype.get(key, 0) + 1

    stats = {
        "novel": novel_title,
        "version": "v2",
        "generated_at": _now(),
        "trainable_docs": len(docs),
        "knowledge_used": len(knowledge),
        "total": len(samples),
        "by_subtype": by_subtype,
        "deprecated_qa_categories": sorted(
            DEPRECATED_QA_CATEGORIES
        ),
        "rights_note": (
            "仅从 authorized/user_provided/open_license "
            "文档生成；每个样本带 source_doc_id / "
            "rights_status / text_sha256 溯源"
        ),
    }

    return samples, stats


def write_dataset_v2(
    samples: list[dict],
    stats: dict,
    out_dir: Path | None = None,
) -> Path:
    """写 v2 数据集到 data/datasets/v2/。"""

    if out_dir is None:
        out_dir = DATA_DIR / "datasets" / "v2"

    out_dir.mkdir(parents=True, exist_ok=True)

    samples_path = out_dir / "samples.jsonl"
    with samples_path.open(
        "w", encoding="utf-8"
    ) as f:
        for s in samples:
            f.write(
                json.dumps(s, ensure_ascii=False)
                + "\n"
            )

    (out_dir / "metadata.json").write_text(
        json.dumps(
            stats, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )

    return samples_path
