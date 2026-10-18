"""叙事动力学分析：Gemini 评审 P0-2 改进。

诊断：原系统只抽"静态设定"（人物/关系/世界观），
缺"叙事动力学"——Hook（钩子）/Escalation（升级）/
Payoff（回收）/断章（cliffhanger）四维。

本模块提供确定性（正则+末180字规则）的章节叙事分析，
挂载到 knowledge_extract.extract_chapter_stats。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------- Hook（开篇钩子）5 类 ----------

HOOK_TYPES = {
    "conflict_hook": "冲突钩：开篇即矛盾（追杀/对峙/危机）",
    "mystery_hook": "悬念钩：抛未解之谜（血字/异象/失踪）",
    "contrast_hook": "反差钩：前后境况强烈对比",
    "dialogue_hook": "对白钩：以带信息量的对白开篇",
    "scene_hook": "场景钩：强画面感的环境/动作开篇",
}

_HOOK_PATTERNS: list[tuple[str, list[str]]] = [
    ("conflict_hook", ["追杀", "围攻", "对峙", "刀架", "枪口", "危机", "逃"]),
    ("mystery_hook", ["血字", "异象", "失踪", "谜", "诡异", "不对劲", "秘密"]),
    ("contrast_hook", ["曾经", "如今", "没想到", "谁能想到", "三天前"]),
    ("dialogue_hook", []),  # 特殊规则：首段引号开头
    ("scene_hook", ["夜色", "血月", "浓雾", "暴雨", "寂静", "轰鸣"]),
]

# ---------- Escalation（冲突升级）5 类 ----------

ESCALATION_TYPES = {
    "stakes_up": "赌注升级：失去的东西变多/更重要",
    "enemy_up": "对手升级：出现更强/更阴的对手",
    "time_pressure": "时间压迫：倒计时/deadline 出现",
    "info_gap": "信息差：主角掌握对手不知道的关键信息",
    "cost_up": "代价升级：赢一次要付更大代价",
}

_ESCALATION_PATTERNS: list[tuple[str, list[str]]] = [
    ("stakes_up", ["不止", "还有", "更糟", "代价", "失去"]),
    ("enemy_up", ["更强", "高手", "背后", "主使", "竟然是"]),
    ("time_pressure", ["来不及", "倒计时", "天亮前", "三天", "必须"]),
    ("info_gap", ["只有", "唯独", "他知道", "没人知道", "底牌"]),
    ("cost_up", ["拼了", "咬牙", "透支", "代价", "以伤换"]),
]

# ---------- Payoff（伏笔回收）5 类 ----------

PAYOFF_TYPES = {
    "foreshadow_payoff": "伏笔回收：前文埋的细节此处兑现",
    "ability_payoff": "能力兑现：之前练/藏的手段派上用场",
    "relation_payoff": "关系兑现：人情/恩怨在此结算",
    "prop_payoff": "道具兑现：关键道具发挥作用",
    "plan_payoff": "计划兑现：主角的谋划收网",
}

_PAYOFF_PATTERNS: list[tuple[str, list[str]]] = [
    ("foreshadow_payoff", ["果然", "正如", "早料到", "伏笔", "之前"]),
    ("ability_payoff", ["早已", "练过", "藏了一手", "底牌", "原来会"]),
    ("relation_payoff", ["还人情", "恩怨", "欠", "报答", "算账"]),
    ("prop_payoff", ["玉镯", "取出", "掏出", "早已准备", "暗器"]),
    ("plan_payoff", ["收网", "瓮中", "请君入瓮", "计划", "谋划"]),
]

# ---------- 断章（章末悬念）5 类 ----------

CLIFFHANGER_TYPES = {
    "danger_cliff": "险境断章：主角陷入即时危险",
    "reveal_cliff": "揭秘断章：关键信息只说一半",
    "twist_cliff": "反转断章：最后一句翻转预期",
    "arrival_cliff": "登场断章：新人物/势力在末尾登场",
    "choice_cliff": "抉择断章：主角面临两难选择",
}

# 末 180 字的确定性规则：出现这些模式即判对应断章类型
_TAIL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "danger_cliff",
        re.compile(r"(身后|门外|黑暗中).{0,20}(脚步|身影|杀气|响动)"),
    ),
    (
        "reveal_cliff",
        re.compile(r"(原来|竟然).{0,30}[。！？]"),
    ),
    (
        "twist_cliff",
        re.compile(r"(却|但).{0,20}(笑了|转身|掏出|亮出)"),
    ),
    (
        "arrival_cliff",
        re.compile(r"(推门|出现|走进|现身).{0,20}(身影|人影|女子|男子|老者)"),
    ),
    (
        "choice_cliff",
        re.compile(r"(要么|或者|两条路|抉择|赌)"),
    ),
]

_TAIL_CHARS = 180


@dataclass
class BeatEvent:
    """单个叙事节拍事件。"""

    dimension: str = ""  # hook/escalation/payoff/cliffhanger
    beat_type: str = ""  # 具体分类 key
    label: str = ""  # 中文说明
    evidence: str = ""  # 触发证据（原文片段）
    position: str = ""  # head/middle/tail


@dataclass
class NarrativeDynamicsProfile:
    """一章的叙事动力学画像。"""

    chapter: str = ""
    hooks: list[BeatEvent] = field(default_factory=list)
    escalations: list[BeatEvent] = field(default_factory=list)
    payoffs: list[BeatEvent] = field(default_factory=list)
    cliffhanger: BeatEvent | None = None

    def to_dict(self) -> dict:
        def ev(e: BeatEvent) -> dict:
            return {
                "dimension": e.dimension,
                "beat_type": e.beat_type,
                "label": e.label,
                "evidence": e.evidence,
                "position": e.position,
            }

        return {
            "chapter": self.chapter,
            "hooks": [ev(e) for e in self.hooks],
            "escalations": [ev(e) for e in self.escalations],
            "payoffs": [ev(e) for e in self.payoffs],
            "cliffhanger": (
                ev(self.cliffhanger)
                if self.cliffhanger
                else None
            ),
        }

    def has_cliffhanger(self) -> bool:
        return self.cliffhanger is not None


def _match_any(
    text: str, keywords: list[str], window: int = 60
) -> str:
    """在文本中找关键词，返回命中证据片段。"""
    for kw in keywords:
        idx = text.find(kw)
        if idx >= 0:
            start = max(0, idx - window // 2)
            return text[start : idx + window // 2].strip()
    return ""


def analyze_hooks(text: str) -> list[BeatEvent]:
    """分析开篇钩子：只看前 500 字。"""
    head = text[:500]
    events: list[BeatEvent] = []

    # 对白钩特殊规则：首个非空段落以引号开头
    paras = [p.strip() for p in head.split("\n\n") if p.strip()]
    if paras and paras[0].startswith(("“", '"', "「")):
        events.append(
            BeatEvent(
                dimension="hook",
                beat_type="dialogue_hook",
                label=HOOK_TYPES["dialogue_hook"],
                evidence=paras[0][:60],
                position="head",
            )
        )

    for beat_type, keywords in _HOOK_PATTERNS:
        if beat_type == "dialogue_hook":
            continue
        ev = _match_any(head, keywords)
        if ev:
            events.append(
                BeatEvent(
                    dimension="hook",
                    beat_type=beat_type,
                    label=HOOK_TYPES[beat_type],
                    evidence=ev,
                    position="head",
                )
            )
    return events


def analyze_escalation(text: str) -> list[BeatEvent]:
    """分析冲突升级：看中段（500字后到末180字前）。"""
    body = text[500:-_TAIL_CHARS] if len(text) > 680 else text[500:]
    if not body:
        return []
    events: list[BeatEvent] = []
    for beat_type, keywords in _ESCALATION_PATTERNS:
        ev = _match_any(body, keywords)
        if ev:
            events.append(
                BeatEvent(
                    dimension="escalation",
                    beat_type=beat_type,
                    label=ESCALATION_TYPES[beat_type],
                    evidence=ev,
                    position="middle",
                )
            )
    return events


def analyze_payoff(text: str) -> list[BeatEvent]:
    """分析伏笔回收：全文扫描。"""
    events: list[BeatEvent] = []
    for beat_type, keywords in _PAYOFF_PATTERNS:
        ev = _match_any(text, keywords)
        if ev:
            events.append(
                BeatEvent(
                    dimension="payoff",
                    beat_type=beat_type,
                    label=PAYOFF_TYPES[beat_type],
                    evidence=ev,
                    position="middle",
                )
            )
    return events


def analyze_cliffhanger(text: str) -> BeatEvent | None:
    """分析断章：只看末 180 字的确定性规则。"""
    tail = text[-_TAIL_CHARS:] if len(text) >= _TAIL_CHARS else text
    for beat_type, pattern in _TAIL_PATTERNS:
        m = pattern.search(tail)
        if m:
            return BeatEvent(
                dimension="cliffhanger",
                beat_type=beat_type,
                label=CLIFFHANGER_TYPES[beat_type],
                evidence=m.group(0),
                position="tail",
            )
    return None


def analyze_chapter(chapter_title: str, text: str) -> NarrativeDynamicsProfile:
    """一章文本 → 叙事动力学画像。确定性，零幻觉。"""
    profile = NarrativeDynamicsProfile(chapter=chapter_title)
    profile.hooks = analyze_hooks(text)
    profile.escalations = analyze_escalation(text)
    profile.payoffs = analyze_payoff(text)
    profile.cliffhanger = analyze_cliffhanger(text)
    return profile
