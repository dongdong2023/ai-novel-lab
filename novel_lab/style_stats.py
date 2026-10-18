from __future__ import annotations

import re

CHINESE_RE = re.compile(r"[\u3400-\u9fff]")

SENTENCE_RE = re.compile(
    r"[^。！？!?；;\n]+[。！？!?；;]?"
)

DIALOGUE_RE = re.compile(r'[“"「『].*?[”"」』]')


def analyze_style(text: str) -> dict:
    """确定性风格统计（无需 LLM，零幻觉）：
    句长/段长/对话占比。"""

    chinese_chars = len(CHINESE_RE.findall(text))

    sentences = [
        m.group(0)
        for m in SENTENCE_RE.finditer(text)
        if CHINESE_RE.search(m.group(0))
    ]

    sentence_lengths = [
        len(CHINESE_RE.findall(s)) for s in sentences
    ]

    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n+", text)
        if p.strip()
    ]

    paragraph_lengths = [
        len(CHINESE_RE.findall(p)) for p in paragraphs
    ]

    dialogue_chars = sum(
        len(CHINESE_RE.findall(m.group(0)))
        for m in DIALOGUE_RE.finditer(text)
    )

    def _mean(values):
        return (
            round(sum(values) / len(values), 2)
            if values
            else 0
        )

    return {
        "chinese_chars": chinese_chars,
        "sentences": len(sentences),
        "paragraphs": len(paragraphs),
        "sentence_length_mean": _mean(sentence_lengths),
        "paragraph_length_mean": _mean(
            paragraph_lengths
        ),
        "dialogue_ratio": (
            round(dialogue_chars / chinese_chars, 4)
            if chinese_chars
            else 0
        ),
    }
