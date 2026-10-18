from __future__ import annotations

import hashlib
from pathlib import Path

PROMPTS_DIR = (
    Path(__file__).resolve().parent / "prompts"
)


def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8")


def prompt_version(name: str) -> str:
    """提示词版本 = 内容哈希前 12 位（借鉴 AIHOT：改词即换版，
    已分析过的文档不重算）。"""
    content = load_prompt(name)
    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()[:12]


def render(name: str, **variables) -> str:
    """加载模板并替换 {var} 占位符（提示词里的 JSON 花括号
    需写成 {{}}，先还原再替换变量，避免 format 撞 JSON）。"""
    text = load_prompt(name)
    text = text.replace("{{", "\x00").replace(
        "}}", "\x01"
    )
    for key, value in variables.items():
        text = text.replace(
            "{" + key + "}", str(value)
        )
    return text.replace("\x00", "{").replace(
        "\x01", "}"
    )
