"""版权三级验证：Gemini 评审版权风险改进。

诊断：user_provided 通道存在"洗稿漏洞"——用户上传的文本
声称是自己写的，实为盗版/洗稿，系统无法识别即入库训练，
版权风险敞口。

三级验证（ingest 时强制执行）：
  L1 .rights.json 签名声明：上传目录必须附带权利声明文件，
     声明 rights 类型 + 来源 + 责任人；缺失则拒绝入库。
  L2 盗版水印检测（BLOCK）：文本含盗版站水印/广告
     （如"笔趣阁""最新章节""请访问"等）直接 BLOCK。
  L3 商业书名交叉比对：文本内出现已知商业书名+作者组合
     且 rights != authorized，标记为可疑，需人工确认。

验证结果：VerifyResult(passed: bool, level_failed, reason)。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# L2 盗版水印/广告关键词（命中即 BLOCK）
# 注意："手机版""最新章节"单独出现是都市正文的正常词，不入
# 字面词表，收紧为下方组合正则特征（P1-2.5）。
PIRACY_WATERMARKS = [
    "笔趣阁",
    "请访问",
    "看书网",
    "顶点小说",
    "八一中文",
    "盗版必究",
    "转载请注明",
    "本书来自",
    "免费下载",
    "全文阅读请",
]

_WATERMARK_RES = [
    re.compile(re.escape(w)) for w in PIRACY_WATERMARKS
]

# L2 组合特征：盗版站广告语境（关键词 + 广告动词/名词相邻）
_WATERMARK_PATTERNS = [
    (
        re.compile(r"最新章节.{0,10}(全文|无弹窗|阅读)"),
        "最新章节+全文/无弹窗/阅读",
    ),
    (
        re.compile(r"手机版.{0,10}(阅读|小说|看书|下载)"),
        "手机版+阅读/小说/看书/下载",
    ),
]

# L3 已知商业书名（示例种子；生产可从榜单数据扩展）
# 命中书名 + rights 非 authorized → 可疑
KNOWN_COMMERCIAL_TITLES = [
    "诡秘之主",
    "斗破苍穹",
    "斗罗大陆",
    "凡人修仙传",
    "遮天",
    "完美世界",
    "大奉打更人",
    "庆余年",
    "赘婿",
    "全职高手",
]


@dataclass
class VerifyResult:
    passed: bool
    level_failed: int = 0  # 0=通过，1/2/3=对应级别失败
    reason: str = ""
    detail: str = ""


def verify_rights_declaration(
    directory: Path,
    expected_rights: str | None = None,
) -> VerifyResult:
    """L1：.rights.json 签名声明检查。

    expected_rights 非空时，声明的 rights 必须与调用方传入的
    rights 一致，防止声明一类、入库标另一类。
    """
    decl = directory / ".rights.json"
    if not decl.exists():
        return VerifyResult(
            passed=False,
            level_failed=1,
            reason="缺失 .rights.json 权利声明文件",
            detail=(
                "请在上传目录放置 .rights.json，声明 "
                '{"rights": "user_provided|authorized|open_license",'
                ' "source": "...", "declarant": "..."}'
            ),
        )
    try:
        data = json.loads(
            decl.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return VerifyResult(
            passed=False,
            level_failed=1,
            reason=".rights.json 解析失败",
            detail=str(exc),
        )
    rights = data.get("rights", "")
    if rights not in (
        "user_provided",
        "authorized",
        "open_license",
    ):
        return VerifyResult(
            passed=False,
            level_failed=1,
            reason=f"rights 类型非法: {rights!r}",
            detail="只允许 user_provided/authorized/open_license",
        )
    if expected_rights and rights != expected_rights:
        return VerifyResult(
            passed=False,
            level_failed=1,
            reason=(
                f"声明 rights({rights}) 与入库 rights"
                f"({expected_rights}) 不一致"
            ),
            detail=".rights.json 的 rights 必须与入库参数一致",
        )
    if not data.get("declarant"):
        return VerifyResult(
            passed=False,
            level_failed=1,
            reason="缺少 declarant（责任人）字段",
            detail="声明文件必须包含责任人信息",
        )
    return VerifyResult(passed=True)


def verify_no_piracy_watermark(text: str) -> VerifyResult:
    """L2：盗版水印检测。命中即 BLOCK。"""
    for pattern, word in zip(
        _WATERMARK_RES, PIRACY_WATERMARKS
    ):
        if pattern.search(text):
            return VerifyResult(
                passed=False,
                level_failed=2,
                reason=f"检测到盗版水印: {word}",
                detail="文本疑似来自盗版站点，已 BLOCK，拒绝入库",
            )
    for pattern, label in _WATERMARK_PATTERNS:
        if pattern.search(text):
            return VerifyResult(
                passed=False,
                level_failed=2,
                reason=f"检测到盗版水印组合特征: {label}",
                detail="文本疑似来自盗版站点，已 BLOCK，拒绝入库",
            )
    return VerifyResult(passed=True)


def verify_title_crosscheck(
    text: str, rights: str, title_hint: str = ""
) -> VerifyResult:
    """L3：商业书名交叉比对。

    文本内出现已知商业书名且 rights 非 authorized
    → 标记可疑（不直接 BLOCK，需人工确认）。
    """
    if rights == "authorized":
        return VerifyResult(passed=True)
    haystack = title_hint + "\n" + text[:2000]
    for known in KNOWN_COMMERCIAL_TITLES:
        if known in haystack:
            return VerifyResult(
                passed=False,
                level_failed=3,
                reason=f"疑似商业作品: 《{known}》",
                detail=(
                    "文本与已知商业书名匹配但 rights 非 "
                    "authorized，需人工确认是否为正版授权"
                ),
            )
    return VerifyResult(passed=True)


def verify_document(
    directory: Path, text: str, rights: str, title_hint: str = ""
) -> VerifyResult:
    """三级验证流水线：L1 → L2 → L3，任一失败即返回。"""
    for check in (
        lambda: verify_rights_declaration(
            directory, expected_rights=rights
        ),
        lambda: verify_no_piracy_watermark(text),
        lambda: verify_title_crosscheck(
            text, rights, title_hint
        ),
    ):
        result = check()
        if not result.passed:
            return result
    return VerifyResult(passed=True)
