from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from typing import Any

import requests

from novel_lab.models import NovelRecord

# 起点 PC 站 (www.qidian.com) 有 JS 反爬挑战（HTTP 202 + probe.js），
# 不绕它；m 站 (m.qidian.com) 榜单页服务端直出 pageContext，可直接读。
# 女生网 m.qdmm.com 同构但 2026-10-16 探测被 WAF 拦，暂不接入。
# 书名/作者在 m 站是干净文本，无番茄那种字体反爬。
#
# read_count 字段语义随板块不同：月票榜=月票数、书友榜=新增粉丝数，
# 阅读指数/签约榜无数值指标（None）。跨平台比较由 popularity 分平台
# 归一化处理。

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0.0.0 Safari/537.36"
)

_PAGE_CTX_RE = re.compile(
    r'<script[^>]*>\s*(\{"pageContext":[\s\S]*?)</script>'
)

_CNT_RE = re.compile(r"([\d.]+)\s*(万)?")

QIDIAN_BOARDS = [
    {"slug": "yuepiao", "name": "月票榜"},
    {"slug": "readindex", "name": "阅读指数榜"},
    {"slug": "sign", "name": "签约榜"},
    {"slug": "newfans", "name": "书友榜"},
]


def _parse_count(value: Any) -> float | None:
    """'8.45万月票'->84500.0；'9608'->9608.0；'415.98万字'->4159800.0。"""
    if value in (None, ""):
        return None
    match = _CNT_RE.search(str(value))
    if not match:
        return None
    number = float(match.group(1))
    if match.group(2):
        number *= 10000
    return number


def extract_page_context(html: str) -> dict:
    match = _PAGE_CTX_RE.search(html)
    if not match:
        raise ValueError(
            "页面里找不到 pageContext，起点m站结构可能已变化"
        )
    data, _ = json.JSONDecoder().raw_decode(
        match.group(1).strip()
    )
    return data["pageContext"]


class QidianAdapter:
    """起点m站榜单适配器：只读公开榜单元数据。"""

    platform = "qidian"

    def __init__(
        self,
        slug: str,
        board_name: str = "",
        timeout: int = 15,
        user_agent: str = DEFAULT_USER_AGENT,
        client=None,
    ):
        self.slug = slug
        self.board_name = board_name or slug
        self.url = f"https://m.qidian.com/rank/{slug}/"
        self.timeout = timeout
        self.headers = {
            "User-Agent": user_agent,
            "Accept-Language": "zh-CN,zh;q=0.9",
        }
        self.board = f"qd_{slug}"
        # 注入 PoliteClient 时抓取统一走礼貌通道；不注入保持直连
        self.client = client

    def fetch(self) -> str:
        if self.client is not None:
            return self.client.get(self.url)
        response = requests.get(
            self.url,
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        response.encoding = (
            response.apparent_encoding or response.encoding
        )
        return response.text

    def parse(self, html: str) -> list[NovelRecord]:

        context = extract_page_context(html)

        page_data = (
            context.get("pageProps", {}).get("pageData", {})
        )
        records = page_data.get("records") or []

        if not records:
            raise ValueError(
                "pageData.records 为空，榜单结构可能已变化"
            )

        discovered_at = datetime.now(
            timezone.utc
        ).isoformat()

        out: list[NovelRecord] = []

        for book in records:

            book_id = str(book.get("bid") or "").strip()
            title = str(book.get("bName") or "").strip()

            if not book_id or not title:
                continue

            word_count = _parse_count(book.get("cnt"))

            rank_num = book.get("rankNum")
            try:
                rank = int(rank_num) if rank_num else None
            except (TypeError, ValueError):
                rank = None

            out.append(
                NovelRecord(
                    platform=self.platform,
                    title=title,
                    author=str(book.get("bAuth") or "").strip(),
                    book_id=book_id,
                    board=self.board,
                    board_name=f"起点/{self.board_name}",
                    gender="男频",
                    rank_type=self.board_name,
                    category_name=(
                        f"{book.get('cat') or ''}/"
                        f"{book.get('subCat') or ''}"
                    ),
                    rank=rank,
                    read_count=_parse_count(
                        book.get("rankCnt")
                    ),
                    word_count=(
                        int(word_count)
                        if word_count
                        else None
                    ),
                    description=str(
                        book.get("desc") or ""
                    )[:600],
                    url=(
                        "https://m.qidian.com/book/"
                        f"{book_id}/"
                    ),
                    discovered_at=discovered_at,
                    title_obfuscated=False,
                )
            )

        unique: dict[tuple[str, str], NovelRecord] = {}
        for record in out:
            unique.setdefault(
                (record.board, record.title), record
            )

        return list(unique.values())


def discover_qidian(
    *,
    timeout: int = 15,
    user_agent: str = DEFAULT_USER_AGENT,
    delay_seconds: float = 1.0,
    client=None,
) -> tuple[list[NovelRecord], list[tuple[str, str]]]:
    """轮询起点m站全部已确认板块，返回 (records, errors)。"""

    all_records: list[NovelRecord] = []
    errors: list[tuple[str, str]] = []

    for index, board in enumerate(QIDIAN_BOARDS):

        adapter = QidianAdapter(
            board["slug"],
            board_name=board["name"],
            timeout=timeout,
            user_agent=user_agent,
            client=client,
        )

        try:
            records = adapter.parse(adapter.fetch())
            all_records.extend(records)
        except Exception as exc:
            errors.append((adapter.board, str(exc)))

        if index < len(QIDIAN_BOARDS) - 1:
            time.sleep(delay_seconds)

    return all_records, errors
