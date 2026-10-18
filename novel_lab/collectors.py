from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

import requests

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0.0.0 Safari/537.36"
)

# 书页元数据采集：只读公开页面的简介/分类/字数/热度等元数据，
# 不碰章节正文。全部标 rights=metadata_only（可进知识库，不可进训练集）。
#
# 附带发现（2026-10-16）：番茄书页 INITIAL_STATE 里的书名/简介是
# 干净文本，没有榜单页那种字体反爬乱码。


def _fetch(url: str, timeout: int, user_agent: str) -> str:
    response = requests.get(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept-Language": "zh-CN,zh;q=0.9",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    response.encoding = (
        response.apparent_encoding or response.encoding
    )
    return response.text


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def fanqie_book_profile(
    book_id: str,
    *,
    timeout: int = 15,
    user_agent: str = DEFAULT_USER_AGENT,
) -> dict:
    html = _fetch(
        f"https://fanqienovel.com/page/{book_id}",
        timeout,
        user_agent,
    )

    match = re.search(
        r"window\.__INITIAL_STATE__\s*=\s*", html
    )
    if not match:
        raise ValueError("书页缺少 __INITIAL_STATE__")

    tail = html[match.end():].replace(":undefined", ":null")
    state, _ = json.JSONDecoder().raw_decode(tail)
    page = state.get("page") or {}

    title = str(page.get("bookName") or "").strip()
    if not title:
        raise ValueError("书页缺 bookName，结构可能已变化")

    category = ""
    category_desc = ""
    cats = page.get("categoryV2") or []
    if isinstance(cats, str):
        # categoryV2 是双重编码的 JSON 字符串
        try:
            cats = json.loads(cats)
        except json.JSONDecodeError:
            cats = []
    if isinstance(cats, list) and cats:
        first = cats[0]
        if isinstance(first, dict):
            category = str(first.get("Name") or "")
            category_desc = str(
                first.get("ExternalDesc") or ""
            )

    last_pub = _safe_int(page.get("lastPublishTime"))

    return {
        "platform": "fanqie",
        "book_id": book_id,
        "title": title,
        "author": str(page.get("author") or ""),
        "category": category,
        "category_desc": category_desc,
        "abstract": str(page.get("abstract") or ""),
        "word_count": _safe_int(page.get("wordNumber")),
        "read_count": _safe_int(page.get("readCount")),
        "creation_status": _safe_int(
            page.get("creationStatus")
        ),
        "last_chapter_title": str(
            page.get("lastChapterTitle") or ""
        ),
        "last_publish_at": (
            datetime.fromtimestamp(
                last_pub, tz=timezone.utc
            ).isoformat()
            if last_pub
            else ""
        ),
        "source_url": (
            f"https://fanqienovel.com/page/{book_id}"
        ),
        "fetched_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }


def qidian_book_profile(
    book_id: str,
    *,
    timeout: int = 15,
    user_agent: str = DEFAULT_USER_AGENT,
) -> dict:
    html = _fetch(
        f"https://m.qidian.com/book/{book_id}/",
        timeout,
        user_agent,
    )

    match = re.search(
        r'<script[^>]*>\s*(\{"pageContext":[\s\S]*?)</script>',
        html,
    )
    if not match:
        raise ValueError("书页缺少 pageContext")

    data, _ = json.JSONDecoder().raw_decode(
        match.group(1).strip()
    )
    page_data = (
        data["pageContext"]
        .get("pageProps", {})
        .get("pageData", {})
    )
    info = page_data.get("bookInfo") or {}

    title = str(info.get("bookName") or "").strip()
    if not title:
        raise ValueError("书页缺 bookName，结构可能已变化")

    tags = []
    book_tag = info.get("bookTag")
    if isinstance(book_tag, str) and book_tag:
        tags = [
            t.strip()
            for t in re.split(r"[,，;；|]", book_tag)
            if t.strip()
        ]

    category = str(info.get("chanName") or "")
    sub = str(info.get("subCateName") or "")
    if sub:
        category = f"{category}·{sub}" if category else sub

    month_ticket = (
        page_data.get("monthTicketInfo", {}) or {}
    ).get("monthCnt")

    return {
        "platform": "qidian",
        "book_id": book_id,
        "title": title,
        "author": str(info.get("authorName") or ""),
        "category": category,
        "category_desc": "",
        "abstract": str(info.get("desc") or "").strip(),
        "word_count": _safe_int(
            info.get("showWordsCnt")
        ),
        "read_count": None,
        "month_ticket": _safe_int(month_ticket),
        "collect": _safe_int(info.get("collect")),
        "total_recommend": _safe_int(
            info.get("recomAll")
        ),
        "click_total": _safe_int(info.get("clickTotal")),
        "chapter_count": _safe_int(
            page_data.get("cTCnt")
        ),
        "creation_status": _safe_int(
            info.get("bookStatus")
        ),
        "tags": tags,
        "source_url": (
            f"https://m.qidian.com/book/{book_id}/"
        ),
        "fetched_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }


def fetch_profile(
    platform: str,
    book_id: str,
    *,
    timeout: int = 15,
    user_agent: str = DEFAULT_USER_AGENT,
) -> dict:
    if platform == "fanqie":
        return fanqie_book_profile(
            book_id, timeout=timeout, user_agent=user_agent
        )
    if platform == "qidian":
        return qidian_book_profile(
            book_id, timeout=timeout, user_agent=user_agent
        )
    raise ValueError(f"未知平台: {platform}")
