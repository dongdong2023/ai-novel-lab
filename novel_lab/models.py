from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class NovelRecord:
    """一本书在某个板块某次观察中的全部信息。

    hotness_score 是板块内部可比的热度（排名+读数归一加权）；
    popularity_score 是跨来源聚合热度（popularity.py 计算）；
    source_count 是本次运行中该书被几个独立来源/板块观察到。
    """

    platform: str
    title: str
    author: str = ""
    book_id: str = ""
    board: str = ""
    board_name: str = ""
    gender: str = ""
    rank_type: str = ""
    category_name: str = ""
    rank: Optional[int] = None
    read_count: Optional[float] = None
    word_count: Optional[int] = None
    chapter_count: Optional[int] = None
    last_chapter_title: str = ""
    # 1=连载中 0=已完结
    creation_status: Optional[int] = None
    updated_at: str = ""
    url: str = ""
    description: str = ""
    discovered_at: str = ""
    title_obfuscated: bool = False
    hotness_score: float = 0.0
    # 跨来源聚合字段
    source_count: int = 1
    popularity_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "NovelRecord":
        fields = set(cls.__dataclass_fields__)
        return cls(
            **{
                k: v
                for k, v in data.items()
                if k in fields
            }
        )
