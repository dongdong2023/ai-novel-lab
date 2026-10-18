"""跨来源热度 v3：Gemini 评审 P1-2 改进。

诊断：原 v2 只有"读数/排名/板块数"静态分量，缺"题材生命
周期"视角——同样热度下，30 万字的新书和 500 万字的老书
含金量完全不同；飙升榜/新书榜的 velocity 信号被平均掉。

v3 四维评分：
  score = 0.20×board + 0.35×rank + 0.25×velocity + 0.20×trend

  - board:    板块覆盖数归一（跨平台可比的"广度"）
  - rank:     1/best_rank 归一（"头部强度"）
  - velocity: 增长率归一；飙升榜/新书榜的书 ×1.25
              （"上升动能"，榜单类型加权）
  - trend:    TrendScore = heat / words^0.65 归一
              （"单位字数热度"，消除字数马太效应）

  最终 × stage_multiplier（题材生命周期）：
    ≤30万字  → 1.35（上升期，值得学）
    ≤150万字 → 1.00（成熟期）
    ＞150万字 → 0.75（长尾期，热度含存量水分）
"""

from __future__ import annotations

import math

# 四维权重（和为 1）
W_BOARD = 0.20
W_RANK = 0.35
W_VELOCITY = 0.25
W_TREND = 0.20

# TrendScore 指数：heat / words^0.65
TREND_EXPONENT = 0.65

# 题材生命周期乘数（按字数）
STAGE_SMALL_MAX = 300_000
STAGE_MID_MAX = 1_500_000
STAGE_MULT_SMALL = 1.35
STAGE_MULT_MID = 1.00
STAGE_MULT_LONG = 0.75

# velocity 加权的榜单类型关键词（飙升/新书 = 上升动能强）
VELOCITY_BOOST_BOARDS = ("飙升", "新书", "上升", "热搜")
VELOCITY_BOOST = 1.25


def minmax(values: list[float]) -> dict[float, float]:
    if not values:
        return {}
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return {v: 1.0 for v in values}
    return {v: (v - lo) / (hi - lo) for v in values}


def stage_multiplier(word_count: int | float | None) -> float:
    """题材生命周期乘数。"""
    words = word_count or 0
    if words <= STAGE_SMALL_MAX:
        return STAGE_MULT_SMALL
    if words <= STAGE_MID_MAX:
        return STAGE_MULT_MID
    return STAGE_MULT_LONG


def trend_score(heat: float, word_count: int | float | None) -> float:
    """单位字数热度：heat / words^0.65。

    heat 取 max_read（在读数）；words 缺失时按 30 万字计
    （取小字数档，避免缺失值被惩罚）。
    """
    words = word_count or STAGE_SMALL_MAX
    words = max(words, 1)
    return heat / math.pow(words, TREND_EXPONENT)


def _velocity_boost(row: dict) -> float:
    """飙升/新书榜的 velocity 加权。"""
    board_name = (
        row.get("board_name")
        or row.get("board")
        or row.get("rank_type")
        or ""
    )
    for kw in VELOCITY_BOOST_BOARDS:
        if kw in board_name:
            return VELOCITY_BOOST
    return 1.0


def popularity_scores(
    rows: list[dict],
    *,
    board_weight: float = W_BOARD,
    rank_weight: float = W_RANK,
    velocity_weight: float = W_VELOCITY,
    trend_weight: float = W_TREND,
) -> list[dict]:
    """跨来源热度 v3（四维 + 生命周期乘数）。"""

    board_norm = minmax(
        [float(r.get("board_count", 0)) for r in rows]
    )
    rank_norm = minmax(
        [
            1.0 / (r.get("best_rank") or 10**6)
            for r in rows
        ]
    )

    # velocity：增长率归一 × 榜单类型加权
    raw_velocity = []
    for r in rows:
        g = r.get("growth_rate")
        base = g if g is not None else 0.0
        raw_velocity.append(base * _velocity_boost(r))
    velocity_norm = minmax(raw_velocity)

    # trend：TrendScore 归一
    raw_trend = [
        trend_score(
            r.get("max_read") or 0,
            r.get("word_count"),
        )
        for r in rows
    ]
    trend_norm = minmax(raw_trend)

    for row in rows:
        parts: list[tuple[float, float]] = [
            (
                board_norm.get(
                    float(row.get("board_count", 0)), 0.0
                ),
                board_weight,
            ),
            (
                rank_norm.get(
                    1.0 / (row.get("best_rank") or 10**6),
                    0.0,
                ),
                rank_weight,
            ),
            (
                velocity_norm.get(
                    (
                        (row.get("growth_rate") or 0.0)
                        * _velocity_boost(row)
                    ),
                    0.0,
                ),
                velocity_weight,
            ),
            (
                trend_norm.get(
                    trend_score(
                        row.get("max_read") or 0,
                        row.get("word_count"),
                    ),
                    0.0,
                ),
                trend_weight,
            ),
        ]

        total = sum(w for _, w in parts)
        base_score = sum(s * w for s, w in parts) / total
        mult = stage_multiplier(row.get("word_count"))
        row["popularity_score"] = round(
            base_score * mult, 6
        )
        row["source_count"] = row.get("board_count", 0)
        row["trend_score"] = round(
            trend_score(
                row.get("max_read") or 0,
                row.get("word_count"),
            ),
            6,
        )
        row["stage_multiplier"] = mult

    return sorted(
        rows,
        key=lambda r: r["popularity_score"],
        reverse=True,
    )
