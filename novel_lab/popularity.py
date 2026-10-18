from __future__ import annotations

import math


def minmax(values: list[float]) -> dict[float, float]:
    if not values:
        return {}
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return {v: 1.0 for v in values}
    return {v: (v - lo) / (hi - lo) for v in values}


def _per_platform_read_norm(
    rows: list[dict],
) -> dict[str, dict[float, float]]:
    """读数分量按平台内部归一。

    不同平台的 read_count 量纲不同（番茄=在读数、起点=月票/新粉数），
    全局归一会让大数值平台垄断该分量。
    """

    groups: dict[str, list[float]] = {}
    for row in rows:
        platform = row.get("platform") or ""
        groups.setdefault(platform, []).append(
            math.log1p(row["max_read"] or 0)
        )

    return {
        platform: minmax(values)
        for platform, values in groups.items()
    }


def popularity_scores(
    rows: list[dict],
    *,
    board_weight: float = 0.15,
    read_weight: float = 0.45,
    rank_weight: float = 0.25,
    growth_weight: float = 0.15,
) -> list[dict]:
    """跨来源热度（v2）。

    score = w_board×norm(板块数) + w_read×norm(log max在读，分平台)
          + w_rank×norm(1/best_rank) + w_growth×norm(增长率)

    归一化在候选池内做；growth 分量只有 row 带 growth_rate 且
    growth_weight>0 时参与，缺行时权重重归一。某分量全体退化
    （如单平台时 board_count 恒为常数）时 minmax 自动给满分。
    """

    board_norm = minmax(
        [float(r["board_count"]) for r in rows]
    )
    read_norms = _per_platform_read_norm(rows)
    rank_norm = minmax(
        [
            1.0 / (r["best_rank"] or 10**6)
            for r in rows
        ]
    )
    growth_values = [
        r["growth_rate"]
        for r in rows
        if r.get("growth_rate") is not None
    ]
    growth_norm = minmax(growth_values)

    for row in rows:

        platform = row.get("platform") or ""

        parts: list[tuple[float, float]] = [
            (
                board_norm.get(
                    float(row["board_count"]), 0.0
                ),
                board_weight,
            ),
            (
                read_norms.get(platform, {}).get(
                    math.log1p(row["max_read"] or 0),
                    0.0,
                ),
                read_weight,
            ),
            (
                rank_norm.get(
                    1.0 / (row["best_rank"] or 10**6), 0.0
                ),
                rank_weight,
            ),
        ]

        growth = row.get("growth_rate")
        if growth_weight > 0 and growth is not None:
            parts.append(
                (
                    growth_norm.get(growth, 0.0),
                    growth_weight,
                )
            )

        total = sum(w for _, w in parts)
        row["popularity_score"] = round(
            sum(s * w for s, w in parts) / total, 6
        )
        row["source_count"] = row["board_count"]

    return sorted(
        rows,
        key=lambda r: r["popularity_score"],
        reverse=True,
    )
