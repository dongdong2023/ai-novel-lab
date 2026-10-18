import tempfile
import unittest
from pathlib import Path

from novel_lab import db, popularity
from novel_lab.discovery.qidian import (
    QidianAdapter,
    _parse_count,
    discover_qidian,
)
from novel_lab.models import NovelRecord

FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "qidian_yuepiao.html"
)


class ParseCountTest(unittest.TestCase):

    def test_variants(self):
        self.assertEqual(
            _parse_count("8.45万月票"), 84500.0
        )
        self.assertEqual(_parse_count("9608"), 9608.0)
        self.assertEqual(
            _parse_count("415.98万字"), 4159800.0
        )
        self.assertIsNone(_parse_count(None))
        self.assertIsNone(_parse_count(""))


class QidianAdapterTest(unittest.TestCase):

    def test_parse_fixture(self):

        adapter = QidianAdapter(
            "yuepiao", board_name="月票榜"
        )

        records = adapter.parse(
            FIXTURE.read_text(encoding="utf-8")
        )

        self.assertEqual(len(records), 20)

        first = records[0]

        self.assertEqual(first.platform, "qidian")
        self.assertEqual(first.title, "夜无疆")
        self.assertEqual(first.author, "辰东")
        self.assertEqual(first.board, "qd_yuepiao")
        self.assertEqual(first.gender, "男频")
        self.assertEqual(first.rank_type, "月票榜")
        self.assertEqual(
            first.category_name, "玄幻/东方玄幻"
        )
        self.assertEqual(first.rank, 1)
        self.assertEqual(first.read_count, 84500.0)
        self.assertEqual(first.word_count, 4159800)
        self.assertFalse(first.title_obfuscated)
        self.assertIn("1040765595", first.url)

    def test_missing_context_raises(self):

        adapter = QidianAdapter("yuepiao")

        with self.assertRaises(ValueError):
            adapter.parse("<html></html>")


class GrowthTest(unittest.TestCase):

    def test_latest_growth(self):

        with tempfile.TemporaryDirectory() as td:

            conn = db.connect(Path(td) / "t.db")

            base = dict(
                platform="fanqie",
                title="书",
                book_id="1",
                board="1_2_8",
                rank=1,
            )

            # 期1：在读 100000；期2：在读 101000
            db.save_discovery(
                conn,
                [
                    NovelRecord(
                        **base,
                        read_count=100000.0,
                        discovered_at=(
                            "2026-10-16T08:00:00+00:00"
                        ),
                    )
                ],
            )
            db.save_discovery(
                conn,
                [
                    NovelRecord(
                        **base,
                        read_count=101000.0,
                        discovered_at=(
                            "2026-10-17T08:00:00+00:00"
                        ),
                    )
                ],
            )

            growth = db.latest_growth(conn)

            self.assertIn(1, growth)
            self.assertAlmostEqual(
                growth[1], 0.01, places=6
            )

            conn.close()


class PlatformPopularityTest(unittest.TestCase):

    def test_mixed_platforms_no_crash(self):

        rows = [
            {
                "novel_rowid": 1,
                "title": "番茄书",
                "platform": "fanqie",
                "board_count": 1,
                "max_read": 2000000,
                "best_rank": 1,
                "growth_rate": 0.001,
            },
            {
                "novel_rowid": 2,
                "title": "起点书",
                "platform": "qidian",
                "board_count": 3,
                "max_read": 84500,
                "best_rank": 1,
                "growth_rate": None,
            },
            {
                "novel_rowid": 3,
                "title": "起点无量纲书",
                "platform": "qidian",
                "board_count": 1,
                "max_read": None,
                "best_rank": 5,
                "growth_rate": None,
            },
        ]

        scored = popularity.popularity_scores(rows)

        self.assertEqual(len(scored), 3)

        for r in scored:
            self.assertGreaterEqual(
                r["popularity_score"], 0.0
            )
            self.assertLessEqual(
                r["popularity_score"], 1.0
            )

        # 起点书在 3 个板块出现，板块分量应给它加分
        by_title = {r["title"]: r for r in scored}
        self.assertGreater(
            by_title["起点书"]["popularity_score"],
            by_title["起点无量纲书"]["popularity_score"],
        )


if __name__ == "__main__":
    unittest.main()
