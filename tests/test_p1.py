import tempfile
import unittest
from pathlib import Path

from novel_lab import db, popularity
from novel_lab.discovery.fanqie import FanqieAdapter
from novel_lab.llm.client import (
    BudgetExceededError,
    LLMClient,
    LLMDisabledError,
)
from novel_lab.models import NovelRecord

FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "fanqie_rank.html"
)


def make_record(**kwargs) -> NovelRecord:
    base = dict(
        platform="fanqie",
        title="测试书",
        book_id="123",
        board="1_2_751",
        gender="男频",
        rank_type="在读榜",
        category_name="悬疑灵异",
        rank=1,
        read_count=100000.0,
        discovered_at="2026-10-16T12:00:00+00:00",
    )
    base.update(kwargs)
    return NovelRecord(**base)


class FanqieAdapterTest(unittest.TestCase):

    def test_parse_fixture(self):

        adapter = FanqieAdapter(
            "https://fanqienovel.com/rank/0_1"
        )

        records = adapter.parse(
            FIXTURE.read_text(encoding="utf-8")
        )

        self.assertEqual(len(records), 10)

        first = records[0]

        self.assertEqual(first.rank, 1)
        self.assertEqual(first.board, "0_1")
        self.assertEqual(
            first.book_id, "7680148815046773822"
        )
        self.assertEqual(first.read_count, 120014.0)
        self.assertEqual(first.chapter_count, 85)
        self.assertEqual(first.creation_status, 1)
        self.assertTrue(first.title_obfuscated)


class DbTest(unittest.TestCase):

    def test_save_discovery_idempotent(self):

        with tempfile.TemporaryDirectory() as td:

            conn = db.connect(Path(td) / "test.db")

            records = [
                make_record(),
                make_record(
                    book_id="456",
                    title="第二本",
                    board="1_2_8",
                    rank=3,
                    read_count=50000.0,
                ),
            ]

            novels = db.save_discovery(conn, records)
            self.assertEqual(novels, 2)

            # 同一轮重复落库：书不重复，观察也不重复
            novels = db.save_discovery(conn, records)
            self.assertEqual(novels, 2)

            count = conn.execute(
                "SELECT COUNT(*) AS n FROM novels"
            ).fetchone()["n"]
            self.assertEqual(count, 2)

            obs = conn.execute(
                "SELECT COUNT(*) AS n FROM observations"
            ).fetchone()["n"]
            self.assertEqual(obs, 2)

            # 不同时间的第二次观察：observations 增长
            later = [
                make_record(
                    discovered_at=(
                        "2026-10-17T12:00:00+00:00"
                    )
                )
            ]
            db.save_discovery(conn, later)

            obs = conn.execute(
                "SELECT COUNT(*) AS n FROM observations"
            ).fetchone()["n"]
            self.assertEqual(obs, 3)

            rows = db.candidate_rows(conn)
            self.assertEqual(len(rows), 2)

            row = next(
                r for r in rows if r["book_id"] == "123"
            )
            self.assertEqual(row["board_count"], 1)
            self.assertEqual(
                row["observation_count"], 2
            )

            conn.close()


class PopularityTest(unittest.TestCase):

    def test_ordering_and_source_count(self):

        rows = [
            {
                "novel_rowid": 1,
                "title": "a",
                "board_count": 2,
                "max_read": 900000,
                "best_rank": 1,
            },
            {
                "novel_rowid": 2,
                "title": "b",
                "board_count": 1,
                "max_read": 100000,
                "best_rank": 5,
            },
            {
                "novel_rowid": 3,
                "title": "c",
                "board_count": 1,
                "max_read": None,
                "best_rank": None,
            },
        ]

        scored = popularity.popularity_scores(rows)

        self.assertEqual(scored[0]["title"], "a")
        self.assertIn("c", [r["title"] for r in scored])

        for r in scored:
            self.assertGreaterEqual(
                r["popularity_score"], 0.0
            )
            self.assertLessEqual(
                r["popularity_score"], 1.0
            )
            self.assertEqual(
                r["source_count"], r["board_count"]
            )


class LLMClientTest(unittest.TestCase):

    def test_disabled_raises(self):

        with tempfile.TemporaryDirectory() as td:
            conn = db.connect(Path(td) / "t.db")
            client = LLMClient(
                conn,
                enabled=False,
                base_url="",
                api_key="",
                model="m",
            )
            with self.assertRaises(LLMDisabledError):
                client.chat("s", "u", purpose="test")
            conn.close()

    def test_receipt_reuse(self):

        calls = []

        def fake_transport(*args):
            calls.append(args)
            return '{"ok": true}'

        with tempfile.TemporaryDirectory() as td:
            conn = db.connect(Path(td) / "t.db")
            client = LLMClient(
                conn,
                enabled=True,
                base_url="http://fake",
                api_key="k",
                model="m",
                transport=fake_transport,
            )

            first = client.chat("s", "u", purpose="p")
            second = client.chat("s", "u", purpose="p")

            self.assertEqual(first, second)
            # 回执复用：同 prompt 只发一次真实请求
            self.assertEqual(len(calls), 1)
            conn.close()

    def test_budget_breaker(self):

        with tempfile.TemporaryDirectory() as td:
            conn = db.connect(Path(td) / "t.db")
            client = LLMClient(
                conn,
                enabled=True,
                base_url="http://fake",
                api_key="k",
                model="m",
                per_minute_limit=1,
                per_hour_limit=100,
                per_day_limit=1000,
                transport=lambda *a: '{"ok": true}',
            )

            client.chat("s", "u1", purpose="p")

            with self.assertRaises(BudgetExceededError):
                client.chat("s", "u2", purpose="p")

            conn.close()


if __name__ == "__main__":
    unittest.main()
