import tempfile
import unittest
from pathlib import Path

from novel_lab import db, ingest
from novel_lab.collectors import (
    fanqie_book_profile,
    qidian_book_profile,
)

FIXTURE_FANQIE = (
    Path(__file__).parent
    / "fixtures"
    / "fanqie_book.html"
)
FIXTURE_QIDIAN = (
    Path(__file__).parent
    / "fixtures"
    / "qidian_book.html"
)


class _FakeResponse:
    def __init__(self, text):
        self.text = text
        self.apparent_encoding = "utf-8"
        self.encoding = "utf-8"

    def raise_for_status(self):
        pass


class CollectorsTest(unittest.TestCase):

    def test_fanqie_profile(self):

        import novel_lab.collectors as c

        original = c.requests.get
        c.requests.get = lambda *a, **kw: _FakeResponse(
            FIXTURE_FANQIE.read_text(encoding="utf-8")
        )
        try:
            profile = fanqie_book_profile(
                "7680148815046773822"
            )
        finally:
            c.requests.get = original

        self.assertEqual(
            profile["title"], "帝王娇宠，不做太子妃做皇后"
        )
        self.assertEqual(profile["author"], "应雪泥")
        self.assertEqual(profile["category"], "古风世情")
        self.assertEqual(profile["word_count"], 178660)
        self.assertEqual(profile["read_count"], 120258)
        self.assertEqual(
            profile["creation_status"], 1
        )
        # 书页书名不应有反爬私有区乱码
        self.assertFalse(
            any(
                0xE000 <= ord(ch) <= 0xF8FF
                for ch in profile["title"]
            )
        )

    def test_qidian_profile(self):

        import novel_lab.collectors as c

        original = c.requests.get
        c.requests.get = lambda *a, **kw: _FakeResponse(
            FIXTURE_QIDIAN.read_text(encoding="utf-8")
        )
        try:
            profile = qidian_book_profile("1040765595")
        finally:
            c.requests.get = original

        self.assertEqual(profile["title"], "夜无疆")
        self.assertEqual(profile["author"], "辰东")
        self.assertIn("玄幻", profile["category"])
        self.assertEqual(profile["month_ticket"], 84511)
        self.assertEqual(profile["chapter_count"], 852)
        self.assertTrue(profile["abstract"])


class IngestTest(unittest.TestCase):

    def _conn(self, td):
        return db.connect(Path(td) / "t.db")

    def test_rights_gate(self):

        with tempfile.TemporaryDirectory() as td:
            conn = self._conn(td)
            with self.assertRaises(ValueError):
                ingest.ingest_document(
                    conn,
                    novel_rowid=None,
                    source_url="https://x/1",
                    source_type="review",
                    rights="unknown",
                    title="t",
                )
            conn.close()

    def test_three_level_dedup(self):

        with tempfile.TemporaryDirectory() as td:
            conn = self._conn(td)

            statuses = []
            kwargs = dict(
                novel_rowid=None,
                source_type="review",
                rights="metadata_only",
                title="同一篇书评",
            )

            _, s1 = ingest.ingest_document(
                conn,
                source_url="https://x/a",
                content_hash="h1",
                **kwargs,
            )
            statuses.append(s1)

            # URL 重复
            _, s2 = ingest.ingest_document(
                conn,
                source_url="https://x/a",
                content_hash="h1",
                **kwargs,
            )
            statuses.append(s2)

            # 内容哈希重复（URL 换了）
            _, s3 = ingest.ingest_document(
                conn,
                source_url="https://x/b",
                content_hash="h1",
                **kwargs,
            )
            statuses.append(s3)

            # 无归属(novel_rowid=None)不做标题判重：
            # 不同书的同名文件不能误合并
            _, s4 = ingest.ingest_document(
                conn,
                source_url="https://x/c",
                content_hash="h2",
                **kwargs,
            )
            statuses.append(s4)

            # 同一本书内的标题判重
            nid = ingest.register_local_novel(
                conn, "测试书"
            )
            _, s5 = ingest.ingest_document(
                conn,
                novel_rowid=nid,
                source_url="https://x/d",
                source_type="review",
                rights="metadata_only",
                title="试毒报告",
                content_hash="h5",
            )
            statuses.append(s5)

            _, s6 = ingest.ingest_document(
                conn,
                novel_rowid=nid,
                source_url="https://x/e",
                source_type="review",
                rights="metadata_only",
                # 空白与标点差异，归一化后同名
                title="试毒 报告！",
                content_hash="h6",
            )
            statuses.append(s6)

            n = conn.execute(
                "SELECT COUNT(*) AS n FROM documents"
            ).fetchone()["n"]
            conn.close()

        self.assertEqual(
            statuses,
            [
                "created",
                "duplicate_url",
                "duplicate_content",
                "created",
                "created",
                "duplicate_title",
            ],
        )
        self.assertEqual(n, 3)

    def test_ingest_directory(self):

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "ch1.txt").write_text(
                "第一章内容" * 100, encoding="utf-8"
            )
            (corpus / "ch2.txt").write_text(
                "第二章内容" * 100, encoding="utf-8"
            )
            (corpus / "cover.png").write_bytes(b"x")

            conn = db.connect(root / "t.db")

            stats = ingest.ingest_directory(
                conn,
                corpus,
                novel_title="测试书",
            )

            # 重复登记：全部判重
            stats2 = ingest.ingest_directory(
                conn,
                corpus,
                novel_title="测试书",
            )

            trainable = ingest.training_ready(conn)
            conn.close()

        self.assertEqual(stats["created"], 2)
        self.assertEqual(stats2["duplicates"], 2)
        # 可训练语料计数（user_provided）
        self.assertEqual(trainable, 2)


if __name__ == "__main__":
    unittest.main()
