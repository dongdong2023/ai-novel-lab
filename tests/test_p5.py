import tempfile
import unittest
from pathlib import Path

from novel_lab import db, ingest
from novel_lab.knowledge_extract import (
    character_totals,
    extract_chapter_llm,
    extract_chapter_stats,
    extract_profile_facts,
    load_character_dict,
    run_extract,
    write_knowledge,
)
from novel_lab.understanding.pipeline import (
    UnderstandingPipeline,
)


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def chat(self, system, user, *, purpose):
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


PROFILE_JSON = (
    '{"platform": "qidian", "title": "夜无疆",'
    ' "author": "辰东", "category": "玄幻·东方玄幻",'
    ' "word_count": 4159800, "month_ticket": 84511,'
    ' "chapter_count": 852, "tags": ["东方玄幻"]}'
)

CHAPTER_TEXT = (
    "第一章 试炼\n\n"
    "林晚站在门口。她看见沈确提着箱子过来。\n\n"
    "“进来。”沈确说。\n\n"
    "林晚点头。李哥在楼下望风。这一夜，林晚数了三遍米袋。"
)


def _setup_db(td):
    conn = db.connect(Path(td) / "t.db")
    nid = ingest.register_local_novel(conn, "测试书")

    profile_path = Path(td) / "p.json"
    profile_path.write_text(PROFILE_JSON, encoding="utf-8")
    ingest.ingest_document(
        conn,
        novel_rowid=nid,
        source_url="https://x/p",
        source_type="book_profile",
        rights="metadata_only",
        title="夜无疆·书页资料",
        content_hash="hp",
        raw_path=str(profile_path),
    )

    chapter_path = Path(td) / "c1.txt"
    chapter_path.write_text(CHAPTER_TEXT, encoding="utf-8")
    ingest.ingest_document(
        conn,
        novel_rowid=nid,
        source_url="file:///c1",
        source_type="user_text",
        rights="user_provided",
        title="第一章 试炼",
        content_hash="hc",
        raw_path=str(chapter_path),
    )
    return conn, nid


class ProfileFactsTest(unittest.TestCase):

    def test_extract(self):
        items = extract_profile_facts(
            {"title": "夜无疆·书页资料"}, PROFILE_JSON
        )
        kinds = [k for k, _, _ in items]
        self.assertIn("genre_fact", kinds)
        # word/month_ticket/chapter_count → 3 条 market_fact
        self.assertEqual(
            kinds.count("market_fact"), 3
        )


class ChapterStatsTest(unittest.TestCase):

    def test_style_and_characters(self):
        items = extract_chapter_stats(
            {"title": "第一章 试炼"},
            CHAPTER_TEXT,
            ["林晚", "沈确", "不存在的人"],
        )

        by_kind = {}
        for kind, payload, _ in items:
            by_kind.setdefault(kind, []).append(payload)

        style = by_kind["style_fact"][0]
        self.assertGreater(style["sentences"], 0)
        self.assertGreater(style["dialogue_ratio"], 0)

        characters = {
            p["name"]: p for p in by_kind["character"]
        }
        # 林晚×3、沈确×2、李哥×1（不在词典里不算）
        self.assertEqual(characters["林晚"]["mentions"], 3)
        self.assertEqual(characters["沈确"]["mentions"], 2)
        self.assertNotIn("不存在的人", characters)
        self.assertNotIn("李哥", characters)


class LlmExtractTest(unittest.TestCase):

    def test_with_quote_kept_without_quote_dropped(self):
        client = FakeClient(
            [
                '{"characters": ['
                '{"name": "林晚", "identity": "验光师", '
                '"traits": ["谨慎"], "quote": "林晚站在门口"}, '
                '{"name": "编的", "identity": "无据", "traits": [], '
                '"quote": ""}, '
                '{"name": "拼接的", "identity": "转述", "traits": [], '
                '"quote": "林晚站在沈确也在门口"}], '
                '"relationships": ['
                '{"from": "林晚", "relation": "合作", '
                '"to": "沈确", "quote": "提着箱子"}], '
                '"events": [], "world_rules": [], "style_notes": []}'
            ]
        )

        items = extract_chapter_llm(
            {"title": "第一章"}, CHAPTER_TEXT, client
        )

        kinds = [k for k, _, _ in items]
        # 无 quote 的"编的"与拼接转述"拼接的"都被门禁丢弃
        self.assertEqual(kinds.count("character"), 1)
        self.assertEqual(kinds.count("relationship"), 1)
        self.assertNotIn("编的", str(items))
        self.assertNotIn("拼接的", str(items))

    def test_garbage_output_returns_empty(self):
        client = FakeClient(["输出失败了一半"])
        items = extract_chapter_llm(
            {"title": "第一章"}, CHAPTER_TEXT, client
        )
        self.assertEqual(items, [])


class RunExtractTest(unittest.TestCase):

    def test_end_to_end_idempotent(self):

        with tempfile.TemporaryDirectory() as td:
            conn, nid = _setup_db(td)

            # 先跑 P4 让文档过预筛
            UnderstandingPipeline(conn, None).run(mock=True)

            stats = run_extract(conn, None, use_llm=False)
            self.assertEqual(stats["docs"], 2)
            self.assertGreater(stats["knowledge"], 0)

            # 溯源：每条知识都挂了 document
            orphans = conn.execute(
                "SELECT COUNT(*) AS n FROM knowledge k "
                "LEFT JOIN documents d ON d.id=k.document_id "
                "WHERE d.id IS NULL"
            ).fetchone()["n"]
            self.assertEqual(orphans, 0)

            # 幂等：重跑不翻倍
            count1 = conn.execute(
                "SELECT COUNT(*) AS n FROM knowledge"
            ).fetchone()["n"]
            run_extract(conn, None, use_llm=False)
            count2 = conn.execute(
                "SELECT COUNT(*) AS n FROM knowledge"
            ).fetchone()["n"]
            self.assertEqual(count1, count2)

            conn.close()


class CharacterDictTest(unittest.TestCase):

    def test_load(self):

        from novel_lab.knowledge_extract import (
            CHARACTER_DICT_DIR,
        )

        CHARACTER_DICT_DIR.mkdir(
            parents=True, exist_ok=True
        )
        path = CHARACTER_DICT_DIR / "__test书__.txt"
        path.write_text(
            "# 注释行\n张三\n\n李四\n", encoding="utf-8"
        )
        try:
            self.assertEqual(
                load_character_dict("__test书__"),
                ["张三", "李四"],
            )
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
