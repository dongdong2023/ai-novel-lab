import tempfile
import unittest
from pathlib import Path

from novel_lab import db, ingest
from novel_lab.dataset_factory import (
    build_dataset,
    samples_writing,
    samples_world,
)
from novel_lab.knowledge_extract import STATS_EXTRACTOR
from novel_lab.understanding.pipeline import (
    UnderstandingPipeline,
)

CHAPTER = (
    "第一章 试炼\n\n"
    "林晚站在门口。她看见沈确提着箱子过来。\n\n"
    "“进来。”沈确说。\n\n"
    "林晚点头。李哥在楼下望风。\n\n"
    "夜里，她数了三遍米袋，记下光值。\n\n"
    "第二天，雨停了。她照常出门验货。\n\n"
    "街口的小卖部开了门，老板娘朝她点头。\n\n"
    "回到铺子，她把账目誊清，锁进铁皮箱。\n\n"
    "窗外的月亮升起来，很淡。\n\n"
    "她拉上窗帘，灭灯，睡了。"
)


def _setup(td):
    conn = db.connect(Path(td) / "t.db")
    nid = ingest.register_local_novel(conn, "测试书")

    ch = Path(td) / "ch1.txt"
    ch.write_text(CHAPTER, encoding="utf-8")
    ingest.ingest_document(
        conn,
        novel_rowid=nid,
        source_url="file:///ch1",
        source_type="user_text",
        rights="user_provided",
        title="ch1",
        content_hash="h1",
        raw_path=str(ch),
    )

    # 一份 metadata_only 资料（不得进入数据集）
    meta = Path(td) / "p.json"
    meta.write_text('{"title": "市场书"}', encoding="utf-8")
    meta_novel = ingest.register_local_novel(conn, "市场书")
    ingest.ingest_document(
        conn,
        novel_rowid=meta_novel,
        source_url="file:///p",
        source_type="book_profile",
        rights="metadata_only",
        title="市场书·书页",
        content_hash="hp",
        raw_path=str(meta),
    )

    # 给测试书过预筛（mock）并塞两条知识
    UnderstandingPipeline(conn, None).run(mock=True)

    conn.execute(
        "INSERT INTO knowledge (novel_rowid, document_id,"
        " kind, payload, confidence, extractor,"
        " prompt_version, created_at)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (
            nid,
            1,
            "world_rule",
            '{"rule": "月光才能充能", "quote": "夜里，她数了三遍米袋"}',
            0.7,
            "llm-v1",
            "t",
            "2026-10-18T00:00:00",
        ),
    )
    conn.execute(
        "INSERT INTO knowledge (novel_rowid, document_id,"
        " kind, payload, confidence, extractor,"
        " prompt_version, created_at)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (
            nid,
            2,
            "world_rule",
            '{"rule": "市场书规则", "quote": "x"}',
            0.7,
            "llm-v1",
            "t",
            "2026-10-18T00:00:00",
        ),
    )
    # 词典人物（确定性统计）+ LLM 人物，供 qa/character 样本
    conn.execute(
        "INSERT INTO knowledge (novel_rowid, document_id,"
        " kind, payload, confidence, extractor,"
        " prompt_version, created_at)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (
            nid,
            1,
            "character",
            '{"name": "林晚", "chapter": "ch1", "mentions": 3}',
            1.0,
            STATS_EXTRACTOR,
            STATS_EXTRACTOR,
            "2026-10-18T00:00:00",
        ),
    )
    conn.execute(
        "INSERT INTO knowledge (novel_rowid, document_id,"
        " kind, payload, confidence, extractor,"
        " prompt_version, created_at)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (
            nid,
            1,
            "character",
            '{"name": "沈确", "identity": "情报搭档",'
            ' "traits": ["话少"], "quote": "沈确提着箱子过来"}',
            0.7,
            "llm-v1",
            "t",
            "2026-10-18T00:00:00",
        ),
    )
    conn.commit()
    return conn


class DatasetTest(unittest.TestCase):

    def test_rights_gate_and_categories(self):

        with tempfile.TemporaryDirectory() as td:
            conn = _setup(td)
            samples, stats = build_dataset(conn, "测试书")
            conn.close()

            cats = {s["category"] for s in samples}
            self.assertIn("qa", cats)  # 人物出场
            self.assertIn("character", cats)
            self.assertIn("world", cats)
            self.assertIn("writing", cats)

            # Rights Gate：metadata_only 的知识绝不出现
            self.assertFalse(
                any(
                    "市场书" in s["output"]
                    or "市场书" in s["instruction"]
                    for s in samples
                )
            )

            # world 样本带原文引用
            world = [
                s for s in samples if s["category"] == "world"
            ]
            self.assertEqual(len(world), 1)
            self.assertIn("夜里，她数了三遍米袋", world[0]["output"])

            self.assertGreater(stats["total"], 0)


class WritingSliceTest(unittest.TestCase):

    def test_continuation_slice_contiguous(self):

        with tempfile.TemporaryDirectory() as td:
            doc = {
                "id": 1,
                "title": "ch1",
                "raw_path": str(
                    Path(td) / "c.txt"
                ),
                "rights": "user_provided",
            }
            (Path(td) / "c.txt").write_text(
                CHAPTER, encoding="utf-8"
            )

            samples = samples_writing([doc])

            self.assertTrue(samples)
            for s in samples:
                self.assertTrue(s["input"])
                self.assertTrue(s["output"])
                # input 结尾和 output 开头必须来自连续段落
                self.assertIn(
                    s["output"].split("\n\n")[0][:6], CHAPTER
                )
                self.assertNotEqual(s["input"], s["output"])


if __name__ == "__main__":
    unittest.main()
