import tempfile
import unittest
from pathlib import Path

from novel_lab import db, ingest
from novel_lab.knowledge_extract import STATS_EXTRACTOR
from novel_lab.quality_gate import (
    check_fact,
    run_gate,
    split_dataset,
)

CHAPTER = (
    "第一章 试炼\n\n"
    "林晚站在门口。她看见沈确提着箱子过来。\n\n"
    "“进来。”沈确说。\n\n"
    "林晚点头。李哥在楼下望风。\n\n"
    "夜里，她数了三遍米袋，记下光值，锁进铁皮箱。\n\n"
    "第二天她照常出门验货，月亮很淡。"
)


def _setup(td):
    conn = db.connect(Path(td) / "t.db")
    nid = ingest.register_local_novel(conn, "双月降临")
    ch = Path(td) / "ch1.txt"
    ch.write_text(CHAPTER, encoding="utf-8")
    ingest.ingest_document(
        conn,
        novel_rowid=nid,
        source_url="file:///ch1",
        source_type="user_text",
        rights="user_provided",
        title="ch1_deai",
        content_hash="h1",
        raw_path=str(ch),
    )
    conn.execute(
        "INSERT INTO knowledge (novel_rowid, document_id,"
        " kind, payload, confidence, extractor,"
        " prompt_version, created_at)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (
            nid,
            1,
            "world_rule",
            '{"rule": "月光充能", "quote": "记下光值"}',
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
            1,
            "character",
            '{"name": "林晚", "chapter": "ch1_deai",'
            ' "mentions": 3}',
            1.0,
            STATS_EXTRACTOR,
            STATS_EXTRACTOR,
            "2026-10-18T00:00:00",
        ),
    )
    conn.commit()
    return conn


def world_sample(output=None):
    return {
        "category": "world",
        "instruction": "这条规则的原文依据是什么？",
        "input": "月光充能",
        "output": output or '月光充能\n原文依据：“记下光值”',
        "source_doc": "ch1_deai",
        "knowledge_id": 1,
    }


def writing_sample():
    return {
        "category": "writing",
        "instruction": "请以相同文风续写。",
        "input": "第一章 试炼\n\n林晚站在门口。她看见沈确提着箱子过来。",
        "output": "“进来。”沈确说。\n\n林晚点头。李哥在楼下望风。",
        "source_doc": "ch1_deai",
        "knowledge_id": None,
    }


class FactCheckTest(unittest.TestCase):

    def test_quote_must_match(self):

        with tempfile.TemporaryDirectory() as td:
            conn = _setup(td)

            # 正版样本：引用一致 -> 通过
            self.assertIsNone(
                check_fact(world_sample(), conn)
            )

            # 造假样本：quote 被篡改 -> 拦下
            bad = world_sample(
                output="月光充能\n原文依据：“编造的引用”"
            )
            err = check_fact(bad, conn)
            self.assertIsNotNone(err)
            self.assertIn("引用", err)

            # 写作切片连续 -> 通过
            self.assertIsNone(
                check_fact(writing_sample(), conn)
            )

            # 写作切片断裂 -> 拦下
            broken = writing_sample()
            broken["input"] = (
                "第一章 试炼\n\n林晚站在门口。"
                "她看见沈确提着箱子过来。\n\n“进来。”沈确说。"
            )
            broken["output"] = "夜里，她数了三遍米袋，记下光值，锁进铁皮箱。"
            # input末段=“进来。”沈确说。 output首段=夜里… —— 原文里中间隔着"林晚点头…"？不，原文顺序：门口/进来/点头/夜里
            # input跳过了"林晚点头。李哥在楼下望风。" -> 不连续
            err = check_fact(broken, conn)
            self.assertIsNotNone(err)

            conn.close()

    def test_mentions_number_verified(self):

        with tempfile.TemporaryDirectory() as td:
            conn = _setup(td)
            sample = {
                "category": "qa",
                "instruction": (
                    "《双月降临》前20章中，林晚主要出现在哪些章节？"
                ),
                "input": "",
                "output": "林晚出现于第一章，共提及3次。",
                "source_doc": "aggregate",
                "knowledge_id": None,
            }
            self.assertIsNone(check_fact(sample, conn))
            sample["output"] = "林晚出现于第一章，共提及9次。"
            err = check_fact(sample, conn)
            self.assertIsNotNone(err)
            conn.close()


class GateAndSplitTest(unittest.TestCase):

    def test_gate_end_to_end(self):

        with tempfile.TemporaryDirectory() as td:
            conn = _setup(td)
            samples = [
                world_sample(),  # 好
                world_sample(),  # 完全重复
                world_sample(
                    output="短"
                ),  # 格式失败(output过短)
                writing_sample(),  # 好
                {
                    "category": "qa",
                    "instruction": "x?",
                    "input": "",
                    "output": "来源不存在的回答内容",
                    "source_doc": "ghost_doc",
                    "knowledge_id": None,
                },  # 来源失败
            ]
            kept, dropped, report = run_gate(
                conn, samples, "双月降临"
            )
            conn.close()

            self.assertEqual(len(kept), 2)
            self.assertEqual(len(dropped), 3)
            reasons = " ".join(
                r["drop_reason"] for r in dropped
            )
            self.assertIn("重复", reasons)
            self.assertIn("格式", reasons)
            self.assertIn("来源", reasons)

    def test_split_stratified(self):
        samples = [
            {"category": "qa", "instruction": f"q{i}", "output": "回答" * 10}
            for i in range(10)
        ] + [
            {"category": "world", "instruction": f"w{i}", "output": "规则" * 10}
            for i in range(4)
        ]
        splits = split_dataset(samples)
        total = sum(len(v) for v in splits.values())
        self.assertEqual(total, 14)
        self.assertEqual(len(splits["train"]) + len(splits["validation"]) + len(splits["test"]), 14)
        # 分层：world 至少各留1
        cats_val = {s["category"] for s in splits["validation"]}
        cats_test = {s["category"] for s in splits["test"]}
        self.assertIn("world", cats_val)
        self.assertIn("world", cats_test)


if __name__ == "__main__":
    unittest.main()
