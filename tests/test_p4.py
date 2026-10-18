import tempfile
import unittest
from pathlib import Path

from novel_lab import db, eval_gold, ingest
from novel_lab.understanding.pipeline import (
    UnderstandingPipeline,
    extract_json,
)
from novel_lab.understanding.prompts import (
    prompt_version,
    render,
)


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, system, user, *, purpose):
        self.calls.append(purpose)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class PromptTest(unittest.TestCase):

    def test_render_keeps_json_braces(self):
        out = render(
            "prefilter",
            title="书",
            source_type="review",
            content="内容",
        )
        self.assertIn('"status"', out)
        self.assertIn("书", out)
        self.assertNotIn("{title}", out)

    def test_version_stable(self):
        self.assertEqual(
            prompt_version("prefilter"),
            prompt_version("prefilter"),
        )


class ExtractJsonTest(unittest.TestCase):

    def test_plain(self):
        self.assertEqual(
            extract_json('{"a": 1}'), {"a": 1}
        )

    def test_fenced_with_chatter(self):
        text = (
            "好的，结果如下：\n```json\n"
            '{"status": "PASS", "reason": "相关"}\n'
            "```\n以上。"
        )
        self.assertEqual(
            extract_json(text)["status"], "PASS"
        )

    def test_invalid(self):
        self.assertIsNone(extract_json("没有JSON"))
        self.assertIsNone(
            extract_json('{"broken": ')
        )


def _make_doc(td, name, content, source_type="review"):
    path = Path(td) / name
    path.write_text(content, encoding="utf-8")
    return path


class PipelineTest(unittest.TestCase):

    def _setup(self, td):
        conn = db.connect(Path(td) / "t.db")
        doc_path = _make_doc(
            td, "d1.txt", "这是一份与某小说相关的书评。"
        )
        _, status = ingest.ingest_document(
            conn,
            novel_rowid=None,
            source_url="https://x/d1",
            source_type="review",
            rights="metadata_only",
            title="某书书评",
            content_hash="h-d1",
            raw_path=str(doc_path),
        )
        assert status == "created"
        return conn

    def test_live_pass_path(self):

        with tempfile.TemporaryDirectory() as td:
            conn = self._setup(td)

            client = FakeClient(
                [
                    '{"status": "PASS", "reason": "小说相关"}',
                    '{"doc_kind": "review", "about_novel": "",'
                    ' "aspects": ["题材"], "facts": [],'
                    ' "notes": "ok"}',
                ]
            )

            pipeline = UnderstandingPipeline(conn, client)
            stats = pipeline.run()

            self.assertEqual(stats["processed"], 1)
            self.assertEqual(stats["pass"], 1)
            # 预筛 + 结构化各一次调用
            self.assertEqual(
                client.calls, ["prefilter", "structure"]
            )

            row = conn.execute(
                "SELECT * FROM doc_structures"
            ).fetchone()
            self.assertEqual(row["prefilter"], "PASS")
            self.assertEqual(row["model"], "live")
            self.assertIn("review", row["structure"])

            # 重跑：同提示词版本下无待处理
            stats2 = pipeline.run()
            self.assertEqual(stats2["processed"], 0)
            self.assertEqual(len(client.calls), 2)

            conn.close()

    def test_block_skips_structure(self):

        with tempfile.TemporaryDirectory() as td:
            conn = self._setup(td)

            client = FakeClient(
                [
                    '{"status": "BLOCK", "reason": "广告"}',
                ]
            )

            pipeline = UnderstandingPipeline(conn, client)
            stats = pipeline.run()

            self.assertEqual(stats["block"], 1)
            self.assertEqual(client.calls, ["prefilter"])
            # BLOCK 不产生结构化（省一次模型调用）

            row = conn.execute(
                "SELECT * FROM doc_structures"
            ).fetchone()
            self.assertIsNone(row["structure"])

            conn.close()

    def test_parse_error_no_crash(self):

        with tempfile.TemporaryDirectory() as td:
            conn = self._setup(td)

            client = FakeClient(
                [
                    '{"status": "PASS", "reason": "ok"}',
                    "抱歉我不知道怎么输出",
                ]
            )

            pipeline = UnderstandingPipeline(conn, client)
            stats = pipeline.run()

            self.assertEqual(stats["processed"], 1)
            self.assertEqual(stats["parse_errors"], 1)

            conn.close()

    def test_mock_mode(self):

        with tempfile.TemporaryDirectory() as td:
            conn = self._setup(td)

            pipeline = UnderstandingPipeline(conn, None)
            stats = pipeline.run(mock=True)

            self.assertEqual(stats["pass"], 1)

            row = conn.execute(
                "SELECT * FROM doc_structures"
            ).fetchone()
            self.assertEqual(row["is_mock"], 1)
            self.assertEqual(row["model"], "mock")
            self.assertIn(
                "review", row["structure"]
            )

            conn.close()


class EvalGoldTest(unittest.TestCase):

    def test_seed_gold_mock_eval(self):

        gold = eval_gold.load_gold()
        self.assertEqual(len(gold), 8)

        with tempfile.TemporaryDirectory() as td:
            conn = db.connect(Path(td) / "t.db")
            report = eval_gold.evaluate(
                gold, conn, None, mock=True
            )
            conn.close()

        # mock 全部有效完成
        self.assertEqual(report["coverage"], 1.0)
        # mock 对两个负例都判不出（广告判成PASS、薄内容判成PASS）
        # ——校准循环正是用来暴露这种与真实模型的差距
        self.assertEqual(report["prefilter_acc"], 0.75)
        self.assertEqual(report["prefilter_n"], 8)
        # 类型映射全对
        self.assertEqual(report["doc_kind_acc"], 1.0)

    def test_live_eval_with_fake_client(self):

        gold = [
            {
                "caseId": "x1",
                "document": {
                    "title": "书评",
                    "source_type": "review",
                    "content": "主角很强",
                },
                "gold": {
                    "prefilter": "PASS",
                    "doc_kind": "review",
                },
            }
        ]

        client = FakeClient(
            [
                '{"status": "PASS", "reason": "相关"}',
                '{"doc_kind": "review", "about_novel": "",'
                ' "aspects": [], "facts": [], "notes": ""}',
            ]
        )

        with tempfile.TemporaryDirectory() as td:
            conn = db.connect(Path(td) / "t.db")
            report = eval_gold.evaluate(
                gold, conn, client, mock=False
            )
            conn.close()

        self.assertEqual(report["coverage"], 1.0)
        self.assertEqual(report["prefilter_acc"], 1.0)
        self.assertEqual(report["doc_kind_acc"], 1.0)


if __name__ == "__main__":
    unittest.main()
