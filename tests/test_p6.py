import unittest

from novel_lab.knowledge_extract import STATS_EXTRACTOR
from novel_lab.knowledge_graph import (
    _base_name,
    character_index,
    detect_gaps,
    relationship_edges,
    rule_conflicts,
)


def char(name, ch, mentions=None, extractor=None, **kw):
    payload = {"name": name, "_ch": ch, "_kind": "character"}
    if mentions is not None:
        payload["mentions"] = mentions
    payload["_extractor"] = (
        extractor or STATS_EXTRACTOR
    )
    payload.update(kw)
    return payload


def rel(a, b, label, ch):
    return {
        "from": a,
        "to": b,
        "relation": label,
        "_ch": ch,
        "_kind": "relationship",
    }


class BaseNameTest(unittest.TestCase):

    def test_variants(self):
        self.assertEqual(
            _base_name("袁经理（袁立军）"), "袁立军"
        )
        self.assertEqual(_base_name("林晚"), "林晚")
        self.assertEqual(_base_name("沈确(A)"), "A")


class CharacterIndexTest(unittest.TestCase):

    def test_merge_dict_and_llm(self):
        knowledge = [
            char("林晚", 1, mentions=10),
            char("林晚", 2, mentions=8),
            char(
                "林晚",
                3,
                identity="验光师",
                traits=["谨慎"],
                extractor="llm-v1",
            ),
            char("袁经理（袁立军）", 2, mentions=5),
            char("袁立军", 4, mentions=20),
        ]
        index = character_index(knowledge)

        self.assertEqual(len(index), 2)
        lin = index["林晚"]
        self.assertEqual(lin["mentions"], 18)
        self.assertEqual(len(lin["chapters"]), 3)
        self.assertIn("验光师", lin["identities"])
        # 括注变体并入本名
        self.assertEqual(index["袁立军"]["mentions"], 25)

    def test_prefix_merge_conservative(self):
        knowledge = [
            char("李哥", 1, mentions=30),
            char("李哥的", 2, extractor="llm-v1"),
        ]
        index = character_index(knowledge)
        # "李哥的"以"李哥"开头且"李哥"是词典实体 -> 合并
        self.assertIn("李哥", index)


class GapsTest(unittest.TestCase):

    def test_rules_fire(self):
        knowledge = [
            # 主角：1-10章出现但5-8断档4章 -> R1
            *[
                char("主角", c, mentions=10)
                for c in (1, 2, 3, 4, 9, 10)
            ],
            # 早期沉底：只在2-3章 -> R2
            char("伏笔人", 2, mentions=5),
            char("伏笔人", 3, mentions=3),
            # 悬空关系 -> R3
            rel("主角", "路人甲", "交易", 3),
        ]
        index = character_index(knowledge)
        findings = detect_gaps(knowledge, index, 20)

        rules = {f["rule"] for f in findings}
        self.assertIn("R1主角断档", rules)
        self.assertIn("R2早期沉底", rules)
        self.assertIn("R3悬空关系", rules)

    def test_no_false_gap_for_steady_character(self):
        knowledge = [
            char("主角", c, mentions=10)
            for c in range(1, 11)
        ]
        index = character_index(knowledge)
        findings = detect_gaps(knowledge, index, 10)
        self.assertEqual(findings, [])


class RuleConflictsTest(unittest.TestCase):

    def test_parse_and_filter(self):
        class FakeClient:
            def chat(self, system, user, *, purpose):
                return (
                    '{"conflicts": ['
                    '{"pair": "R3×R17", "reason": "扣光速度矛盾"}, '
                    '{"pair": "随便", "reason": "格式错"}, '
                    '{"pair": "R1×R2", "reason": ""}]}'
                )

        knowledge = [
            {
                "_kind": "world_rule",
                "rule": f"规则{i}",
                "_ch": i,
            }
            for i in range(1, 20)
        ]
        result = rule_conflicts(knowledge, FakeClient())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["pair"], "R3×R17")


if __name__ == "__main__":
    unittest.main()
