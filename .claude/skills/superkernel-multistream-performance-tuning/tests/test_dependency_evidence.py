# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import multistream_dependency_evidence  # noqa: E402


class DependencyEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.provider = self.root / "provider.py"
        self.provider.write_text("# reviewed dependency provider\n")
        self.graph = self.root / "graph.json"
        self.graph.write_text("{}\n")

    def tearDown(self):
        self.temporary.cleanup()

    def _value(self, *, coverage=None, edges=None, blockers=None):
        value = {
            "schema_version": multistream_dependency_evidence.FRAGMENT_SET_SCHEMA,
            "fragment_set_id": "dependencies-1",
            "request_fingerprint": "request-fp-1",
            "range_id": "range-1",
            "graph_occurrence_fingerprint": "occurrence-1",
            "statement_ids": ["vector", "cube"],
            "required_dependency_kinds": sorted(
                multistream_dependency_evidence.HARD_DEPENDENCY_KINDS
            ),
            "providers": [
                {
                    "provider_id": "source-review-1",
                    "provider_api_version": multistream_dependency_evidence.PROVIDER_API_VERSION,
                    "provider_kind": "source_review",
                    "implementation": multistream_dependency_evidence.source_file_record(
                        self.provider, self.root
                    ),
                    "input_files": [
                        multistream_dependency_evidence.source_file_record(
                            self.graph, self.root
                        )
                    ],
                    "covered_dependency_kinds": coverage
                    or sorted(multistream_dependency_evidence.HARD_DEPENDENCY_KINDS),
                    "edges": edges or [],
                    "blockers": blockers or [],
                }
            ],
        }
        value["fragment_set_fingerprint"] = multistream_dependency_evidence.fingerprint(
            value
        )
        return value

    def _build(self, value):
        path = self.root / "fragments.json"
        path.write_text(json.dumps(value) + "\n")
        return multistream_dependency_evidence.build(path, self.root)

    def test_builds_complete_audited_hard_edge_union(self):
        edge = {
            "before": "vector",
            "after": "cube",
            "kind": "DATA",
            "evidence_locator": "graph.nodes[7]",
        }
        evidence = self._build(self._value(edges=[edge]))
        self.assertTrue(evidence["complete"])
        self.assertEqual(
            evidence["hard_dependencies"],
            [{"before": "vector", "after": "cube", "kind": "DATA"}],
        )
        self.assertEqual(
            evidence["edge_sources"][0]["sources"][0]["provider_id"], "source-review-1"
        )
        path = self.root / "evidence.json"
        path.write_text(json.dumps(evidence) + "\n")
        replay = multistream_dependency_evidence.validate(
            path,
            self.root,
            require_complete=True,
            request_fingerprint="request-fp-1",
            range_id="range-1",
            graph_occurrence_fingerprint="occurrence-1",
            statement_ids=["vector", "cube"],
        )
        self.assertTrue(replay["complete"])

    def test_missing_kind_coverage_is_a_blocker(self):
        evidence = self._build(self._value(coverage=["DATA"]))
        self.assertFalse(evidence["complete"])
        blocker = next(
            item
            for item in evidence["blockers"]
            if item["code"] == "dependency_kind_coverage_missing"
        )
        self.assertIn("SIDE_EFFECT", blocker["kinds"])

    def test_provider_blocker_is_preserved(self):
        evidence = self._build(self._value(blockers=["control-flow review incomplete"]))
        self.assertFalse(evidence["complete"])
        self.assertIn(
            "provider_blocked", {item["code"] for item in evidence["blockers"]}
        )

    def test_cycle_is_a_blocker(self):
        edges = [
            {
                "before": "vector",
                "after": "cube",
                "kind": "DATA",
                "evidence_locator": "edge-1",
            },
            {
                "before": "cube",
                "after": "vector",
                "kind": "CONTROL_FLOW",
                "evidence_locator": "edge-2",
            },
        ]
        evidence = self._build(self._value(edges=edges))
        self.assertIn(
            "hard_dependency_cycle", {item["code"] for item in evidence["blockers"]}
        )

    def test_changed_provider_implementation_is_rejected(self):
        value = self._value()
        self.provider.write_text("# changed\n")
        with self.assertRaisesRegex(ValueError, "file changed"):
            self._build(value)


if __name__ == "__main__":
    unittest.main()
