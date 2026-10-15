# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(TEST_DIR))

import multistream_critical_path  # noqa: E402
import multistream_event_stage_action  # noqa: E402
from test_critical_path import CriticalPathTest  # noqa: E402


class EventStageActionTest(unittest.TestCase):
    def _analysis_and_graph(self):
        fixture = CriticalPathTest()
        capture = fixture._capture(aux_start=8.0, event_delay=8.0)
        return multistream_critical_path.analyze(capture), capture["logical_graph"]

    def test_builds_event_first_then_stage_derivative(self):
        analysis, graph = self._analysis_and_graph()
        catalog = multistream_event_stage_action.build(analysis, graph)
        self.assertEqual(
            [item["change_kind"] for item in catalog["actions"]],
            [
                "event_edge_refinement",
                "stage_split",
            ],
        )
        event, stage = catalog["actions"]
        self.assertEqual(event["activation_condition"], "immediate")
        self.assertEqual(stage["parent_action_id"], event["action_id"])
        self.assertEqual(
            stage["activation_condition"],
            "event_clean3_nonregressing_without_incremental_gain",
        )

    def test_stage_delay_without_event_delay_builds_immediate_stage_only(self):
        fixture = CriticalPathTest()
        capture = fixture._capture()
        analysis = multistream_critical_path.analyze(capture)
        catalog = multistream_event_stage_action.build(
            analysis, capture["logical_graph"]
        )
        self.assertEqual(
            [item["change_kind"] for item in catalog["actions"]], ["stage_split"]
        )
        self.assertEqual(catalog["actions"][0]["activation_condition"], "immediate")
        self.assertIsNone(catalog["actions"][0]["parent_action_id"])

    def test_no_actionable_analysis_produces_empty_catalog(self):
        fixture = CriticalPathTest()
        capture = fixture._capture(aux_start=0.0)
        analysis = multistream_critical_path.analyze(capture)
        catalog = multistream_event_stage_action.build(
            analysis, capture["logical_graph"]
        )
        self.assertEqual(catalog["actions"], [])

    def test_rejects_analysis_fingerprint_tampering(self):
        analysis, graph = self._analysis_and_graph()
        analysis["decision"] = "no_event_or_stage_candidate"
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            multistream_event_stage_action.build(analysis, graph)


if __name__ == "__main__":
    unittest.main()
