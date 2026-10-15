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
import multistream_event_stage_planner  # noqa: E402
from test_critical_path import CriticalPathTest  # noqa: E402


class EventStagePlannerTest(unittest.TestCase):
    def _inputs(self):
        fixture = CriticalPathTest()
        capture = fixture._capture(aux_start=8.0, event_delay=8.0)
        analysis = multistream_critical_path.analyze(capture)
        catalog = multistream_event_stage_action.build(
            analysis, capture["logical_graph"]
        )
        return analysis, catalog

    def test_event_is_planned_before_stage(self):
        analysis, catalog = self._inputs()
        matrix = multistream_event_stage_planner.plan(
            analysis, catalog, {"settled_actions": []}, 3
        )
        self.assertEqual(len(matrix["candidates"]), 1)
        self.assertEqual(
            matrix["candidates"][0]["change_kind"], "event_edge_refinement"
        )
        self.assertTrue(matrix["candidates"][0]["selected_for_execution"])

    def test_stage_is_released_only_after_nonregressing_event_no_gain(self):
        analysis, catalog = self._inputs()
        event = catalog["actions"][0]
        history = {
            "settled_actions": [
                {
                    "action_id": event["action_id"],
                    "status": "rejected",
                    "correctness_passed": True,
                    "mechanism_validated": True,
                    "clean3_nonregressing": True,
                }
            ]
        }
        matrix = multistream_event_stage_planner.plan(analysis, catalog, history, 3)
        self.assertEqual(
            [item["change_kind"] for item in matrix["candidates"]], ["stage_split"]
        )
        self.assertEqual(
            matrix["candidates"][0]["parent_action_id"], event["action_id"]
        )

    def test_stage_stays_blocked_when_event_mechanism_is_not_validated(self):
        analysis, catalog = self._inputs()
        event = catalog["actions"][0]
        history = {
            "settled_actions": [
                {
                    "action_id": event["action_id"],
                    "status": "rejected",
                    "correctness_passed": True,
                    "mechanism_validated": False,
                    "clean3_nonregressing": True,
                }
            ]
        }
        matrix = multistream_event_stage_planner.plan(analysis, catalog, history, 3)
        self.assertEqual(matrix["candidates"], [])
        self.assertEqual(
            matrix["blocked_actions"][0]["code"], "PARENT_MECHANISM_NOT_VALIDATED"
        )

    def test_scope_derivative_requires_settled_parent(self):
        analysis, catalog = self._inputs()
        event = catalog["actions"][0]
        history = {
            "settled_actions": [
                {
                    "action_id": event["action_id"],
                    "status": "rejected",
                    "correctness_passed": True,
                    "mechanism_validated": True,
                    "clean3_nonregressing": True,
                }
            ]
        }
        derivatives = [
            {
                "derivative_id": "scope-1",
                "parent_action_id": event["action_id"],
                "source_action": {
                    "range_id": "range-1",
                    "change_kind": "scope_split",
                    "factor_id": "align-ready-boundary",
                },
            }
        ]
        matrix = multistream_event_stage_planner.plan(
            analysis, catalog, history, 3, derivatives
        )
        self.assertIn(
            "scope_event_derivative",
            [item["change_kind"] for item in matrix["candidates"]],
        )
        scope = next(
            item
            for item in matrix["candidates"]
            if item["change_kind"] == "scope_event_derivative"
        )
        self.assertEqual(
            scope["comparison_baselines"], ["incumbent", "parent_candidate"]
        )
        self.assertEqual(scope["lineage_depth"], 2)

    def test_scope_derivative_rejects_ambiguous_source_factor(self):
        analysis, catalog = self._inputs()
        event = catalog["actions"][0]
        history = {
            "settled_actions": [
                {
                    "action_id": event["action_id"],
                    "status": "rejected",
                    "correctness_passed": True,
                    "mechanism_validated": True,
                    "clean3_nonregressing": True,
                }
            ]
        }
        derivatives = [
            {
                "derivative_id": "scope-1",
                "parent_action_id": event["action_id"],
                "source_action": {"range_id": "range-1", "change_kind": "scope_split"},
            }
        ]
        with self.assertRaisesRegex(ValueError, "one exact factor"):
            multistream_event_stage_planner.plan(
                analysis, catalog, history, 3, derivatives
            )


if __name__ == "__main__":
    unittest.main()
