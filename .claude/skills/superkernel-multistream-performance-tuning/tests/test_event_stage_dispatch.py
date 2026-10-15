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
sys.path.insert(0, str(SCRIPT_DIR))

import multistream_event_stage_dispatch  # noqa: E402
import multistream_source_transform  # noqa: E402


class EventStageDispatchTest(unittest.TestCase):
    def _action(self, kind="event_notify_earlier", logical_id="aux.ready"):
        return {
            "schema_version": multistream_source_transform.ACTION_MANIFEST_SCHEMA,
            "trial_id": "trial-1",
            "action_id": "action-1",
            "expected_dispatch_change": {"kind": kind, "logical_id": logical_id},
        }

    @staticmethod
    def _row(logical_id, kind, role, ordinal):
        return {
            "logical_id": logical_id,
            "kind": kind,
            "stream_role": role,
            "dispatch_ordinal": ordinal,
        }

    def _occurrences(self, candidate_ordinal=1):
        return [
            {
                "alignment_id": f"decode-{index}",
                "baseline": [
                    self._row("main.cube", "stage", "main", 0),
                    self._row("aux.ready", "event_notify", "auxiliary", 3),
                ],
                "candidate": [
                    self._row("main.cube", "stage", "main", 0),
                    self._row(
                        "aux.ready", "event_notify", "auxiliary", candidate_ordinal
                    ),
                ],
            }
            for index in range(3)
        ]

    def test_proves_event_notify_advanced_in_three_occurrences(self):
        action = self._action()
        evidence = multistream_event_stage_dispatch.build(
            "trial-1", "request-1", action, self._occurrences()
        )
        self.assertEqual(evidence["decision"], "effective")
        self.assertEqual(
            multistream_event_stage_dispatch.validate(evidence, action)["decision"],
            "effective",
        )

    def test_unchanged_dispatch_is_not_effective(self):
        evidence = multistream_event_stage_dispatch.build(
            "trial-1",
            "request-1",
            self._action(),
            self._occurrences(candidate_ordinal=3),
        )
        self.assertEqual(evidence["decision"], "not_effective")

    def test_rejects_single_stream_observation(self):
        occurrences = self._occurrences()
        for occurrence in occurrences:
            occurrence["baseline"][1]["stream_role"] = "main"
            occurrence["candidate"][1]["stream_role"] = "main"
        with self.assertRaisesRegex(ValueError, "multi-stream"):
            multistream_event_stage_dispatch.build(
                "trial-1", "request-1", self._action(), occurrences
            )

    def test_rejects_missing_occurrence(self):
        with self.assertRaisesRegex(ValueError, "at least three"):
            multistream_event_stage_dispatch.build(
                "trial-1", "request-1", self._action(), self._occurrences()[:2]
            )


if __name__ == "__main__":
    unittest.main()
