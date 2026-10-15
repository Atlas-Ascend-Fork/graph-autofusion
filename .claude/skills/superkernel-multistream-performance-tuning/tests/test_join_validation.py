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
import multistream_join_validation  # noqa: E402
from test_critical_path import CriticalPathTest  # noqa: E402


class JoinValidationTest(unittest.TestCase):
    ACTION_FP = "sha256:" + "0" * 64

    def _analyses(self):
        fixture = CriticalPathTest()
        baseline = multistream_critical_path.analyze(fixture._capture())
        candidate = multistream_critical_path.analyze(
            fixture._capture(aux_start=0.0, downstream_start=18.0)
        )
        return baseline, candidate

    def test_validates_earlier_join_without_stall_regression(self):
        baseline, candidate = self._analyses()
        result = multistream_join_validation.compare(
            "trial-1", self.ACTION_FP, baseline, candidate
        )
        self.assertEqual(result["decision"], "validated")
        target = result["targets"][0]
        self.assertLess(target["join_ready_delta_us"]["p50"], 0.0)
        self.assertLessEqual(target["join_stall_delta_us"]["p90"], 0.0)

    def test_rejects_candidate_when_join_stall_grows(self):
        fixture = CriticalPathTest()
        baseline = multistream_critical_path.analyze(fixture._capture())
        candidate = multistream_critical_path.analyze(fixture._capture(aux_start=0.0))
        result = multistream_join_validation.compare(
            "trial-1", self.ACTION_FP, baseline, candidate
        )
        self.assertEqual(result["decision"], "not_effective")
        self.assertEqual(result["targets"][0]["reason"], "join_stall_regressed")

    def test_rejects_misaligned_occurrences(self):
        baseline, candidate = self._analyses()
        candidate["targets"][0]["occurrences"][0]["alignment_id"] = "other"
        candidate["analysis_fingerprint"] = multistream_critical_path.fingerprint(
            {
                key: value
                for key, value in candidate.items()
                if key != "analysis_fingerprint"
            }
        )
        with self.assertRaisesRegex(ValueError, "alignment"):
            multistream_join_validation.compare(
                "trial-1", self.ACTION_FP, baseline, candidate
            )


if __name__ == "__main__":
    unittest.main()
