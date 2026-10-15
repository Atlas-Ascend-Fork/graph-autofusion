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

import multistream_evidence  # noqa: E402


class MultistreamEvidenceTest(unittest.TestCase):
    def test_multistream_clean_has_no_fixed_percentage_floor(self):
        self.assertEqual(multistream_evidence.DEFAULT_CLEAN_MIN_IMPROVEMENT_PCT, 0.0)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def _run(self, parent, index, decode_ms):
        run = self.root / parent / f"run-{index}"
        run.mkdir(parents=True)
        lines = [
            f"Inference time (decode): {decode_ms + offset / 100:.2f} ms"
            for offset in range(4)
        ]
        (run / "log_0.log").write_text("\n".join(lines) + "\n")

    def test_correctness_evidence_binds_rank_logs_and_detects_tampering(self):
        run = self.root / "correctness"
        run.mkdir()
        (run / "launcher.exit-code").write_text("0\n")
        (run / "correctness.status").write_text("passed\n")
        (run / "log_0.log").write_text("Finished inference\n")
        evidence = multistream_evidence.build_correctness(
            self.root,
            "correctness_passed",
            "MS-O1",
            "sha256:request",
            "correctness",
            1,
        )
        output = self.root / "evidence.json"
        output.write_text(json.dumps(evidence) + "\n")
        validated = multistream_evidence.validate_evidence(
            output,
            self.root,
            trial_id="MS-O1",
            request_fingerprint="sha256:request",
            state_after="correctness_passed",
        )
        self.assertEqual(validated["decision"], "pass")

        (run / "log_0.log").write_text("RuntimeError: device failed\n")
        with self.assertRaisesRegex(ValueError, "source changed"):
            multistream_evidence.validate_evidence(output, self.root)

    def test_clean3_no_gain_is_rejected_with_compact_evidence(self):
        for index in range(1, 6):
            self._run("baseline", index, 10.0)
        for index in range(1, 4):
            self._run("candidate", index, 11.0)
        evidence = multistream_evidence.build_clean(
            self.root,
            "clean3_passed",
            "MS-O1",
            "sha256:request",
            "baseline",
            "candidate",
            "MS-O1",
            expected_ranks=1,
            warmup=1,
            expected_runs=3,
            min_improvement_pct=2.0,
            allow_p90_regression_pct=0.0,
            allow_stddev_regression_pct=0.0,
        )
        self.assertEqual(evidence["decision"], "reject")
        self.assertFalse(
            evidence["semantic_result"]["selection"]["option_trial_accepted"]
        )
        self.assertNotIn("samples_ms", json.dumps(evidence))
        self.assertLess(len(json.dumps(evidence)), 12000)

    def test_clean_state_requires_exact_run_count(self):
        for index in range(1, 6):
            self._run("baseline", index, 10.0)
        for index in range(1, 5):
            self._run("candidate", index, 9.0)
        with self.assertRaisesRegex(ValueError, "exactly 3 candidate runs"):
            multistream_evidence.build_clean(
                self.root,
                "clean3_passed",
                "MS-O1",
                "sha256:request",
                "baseline",
                "candidate",
                "MS-O1",
                expected_ranks=1,
                warmup=1,
                expected_runs=3,
                min_improvement_pct=2.0,
                allow_p90_regression_pct=0.0,
                allow_stddev_regression_pct=0.0,
            )

    def test_component_analysis_evidence_accepts_complete_three_occurrence_result(self):
        analysis = {
            "layers": [{"range_id": "range-1"}],
            "summary": {
                "schema_version": "superkernel-multistream-component-performance-analysis-v1",
                "status": "complete",
                "target_count": 1,
                "occurrences_per_target": 3,
                "improved_layer_count": 1,
                "regressed_layer_count": 0,
                "unchanged_layer_count": 0,
                "incumbent_parent_total_mean_us": 10.0,
                "candidate_parent_total_mean_us": 9.0,
                "weighted_parent_improvement_us": 1.0,
                "weighted_parent_improvement_pct": 10.0,
                "incumbent_overlap_mean_us": 0.0,
                "candidate_overlap_mean_us": 1.0,
                "candidate_same_engine_contention_mean_us": 0.0,
            },
        }
        path = self.root / "component-analysis.json"
        path.write_text(json.dumps(analysis) + "\n")
        evidence = multistream_evidence.build_analysis(
            self.root,
            "analysis_validated",
            "MS-C1",
            "sha256:request",
            path,
        )
        self.assertEqual(evidence["decision"], "pass")
        self.assertEqual(evidence["semantic_result"]["range_id_count"], 1)

    def test_component_analysis_evidence_rejects_short_capture(self):
        analysis = {
            "layers": [{"range_id": "range-1"}],
            "summary": {
                "schema_version": "superkernel-multistream-component-performance-analysis-v1",
                "status": "complete",
                "target_count": 1,
                "occurrences_per_target": 2,
            },
        }
        path = self.root / "component-analysis.json"
        path.write_text(json.dumps(analysis) + "\n")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            multistream_evidence.build_analysis(
                self.root,
                "analysis_validated",
                "MS-C1",
                "sha256:request",
                path,
            )


if __name__ == "__main__":
    unittest.main()
