import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import multistream_contract  # noqa: E402
import multistream_opportunity_discovery  # noqa: E402
import multistream_trace_analysis  # noqa: E402


class OpportunityDiscoveryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.analysis_path = self.root / "profiling-analysis.json"
        decisions = [
            self._decision("beneficial", "range-beneficial", "fp-beneficial"),
            self._decision("regressed", "range-regressed", "fp-regressed"),
            self._decision(
                "beneficial", "range-single-stream", "fp-single", stream_count=1
            ),
        ]
        analysis = {
            "schema_version": "1.2",
            "per_sk_decisions": decisions,
        }
        analysis["analysis_content_fingerprint"] = (
            multistream_contract._analysis_fingerprint(analysis)
        )
        self.analysis_path.write_text(json.dumps(analysis) + "\n")

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _decision(classification, range_id, occurrence_fp, *, stream_count=2):
        return {
            "range_id": range_id,
            "graph_occurrence_fingerprint": occurrence_fp,
            "classification": classification,
            "mapping_method": "kernel_projection_structural",
            "mapping_confidence": "exact_projected_trace",
            "candidate_occurrence_count": 3,
            "child_count": 2,
            "original": {
                "interval_us": {"count": 3},
                "example_occurrence": {
                    "stream_count": stream_count,
                    "stream_ids": list(range(stream_count)),
                    "multi_stream_analysis": {
                        "multi_stream_detected": stream_count >= 2,
                    },
                },
            },
        }

    def test_discovery_includes_beneficial_and_regressed_multistream_targets(self):
        discovery = multistream_opportunity_discovery.discover(
            self.analysis_path, self.root
        )

        self.assertEqual(discovery["summary"]["eligible_target_count"], 2)
        self.assertEqual(discovery["summary"]["beneficial_target_count"], 1)
        self.assertEqual(discovery["summary"]["regressed_target_count"], 1)
        self.assertEqual(
            {item["range_id"] for item in discovery["eligible_targets"]},
            {"range-beneficial", "range-regressed"},
        )
        beneficial = next(
            item
            for item in discovery["eligible_targets"]
            if item["range_id"] == "range-beneficial"
        )
        self.assertEqual(beneficial["parallelism_effect"], "unknown")
        self.assertEqual(beneficial["optimization_status"], "blocked")

    def test_discovery_accepts_fusion_analysis_fingerprint_with_chinese_text(self):
        analysis = json.loads(self.analysis_path.read_text())
        analysis["next_agent_guidance_zh"] = "仅按证据充分的逐 SK 判定处理本轮 scope。"
        analysis["analysis_content_fingerprint"] = (
            multistream_contract._analysis_fingerprint(analysis)
        )
        self.analysis_path.write_text(json.dumps(analysis, ensure_ascii=False) + "\n")

        discovery = multistream_opportunity_discovery.discover(
            self.analysis_path, self.root
        )

        self.assertEqual(discovery["summary"]["eligible_target_count"], 2)

    def test_coverage_fails_when_beneficial_target_is_omitted(self):
        discovery = multistream_opportunity_discovery.discover(
            self.analysis_path, self.root
        )
        discovery_path = self.root / "discovery.json"
        discovery_path.write_text(json.dumps(discovery) + "\n")
        trace = {
            "schema_version": multistream_trace_analysis.ANALYSIS_SCHEMA,
            "targets": [
                {
                    "range_id": "range-regressed",
                    "graph_occurrence_fingerprint": "fp-regressed",
                    "latent_opportunity": False,
                }
            ],
            "blockers": [],
        }
        trace["analysis_fingerprint"] = multistream_trace_analysis.fingerprint(trace)
        trace_path = self.root / "trace-analysis.json"
        trace_path.write_text(json.dumps(trace) + "\n")

        with mock.patch(
            "multistream_opportunity_discovery.multistream_trace_analysis.validate_bound_analysis",
            return_value=trace,
        ):
            coverage = multistream_opportunity_discovery.audit_coverage(
                discovery_path, trace_path, self.root
            )

        self.assertFalse(coverage["complete"])
        self.assertEqual(
            coverage["missing_targets"],
            [
                {
                    "range_id": "range-beneficial",
                    "graph_occurrence_fingerprint": "fp-beneficial",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
