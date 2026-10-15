# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import multistream_event_stage_action  # noqa: E402
import multistream_source_transform  # noqa: E402


class SourceTransformTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "model.py"
        self.source.write_text(
            "def run(enable_multi_streams):\n"
            "    if enable_multi_streams:\n"
            "        record_event('done')\n"
            "        wait_event('done')\n"
            "    return 1\n"
        )
        self.catalog = {
            "schema_version": multistream_event_stage_action.SCHEMA,
            "request_fingerprint": "request-1",
            "critical_path_analysis_fingerprint": "analysis-1",
            "actions": [
                {
                    "action_id": "event:join:aux:ready",
                    "change_kind": "event_edge_refinement",
                    "join_id": "join",
                    "stage_id": "aux",
                    "event_edge_id": "ready",
                    "risk": "low",
                    "activation_condition": "immediate",
                    "parent_action_id": None,
                    "expected_dispatch_change": {
                        "kind": "event_notify_earlier",
                        "logical_id": "ready",
                    },
                    "safety_requirements": list(
                        multistream_event_stage_action.EVENT_REQUIREMENTS
                    ),
                }
            ],
        }
        self.catalog["action_catalog_fingerprint"] = (
            multistream_event_stage_action.fingerprint(self.catalog)
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _transform(self):
        original = self.source.read_bytes()
        old = b"record_event('done')"
        start = original.index(old)
        transform = {
            "schema_version": multistream_source_transform.TRANSFORM_SCHEMA,
            "trial_id": "trial-1",
            "action_catalog_fingerprint": self.catalog["action_catalog_fingerprint"],
            "action_id": "event:join:aux:ready",
            "source_file": "model.py",
            "input_source_fingerprint": multistream_source_transform.file_fingerprint(
                self.source
            ),
            "single_stream_projection_fingerprint_before": "projection-fp-1",
            "single_stream_projection_fingerprint_after": "projection-fp-1",
            "allowed_multistream_ranges": [
                {
                    "start_offset": original.index(b"    if enable_multi_streams:"),
                    "end_offset": original.index(b"    return 1"),
                }
            ],
            "dependency_evidence_fingerprint_before": "dependency-before",
            "dependency_evidence_fingerprint_after": "dependency-after",
            "safety_proofs": {
                "producer_before_record": True,
                "consumer_after_wait": True,
                "event_reuse_safe": True,
                "record_stream_lifetime_preserved": True,
                "modified_dependency_complete": True,
            },
            "stage_state_contract": None,
            "replacements": [
                {
                    "start_offset": start,
                    "end_offset": start + len(old),
                    "before_fingerprint": "sha256:" + hashlib.sha256(old).hexdigest(),
                    "replacement": "record_event('ready')",
                }
            ],
        }
        transform["transform_fingerprint"] = multistream_source_transform.fingerprint(
            transform
        )
        return transform

    def test_materializes_authorized_single_factor_transform(self):
        manifest = multistream_source_transform.materialize(
            self._transform(), self.catalog, self.root, self.root / "candidate.py"
        )
        self.assertEqual(manifest["change_kind"], "event_edge_refinement")
        self.assertIn("record_event('ready')", (self.root / "candidate.py").read_text())
        self.assertTrue(manifest["single_change_verified"])
        self.assertEqual(
            manifest["expected_dispatch_change"]["kind"], "event_notify_earlier"
        )

    def test_rejects_stale_replacement_before_write(self):
        transform = self._transform()
        transform["replacements"][0]["before_fingerprint"] = "sha256:" + "0" * 64
        transform["transform_fingerprint"] = multistream_source_transform.fingerprint(
            {
                key: value
                for key, value in transform.items()
                if key != "transform_fingerprint"
            }
        )
        with self.assertRaisesRegex(ValueError, "before_fingerprint"):
            multistream_source_transform.materialize(
                transform, self.catalog, self.root, self.root / "candidate.py"
            )
        self.assertFalse((self.root / "candidate.py").exists())

    def test_rejects_overlapping_replacements(self):
        transform = self._transform()
        hunk = dict(transform["replacements"][0])
        hunk["start_offset"] += 1
        hunk["end_offset"] -= 1
        hunk["before_fingerprint"] = (
            "sha256:"
            + hashlib.sha256(
                self.source.read_bytes()[hunk["start_offset"] : hunk["end_offset"]]
            ).hexdigest()
        )
        transform["replacements"].append(hunk)
        transform["transform_fingerprint"] = multistream_source_transform.fingerprint(
            {
                key: value
                for key, value in transform.items()
                if key != "transform_fingerprint"
            }
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            multistream_source_transform.validate(transform, self.catalog, self.root)

    def test_rejects_changed_single_stream_projection(self):
        transform = self._transform()
        transform["single_stream_projection_fingerprint_after"] = "other"
        transform["transform_fingerprint"] = multistream_source_transform.fingerprint(
            {
                key: value
                for key, value in transform.items()
                if key != "transform_fingerprint"
            }
        )
        with self.assertRaisesRegex(ValueError, "single_stream_projection"):
            multistream_source_transform.validate(transform, self.catalog, self.root)

    def test_rejects_replacement_outside_multistream_ranges(self):
        transform = self._transform()
        transform["allowed_multistream_ranges"] = [
            {"start_offset": 0, "end_offset": 10}
        ]
        transform["transform_fingerprint"] = multistream_source_transform.fingerprint(
            {
                key: value
                for key, value in transform.items()
                if key != "transform_fingerprint"
            }
        )
        with self.assertRaisesRegex(ValueError, "outside declared multistream"):
            multistream_source_transform.validate(transform, self.catalog, self.root)

    def test_rejects_incomplete_event_safety_proof(self):
        transform = self._transform()
        transform["safety_proofs"]["event_reuse_safe"] = False
        transform["transform_fingerprint"] = multistream_source_transform.fingerprint(
            {
                key: value
                for key, value in transform.items()
                if key != "transform_fingerprint"
            }
        )
        with self.assertRaisesRegex(ValueError, "safety proofs"):
            multistream_source_transform.validate(transform, self.catalog, self.root)

    def test_wraps_one_scope_factor_with_parent_lineage(self):
        scope = {
            "schema_version": "superkernel-multistream-action-manifest-v1",
            "trial_id": "scope-trial",
            "change_kind": "scope_split",
            "single_change_verified": True,
            "source_adapter_validation": "passed",
            "immutable_input_source": "model.py",
            "materialized_source": "candidate.py",
            "input_source_fingerprint": "source-before",
            "output_source_fingerprint": "source-after",
        }
        parent = {
            "schema_version": multistream_source_transform.ACTION_MANIFEST_SCHEMA,
            "action_id": "event-parent",
        }
        candidate = {
            "action_id": "scope-child",
            "parent_action_id": "event-parent",
            "change_kind": "scope_event_derivative",
            "source_action": {
                "change_kind": "scope_split",
                "range_id": "range-1",
                "factor_id": "align-ready",
            },
        }
        manifest = multistream_source_transform.wrap_scope_derivative(
            scope,
            candidate,
            parent,
            single_stream_projection_fingerprint_before="projection-1",
            single_stream_projection_fingerprint_after="projection-1",
            dependency_evidence_fingerprint_before="dependency-1",
            dependency_evidence_fingerprint_after="dependency-2",
        )
        self.assertEqual(manifest["change_kind"], "scope_event_derivative")
        self.assertEqual(manifest["parent_action_id"], "event-parent")
        self.assertTrue(manifest["single_change_verified"])


if __name__ == "__main__":
    unittest.main()
