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
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import multistream_network_conformance as conformance  # noqa: E402


class NetworkConformanceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.skill = self.root / "skill"
        self.skill.mkdir()
        (self.skill / "core.py").write_text("GENERIC_POLICY = 'cube_vector_only'\n")
        self.patch = mock.patch.object(conformance, "SKILL_ROOT", self.skill)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        closure = {
            "schema_version": "superkernel-multistream-real-npu-receipt-v1",
            "outcome": {
                "result_status": "no_gain",
                "incumbent_unchanged": True,
            },
        }
        screening = {
            "schema_version": "superkernel-multistream-resource-screening-receipt-v1",
            "decision": "no_reorder_candidate",
            "authorization": "diagnostic_only",
            "stable_parent_match_count": 0,
            "resource_classification_policy": "accelerator_core_block_num_mix_block_num_v1",
        }
        (self.root / "closure.json").write_text(json.dumps(closure) + "\n")
        (self.root / "screening.json").write_text(json.dumps(screening) + "\n")
        self._adapter(
            "adapter-reference",
            "minimal_reference",
            "real_npu_closure",
            "closure.json",
            "no_gain",
            "adapter-reference.json",
        )
        self._adapter(
            "adapter-large-model",
            "distributed_large_model",
            "resource_screening",
            "screening.json",
            "no_reorder_candidate",
            "adapter-large-model.json",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _adapter(self, adapter_id, network_class, kind, evidence, decision, output):
        draft = {
            "schema_version": conformance.ADAPTER_SCHEMA,
            "adapter_id": adapter_id,
            "network_class": network_class,
            "capabilities": [kind],
            "evidence": [{"kind": kind, "path": evidence}],
            "expected_decision": decision,
        }
        value = conformance.freeze_adapter(draft, self.root)
        (self.root / output).write_text(json.dumps(value) + "\n")

    def _plan(self, *, adapters=None):
        draft = {
            "schema_version": conformance.PLAN_SCHEMA,
            "suite_id": "two-network-suite",
            "core_files": ["core.py"],
            "adapters": adapters
            or ["adapter-reference.json", "adapter-large-model.json"],
        }
        plan = conformance.freeze_plan(draft, self.root)
        path = self.root / "plan.json"
        path.write_text(json.dumps(plan) + "\n")
        return path

    def test_two_network_classes_pass_with_unchanged_generic_core(self):
        report = conformance.build_report(self._plan(), self.root)
        self.assertEqual(report["decision"], "pass")
        self.assertEqual(report["adapter_count"], 2)
        self.assertEqual(report["network_class_count"], 2)
        self.assertEqual(
            {item["decision"] for item in report["adapters"]},
            {"no_gain", "no_reorder_candidate"},
        )

    def test_requires_two_distinct_network_classes(self):
        value = json.loads((self.root / "adapter-large-model.json").read_text())
        value["network_class"] = "minimal_reference"
        value.pop("adapter_fingerprint")
        value["adapter_fingerprint"] = conformance.fingerprint(value)
        (self.root / "adapter-large-model.json").write_text(json.dumps(value) + "\n")
        with self.assertRaisesRegex(ValueError, "two network classes"):
            self._plan()

    def test_no_candidate_requires_zero_parent_matches(self):
        screening = json.loads((self.root / "screening.json").read_text())
        screening["stable_parent_match_count"] = 1
        (self.root / "screening.json").write_text(json.dumps(screening) + "\n")
        with self.assertRaisesRegex(ValueError, "zero stable parent matches"):
            conformance.freeze_adapter(
                {
                    "schema_version": conformance.ADAPTER_SCHEMA,
                    "adapter_id": "adapter-large-model",
                    "network_class": "distributed_large_model",
                    "capabilities": ["resource_screening"],
                    "evidence": [
                        {"kind": "resource_screening", "path": "screening.json"}
                    ],
                    "expected_decision": "no_reorder_candidate",
                },
                self.root,
            )

    def test_adapter_identity_in_generic_core_is_rejected(self):
        (self.skill / "core.py").write_text("SPECIAL = 'adapter-large-model'\n")
        path = self._plan()
        with self.assertRaisesRegex(
            ValueError, "generic core contains adapter identities"
        ):
            conformance.build_report(path, self.root)

    def test_sealed_adapter_evidence_tampering_is_rejected(self):
        (self.root / "closure.json").write_text('{"tampered": true}\n')
        with self.assertRaisesRegex(ValueError, "sealed identity mismatch"):
            conformance.validate_adapter(
                json.loads((self.root / "adapter-reference.json").read_text()),
                self.root,
            )


if __name__ == "__main__":
    unittest.main()
