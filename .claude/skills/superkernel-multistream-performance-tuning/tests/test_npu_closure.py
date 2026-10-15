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

import multistream_npu_closure  # noqa: E402


class NpuClosureTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.request_fingerprint = "sha256:request"
        self.trial_id = "MS-R1"
        payloads = {
            "request": {},
            "operator_order_capture": {},
            "action_manifest": {
                "trial_id": self.trial_id,
                "change_kind": "dependency_safe_operator_reorder",
                "multistream_only_verified": True,
            },
            "dispatch_order_evidence": {},
            "four_profile_plan": {},
            "four_profile_summary": {},
            "clean_evidence": {},
            "result": {
                "status": "no_gain",
                "trials": [{"trial_id": self.trial_id, "decision": "rejected"}],
            },
        }
        self.paths = {}
        for name, schema in multistream_npu_closure.ARTIFACT_SCHEMAS.items():
            payload = {"schema_version": schema, **payloads[name]}
            path = self.root / f"{name}.json"
            path.write_text(json.dumps(payload) + "\n")
            self.paths[name] = path.name

    def tearDown(self):
        self.temporary.cleanup()

    def _draft(self):
        return {
            "schema_version": multistream_npu_closure.SCHEMA,
            "closure_id": "real-npu-closure-1",
            "request_fingerprint": self.request_fingerprint,
            "trial_id": self.trial_id,
            "runtime": {
                "device_name": "Ascend-test",
                "device_count": 1,
                "backend": "npugraph_ex",
                "npugraph_ex": True,
                "static_kernel_compile": True,
                "super_kernel_scope": True,
                "stream_count": 2,
            },
            "artifacts": self.paths,
            "outcome": "no_gain",
        }

    def _patch_replay(self, *, authorized=True):
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(
            multistream_npu_closure.multistream_contract,
            "validate_request",
            return_value={"request_fingerprint": self.request_fingerprint},
        ).start()
        mock.patch.object(
            multistream_npu_closure.multistream_operator_order,
            "analyze",
            return_value={
                "targets": [
                    {
                        "range_id": "range-1",
                        "multistream_reorder_authorized": authorized,
                    }
                ]
            },
        ).start()
        mock.patch.object(
            multistream_npu_closure.multistream_operator_order,
            "validate_dispatch_evidence",
            return_value={"valid": True, "decision": "pass"},
        ).start()
        mock.patch.object(
            multistream_npu_closure.multistream_four_profile,
            "validate_plan",
            return_value={
                "trial_id": self.trial_id,
                "request_fingerprint": self.request_fingerprint,
            },
        ).start()
        mock.patch.object(
            multistream_npu_closure.multistream_four_profile,
            "validate_summary",
            return_value={"complete": True},
        ).start()
        mock.patch.object(
            multistream_npu_closure.multistream_evidence,
            "validate_evidence",
            return_value={"decision": "reject"},
        ).start()
        mock.patch.object(
            multistream_npu_closure.multistream_contract,
            "validate_result",
            return_value={"status": "no_gain"},
        ).start()

    def test_seals_and_replays_complete_no_gain_closure(self):
        self._patch_replay()
        sealed = multistream_npu_closure.seal(self._draft(), self.root)
        result = multistream_npu_closure.validate(sealed, self.root)
        self.assertTrue(result["valid"])
        self.assertEqual(result["result_status"], "no_gain")
        self.assertEqual(result["authorized_range_ids"], ["range-1"])
        self.assertEqual(
            set(sealed["artifacts"]), set(multistream_npu_closure.ARTIFACT_SCHEMAS)
        )

    def test_no_candidate_cannot_substitute_for_executed_reorder(self):
        self._patch_replay(authorized=False)
        with self.assertRaisesRegex(
            ValueError, "analyzer-authorized reorder candidate"
        ):
            multistream_npu_closure.seal(self._draft(), self.root)

    def test_requires_real_npu_execution_capabilities(self):
        self._patch_replay()
        draft = self._draft()
        draft["runtime"]["stream_count"] = 1
        with self.assertRaisesRegex(ValueError, "at least two"):
            multistream_npu_closure.seal(draft, self.root)

    def test_rejects_artifact_tampering_after_seal(self):
        self._patch_replay()
        sealed = multistream_npu_closure.seal(self._draft(), self.root)
        (self.root / self.paths["clean_evidence"]).write_text('{"tampered": true}\n')
        with self.assertRaisesRegex(ValueError, "must use|identity mismatch"):
            multistream_npu_closure.validate(sealed, self.root)

    def test_no_gain_must_bind_one_rejected_trial(self):
        self._patch_replay()
        result_path = self.root / self.paths["result"]
        value = json.loads(result_path.read_text())
        value["trials"] = []
        result_path.write_text(json.dumps(value) + "\n")
        with self.assertRaisesRegex(ValueError, "exactly one matching executed trial"):
            multistream_npu_closure.seal(self._draft(), self.root)


if __name__ == "__main__":
    unittest.main()
