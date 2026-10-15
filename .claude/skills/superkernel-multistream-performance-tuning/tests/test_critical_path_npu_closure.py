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

import multistream_critical_path_npu_closure  # noqa: E402


class CriticalPathNpuClosureTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.trial_id = "MS-CP-1"
        self.request_fp = "request-fp-1"
        payloads = {
            "request": {},
            "action_manifest": {
                "trial_id": self.trial_id,
                "action_id": "event-1",
                "change_kind": "event_edge_refinement",
                "multistream_only_verified": True,
                "single_change_verified": True,
            },
            "event_stage_dispatch": {"decision": "effective"},
            "four_profile_plan": {},
            "four_profile_summary": {},
            "join_validation": {
                "trial_id": self.trial_id,
                "request_fingerprint": self.request_fp,
            },
            "clean_evidence": {},
            "result": {
                "trials": [
                    {
                        "trial_id": self.trial_id,
                        "decision": "rejected",
                        "clean_state": "clean3_passed",
                    }
                ],
            },
            "cleanup_plan": {},
            "cleanup_receipt": {},
        }
        self.paths = {}
        for (
            name,
            schema,
        ) in multistream_critical_path_npu_closure.ARTIFACT_SCHEMAS.items():
            path = self.root / f"{name}.json"
            path.write_text(
                json.dumps({"schema_version": schema, **payloads[name]}) + "\n"
            )
            self.paths[name] = path.name

    def tearDown(self):
        self.temporary.cleanup()

    def _draft(self):
        return {
            "schema_version": multistream_critical_path_npu_closure.SCHEMA,
            "closure_id": "critical-path-npu-1",
            "request_fingerprint": self.request_fp,
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
            "action_closed": True,
            "four_profile_closed": True,
            "outcome": {},
        }

    def _patch(self, dispatch="effective"):
        self.addCleanup(mock.patch.stopall)
        module = multistream_critical_path_npu_closure
        mock.patch.object(
            module.multistream_critical_path_contract,
            "validate_request",
            return_value={"request_fingerprint": self.request_fp},
        ).start()
        mock.patch.object(
            module.multistream_event_stage_dispatch,
            "validate",
            return_value={"decision": dispatch},
        ).start()
        mock.patch.object(
            module.multistream_four_profile,
            "validate_plan",
            return_value={
                "trial_id": self.trial_id,
                "request_fingerprint": self.request_fp,
            },
        ).start()
        mock.patch.object(
            module.multistream_four_profile,
            "validate_summary",
            return_value={"complete": True},
        ).start()
        action = json.loads((self.root / self.paths["action_manifest"]).read_text())
        join = json.loads((self.root / self.paths["join_validation"]).read_text())
        join["action_manifest_fingerprint"] = (
            module.multistream_source_transform.fingerprint(action)
        )
        join["decision"] = "effective"
        mock.patch.object(
            module.multistream_join_validation, "validate", return_value=join
        ).start()
        mock.patch.object(
            module.multistream_critical_path_contract,
            "validate_result",
            return_value={"status": "no_gain"},
        ).start()
        mock.patch.object(
            module.multistream_evidence,
            "validate_evidence",
            return_value={"decision": "reject"},
        ).start()
        mock.patch.object(
            module.multistream_cleanup, "validate_receipt", return_value={"valid": True}
        ).start()

    def test_seals_complete_event_stage_no_gain_closure(self):
        self._patch()
        sealed = multistream_critical_path_npu_closure.seal(self._draft(), self.root)
        replay = multistream_critical_path_npu_closure.validate(sealed, self.root)
        self.assertTrue(replay["valid"])
        self.assertEqual(replay["result_status"], "no_gain")
        self.assertTrue(sealed["action_closed"])
        self.assertTrue(sealed["four_profile_closed"])

    def test_rejects_action_that_did_not_change_dispatch(self):
        self._patch(dispatch="not_effective")
        with self.assertRaisesRegex(ValueError, "did not change"):
            multistream_critical_path_npu_closure.seal(self._draft(), self.root)

    def test_rejects_single_stream_runtime(self):
        self._patch()
        draft = self._draft()
        draft["runtime"]["stream_count"] = 1
        with self.assertRaisesRegex(ValueError, "at least two"):
            multistream_critical_path_npu_closure.seal(draft, self.root)


if __name__ == "__main__":
    unittest.main()
