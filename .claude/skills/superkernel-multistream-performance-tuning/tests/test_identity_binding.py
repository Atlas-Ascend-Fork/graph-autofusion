# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import multistream_identity_binding  # noqa: E402


class IdentityBindingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.evidence = self.root / "mapping.json"
        self.evidence.write_text("{}\n")

    def tearDown(self):
        self.temporary.cleanup()

    def _observation(self, domain, alignment, identity, method):
        return {
            "domain": domain,
            "identity": identity,
            "operator_id": "operator-1",
            "statement_id": "statement-1",
            "alignment_id": alignment,
            "binding_method": method,
            "evidence_path": "mapping.json",
            "evidence_locator": f"{domain}:{alignment or 'source'}",
        }

    def _value(self):
        observations = [
            self._observation(
                "source",
                None,
                {"source_file": "model.py", "start_offset": 10, "end_offset": 20},
                "exact_source_span",
            )
        ]
        for index in range(3):
            alignment = f"decode-{index}"
            observations.extend(
                [
                    self._observation(
                        "sk_off",
                        alignment,
                        {"origin_uid": f"off-{index}"},
                        "compiler_origin_uid",
                    ),
                    self._observation(
                        "sk_on",
                        alignment,
                        {"origin_uid": f"on-{index}"},
                        "sk_meta_origin_exact",
                    ),
                ]
            )
        value = {
            "schema_version": multistream_identity_binding.OBSERVATIONS_SCHEMA,
            "observation_set_id": "identity-set-1",
            "request_fingerprint": "request-fp-1",
            "evidence_files": [
                {
                    "path": "mapping.json",
                    "size_bytes": self.evidence.stat().st_size,
                    "file_fingerprint": multistream_identity_binding.file_fingerprint(
                        self.evidence
                    ),
                }
            ],
            "observations": observations,
        }
        value["observations_fingerprint"] = multistream_identity_binding.fingerprint(
            value
        )
        return value

    def _build(self, value):
        path = self.root / "observations.json"
        path.write_text(json.dumps(value) + "\n")
        return multistream_identity_binding.build(path, self.root)

    def test_builds_complete_bidirectional_registry(self):
        registry = self._build(self._value())
        self.assertTrue(registry["complete"])
        self.assertEqual(registry["blockers"], [])
        self.assertEqual(
            registry["entries"][0]["aligned_occurrence_ids"],
            ["decode-0", "decode-1", "decode-2"],
        )
        self.assertEqual(len(registry["forward_index"]), 7)
        registry_path = self.root / "registry.json"
        registry_path.write_text(json.dumps(registry) + "\n")
        summary = multistream_identity_binding.validate(
            registry_path, self.root, require_complete=True
        )
        self.assertTrue(summary["complete"])

    def test_reports_ambiguous_raw_identity(self):
        value = self._value()
        duplicate = copy.deepcopy(value["observations"][1])
        duplicate["operator_id"] = "operator-2"
        duplicate["statement_id"] = "statement-2"
        value["observations"].append(duplicate)
        value["observations_fingerprint"] = multistream_identity_binding.fingerprint(
            {
                key: item
                for key, item in value.items()
                if key != "observations_fingerprint"
            }
        )
        registry = self._build(value)
        codes = {item["code"] for item in registry["blockers"]}
        self.assertIn("ambiguous_raw_identity", codes)
        self.assertFalse(registry["complete"])

    def test_reports_occurrence_alignment_mismatch(self):
        value = self._value()
        value["observations"][-1]["alignment_id"] = "decode-other"
        value["observations_fingerprint"] = multistream_identity_binding.fingerprint(
            {
                key: item
                for key, item in value.items()
                if key != "observations_fingerprint"
            }
        )
        registry = self._build(value)
        self.assertIn(
            "occurrence_alignment_mismatch",
            {item["code"] for item in registry["blockers"]},
        )

    def test_rejects_name_or_ordinal_binding_method(self):
        value = self._value()
        value["observations"][1]["binding_method"] = "operator_name"
        value["observations_fingerprint"] = multistream_identity_binding.fingerprint(
            {
                key: item
                for key, item in value.items()
                if key != "observations_fingerprint"
            }
        )
        with self.assertRaisesRegex(ValueError, "not an exact method"):
            self._build(value)


if __name__ == "__main__":
    unittest.main()
