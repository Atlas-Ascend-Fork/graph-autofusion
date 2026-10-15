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

import multistream_execution  # noqa: E402


class MultistreamExecutionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input_config = self.root / "incumbent.yaml"
        self.output_config = self.root / "trial.yaml"
        self.input_config.write_text(
            "model_config:\n"
            "  custom_params:\n"
            "    super_kernel_optimize_options:\n"
            "      auto_op_parallel: 0\n"
            "    super_kernel_debug_options: {}\n"
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_materialize_option_proves_one_structural_change(self):
        pointer = (
            "/model_config/custom_params/super_kernel_optimize_options/auto_op_parallel"
        )
        manifest = multistream_execution.materialize_option(
            self.input_config,
            self.output_config,
            pointer,
            0,
            1,
            "MS-O1",
        )

        self.assertEqual(manifest["changed_pointers"], [pointer])
        self.assertTrue(manifest["single_change_verified"])
        self.assertEqual(
            multistream_execution._load_structured(self.output_config)["model_config"][
                "custom_params"
            ]["super_kernel_optimize_options"]["auto_op_parallel"],
            1,
        )
        self.assertEqual(
            multistream_execution._load_structured(self.input_config)["model_config"][
                "custom_params"
            ]["super_kernel_optimize_options"]["auto_op_parallel"],
            0,
        )

    def test_materialize_option_rejects_stale_before_value(self):
        with self.assertRaisesRegex(ValueError, "before value"):
            multistream_execution.materialize_option(
                self.input_config,
                self.output_config,
                "/model_config/custom_params/super_kernel_optimize_options/auto_op_parallel",
                7,
                1,
                "MS-O1",
            )

    def test_materialize_exact_none_range_exclusion(self):
        source = self.root / "model.py"
        output = self.root / "model-trial.py"
        text = "def run():\n    value = compute()\n    return value\n"
        source.write_text(text)
        start = text.index("    value")
        end = text.index("    return")

        manifest = multistream_execution.materialize_source_action(
            source,
            output,
            "model.py",
            start,
            end,
            "exclude one exact range from fusion",
            "range_exclusion",
            [
                {
                    "offset": start,
                    "text": "    torch.npu.super_kernel_scope_begin(None)\n",
                },
                {
                    "offset": end,
                    "text": "    torch.npu.super_kernel_scope_end(None)\n",
                },
            ],
            "MS-X1",
        )

        self.assertEqual(manifest["mechanism"], "explicit_none_exclusion")
        self.assertEqual(manifest["source_adapter_validation"], "passed")
        self.assertIn("super_kernel_scope_begin(None)", output.read_text())
        self.assertEqual(source.read_text(), text)

    def test_source_action_rejects_business_statement_insertion(self):
        source = self.root / "model.py"
        output = self.root / "model-trial.py"
        text = "value = compute()\nreturn_value = value\n"
        source.write_text(text)
        start = 0
        end = text.index("return_value")

        with self.assertRaisesRegex(ValueError, "non-scope statement"):
            multistream_execution.materialize_source_action(
                source,
                output,
                "model.py",
                start,
                end,
                "exclude one exact range from fusion",
                "range_exclusion",
                [
                    {"offset": start, "text": "value = mutate_business_value()\n"},
                    {
                        "offset": end,
                        "text": "torch.npu.super_kernel_scope_end(None)\n",
                    },
                ],
                "MS-X1",
            )

    def test_trial_state_requires_gate_order_and_evidence(self):
        action_path = self.root / "action.json"
        action_path.write_text(
            json.dumps(
                {
                    "schema_version": multistream_execution.ACTION_SCHEMA,
                    "trial_id": "MS-O1",
                }
            )
            + "\n"
        )
        state = multistream_execution.initialize_state(
            "MS-O1", "sha256:request", action_path
        )
        with self.assertRaisesRegex(ValueError, "one gate at a time"):
            multistream_execution.advance_state(
                state, "correctness_passed", "correctness.json"
            )
        state = multistream_execution.advance_state(state, "materialized")
        with self.assertRaisesRegex(ValueError, "requires an evidence"):
            multistream_execution.advance_state(state, "diff_verified")

    def test_accepted_state_replays_complete_history(self):
        action_path = self.root / "action.json"
        action_path.write_text(
            json.dumps(
                {
                    "schema_version": multistream_execution.ACTION_SCHEMA,
                    "trial_id": "MS-O1",
                }
            )
            + "\n"
        )
        state = multistream_execution.initialize_state(
            "MS-O1", "sha256:request", action_path
        )
        for target in multistream_execution.ORDERED_STATES[1:]:
            state = multistream_execution.advance_state(
                state,
                target,
                None if target == "materialized" else "evidence.json",
            )
        summary = multistream_execution.validate_state(
            state,
            trial_id="MS-O1",
            request_fingerprint="sha256:request",
            require_accepted=True,
        )
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["state"], "accepted")

    def test_model_source_snapshot_is_generated_from_explicit_file_list(self):
        source = self.root / "source"
        source.mkdir()
        (source / "model.py").write_text("value = 1\n")
        (source / "layers.py").write_text("value = 2\n")
        manifest = multistream_execution.generate_source_snapshot(
            source,
            self.root,
            "revision-1",
            ["model.py", "layers.py"],
        )
        self.assertEqual(
            [item["path"] for item in manifest["files"]],
            ["source/layers.py", "source/model.py"],
        )
        validated = multistream_execution.validate_source_snapshot(manifest, self.root)
        self.assertEqual(
            validated["source_fingerprint"], manifest["source_fingerprint"]
        )
