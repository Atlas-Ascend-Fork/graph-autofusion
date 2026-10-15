# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import generate_original_fused_inventory as generator  # noqa: E402 - load sibling scripts after sys.path setup
from projected_trace_mapping import graph_occurrence_fingerprint  # noqa: E402 - load sibling scripts after sys.path setup
from source_calibration_common import canonical_sha256  # noqa: E402 - load sibling scripts after sys.path setup


class GenerateOriginalFusedInventoryTest(unittest.TestCase):
    def test_emits_only_groups_with_a_unique_proven_block(self):
        trace = {
            "protocol": "kernel_projection_trace_v2",
            "mappings": [
                {
                    "status": "exact",
                    "device_id": 0,
                    "model_id": 48,
                    "sk_id": 7,
                    "candidate_source_scope": "decode",
                    "ordered_child_node_keys": ["original-a"],
                    "ordered_child_ops": ["Add"],
                    "metadata_fingerprint": "a" * 64,
                },
                {
                    "status": "exact",
                    "device_id": 0,
                    "model_id": 48,
                    "sk_id": 8,
                    "candidate_source_scope": "decode",
                    "ordered_child_node_keys": ["original-b"],
                    "ordered_child_ops": ["Mul"],
                    "metadata_fingerprint": "b" * 64,
                },
            ],
        }
        trace["mapping_fingerprint"] = canonical_sha256(trace)
        projection = {
            "protocol": "source_calibration_projection_v2",
            "occurrences": [
                {
                    "original_node_key": "original-a",
                    "calibration_node_key": "calibration-a",
                    "original": ["role-a", 1, "Add", "AI_VECTOR_CORE"],
                },
                {
                    "original_node_key": "original-b",
                    "calibration_node_key": "calibration-b",
                    "original": ["role-a", 2, "Mul", "AI_VECTOR_CORE"],
                },
            ],
        }
        projection["projection_fingerprint"] = canonical_sha256(projection)
        blocks = {
            "instances": [
                {
                    "block_instance_id": "0",
                    "runtime_scope_binding": {
                        "method": "exact_scope_name",
                        "value": "skcal.block.0",
                    },
                }
            ]
        }
        assignments = {
            "protocol": "calibration_unit_assignments_v1",
            "assignments": [
                {
                    "calibration_node_key": "calibration-a",
                    "block_instance_id": "0",
                }
            ],
        }

        inventory, evidence = generator.build_inventory(
            trace, projection, blocks, assignments
        )

        self.assertEqual(inventory["generation_summary"]["input_sk_group_count"], 2)
        self.assertEqual(inventory["generation_summary"]["emitted_sk_group_count"], 1)
        self.assertEqual(
            inventory["generation_summary"]["skip_reasons"],
            {"source_block_unassigned": 1},
        )
        self.assertEqual(inventory["sk_groups"][0]["source_scope"], "skcal.block.0")
        self.assertEqual(inventory["sk_groups"][0]["candidate_source_scope"], "decode")
        self.assertEqual(inventory["sk_groups"][0]["block_local_sk_ordinal"], 0)
        self.assertEqual(
            inventory["sk_groups"][0]["sk_occurrence_fingerprint"],
            graph_occurrence_fingerprint(
                trace["mapping_fingerprint"],
                device_id=0,
                model_id=48,
                sk_id=7,
                ordered_child_node_keys=["original-a"],
            ),
        )
        self.assertEqual(len(evidence["records"]), 1)


if __name__ == "__main__":
    unittest.main()
