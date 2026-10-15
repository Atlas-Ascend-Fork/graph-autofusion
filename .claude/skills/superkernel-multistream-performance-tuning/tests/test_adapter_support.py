# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import multistream_adapter_support  # noqa: E402


class AdapterSupportTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        payloads = {
            "cleanup_evaluation": {
                "acceptance": {
                    "failure_injection_recovery": "passed",
                    "idempotent_replay": "passed",
                    "incumbent_unchanged": "passed",
                },
            },
            "critical_path_npu_closure_receipt": {
                "outcome": {"result_status": "no_gain"},
                "action_closed": True,
                "four_profile_closed": True,
            },
            "network_conformance_report": {
                "decision": "pass",
                "core_adapter_identity_leaks": [],
            },
            "real_npu_closure_receipt": {
                "outcome": {"result_status": "no_gain"},
            },
            "resource_screening_receipt": {
                "decision": "no_reorder_candidate",
                "stable_parent_match_count": 0,
            },
            "shared_npu_lease_inventory": {"status": "passed", "blockers": []},
        }
        self.artifacts = {}
        for kind, schema in multistream_adapter_support.ARTIFACT_SCHEMAS.items():
            path = self.root / f"{kind}.json"
            path.write_text(
                json.dumps({"schema_version": schema, **payloads[kind]}) + "\n"
            )
            self.artifacts[kind] = path.name

    def tearDown(self):
        self.temporary.cleanup()

    def _draft(self, *, adapter_id="full-v1", blocked=(), resource_screening=False):
        kinds = [
            "cleanup_evaluation",
            "network_conformance_report",
            "resource_screening_receipt"
            if resource_screening
            else "real_npu_closure_receipt",
            "shared_npu_lease_inventory",
        ]
        capabilities = []
        for name in multistream_adapter_support.CAPABILITIES:
            if name in blocked:
                capabilities.append(
                    {
                        "name": name,
                        "status": "unsupported",
                        "evidence_artifacts": [],
                        "blocker": {
                            "code": f"MISSING_{name.upper()}",
                            "detail": f"{name} is not integrated",
                        },
                    }
                )
            else:
                evidence = (
                    "resource_screening_receipt"
                    if resource_screening
                    else "real_npu_closure_receipt"
                )
                if name == "isolated_cleanup":
                    evidence = "cleanup_evaluation"
                elif name == "shared_npu_lease":
                    evidence = "shared_npu_lease_inventory"
                capabilities.append(
                    {
                        "name": name,
                        "status": "supported",
                        "evidence_artifacts": [evidence],
                        "blocker": None,
                    }
                )
        return {
            "schema_version": multistream_adapter_support.ADAPTER_SCHEMA,
            "adapter_id": adapter_id,
            "network_class": adapter_id.split("-")[0],
            "artifacts": [
                {"kind": kind, "path": self.artifacts[kind]} for kind in sorted(kinds)
            ],
            "capabilities": capabilities,
        }

    def _freeze(self, draft, name):
        adapter = multistream_adapter_support.freeze_adapter(draft, self.root)
        path = self.root / name
        path.write_text(json.dumps(adapter, sort_keys=True) + "\n")
        return adapter, path

    def _collection_plan(
        self,
        actions,
        *,
        adapter_id="full-v1",
        round_kind="multistream",
        omit_roles=(),
        omit_outputs=(),
    ):
        roles = sorted(
            (
                {
                    role
                    for action in actions
                    for role in multistream_adapter_support.COLLECTION_BINDING_REQUIREMENTS[
                        action
                    ]
                }
                | (
                    set(multistream_adapter_support.ROUND_COLLECTION_BINDINGS)
                    if round_kind in {"P", "FINAL"}
                    else set()
                )
            )
            - set(omit_roles)
        )
        outputs = sorted(
            (
                {
                    output
                    for action in actions
                    for output in multistream_adapter_support.COLLECTION_OUTPUT_REQUIREMENTS[
                        action
                    ]
                }
                | (
                    set(multistream_adapter_support.ROUND_EXPECTED_OUTPUTS)
                    if round_kind in {"P", "FINAL"}
                    else set()
                )
            )
            - set(omit_outputs)
        )
        source_fingerprint = "sha256:" + "1" * 64
        config_fingerprint = "sha256:" + "2" * 64
        control_fingerprint = "sha256:" + "3" * 64
        workload_fingerprint = "sha256:" + "4" * 64
        identities = {
            "config_identity": {"config_fingerprint": config_fingerprint},
            "control_identity": {
                "control_fingerprint": control_fingerprint,
                "round_kind": round_kind,
            },
            "source_identity": {
                "source_revision": "source-r1",
                "source_fingerprint": source_fingerprint,
            },
            "workload_identity": {"workload_fingerprint": workload_fingerprint},
        }
        for role, value in identities.items():
            (self.root / f"{role}.json").write_text(json.dumps(value) + "\n")
        for role in roles:
            path = self.root / f"{role}.json"
            if role == "model_run_spec":
                value = {
                    "schema_version": multistream_adapter_support.MODEL_RUN_SPEC_SCHEMA
                }
            elif role == "round_action_manifest":
                value = {
                    "schema_version": multistream_adapter_support.ROUND_ACTION_MANIFEST_SCHEMA,
                    "round_kind": round_kind,
                    "source_revision": "source-r1",
                    "requested_actions": sorted(actions),
                }
            else:
                value = {"provider": role}
            path.write_text(json.dumps(value) + "\n")
        return multistream_adapter_support.freeze_collection_plan(
            {
                "schema_version": multistream_adapter_support.COLLECTION_PLAN_SCHEMA,
                "plan_id": "collection-1",
                "adapter_id": adapter_id,
                "round_kind": round_kind,
                "source_revision": "source-r1",
                "source_fingerprint": source_fingerprint,
                "config_fingerprint": config_fingerprint,
                "control_fingerprint": control_fingerprint,
                "workload_fingerprint": workload_fingerprint,
                "requested_actions": sorted(actions),
                "experiment_artifacts": [
                    {"role": role, "path": f"{role}.json"}
                    for role in sorted(identities)
                ],
                "bindings": [{"role": role, "path": f"{role}.json"} for role in roles],
                "expected_outputs": outputs,
            },
            self.root,
        )

    def test_complete_evidence_makes_all_actions_available(self):
        _, path = self._freeze(self._draft(), "full.json")
        matrix = multistream_adapter_support.build_matrix(
            "matrix-1", [path.name], self.root
        )
        self.assertTrue(
            all(
                item["status"] == "available"
                for item in matrix["adapters"][0]["actions"]
            )
        )
        self.assertTrue(
            multistream_adapter_support.validate_matrix(matrix, self.root)["valid"]
        )

    def test_capability_blockers_are_reported_per_action(self):
        blocked = {"dependency_evidence", "model_execution", "source_exact_mapping"}
        _, path = self._freeze(
            self._draft(adapter_id="partial-v1", blocked=blocked), "partial.json"
        )
        matrix = multistream_adapter_support.build_matrix(
            "matrix-2", [path.name], self.root
        )
        actions = {item["action"]: item for item in matrix["adapters"][0]["actions"]}
        self.assertEqual(actions["option"]["status"], "blocked")
        reorder_codes = {
            item["code"]
            for item in actions["dependency_safe_operator_reorder"]["blockers"]
        }
        self.assertIn("MISSING_DEPENDENCY_EVIDENCE", reorder_codes)
        self.assertIn("MISSING_SOURCE_EXACT_MAPPING", reorder_codes)

    def test_resource_screening_adds_current_candidate_blocker(self):
        _, path = self._freeze(
            self._draft(adapter_id="screened-v1", resource_screening=True),
            "screened.json",
        )
        matrix = multistream_adapter_support.build_matrix(
            "matrix-3", [path.name], self.root
        )
        actions = {item["action"]: item for item in matrix["adapters"][0]["actions"]}
        self.assertEqual(actions["option"]["status"], "available")
        self.assertEqual(
            actions["dependency_safe_operator_reorder"]["status"], "blocked"
        )
        self.assertIn(
            "NO_STABLE_SAME_PARENT_PAIR",
            {
                item["code"]
                for item in actions["dependency_safe_operator_reorder"]["blockers"]
            },
        )

    def test_requires_complete_catalog_and_sealed_artifacts(self):
        draft = self._draft()
        draft["capabilities"] = draft["capabilities"][:-1]
        with self.assertRaisesRegex(ValueError, "complete ordered capability catalog"):
            multistream_adapter_support.freeze_adapter(draft, self.root)
        adapter, _ = self._freeze(self._draft(), "tampered.json")
        (self.root / self.artifacts["real_npu_closure_receipt"]).write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "content must use|sealed identity"):
            multistream_adapter_support.validate_adapter(adapter, self.root)

    def test_matrix_replay_detects_adapter_change(self):
        adapter, path = self._freeze(self._draft(), "full.json")
        matrix = multistream_adapter_support.build_matrix(
            "matrix-4", [path.name], self.root
        )
        adapter["adapter_id"] = "changed"
        path.write_text(json.dumps(adapter) + "\n")
        with self.assertRaisesRegex(ValueError, "fingerprint|sealed|differs"):
            multistream_adapter_support.validate_matrix(matrix, self.root)

    def test_v2_adapter_exposes_critical_path_actions_only_with_closure(self):
        draft = self._draft(adapter_id="critical-v2")
        draft["schema_version"] = multistream_adapter_support.ADAPTER_SCHEMA_V2
        draft["artifacts"].append(
            {
                "kind": "critical_path_npu_closure_receipt",
                "path": self.artifacts["critical_path_npu_closure_receipt"],
            }
        )
        draft["artifacts"] = sorted(draft["artifacts"], key=lambda item: item["kind"])
        closure_evidence = "critical_path_npu_closure_receipt"
        for name in multistream_adapter_support.CAPABILITIES_V2[
            len(multistream_adapter_support.CAPABILITIES) :
        ]:
            draft["capabilities"].append(
                {
                    "name": name,
                    "status": "supported",
                    "evidence_artifacts": [closure_evidence],
                    "blocker": None,
                }
            )
        _, path = self._freeze(draft, "critical-v2.json")
        matrix = multistream_adapter_support.build_matrix(
            "matrix-v2", [path.name], self.root
        )
        self.assertEqual(
            matrix["schema_version"], multistream_adapter_support.MATRIX_SCHEMA_V2
        )
        actions = {item["action"]: item for item in matrix["adapters"][0]["actions"]}
        self.assertEqual(actions["event_edge_refinement"]["status"], "available")
        self.assertEqual(actions["stage_split"]["status"], "available")
        self.assertTrue(
            multistream_adapter_support.validate_matrix(matrix, self.root)["valid"]
        )

    def test_v3_adapter_requires_component_specific_capabilities_and_collection_evidence(
        self,
    ):
        draft = self._draft(adapter_id="component-v3")
        draft["schema_version"] = multistream_adapter_support.ADAPTER_SCHEMA_V3
        for name in multistream_adapter_support.CAPABILITIES_V3[
            len(multistream_adapter_support.CAPABILITIES) :
        ]:
            draft["capabilities"].append(
                {
                    "name": name,
                    "status": "supported",
                    "evidence_artifacts": ["real_npu_closure_receipt"],
                    "blocker": None,
                }
            )
        _, path = self._freeze(draft, "component-v3.json")
        matrix = multistream_adapter_support.build_matrix(
            "matrix-component-v3", [path.name], self.root
        )
        actions = {item["action"]: item for item in matrix["adapters"][0]["actions"]}
        self.assertEqual(
            matrix["schema_version"], multistream_adapter_support.MATRIX_SCHEMA_V3
        )
        self.assertEqual(actions["component_overlap_reorder"]["status"], "available")

        plan = self._collection_plan(
            ["component_overlap_reorder"], adapter_id="component-v3"
        )
        receipt = multistream_adapter_support.build_preflight(
            "preflight-component",
            matrix,
            plan,
            self.root,
            "component-v3",
            ["component_overlap_reorder"],
        )
        self.assertEqual(receipt["status"], "ready")
        roles = {item["role"] for item in plan["bindings"]}
        outputs = set(plan["expected_outputs"])
        self.assertIn("component_reorder_materializer", roles)
        self.assertIn("component_source_mapping_provider", roles)
        self.assertIn("component_capture_provider", roles)
        self.assertIn("component_source_map", outputs)
        self.assertIn("bound_component_capture", outputs)
        self.assertIn("post_component_dispatch_evidence", outputs)

    def test_preflight_requires_available_actions_and_complete_collection_plan(self):
        _, path = self._freeze(self._draft(), "full.json")
        matrix = multistream_adapter_support.build_matrix(
            "matrix-preflight", [path.name], self.root
        )
        actions = ["dependency_safe_operator_reorder"]
        plan = self._collection_plan(actions)

        receipt = multistream_adapter_support.build_preflight(
            "preflight-1", matrix, plan, self.root, "full-v1", actions
        )

        self.assertEqual(receipt["status"], "ready")
        self.assertEqual(receipt["blockers"], [])
        self.assertTrue(
            multistream_adapter_support.validate_preflight(
                receipt, matrix, plan, self.root
            )["valid"]
        )

    def test_preflight_blocks_before_execution_when_collection_inputs_are_missing(self):
        _, path = self._freeze(self._draft(), "full.json")
        matrix = multistream_adapter_support.build_matrix(
            "matrix-preflight", [path.name], self.root
        )
        actions = ["dependency_safe_operator_reorder"]
        plan = self._collection_plan(
            actions,
            omit_roles={"dependency_provider"},
            omit_outputs={"bound_short_trace_analysis"},
        )

        receipt = multistream_adapter_support.build_preflight(
            "preflight-2", matrix, plan, self.root, "full-v1", actions
        )

        self.assertEqual(receipt["status"], "blocked")
        self.assertIn(
            ("MISSING_COLLECTION_BINDING", "dependency_provider"),
            {(item["code"], item["name"]) for item in receipt["blockers"]},
        )
        self.assertIn(
            ("MISSING_EXPECTED_OUTPUT", "bound_short_trace_analysis"),
            {(item["code"], item["name"]) for item in receipt["blockers"]},
        )

    def test_event_preflight_uses_event_evidence_not_operator_reorder_evidence(self):
        draft = self._draft(adapter_id="critical-v2")
        draft["schema_version"] = multistream_adapter_support.ADAPTER_SCHEMA_V2
        draft["artifacts"].append(
            {
                "kind": "critical_path_npu_closure_receipt",
                "path": self.artifacts["critical_path_npu_closure_receipt"],
            }
        )
        draft["artifacts"] = sorted(draft["artifacts"], key=lambda item: item["kind"])
        for name in multistream_adapter_support.CAPABILITIES_V2[
            len(multistream_adapter_support.CAPABILITIES) :
        ]:
            draft["capabilities"].append(
                {
                    "name": name,
                    "status": "supported",
                    "evidence_artifacts": ["critical_path_npu_closure_receipt"],
                    "blocker": None,
                }
            )
        _, path = self._freeze(draft, "critical-v2.json")
        matrix = multistream_adapter_support.build_matrix(
            "matrix-event-preflight", [path.name], self.root
        )
        actions = ["event_edge_refinement"]
        plan = self._collection_plan(actions)
        plan["adapter_id"] = "critical-v2"
        plan["collection_plan_fingerprint"] = multistream_adapter_support.fingerprint(
            {
                key: value
                for key, value in plan.items()
                if key != "collection_plan_fingerprint"
            }
        )

        receipt = multistream_adapter_support.build_preflight(
            "preflight-event", matrix, plan, self.root, "critical-v2", actions
        )

        self.assertEqual(receipt["status"], "ready")
        roles = set(
            multistream_adapter_support.COLLECTION_BINDING_REQUIREMENTS[actions[0]]
        )
        outputs = set(
            multistream_adapter_support.COLLECTION_OUTPUT_REQUIREMENTS[actions[0]]
        )
        self.assertNotIn("reorder_materializer", roles)
        self.assertNotIn("operator_order_capture", outputs)
        self.assertIn("event_materializer", roles)
        self.assertIn("event_stage_dispatch_evidence", outputs)

    def test_collection_identity_change_invalidates_preflight_replay(self):
        _, path = self._freeze(self._draft(), "full.json")
        matrix = multistream_adapter_support.build_matrix(
            "matrix-preflight", [path.name], self.root
        )
        actions = ["range_exclusion"]
        plan = self._collection_plan(actions)
        receipt = multistream_adapter_support.build_preflight(
            "preflight-identity", matrix, plan, self.root, "full-v1", actions
        )
        plan["source_revision"] = "source-r2"
        plan["collection_plan_fingerprint"] = multistream_adapter_support.fingerprint(
            {
                key: value
                for key, value in plan.items()
                if key != "collection_plan_fingerprint"
            }
        )

        with self.assertRaisesRegex(ValueError, "differs from deterministic replay"):
            multistream_adapter_support.validate_preflight(
                receipt, matrix, plan, self.root
            )

    def test_p_final_rounds_require_round_specific_actions_and_evidence(self):
        _, path = self._freeze(self._draft(), "full.json")
        matrix = multistream_adapter_support.build_matrix(
            "matrix-rounds", [path.name], self.root
        )
        cases = (
            ("P", ["range_exclusion"]),
            ("FINAL", ["range_exclusion", "scope_split"]),
        )
        for round_kind, actions in cases:
            with self.subTest(round_kind=round_kind):
                plan = self._collection_plan(actions, round_kind=round_kind)
                receipt = multistream_adapter_support.build_preflight(
                    f"preflight-{round_kind}",
                    matrix,
                    plan,
                    self.root,
                    "full-v1",
                    sorted(actions),
                )
                self.assertEqual(receipt["status"], "ready")
                self.assertEqual(receipt["round_kind"], round_kind)
        for round_kind in ("P", "FINAL"):
            with (
                self.subTest(round_kind=round_kind),
                self.assertRaisesRegex(ValueError, "forbids option action"),
            ):
                self._collection_plan(["option", "scope_split"], round_kind=round_kind)

    def test_round_preflight_blocks_when_profile_launcher_is_not_bound(self):
        _, path = self._freeze(self._draft(), "full.json")
        matrix = multistream_adapter_support.build_matrix(
            "matrix-round", [path.name], self.root
        )
        actions = ["range_exclusion"]
        plan = self._collection_plan(
            actions, round_kind="P", omit_roles={"profile_launcher"}
        )
        receipt = multistream_adapter_support.build_preflight(
            "preflight-P", matrix, plan, self.root, "full-v1", actions
        )
        self.assertEqual(receipt["status"], "blocked")
        self.assertIn(
            ("MISSING_COLLECTION_BINDING", "profile_launcher"),
            {(item["code"], item["name"]) for item in receipt["blockers"]},
        )

    def test_missing_adapter_and_v2_action_produce_blocked_receipts(self):
        _, path = self._freeze(self._draft(), "full.json")
        matrix = multistream_adapter_support.build_matrix(
            "matrix-v1", [path.name], self.root
        )
        reorder_plan = self._collection_plan(["dependency_safe_operator_reorder"])
        missing = multistream_adapter_support.build_preflight(
            "preflight-missing",
            matrix,
            reorder_plan,
            self.root,
            "missing-adapter",
            ["dependency_safe_operator_reorder"],
        )
        self.assertEqual(missing["status"], "blocked")
        self.assertEqual(missing["blockers"][0]["code"], "ADAPTER_NOT_IN_MATRIX")

        event_plan = self._collection_plan(["event_edge_refinement"])
        unavailable = multistream_adapter_support.build_preflight(
            "preflight-v2-action",
            matrix,
            event_plan,
            self.root,
            "full-v1",
            ["event_edge_refinement"],
        )
        self.assertEqual(unavailable["status"], "blocked")
        self.assertEqual(
            unavailable["blockers"][0]["code"], "ACTION_NOT_IN_MATRIX_SCHEMA"
        )

    def test_deleted_binding_after_freeze_produces_blocked_receipt(self):
        _, path = self._freeze(self._draft(), "full.json")
        matrix = multistream_adapter_support.build_matrix(
            "matrix-deleted", [path.name], self.root
        )
        actions = ["dependency_safe_operator_reorder"]
        plan = self._collection_plan(actions)
        (self.root / "dependency_provider.json").unlink()
        receipt = multistream_adapter_support.build_preflight(
            "preflight-deleted", matrix, plan, self.root, "full-v1", actions
        )
        self.assertEqual(receipt["status"], "blocked")
        self.assertEqual(receipt["blockers"][0]["code"], "COLLECTION_PLAN_INVALID")

    def test_cli_persists_blocked_receipt_and_returns_two(self):
        _, path = self._freeze(self._draft(), "full.json")
        matrix = multistream_adapter_support.build_matrix(
            "matrix-cli", [path.name], self.root
        )
        plan = self._collection_plan(["dependency_safe_operator_reorder"])
        matrix_path = self.root / "matrix.json"
        plan_path = self.root / "plan.json"
        output_path = self.root / "blocked-preflight.json"
        matrix_path.write_text(json.dumps(matrix) + "\n")
        plan_path.write_text(json.dumps(plan) + "\n")

        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "multistream_adapter_support.py"),
                "preflight",
                "--preflight-id",
                "preflight-cli",
                "--matrix",
                str(matrix_path),
                "--collection-plan",
                str(plan_path),
                "--adapter-id",
                "missing-adapter",
                "--action",
                "dependency_safe_operator_reorder",
                "--root",
                str(self.root),
                "--out",
                str(output_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertTrue(output_path.is_file())
        receipt = json.loads(output_path.read_text())
        self.assertEqual(receipt["status"], "blocked")
        self.assertTrue(
            multistream_adapter_support.validate_preflight(
                receipt, matrix, plan, self.root
            )["valid"]
        )


if __name__ == "__main__":
    unittest.main()
