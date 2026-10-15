# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib.util
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "auto_tune_session", SCRIPTS / "auto_tune_session.py"
)
auto_tune_session = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(auto_tune_session)


def phase_result(session, *, status=None, details_zh=None):
    step = auto_tune_session.next_step(session)
    if status is None:
        status = {
            "intake_preparation": "succeeded",
            "s0_baseline": "succeeded",
            "stage_a_scope_selection": "accepted",
            "stage_o_option_tuning": "no_gain",
            "base_profile_source_mapping": "succeeded",
            "optional_experiments": "not_requested"
            if step["optional_branch"] == "none"
            else "no_gain",
            "final_e2e_report": "accepted",
        }[step["phase"]]
    result = {
        "schema_version": "superkernel-auto-tune-phase-result-v1",
        "session_id": session["session_id"],
        "step_id": step["step_id"],
        "phase": step["phase"],
        "agent_id": step["agent_id"],
        "status": status,
        "summary_zh": f"{step['phase']} 阶段已结算。",
        "input_fingerprints": {"control": "a" * 64},
        "consumed_artifacts": [],
        "produced_artifacts": [],
        "decisions": [],
        "blockers": [],
        "next_step_guidance_zh": "交由控制 Agent 校验并调度下一阶段。",
        "details_zh": details_zh or {},
    }
    if step["phase"] == "final_e2e_report":
        ledger = Path(session["artifact_root"]) / "ledger.json"
        classification = {
            "accepted": "beneficial",
            "no_gain": "no_gain",
            "not_run": "not_run",
            "failed": "failed",
            "blocked": "blocked",
        }[status]
        final_e2e = {
            "classification": classification,
            "candidate_id": "Sbest",
            "scope_strategy": "explicit_candidates",
            "option_config": {"auto_op_parallel": True},
            "evidence_artifacts": [],
            "reason_zh": "依据最终 clean E2E 结算。",
        }
        if classification in {"beneficial", "no_gain"}:
            candidate = 9.0 if classification == "beneficial" else 10.0
            final_e2e.update(
                {
                    "baseline": {"median_ms": 10.0, "sample_count": 5},
                    "candidate": {"median_ms": candidate, "sample_count": 5},
                    "improvement_pct": (10.0 - candidate) / 10.0 * 100.0,
                }
            )
        ledger.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "final_e2e": final_e2e,
                    "experiments": {
                        "Sbest": {
                            "source_revision": "revision-abc",
                            "baseline_config_fingerprint": "config-123",
                            "rounds": [
                                {
                                    "round_id": "Sbest-BASE",
                                    "declared_option_changes": [
                                        {
                                            "pointer": "/auto_op_parallel",
                                            "after": True,
                                        }
                                    ],
                                    "lifecycle": {
                                        "correctness": "passed",
                                        "clean": "passed",
                                        "profiling": "passed",
                                    },
                                }
                            ],
                            "performance_scope_decisions": [
                                {"classification": "beneficial"},
                                {"classification": "neutral"},
                            ],
                            "blockers": [],
                        }
                    },
                },
                ensure_ascii=False,
            )
        )
        result["ledger_path"] = "ledger.json"
    return result


def canonical_fingerprint(value, *, ensure_ascii=False):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=ensure_ascii,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class AutoTuneSessionTest(unittest.TestCase):
    def report_summary(self):
        return {
            "schema_version": "superkernel-report-summary-v1",
            "baseline": {
                "metric": "TP worst-rank mean",
                "value_ms": 10.0,
                "run_count": 5,
                "reason_zh": "五次基线稳定。",
                "evidence_artifacts": ["evidence.json"],
            },
            "stages": [
                {
                    "stage": "stage_a",
                    "candidate_id": "S1",
                    "kind": "screening",
                    "metric": "TP worst-rank mean",
                    "baseline_ms": 10.0,
                    "candidate_ms": 11.0,
                    "run_count": 3,
                    "eligible": False,
                    "selector_rank": 1,
                    "status": "no_gain",
                    "reason_zh": "收益未达标。",
                    "gates_zh": "阈值 1%；P90 不通过；stddev 不通过",
                    "evidence_artifacts": ["evidence.json"],
                },
            ],
            "fallback": {
                "candidate_id": "S0",
                "scope_strategy": "SK-off",
                "reason_zh": "已验证并恢复 SK-off。",
                "evidence_artifacts": ["evidence.json"],
            },
        }

    def final_with_summary(self, directory, status="not_run", summary=None):
        _, session_path = self.initialize(directory)
        self.advance_to_final(session_path)
        session = auto_tune_session.load_session(session_path)
        result = phase_result(
            session,
            status=status,
            details_zh={key: "已结算。" for key in auto_tune_session.FINAL_DETAIL_KEYS},
        )
        root = Path(session["artifact_root"])
        (root / "evidence.json").write_text("{}")
        ledger_path = root / result["ledger_path"]
        ledger = json.loads(ledger_path.read_text())
        ledger["experiments"] = {}
        ledger["report_summary"] = (
            summary if summary is not None else self.report_summary()
        )
        ledger_path.write_text(json.dumps(ledger))
        return session_path, result

    def test_failure_report_starts_with_comparison_and_keeps_best_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            session_path, result = self.final_with_summary(directory)
            completed = auto_tune_session.seal_handoff(session_path, result)
            report = (
                Path(completed["artifact_root"]) / "FINAL_E2E_REPORT.md"
            ).read_text()
            self.assertLess(report.index("## 结果速览"), report.index("Session:"))
            self.assertLess(
                report.index("## 关键数据对比"), report.index("## 配置与建议")
            )
            self.assertIn("10.000000", report)
            self.assertIn("11.000000", report)
            self.assertIn("+1.000000", report)
            self.assertIn("-10.000000%", report)
            self.assertIn("未通过门禁，非优胜者", report)
            self.assertIn("优胜者：无", report)
            self.assertIn("最终 E2E 结论: 未执行", report)
            self.assertIn("P90 不通过", report)

    def test_summary_override_is_read_only_and_legacy_report_has_missing_data(self):
        with tempfile.TemporaryDirectory() as directory:
            _, session_path = self.initialize(directory)
            self.advance_to_final(session_path)
            completed = self.seal_current(
                session_path,
                status="not_run",
                details_zh={
                    key: "已结算。" for key in auto_tune_session.FINAL_DETAIL_KEYS
                },
            )
            root = Path(completed["artifact_root"])
            self.assertIn("N/A", (root / "FINAL_E2E_REPORT.md").read_text())
            sealed_before = {path: path.read_bytes() for path in root.rglob("*.json")}
            (root / "evidence.json").write_text("{}")
            summary_path = root / "summary.json"
            summary_path.write_text(json.dumps(self.report_summary()))
            output = root / "REPORT.summary.md"
            process = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "auto_tune_session.py"),
                    "render-final-report",
                    "--session",
                    str(session_path),
                    "--summary",
                    str(summary_path),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
            self.assertIn("11.000000", output.read_text())
            for path, content in sealed_before.items():
                self.assertEqual(path.read_bytes(), content)

    def test_summary_rejects_wrong_metrics_missing_evidence_and_fake_final(self):
        for field, value in (
            ("metric", "pooled P50"),
            ("baseline_ms", 20.0),
            ("stage", "final_e2e"),
            ("kind", "final_clean"),
            ("candidate_ms", float("nan")),
            ("run_count", True),
            ("evidence_artifacts", ["../outside.json"]),
            ("evidence_artifacts", ["missing.json"]),
        ):
            with (
                self.subTest(field=field, value=value),
                tempfile.TemporaryDirectory() as directory,
            ):
                summary = self.report_summary()
                summary["stages"][0][field] = value
                session_path, result = self.final_with_summary(
                    directory, summary=summary
                )
                with self.assertRaisesRegex(ValueError, "report_summary"):
                    auto_tune_session.seal_handoff(session_path, result)
                self.assertNotEqual(
                    auto_tune_session.load_session(session_path)["status"], "completed"
                )

    def test_success_summary_binds_winner_to_final_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = self.report_summary()
            summary["winner"] = {
                "candidate_id": "Sbest",
                "scope_strategy": "explicit_candidates",
                "promotion_path": "whole-scope",
                "option_config": {"auto_op_parallel": True},
                "debug_option_config": {},
                "config_path": "evidence.json",
                "config_fingerprint": "config-123",
                "reason_zh": "最终验证通过。",
                "evidence_artifacts": ["evidence.json"],
            }
            session_path, result = self.final_with_summary(
                directory, "accepted", summary
            )
            completed = auto_tune_session.seal_handoff(session_path, result)
            report = (
                Path(completed["artifact_root"]) / "FINAL_E2E_REPORT.md"
            ).read_text()
            for text in (
                "优胜者：`Sbest`",
                "whole-scope",
                "config-123",
                "Debug options: `{}`",
                "+10.000000%",
            ):
                self.assertIn(text, report)
            summary["winner"]["candidate_id"] = "fake-winner"
            summary_path = Path(completed["artifact_root"]) / "override.json"
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "report_summary"):
                auto_tune_session.render_final_report(
                    session_path, summary_path=summary_path
                )

    def test_summary_uses_selector_rank_and_accepted_candidate_before_best_attempt(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            summary = self.report_summary()
            other = dict(
                summary["stages"][0],
                candidate_id="S2",
                selector_rank=2,
                candidate_ms=10.5,
            )
            summary["stages"].append(other)
            session_path, result = self.final_with_summary(directory, summary=summary)
            completed = auto_tune_session.seal_handoff(session_path, result)
            report = (
                Path(completed["artifact_root"]) / "FINAL_E2E_REPORT.md"
            ).read_text()
            self.assertIn("stage_a / S1", report)
            self.assertNotIn("stage_a / S2", report)
            other.update(eligible=True, status="accepted")
            summary_path = Path(completed["artifact_root"]) / "override.json"
            summary_path.write_text(json.dumps(summary))
            output = auto_tune_session.render_final_report(
                session_path, summary_path=summary_path
            )
            self.assertIn("stage_a / S2", output.read_text())

    def test_missing_nested_option_map_is_not_reported_as_effective_options(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = self.report_summary()
            summary["fallback"]["option_config"] = {"super_kernel_debug_options": {}}
            session_path, result = self.final_with_summary(directory, summary=summary)
            completed = auto_tune_session.seal_handoff(session_path, result)
            report = (
                Path(completed["artifact_root"]) / "FINAL_E2E_REPORT.md"
            ).read_text()
            self.assertIn("Optimize options: N/A", report)
            self.assertIn("Debug options: `{}`", report)

    def test_final_run_count_does_not_reuse_sample_count(self):
        with tempfile.TemporaryDirectory() as directory:
            session_path, result = self.final_with_summary(directory, "accepted")
            root = session_path.parent
            ledger_path = root / result["ledger_path"]
            ledger = json.loads(ledger_path.read_text())
            ledger["final_e2e"]["candidate"].update(run_count=3, sample_count=120)
            ledger_path.write_text(json.dumps(ledger))
            auto_tune_session.seal_handoff(session_path, result)
            report = (root / "FINAL_E2E_REPORT.md").read_text()
            final_row = next(
                line for line in report.splitlines() if line.startswith("| 最终 E2E")
            )
            self.assertIn("| 3 |", final_row)
            self.assertNotIn("120", final_row)

    def test_summary_rejects_symlink_evidence_and_inconsistent_improvement(self):
        for failure in ("symlink", "improvement"):
            with (
                self.subTest(failure=failure),
                tempfile.TemporaryDirectory() as directory,
            ):
                summary = self.report_summary()
                if failure == "improvement":
                    summary["stages"][0]["improvement_pct"] = 10.0
                session_path, result = self.final_with_summary(
                    directory, summary=summary
                )
                if failure == "symlink":
                    evidence = session_path.parent / "evidence.json"
                    evidence.unlink()
                    outside = Path(directory) / "outside.json"
                    outside.write_text("{}")
                    evidence.symlink_to(outside)
                with self.assertRaisesRegex(ValueError, "report_summary"):
                    auto_tune_session.seal_handoff(session_path, result)

    def test_summary_table_escapes_markdown_and_keeps_profile_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = self.report_summary()
            summary["stages"][0]["reason_zh"] = "失败|原因\n下一行"
            profile = dict(
                summary["stages"][0],
                stage="base_profile_source_mapping",
                kind="profiling",
                metric="rank0 profile mean",
                baseline_ms=20.0,
                candidate_ms=15.0,
            )
            summary["stages"].append(profile)
            session_path, result = self.final_with_summary(directory, summary=summary)
            completed = auto_tune_session.seal_handoff(session_path, result)
            report = (
                Path(completed["artifact_root"]) / "FINAL_E2E_REPORT.md"
            ).read_text()
            self.assertIn(r"失败\|原因<br>下一行", report)
            self.assertIn("仅诊断，非优胜者", report)
            self.assertIn("最终 E2E 结论: 未执行", report)

    def initialize(self, directory, mode="none"):
        root = Path(directory) / "artifacts"
        session_path = root / "auto-tune-session.json"
        return auto_tune_session.initialize_session(
            session_path=session_path,
            artifact_root=root,
            session_id="session-20260905-a",
            optional_mode=mode,
        ), session_path

    def seal_current(self, session_path, *, status=None, details_zh=None):
        session = auto_tune_session.load_session(session_path)
        return auto_tune_session.seal_handoff(
            session_path, phase_result(session, status=status, details_zh=details_zh)
        )

    def advance_to_final(self, session_path):
        while True:
            session = auto_tune_session.load_session(session_path)
            step = auto_tune_session.next_step(session)
            if step["phase"] == "final_e2e_report":
                return
            auto_tune_session.seal_handoff(session_path, phase_result(session))

    def completed_parent_with_import_evidence(self, directory):
        parent_root = Path(directory) / "parent-artifacts"
        session_path = parent_root / "auto-tune-session.json"
        auto_tune_session.initialize_session(
            session_path=session_path,
            artifact_root=parent_root,
            session_id="parent-session",
            optional_mode="none",
        )
        evidence_root = parent_root / "evidence"
        evidence_root.mkdir(parents=True)
        analysis = {
            "schema_version": "1.2",
            "experiment_id": "Sbest",
            "round_id": "Sbest-BASE",
            "candidate_name": "Sbest",
            "source_revision": "revision-abc",
            "summary_zh": "BASE 独立性能分析已完成。",
        }
        analysis["analysis_content_fingerprint"] = canonical_fingerprint(
            analysis, ensure_ascii=True
        )
        analysis_path = evidence_root / "profiling-analysis.json"
        analysis_path.write_text(
            json.dumps(analysis, ensure_ascii=False), encoding="utf-8"
        )

        source_map = {
            "schema_version": "2.0",
            "protocol": "source_scope_map_v2",
            "provenance": {"stable_marker_revision": "revision-abc"},
            "source_ranges": [
                {
                    "range_id": "range-1",
                    "relation": "exact_cover",
                    "source_scope": "model.layers[0]",
                }
            ],
        }
        source_map["provenance"]["source_scope_map_content_fingerprint"] = (
            canonical_fingerprint(source_map)
        )
        source_map_path = evidence_root / "source-scope-map.json"
        source_map_path.write_text(
            json.dumps(source_map, ensure_ascii=False), encoding="utf-8"
        )

        ledger = {
            "schema_version": 2,
            "experiments": {
                "Sbest": {
                    "source_revision": "revision-abc",
                    "rounds": [{"round_id": "Sbest-BASE"}],
                    "blockers": [],
                }
            },
        }
        ledger_path = evidence_root / "experiment-ledger.json"
        ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

        imported_paths = [
            str(path.relative_to(parent_root))
            for path in (analysis_path, source_map_path, ledger_path)
        ]
        while True:
            session = auto_tune_session.load_session(session_path)
            step = auto_tune_session.next_step(session)
            result = phase_result(
                session,
                details_zh=(
                    {key: "已结算。" for key in auto_tune_session.FINAL_DETAIL_KEYS}
                    if step["phase"] == "final_e2e_report"
                    else None
                ),
            )
            if step["phase"] == "base_profile_source_mapping":
                result["produced_artifacts"] = imported_paths
            auto_tune_session.seal_handoff(session_path, result)
            if step["phase"] == "final_e2e_report":
                break
        return {
            "session_path": session_path,
            "root": parent_root,
            "analysis": analysis_path,
            "source_map": source_map_path,
            "ledger": ledger_path,
        }

    def derive_from_parent(self, directory, parent, mode="source-range"):
        child_root = Path(directory) / f"child-{mode}"
        child_session = child_root / "auto-tune-session.json"
        return auto_tune_session.derive_session(
            parent_session_path=parent["session_path"],
            session_path=child_session,
            artifact_root=child_root,
            session_id=f"derived-{mode}",
            optional_mode=mode,
            profiling_analysis=parent["analysis"],
            ledger=parent["ledger"],
            source_scope_map=parent["source_map"],
            approve_imported_evidence=True,
        ), child_session

    def test_source_range_from_smap_derives_an_independent_verified_session(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = self.completed_parent_with_import_evidence(directory)
            parent_before = parent["session_path"].read_bytes()

            session, session_path = self.derive_from_parent(directory, parent)

            self.assertEqual(parent["session_path"].read_bytes(), parent_before)
            self.assertEqual(session["entrypoint"], "source-range-from-smap")
            self.assertEqual(
                [step["step_id"] for step in session["steps"]],
                ["optional-source-range", "final-e2e-report"],
            )
            self.assertEqual(
                auto_tune_session.next_step(session)["step_id"], "optional-source-range"
            )
            self.assertTrue(auto_tune_session.verify_session(session_path)["valid"])
            manifest_path = (
                Path(session["artifact_root"]) / session["evidence_import"]["path"]
            )
            manifest = json.loads(manifest_path.read_text())
            self.assertTrue(manifest["authorization"]["approved"])
            self.assertEqual(manifest["parent"]["session_id"], "parent-session")
            task = auto_tune_session.create_dispatch_task(session_path)
            self.assertEqual(task["evidence_import"], session["evidence_import"])
            auto_tune_session.seal_handoff(
                session_path, phase_result(session, status="no_gain")
            )
            final_session = auto_tune_session.load_session(session_path)
            auto_tune_session.seal_handoff(
                session_path,
                phase_result(
                    final_session,
                    status="accepted",
                    details_zh={
                        key: "已从导入证据和派生会话结算。"
                        for key in auto_tune_session.FINAL_DETAIL_KEYS
                    },
                ),
            )
            report = (
                Path(session["artifact_root"]) / "FINAL_E2E_REPORT.md"
            ).read_text()
            self.assertIn("Entrypoint: `source-range-from-smap`", report)
            self.assertIn("Imported parent: `parent-session`", report)
            self.assertEqual(parent["session_path"].read_bytes(), parent_before)

    def test_derivation_requires_explicit_approval_before_writing_child(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = self.completed_parent_with_import_evidence(directory)
            child_root = Path(directory) / "unapproved-child"
            with self.assertRaisesRegex(ValueError, "explicit approval"):
                auto_tune_session.derive_session(
                    parent_session_path=parent["session_path"],
                    session_path=child_root / "auto-tune-session.json",
                    artifact_root=child_root,
                    session_id="unapproved",
                    optional_mode="source-range",
                    profiling_analysis=parent["analysis"],
                    ledger=parent["ledger"],
                    source_scope_map=parent["source_map"],
                    approve_imported_evidence=False,
                )
            self.assertFalse(child_root.exists())

    def test_imported_evidence_is_digest_bound_on_every_resume_and_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = self.completed_parent_with_import_evidence(directory)
            _, session_path = self.derive_from_parent(directory, parent)
            parent["analysis"].write_text("{}\n", encoding="utf-8")

            validation = auto_tune_session.verify_session(session_path)
            self.assertFalse(validation["valid"])
            self.assertIn("digest mismatch", validation["errors"][0])
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                auto_tune_session.create_dispatch_task(session_path)

    def test_source_range_derivation_rejects_non_exact_smap(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = self.completed_parent_with_import_evidence(directory)
            source_map = json.loads(parent["source_map"].read_text())
            source_map["source_ranges"][0]["relation"] = "diagnostic_only"
            source_map["provenance"].pop("source_scope_map_content_fingerprint")
            source_map["provenance"]["source_scope_map_content_fingerprint"] = (
                canonical_fingerprint(source_map)
            )
            parent["source_map"].write_text(json.dumps(source_map), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "exact_cover"):
                self.derive_from_parent(directory, parent)

    def test_derived_both_branch_still_rebases_after_accepted_multistream(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = self.completed_parent_with_import_evidence(directory)
            session, session_path = self.derive_from_parent(
                directory, parent, mode="both"
            )
            self.assertEqual(
                [step["step_id"] for step in session["steps"]],
                ["optional-multistream", "optional-source-range", "final-e2e-report"],
            )
            result = phase_result(session, status="accepted")
            settled = auto_tune_session.seal_handoff(session_path, result)
            self.assertEqual(
                [step["step_id"] for step in settled["steps"]],
                [
                    "optional-multistream",
                    "base-profile-derived",
                    "optional-source-range",
                    "final-e2e-report",
                ],
            )
            self.assertEqual(
                auto_tune_session.next_step(settled)["step_id"], "base-profile-derived"
            )

    def test_multistream_derivation_uses_the_same_import_without_requiring_smap(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = self.completed_parent_with_import_evidence(directory)
            child_root = Path(directory) / "child-multistream-only"
            session = auto_tune_session.derive_session(
                parent_session_path=parent["session_path"],
                session_path=child_root / "auto-tune-session.json",
                artifact_root=child_root,
                session_id="derived-multistream-only",
                optional_mode="multistream",
                profiling_analysis=parent["analysis"],
                ledger=parent["ledger"],
                approve_imported_evidence=True,
            )
            self.assertEqual(session["entrypoint"], "optional-from-base")
            self.assertEqual(
                [step["step_id"] for step in session["steps"]],
                ["optional-multistream", "final-e2e-report"],
            )
            manifest = json.loads(
                (child_root / session["evidence_import"]["path"]).read_text()
            )
            self.assertNotIn("source_scope_map", manifest["artifacts"])

    def test_source_range_derivation_rejects_an_incomplete_parent_session(self):
        with tempfile.TemporaryDirectory() as directory:
            parent_root = Path(directory) / "incomplete-parent"
            parent_session = parent_root / "auto-tune-session.json"
            auto_tune_session.initialize_session(
                session_path=parent_session,
                artifact_root=parent_root,
                session_id="incomplete-parent",
                optional_mode="none",
            )
            child_root = Path(directory) / "child"
            with self.assertRaisesRegex(ValueError, "completed"):
                auto_tune_session.derive_session(
                    parent_session_path=parent_session,
                    session_path=child_root / "auto-tune-session.json",
                    artifact_root=child_root,
                    session_id="derived-from-incomplete",
                    optional_mode="source-range",
                    profiling_analysis=parent_root / "missing-analysis.json",
                    ledger=parent_root / "missing-ledger.json",
                    source_scope_map=parent_root / "missing-map.json",
                    approve_imported_evidence=True,
                )

    def test_initializes_serial_schedule_and_records_not_requested_optional_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            session, session_path = self.initialize(directory)

            self.assertEqual(
                session["schema_version"], "superkernel-auto-tune-session-v1"
            )
            self.assertEqual(session["optional_mode"], "none")
            self.assertEqual(
                [step["phase"] for step in session["steps"]],
                [
                    "intake_preparation",
                    "s0_baseline",
                    "stage_a_scope_selection",
                    "stage_o_option_tuning",
                    "base_profile_source_mapping",
                    "optional_experiments",
                    "final_e2e_report",
                ],
            )
            self.assertEqual(
                auto_tune_session.next_step(
                    auto_tune_session.load_session(session_path)
                )["agent_id"],
                "sk-intake-preparation",
            )

    def test_rejects_out_of_order_or_impersonated_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            session, session_path = self.initialize(directory)
            handoff = phase_result(session)
            handoff["agent_id"] = "sk-s0-baseline"

            with self.assertRaisesRegex(ValueError, "agent_id"):
                auto_tune_session.seal_handoff(session_path, handoff)

            self.assertEqual(
                auto_tune_session.next_step(
                    auto_tune_session.load_session(session_path)
                )["phase"],
                "intake_preparation",
            )

    def test_external_handoff_cannot_spoof_the_controller_generated_bypass(self):
        with tempfile.TemporaryDirectory() as directory:
            session, session_path = self.initialize(directory)
            handoff = phase_result(session, status="not_run")
            handoff["system_generated_by"] = "superkernel-auto-tune"

            with self.assertRaisesRegex(ValueError, "reserved"):
                auto_tune_session.seal_handoff(session_path, handoff)
            self.assertEqual(
                auto_tune_session.next_step(
                    auto_tune_session.load_session(session_path)
                )["step_id"],
                "intake-preparation",
            )

    def test_rejects_a_tampered_session_schedule_before_dispatch_or_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            _, session_path = self.initialize(directory)
            tampered = json.loads(session_path.read_text())
            tampered["steps"] = [tampered["steps"][-1]]
            session_path.write_text(json.dumps(tampered), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "canonical schedule"):
                auto_tune_session.load_session(session_path)
            with self.assertRaisesRegex(ValueError, "canonical schedule"):
                auto_tune_session.resume_session(session_path)

    def test_seals_a_handoff_and_generates_immutable_phase_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            _, session_path = self.initialize(directory)
            before = auto_tune_session.load_session(session_path)
            handoff = phase_result(before)
            session = auto_tune_session.seal_handoff(session_path, handoff)
            sealed = Path(session["artifact_root"]) / "phases" / "intake-preparation"

            self.assertTrue((sealed / "phase-result.json").is_file())
            self.assertTrue((sealed / "PHASE_REPORT.md").is_file())
            with self.assertRaisesRegex(ValueError, "already sealed"):
                auto_tune_session.seal_handoff(
                    session_path,
                    handoff,
                )

    def test_both_mode_inserts_derived_base_only_after_accepted_multistream(self):
        with tempfile.TemporaryDirectory() as directory:
            _, session_path = self.initialize(directory, mode="both")
            for _ in range(5):
                self.seal_current(session_path)
            self.assertEqual(
                auto_tune_session.next_step(
                    auto_tune_session.load_session(session_path)
                )["step_id"],
                "optional-multistream",
            )

            self.seal_current(session_path, status="accepted")
            inserted = auto_tune_session.next_step(
                auto_tune_session.load_session(session_path)
            )
            self.assertEqual(inserted["step_id"], "base-profile-derived")
            self.assertEqual(inserted["phase"], "base_profile_source_mapping")
            self.assertTrue(inserted["derived_family_rebase"])

            self.seal_current(session_path)
            self.assertEqual(
                auto_tune_session.next_step(
                    auto_tune_session.load_session(session_path)
                )["step_id"],
                "optional-source-range",
            )

    def test_both_mode_no_gain_keeps_incumbent_and_skips_derived_base(self):
        with tempfile.TemporaryDirectory() as directory:
            _, session_path = self.initialize(directory, mode="both")
            for _ in range(5):
                self.seal_current(session_path)
            self.seal_current(session_path, status="no_gain")

            self.assertEqual(
                auto_tune_session.next_step(
                    auto_tune_session.load_session(session_path)
                )["step_id"],
                "optional-source-range",
            )

    def test_both_mode_rejects_nonaccepted_multistream_status(self):
        with tempfile.TemporaryDirectory() as directory:
            _, session_path = self.initialize(directory, mode="both")
            for _ in range(5):
                self.seal_current(session_path)

            with self.assertRaisesRegex(ValueError, "optional_experiments"):
                self.seal_current(session_path, status="succeeded")

    def test_dispatch_creates_a_structured_task_for_only_the_current_agent(self):
        with tempfile.TemporaryDirectory() as directory:
            _, session_path = self.initialize(directory)
            task = auto_tune_session.create_dispatch_task(session_path)

            self.assertEqual(
                task["schema_version"], "superkernel-auto-tune-phase-task-v1"
            )
            self.assertEqual(task["step"]["agent_id"], "sk-intake-preparation")
            self.assertEqual(
                task["handoff"]["result_schema"], auto_tune_session.RESULT_SCHEMA
            )
            self.seal_current(session_path)
            with self.assertRaisesRegex(ValueError, "stale dispatch task"):
                auto_tune_session.validate_dispatch_task(task, session_path)

    def test_run_agent_step_invokes_host_and_records_a_sealed_dispatch_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            _, session_path = self.initialize(directory)
            runner = Path(directory) / "fake_agent.py"
            runner.write_text(
                "import argparse,json\n"
                "p=argparse.ArgumentParser(); p.add_argument('--task'); p.add_argument('--result'); a=p.parse_args()\n"
                "task=json.load(open(a.task)); step=task['step']\n"
                "result={'schema_version':'superkernel-auto-tune-phase-result-v1','session_id':task['session_id'],'step_id':step['step_id'],'phase':step['phase'],'agent_id':step['agent_id'],'status':'succeeded','summary_zh':'Intake 完成。','input_fingerprints':{'task':task['session_fingerprint']},'consumed_artifacts':[],'produced_artifacts':[],'decisions':[],'blockers':[],'next_step_guidance_zh':'进入 S0。','details_zh':{}}\n"
                "json.dump(result,open(a.result,'w'))\n",
                encoding="utf-8",
            )

            session = auto_tune_session.run_agent_step(
                session_path, [sys.executable, str(runner)]
            )

            self.assertEqual(
                auto_tune_session.next_step(session)["phase"], "s0_baseline"
            )
            receipts = list(
                (Path(session["artifact_root"]) / "dispatches").glob(
                    "*/dispatch-receipt.json"
                )
            )
            self.assertEqual(len(receipts), 1)
            receipt = json.loads(receipts[0].read_text())
            self.assertEqual(receipt["agent_id"], "sk-intake-preparation")
            self.assertEqual(receipt["runner_status"], "passed")

    def test_failed_agent_host_attempt_preserves_scene_and_can_be_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            _, session_path = self.initialize(directory)
            with self.assertRaisesRegex(ValueError, "status 7"):
                auto_tune_session.run_agent_step(
                    session_path, [sys.executable, "-c", "import sys;sys.exit(7)"]
                )

            current = auto_tune_session.load_session(session_path)
            self.assertEqual(
                auto_tune_session.next_step(current)["step_id"], "intake-preparation"
            )
            receipt_path = next(
                (Path(current["artifact_root"]) / "dispatches").glob(
                    "*/dispatch-receipt.json"
                )
            )
            self.assertEqual(
                json.loads(receipt_path.read_text())["runner_status"], "failed"
            )

            runner = Path(directory) / "retry_agent.py"
            runner.write_text(
                "import argparse,json\n"
                "p=argparse.ArgumentParser(); p.add_argument('--task'); p.add_argument('--result'); a=p.parse_args()\n"
                "task=json.load(open(a.task)); step=task['step']\n"
                "result={'schema_version':'superkernel-auto-tune-phase-result-v1','session_id':task['session_id'],'step_id':step['step_id'],'phase':step['phase'],'agent_id':step['agent_id'],'status':'succeeded','summary_zh':'重试成功。','input_fingerprints':{'task':task['session_fingerprint']},'consumed_artifacts':[],'produced_artifacts':[],'decisions':[],'blockers':[],'next_step_guidance_zh':'进入下一阶段。','details_zh':{}}\n"
                "json.dump(result,open(a.result,'w'))\n",
                encoding="utf-8",
            )
            recovered = auto_tune_session.run_agent_step(
                session_path, [sys.executable, str(runner)]
            )
            self.assertEqual(
                auto_tune_session.next_step(recovered)["phase"], "s0_baseline"
            )
            self.assertTrue(auto_tune_session.verify_session(session_path)["valid"])

    def test_wrong_agent_host_result_is_archived_without_advancing_session(self):
        with tempfile.TemporaryDirectory() as directory:
            _, session_path = self.initialize(directory)
            runner = Path(directory) / "wrong_agent.py"
            runner.write_text(
                "import argparse,json\n"
                "p=argparse.ArgumentParser(); p.add_argument('--task'); p.add_argument('--result'); a=p.parse_args()\n"
                "task=json.load(open(a.task)); step=task['step']\n"
                "result={'schema_version':'superkernel-auto-tune-phase-result-v1','session_id':task['session_id'],'step_id':step['step_id'],'phase':step['phase'],'agent_id':'sk-s0-baseline','status':'succeeded','summary_zh':'错误 Agent。','input_fingerprints':{},'consumed_artifacts':[],'produced_artifacts':[],'decisions':[],'blockers':[],'next_step_guidance_zh':'不应推进。','details_zh':{}}\n"
                "json.dump(result,open(a.result,'w'))\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "agent_id"):
                auto_tune_session.run_agent_step(
                    session_path, [sys.executable, str(runner)]
                )
            session = auto_tune_session.load_session(session_path)
            self.assertEqual(
                auto_tune_session.next_step(session)["step_id"], "intake-preparation"
            )
            receipt = json.loads(
                next(
                    (Path(session["artifact_root"]) / "dispatches").glob(
                        "*/dispatch-receipt.json"
                    )
                ).read_text()
            )
            self.assertEqual(receipt["runner_status"], "failed")
            self.assertIn("agent_id", receipt["error"])

    def test_fake_agent_host_runs_the_complete_both_branch_orchestration(self):
        with tempfile.TemporaryDirectory() as directory:
            _, session_path = self.initialize(directory, mode="both")
            runner = Path(directory) / "orchestration_agent.py"
            runner.write_text(
                "import argparse,json,pathlib\n"
                "p=argparse.ArgumentParser(); p.add_argument('--task'); p.add_argument('--result'); a=p.parse_args()\n"
                "task=json.load(open(a.task)); step=task['step']; phase=step['phase']\n"
                "status={'intake_preparation':'succeeded','s0_baseline':'succeeded','stage_a_scope_selection':'accepted','stage_o_option_tuning':'no_gain','base_profile_source_mapping':'succeeded','final_e2e_report':'accepted'}.get(phase)\n"
                "status=('accepted' if step.get('optional_branch')=='multistream' else 'no_gain') if phase=='optional_experiments' else status\n"
                "details={}\n"
                "result={'schema_version':'superkernel-auto-tune-phase-result-v1','session_id':task['session_id'],'step_id':step['step_id'],'phase':phase,'agent_id':step['agent_id'],'status':status,'summary_zh':phase+' 已完成。','input_fingerprints':{'task':task['session_fingerprint']},'consumed_artifacts':[],'produced_artifacts':[],'decisions':[],'blockers':[],'next_step_guidance_zh':'交回控制 Agent。','details_zh':details}\n"
                "if phase=='final_e2e_report':\n"
                " details.update({k:'已从 ledger 汇总。' for k in ['environment','s0','stage_a','stage_o','base_profile_source_mapping','optional_experiments','final_e2e']})\n"
                " ledger={'schema_version':2,'experiments':{},'final_e2e':{'classification':'beneficial','candidate_id':'Sbest-derived','scope_strategy':'explicit_candidates','option_config':{'auto_op_parallel':True},'evidence_artifacts':[],'reason_zh':'最终 clean E2E 有收益。','baseline':{'median_ms':10.0,'sample_count':5},'candidate':{'median_ms':9.0,'sample_count':5},'improvement_pct':10.0}}\n"
                " pathlib.Path(task['artifact_root'],'ledger.json').write_text(json.dumps(ledger))\n"
                " result['ledger_path']='ledger.json'\n"
                "json.dump(result,open(a.result,'w'))\n",
                encoding="utf-8",
            )

            while auto_tune_session.load_session(session_path)["status"] != "completed":
                auto_tune_session.run_agent_step(
                    session_path, [sys.executable, str(runner)]
                )

            session = auto_tune_session.load_session(session_path)
            self.assertEqual(
                [step["step_id"] for step in session["steps"]],
                [
                    "intake-preparation",
                    "s0-baseline",
                    "stage-a-scope-selection",
                    "stage-o-option-tuning",
                    "base-profile-source-mapping",
                    "optional-multistream",
                    "base-profile-derived",
                    "optional-source-range",
                    "final-e2e-report",
                ],
            )
            self.assertTrue(
                all("dispatch_receipt_path" in step for step in session["steps"])
            )
            self.assertTrue(auto_tune_session.verify_session(session_path)["valid"])
            report = (
                Path(session["artifact_root"]) / "FINAL_E2E_REPORT.md"
            ).read_text()
            self.assertIn("最终 E2E 结论: 有收益", report)
            self.assertIn("Sbest-derived", report)
            self.assertIn("10.000000%", report)

    def test_controller_and_stages_use_runtime_common_ownership(self):
        root = Path(__file__).resolve().parents[2]
        runtime = root / "superkernel-runtime-common"
        self.assertTrue((runtime / "scripts" / "analyze_performance.py").is_file())
        self.assertFalse(
            (
                root / "superkernel-auto-tune" / "scripts" / "analyze_performance.py"
            ).is_file()
        )
        for skill in set(
            auto_tune_session.PHASES[phase][1] for phase in auto_tune_session.PHASES
        ):
            manifest = json.loads((root / skill / "phase.json").read_text())
            self.assertEqual(manifest["skill"], skill)
            self.assertTrue(manifest["runtime_tools"])

    def test_each_stage_has_an_executable_task_validating_entrypoint(self):
        with tempfile.TemporaryDirectory() as directory:
            _, session_path = self.initialize(directory, mode="both")
            root = Path(__file__).resolve().parents[2]
            visited = set()
            while True:
                session = auto_tune_session.load_session(session_path)
                step = auto_tune_session.next_step(session)
                task_path = Path(directory) / f"{step['step_id']}-task.json"
                auto_tune_session.write_dispatch_task(session_path, task_path)
                executable = root / step["skill"] / "scripts" / "execute_phase.py"
                completed = subprocess.run(
                    ["python3", str(executable), "--task", str(task_path)],
                    check=False,
                    capture_output=True,
                    text=True,
                )

                self.assertEqual(completed.returncode, 0, completed.stderr)
                plan = json.loads(completed.stdout)
                self.assertEqual(plan["phase"], step["phase"])
                self.assertEqual(plan["agent_id"], step["agent_id"])
                self.assertTrue(plan["commands"])
                visited.add(step["skill"])
                if step["phase"] == "final_e2e_report":
                    break
                auto_tune_session.seal_handoff(session_path, phase_result(session))
            self.assertEqual(
                visited,
                set(
                    auto_tune_session.PHASES[phase][1]
                    for phase in auto_tune_session.PHASES
                ),
            )

    def test_stage_entrypoint_executes_only_an_owned_tool_after_task_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            _, session_path = self.initialize(directory)
            task_path = Path(directory) / "intake-task.json"
            auto_tune_session.write_dispatch_task(session_path, task_path)
            executable = (
                Path(__file__).resolve().parents[2]
                / "superkernel-intake-preparation"
                / "scripts"
                / "execute_phase.py"
            )
            completed = subprocess.run(
                [
                    "python3",
                    str(executable),
                    "--task",
                    str(task_path),
                    "--tool",
                    "check_environment.py",
                    "--lease-root",
                    str(Path(directory) / "leases"),
                    "--device-id",
                    "0",
                    "--",
                    "--help",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn('"status": "passed"', completed.stdout)

    def test_complete_both_lifecycle_records_every_stage_and_final_ledger_report(self):
        with tempfile.TemporaryDirectory() as directory:
            _, session_path = self.initialize(directory, mode="both")
            while True:
                session = auto_tune_session.load_session(session_path)
                step = auto_tune_session.next_step(session)
                if step["phase"] == "final_e2e_report":
                    details = {
                        key: f"{key} 已从会话和 ledger 汇总。"
                        for key in auto_tune_session.FINAL_DETAIL_KEYS
                    }
                    finished = self.seal_current(
                        session_path, status="accepted", details_zh=details
                    )
                    break
                status = (
                    "accepted" if step["step_id"] == "optional-multistream" else None
                )
                auto_tune_session.seal_handoff(
                    session_path, phase_result(session, status=status)
                )

            self.assertEqual(finished["status"], "completed")
            self.assertEqual(
                [step["step_id"] for step in finished["steps"]],
                [
                    "intake-preparation",
                    "s0-baseline",
                    "stage-a-scope-selection",
                    "stage-o-option-tuning",
                    "base-profile-source-mapping",
                    "optional-multistream",
                    "base-profile-derived",
                    "optional-source-range",
                    "final-e2e-report",
                ],
            )
            self.assertTrue(
                all(step["state"] == "sealed" for step in finished["steps"])
            )
            self.assertTrue(auto_tune_session.verify_session(session_path)["valid"])

    def test_s0_blocker_short_circuits_to_final_and_seals_not_run_phase_scenes(self):
        with tempfile.TemporaryDirectory() as directory:
            _, session_path = self.initialize(directory)
            self.seal_current(session_path)
            session = self.seal_current(session_path, status="blocked")

            self.assertEqual(
                auto_tune_session.next_step(session)["phase"], "final_e2e_report"
            )
            skipped = [
                step
                for step in session["steps"]
                if step["step_id"]
                in {
                    "stage-a-scope-selection",
                    "stage-o-option-tuning",
                    "base-profile-source-mapping",
                    "optional-experiments",
                }
            ]
            self.assertTrue(all(step["state"] == "sealed" for step in skipped))
            for step in skipped:
                record = Path(session["artifact_root"]) / step["result_path"]
                expected_status = (
                    "not_requested"
                    if step["step_id"] == "optional-experiments"
                    else "not_run"
                )
                self.assertEqual(
                    json.loads(record.read_text())["status"], expected_status
                )
                self.assertTrue((record.parent / "PHASE_REPORT.md").is_file())

    def test_every_required_failure_gate_short_circuits_and_preserves_phase_scenes(
        self,
    ):
        cases = (
            ("intake_preparation", "blocked"),
            ("s0_baseline", "failed"),
            ("stage_a_scope_selection", "no_gain"),
            ("stage_o_option_tuning", "blocked"),
            ("base_profile_source_mapping", "invalid"),
        )
        with tempfile.TemporaryDirectory() as directory:
            for phase, status in cases:
                with self.subTest(phase=phase, status=status):
                    root = Path(directory) / phase
                    _, session_path = (
                        auto_tune_session.initialize_session(
                            session_path=root / "session.json",
                            artifact_root=root,
                            session_id=f"gate-{phase}",
                            optional_mode="none",
                        ),
                        root / "session.json",
                    )
                    while (
                        auto_tune_session.next_step(
                            auto_tune_session.load_session(session_path)
                        )["phase"]
                        != phase
                    ):
                        session = auto_tune_session.load_session(session_path)
                        auto_tune_session.seal_handoff(
                            session_path, phase_result(session)
                        )
                    session = auto_tune_session.load_session(session_path)
                    settled = auto_tune_session.seal_handoff(
                        session_path, phase_result(session, status=status)
                    )
                    self.assertEqual(
                        auto_tune_session.next_step(settled)["phase"],
                        "final_e2e_report",
                    )
                    self.assertTrue(
                        auto_tune_session.verify_session(session_path)["valid"]
                    )
                    completed = self.seal_current(
                        session_path,
                        status="not_run",
                        details_zh={
                            key: "前置失败，最终未执行。"
                            for key in auto_tune_session.FINAL_DETAIL_KEYS
                        },
                    )
                    report = (
                        Path(completed["artifact_root"]) / "FINAL_E2E_REPORT.md"
                    ).read_text()
                    self.assertIn("## 关键数据对比", report)
                    self.assertIn("最终 E2E 结论: 未执行", report)
                    self.assertTrue(
                        auto_tune_session.verify_session(session_path)["valid"]
                    )

    def test_budget_and_final_failure_keep_available_baseline_and_report(self):
        for status, reason in (
            ("not_run", "预算耗尽，未运行最终验证。"),
            ("failed", "最终验证正确性失败。"),
            ("blocked", "设备阻塞。"),
            ("no_gain", "最终性能无收益。"),
        ):
            with (
                self.subTest(status=status),
                tempfile.TemporaryDirectory() as directory,
            ):
                session_path, result = self.final_with_summary(directory, status)
                result["summary_zh"] = reason
                completed = auto_tune_session.seal_handoff(session_path, result)
                report = (
                    Path(completed["artifact_root"]) / "FINAL_E2E_REPORT.md"
                ).read_text()
                self.assertIn(reason, report)
                self.assertIn("10.000000", report)
                self.assertIn("优胜者：无", report)
                self.assertTrue(auto_tune_session.verify_session(session_path)["valid"])

    def test_resume_returns_the_same_pending_step_after_a_process_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            _, session_path = self.initialize(directory)
            self.seal_current(session_path)

            resumed = auto_tune_session.resume_session(session_path)
            self.assertEqual(resumed["session_id"], "session-20260905-a")
            self.assertEqual(resumed["next_step"]["step_id"], "s0-baseline")

    def test_final_stage_requires_complete_chinese_terminal_sections_and_finishes_session(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            _, session_path = self.initialize(directory)
            self.advance_to_final(session_path)
            required = {
                "environment": "环境确认完成。",
                "s0": "S0 已结算。",
                "stage_a": "框定方式已结算。",
                "stage_o": "Option 已结算。",
                "base_profile_source_mapping": "BASE/SMAP 已结算。",
                "optional_experiments": "未请求可选实验。",
                "final_e2e": "没有合法 clean E2E，记录 not_run。",
            }
            session = self.seal_current(
                session_path, status="not_run", details_zh=required
            )

            self.assertEqual(session["status"], "completed")
            report = Path(session["artifact_root"]) / "FINAL_E2E_REPORT.md"
            self.assertTrue(report.is_file())
            self.assertIn("没有合法 clean E2E", report.read_text())
            self.assertIn("Ledger 汇总", report.read_text())
            self.assertIn("Sbest-BASE", report.read_text())
            self.assertIn("revision-abc", report.read_text())
            self.assertIn("beneficial=1", report.read_text())
            self.assertIn("/auto_op_parallel", report.read_text())
            self.assertIn("自动结论", report.read_text())
            self.assertIn("最终 E2E 结论: 未执行", report.read_text())
            verified = auto_tune_session.verify_session(session_path)
            self.assertTrue(verified["valid"], json.dumps(verified, ensure_ascii=False))

    def test_final_stage_rejects_missing_terminal_section(self):
        with tempfile.TemporaryDirectory() as directory:
            _, session_path = self.initialize(directory)
            self.advance_to_final(session_path)
            with self.assertRaisesRegex(ValueError, "details_zh"):
                self.seal_current(session_path, details_zh={"environment": "环境"})

    def test_final_stage_rejects_missing_or_invalid_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            _, session_path = self.initialize(directory)
            self.advance_to_final(session_path)
            session = auto_tune_session.load_session(session_path)
            result = phase_result(
                session,
                details_zh={
                    key: "已结算。" for key in auto_tune_session.FINAL_DETAIL_KEYS
                },
            )
            result.pop("ledger_path")
            with self.assertRaisesRegex(ValueError, "ledger_path"):
                auto_tune_session.seal_handoff(session_path, result)

            result["ledger_path"] = "../outside.json"
            with self.assertRaisesRegex(ValueError, "ledger_path"):
                auto_tune_session.seal_handoff(session_path, result)

    def test_final_stage_rejects_a_ledger_without_structured_final_e2e(self):
        with tempfile.TemporaryDirectory() as directory:
            _, session_path = self.initialize(directory)
            self.advance_to_final(session_path)
            session = auto_tune_session.load_session(session_path)
            result = phase_result(
                session,
                status="accepted",
                details_zh={
                    key: "已结算。" for key in auto_tune_session.FINAL_DETAIL_KEYS
                },
            )
            ledger = Path(session["artifact_root"]) / result["ledger_path"]
            payload = json.loads(ledger.read_text())
            payload.pop("final_e2e", None)
            ledger.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "final_e2e"):
                auto_tune_session.seal_handoff(session_path, result)

    def test_final_e2e_supports_every_terminal_class_and_rejects_metric_drift(self):
        cases = {
            "accepted": "有收益",
            "no_gain": "无收益",
            "not_run": "未执行",
            "failed": "失败或阻塞",
            "blocked": "失败或阻塞",
        }
        with tempfile.TemporaryDirectory() as directory:
            for status, label in cases.items():
                with self.subTest(status=status):
                    root = Path(directory) / status
                    _, session_path = self.initialize(root)
                    self.advance_to_final(session_path)
                    session = auto_tune_session.load_session(session_path)
                    result = phase_result(
                        session,
                        status=status,
                        details_zh={
                            key: "已结算。"
                            for key in auto_tune_session.FINAL_DETAIL_KEYS
                        },
                    )
                    completed = auto_tune_session.seal_handoff(session_path, result)
                    self.assertEqual(completed["status"], "completed")
                    report = (
                        Path(completed["artifact_root"]) / "FINAL_E2E_REPORT.md"
                    ).read_text()
                    self.assertIn(f"最终 E2E 结论: {label}", report)

            root = Path(directory) / "drift"
            _, session_path = self.initialize(root)
            self.advance_to_final(session_path)
            session = auto_tune_session.load_session(session_path)
            result = phase_result(
                session,
                status="accepted",
                details_zh={
                    key: "已结算。" for key in auto_tune_session.FINAL_DETAIL_KEYS
                },
            )
            ledger_path = Path(session["artifact_root"]) / result["ledger_path"]
            ledger = json.loads(ledger_path.read_text())
            ledger["final_e2e"]["improvement_pct"] = 99.0
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "improvement_pct"):
                auto_tune_session.seal_handoff(session_path, result)


if __name__ == "__main__":
    unittest.main()
