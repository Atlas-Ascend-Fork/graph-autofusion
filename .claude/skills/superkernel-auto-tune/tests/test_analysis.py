# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import csv
import io
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[2] / "superkernel-runtime-common" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import analyze_performance  # noqa: E402 - load sibling scripts after sys.path setup
import experiment_ledger  # noqa: E402 - load sibling scripts after sys.path setup
import analyze_sk_meta  # noqa: E402 - load sibling scripts after sys.path setup
import check_environment  # noqa: E402 - load sibling scripts after sys.path setup
import recommend_sk_strategy  # noqa: E402 - load sibling scripts after sys.path setup
import render_round_report  # noqa: E402 - load sibling scripts after sys.path setup


class EnvironmentProbeTest(unittest.TestCase):
    def test_default_option_samples_do_not_assume_grouped_matmul_symbols(self):
        default_samples = {
            "optimize_options": check_environment.OPTIMIZE_OPTION_SAMPLES,
            "debug_options": check_environment.DEBUG_OPTION_SAMPLES,
        }

        self.assertNotIn("GroupedMatmul", json.dumps(default_samples))
        self.assertEqual(
            check_environment.OPTIMIZE_OPTION_SAMPLES["dcci_before_kernel_start"],
            [[".*"]],
        )
        self.assertEqual(
            check_environment.DEBUG_OPTION_SAMPLES["debug_dcci_disable_on_kernel"],
            [[".*"]],
        )

    def test_custom_option_samples_can_add_an_exact_kernel_regex(self):
        custom_regex = ["^static_kernel_current_graph$"]

        merged = check_environment._merge_probe_samples(
            check_environment.OPTIMIZE_OPTION_SAMPLES,
            {"dcci_before_kernel_start": [custom_regex]},
        )

        self.assertEqual(
            merged["dcci_before_kernel_start"],
            [[".*"], custom_regex],
        )

    def test_explicit_none_exclusion_uses_python_none_in_balanced_pair(self):
        calls = []

        class Npu:
            @staticmethod
            def super_kernel_scope_begin(scope_name):
                calls.append(("begin", scope_name))

            @staticmethod
            def super_kernel_scope_end(scope_name):
                calls.append(("end", scope_name))

        result = check_environment._probe_explicit_none_exclusion(Npu())

        self.assertTrue(result["accepted"])
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(calls, [("begin", None), ("end", None)])
        self.assertEqual(result["semantic_validation"], "not_proven_by_api_probe")

    def test_explicit_none_exclusion_reports_wrapper_rejection(self):
        class Npu:
            @staticmethod
            def super_kernel_scope_begin(scope_name):
                raise RuntimeError("None scope unsupported")

            @staticmethod
            def super_kernel_scope_end(scope_name):
                raise AssertionError("end must not run after begin rejection")

        result = check_environment._probe_explicit_none_exclusion(Npu())

        self.assertFalse(result["accepted"])
        self.assertEqual(result["status"], "rejected")
        self.assertIn("unsupported", result["error"])

    def test_requires_cann_home_and_loader_paths_before_real_torch_import(self):
        with self.assertRaisesRegex(
            check_environment.RuntimeEnvironmentError,
            "Option probing was not run",
        ):
            check_environment._require_cann_loader_environment(
                {
                    "LD_LIBRARY_PATH": "/usr/local/Ascend/driver/lib64",
                    "PATH": "/usr/bin",
                }
            )

    def test_accepts_environment_loaded_by_cann_setenv(self):
        result = check_environment._require_cann_loader_environment(
            {
                "ASCEND_HOME_PATH": "/usr/local/Ascend/cann",
                "LD_LIBRARY_PATH": (
                    "/usr/local/Ascend/cann/lib64:"
                    "/usr/local/Ascend/cann/x86_64-linux/lib64"
                ),
                "PATH": "/usr/local/Ascend/cann/bin:/usr/bin",
            }
        )

        self.assertTrue(result["loaded"])
        self.assertEqual(len(result["cann_library_paths"]), 2)

    def test_cli_marks_missing_runtime_as_not_run_instead_of_rejected(self):
        output = io.StringIO()
        with (
            mock.patch.object(
                check_environment,
                "collect_environment",
                side_effect=check_environment.RuntimeEnvironmentError("missing CANN"),
            ),
            redirect_stdout(output),
        ):
            exit_code = check_environment.main(["--json"])

        report = json.loads(output.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertEqual(report["error_code"], "cann_environment_not_loaded")
        self.assertEqual(report["option_probe_status"], "not_run")


def _write_performance_runs(root, desired_worst_rank_means):
    root.mkdir()
    for run_number, desired_mean in enumerate(desired_worst_rank_means, start=1):
        run_path = root / f"run-{run_number}"
        run_path.mkdir()
        (run_path / "config.yaml").write_text(
            "model_config:\n"
            "  custom_params:\n"
            "    super_kernel_optimize_options: {}\n"
            "    super_kernel_debug_options: {}\n"
        )
        samples = (
            desired_mean
            if isinstance(desired_mean, (list, tuple))
            else [desired_mean] * 9
        )
        for rank in range(8):
            (run_path / f"log_{rank}.log").write_text(
                "\n".join(f"Inference time (decode): {sample} ms" for sample in samples)
                + "\n"
            )


def _screening_matrix(*, executed=("S1", "S2", "S3"), s4_blocker=None):
    strategies = (
        ("S1", "automatic_aot"),
        ("S2", "broad_decode"),
        ("S3", "per_block"),
        ("S4", "semantic_segment"),
    )
    entries = []
    for candidate_id, strategy_kind in strategies:
        entry = {
            "id": candidate_id,
            "strategy_kind": strategy_kind,
            "only_change": f"test {strategy_kind}",
            "status": "executed" if candidate_id in executed else "blocked",
        }
        if entry["status"] == "blocked":
            entry["blocker"] = s4_blocker or {
                "code": "semantic_boundary_unavailable",
                "reason_zh": "源码无法建立保持依赖的语义分段边界。",
                "evidence": "intake/S4-BLOCKER.md",
            }
        entries.append(entry)
    return {
        "schema_version": "superkernel-screening-matrix-v2",
        "candidate_set_frozen_before_execution": True,
        "candidates": entries,
    }


def _write_fused_metadata(
    root,
    scope,
    op_sequence,
    *,
    function_token,
    node_start,
    model_directory="model_1_1",
):
    model = root / model_directory
    model.mkdir(parents=True)
    first_op = op_sequence[0]
    last_op = op_sequence[-1]
    function = (
        f"sk_17_{scope}_start_static_kernel_{first_op}_{function_token}_"
        f"end_static_kernel_{last_op}_{function_token}"
    )
    lines = [f"SK Function: {function}, scope id: 17, Node Count: {len(op_sequence)}"]
    for offset, op_type in enumerate(op_sequence):
        lines.append(
            f"[nodeId:{node_start + offset}, streamId:1] - "
            f"KernelInfos{{funcName:static_kernel_{op_type}_{function_token}, "
            "kernelType:AIV_ONLY, numBlocks:1, cubeNum:0, vecNum:1, "
            "isScheModeOn:0}}"
        )
    (model / "sk_fused_nodes.log").write_text("\n".join(lines) + "\n")


def _write_config_manifest(path, config):
    path.write_text(json.dumps(config, sort_keys=True) + "\n")
    return path


def _write_replay_evidence(
    path,
    *,
    candidate_name="S1",
    reproducible=True,
    config_fingerprint_match=True,
    distinct_roots=True,
    matched_effective_group_count=1,
):
    root = path.parent / path.stem
    compat_root = root / "compat"
    verify_root = root / "verify" if distinct_roots else compat_root
    child_ops = ["A", "B", "C", "D", "E"]
    _write_fused_metadata(
        compat_root, "scope", child_ops, function_token="compat", node_start=1
    )
    if distinct_roots:
        _write_fused_metadata(
            verify_root, "scope", child_ops, function_token="verify", node_start=100
        )
    compat_config = _write_config_manifest(root / "compat-config.json", {"scope": "S"})
    verify_config = _write_config_manifest(
        root / "verify-config.json",
        {"scope": "S" if config_fingerprint_match else "different"},
    )
    replay = analyze_sk_meta.compare_deep_fusion_replay(
        compat_root,
        verify_root,
        compat_config_manifest=compat_config,
        verify_config_manifest=verify_config,
        candidate_name=candidate_name,
        round_id=f"{candidate_name}-BASE",
        source_revision="fixture-source-revision",
        artifact_base=path.parent,
    )
    if not reproducible:
        replay["deep_fusion_reproducible"] = False
    if matched_effective_group_count != replay["matched_effective_group_count"]:
        replay["matched_effective_group_count"] = matched_effective_group_count
    path.write_text(json.dumps(replay) + "\n")
    return path


class PerformanceAnalysisTest(unittest.TestCase):
    def test_stage_a_config_requires_explicit_empty_option_maps(self):
        cases = {
            "missing_optimize": (
                "    super_kernel_debug_options: {}\n",
                "explicitly define super_kernel_optimize_options",
            ),
            "missing_debug": (
                "    super_kernel_optimize_options: {}\n",
                "explicitly define super_kernel_debug_options",
            ),
            "nonempty_optimize": (
                "    super_kernel_optimize_options:\n"
                "      dcci_disable_on_kernel: ['.*']\n"
                "    super_kernel_debug_options: {}\n",
                "super_kernel_optimize_options must be an explicitly empty mapping",
            ),
            "nonempty_debug": (
                "    super_kernel_optimize_options: {}\n"
                "    super_kernel_debug_options:\n"
                "      debug_sync_all: 1\n",
                "super_kernel_debug_options must be an explicitly empty mapping",
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.yaml"
            valid.write_text(
                "model_config:\n"
                "  custom_params:\n"
                "    super_kernel_optimize_options: {}\n"
                "    super_kernel_debug_options: {}\n"
            )
            result = analyze_performance.validate_stage_a_config(valid)
            self.assertTrue(result["explicitly_empty"])

            for name, (custom_params, error) in cases.items():
                with self.subTest(name=name):
                    config = root / f"{name}.yaml"
                    config.write_text(
                        "model_config:\n  custom_params:\n" + custom_params
                    )
                    with self.assertRaisesRegex(ValueError, error):
                        analyze_performance.validate_stage_a_config(config)

    def test_decimal_five_percent_baseline_spread_is_stable(self):
        summary = {
            "run_count": 5,
            "run_worst_rank_means_ms": [9.75, 9.875, 10.0, 10.125, 10.25],
        }

        result = analyze_performance.evaluate_baseline_stability(summary)

        self.assertTrue(result["stable"])
        self.assertAlmostEqual(result["spread_ms"], 0.5)
        self.assertAlmostEqual(result["mean_ms"], 10.0)
        self.assertAlmostEqual(result["spread_pct"], 5.0)

    def test_absolute_half_ms_spread_can_be_unstable_when_relative_spread_exceeds_five_percent(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "S0"
            _write_performance_runs(baseline, [9.6, 9.7, 9.8, 9.9, 10.1])
            summary = analyze_performance.summarize_candidate(baseline)

        result = analyze_performance.evaluate_baseline_stability(summary)

        self.assertFalse(result["stable"])
        self.assertEqual(result["required_run_count"], 5)
        self.assertEqual(result["run_worst_rank_means_ms"], [9.6, 9.7, 9.8, 9.9, 10.1])
        self.assertAlmostEqual(result["spread_ms"], 0.5)
        self.assertGreater(result["spread_pct"], 5.0)

    def test_raw_log_decimal_five_percent_spread_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "S0"
            _write_performance_runs(
                baseline,
                [
                    [9.70, 9.80] * 9,
                    [9.825, 9.925] * 9,
                    [9.95, 10.05] * 9,
                    [10.075, 10.175] * 9,
                    [10.20, 10.30] * 9,
                ],
            )
            summary = analyze_performance.summarize_candidate(baseline)

        result = analyze_performance.evaluate_baseline_stability(summary)

        self.assertTrue(result["stable"])
        self.assertEqual(result["spread_ms"], 0.5)
        self.assertEqual(result["spread_pct"], 5.0)
        self.assertEqual(
            summary["run_worst_rank_means_exact_ms"],
            ["9.75", "9.875", "10.00", "10.125", "10.25"],
        )

    def test_baseline_spread_over_five_percent_is_unstable(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "S0"
            _write_performance_runs(baseline, [10.0, 10.1, 10.2, 10.3, 10.52])
            summary = analyze_performance.summarize_candidate(baseline)

        result = analyze_performance.evaluate_baseline_stability(summary)

        self.assertFalse(result["stable"])
        self.assertGreater(result["spread_pct"], 5.0)

    def test_faster_three_run_candidate_is_not_promoted_when_baseline_is_unstable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            candidate = root / "candidate"
            _write_performance_runs(baseline, [10.0, 10.1, 10.2, 10.3, 10.52])
            _write_performance_runs(candidate, [8.0, 8.1, 8.2])

            evidence = _write_replay_evidence(root / "S1-replay.json")
            report = analyze_performance.compare_candidates(
                baseline, {"S1": candidate}, replay_evidence={"S1": evidence}
            )

        promotion = report["candidates"]["S1"]["promotion"]
        self.assertFalse(report["baseline_stability"]["stable"])
        self.assertFalse(promotion["promoted"])
        self.assertFalse(promotion["checks"]["baseline_stable"])

    def test_forged_baseline_stability_cannot_promote_unstable_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            candidate = root / "candidate"
            _write_performance_runs(baseline, [10.0, 10.1, 10.2, 10.3, 10.52])
            _write_performance_runs(candidate, [8.0, 8.1, 8.2])
            baseline_summary = analyze_performance.summarize_candidate(baseline)
            candidate_summary = analyze_performance.summarize_candidate(candidate)

        promotion = analyze_performance.evaluate_promotion(
            baseline_summary, candidate_summary
        )

        self.assertFalse(promotion["checks"]["baseline_stable"])
        self.assertFalse(promotion["promoted"])

    def test_single_run_candidate_cannot_promote_and_missing_replay_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            candidate = root / "candidate"
            _write_performance_runs(baseline, [10.0, 10.1, 10.2, 10.3, 10.4])
            _write_performance_runs(candidate, [8.0])

            candidate_summary = analyze_performance.summarize_candidate(candidate)
            baseline_summary = analyze_performance.summarize_candidate(baseline)
            with self.assertRaisesRegex(ValueError, "replay evidence"):
                analyze_performance.compare_candidates(baseline, {"S1": candidate})

        promotion = analyze_performance.evaluate_promotion(
            baseline_summary, candidate_summary
        )
        self.assertFalse(promotion["promoted"])
        self.assertFalse(promotion["checks"]["candidate_min_runs"])
        self.assertFalse(promotion["checks"]["deep_fusion_replay"])

    def test_clean_promotion_requires_passing_candidate_replay_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            candidate = root / "candidate"
            evidence = _write_replay_evidence(root / "S1-replay.json")
            _write_performance_runs(baseline, [10.0, 10.1, 10.2, 10.3, 10.4])
            _write_performance_runs(candidate, [8.0, 8.0, 8.0])

            report = analyze_performance.compare_candidates(
                baseline,
                {"S1": candidate},
                replay_evidence={"S1": evidence},
            )

        promotion = report["candidates"]["S1"]["promotion"]
        self.assertTrue(report["candidates"]["S1"]["deep_fusion_replay"]["valid"])
        self.assertTrue(promotion["checks"]["deep_fusion_replay"])
        self.assertTrue(promotion["promoted"])

    def test_whole_scope_clean_promotion_does_not_require_replay_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            candidate = root / "candidate"
            _write_performance_runs(baseline, [10.0, 10.1, 10.2, 10.3, 10.4])
            _write_performance_runs(candidate, [8.0, 8.0, 8.0])

            report = analyze_performance.compare_candidates(
                baseline,
                {"S2": candidate},
                promotion_modes={"S2": "whole_scope"},
            )

        summary = report["candidates"]["S2"]
        self.assertEqual(summary["promotion_mode"], "whole_scope")
        self.assertEqual(summary["deep_fusion_replay"]["status"], "not_required")
        self.assertTrue(summary["promotion"]["checks"]["deep_fusion_replay"])
        self.assertTrue(summary["promotion"]["promoted"])

    def test_selection_only_ranks_clean_candidates_without_promoting_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            s1 = root / "S1"
            s2 = root / "S2"
            s3 = root / "S3"
            _write_performance_runs(baseline, [10.0, 10.1, 10.2, 10.3, 10.4])
            _write_performance_runs(s1, [9.4, 9.4, 9.4])
            _write_performance_runs(s2, [8.0, 8.0, 8.0])
            _write_performance_runs(s3, [10.3, 10.3, 10.3])

            report = analyze_performance.compare_candidates(
                baseline,
                {"S1": s1, "S2": s2, "S3": s3},
                selection_only=True,
                screening_matrix=_screening_matrix(),
            )

        self.assertEqual(report["selection"]["stage"], "candidate_screening")
        self.assertEqual(report["selection"]["selected_for_deep_analysis"], "S2")
        self.assertEqual(report["selection"]["recommended"], "S2")
        self.assertEqual(report["selection"]["promoted_candidates"], [])
        self.assertEqual(
            report["selection"]["final_promotion_decision"],
            "deferred_until_winner_profiling",
        )
        self.assertEqual(report["selection"]["ranked_candidates"], ["S2", "S1", "S3"])
        self.assertEqual(
            report["selection"]["ranked_eligible_candidates"], ["S2", "S1"]
        )
        self.assertTrue(report["candidates"]["S2"]["screening_eligibility"]["eligible"])
        self.assertNotIn("promotion", report["candidates"]["S2"])
        self.assertNotIn("deep_fusion_replay", report["candidates"]["S2"])
        self.assertTrue(report["screening_matrix"]["settlement_complete"])
        self.assertTrue(report["stage_a_option_controls"]["all_explicitly_empty"])
        self.assertEqual(
            report["screening_matrix"]["executed_candidate_ids"],
            ["S1", "S2", "S3"],
        )
        self.assertEqual(
            report["screening_matrix"]["nonexecuted_candidates"][0]["id"],
            "S4",
        )

    def test_selection_only_rejects_missing_or_nonempty_stage_a_options(self):
        mutations = {
            "missing": (
                "model_config:\n  custom_params:\n    super_kernel_debug_options: {}\n",
                "explicitly define super_kernel_optimize_options",
            ),
            "nonempty": (
                "model_config:\n"
                "  custom_params:\n"
                "    super_kernel_optimize_options:\n"
                "      dcci_disable_on_kernel: ['.*']\n"
                "    super_kernel_debug_options: {}\n",
                "must be an explicitly empty mapping",
            ),
        }
        for name, (config_text, error) in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                baseline = root / "baseline"
                candidate = root / "S1"
                _write_performance_runs(baseline, [10.0, 10.1, 10.2, 10.3, 10.4])
                _write_performance_runs(candidate, [9.0, 9.0, 9.0])
                (candidate / "run-1" / "config.yaml").write_text(config_text)

                with self.assertRaisesRegex(ValueError, error):
                    analyze_performance.compare_candidates(
                        baseline,
                        {"S1": candidate},
                        selection_only=True,
                        screening_matrix=_screening_matrix(executed=("S1",)),
                    )

    def test_option_trial_accepts_any_strictly_positive_gain_without_replay_or_profiling(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            incumbent = root / "O0-incumbent"
            trial = root / "O1-auto-op-parallel"
            _write_performance_runs(incumbent, [10.0, 10.1, 10.2, 10.3, 10.4])
            _write_performance_runs(trial, [10.1, 10.1, 10.1])

            report = analyze_performance.compare_candidates(
                incumbent,
                {"O1": trial},
                option_trial=True,
            )

        summary = report["candidates"]["O1"]
        self.assertEqual(report["selection"]["stage"], "winner_option_trial")
        self.assertTrue(summary["option_trial_evaluation"]["accepted"])
        self.assertEqual(summary["option_trial_evaluation"]["min_improvement_pct"], 0.0)
        self.assertEqual(
            summary["option_trial_evaluation"]["mean_improvement_rule"],
            "strictly_positive",
        )
        self.assertGreater(
            summary["option_trial_evaluation"]["mean_improvement_pct"], 0.0
        )
        self.assertLess(summary["option_trial_evaluation"]["mean_improvement_pct"], 2.0)
        self.assertTrue(report["selection"]["option_trial_accepted"])
        self.assertEqual(report["selection"]["retained_incumbent"], "O1")
        self.assertEqual(
            report["selection"]["final_promotion_decision"],
            "not_applicable_option_trial",
        )
        self.assertNotIn("promotion", summary)
        self.assertNotIn("deep_fusion_replay", summary)

    def test_option_trial_rejects_noisy_or_nonbeneficial_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            incumbent = root / "O0-incumbent"
            trial = root / "O1-dcci"
            _write_performance_runs(incumbent, [10.0, 10.1, 10.2, 10.3, 10.4])
            _write_performance_runs(trial, [10.2, 10.2, 10.2])

            report = analyze_performance.compare_candidates(
                incumbent,
                {"O1": trial},
                option_trial=True,
            )

        self.assertFalse(
            report["candidates"]["O1"]["option_trial_evaluation"]["accepted"]
        )
        self.assertFalse(
            report["candidates"]["O1"]["option_trial_evaluation"]["checks"][
                "mean_improvement"
            ]
        )
        self.assertFalse(report["selection"]["option_trial_accepted"])
        self.assertEqual(report["selection"]["retained_incumbent"], "baseline")

    def test_option_trial_requires_one_candidate_and_rejects_other_modes(self):
        with self.assertRaisesRegex(ValueError, "exactly one candidate"):
            analyze_performance.compare_candidates("/tmp/O0", {}, option_trial=True)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            analyze_performance.compare_candidates(
                "/tmp/S0",
                {"O1": "/tmp/O1"},
                selection_only=True,
                option_trial=True,
            )
        with self.assertRaisesRegex(ValueError, "option-trial"):
            analyze_performance.compare_candidates(
                "/tmp/O0",
                {"O1": "/tmp/O1"},
                option_trial=True,
                promotion_modes={"O1": "whole_scope"},
            )

    def test_selection_only_rejects_missing_required_strategy_kind(self):
        matrix = _screening_matrix()
        matrix["candidates"] = matrix["candidates"][:3]

        with self.assertRaisesRegex(ValueError, "semantic_segment"):
            analyze_performance.validate_screening_matrix(matrix, {"S1", "S2", "S3"})

    def test_selection_only_rejects_existing_eligible_as_s4_skip_reason(self):
        matrix = _screening_matrix(
            s4_blocker={
                "code": "deferred_until_no_eligible",
                "reason_zh": "已有 eligible candidate，所以暂不实现 S4。",
                "evidence": "intake/candidate-matrix.json",
            }
        )

        with self.assertRaisesRegex(ValueError, "illegal early-stop blocker"):
            analyze_performance.validate_screening_matrix(matrix, {"S1", "S2", "S3"})

    def test_selection_only_rejects_executed_candidate_set_mismatch(self):
        matrix = _screening_matrix(executed=("S1", "S2", "S3", "S4"))

        with self.assertRaisesRegex(ValueError, "missing executed candidates: S4"):
            analyze_performance.validate_screening_matrix(matrix, {"S1", "S2", "S3"})

    def test_selection_only_cli_requires_screening_matrix(self):
        with self.assertRaises(SystemExit):
            analyze_performance.main(
                [
                    "--baseline",
                    "/tmp/S0",
                    "--candidate",
                    "S1=/tmp/S1",
                    "--selection-only",
                ]
            )

    def test_selection_only_rejects_final_promotion_inputs(self):
        with self.assertRaisesRegex(ValueError, "selection-only"):
            analyze_performance.compare_candidates(
                "/tmp/S0",
                {"S1": "/tmp/S1"},
                selection_only=True,
                promotion_modes={"S1": "whole_scope"},
            )

    def test_whole_scope_mode_rejects_unrelated_replay_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            candidate = root / "candidate"
            evidence = _write_replay_evidence(
                root / "S2-replay.json", candidate_name="S2"
            )
            _write_performance_runs(baseline, [10.0, 10.1, 10.2, 10.3, 10.4])
            _write_performance_runs(candidate, [8.0, 8.0, 8.0])

            with self.assertRaisesRegex(ValueError, "extra: S2"):
                analyze_performance.compare_candidates(
                    baseline,
                    {"S2": candidate},
                    replay_evidence={"S2": evidence},
                    promotion_modes={"S2": "whole_scope"},
                )

    def test_candidate_comparison_rejects_cross_candidate_replay_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            candidate = root / "candidate"
            _write_performance_runs(baseline, [10.0, 10.1, 10.2, 10.3, 10.4])
            _write_performance_runs(candidate, [8.0, 8.0, 8.0])
            evidence = _write_replay_evidence(
                root / "S1-replay.json", candidate_name="S1"
            )

            with self.assertRaisesRegex(ValueError, "candidate_name"):
                analyze_performance.compare_candidates(
                    baseline,
                    {"S2": candidate},
                    replay_evidence={"S2": evidence},
                )

    def test_candidate_comparison_rejects_extra_or_duplicate_replay_mapping(self):
        with self.assertRaises(SystemExit):
            analyze_performance.main(
                [
                    "--baseline",
                    "/tmp/S0",
                    "--candidate",
                    "S1=/tmp/S1",
                    "--replay-evidence",
                    "S1=/tmp/S1-replay.json",
                    "--replay-evidence",
                    "S1=/tmp/S1-replay-duplicate.json",
                ]
            )
        with self.assertRaises(SystemExit):
            analyze_performance.main(
                [
                    "--baseline",
                    "/tmp/S0",
                    "--candidate",
                    "S1=/tmp/S1",
                    "--replay-evidence",
                    "S1=/tmp/S1-replay.json",
                    "--replay-evidence",
                    "S2=/tmp/S2-replay.json",
                ]
            )

    def test_performance_cli_rejects_invalid_replay_without_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            candidate = root / "candidate"
            evidence = root / "invalid-replay.json"
            _write_performance_runs(baseline, [10.0, 10.1, 10.2, 10.3, 10.4])
            _write_performance_runs(candidate, [8.0, 8.0, 8.0])
            evidence.write_text(
                json.dumps(
                    {
                        "candidate_name": "S1",
                        "compat_metadata_root": 1,
                        "verify_metadata_root": "/tmp/verify",
                        "compat_config_manifest_path": "/tmp/compat.json",
                        "verify_config_manifest_path": "/tmp/verify.json",
                    }
                )
            )

            with self.assertRaises(SystemExit):
                analyze_performance.main(
                    [
                        "--baseline",
                        str(baseline),
                        "--candidate",
                        f"S1={candidate}",
                        "--replay-evidence",
                        f"S1={evidence}",
                    ]
                )

    def test_performance_policy_overrides_are_not_public_or_accepted_by_cli(self):
        summary = {
            "run_count": 1,
            "run_worst_rank_means_ms": [10.0],
            "worst_rank_mean_ms": 10.0,
            "median_run_worst_rank_mean_ms": 10.0,
            "decode": {"p90_ms": 10.0, "stddev_ms": 0.0},
        }
        with self.assertRaises(TypeError):
            analyze_performance.evaluate_baseline_stability(
                summary, required_run_count=1
            )
        with self.assertRaises(TypeError):
            analyze_performance.evaluate_promotion(summary, summary, min_runs=1)
        for flag, value in (
            ("--min-runs", "1"),
            ("--baseline-required-runs", "1"),
            ("--baseline-max-spread-ms", "999"),
            ("--baseline-max-spread-pct", "999"),
        ):
            with self.assertRaises(SystemExit):
                analyze_performance.main(["--baseline", "/tmp/S0", flag, value])
        with self.assertRaises(SystemExit):
            analyze_performance.main(
                [
                    "--baseline",
                    "/tmp/S0",
                    "--candidate",
                    "S1=/tmp/S1",
                ]
            )


class MetadataAnalysisTest(unittest.TestCase):
    def test_per_layer_sk_inventory_disconnect_and_control_core(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model_1_1"
            model.mkdir()
            (model / "sk_fused_nodes.log").write_text(
                "SK Function: sk_1_decoder.layer.0.attention_start_static_kernel_A_hash_"
                "end_static_kernel_B_hash, scope id: 1, Node Count: 2\n"
                "[nodeId:10, streamId:1] - KernelInfos{funcName:static_kernel_A_hash, "
                "kernelType:AIC_ONLY, numBlocks:2, cubeNum:2, vecNum:0, isScheModeOn:1}\n"
                "[nodeId:11, streamId:2] - KernelInfos{funcName:static_kernel_B_hash, "
                "kernelType:AIV_ONLY, numBlocks:2, cubeNum:0, vecNum:2, isScheModeOn:1}\n"
                "SK Function: sk_2_decoder.layer.0.mlp_start_static_kernel_C_hash_"
                "end_static_kernel_C_hash, scope id: 2, Node Count: 1\n"
                "[nodeId:12, streamId:1] - KernelInfos{funcName:static_kernel_C_hash, "
                "kernelType:AIV_ONLY, numBlocks:1, cubeNum:0, vecNum:1, isScheModeOn:0}\n"
            )
            (model / "sk_scope_split.log").write_text(
                "Scope 1 (scopeId=1): 3 nodes, 2 streams, scopeNames=[decoder.layer.0.attention]\n"
                "BreakInfo: breakReason=There exists unfusible node in scope, triggerNode=13, "
                'triggerStream=1, fusionFailReason=Insufficient resources, detail="split"\n'
                "Scope 1 (scopeId=1): 3 nodes, 2 streams, scopeNames=[decoder.layer.0.attention]\n"
                "BreakInfo: breakReason=There exists unfusible node in scope, triggerNode=13, "
                'triggerStream=1, fusionFailReason=Insufficient resources, detail="split"\n'
            )
            (model / "sk_fusion_fail_reasons.log").write_text("")
            report = analyze_sk_meta.analyze_metadata(directory)

        layer = report["layer_summary"]["layers"]["0"]
        self.assertEqual(layer["sk_count"], 2)
        self.assertEqual(layer["child_op_count_histogram"], {"1": 1, "2": 1})
        self.assertEqual(layer["scope_fragment_count"], 1)
        self.assertEqual(report["scope_break_summary"]["raw_record_count"], 2)
        self.assertEqual(report["scope_break_summary"]["record_count"], 1)
        self.assertEqual(report["control_core"]["status"], "mixed")
        self.assertEqual(len(report["sk_inventory"]), 2)

    def test_round_report_filters_shallow_fusion_and_lists_unfused_ops(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model_1_1"
            model.mkdir()
            (model / "sk_fused_nodes.log").write_text(
                "SK Function: sk_1_decoder.layer.0.auto_start_static_kernel_A_hash_"
                "end_static_kernel_D_hash, scope id: 1, Node Count: 4\n"
                "[nodeId:10, streamId:1] - KernelInfos{funcName:static_kernel_A_hash, "
                "kernelType:AIV_ONLY, numBlocks:1, cubeNum:0, vecNum:1, isScheModeOn:0}\n"
                "[nodeId:11, streamId:1] - KernelInfos{funcName:static_kernel_B_hash, "
                "kernelType:AIV_ONLY, numBlocks:1, cubeNum:0, vecNum:1, isScheModeOn:0}\n"
                "[nodeId:12, streamId:1] - KernelInfos{funcName:static_kernel_C_hash, "
                "kernelType:AIV_ONLY, numBlocks:1, cubeNum:0, vecNum:1, isScheModeOn:0}\n"
                "[nodeId:13, streamId:1] - KernelInfos{funcName:static_kernel_D_hash, "
                "kernelType:AIV_ONLY, numBlocks:1, cubeNum:0, vecNum:1, isScheModeOn:0}\n"
                "SK Function: sk_2_decoder.layer.0.auto_start_static_kernel_E_hash_"
                "end_static_kernel_I_hash, scope id: 2, Node Count: 5\n"
                "[nodeId:20, streamId:1] - KernelInfos{funcName:static_kernel_E_hash, "
                "kernelType:AIV_ONLY, numBlocks:1, cubeNum:0, vecNum:1, isScheModeOn:0}\n"
                "[nodeId:21, streamId:1] - KernelInfos{funcName:static_kernel_F_hash, "
                "kernelType:AIV_ONLY, numBlocks:1, cubeNum:0, vecNum:1, isScheModeOn:0}\n"
                "[nodeId:22, streamId:1] - KernelInfos{funcName:static_kernel_G_hash, "
                "kernelType:AIV_ONLY, numBlocks:1, cubeNum:0, vecNum:1, isScheModeOn:0}\n"
                "[nodeId:23, streamId:1] - KernelInfos{funcName:static_kernel_H_hash, "
                "kernelType:AIV_ONLY, numBlocks:1, cubeNum:0, vecNum:1, isScheModeOn:0}\n"
                "[nodeId:24, streamId:1] - KernelInfos{funcName:static_kernel_I_hash, "
                "kernelType:AIV_ONLY, numBlocks:1, cubeNum:0, vecNum:1, isScheModeOn:0}\n"
            )
            (model / "sk_fusion_fail_reasons.log").write_text(
                "Node [nodeId:30, streamId:2] - KernelInfos{funcName:static_kernel_Custom_hash, "
                "kernelType:AIV_ONLY, numBlocks:1, cubeNum:0, vecNum:1}: reason: OP_UNSUPPORT\n"
                "Node [nodeId:31, streamId:2] - KernelInfos{funcName:static_kernel_Runtime_hash, "
                "kernelType:AIV_ONLY, numBlocks:1, cubeNum:0, vecNum:1}: reason: "
                "refresh task information at runtime\n"
            )
            report = analyze_sk_meta.analyze_metadata(directory)

        round_report = report["round_report"]
        self.assertEqual(round_report["min_child_nodes"], 5)
        self.assertEqual(round_report["effective_fusion"]["effective_group_count"], 1)
        self.assertEqual(round_report["effective_fusion"]["shallow_group_count"], 1)
        self.assertEqual(
            round_report["effective_fusion"]["effective_child_node_total"], 5
        )
        self.assertEqual(
            round_report["effective_fusion"]["effective_launch_reduction"], 4
        )
        self.assertEqual(
            [item["child_count"] for item in round_report["per_sk_fusion_table"]],
            [5, 4],
        )
        self.assertFalse(
            round_report["per_sk_fusion_table"][1]["counts_as_effective_fusion"]
        )
        op_rows = round_report["non_fusion_operator_table"]
        self.assertIn(
            ("OP_UNSUPPORT", "Custom"),
            {(row["reason"], row["op_type"]) for row in op_rows},
        )
        self.assertIn(
            ("DYNAMIC_TASK_UNSUPPORT", "Runtime"),
            {(row["reason"], row["op_type"]) for row in op_rows},
        )
        evidence_fields = round_report["required_per_round_report_fields"]
        self.assertTrue(
            any("round-evidence.json" in field for field in evidence_fields)
        )
        self.assertFalse(
            any(
                "clean performance versus S0 baseline" == field
                for field in evidence_fields
            )
        )

    def test_round_report_keeps_child_count_descriptive_without_scope_action(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model_1_1"
            model.mkdir()
            lines = []
            for group_index, child_count in enumerate([1, 2, 4, 5], start=1):
                ops = [f"Op{group_index}_{offset}" for offset in range(child_count)]
                lines.append(
                    f"SK Function: sk_{group_index}_decoder.layer.0.segment_"
                    f"start_static_kernel_{ops[0]}_hash_end_static_kernel_"
                    f"{ops[-1]}_hash, scope id: {group_index}, Node Count: {child_count}"
                )
                for offset, op_type in enumerate(ops):
                    lines.append(
                        f"[nodeId:{group_index * 100 + offset}, streamId:1] - "
                        f"KernelInfos{{funcName:static_kernel_{op_type}_hash, "
                        "kernelType:AIV_ONLY, numBlocks:1, cubeNum:0, vecNum:1, "
                        "isScheModeOn:0}}"
                    )
            (model / "sk_fused_nodes.log").write_text("\n".join(lines) + "\n")

            report = analyze_sk_meta.analyze_metadata(
                directory, min_effective_child_nodes=5, round_name="S3-BASE"
            )

        round_report = report["round_report"]
        adjustment = round_report["single_child_exclusion"]
        self.assertFalse(adjustment["requires_scope_adjustment"])
        self.assertEqual(adjustment["next_round_policy"], "profiling_decides")
        self.assertEqual(adjustment["performance_action"], "none")
        self.assertEqual(adjustment["single_child_group_count"], 1)
        self.assertEqual(adjustment["shallow_non_single_child_group_count"], 2)
        self.assertEqual(adjustment["candidates"][0]["child_count"], 1)
        self.assertEqual(
            adjustment["candidates"][0]["performance_action"],
            "pending_profiling_analysis",
        )
        self.assertEqual(round_report["effective_fusion"]["effective_group_count"], 1)
        self.assertIn("descriptive", round_report["filter_policy"])
        self.assertIn("profiling", round_report["filter_policy"])

    def test_round_report_does_not_force_adjustment_for_two_to_four_child_shallow_groups(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model_1_1"
            model.mkdir()
            lines = []
            for group_index, child_count in enumerate([2, 4, 5], start=1):
                ops = [f"Op{group_index}_{offset}" for offset in range(child_count)]
                lines.append(
                    f"SK Function: sk_{group_index}_decoder.layer.0.segment_"
                    f"start_static_kernel_{ops[0]}_hash_end_static_kernel_"
                    f"{ops[-1]}_hash, scope id: {group_index}, Node Count: {child_count}"
                )
                for offset, op_type in enumerate(ops):
                    lines.append(
                        f"[nodeId:{group_index * 100 + offset}, streamId:1] - "
                        f"KernelInfos{{funcName:static_kernel_{op_type}_hash, "
                        "kernelType:AIV_ONLY, numBlocks:1, cubeNum:0, vecNum:1, "
                        "isScheModeOn:0}}"
                    )
            (model / "sk_fused_nodes.log").write_text("\n".join(lines) + "\n")

            report = analyze_sk_meta.analyze_metadata(
                directory, min_effective_child_nodes=5, round_name="S3-A1"
            )

        adjustment = report["round_report"]["single_child_exclusion"]
        self.assertFalse(adjustment["requires_scope_adjustment"])
        self.assertEqual(adjustment["single_child_group_count"], 0)
        self.assertEqual(adjustment["shallow_non_single_child_group_count"], 2)
        self.assertEqual(
            report["round_report"]["effective_fusion"]["effective_group_count"], 1
        )

    def test_automatic_aot_single_child_defers_action_to_profiling(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model_1_1"
            model.mkdir()
            (model / "sk_fused_nodes.log").write_text(
                "SK Function: sk_1_decoder.layer.0.auto_start_static_kernel_A_hash_"
                "end_static_kernel_A_hash, scope id: 1, Node Count: 1\n"
                "[nodeId:10, streamId:1] - KernelInfos{funcName:static_kernel_A_hash, "
                "kernelType:AIV_ONLY, numBlocks:1, cubeNum:0, vecNum:1, isScheModeOn:0}\n"
            )
            report = analyze_sk_meta.analyze_metadata(
                directory,
                min_effective_child_nodes=5,
                round_name="S1",
                scope_kind="automatic_aot",
            )

        adjustment = report["round_report"]["single_child_exclusion"]
        self.assertEqual(adjustment["scope_kind"], "automatic_aot")
        self.assertFalse(adjustment["requires_scope_adjustment"])
        self.assertEqual(adjustment["next_round_policy"], "profiling_decides")
        self.assertEqual(adjustment["s1_auto_action"], "none")

    def test_node_count_mismatch_is_untrusted_and_cannot_be_effective_or_adjusted(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model_1_1"
            model.mkdir()
            (model / "sk_fused_nodes.log").write_text(
                "SK Function: sk_1_decoder.layer.0.auto_start_static_kernel_A_hash_"
                "end_static_kernel_E_hash, scope id: 1, Node Count: 5\n"
                "[nodeId:10, streamId:1] - KernelInfos{funcName:static_kernel_A_hash, "
                "kernelType:AIV_ONLY, numBlocks:1, cubeNum:0, vecNum:1, isScheModeOn:0}\n"
                "[nodeId:11, streamId:1] - KernelInfos{funcName:static_kernel_B_hash, "
                "kernelType:AIV_ONLY, numBlocks:1, cubeNum:0, vecNum:1, isScheModeOn:0}\n"
                "[nodeId:12, streamId:1] - KernelInfos{funcName:static_kernel_C_hash, "
                "kernelType:AIV_ONLY, numBlocks:1, cubeNum:0, vecNum:1, isScheModeOn:0}\n"
                "[nodeId:13, streamId:1] - KernelInfos{funcName:static_kernel_D_hash, "
                "kernelType:AIV_ONLY, numBlocks:1, cubeNum:0, vecNum:1, isScheModeOn:0}\n"
            )
            report = analyze_sk_meta.analyze_metadata(
                directory, min_effective_child_nodes=5, round_name="S3-BASE"
            )

        row = report["round_report"]["per_sk_fusion_table"][0]
        self.assertTrue(row["node_count_mismatch"])
        self.assertEqual(row["declared_child_count"], 5)
        self.assertEqual(row["child_count"], 4)
        self.assertFalse(row["counts_as_effective_fusion"])
        self.assertFalse(
            report["round_report"]["single_child_exclusion"][
                "requires_scope_adjustment"
            ]
        )

    def test_metadata_artifact_paths_are_relative_to_the_report_root(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model_1_1"
            model.mkdir()
            (model / "sk_fused_nodes.log").write_text(
                "SK Function: sk_1_decoder.layer.0.auto_start_static_kernel_A_hash_"
                "end_static_kernel_A_hash, scope id: 1, Node Count: 1\n"
                "[nodeId:10, streamId:1] - KernelInfos{funcName:static_kernel_A_hash, "
                "kernelType:AIV_ONLY, numBlocks:1, cubeNum:0, vecNum:1, isScheModeOn:0}\n"
            )
            report = analyze_sk_meta.analyze_metadata(directory)

        self.assertEqual(report["root"], ".")
        paths = [item["path"] for item in report["round_report"]["per_sk_fusion_table"]]
        self.assertTrue(paths)
        self.assertTrue(all(not Path(path).is_absolute() for path in paths))

    def test_deep_auto_scope_plan_does_not_filter_reliable_groups_by_child_count(self):
        summary = {
            "sk_inventory": [
                {
                    "scope_id": 1,
                    "function": "sk_1_none_start_A_end_B",
                    "node_count": 2,
                    "boundary": {"scope": "none", "start_op": "A", "end_op": "B"},
                    "nodes": [{"op_type": "A"}, {"op_type": "B"}],
                },
                {
                    "scope_id": 2,
                    "function": "sk_2_none_start_MoeGatingTopK_end_MlaPrologV3",
                    "node_count": 13,
                    "boundary": {
                        "scope": "none",
                        "start_op": "MoeGatingTopK",
                        "end_op": "MlaPrologV3",
                    },
                    "nodes": [
                        {"op_type": "MoeGatingTopK"},
                        {"op_type": "MoeDistributeDispatchV2"},
                        {"op_type": "DynamicQuant"},
                        {"op_type": "GroupedMatmul"},
                        {"op_type": "DequantSwigluQuant"},
                        {"op_type": "GroupedMatmul"},
                        {"op_type": "MoeDistributeCombineV2"},
                        {"op_type": "DynamicQuant"},
                        {"op_type": "QuantBatchMatmulV3"},
                        {"op_type": "DequantSwigluQuant"},
                        {"op_type": "QuantBatchMatmulV3"},
                        {"op_type": "AddRmsNorm"},
                        {"op_type": "MlaPrologV3"},
                    ],
                },
                {
                    "scope_id": 3,
                    "function": "sk_3_none_start_aiv_end_QuantBatchMatmulV3",
                    "node_count": 5,
                    "boundary": {
                        "scope": "none",
                        "start_op": "aiv",
                        "end_op": "QuantBatchMatmulV3",
                    },
                    "nodes": [
                        {"op_type": "aiv"},
                        {"op_type": "AddRmsNormDynamicQuant"},
                        {"op_type": "QuantBatchMatmulV3"},
                        {"op_type": "DequantSwigluQuant"},
                        {"op_type": "QuantBatchMatmulV3"},
                    ],
                },
            ]
        }

        plan = analyze_sk_meta.build_deep_auto_scope_plan(summary, min_child_nodes=5)

        self.assertEqual(plan["min_child_nodes"], 5)
        self.assertEqual(plan["selected_group_count"], 3)
        self.assertEqual(plan["skipped_group_count"], 0)
        self.assertEqual(plan["selected_child_node_total"], 20)
        self.assertEqual(
            [item["child_count"] for item in plan["manual_scope_candidates"]],
            [13, 5, 2],
        )
        self.assertTrue(
            all(
                item["performance_action"] == "pending_profiling_analysis"
                for item in plan["manual_scope_candidates"]
            )
        )
        self.assertEqual(
            plan["manual_scope_candidates"][0]["op_sequence"][0],
            "MoeGatingTopK",
        )
        self.assertEqual(
            plan["manual_scope_candidates"][1]["op_sequence"],
            [
                "aiv",
                "AddRmsNormDynamicQuant",
                "QuantBatchMatmulV3",
                "DequantSwigluQuant",
                "QuantBatchMatmulV3",
            ],
        )

    def test_cli_help_describes_child_thresholds_as_non_filtering(self):
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as raised:
            analyze_sk_meta.main(["--help"])

        help_text = output.getvalue()
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("descriptive", help_text)
        self.assertIn("never filters profiling", help_text)
        self.assertNotIn("filtered from scope decisions", help_text)
        self.assertNotIn("only groups with child count", help_text)

    def test_deep_fusion_replay_matches_fresh_roots_with_different_sk_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child_ops = ["A", "B", "C", "D", "E"]
            compat_root = root / "compat"
            verify_root = root / "verify"
            _write_fused_metadata(
                compat_root,
                "decoder.layer.0.attention",
                child_ops,
                function_token="compat_hash",
                node_start=100,
            )
            _write_fused_metadata(
                verify_root,
                "decoder.layer.0.attention",
                child_ops,
                function_token="verify_hash",
                node_start=900,
            )
            compat_config = _write_config_manifest(
                root / "compat-config.json", {"scope": "attention", "sk": True}
            )
            verify_config = _write_config_manifest(
                root / "verify-config.json", {"sk": True, "scope": "attention"}
            )

            replay = analyze_sk_meta.compare_fusion_replay(
                compat_root,
                verify_root,
                compat_config_manifest=compat_config,
                verify_config_manifest=verify_config,
            )

        self.assertTrue(replay["fusion_reproducible"])
        self.assertTrue(replay["deep_fusion_reproducible"])
        self.assertEqual(replay["replay_min_child_nodes"], 1)
        self.assertEqual(replay["matched_fusion_group_count"], 1)
        self.assertIsNone(replay["failure_reason"])
        self.assertEqual(replay["compat_effective_group_count"], 1)
        self.assertEqual(replay["verify_effective_group_count"], 1)
        self.assertEqual(replay["matched_effective_group_count"], 1)
        self.assertEqual(replay["compat_only_effective_groups"], [])
        self.assertEqual(replay["verify_only_effective_groups"], [])
        self.assertEqual(replay["matched_effective_groups"][0]["child_count"], 5)
        self.assertTrue(replay["config_fingerprint_match"])
        self.assertEqual(
            replay["compat_config_fingerprint"], replay["verify_config_fingerprint"]
        )
        self.assertEqual(replay["artifact_base"], ".")
        self.assertEqual(replay["compat_metadata_root"], "compat")
        self.assertEqual(replay["verify_metadata_root"], "verify")
        self.assertEqual(replay["compat_config_manifest_path"], "compat-config.json")
        self.assertEqual(replay["verify_config_manifest_path"], "verify-config.json")
        self.assertFalse(Path(replay["compat_config_manifest_path"]).is_absolute())
        self.assertFalse(Path(replay["verify_config_manifest_path"]).is_absolute())
        with self.assertRaises(TypeError):
            analyze_sk_meta.compare_deep_fusion_replay(
                compat_root,
                verify_root,
                compat_config_manifest=compat_config,
                verify_config_manifest=verify_config,
                min_effective_child_nodes=1,
            )

    def test_fusion_replay_accepts_reproducible_single_child_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compat_root = root / "compat"
            verify_root = root / "verify"
            _write_fused_metadata(
                compat_root,
                "decoder.layer.0.attention",
                ["A"],
                function_token="compat_hash",
                node_start=100,
                model_directory="model_48_1",
            )
            _write_fused_metadata(
                verify_root,
                "decoder.layer.0.attention",
                ["A"],
                function_token="verify_hash",
                node_start=900,
                model_directory="model_48_2",
            )
            compat_config = _write_config_manifest(
                root / "compat-config.json", {"scope": "attention", "sk": True}
            )
            verify_config = _write_config_manifest(
                root / "verify-config.json", {"sk": True, "scope": "attention"}
            )

            replay = analyze_sk_meta.compare_fusion_replay(
                compat_root,
                verify_root,
                compat_config_manifest=compat_config,
                verify_config_manifest=verify_config,
            )

        self.assertTrue(replay["fusion_reproducible"])
        self.assertEqual(replay["replay_min_child_nodes"], 1)
        self.assertEqual(replay["matched_fusion_group_count"], 1)
        self.assertFalse(replay["deep_fusion_reproducible"])
        matched = replay["matched_fusion_groups"][0]
        self.assertEqual(matched["model_id"], "48")
        self.assertEqual(matched["signature"]["model_id"], "48")
        self.assertEqual(matched["source_scope"], "decoder.layer.0.attention")
        self.assertEqual(matched["segments"], ["attention"])
        self.assertEqual(matched["ordered_child_op_sequence"], ["A"])
        self.assertEqual(matched["child_count"], 1)
        self.assertTrue(matched["count_reliable"])

    def test_fusion_replay_excludes_duplicate_identity_from_actionable_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compat_root = root / "compat"
            verify_root = root / "verify"
            for metadata_root, token, node_start in (
                (compat_root, "compat", 100),
                (verify_root, "verify", 900),
            ):
                _write_fused_metadata(
                    metadata_root,
                    "decoder.layer.0.attention",
                    ["A"],
                    function_token=token,
                    node_start=node_start,
                    model_directory="model_48",
                )
                metadata_log = metadata_root / "model_48" / "sk_fused_nodes.log"
                record = metadata_log.read_text()
                metadata_log.write_text(record + record)
            compat_config = _write_config_manifest(root / "compat.json", {"x": 1})
            verify_config = _write_config_manifest(root / "verify.json", {"x": 1})

            replay = analyze_sk_meta.compare_fusion_replay(
                compat_root,
                verify_root,
                compat_config_manifest=compat_config,
                verify_config_manifest=verify_config,
                candidate_name="S1",
                artifact_base=root,
            )
            replay_path = root / "replay.json"
            replay_path.write_text(json.dumps(replay) + "\n")
            validation = analyze_sk_meta.validate_replay_report(
                replay_path, expected_candidate="S1"
            )

        self.assertFalse(replay["fusion_reproducible"])
        self.assertEqual(replay["matched_fusion_group_count"], 0)
        self.assertEqual(len(replay["compat_only_fusion_groups"]), 2)
        self.assertEqual(len(replay["verify_only_fusion_groups"]), 2)
        self.assertEqual(
            replay["failure_reason"], "no_matching_effective_fusion_signature"
        )
        self.assertFalse(validation["valid"])

    def test_fusion_replay_preserves_graph_occurrence_fingerprint_in_signature(self):
        base = {
            "model_id": "1",
            "source_scope": "decoder.layer.0",
            "segments": ["decoder", "layer.0"],
            "boundary": {"start_op": "Add", "end_op": "MatMul"},
            "op_sequence": ["Add", "MatMul"],
            "child_count": 2,
            "count_reliable": True,
        }
        compat = analyze_sk_meta._replay_group(
            {**base, "graph_occurrence_fingerprint": "a" * 64}
        )
        verify = analyze_sk_meta._replay_group(
            {**base, "graph_occurrence_fingerprint": "b" * 64}
        )

        matched, compat_only, verify_only = analyze_sk_meta._match_replay_groups(
            [compat], [verify]
        )

        self.assertEqual(matched, [])
        self.assertEqual(compat_only[0]["graph_occurrence_fingerprint"], "a" * 64)
        self.assertEqual(verify_only[0]["graph_occurrence_fingerprint"], "b" * 64)

    def test_fusion_replay_accepts_auto_round_for_candidate_family(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compat = root / "compat"
            verify = root / "verify"
            _write_fused_metadata(
                compat, "scope", ["A"], function_token="compat", node_start=1
            )
            _write_fused_metadata(
                verify, "scope", ["A"], function_token="verify", node_start=2
            )
            compat_config = _write_config_manifest(
                root / "compat.json", {"scope": "S1"}
            )
            verify_config = _write_config_manifest(
                root / "verify.json", {"scope": "S1"}
            )

            report = analyze_sk_meta.compare_fusion_replay(
                compat,
                verify,
                compat_config_manifest=compat_config,
                verify_config_manifest=verify_config,
                candidate_name="S1",
                round_id="S1-AUTO",
                source_revision="source-revision",
            )

        self.assertEqual(report["candidate_name"], "S1")
        self.assertEqual(report["round_id"], "S1-AUTO")

    def test_fusion_replay_normalizes_scope_and_filters_blank_segments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compat_root = root / "compat"
            verify_root = root / "verify"
            for metadata_root, token, node_start in (
                (compat_root, "compat", 100),
                (verify_root, "verify", 900),
            ):
                _write_fused_metadata(
                    metadata_root,
                    "decoder.layer.0.   ",
                    ["A"],
                    function_token=token,
                    node_start=node_start,
                    model_directory="model_48",
                )
            compat_config = _write_config_manifest(root / "compat.json", {"x": 1})
            verify_config = _write_config_manifest(root / "verify.json", {"x": 1})

            replay = analyze_sk_meta.compare_fusion_replay(
                compat_root,
                verify_root,
                compat_config_manifest=compat_config,
                verify_config_manifest=verify_config,
            )

        self.assertTrue(replay["fusion_reproducible"])
        matched = replay["matched_fusion_groups"][0]
        self.assertEqual(matched["source_scope"], "decoder.layer.0.")
        self.assertEqual(matched["segments"], [])

    def test_fusion_replay_does_not_match_signatures_swapped_between_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compat_root = root / "compat"
            verify_root = root / "verify"
            for metadata_root, model_48_ops, model_49_ops, token in (
                (compat_root, ["A"], ["B"], "compat"),
                (verify_root, ["B"], ["A"], "verify"),
            ):
                _write_fused_metadata(
                    metadata_root,
                    "scope",
                    model_48_ops,
                    function_token=f"{token}_48",
                    node_start=100,
                    model_directory="model_48",
                )
                _write_fused_metadata(
                    metadata_root,
                    "scope",
                    model_49_ops,
                    function_token=f"{token}_49",
                    node_start=200,
                    model_directory="model_49_1",
                )
            compat_config = _write_config_manifest(root / "compat.json", {"x": 1})
            verify_config = _write_config_manifest(root / "verify.json", {"x": 1})

            replay = analyze_sk_meta.compare_fusion_replay(
                compat_root,
                verify_root,
                compat_config_manifest=compat_config,
                verify_config_manifest=verify_config,
            )

        self.assertFalse(replay["fusion_reproducible"])
        self.assertEqual(replay["matched_fusion_group_count"], 0)
        self.assertEqual(
            replay["failure_reason"], "no_matching_effective_fusion_signature"
        )

    def test_fusion_replay_and_validator_reject_incomplete_generated_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compat_root = root / "compat"
            verify_root = root / "verify"
            for metadata_root, token in (
                (compat_root, "compat"),
                (verify_root, "verify"),
            ):
                model = metadata_root / "model_48"
                model.mkdir(parents=True)
                (model / "sk_fused_nodes.log").write_text(
                    f"SK Function: generated_{token}, scope id: 1, Node Count: 1\n"
                    "[nodeId:10, streamId:1] - "
                    "KernelInfos{funcName:static_kernel_A_hash, "
                    "kernelType:AIV_ONLY, numBlocks:1, cubeNum:0, vecNum:1, "
                    "isScheModeOn:0}\n"
                )
            compat_config = _write_config_manifest(root / "compat.json", {"x": 1})
            verify_config = _write_config_manifest(root / "verify.json", {"x": 1})
            replay = analyze_sk_meta.compare_fusion_replay(
                compat_root,
                verify_root,
                compat_config_manifest=compat_config,
                verify_config_manifest=verify_config,
                candidate_name="S1",
                artifact_base=root,
            )
            replay_path = root / "replay.json"
            replay_path.write_text(json.dumps(replay) + "\n")

            validation = analyze_sk_meta.validate_replay_report(
                replay_path, expected_candidate="S1"
            )

        self.assertFalse(replay["fusion_reproducible"])
        self.assertEqual(replay["matched_fusion_group_count"], 0)
        self.assertFalse(validation["valid"])
        self.assertFalse(validation["checks"]["matched_reliable_signature"])

    def test_fusion_replay_excludes_unreliable_child_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compat_root = root / "compat"
            verify_root = root / "verify"
            _write_fused_metadata(
                compat_root, "scope", ["A"], function_token="compat", node_start=1
            )
            _write_fused_metadata(
                verify_root, "scope", ["A"], function_token="verify", node_start=2
            )
            for metadata_root in (compat_root, verify_root):
                metadata_log = metadata_root / "model_1_1" / "sk_fused_nodes.log"
                metadata_log.write_text(
                    metadata_log.read_text().replace("Node Count: 1", "Node Count: 2")
                )
            compat_config = _write_config_manifest(root / "compat.json", {"x": 1})
            verify_config = _write_config_manifest(root / "verify.json", {"x": 1})

            replay = analyze_sk_meta.compare_fusion_replay(
                compat_root,
                verify_root,
                compat_config_manifest=compat_config,
                verify_config_manifest=verify_config,
            )

        self.assertFalse(replay["fusion_reproducible"])
        self.assertEqual(replay["matched_fusion_group_count"], 0)
        self.assertEqual(replay["failure_reason"], "compat_has_no_effective_fusion")

    def test_deep_fusion_replay_rejects_shallow_verify_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compat_root = root / "compat"
            verify_root = root / "verify"
            _write_fused_metadata(
                compat_root,
                "decoder.layer.0.attention",
                ["A", "B", "C", "D", "E"],
                function_token="compat_hash",
                node_start=100,
            )
            _write_fused_metadata(
                verify_root,
                "decoder.layer.0.attention",
                ["A", "B", "C", "D"],
                function_token="verify_hash",
                node_start=900,
            )
            compat_config = _write_config_manifest(
                root / "compat-config.json", {"x": 1}
            )
            verify_config = _write_config_manifest(
                root / "verify-config.json", {"x": 1}
            )

            replay = analyze_sk_meta.compare_deep_fusion_replay(
                compat_root,
                verify_root,
                compat_config_manifest=compat_config,
                verify_config_manifest=verify_config,
            )

        self.assertFalse(replay["deep_fusion_reproducible"])
        self.assertEqual(replay["failure_reason"], "verify_has_no_effective_fusion")

    def test_deep_fusion_replay_requires_ordered_child_sequence_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compat_root = root / "compat"
            verify_root = root / "verify"
            _write_fused_metadata(
                compat_root,
                "decoder.layer.0.attention",
                ["A", "B", "C", "D", "E"],
                function_token="compat_hash",
                node_start=100,
            )
            _write_fused_metadata(
                verify_root,
                "decoder.layer.0.attention",
                ["A", "C", "B", "D", "E"],
                function_token="verify_hash",
                node_start=900,
            )
            compat_config = _write_config_manifest(
                root / "compat-config.json", {"x": 1}
            )
            verify_config = _write_config_manifest(
                root / "verify-config.json", {"x": 1}
            )

            replay = analyze_sk_meta.compare_deep_fusion_replay(
                compat_root,
                verify_root,
                compat_config_manifest=compat_config,
                verify_config_manifest=verify_config,
            )

        self.assertFalse(replay["deep_fusion_reproducible"])
        self.assertEqual(
            replay["failure_reason"], "no_matching_effective_fusion_signature"
        )

    def test_deep_fusion_replay_rejects_same_metadata_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "metadata"
            _write_fused_metadata(
                root,
                "decoder.layer.0.attention",
                ["A", "B", "C", "D", "E"],
                function_token="same_hash",
                node_start=100,
            )
            config = _write_config_manifest(root.parent / "config.json", {"x": 1})

            replay = analyze_sk_meta.compare_deep_fusion_replay(
                root,
                root,
                compat_config_manifest=config,
                verify_config_manifest=config,
            )

        self.assertFalse(replay["deep_fusion_reproducible"])
        self.assertEqual(replay["failure_reason"], "verify_metadata_not_fresh")

    def test_cli_replay_writes_passing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compat_root = root / "compat"
            verify_root = root / "verify"
            replay_out = root / "replay.json"
            child_ops = ["A", "B", "C", "D", "E"]
            _write_fused_metadata(
                compat_root,
                "decoder.layer.0.attention",
                child_ops,
                function_token="compat_hash",
                node_start=100,
            )
            _write_fused_metadata(
                verify_root,
                "decoder.layer.0.attention",
                child_ops,
                function_token="verify_hash",
                node_start=900,
            )
            compat_config = _write_config_manifest(
                root / "compat-config.json", {"x": 1}
            )
            verify_config = _write_config_manifest(
                root / "verify-config.json", {"x": 1}
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = analyze_sk_meta.main(
                    [
                        str(compat_root),
                        "--verify-root",
                        str(verify_root),
                        "--compat-config-manifest",
                        str(compat_config),
                        "--verify-config-manifest",
                        str(verify_config),
                        "--candidate-name",
                        "S2",
                        "--round-id",
                        "S2-BASE",
                        "--source-revision",
                        "source-revision-1",
                        "--replay-report-out",
                        str(replay_out),
                    ]
                )

            report = json.loads(replay_out.read_text())
            validation = analyze_sk_meta.validate_replay_report(
                replay_out, expected_candidate="S2"
            )

        self.assertEqual(exit_code, 0)
        self.assertTrue(report["deep_fusion_reproducible"])
        self.assertEqual(report["candidate_name"], "S2")
        self.assertEqual(report["round_id"], "S2-BASE")
        self.assertEqual(report["source_revision"], "source-revision-1")
        self.assertTrue(validation["valid"])
        self.assertEqual(json.loads(stdout.getvalue()), report)

    def test_cli_and_validator_accept_reproducible_single_child_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compat_root = root / "compat"
            verify_root = root / "verify"
            replay_out = root / "replay.json"
            _write_fused_metadata(
                compat_root,
                "decoder.layer.0.attention",
                ["A"],
                function_token="compat_hash",
                node_start=100,
            )
            _write_fused_metadata(
                verify_root,
                "decoder.layer.0.attention",
                ["A"],
                function_token="verify_hash",
                node_start=900,
            )
            compat_config = _write_config_manifest(
                root / "compat-config.json", {"x": 1}
            )
            verify_config = _write_config_manifest(
                root / "verify-config.json", {"x": 1}
            )

            with redirect_stdout(io.StringIO()):
                exit_code = analyze_sk_meta.main(
                    [
                        str(compat_root),
                        "--verify-root",
                        str(verify_root),
                        "--compat-config-manifest",
                        str(compat_config),
                        "--verify-config-manifest",
                        str(verify_config),
                        "--candidate-name",
                        "S1",
                        "--round-id",
                        "S1-BASE",
                        "--source-revision",
                        "source-revision-1",
                        "--replay-report-out",
                        str(replay_out),
                    ]
                )
            report = json.loads(replay_out.read_text())
            validation = analyze_sk_meta.validate_replay_report(
                replay_out, expected_candidate="S1"
            )

        self.assertEqual(exit_code, 0)
        self.assertTrue(report["fusion_reproducible"])
        self.assertFalse(report["deep_fusion_reproducible"])
        self.assertEqual(report["replay_min_child_nodes"], 1)
        self.assertTrue(validation["valid"])

    def test_cli_replay_min_child_nodes_can_require_deep_fusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compat_root = root / "compat"
            verify_root = root / "verify"
            _write_fused_metadata(
                compat_root, "scope", ["A"], function_token="compat", node_start=1
            )
            _write_fused_metadata(
                verify_root, "scope", ["A"], function_token="verify", node_start=2
            )
            compat_config = _write_config_manifest(root / "compat.json", {"x": 1})
            verify_config = _write_config_manifest(root / "verify.json", {"x": 1})

            with redirect_stdout(io.StringIO()):
                exit_code = analyze_sk_meta.main(
                    [
                        str(compat_root),
                        "--verify-root",
                        str(verify_root),
                        "--compat-config-manifest",
                        str(compat_config),
                        "--verify-config-manifest",
                        str(verify_config),
                        "--candidate-name",
                        "S1",
                        "--round-id",
                        "S1-BASE",
                        "--source-revision",
                        "source-revision-1",
                        "--replay-min-child-nodes",
                        "5",
                    ]
                )

        self.assertEqual(exit_code, 1)

    def test_replay_validator_recomputes_artifacts_and_rejects_forged_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "forged.json"
            replay_path.write_text(
                json.dumps(
                    {
                        "candidate_name": "S1",
                        "round_id": "S1-BASE",
                        "source_revision": "source-revision-1",
                        "deep_fusion_reproducible": True,
                        "config_fingerprint_match": True,
                        "compat_config_fingerprint": "a" * 64,
                        "verify_config_fingerprint": "a" * 64,
                        "compat_metadata_root": "/nonexistent/compat",
                        "verify_metadata_root": "/nonexistent/verify",
                        "compat_config_manifest_path": "/nonexistent/compat.json",
                        "verify_config_manifest_path": "/nonexistent/verify.json",
                        "matched_effective_group_count": 1,
                    }
                )
            )

            validation = analyze_sk_meta.validate_replay_report(
                replay_path, expected_candidate="S1"
            )

        self.assertFalse(validation["valid"])
        self.assertEqual(validation["failure_reason"], "replay_artifacts_invalid")

    def test_replay_validator_rejects_tampered_threshold_and_matched_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compat_root = root / "compat"
            verify_root = root / "verify"
            _write_fused_metadata(
                compat_root, "scope", ["A"], function_token="compat", node_start=1
            )
            _write_fused_metadata(
                verify_root, "scope", ["A"], function_token="verify", node_start=2
            )
            compat_config = _write_config_manifest(root / "compat.json", {"x": 1})
            verify_config = _write_config_manifest(root / "verify.json", {"x": 1})
            report = analyze_sk_meta.compare_fusion_replay(
                compat_root,
                verify_root,
                compat_config_manifest=compat_config,
                verify_config_manifest=verify_config,
                candidate_name="S1",
                round_id="S1-BASE",
                source_revision="source-revision-1",
                artifact_base=root,
            )
            replay_path = root / "replay.json"

            mutations = {
                "threshold": lambda value: value.__setitem__(
                    "replay_min_child_nodes", 5
                ),
                "matched_groups": lambda value: value["matched_fusion_groups"][
                    0
                ].__setitem__("source_scope", "other_scope"),
                "lifecycle_claim": lambda value: value.__setitem__(
                    "fusion_reproducible", False
                ),
                "compat_config_manifest": lambda value: value[
                    "compat_config_manifest"
                ].__setitem__("fingerprint", "f" * 64),
                "verify_config_manifest": lambda value: value[
                    "verify_config_manifest"
                ].__setitem__("fingerprint", "f" * 64),
                "compat_only_effective_groups": lambda value: value[
                    "compat_only_effective_groups"
                ].append({"forged": True}),
                "verify_only_effective_groups": lambda value: value[
                    "verify_only_effective_groups"
                ].append({"forged": True}),
                "missing_nullable_producer_field": lambda value: value.pop(
                    "failure_reason"
                ),
            }
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    forged = json.loads(json.dumps(report))
                    mutate(forged)
                    replay_path.write_text(json.dumps(forged) + "\n")
                    validation = analyze_sk_meta.validate_replay_report(replay_path)
                    self.assertFalse(validation["valid"])
                    self.assertEqual(
                        validation["failure_reason"],
                        "replay_report_stale_or_tampered",
                    )

    def test_replay_validator_rejects_malformed_artifact_paths_without_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "malformed.json"
            replay_path.write_text(
                json.dumps(
                    {
                        "candidate_name": "S1",
                        "round_id": "S1-BASE",
                        "source_revision": "source-revision-1",
                        "compat_metadata_root": 1,
                        "verify_metadata_root": "/tmp/verify",
                        "compat_config_manifest_path": "/tmp/compat.json",
                        "verify_config_manifest_path": "/tmp/verify.json",
                    }
                )
            )

            validation = analyze_sk_meta.validate_replay_report(replay_path)

        self.assertFalse(validation["valid"])
        self.assertEqual(validation["failure_reason"], "replay_artifact_path_invalid")

    def test_replay_validator_rejects_cross_candidate_name(self):
        with tempfile.TemporaryDirectory() as directory:
            replay_path = _write_replay_evidence(
                Path(directory) / "S1-replay.json", candidate_name="S1"
            )

            validation = analyze_sk_meta.validate_replay_report(
                replay_path, expected_candidate="S2"
            )

        self.assertFalse(validation["valid"])
        self.assertEqual(validation["failure_reason"], "replay_candidate_name_mismatch")

    def test_cli_replay_returns_one_for_shallow_verify(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compat_root = root / "compat"
            verify_root = root / "verify"
            _write_fused_metadata(
                compat_root,
                "decoder.layer.0.attention",
                ["A", "B", "C", "D", "E"],
                function_token="compat_hash",
                node_start=100,
            )
            _write_fused_metadata(
                verify_root,
                "decoder.layer.0.attention",
                ["A", "B", "C", "D"],
                function_token="verify_hash",
                node_start=900,
            )
            compat_config = _write_config_manifest(
                root / "compat-config.json", {"x": 1}
            )
            verify_config = _write_config_manifest(
                root / "verify-config.json", {"x": 1}
            )

            with redirect_stdout(io.StringIO()):
                exit_code = analyze_sk_meta.main(
                    [
                        str(compat_root),
                        "--verify-root",
                        str(verify_root),
                        "--compat-config-manifest",
                        str(compat_config),
                        "--verify-config-manifest",
                        str(verify_config),
                        "--candidate-name",
                        "S2",
                        "--round-id",
                        "S2-BASE",
                        "--source-revision",
                        "source-revision-1",
                    ]
                )

        self.assertEqual(exit_code, 1)

    def test_deep_fusion_replay_rejects_mismatched_config_and_validator_reports_it(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compat_root = root / "compat"
            verify_root = root / "verify"
            replay_out = root / "replay.json"
            child_ops = ["A", "B", "C", "D", "E"]
            _write_fused_metadata(
                compat_root, "scope", child_ops, function_token="a", node_start=1
            )
            _write_fused_metadata(
                verify_root, "scope", child_ops, function_token="b", node_start=20
            )
            compat_config = _write_config_manifest(
                root / "compat-config.json", {"scope": "S2"}
            )
            verify_config = _write_config_manifest(
                root / "verify-config.json", {"scope": "S3"}
            )

            replay = analyze_sk_meta.compare_deep_fusion_replay(
                compat_root,
                verify_root,
                compat_config_manifest=compat_config,
                verify_config_manifest=verify_config,
            )
            replay_out.write_text(json.dumps(replay))
            validation = analyze_sk_meta.validate_replay_report(replay_out)

        self.assertFalse(replay["deep_fusion_reproducible"])
        self.assertEqual(replay["failure_reason"], "config_fingerprint_mismatch")
        self.assertFalse(replay["config_fingerprint_match"])
        self.assertFalse(validation["valid"])
        self.assertFalse(validation["checks"]["config_fingerprint_match"])

    def test_replay_cli_requires_both_config_manifests(self):
        with self.assertRaises(SystemExit):
            analyze_sk_meta.main(["/tmp/compat", "--verify-root", "/tmp/verify"])

    def test_deep_fusion_replay_reports_missing_config_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compat_root = root / "compat"
            verify_root = root / "verify"
            child_ops = ["A", "B", "C", "D", "E"]
            _write_fused_metadata(
                compat_root, "scope", child_ops, function_token="a", node_start=1
            )
            _write_fused_metadata(
                verify_root, "scope", child_ops, function_token="b", node_start=20
            )
            verify_config = _write_config_manifest(
                root / "verify-config.json", {"scope": "S2"}
            )

            replay = analyze_sk_meta.compare_deep_fusion_replay(
                compat_root,
                verify_root,
                verify_config_manifest=verify_config,
            )

        self.assertFalse(replay["deep_fusion_reproducible"])
        self.assertEqual(replay["failure_reason"], "config_manifest_missing")
        self.assertIsNone(replay["compat_config_fingerprint"])

    def test_deep_fusion_replay_reports_invalid_config_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compat_root = root / "compat"
            verify_root = root / "verify"
            child_ops = ["A", "B", "C", "D", "E"]
            _write_fused_metadata(
                compat_root, "scope", child_ops, function_token="a", node_start=1
            )
            _write_fused_metadata(
                verify_root, "scope", child_ops, function_token="b", node_start=20
            )
            invalid_config = root / "compat-config.json"
            invalid_config.write_text("{")
            verify_config = _write_config_manifest(
                root / "verify-config.json", {"scope": "S2"}
            )

            replay = analyze_sk_meta.compare_deep_fusion_replay(
                compat_root,
                verify_root,
                compat_config_manifest=invalid_config,
                verify_config_manifest=verify_config,
            )

        self.assertFalse(replay["deep_fusion_reproducible"])
        self.assertEqual(replay["failure_reason"], "config_manifest_invalid")
        self.assertFalse(replay["compat_config_manifest"]["valid"])

    def test_replay_validator_requires_actual_matching_fingerprints(self):
        with tempfile.TemporaryDirectory() as directory:
            replay_path = _write_replay_evidence(Path(directory) / "replay.json")
            replay = json.loads(replay_path.read_text())
            del replay["compat_config_fingerprint"]
            del replay["verify_config_fingerprint"]
            replay_path.write_text(json.dumps(replay))

            validation = analyze_sk_meta.validate_replay_report(replay_path)

        self.assertFalse(validation["valid"])
        self.assertTrue(validation["checks"]["config_fingerprint_match"])
        self.assertFalse(validation["checks"]["report_matches_recomputed"])


class RoundReportRenderingTest(unittest.TestCase):
    def _round_report(self, round_name="S2"):
        return {
            "round_name": round_name,
            "min_child_nodes": 5,
            "effective_fusion": {
                "total_group_count": 2,
                "effective_group_count": 1,
                "shallow_group_count": 1,
                "effective_child_node_total": 6,
                "shallow_child_node_total": 2,
                "effective_child_count_histogram": {"6": 1},
                "shallow_child_count_histogram": {"2": 1},
                "effective_launch_reduction": 5,
                "all_fused_launch_reduction": 6,
            },
            "per_sk_fusion_table": [
                {
                    "function": "sk_eff|pipe",
                    "scope_id": 7,
                    "boundary": {"start_op": "MatMul", "end_op": "Add"},
                    "source_scope": "decoder.layer.0.attention",
                    "layer": "0",
                    "segments": ["attention"],
                    "child_count": 6,
                    "counts_as_effective_fusion": True,
                    "launch_count_without_sk": 6,
                    "launch_count_with_sk": 1,
                    "estimated_launch_reduction": 5,
                    "stream_ids": [1, 2],
                    "control_core_mode": "off",
                    "op_sequence": ["MatMul", "Add|Pipe", "Norm"],
                    "path": "/tmp/sk_eff",
                },
                {
                    "function": "sk_shallow",
                    "source_scope": "decoder.layer.0.mlp",
                    "child_count": 2,
                    "counts_as_effective_fusion": False,
                    "launch_count_without_sk": 2,
                    "launch_count_with_sk": 1,
                    "estimated_launch_reduction": 1,
                    "op_sequence": ["Mul", "Add"],
                },
            ],
            "non_fusion_operator_table": [
                {
                    "source": "sk_fusion_fail_reasons.log",
                    "reason": "OP_UNSUPPORT",
                    "count": 3,
                    "percent": 75.0,
                    "op_type": "Custom|Op",
                    "kernel_type": "AIV_ONLY",
                    "example_function": "static_kernel_Custom",
                    "example_node_id": 10,
                    "example_stream_id": 2,
                    "example_detail": "missing\nline",
                    "action": "move out",
                }
            ],
        }

    def test_render_markdown_summarizes_round_fusion_for_humans(self):
        markdown = render_round_report.render_markdown(
            [(Path("S2/round-report.child-ge-5.json"), self._round_report())],
            top_sk=1,
            top_non_fusion=1,
        )

        self.assertIn("# SuperKernel 单轮融合报告", markdown)
        self.assertIn("## S2", markdown)
        self.assertIn("child_count >= 5", markdown)
        self.assertIn("| 有效融合组 | 1 |", markdown)
        self.assertIn("| 浅融合组 | 1 |", markdown)
        self.assertIn("sk_eff\\|pipe", markdown)
        self.assertIn("decoder.layer.0.attention; MatMul -> Add; scope_id=7", markdown)
        self.assertIn("MatMul -> Add\\|Pipe -> Norm", markdown)
        self.assertIn("OP_UNSUPPORT", markdown)
        self.assertIn("Custom\\|Op", markdown)
        self.assertIn("missing<br>line", markdown)
        self.assertNotIn("sk_shallow", markdown)

    def test_render_markdown_keeps_single_child_descriptive_without_a1_or_gate(self):
        report = self._round_report()
        report["single_child_exclusion"] = {
            "requires_scope_adjustment": True,
            "single_child_group_count": 1,
            "shallow_non_single_child_group_count": 1,
            "s1_auto_action": "record_and_inherit_only",
            "candidates": [
                {
                    "function": "sk_single",
                    "source_scope": "decoder.layer.0",
                    "boundary": {"start_op": "MatMul", "end_op": "MatMul"},
                    "child_functions": ["static_kernel_MatMul"],
                    "path": "model_1_1/sk_fused_nodes.log",
                    "exclusion_status": "proposed",
                }
            ],
        }

        markdown = render_round_report.render_markdown(
            [(Path("S3-BASE/round-report.child-ge-5.json"), report)]
        )

        self.assertIn("### Child count 结构描述", markdown)
        self.assertNotIn("S3-A1", markdown)
        self.assertNotIn("必须调整", markdown)
        self.assertIn("profiling", markdown)
        self.assertIn("static_kernel_MatMul", markdown)
        self.assertNotIn("proposed", markdown)

    def test_render_cli_accepts_full_metadata_summary_and_writes_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sk-meta-summary.json"
            target = root / "round.md"
            source.write_text(
                json.dumps({"round_report": self._round_report("S1")}) + "\n"
            )

            exit_code = render_round_report.main(
                [str(source), "--markdown-out", str(target)]
            )

            self.assertEqual(exit_code, 0)
            markdown = target.read_text()

        self.assertIn("## S1", markdown)
        self.assertIn("来源:", markdown)
        self.assertIn("SK 融合范围明细", markdown)

    def test_rejects_json_without_round_report_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad.json"
            source.write_text(json.dumps({"effective_fusion": {}}) + "\n")

            with self.assertRaisesRegex(ValueError, "missing round-report fields"):
                render_round_report.load_round_report(source)


class ExperimentLedgerTest(unittest.TestCase):
    def _legacy_v1_ledger(self, experiment_id="S2"):
        legacy_result = {
            "experiment_id": experiment_id,
            "parent_experiment_id": experiment_id,
            "child_agent_id": "legacy-agent",
            "baseline_revision": "legacy-baseline-sha",
            "source_revision": "legacy-source-sha",
            "rounds": [
                {
                    "round_id": f"{experiment_id}-BASE",
                    "child_agent_id": "legacy-agent",
                    "scope_kind": "manual",
                    "single_child_group_count": 1,
                    "next_round_policy": "adjust_same_family",
                    "lifecycle": {"metadata": "passed"},
                    "artifacts": {
                        "round_evidence": f"{experiment_id}/round-evidence.json",
                        "round_report": f"{experiment_id}/round-report.json",
                    },
                }
            ],
            "single_child_exclusions": [
                {
                    "round_id": f"{experiment_id}-BASE",
                    "source_scope": "decoder.layer.legacy",
                    "scope_id": 7,
                    "child_functions": ["static_kernel_MatMul"],
                    "source_evidence": f"{experiment_id}/source-map.json",
                    "artifact": f"{experiment_id}/round-report.json",
                    "exclusion_status": "verified",
                    "retested_in": f"{experiment_id}-A1",
                }
            ],
            "inherited_exclusions_for_next_agents": [],
            "lifecycle": {"metadata": "passed", "replay": "passed"},
            "blockers": [],
            "next_agent_guidance_zh": "保留旧版单子算子排除记录。",
        }
        verified = {
            **json.loads(json.dumps(legacy_result["single_child_exclusions"][0])),
            "experiment_id": experiment_id,
            "child_agent_id": "legacy-agent",
            "source_revision": "legacy-source-sha",
        }
        return {
            "schema_version": 1,
            "experiments": {experiment_id: legacy_result},
            "verified_exclusions": [verified],
        }

    def _round(
        self,
        round_id,
        round_kind,
        candidate_fingerprint,
        performance_decision_range_ids,
        **extra,
    ):
        round_data = {
            "round_id": round_id,
            "round_kind": round_kind,
            "child_agent_id": "agent-7",
            "scope_kind": "manual",
            "source_revision": "source-sha",
            "baseline_config_fingerprint": "baseline-config-fp",
            "candidate_config_fingerprint": candidate_fingerprint,
            "control_fingerprint": "control-fp",
            "workload_fingerprint": "workload-fp",
            "baseline_profile": "profiles/frozen-baseline.json",
            "baseline_profile_fingerprint": "baseline-profile-fp",
            "candidate_profile": f"{round_id}/profiles/candidate.json",
            "candidate_profile_fingerprint": f"{round_id}-candidate-profile-fp",
            "declared_change_set": {
                "allowed_json_pointers": [
                    "/superkernel/enabled",
                    "/superkernel/scope",
                ],
                "only_change_zh": "只启用 SuperKernel 并应用本轮范围。",
            },
            "performance_decision_range_ids": performance_decision_range_ids,
            "lifecycle": {
                "compat": "passed",
                "correctness": "passed",
                "replay": "passed",
                "profiling": "passed",
            },
            "artifacts": {
                "round_evidence": f"{round_id}/round-evidence.json",
                "round_report": f"{round_id}/round-report.json",
            },
        }
        if round_kind in {"performance_prune", "final"}:
            round_data["declared_option_changes"] = []
        round_data.update(extra)
        return round_data

    def _decision(
        self,
        *,
        round_id,
        range_id,
        classification,
        action,
        status,
        candidate_fingerprint,
        verified_in_round_id=None,
    ):
        decision = {
            "round_id": round_id,
            "range_id": range_id,
            "sk_id": f"sk-{range_id}",
            "classification": classification,
            "action": action,
            "status": status,
            "mapping_method": "source_scope_map",
            "mapping_confidence": "exact",
            "source_scope": f"decoder.layer.0.{range_id}",
            "boundary": {
                "start_op": f"{range_id}-start",
                "end_op": f"{range_id}-end",
                "source_file": "model.py",
                "start_offset": 100,
                "end_offset": 200,
            },
            "source_revision": "source-sha",
            "baseline_config_fingerprint": "baseline-config-fp",
            "candidate_config_fingerprint": candidate_fingerprint,
            "control_fingerprint": "control-fp",
            "workload_fingerprint": "workload-fp",
            "baseline_profile": "profiles/frozen-baseline.json",
            "baseline_profile_fingerprint": "baseline-profile-fp",
            "candidate_profile": f"{round_id}/profiles/candidate.json",
            "candidate_profile_fingerprint": f"{round_id}-candidate-profile-fp",
            "declared_change_set": {
                "allowed_json_pointers": [
                    "/superkernel/enabled",
                    "/superkernel/scope",
                ],
                "only_change_zh": "只启用 SuperKernel 并应用本轮范围。",
            },
            "source_profiling_analysis_result": (
                f"{round_id}/profiling-analysis/profiling-analysis-result.json"
            ),
        }
        if verified_in_round_id is not None:
            decision["verified_in_round_id"] = verified_in_round_id
            decision["profiling_analysis_result"] = (
                f"{verified_in_round_id}/profiling-analysis/"
                "profiling-analysis-result.json"
            )
        return decision

    def _result(self):
        return {
            "experiment_id": "S3",
            "parent_experiment_id": "S3",
            "child_agent_id": "agent-7",
            "baseline_revision": "baseline-sha",
            "source_revision": "source-sha",
            "baseline_config_fingerprint": "baseline-config-fp",
            "control_fingerprint": "control-fp",
            "workload_fingerprint": "workload-fp",
            "profiling_analysis_agent_ids": {
                "S3-BASE": "prof-agent-1",
                "S3-P1": "prof-agent-2",
                "S3-FINAL": "prof-agent-3",
            },
            "profiling_analysis_result": {
                "S3-BASE": (
                    "S3-BASE/profiling-analysis/profiling-analysis-result.json"
                ),
                "S3-P1": "S3-P1/profiling-analysis/profiling-analysis-result.json",
                "S3-FINAL": (
                    "S3-FINAL/profiling-analysis/profiling-analysis-result.json"
                ),
            },
            "rounds": [
                self._round(
                    "S3-BASE",
                    "base",
                    "candidate-base-fp",
                    ["range-beneficial", "range-neutral", "range-unknown"],
                ),
                self._round(
                    "S3-P1",
                    "performance_prune",
                    "candidate-pruned-fp",
                    ["range-neutral"],
                ),
                self._round(
                    "S3-FINAL",
                    "final",
                    "candidate-final-fp",
                    ["range-beneficial"],
                    retained_range_ids=["range-beneficial"],
                ),
            ],
            "performance_scope_decisions": [
                self._decision(
                    round_id="S3-BASE",
                    range_id="range-beneficial",
                    classification="beneficial",
                    action="keep",
                    status="applied",
                    candidate_fingerprint="candidate-base-fp",
                ),
                self._decision(
                    round_id="S3-BASE",
                    range_id="range-neutral",
                    classification="neutral",
                    action="prune",
                    status="verified",
                    candidate_fingerprint="candidate-base-fp",
                    verified_in_round_id="S3-P1",
                ),
                self._decision(
                    round_id="S3-FINAL",
                    range_id="range-beneficial",
                    classification="beneficial",
                    action="keep",
                    status="applied",
                    candidate_fingerprint="candidate-final-fp",
                ),
            ],
            "unresolved_performance_ranges": [
                self._decision(
                    round_id="S3-BASE",
                    range_id="range-unknown",
                    classification="insufficient_evidence",
                    action="reprofile",
                    status="proposed",
                    candidate_fingerprint="candidate-base-fp",
                )
            ],
            "inherited_exclusions_for_next_agents": [],
            "lifecycle": {
                "metadata": "passed",
                "replay": "passed",
                "profiling": "passed",
                "clean": "not_run",
            },
            "blockers": [],
            "next_agent_guidance_zh": "后续实验沿用已验证的性能范围结论。",
        }

    def _freshen_profiling_evidence(self, result, prefix, *, fresh_baseline=False):
        rounds_by_id = {}
        for round_data in result["rounds"]:
            round_id = round_data["round_id"]
            rounds_by_id[round_id] = round_data
            round_data["artifacts"] = {
                "round_evidence": f"{prefix}/{round_id}/round-evidence.json",
                "round_report": f"{prefix}/{round_id}/round-report.json",
            }
            round_data["candidate_profile"] = (
                f"{prefix}/{round_id}/profiles/candidate.json"
            )
            round_data["candidate_profile_fingerprint"] = (
                f"{prefix}-{round_id}-candidate-profile-fp"
            )
            if fresh_baseline:
                round_data["baseline_profile"] = (
                    f"{prefix}/{round_id}/profiles/baseline.json"
                )
                round_data["baseline_profile_fingerprint"] = (
                    f"{prefix}-{round_id}-baseline-profile-fp"
                )
        result["profiling_analysis_agent_ids"] = {
            round_id: f"{prefix}-{round_id}-prof-agent" for round_id in rounds_by_id
        }
        result["profiling_analysis_result"] = {
            round_id: f"{prefix}/{round_id}/profiling-analysis/result.json"
            for round_id in rounds_by_id
        }
        for collection in (
            "performance_scope_decisions",
            "unresolved_performance_ranges",
        ):
            for decision in result[collection]:
                source_round = rounds_by_id[decision["round_id"]]
                for field in (
                    "baseline_profile",
                    "baseline_profile_fingerprint",
                    "candidate_profile",
                    "candidate_profile_fingerprint",
                    "declared_change_set",
                ):
                    decision[field] = json.loads(json.dumps(source_round[field]))
                decision["source_profiling_analysis_result"] = result[
                    "profiling_analysis_result"
                ][decision["round_id"]]
                if "verified_in_round_id" in decision:
                    decision["profiling_analysis_result"] = result[
                        "profiling_analysis_result"
                    ][decision["verified_in_round_id"]]

    def _sync_round_binding_field(self, result, round_id, field):
        round_data = next(
            item for item in result["rounds"] if item["round_id"] == round_id
        )
        for collection in (
            "performance_scope_decisions",
            "unresolved_performance_ranges",
        ):
            for decision in result[collection]:
                if decision["round_id"] == round_id:
                    decision[field] = json.loads(json.dumps(round_data[field]))

    def _run_merge_cli(self, directory, ledger, result):
        directory = Path(directory)
        ledger_path = directory / "ledger.json"
        result_path = directory / "result.json"
        output_path = directory / "merged.json"
        if isinstance(ledger, str):
            ledger_path.write_text(ledger)
        elif ledger is not None:
            ledger_path.write_text(json.dumps(ledger) + "\n")
        if isinstance(result, str):
            result_path.write_text(result)
        else:
            result_path.write_text(json.dumps(result) + "\n")
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = experiment_ledger.main(
                [
                    "merge",
                    "--ledger",
                    str(ledger_path),
                    "--result",
                    str(result_path),
                    "--output",
                    str(output_path),
                ]
            )
        return exit_code, stdout.getvalue(), stderr.getvalue(), output_path

    def _automatic_result_from_analysis(self, root, analysis_path):
        root = Path(root).resolve()
        analysis_path = Path(analysis_path).resolve()
        analysis = json.loads(analysis_path.read_text())
        validated = recommend_sk_strategy._validate_analysis("S4", analysis)
        producers = validated["decisions"]
        round_id = analysis["round_id"]

        def artifact(field):
            return str(
                (analysis_path.parent / analysis["inputs"][field])
                .resolve()
                .relative_to(root)
            )

        round_data = {
            "round_id": round_id,
            "round_kind": "automatic_aot",
            "child_agent_id": "agent-real",
            "scope_kind": "automatic_aot",
            "source_revision": analysis["source_revision"],
            "baseline_config_fingerprint": analysis["baseline_config_fingerprint"],
            "candidate_config_fingerprint": analysis["candidate_config_fingerprint"],
            "control_fingerprint": analysis["control_fingerprint"],
            "workload_fingerprint": analysis["workload_fingerprint"],
            "baseline_profile": artifact("baseline_profile"),
            "baseline_profile_fingerprint": analysis["baseline_profile_fingerprint"],
            "candidate_profile": artifact("candidate_profile"),
            "candidate_profile_fingerprint": analysis["candidate_profile_fingerprint"],
            "declared_change_set": json.loads(
                json.dumps(analysis["declared_change_set"])
            ),
            "performance_decision_range_ids": [
                producer["range_id"] for producer in producers
            ],
            "lifecycle": {
                "compat": "passed",
                "correctness": "passed",
                "replay": "passed",
                "profiling": "passed",
            },
            "artifacts": {
                "round_evidence": f"{round_id}/round-evidence.json",
                "round_report": f"{round_id}/round-report.json",
            },
        }
        decisions = []
        unresolved = []
        for producer in producers:
            decision = {
                field: json.loads(json.dumps(producer[field]))
                for field in (
                    "sk_id",
                    "range_id",
                    "classification",
                    "action",
                    "mapping_method",
                    "mapping_confidence",
                    "source_scope",
                    "boundary",
                )
            }
            for field in (
                "candidate_binding_status",
                "graph_occurrence_fingerprint",
                "mapping_blockers",
            ):
                if field in producer:
                    decision[field] = json.loads(json.dumps(producer[field]))
            decision.update(
                {
                    "round_id": round_id,
                    "status": (
                        "blocked" if producer["action"] == "block" else "proposed"
                    ),
                    "source_revision": analysis["source_revision"],
                    "baseline_config_fingerprint": analysis[
                        "baseline_config_fingerprint"
                    ],
                    "candidate_config_fingerprint": analysis[
                        "candidate_config_fingerprint"
                    ],
                    "control_fingerprint": analysis["control_fingerprint"],
                    "workload_fingerprint": analysis["workload_fingerprint"],
                    "baseline_profile": round_data["baseline_profile"],
                    "baseline_profile_fingerprint": analysis[
                        "baseline_profile_fingerprint"
                    ],
                    "candidate_profile": round_data["candidate_profile"],
                    "candidate_profile_fingerprint": analysis[
                        "candidate_profile_fingerprint"
                    ],
                    "declared_change_set": json.loads(
                        json.dumps(analysis["declared_change_set"])
                    ),
                    "source_profiling_analysis_result": str(
                        analysis_path.relative_to(root)
                    ),
                }
            )
            target = (
                decisions if producer["action"] in {"keep", "prune"} else unresolved
            )
            target.append(decision)
        return {
            "experiment_id": "S4",
            "parent_experiment_id": "S4",
            "child_agent_id": "agent-real",
            "baseline_revision": "baseline-real",
            "source_revision": analysis["source_revision"],
            "baseline_config_fingerprint": analysis["baseline_config_fingerprint"],
            "control_fingerprint": analysis["control_fingerprint"],
            "workload_fingerprint": analysis["workload_fingerprint"],
            "profiling_analysis_agent_ids": {round_id: analysis["analysis_agent_id"]},
            "profiling_analysis_result": {
                round_id: str(analysis_path.relative_to(root))
            },
            "rounds": [round_data],
            "performance_scope_decisions": decisions,
            "unresolved_performance_ranges": unresolved,
            "inherited_exclusions_for_next_agents": [],
            "lifecycle": {
                "metadata": "passed",
                "replay": "passed",
                "profiling": "passed",
                "clean": "not_run",
            },
            "blockers": [],
            "next_agent_guidance_zh": "后续实验只消费真实 analyzer 范围判定。",
        }

    def _mark_artifact_validated(self, ledger, *experiment_ids):
        marked = json.loads(json.dumps(ledger))
        validated = set(marked.get("artifact_validated_experiment_ids", []))
        validated.update(experiment_ids)
        marked["artifact_validated_experiment_ids"] = sorted(validated)
        marked["conditional_performance_evidence"] = (
            experiment_ledger._rebuild_conditional_evidence(
                marked["experiments"], validated
            )
        )
        return marked

    def test_validates_base_prune_final_rounds(self):
        validation = experiment_ledger.validate_experiment_result(self._result())

        self.assertTrue(validation["valid"])
        self.assertEqual(validation["errors"], [])

    def test_rejects_removed_round_kind_contract(self):
        result = self._result()
        result["rounds"].insert(
            2,
            self._round(
                "S3-X1",
                "legacy",
                "candidate-pruned-fp",
                ["range-neutral"],
            ),
        )

        validation = experiment_ledger.validate_experiment_result(result)

        self.assertFalse(validation["valid"])
        self.assertTrue(
            any("round_kind is invalid" in error for error in validation["errors"]),
            validation["errors"],
        )

    def test_rejects_historical_raw_id_exact_mapping_for_lifecycle_progress(self):
        result = self._result()
        for collection in (
            "performance_scope_decisions",
            "unresolved_performance_ranges",
        ):
            for decision in result[collection]:
                decision["mapping_method"] = "sk_meta_node_ids"
                decision["mapping_confidence"] = "exact"

        validation = experiment_ledger.validate_experiment_result(result)

        self.assertFalse(validation["valid"])
        self.assertTrue(
            any("source-actionable" in error for error in validation["errors"]),
            validation["errors"],
        )

    def test_structural_exact_is_performance_exact_but_not_source_actionable(self):
        decision = {
            "mapping_method": "kernel_projection_structural",
            "mapping_confidence": "exact_projected_trace",
            "candidate_binding_status": "bound",
            "graph_occurrence_fingerprint": "a" * 64,
            "mapping_blockers": [],
            "boundary": {"start_op": "A", "end_op": "B"},
        }

        self.assertTrue(experiment_ledger._mapping_is_performance_exact(decision))
        self.assertFalse(experiment_ledger._mapping_is_source_actionable(decision))

    def test_structural_exact_proposed_classification_cannot_be_applied(self):
        proposed = self._result()
        decision = self._decision(
            round_id="S3-BASE",
            range_id="range-structural",
            classification="beneficial",
            action="keep",
            status="proposed",
            candidate_fingerprint="candidate-base-fp",
        )
        decision.update(
            {
                "mapping_method": "kernel_projection_structural",
                "mapping_confidence": "exact_projected_trace",
                "candidate_binding_status": "bound",
                "graph_occurrence_fingerprint": "a" * 64,
                "mapping_blockers": [],
            }
        )
        decision["boundary"] = {"start_op": "A", "end_op": "B"}
        proposed["performance_scope_decisions"].append(decision)
        proposed["rounds"][0]["performance_decision_range_ids"].append(
            "range-structural"
        )

        validation = experiment_ledger.validate_experiment_result(proposed)
        self.assertTrue(validation["valid"], validation["errors"])

        decision["status"] = "applied"
        validation = experiment_ledger.validate_experiment_result(proposed)
        self.assertFalse(validation["valid"])
        self.assertTrue(
            any("source-actionable" in error for error in validation["errors"]),
            validation["errors"],
        )

    def test_ambiguous_and_diagnostic_mappings_are_not_performance_exact(self):
        for method, confidence in (
            ("kernel_projection_structural", "ambiguous"),
            ("kernel_projection_structural", "unmapped"),
            ("sk_meta_node_ids", "diagnostic_only"),
        ):
            with self.subTest(method=method, confidence=confidence):
                self.assertFalse(
                    experiment_ledger._mapping_is_performance_exact(
                        {
                            "mapping_method": method,
                            "mapping_confidence": confidence,
                        }
                    )
                )

    def test_non_exact_mappings_must_be_insufficient_in_full_ledger(self):
        for method, confidence in (
            ("kernel_projection_structural", "ambiguous"),
            ("kernel_projection_structural", "unmapped"),
            ("sk_meta_node_ids", "diagnostic_only"),
        ):
            with self.subTest(method=method, confidence=confidence):
                result = self._result()
                decision = result["unresolved_performance_ranges"][0]
                decision.update(
                    {
                        "classification": "regressed",
                        "action": "block",
                        "status": "blocked",
                        "mapping_method": method,
                        "mapping_confidence": confidence,
                    }
                )
                validation = experiment_ledger.validate_experiment_result(result)
                self.assertFalse(validation["valid"])
                self.assertTrue(
                    any(
                        "insufficient_evidence" in error
                        for error in validation["errors"]
                    ),
                    validation["errors"],
                )

    def test_object_validation_without_artifact_root_is_not_promotion_eligible(self):
        validation = experiment_ledger.validate_experiment_result(self._result())

        self.assertTrue(validation["valid"])
        self.assertFalse(validation["artifact_evidence_validated"])
        self.assertFalse(validation["conditional_evidence_eligible"])

    def test_validate_cli_rejects_missing_profiling_analysis_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment-result.json"
            path.write_text(json.dumps(self._result()) + "\n")
            stdout = io.StringIO()
            stderr = io.StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = experiment_ledger.main(["validate", str(path)])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertFalse(payload["valid"])
        self.assertTrue(
            any("profiling_analysis_result" in error for error in payload["errors"])
        )
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_object_merge_without_artifact_root_cannot_promote_conditional_evidence(
        self,
    ):
        ledger = experiment_ledger.merge_experiment_result({}, self._result())

        self.assertIn("S3", ledger["experiments"])
        self.assertEqual(ledger["artifact_validated_experiment_ids"], [])
        self.assertEqual(ledger["conditional_performance_evidence"], {})

    def test_merge_cli_rejects_missing_profiling_analysis_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            exit_code, stdout, stderr, output = self._run_merge_cli(
                directory, {}, self._result()
            )

        payload = json.loads(stdout)
        self.assertEqual(exit_code, 1)
        self.assertFalse(payload["valid"])
        self.assertTrue(
            any("profiling_analysis_result" in error for error in payload["errors"])
        )
        self.assertNotIn("Traceback", stderr)
        self.assertFalse(output.exists())

    def test_real_analyzer_result_flows_through_strategy_and_schema2_ledger_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis_path, _ = StrategyTest._write_real_analysis_fixture(root)
            analysis = json.loads(analysis_path.read_text())

            strategy = recommend_sk_strategy.build_strategy(
                {"S4": analysis}, analysis_paths={"S4": analysis_path}
            )
            self.assertEqual(strategy["candidates"]["S4"]["source_round_id"], "S4-BASE")

            result = self._automatic_result_from_analysis(root, analysis_path)
            result_path = root / "experiment-result.json"
            result_path.write_text(json.dumps(result) + "\n")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                validate_exit = experiment_ledger.main(["validate", str(result_path)])
            validation = json.loads(stdout.getvalue())
            self.assertEqual(validate_exit, 0, validation["errors"])
            self.assertTrue(validation["artifact_evidence_validated"])

            merge_exit, _, merge_stderr, output = self._run_merge_cli(root, {}, result)
            self.assertEqual(merge_exit, 0, merge_stderr)
            ledger = json.loads(output.read_text())
            self.assertEqual(ledger["artifact_validated_experiment_ids"], ["S4"])
            self.assertIn("S4", ledger["experiments"])

    @unittest.skip(
        "legacy task-range exact fixture; source_scope_map_v2 flow is covered in sibling tests"
    )
    def test_real_base_p_strategy_rounds_flow_into_complete_schema2_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path, base_artifacts = StrategyTest._write_real_analysis_fixture(
                root / "base",
                range_count=2,
                dcci_recommendation=True,
            )
            p_path, _ = StrategyTest._write_real_analysis_fixture(
                root / "p1",
                range_count=2,
                round_id="S4-P1",
                candidate_ranges=("B",),
                dcci_recommendation=True,
                shared_baseline=base_artifacts,
            )
            final_path, _ = StrategyTest._write_real_analysis_fixture(
                root / "final",
                range_count=2,
                round_id="S4-FINAL",
                candidate_ranges=("B",),
                candidate_durations={"B": 5},
                dcci_recommendation=True,
                shared_baseline=base_artifacts,
            )
            base_analysis = json.loads(base_path.read_text())
            p_analysis = json.loads(p_path.read_text())
            prune_plan = recommend_sk_strategy.build_strategy(
                {"S4": base_analysis}, analysis_paths={"S4": base_path}
            )["candidates"]["S4"]["next_candidate_matrix"][0]
            final_plan = recommend_sk_strategy.build_strategy(
                {"S4": [base_analysis, p_analysis]},
                analysis_paths={"S4": [base_path, p_path]},
            )["candidates"]["S4"]["next_candidate_matrix"]
            self.assertEqual(prune_plan["id"], "S4-P1")
            self.assertEqual({item["comparison_to"] for item in final_plan}, {"S4-P1"})

            base_result = self._automatic_result_from_analysis(root, base_path)
            p_result = self._automatic_result_from_analysis(root, p_path)
            final_result = self._automatic_result_from_analysis(root, final_path)
            removed_range_ids = prune_plan["target_range_ids"]
            base_result["rounds"][0].update(
                {"round_kind": "base", "scope_kind": "manual"}
            )
            for decision in base_result["performance_scope_decisions"]:
                decision["status"] = "applied"
            p_result["rounds"][0].update(
                {
                    "round_kind": "performance_prune",
                    "scope_kind": "manual",
                    "performance_decision_range_ids": removed_range_ids,
                }
            )
            retained_range_ids = [
                decision["range_id"]
                for decision in final_result["performance_scope_decisions"]
            ]
            final_result["rounds"][0].update(
                {
                    "round_kind": "final",
                    "scope_kind": "manual",
                    "performance_decision_range_ids": retained_range_ids,
                    "retained_range_ids": retained_range_ids,
                }
            )
            for decision in final_result["performance_scope_decisions"]:
                decision["status"] = "applied"

            for addition in (p_result, final_result):
                base_result["rounds"].extend(addition["rounds"])
                base_result["profiling_analysis_agent_ids"].update(
                    addition["profiling_analysis_agent_ids"]
                )
                base_result["profiling_analysis_result"].update(
                    addition["profiling_analysis_result"]
                )
                for collection in (
                    "performance_scope_decisions",
                    "unresolved_performance_ranges",
                ):
                    base_result[collection].extend(addition[collection])

            result_path = root / "experiment-result.json"
            result_path.write_text(json.dumps(base_result) + "\n")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                validate_exit = experiment_ledger.main(["validate", str(result_path)])
            validation = json.loads(stdout.getvalue())
            self.assertEqual(validate_exit, 0, validation["errors"])
            self.assertTrue(validation["artifact_evidence_validated"])

            merge_exit, _, merge_stderr, output = self._run_merge_cli(
                root, {}, base_result
            )
            self.assertEqual(merge_exit, 0, merge_stderr)
            ledger = json.loads(output.read_text())
            self.assertEqual(ledger["artifact_validated_experiment_ids"], ["S4"])
            self.assertEqual(
                set(ledger["experiments"]["S4"]["profiling_analysis_result"]),
                {"S4-BASE", "S4-P1", "S4-FINAL"},
            )

    def test_idempotent_merge_revalidates_new_result_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis_path, _ = StrategyTest._write_real_analysis_fixture(root)
            result = self._automatic_result_from_analysis(root, analysis_path)
            first_exit, _, first_stderr, output = self._run_merge_cli(root, {}, result)
            self.assertEqual(first_exit, 0, first_stderr)
            ledger = json.loads(output.read_text())
            self.assertEqual(ledger["artifact_validated_experiment_ids"], ["S4"])

            tampered = json.loads(analysis_path.read_text())
            tampered["per_sk_decisions"][0]["classification"] = "beneficial"
            analysis_path.write_text(json.dumps(tampered) + "\n")
            second_exit, second_stdout, second_stderr, _ = self._run_merge_cli(
                root, ledger, result
            )
            failure = json.loads(second_stdout)
            self.assertEqual(second_exit, 1)
            self.assertFalse(failure["valid"])
            self.assertTrue(
                any(
                    "analysis_content_fingerprint" in error
                    for error in failure["errors"]
                ),
                failure["errors"],
            )
            self.assertNotIn("Traceback", second_stderr)

    def test_cli_rejects_real_analysis_when_all_producer_ranges_are_omitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis_path, _ = StrategyTest._write_real_analysis_fixture(root)
            result = self._automatic_result_from_analysis(root, analysis_path)
            result["performance_scope_decisions"] = []
            result["unresolved_performance_ranges"] = []
            result_path = root / "experiment-result.json"
            result_path.write_text(json.dumps(result) + "\n")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                validate_exit = experiment_ledger.main(["validate", str(result_path)])
            validation = json.loads(stdout.getvalue())
            self.assertEqual(validate_exit, 1)
            self.assertTrue(
                any(
                    "complete analyzer coverage" in error
                    for error in validation["errors"]
                ),
                validation["errors"],
            )

            merge_exit, merge_stdout, _, output = self._run_merge_cli(root, {}, result)
            self.assertEqual(merge_exit, 1)
            self.assertTrue(
                any(
                    "complete analyzer coverage" in error
                    for error in json.loads(merge_stdout)["errors"]
                )
            )
            self.assertFalse(output.exists())

    def test_cli_rejects_real_analysis_when_one_of_multiple_ranges_is_omitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis_path, _ = StrategyTest._write_real_analysis_fixture(
                root, range_count=2
            )
            result = self._automatic_result_from_analysis(root, analysis_path)
            result["performance_scope_decisions"] = result[
                "performance_scope_decisions"
            ][:1]
            result["unresolved_performance_ranges"] = []
            result_path = root / "experiment-result.json"
            result_path.write_text(json.dumps(result) + "\n")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                validate_exit = experiment_ledger.main(["validate", str(result_path)])
            validation = json.loads(stdout.getvalue())
            self.assertEqual(validate_exit, 1)
            self.assertTrue(
                any(
                    "complete analyzer coverage" in error
                    for error in validation["errors"]
                ),
                validation["errors"],
            )

            merge_exit, merge_stdout, _, output = self._run_merge_cli(root, {}, result)
            self.assertEqual(merge_exit, 1)
            self.assertTrue(
                any(
                    "complete analyzer coverage" in error
                    for error in json.loads(merge_stdout)["errors"]
                )
            )
            self.assertFalse(output.exists())

    def test_artifact_validation_rejects_symlink_escape_and_report_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            container = Path(directory)
            root = container / "artifacts"
            root.mkdir()
            analysis_path, _ = StrategyTest._write_real_analysis_fixture(root)
            result = self._automatic_result_from_analysis(root, analysis_path)
            round_id = next(iter(result["profiling_analysis_result"]))

            outside = container / "outside-analysis.json"
            outside.write_bytes(analysis_path.read_bytes())
            escaped = root / "linked-analysis.json"
            escaped.symlink_to(outside)
            escaped_result = json.loads(json.dumps(result))
            escaped_result["profiling_analysis_result"][round_id] = escaped.name
            for collection in (
                "performance_scope_decisions",
                "unresolved_performance_ranges",
            ):
                for decision in escaped_result[collection]:
                    if decision["round_id"] == round_id:
                        decision["source_profiling_analysis_result"] = escaped.name
            validation = experiment_ledger.validate_experiment_result(
                escaped_result, artifact_root=root
            )
            self.assertFalse(validation["valid"])
            self.assertTrue(any("escapes" in error for error in validation["errors"]))

            original = analysis_path.read_bytes()
            mutations = {
                "schema": lambda report: report.update({"schema_version": "0.9"}),
                "identity": lambda report: report.update({"experiment_id": "S9"}),
                "agent": lambda report: report.update(
                    {"analysis_agent_id": "forged-agent"}
                ),
                "round": lambda report: report.update({"round_id": "S4-P9"}),
                "fingerprint": lambda report: report.update(
                    {"candidate_profile_fingerprint": "f" * 64}
                ),
            }
            for label, mutate in mutations.items():
                with self.subTest(label=label):
                    report = json.loads(original)
                    mutate(report)
                    StrategyTest._rebind_analysis_identity(report)
                    analysis_path.write_text(json.dumps(report))
                    validation = experiment_ledger.validate_experiment_result(
                        result, artifact_root=root
                    )
                    self.assertFalse(validation["valid"], label)
                    analysis_path.write_bytes(original)

            report = json.loads(original)
            report["blockers"] = ["tampered after producer seal"]
            analysis_path.write_text(json.dumps(report))
            validation = experiment_ledger.validate_experiment_result(
                result, artifact_root=root
            )
            self.assertFalse(validation["valid"])
            self.assertTrue(
                any("content_fingerprint" in error for error in validation["errors"])
            )

            analysis_path.write_bytes(original)
            candidate_profile = (
                analysis_path.parent
                / json.loads(original)["inputs"]["candidate_profile"]
            ).resolve()
            candidate_profile.write_text(candidate_profile.read_text() + "tampered\n")
            validation = experiment_ledger.validate_experiment_result(
                result, artifact_root=root
            )
            self.assertFalse(validation["valid"])
            self.assertTrue(
                any("fingerprint" in error for error in validation["errors"])
            )

    def test_artifact_validation_rejects_experiment_decision_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis_path, _ = StrategyTest._write_real_analysis_fixture(root)
            result = self._automatic_result_from_analysis(root, analysis_path)
            decision = (
                result["performance_scope_decisions"]
                or result["unresolved_performance_ranges"]
            )[0]
            decision["boundary"]["start_op"] += "-forged"

            validation = experiment_ledger.validate_experiment_result(
                result, artifact_root=root
            )

            self.assertFalse(validation["valid"])
            self.assertTrue(
                any("per_sk_decisions" in error for error in validation["errors"]),
                validation["errors"],
            )

    def test_defines_exact_performance_enums(self):
        self.assertEqual(
            experiment_ledger.ROUND_KINDS,
            {"base", "performance_prune", "final", "automatic_aot"},
        )
        self.assertEqual(
            experiment_ledger.PERFORMANCE_CLASSES,
            {"beneficial", "regressed", "neutral", "insufficient_evidence"},
        )
        self.assertEqual(
            experiment_ledger.PERFORMANCE_ACTIONS,
            {"keep", "prune", "reprofile", "block"},
        )
        self.assertEqual(
            experiment_ledger.DECISION_STATUSES,
            {"proposed", "applied", "verified", "blocked"},
        )

    def test_rejects_mixed_parent_experiment_ownership(self):
        result = self._result()
        result["rounds"][2]["child_agent_id"] = "agent-other"

        validation = experiment_ledger.validate_experiment_result(result)

        self.assertFalse(validation["valid"])
        self.assertTrue(
            any("child_agent_id" in error for error in validation["errors"])
        )

    def test_rejects_duplicate_profiling_analysis_agent_ids(self):
        result = self._result()
        result["profiling_analysis_agent_ids"]["S3-FINAL"] = "prof-agent-2"

        validation = experiment_ledger.validate_experiment_result(result)

        self.assertFalse(validation["valid"])
        self.assertTrue(any("unique" in error for error in validation["errors"]))

    def test_rejects_final_reusing_a_profiling_agent_or_result_path(self):
        for mapping_name, reused_value in (
            ("profiling_analysis_agent_ids", "prof-agent-2"),
            (
                "profiling_analysis_result",
                "S3-P1/profiling-analysis/profiling-analysis-result.json",
            ),
        ):
            with self.subTest(mapping_name=mapping_name):
                result = self._result()
                result[mapping_name]["S3-FINAL"] = reused_value

                validation = experiment_ledger.validate_experiment_result(result)

                self.assertFalse(validation["valid"])
                self.assertTrue(
                    any("unique" in error for error in validation["errors"]),
                    validation["errors"],
                )

    def test_rejects_absolute_analysis_result_path(self):
        result = self._result()
        result["profiling_analysis_result"]["S3-P1"] = "/tmp/analysis.json"

        validation = experiment_ledger.validate_experiment_result(result)

        self.assertFalse(validation["valid"])
        self.assertTrue(any("relative" in error for error in validation["errors"]))

    def test_rejects_parent_traversal_in_any_artifact_path(self):
        result = self._result()
        result["single_child_exclusions"] = [
            {"source_evidence": "../outside/source-map.json"}
        ]

        validation = experiment_ledger.validate_experiment_result(result)

        self.assertFalse(validation["valid"])
        self.assertTrue(any("relative" in error for error in validation["errors"]))

    def test_rejects_missing_or_unknown_profiling_round_mapping(self):
        for mapping_name in (
            "profiling_analysis_agent_ids",
            "profiling_analysis_result",
        ):
            with self.subTest(mapping_name=mapping_name, issue="missing"):
                result = self._result()
                del result[mapping_name]["S3-P1"]
                self.assertFalse(
                    experiment_ledger.validate_experiment_result(result)["valid"]
                )
            with self.subTest(mapping_name=mapping_name, issue="unknown"):
                result = self._result()
                result[mapping_name]["S3-UNKNOWN"] = "unexpected"
                self.assertFalse(
                    experiment_ledger.validate_experiment_result(result)["valid"]
                )

    def test_rejects_beneficial_prune(self):
        result = self._result()
        result["performance_scope_decisions"][0]["action"] = "prune"

        validation = experiment_ledger.validate_experiment_result(result)

        self.assertFalse(validation["valid"])
        self.assertTrue(any("beneficial" in error for error in validation["errors"]))

    def test_rejects_insufficient_evidence_prune(self):
        result = self._result()
        result["unresolved_performance_ranges"][0]["action"] = "prune"

        validation = experiment_ledger.validate_experiment_result(result)

        self.assertFalse(validation["valid"])
        self.assertTrue(
            any("insufficient_evidence" in error for error in validation["errors"])
        )

    def test_rejects_effective_prune_without_exact_producer_mapping(self):
        result = self._result()
        result["performance_scope_decisions"][1]["mapping_confidence"] = (
            "diagnostic_only"
        )

        validation = experiment_ledger.validate_experiment_result(result)

        self.assertFalse(validation["valid"])
        self.assertTrue(any("exact" in error for error in validation["errors"]))

    def test_rejects_effective_keep_without_exact_producer_mapping(self):
        result = self._result()
        result["performance_scope_decisions"][0]["mapping_confidence"] = (
            "diagnostic_only"
        )

        validation = experiment_ledger.validate_experiment_result(result)

        self.assertFalse(validation["valid"])
        self.assertTrue(
            any("effective keep" in error for error in validation["errors"])
        )

    def test_rejects_prune_round_without_eligible_decision(self):
        result = self._result()
        result["rounds"][1]["performance_decision_range_ids"] = ["range-beneficial"]

        validation = experiment_ledger.validate_experiment_result(result)

        self.assertFalse(validation["valid"])
        self.assertTrue(
            any("performance_prune" in error for error in validation["errors"])
        )

    def test_prune_round_accepts_applied_decision_without_verification_fields(self):
        result = self._result()
        decision = result["performance_scope_decisions"][1]
        decision["status"] = "applied"
        del decision["verified_in_round_id"]
        del decision["profiling_analysis_result"]

        validation = experiment_ledger.validate_experiment_result(result)

        self.assertTrue(validation["valid"], validation["errors"])

    def test_prune_round_cannot_self_reference_as_its_only_source_decision(self):
        result = self._result()
        source_decision = result["performance_scope_decisions"][1]
        source_decision["action"] = "reprofile"
        source_decision["status"] = "proposed"
        del source_decision["verified_in_round_id"]
        del source_decision["profiling_analysis_result"]
        result["performance_scope_decisions"].append(
            self._decision(
                round_id="S3-P1",
                range_id="range-neutral",
                classification="regressed",
                action="prune",
                status="applied",
                candidate_fingerprint="candidate-pruned-fp",
            )
        )

        validation = experiment_ledger.validate_experiment_result(result)

        self.assertFalse(validation["valid"])
        self.assertTrue(
            any("performance_prune" in error for error in validation["errors"]),
            validation["errors"],
        )

    def test_prune_round_rejects_keep_evidence_for_a_removed_range(self):
        result = self._result()
        result["performance_scope_decisions"].append(
            self._decision(
                round_id="S3-P1",
                range_id="range-neutral",
                classification="beneficial",
                action="keep",
                status="applied",
                candidate_fingerprint="candidate-pruned-fp",
            )
        )

        validation = experiment_ledger.validate_experiment_result(result)

        self.assertFalse(validation["valid"])
        self.assertTrue(
            any("removed range" in error for error in validation["errors"]),
            validation["errors"],
        )

    def test_multi_range_prune_allows_evidence_for_an_unpruned_range(self):
        result = self._result()
        result["rounds"][0]["performance_decision_range_ids"].append("range-neutral-2")
        result["rounds"][1]["performance_decision_range_ids"].append("range-neutral-2")
        result["performance_scope_decisions"].extend(
            [
                self._decision(
                    round_id="S3-BASE",
                    range_id="range-neutral-2",
                    classification="neutral",
                    action="prune",
                    status="verified",
                    candidate_fingerprint="candidate-base-fp",
                    verified_in_round_id="S3-P1",
                ),
                self._decision(
                    round_id="S3-P1",
                    range_id="range-beneficial",
                    classification="beneficial",
                    action="keep",
                    status="applied",
                    candidate_fingerprint="candidate-pruned-fp",
                ),
            ]
        )

        validation = experiment_ledger.validate_experiment_result(result)

        self.assertTrue(validation["valid"], validation["errors"])

    def test_prune_and_final_forbid_option_changes(self):
        for index, round_kind in ((1, "performance_prune"), (2, "final")):
            with self.subTest(round_kind=round_kind):
                result = self._result()
                result["rounds"][index]["declared_option_changes"] = [
                    {"option": "auto_op_parallel", "before": 0, "after": 1}
                ]

                validation = experiment_ledger.validate_experiment_result(result)

                self.assertFalse(validation["valid"])
                self.assertTrue(
                    any(
                        "forbids option changes" in error
                        for error in validation["errors"]
                    ),
                    validation["errors"],
                )

    def test_prune_and_final_reject_option_pointers(self):
        for index, round_kind in ((1, "performance_prune"), (2, "final")):
            with self.subTest(round_kind=round_kind):
                result = self._result()
                result["rounds"][index]["declared_change_set"][
                    "allowed_json_pointers"
                ] = ["/superkernel/options/auto_op_parallel"]

                validation = experiment_ledger.validate_experiment_result(result)

                self.assertFalse(validation["valid"])
                self.assertTrue(
                    any(
                        "must not target options" in error
                        for error in validation["errors"]
                    ),
                    validation["errors"],
                )

    def test_rejects_final_without_fresh_profiling(self):
        result = self._result()
        result["rounds"][2]["lifecycle"]["profiling"] = "not_run"
        del result["profiling_analysis_agent_ids"]["S3-FINAL"]
        del result["profiling_analysis_result"]["S3-FINAL"]

        validation = experiment_ledger.validate_experiment_result(result)

        self.assertFalse(validation["valid"])
        self.assertTrue(
            any(
                "final requires fresh profiling" in error
                for error in validation["errors"]
            )
        )

    def test_rejects_conflicting_records_for_one_round_and_range(self):
        result = self._result()
        result["unresolved_performance_ranges"].append(
            self._decision(
                round_id="S3-BASE",
                range_id="range-neutral",
                classification="neutral",
                action="block",
                status="blocked",
                candidate_fingerprint="candidate-base-fp",
            )
        )

        validation = experiment_ledger.validate_experiment_result(result)

        self.assertFalse(validation["valid"])
        self.assertTrue(any("conflicting" in error for error in validation["errors"]))

    def test_verified_decision_references_declared_path_and_round_fingerprint(self):
        mutations = (
            (
                "analysis path",
                lambda result: result["performance_scope_decisions"][1].update(
                    {
                        "profiling_analysis_result": result[
                            "profiling_analysis_result"
                        ]["S3-BASE"]
                    }
                ),
            ),
            (
                "candidate fingerprint",
                lambda result: result["performance_scope_decisions"][1].update(
                    {"candidate_config_fingerprint": "candidate-final-fp"}
                ),
            ),
        )
        for issue, mutate in mutations:
            with self.subTest(issue=issue):
                result = self._result()
                mutate(result)

                validation = experiment_ledger.validate_experiment_result(result)

                self.assertFalse(validation["valid"])
                self.assertTrue(
                    any(
                        token in error
                        for token in ("profiling_analysis_result", "fingerprint")
                        for error in validation["errors"]
                    ),
                    validation["errors"],
                )

    def test_malformed_round_references_return_validation_errors(self):
        for mutate in (
            lambda result: result["rounds"][1].update(
                {"performance_decision_range_ids": [{}]}
            ),
            lambda result: result["performance_scope_decisions"][1].update(
                {"round_id": []}
            ),
            lambda result: result["rounds"][1].update({"round_kind": []}),
            lambda result: result["performance_scope_decisions"][1].update(
                {"classification": []}
            ),
        ):
            with self.subTest(mutate=mutate):
                result = self._result()
                mutate(result)

                validation = experiment_ledger.validate_experiment_result(result)

                self.assertFalse(validation["valid"])

    def test_json_type_mutations_never_escape_validation(self):
        mutations = []
        for invalid in ([], {}, True, None):
            mutations.extend(
                (
                    (
                        f"round_id={invalid!r}",
                        lambda result, value=invalid: result["rounds"][1].update(
                            {"round_id": value}
                        ),
                    ),
                    (
                        f"decision_range_id={invalid!r}",
                        lambda result, value=invalid: result[
                            "performance_scope_decisions"
                        ][1].update({"range_id": value}),
                    ),
                    (
                        f"verified_round_id={invalid!r}",
                        lambda result, value=invalid: result[
                            "performance_scope_decisions"
                        ][1].update({"verified_in_round_id": value}),
                    ),
                    (
                        f"declared_change_set={invalid!r}",
                        lambda result, value=invalid: result["rounds"][2].update(
                            {"declared_change_set": value}
                        ),
                    ),
                )
            )
        for name, mutate in mutations:
            with self.subTest(name=name):
                result = self._result()
                mutate(result)

                validation = experiment_ledger.validate_experiment_result(result)

                self.assertFalse(validation["valid"])
                self.assertTrue(validation["errors"])

    def test_validate_cli_returns_structured_errors_without_traceback(self):
        result = self._result()
        result["rounds"][1]["round_id"] = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(result) + "\n")
            stdout = io.StringIO()
            stderr = io.StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = experiment_ledger.main(["validate", str(path)])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertFalse(payload["valid"])
        self.assertTrue(payload["errors"])
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_validate_cli_handles_invalid_json_syntax_without_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text('{"rounds": [}\n')
            stdout = io.StringIO()
            stderr = io.StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = experiment_ledger.main(["validate", str(path)])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertFalse(payload["valid"])
        self.assertTrue(any("JSON" in error for error in payload["errors"]))
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_rejects_non_finite_and_non_json_values_globally(self):
        for invalid in (float("nan"), float("inf"), float("-inf"), ("tuple",)):
            with self.subTest(invalid=invalid):
                result = self._result()
                result["blockers"] = [invalid]

                validation = experiment_ledger.validate_experiment_result(result)

                self.assertFalse(validation["valid"])
                self.assertTrue(
                    any("JSON" in error for error in validation["errors"]),
                    validation["errors"],
                )

    def test_validate_cli_rejects_nonstandard_json_constants(self):
        for constant in ("NaN", "Infinity", "-Infinity"):
            with (
                self.subTest(constant=constant),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory) / "invalid.json"
                payload = json.dumps(self._result())
                path.write_text(
                    payload.replace('"blockers": []', f'"blockers": [{constant}]')
                )
                stdout = io.StringIO()
                stderr = io.StringIO()

                with redirect_stdout(stdout), redirect_stderr(stderr):
                    exit_code = experiment_ledger.main(["validate", str(path)])

                validation = json.loads(stdout.getvalue())
                self.assertEqual(exit_code, 1)
                self.assertFalse(validation["valid"])
                self.assertTrue(
                    any(
                        "non-standard JSON constant" in error
                        for error in validation["errors"]
                    ),
                    validation["errors"],
                )
                self.assertNotIn("Traceback", stderr.getvalue())

    def test_merge_cli_returns_structured_errors_for_malformed_ledger_json(self):
        with tempfile.TemporaryDirectory() as directory:
            exit_code, stdout, stderr, output = self._run_merge_cli(
                directory, '{"experiments": [}\n', self._result()
            )

            payload = json.loads(stdout)
            self.assertEqual(exit_code, 1)
            self.assertFalse(payload["valid"])
            self.assertTrue(any("JSON" in error for error in payload["errors"]))
            self.assertNotIn("Traceback", stderr)
            self.assertFalse(output.exists())

    def test_merge_cli_returns_structured_errors_for_malformed_result_json(self):
        with tempfile.TemporaryDirectory() as directory:
            exit_code, stdout, stderr, output = self._run_merge_cli(
                directory, {}, '{"rounds": [}\n'
            )

            payload = json.loads(stdout)
            self.assertEqual(exit_code, 1)
            self.assertFalse(payload["valid"])
            self.assertTrue(any("JSON" in error for error in payload["errors"]))
            self.assertNotIn("Traceback", stderr)
            self.assertFalse(output.exists())

    def test_merge_cli_returns_structured_errors_for_invalid_result(self):
        with tempfile.TemporaryDirectory() as directory:
            exit_code, stdout, stderr, output = self._run_merge_cli(
                directory, {}, {"experiment_id": "incomplete"}
            )

            payload = json.loads(stdout)
            self.assertEqual(exit_code, 1)
            self.assertFalse(payload["valid"])
            self.assertTrue(payload["errors"])
            self.assertNotIn("Traceback", stderr)
            self.assertFalse(output.exists())

    def test_merge_cli_returns_structured_errors_for_non_object_inputs(self):
        cases = (([], self._result()), ({}, []))
        for ledger, result in cases:
            with self.subTest(ledger_type=type(ledger), result_type=type(result)):
                with tempfile.TemporaryDirectory() as directory:
                    exit_code, stdout, stderr, output = self._run_merge_cli(
                        directory, ledger, result
                    )

                    payload = json.loads(stdout)
                    self.assertEqual(exit_code, 1)
                    self.assertFalse(payload["valid"])
                    self.assertTrue(payload["errors"])
                    self.assertNotIn("Traceback", stderr)
                    self.assertFalse(output.exists())

    def test_merge_cli_returns_structured_errors_for_output_os_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis_path, _ = StrategyTest._write_real_analysis_fixture(root)
            result = self._automatic_result_from_analysis(root, analysis_path)
            output_path = root / "merged.json"
            output_path.mkdir()

            exit_code, stdout, stderr, _ = self._run_merge_cli(directory, {}, result)

            payload = json.loads(stdout)
            self.assertEqual(exit_code, 1)
            self.assertFalse(payload["valid"])
            self.assertTrue(
                any("IsADirectoryError" in error for error in payload["errors"])
            )
            self.assertNotIn("Traceback", stderr)

    def test_merge_cli_returns_structured_errors_for_tampered_ledger(self):
        ledger = experiment_ledger.merge_experiment_result({}, self._result())
        ledger = self._mark_artifact_validated(ledger, "S3")
        key = next(iter(ledger["conditional_performance_evidence"]))
        ledger["conditional_performance_evidence"][key]["evidence_kind"] = "invalid"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis_path, _ = StrategyTest._write_real_analysis_fixture(root)
            result = self._automatic_result_from_analysis(root, analysis_path)
            exit_code, stdout, stderr, output = self._run_merge_cli(
                directory, ledger, result
            )

            payload = json.loads(stdout)
            self.assertEqual(exit_code, 1)
            self.assertFalse(payload["valid"])
            self.assertTrue(any("conditional" in error for error in payload["errors"]))
            self.assertNotIn("Traceback", stderr)
            self.assertFalse(output.exists())

    def test_merge_rejects_non_object_ledgers(self):
        for ledger in ([], "ledger", None, 7, True):
            with self.subTest(ledger=ledger):
                with self.assertRaisesRegex(ValueError, "ledger must be an object"):
                    experiment_ledger.merge_experiment_result(ledger, self._result())

    def test_migrates_real_legacy_v1_ledger_without_promoting_legacy_evidence(self):
        legacy = self._legacy_v1_ledger()
        legacy["experiments"]["S2"]["metadata"] = {
            "compat": ["S2/sk-meta/compat.json"],
            "verify": {"artifact": "S2/sk-meta/verify.json"},
        }

        merged = experiment_ledger.merge_experiment_result(legacy, self._result())

        self.assertEqual(merged["schema_version"], 2)
        self.assertEqual(merged["migrated_from_schema_version"], 1)
        self.assertIn("S2", merged["legacy_experiments"])
        self.assertIn("S3", merged["experiments"])
        self.assertEqual(
            merged["legacy_verified_exclusions"], legacy["verified_exclusions"]
        )
        self.assertEqual(
            merged["legacy_experiments"]["S2"]["metadata"],
            legacy["experiments"]["S2"]["metadata"],
        )
        self.assertNotIn(
            "decoder.layer.legacy",
            json.dumps(merged["conditional_performance_evidence"], sort_keys=True),
        )

    def test_v1_rejects_invalid_nested_legacy_experiment_artifact_paths(self):
        for invalid_path in ("/tmp/legacy-sk-meta.json", "../legacy-sk-meta.json"):
            with self.subTest(invalid_path=invalid_path):
                legacy = self._legacy_v1_ledger()
                legacy["experiments"]["S2"]["metadata"] = {
                    "nested": [{"artifact": invalid_path}]
                }

                with self.assertRaisesRegex(ValueError, "relative artifact path"):
                    experiment_ledger.merge_experiment_result(legacy, self._result())

    def test_v1_rejects_invalid_legacy_verified_exclusion_artifact_paths(self):
        for invalid_path in ("/tmp/verified.json", "../verified.json", "   "):
            with self.subTest(invalid_path=invalid_path):
                legacy = self._legacy_v1_ledger()
                legacy["verified_exclusions"][0]["metadata"] = {
                    "nested": [invalid_path]
                }

                with self.assertRaisesRegex(ValueError, "relative artifact path"):
                    experiment_ledger.merge_experiment_result(legacy, self._result())

    def test_migrates_intermediate_v1_performance_experiment_as_active(self):
        first = self._result()
        intermediate = {
            "schema_version": 1,
            "experiments": {"S3": first},
            "conditional_performance_evidence": {},
        }
        second = self._result()
        second["experiment_id"] = "S4"
        second["parent_experiment_id"] = "S4"
        for round_data in second["rounds"]:
            round_data["candidate_config_fingerprint"] = (
                "S4-" + round_data["candidate_config_fingerprint"]
            )
        for collection in (
            "performance_scope_decisions",
            "unresolved_performance_ranges",
        ):
            for decision in second[collection]:
                decision["candidate_config_fingerprint"] = (
                    "S4-" + decision["candidate_config_fingerprint"]
                )
        self._freshen_profiling_evidence(second, "S4")

        merged = experiment_ledger.merge_experiment_result(intermediate, second)

        self.assertEqual(set(merged["experiments"]), {"S3", "S4"})
        self.assertEqual(merged["artifact_validated_experiment_ids"], [])
        self.assertEqual(merged["conditional_performance_evidence"], {})

    def test_legacy_experiment_id_cannot_be_reused_by_active_experiment(self):
        with self.assertRaisesRegex(
            ValueError, "legacy.*cannot be reused|cannot.*legacy"
        ):
            experiment_ledger.merge_experiment_result(
                self._legacy_v1_ledger("S3"), self._result()
            )

    def test_legacy_artifacts_remain_reserved_after_schema_migration(self):
        result = self._result()
        result["rounds"][0]["candidate_profile"] = "S2/round-report.json"
        self._sync_round_binding_field(result, "S3-BASE", "candidate_profile")

        with self.assertRaisesRegex(ValueError, "artifact|role"):
            experiment_ledger.merge_experiment_result(self._legacy_v1_ledger(), result)

    def test_legacy_single_segment_artifact_remains_reserved_after_migration(self):
        legacy = self._legacy_v1_ledger()
        legacy["experiments"]["S2"]["single_child_exclusions"][0]["source_evidence"] = (
            "reserved"
        )
        legacy["verified_exclusions"][0]["source_evidence"] = "reserved"
        result = self._result()
        result["rounds"][0]["candidate_profile"] = "reserved"
        self._sync_round_binding_field(result, "S3-BASE", "candidate_profile")

        with self.assertRaisesRegex(ValueError, "artifact|role"):
            experiment_ledger.merge_experiment_result(legacy, result)

    def test_legacy_source_evidence_remains_reserved_after_schema_migration(self):
        legacy = self._legacy_v1_ledger()
        result = self._result()
        result["rounds"][0]["candidate_profile"] = "S2/source-map.json"
        self._sync_round_binding_field(result, "S3-BASE", "candidate_profile")

        with self.assertRaisesRegex(ValueError, "artifact|role"):
            experiment_ledger.merge_experiment_result(legacy, result)

    def test_legacy_verified_exclusion_artifacts_remain_reserved(self):
        legacy = self._legacy_v1_ledger()
        legacy["experiments"]["S2"]["single_child_exclusions"] = []
        result = self._result()
        result["rounds"][0]["candidate_profile"] = "S2/source-map.json"
        self._sync_round_binding_field(result, "S3-BASE", "candidate_profile")

        with self.assertRaisesRegex(ValueError, "artifact|role"):
            experiment_ledger.merge_experiment_result(legacy, result)

    def test_v2_rejects_tampered_legacy_experiment_identity(self):
        ledger = experiment_ledger.merge_experiment_result(
            self._legacy_v1_ledger(), self._result()
        )
        ledger["legacy_experiments"]["S2"]["experiment_id"] = "other-id"

        with self.assertRaisesRegex(ValueError, "legacy.*identity"):
            experiment_ledger.merge_experiment_result(ledger, self._result())

    def test_idempotent_merge_rejects_tampered_legacy_artifact_roles(self):
        result = self._result()
        ledger = experiment_ledger.merge_experiment_result(
            self._legacy_v1_ledger(), result
        )
        ledger["legacy_experiments"]["S2"]["metadata"] = "reserved"
        ledger["legacy_experiments"]["S2"]["replay"] = "reserved"

        with self.assertRaisesRegex(ValueError, "conflicting roles"):
            experiment_ledger.merge_experiment_result(ledger, result)

    def test_rejects_unknown_ledger_schema_version(self):
        for version in (99, True, None, [], {}):
            with self.subTest(version=version):
                with self.assertRaisesRegex(
                    ValueError, "unsupported ledger schema version"
                ):
                    experiment_ledger.merge_experiment_result(
                        {"schema_version": version}, self._result()
                    )

    def test_rejects_tampered_or_incomplete_conditional_index(self):
        for mutation in ("tampered", "missing"):
            with self.subTest(mutation=mutation):
                ledger = experiment_ledger.merge_experiment_result({}, self._result())
                ledger = self._mark_artifact_validated(ledger, "S3")
                key = next(iter(ledger["conditional_performance_evidence"]))
                if mutation == "tampered":
                    ledger["conditional_performance_evidence"][key]["evidence_kind"] = (
                        "invalid"
                    )
                else:
                    del ledger["conditional_performance_evidence"][key]
                second = self._result()
                second["experiment_id"] = "S4"
                second["parent_experiment_id"] = "S4"
                self._freshen_profiling_evidence(second, "S4")

                with self.assertRaisesRegex(ValueError, "derived|conditional"):
                    experiment_ledger.merge_experiment_result(ledger, second)

    def test_rejects_conflicting_evidence_for_same_conditional_key(self):
        ledger = experiment_ledger.merge_experiment_result({}, self._result())
        second = self._result()
        second["experiment_id"] = "S4"
        second["parent_experiment_id"] = "S4"
        self._freshen_profiling_evidence(second, "S4")

        ledger = experiment_ledger.merge_experiment_result(ledger, second)
        with self.assertRaisesRegex(ValueError, "conflicting conditional"):
            self._mark_artifact_validated(ledger, "S3", "S4")

    def test_round_report_and_evidence_have_exclusive_artifact_roles(self):
        mutations = (
            lambda result: result["rounds"][0]["artifacts"].update(
                {
                    "round_report": result["rounds"][0]["candidate_profile"],
                }
            ),
            lambda result: result["rounds"][1]["artifacts"].update(
                {"round_evidence": result["rounds"][0]["artifacts"]["round_evidence"]}
            ),
            lambda result: result["rounds"][0]["artifacts"].update(
                {"round_report": result["rounds"][0]["artifacts"]["round_evidence"]}
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                result = self._result()
                mutate(result)

                validation = experiment_ledger.validate_experiment_result(result)

                self.assertFalse(validation["valid"])
                self.assertTrue(
                    any(
                        "artifact" in error or "role" in error
                        for error in validation["errors"]
                    ),
                    validation["errors"],
                )

    def test_all_declared_artifact_keys_have_exclusive_roles(self):
        mutations = (
            lambda result: result["rounds"][0]["artifacts"].update(
                {
                    "metadata": result["rounds"][0]["artifacts"]["round_report"],
                }
            ),
            lambda result: result["rounds"][0]["artifacts"].update(
                {"metadata": "S3-BASE/sk-meta.json", "replay": "S3-BASE/sk-meta.json"}
            ),
            lambda result: result["performance_scope_decisions"][0].update(
                {"source_evidence": result["rounds"][0]["candidate_profile"]}
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                result = self._result()
                mutate(result)

                validation = experiment_ledger.validate_experiment_result(result)

                self.assertFalse(validation["valid"])
                self.assertTrue(
                    any(
                        "artifact" in error or "role" in error
                        for error in validation["errors"]
                    ),
                    validation["errors"],
                )

    def test_single_segment_artifact_paths_have_exclusive_roles(self):
        result = self._result()
        result["rounds"][0]["artifacts"].update(
            {"metadata": "reserved", "replay": "reserved"}
        )

        validation = experiment_ledger.validate_experiment_result(result)

        self.assertFalse(validation["valid"])
        self.assertTrue(
            any(
                "artifact" in error or "role" in error for error in validation["errors"]
            ),
            validation["errors"],
        )

    def test_nested_artifact_path_container_rejects_absolute_string_leaf(self):
        result = self._result()
        result["rounds"][0]["artifacts"]["metadata"] = [
            {"compat": ["S3-BASE/compat-sk-meta.json"]},
            {"verify": [{"path_value": "/tmp/absolute-sk-meta.json"}]},
        ]

        validation = experiment_ledger.validate_experiment_result(result)

        self.assertFalse(validation["valid"])
        self.assertTrue(
            any("relative artifact path" in error for error in validation["errors"]),
            validation["errors"],
        )

    def test_nested_artifact_path_container_rejects_blank_string_leaf(self):
        for invalid_path in ("", "   ", " reserved "):
            with self.subTest(invalid_path=invalid_path):
                result = self._result()
                result["rounds"][0]["artifacts"]["metadata"] = {
                    "compat": [{"path_value": invalid_path}]
                }

                validation = experiment_ledger.validate_experiment_result(result)

                self.assertFalse(validation["valid"])
                self.assertTrue(
                    any(
                        "relative artifact path" in error
                        for error in validation["errors"]
                    ),
                    validation["errors"],
                )

    def test_nested_same_role_relative_artifacts_are_registered(self):
        result = self._result()
        result["rounds"][0]["artifacts"]["metadata"] = {
            "compat": [
                "reserved",
                {"copy": "S3-BASE/compat-sk-meta.json", "replay": "reserved"},
            ],
            "verify": {"nested": [None, "S3-BASE/verify-sk-meta.json"]},
        }
        result["metadata"] = {"alias": [{"path_value": "reserved"}]}

        validation = experiment_ledger.validate_experiment_result(result)
        paths_by_role = experiment_ledger._artifact_paths_by_role(result)

        self.assertTrue(validation["valid"], validation["errors"])
        self.assertTrue(
            {
                "reserved",
                "S3-BASE/compat-sk-meta.json",
                "S3-BASE/verify-sk-meta.json",
            }.issubset(paths_by_role["metadata"])
        )

    def test_same_semantic_artifact_aliases_are_allowed(self):
        result = self._result()
        analysis_path = result["profiling_analysis_result"]["S3-BASE"]
        result["performance_scope_decisions"][0]["profiling_analysis_result"] = (
            analysis_path
        )
        result["rounds"][0]["artifacts"]["metadata"] = "S3-BASE/sk-meta.json"
        result["metadata"] = {"compat": "S3-BASE/sk-meta.json"}
        result["notes"] = {
            "path_like_but_not_declared": result["rounds"][0]["candidate_profile"]
        }

        validation = experiment_ledger.validate_experiment_result(result)

        self.assertTrue(validation["valid"], validation["errors"])

    def test_merge_rejects_round_artifact_role_reuse_across_experiments(self):
        first = self._result()
        ledger = experiment_ledger.merge_experiment_result({}, first)
        for direction in ("report_as_profile", "profile_as_report"):
            with self.subTest(direction=direction):
                second = self._result()
                second["experiment_id"] = "S4"
                second["parent_experiment_id"] = "S4"
                self._freshen_profiling_evidence(second, "S4", fresh_baseline=True)
                if direction == "report_as_profile":
                    second["rounds"][0]["candidate_profile"] = first["rounds"][0][
                        "artifacts"
                    ]["round_report"]
                    self._sync_round_binding_field(
                        second, "S3-BASE", "candidate_profile"
                    )
                else:
                    second["rounds"][0]["artifacts"]["round_report"] = first["rounds"][
                        0
                    ]["candidate_profile"]

                with self.assertRaisesRegex(ValueError, "artifact|role"):
                    experiment_ledger.merge_experiment_result(ledger, second)

    def test_merge_rejects_metadata_replay_role_exchange_from_v2_history(self):
        first = self._result()
        first["rounds"][0]["artifacts"]["metadata"] = "S3-BASE/sk-meta.json"
        ledger = experiment_ledger.merge_experiment_result({}, first)
        second = self._result()
        second["experiment_id"] = "S4"
        second["parent_experiment_id"] = "S4"
        self._freshen_profiling_evidence(second, "S4", fresh_baseline=True)
        second["rounds"][0]["artifacts"]["replay"] = "S3-BASE/sk-meta.json"

        with self.assertRaisesRegex(ValueError, "artifact|role"):
            experiment_ledger.merge_experiment_result(ledger, second)

    def test_conditional_evidence_key_uses_all_declared_fingerprints(self):
        result = self._result()

        ledger = experiment_ledger.merge_experiment_result({}, result)
        ledger = self._mark_artifact_validated(ledger, "S3")

        for key, evidence in ledger["conditional_performance_evidence"].items():
            self.assertEqual(
                json.loads(key),
                [
                    evidence["source_revision"],
                    evidence["baseline_config_fingerprint"],
                    evidence["candidate_config_fingerprint"],
                    evidence["control_fingerprint"],
                    evidence["workload_fingerprint"],
                    evidence["range_id"],
                ],
            )

    def test_automatic_aot_does_not_require_manual_prune_fields(self):
        result = self._automatic_result()
        del result["rounds"][0]["performance_decision_range_ids"]

        validation = experiment_ledger.validate_experiment_result(result)

        self.assertTrue(validation["valid"])

    def test_legacy_single_child_fields_are_optional_and_never_promoted(self):
        result = self._result()
        result["single_child_exclusions"] = [
            {
                "round_id": "S3-BASE",
                "source_scope": "decoder.layer.legacy",
                "exclusion_status": "verified",
            }
        ]
        result["rounds"][0]["single_child_group_count"] = 9

        ledger = experiment_ledger.merge_experiment_result({}, result)
        ledger = self._mark_artifact_validated(ledger, "S3")

        self.assertNotIn("verified_exclusions", ledger)
        self.assertEqual(len(ledger["conditional_performance_evidence"]), 1)
        self.assertNotIn(
            "decoder.layer.legacy",
            json.dumps(ledger["conditional_performance_evidence"], sort_keys=True),
        )

    def test_conditional_evidence_deduplicates_same_composite_key(self):
        result = self._result()
        ledger = experiment_ledger.merge_experiment_result({}, result)
        ledger = self._mark_artifact_validated(ledger, "S3")
        original = json.loads(json.dumps(ledger))

        ledger = experiment_ledger.merge_experiment_result(ledger, result)

        self.assertEqual(ledger, original)
        self.assertEqual(len(ledger["conditional_performance_evidence"]), 1)

    def test_changed_fingerprint_does_not_overwrite_conditional_evidence(self):
        first = self._result()
        ledger = experiment_ledger.merge_experiment_result({}, first)
        second = self._result()
        second["experiment_id"] = "S4"
        second["parent_experiment_id"] = "S4"
        second["workload_fingerprint"] = "workload-v2-fp"
        for round_data in second["rounds"]:
            round_data["workload_fingerprint"] = "workload-v2-fp"
        for decision in (
            second["performance_scope_decisions"]
            + second["unresolved_performance_ranges"]
        ):
            decision["workload_fingerprint"] = "workload-v2-fp"
        self._freshen_profiling_evidence(second, "S4", fresh_baseline=True)

        ledger = experiment_ledger.merge_experiment_result(ledger, second)
        ledger = self._mark_artifact_validated(ledger, "S3", "S4")

        evidence = ledger["conditional_performance_evidence"]
        self.assertEqual(len(ledger["experiments"]), 2)
        self.assertEqual(len(evidence), 2)
        self.assertEqual(
            {item["workload_fingerprint"] for item in evidence.values()},
            {"workload-fp", "workload-v2-fp"},
        )

    def test_rejects_duplicate_candidate_profile_artifacts_within_result(self):
        for field in ("candidate_profile", "candidate_profile_fingerprint"):
            with self.subTest(field=field):
                result = self._result()
                result["rounds"][1][field] = result["rounds"][0][field]

                validation = experiment_ledger.validate_experiment_result(result)

                self.assertFalse(validation["valid"])
                self.assertTrue(any(field in error for error in validation["errors"]))

    def test_baseline_profile_pair_is_immutable_within_result(self):
        for field, replacement in (
            ("baseline_profile_fingerprint", "different-baseline-fp"),
            ("baseline_profile", "profiles/copied-baseline.json"),
        ):
            with self.subTest(field=field):
                result = self._result()
                result["rounds"][1][field] = replacement

                validation = experiment_ledger.validate_experiment_result(result)

                self.assertFalse(validation["valid"])
                self.assertTrue(
                    any("baseline profile" in error for error in validation["errors"]),
                    validation["errors"],
                )

    def test_identical_baseline_profile_pair_may_repeat_within_result(self):
        validation = experiment_ledger.validate_experiment_result(self._result())

        self.assertTrue(validation["valid"], validation["errors"])

    def test_rejects_candidate_profile_artifact_used_as_baseline_in_same_result(self):
        for suffix in ("profile", "profile_fingerprint"):
            with self.subTest(suffix=suffix):
                result = self._result()
                candidate_field = f"candidate_{suffix}"
                baseline_field = f"baseline_{suffix}"
                result["rounds"][1][baseline_field] = result["rounds"][0][
                    candidate_field
                ]

                validation = experiment_ledger.validate_experiment_result(result)

                self.assertFalse(validation["valid"])
                self.assertTrue(
                    any(
                        "baseline" in error and "candidate" in error
                        for error in validation["errors"]
                    ),
                    validation["errors"],
                )

    def test_rejects_same_round_candidate_profile_as_baseline(self):
        for suffix in ("profile", "profile_fingerprint"):
            with self.subTest(suffix=suffix):
                result = self._result()
                candidate_field = f"candidate_{suffix}"
                baseline_field = f"baseline_{suffix}"
                result["rounds"][0][candidate_field] = result["rounds"][0][
                    baseline_field
                ]
                self._sync_round_binding_field(result, "S3-BASE", candidate_field)

                validation = experiment_ledger.validate_experiment_result(result)

                self.assertFalse(validation["valid"])
                self.assertTrue(
                    any(
                        "baseline" in error and "candidate" in error
                        for error in validation["errors"]
                    ),
                    validation["errors"],
                )

    def test_decision_profile_fingerprints_must_match_source_round(self):
        for mutation in (
            lambda decision: decision.pop("baseline_profile_fingerprint"),
            lambda decision: decision.update(
                {"candidate_profile_fingerprint": "other-profile-fp"}
            ),
        ):
            with self.subTest(mutation=mutation):
                result = self._result()
                mutation(result["performance_scope_decisions"][0])

                validation = experiment_ledger.validate_experiment_result(result)

                self.assertFalse(validation["valid"])
                self.assertTrue(
                    any(
                        "profile_fingerprint" in error for error in validation["errors"]
                    ),
                    validation["errors"],
                )

    def test_merge_rejects_candidate_profile_reuse_with_fresh_analysis(self):
        ledger = experiment_ledger.merge_experiment_result({}, self._result())
        for field in ("candidate_profile", "candidate_profile_fingerprint"):
            with self.subTest(field=field):
                second = self._result()
                second["experiment_id"] = "S4"
                second["parent_experiment_id"] = "S4"
                original_values = {
                    item["round_id"]: item[field] for item in second["rounds"]
                }
                self._freshen_profiling_evidence(second, "S4")
                rounds_by_id = {item["round_id"]: item for item in second["rounds"]}
                for round_id, value in original_values.items():
                    rounds_by_id[round_id][field] = value
                for collection in (
                    "performance_scope_decisions",
                    "unresolved_performance_ranges",
                ):
                    for decision in second[collection]:
                        decision[field] = rounds_by_id[decision["round_id"]][field]

                with self.assertRaisesRegex(ValueError, "candidate profile"):
                    experiment_ledger.merge_experiment_result(ledger, second)

    def test_changed_workload_cannot_reuse_profiles_with_fresh_analysis(self):
        ledger = experiment_ledger.merge_experiment_result({}, self._result())
        second = self._result()
        second["experiment_id"] = "S4"
        second["parent_experiment_id"] = "S4"
        second["workload_fingerprint"] = "workload-v2-fp"
        for round_data in second["rounds"]:
            round_data["workload_fingerprint"] = "workload-v2-fp"
        for collection in (
            "performance_scope_decisions",
            "unresolved_performance_ranges",
        ):
            for decision in second[collection]:
                decision["workload_fingerprint"] = "workload-v2-fp"
        reused_profiles = {
            round_data["round_id"]: {
                field: round_data[field]
                for field in (
                    "baseline_profile",
                    "baseline_profile_fingerprint",
                    "candidate_profile",
                    "candidate_profile_fingerprint",
                )
            }
            for round_data in second["rounds"]
        }
        self._freshen_profiling_evidence(second, "S4")
        for round_data in second["rounds"]:
            round_data.update(reused_profiles[round_data["round_id"]])
        for collection in (
            "performance_scope_decisions",
            "unresolved_performance_ranges",
        ):
            for decision in second[collection]:
                source_round = next(
                    item
                    for item in second["rounds"]
                    if item["round_id"] == decision["round_id"]
                )
                for field in reused_profiles[source_round["round_id"]]:
                    decision[field] = source_round[field]

        with self.assertRaisesRegex(ValueError, "profile"):
            experiment_ledger.merge_experiment_result(ledger, second)

    def test_frozen_baseline_profiles_may_be_reused_across_experiments(self):
        ledger = experiment_ledger.merge_experiment_result({}, self._result())
        second = self._result()
        second["experiment_id"] = "S4"
        second["parent_experiment_id"] = "S4"
        for round_data in second["rounds"]:
            round_data["candidate_config_fingerprint"] = (
                "S4-" + round_data["candidate_config_fingerprint"]
            )
        for collection in (
            "performance_scope_decisions",
            "unresolved_performance_ranges",
        ):
            for decision in second[collection]:
                decision["candidate_config_fingerprint"] = (
                    "S4-" + decision["candidate_config_fingerprint"]
                )
        self._freshen_profiling_evidence(second, "S4")

        merged = experiment_ledger.merge_experiment_result(ledger, second)

        self.assertEqual(set(merged["experiments"]), {"S3", "S4"})

    def test_merge_requires_exact_historical_baseline_profile_pair(self):
        ledger = experiment_ledger.merge_experiment_result({}, self._result())
        for field, replacement in (
            ("baseline_profile_fingerprint", "different-baseline-fp"),
            ("baseline_profile", "profiles/copied-baseline.json"),
        ):
            with self.subTest(field=field):
                second = self._result()
                second["experiment_id"] = "S4"
                second["parent_experiment_id"] = "S4"
                self._freshen_profiling_evidence(second, "S4")
                for round_data in second["rounds"]:
                    round_data[field] = replacement
                    self._sync_round_binding_field(
                        second, round_data["round_id"], field
                    )

                with self.assertRaisesRegex(ValueError, "baseline profile"):
                    experiment_ledger.merge_experiment_result(ledger, second)

    def test_terminal_failed_round_baseline_pair_remains_registered(self):
        first = self._result()
        first["rounds"] = first["rounds"][:2]
        terminal = first["rounds"][-1]
        terminal["lifecycle"]["replay"] = "failed"
        terminal["lifecycle"]["profiling"] = "not_run"
        terminal["baseline_profile"] = "profiles/terminal-failed-baseline.json"
        terminal["baseline_profile_fingerprint"] = "terminal-failed-baseline-fp"
        first["profiling_analysis_agent_ids"] = {
            "S3-BASE": first["profiling_analysis_agent_ids"]["S3-BASE"]
        }
        first["profiling_analysis_result"] = {
            "S3-BASE": first["profiling_analysis_result"]["S3-BASE"]
        }
        first["performance_scope_decisions"] = [
            self._decision(
                round_id="S3-BASE",
                range_id="range-neutral",
                classification="neutral",
                action="prune",
                status="applied",
                candidate_fingerprint="candidate-base-fp",
            )
        ]
        first["unresolved_performance_ranges"] = []
        first["rounds"][0]["performance_decision_range_ids"] = ["range-neutral"]
        first["rounds"][1]["performance_decision_range_ids"] = ["range-neutral"]
        first["blockers"] = ["S3-P1 replay failed"]
        ledger = experiment_ledger.merge_experiment_result({}, first)

        second = self._result()
        second["experiment_id"] = "S4"
        second["parent_experiment_id"] = "S4"
        self._freshen_profiling_evidence(second, "S4", fresh_baseline=True)
        second["rounds"][0]["baseline_profile"] = terminal["baseline_profile"]
        second["rounds"][0]["baseline_profile_fingerprint"] = "reinterpreted-fp"
        self._sync_round_binding_field(second, "S3-BASE", "baseline_profile")
        self._sync_round_binding_field(
            second, "S3-BASE", "baseline_profile_fingerprint"
        )

        with self.assertRaisesRegex(ValueError, "baseline profile"):
            experiment_ledger.merge_experiment_result(ledger, second)

    def test_merge_rejects_historical_candidate_reused_as_baseline(self):
        first = self._result()
        ledger = experiment_ledger.merge_experiment_result({}, first)
        for suffix in ("profile", "profile_fingerprint"):
            with self.subTest(suffix=suffix):
                second = self._result()
                second["experiment_id"] = "S4"
                second["parent_experiment_id"] = "S4"
                self._freshen_profiling_evidence(second, "S4", fresh_baseline=True)
                baseline_field = f"baseline_{suffix}"
                candidate_field = f"candidate_{suffix}"
                second["rounds"][0][baseline_field] = first["rounds"][0][
                    candidate_field
                ]
                self._sync_round_binding_field(second, "S3-BASE", baseline_field)

                with self.assertRaisesRegex(ValueError, "baseline.*candidate"):
                    experiment_ledger.merge_experiment_result(ledger, second)

    def test_merge_rejects_historical_baseline_reused_as_candidate(self):
        first = self._result()
        ledger = experiment_ledger.merge_experiment_result({}, first)
        for suffix in ("profile", "profile_fingerprint"):
            with self.subTest(suffix=suffix):
                second = self._result()
                second["experiment_id"] = "S4"
                second["parent_experiment_id"] = "S4"
                self._freshen_profiling_evidence(second, "S4", fresh_baseline=True)
                baseline_field = f"baseline_{suffix}"
                candidate_field = f"candidate_{suffix}"
                second["rounds"][0][candidate_field] = first["rounds"][0][
                    baseline_field
                ]
                self._sync_round_binding_field(second, "S3-BASE", candidate_field)

                with self.assertRaisesRegex(ValueError, "candidate.*baseline"):
                    experiment_ledger.merge_experiment_result(ledger, second)

    def test_merge_rejects_changed_content_for_existing_experiment_id(self):
        first = self._result()
        ledger = experiment_ledger.merge_experiment_result({}, first)
        changed = self._result()
        changed["workload_fingerprint"] = "changed-workload-fp"
        for round_data in changed["rounds"]:
            round_data["workload_fingerprint"] = "changed-workload-fp"
        for collection in (
            "performance_scope_decisions",
            "unresolved_performance_ranges",
        ):
            for decision in changed[collection]:
                decision["workload_fingerprint"] = "changed-workload-fp"

        with self.assertRaisesRegex(ValueError, "immutable"):
            experiment_ledger.merge_experiment_result(ledger, changed)

    def test_merge_rejects_cross_experiment_analysis_evidence_reuse(self):
        ledger = experiment_ledger.merge_experiment_result({}, self._result())
        for reused_field in (
            "profiling_analysis_agent_ids",
            "profiling_analysis_result",
        ):
            with self.subTest(reused_field=reused_field):
                second = self._result()
                second["experiment_id"] = "S4"
                second["parent_experiment_id"] = "S4"
                other_field = (
                    "profiling_analysis_result"
                    if reused_field == "profiling_analysis_agent_ids"
                    else "profiling_analysis_agent_ids"
                )
                second[other_field] = {
                    round_id: f"fresh-{value}"
                    for round_id, value in second[other_field].items()
                }
                if other_field == "profiling_analysis_result":
                    for collection in (
                        "performance_scope_decisions",
                        "unresolved_performance_ranges",
                    ):
                        for decision in second[collection]:
                            decision["source_profiling_analysis_result"] = second[
                                other_field
                            ][decision["round_id"]]
                            if "verified_in_round_id" in decision:
                                decision["profiling_analysis_result"] = second[
                                    other_field
                                ][decision["verified_in_round_id"]]

                with self.assertRaisesRegex(ValueError, "fresh"):
                    experiment_ledger.merge_experiment_result(ledger, second)

    def test_profiling_passed_requires_correctness(self):
        result = self._result()
        result["rounds"][1]["lifecycle"]["correctness"] = "failed"

        validation = experiment_ledger.validate_experiment_result(result)

        self.assertFalse(validation["valid"])
        self.assertTrue(any("correctness" in error for error in validation["errors"]))

    def test_analysis_handoff_accepts_source_scope_from_scope_actions(self):
        producer = {
            "sk_id": "sk-1",
            "range_id": "range-1",
            "classification": "beneficial",
            "action": "keep",
            "mapping_method": "kernel_projection_structural",
            "mapping_confidence": "exact_projected_trace",
            "boundary": {"start_op": "A", "end_op": "A"},
            "candidate_binding_status": "bound",
            "graph_occurrence_fingerprint": "a" * 64,
            "mapping_blockers": [],
            "identity": {
                "source_scope": "none",
                "ordered_child_op_sequence": ["A"],
            },
        }
        action = {
            "sk_id": "sk-1",
            "range_id": "range-1",
            "classification": "beneficial",
            "action": "keep",
            "source_scope": "none",
            "boundary": {"start_op": "A", "end_op": "A"},
        }
        submitted = {
            **producer,
            "round_id": "S1-BASE",
            "source_scope": action["source_scope"],
        }
        submitted.pop("identity")
        result = {
            "performance_scope_decisions": [submitted],
            "unresolved_performance_ranges": [],
        }

        experiment_ledger._validate_analysis_decisions(
            result,
            {
                "S1-BASE": {
                    "per_sk_decisions": [producer],
                    "scope_actions": [action],
                }
            },
        )

    def test_optional_compat_and_replay_audits_do_not_gate_profiling(self):
        for audit_gate in ("compat", "replay"):
            with self.subTest(audit_gate=audit_gate):
                result = self._result()
                result["rounds"][0]["lifecycle"][audit_gate] = "failed"

                validation = experiment_ledger.validate_experiment_result(result)

                self.assertTrue(validation["valid"], validation["errors"])

    def test_manual_round_cannot_continue_after_incomplete_round(self):
        for round_index in (0, 1):
            with self.subTest(
                round_id=self._result()["rounds"][round_index]["round_id"]
            ):
                result = self._result()
                round_id = result["rounds"][round_index]["round_id"]
                result["rounds"][round_index]["lifecycle"]["profiling"] = "not_run"
                result["profiling_analysis_agent_ids"].pop(round_id)
                result["profiling_analysis_result"].pop(round_id)

                validation = experiment_ledger.validate_experiment_result(result)

                self.assertFalse(validation["valid"])
                self.assertTrue(
                    any(
                        "cannot be followed" in error for error in validation["errors"]
                    ),
                    validation["errors"],
                )

    def test_terminal_failed_manual_round_may_retain_blocker(self):
        result = self._result()
        result["rounds"] = result["rounds"][:2]
        terminal = result["rounds"][-1]
        terminal["lifecycle"]["replay"] = "failed"
        terminal["lifecycle"]["profiling"] = "not_run"
        result["profiling_analysis_agent_ids"] = {
            "S3-BASE": result["profiling_analysis_agent_ids"]["S3-BASE"]
        }
        result["profiling_analysis_result"] = {
            "S3-BASE": result["profiling_analysis_result"]["S3-BASE"]
        }
        result["performance_scope_decisions"] = [
            self._decision(
                round_id="S3-BASE",
                range_id="range-neutral",
                classification="neutral",
                action="prune",
                status="applied",
                candidate_fingerprint="candidate-base-fp",
            )
        ]
        result["unresolved_performance_ranges"] = []
        result["rounds"][0]["performance_decision_range_ids"] = ["range-neutral"]
        result["rounds"][1]["performance_decision_range_ids"] = ["range-neutral"]
        result["blockers"] = ["S3-P1 replay failed"]

        validation = experiment_ledger.validate_experiment_result(result)

        self.assertTrue(validation["valid"], validation["errors"])
        ledger = experiment_ledger.merge_experiment_result({}, result)
        self.assertEqual(ledger["conditional_performance_evidence"], {})

    def test_terminal_failed_manual_round_requires_an_explicit_blocker(self):
        result = self._result()
        result["rounds"] = result["rounds"][:2]
        terminal = result["rounds"][-1]
        terminal["lifecycle"]["replay"] = "failed"
        terminal["lifecycle"]["profiling"] = "not_run"
        result["profiling_analysis_agent_ids"] = {
            "S3-BASE": result["profiling_analysis_agent_ids"]["S3-BASE"]
        }
        result["profiling_analysis_result"] = {
            "S3-BASE": result["profiling_analysis_result"]["S3-BASE"]
        }
        result["performance_scope_decisions"] = [
            self._decision(
                round_id="S3-BASE",
                range_id="range-neutral",
                classification="neutral",
                action="prune",
                status="applied",
                candidate_fingerprint="candidate-base-fp",
            )
        ]
        result["rounds"][0]["performance_decision_range_ids"] = ["range-neutral"]
        result["rounds"][1]["performance_decision_range_ids"] = ["range-neutral"]
        result["unresolved_performance_ranges"] = []
        result["blockers"] = [" "]

        validation = experiment_ledger.validate_experiment_result(result)

        self.assertFalse(validation["valid"])
        self.assertTrue(
            any(
                "explicit non-empty blockers" in error for error in validation["errors"]
            ),
            validation["errors"],
        )

    def test_successful_manual_experiment_requires_fresh_final_round(self):
        result = self._result()
        result["rounds"] = result["rounds"][:2]
        result["profiling_analysis_agent_ids"] = {
            round_id: value
            for round_id, value in result["profiling_analysis_agent_ids"].items()
            if round_id in {"S3-BASE", "S3-P1"}
        }
        result["profiling_analysis_result"] = {
            round_id: value
            for round_id, value in result["profiling_analysis_result"].items()
            if round_id in {"S3-BASE", "S3-P1"}
        }
        result["performance_scope_decisions"] = result["performance_scope_decisions"][
            :2
        ]

        validation = experiment_ledger.validate_experiment_result(result)

        self.assertFalse(validation["valid"])
        self.assertTrue(
            any(
                "successful manual experiment requires" in error
                for error in validation["errors"]
            ),
            validation["errors"],
        )

    def test_whole_scope_promotion_does_not_require_source_range_final(self):
        result = self._result()
        result["promotion_mode"] = "whole_scope"
        result["rounds"] = result["rounds"][:1]
        result["rounds"][0]["performance_decision_range_ids"] = [
            "range-beneficial",
            "range-neutral",
        ]
        result["profiling_analysis_agent_ids"] = {
            "S3-BASE": result["profiling_analysis_agent_ids"]["S3-BASE"]
        }
        result["profiling_analysis_result"] = {
            "S3-BASE": result["profiling_analysis_result"]["S3-BASE"]
        }
        beneficial = self._decision(
            round_id="S3-BASE",
            range_id="range-beneficial",
            classification="beneficial",
            action="keep",
            status="proposed",
            candidate_fingerprint="candidate-base-fp",
        )
        neutral = self._decision(
            round_id="S3-BASE",
            range_id="range-neutral",
            classification="neutral",
            action="block",
            status="blocked",
            candidate_fingerprint="candidate-base-fp",
        )
        neutral.update(
            {
                "mapping_method": "kernel_projection_structural",
                "mapping_confidence": "exact_projected_trace",
                "candidate_binding_status": "bound",
                "graph_occurrence_fingerprint": "a" * 64,
                "mapping_blockers": [],
                "boundary": {"start_op": "A", "end_op": "B"},
            }
        )
        result["performance_scope_decisions"] = [beneficial]
        result["unresolved_performance_ranges"] = [neutral]
        result["blockers"] = []

        validation = experiment_ledger.validate_experiment_result(result)

        self.assertTrue(validation["valid"], validation["errors"])

    def test_whole_scope_promotion_rejects_insufficient_evidence(self):
        result = self._automatic_result()
        result["promotion_mode"] = "whole_scope"
        result["rounds"][0]["performance_decision_range_ids"] = ["range-auto"]
        result["unresolved_performance_ranges"] = [
            self._decision(
                round_id="S1",
                range_id="range-auto",
                classification="insufficient_evidence",
                action="block",
                status="blocked",
                candidate_fingerprint="candidate-auto-fp",
            )
        ]

        validation = experiment_ledger.validate_experiment_result(result)

        self.assertFalse(validation["valid"])
        self.assertTrue(
            any(
                "whole_scope" in error and "insufficient_evidence" in error
                for error in validation["errors"]
            ),
            validation["errors"],
        )

    def test_analysis_result_path_cannot_overlap_profile_path_within_result(self):
        result = self._result()
        result["profiling_analysis_result"]["S3-FINAL"] = result["rounds"][2][
            "candidate_profile"
        ]
        for decision in result["performance_scope_decisions"]:
            if decision.get("verified_in_round_id") == "S3-FINAL":
                decision["profiling_analysis_result"] = result[
                    "profiling_analysis_result"
                ]["S3-FINAL"]

        validation = experiment_ledger.validate_experiment_result(result)

        self.assertFalse(validation["valid"])
        self.assertTrue(
            any(
                "analysis" in error and "profile" in error
                for error in validation["errors"]
            ),
            validation["errors"],
        )

    def test_merge_rejects_analysis_and_profile_role_exchange(self):
        first = self._result()
        ledger = experiment_ledger.merge_experiment_result({}, first)
        for direction in ("analysis_as_profile", "profile_as_analysis"):
            with self.subTest(direction=direction):
                second = self._result()
                second["experiment_id"] = "S4"
                second["parent_experiment_id"] = "S4"
                self._freshen_profiling_evidence(second, "S4", fresh_baseline=True)
                if direction == "analysis_as_profile":
                    second["rounds"][0]["candidate_profile"] = first[
                        "profiling_analysis_result"
                    ]["S3-BASE"]
                    self._sync_round_binding_field(
                        second, "S3-BASE", "candidate_profile"
                    )
                else:
                    second["profiling_analysis_result"]["S3-BASE"] = first["rounds"][0][
                        "candidate_profile"
                    ]
                    for collection in (
                        "performance_scope_decisions",
                        "unresolved_performance_ranges",
                    ):
                        for decision in second[collection]:
                            if decision["round_id"] == "S3-BASE":
                                decision["source_profiling_analysis_result"] = second[
                                    "profiling_analysis_result"
                                ]["S3-BASE"]

                with self.assertRaisesRegex(
                    ValueError, "analysis.*profile|profile.*analysis"
                ):
                    experiment_ledger.merge_experiment_result(ledger, second)

    def test_decision_source_requires_passed_profiling_and_analysis_evidence(self):
        for issue in ("profiling", "agent", "path"):
            with self.subTest(issue=issue):
                result = self._result()
                if issue == "profiling":
                    result["rounds"][0]["lifecycle"]["profiling"] = "not_run"
                    del result["profiling_analysis_agent_ids"]["S3-BASE"]
                    del result["profiling_analysis_result"]["S3-BASE"]
                elif issue == "agent":
                    del result["profiling_analysis_agent_ids"]["S3-BASE"]
                else:
                    del result["profiling_analysis_result"]["S3-BASE"]

                validation = experiment_ledger.validate_experiment_result(result)

                self.assertFalse(validation["valid"])
                self.assertTrue(
                    any(
                        "source" in error or "missing" in error
                        for error in validation["errors"]
                    ),
                    validation["errors"],
                )

    def test_verified_prune_cannot_self_verify_in_source_round(self):
        result = self._result()
        decision = result["performance_scope_decisions"][1]
        decision["verified_in_round_id"] = "S3-BASE"
        decision["profiling_analysis_result"] = result["profiling_analysis_result"][
            "S3-BASE"
        ]

        validation = experiment_ledger.validate_experiment_result(result)

        self.assertFalse(validation["valid"])
        self.assertTrue(
            any("strictly later" in error for error in validation["errors"])
        )

    def test_verified_prune_requires_a_later_performance_prune_round(self):
        for verification_round_id in ("S3-FINAL",):
            with self.subTest(verification_round_id=verification_round_id):
                result = self._result()
                decision = result["performance_scope_decisions"][1]
                decision["verified_in_round_id"] = verification_round_id
                decision["profiling_analysis_result"] = result[
                    "profiling_analysis_result"
                ][verification_round_id]

                validation = experiment_ledger.validate_experiment_result(result)

                self.assertFalse(validation["valid"])
                self.assertTrue(
                    any("performance_prune" in error for error in validation["errors"]),
                    validation["errors"],
                )

    def test_verified_prune_must_be_referenced_by_its_verification_round(self):
        result = self._result()
        result["rounds"][1]["performance_decision_range_ids"] = ["range-beneficial"]

        validation = experiment_ledger.validate_experiment_result(result)

        self.assertFalse(validation["valid"])
        self.assertTrue(
            any("verified prune range" in error for error in validation["errors"]),
            validation["errors"],
        )

    def test_final_requires_fresh_applied_keep_decision_for_every_final_range(self):
        for missing_range in ("range-beneficial",):
            with self.subTest(missing_range=missing_range):
                result = self._result()
                result["performance_scope_decisions"] = [
                    decision
                    for decision in result["performance_scope_decisions"]
                    if not (
                        decision["round_id"] == "S3-FINAL"
                        and decision["range_id"] == missing_range
                    )
                ]

                validation = experiment_ledger.validate_experiment_result(result)

                self.assertFalse(validation["valid"])
                self.assertTrue(
                    any(
                        "fresh applied keep decision" in error
                        for error in validation["errors"]
                    ),
                    validation["errors"],
                )

    def test_final_rejects_regressed_or_blocked_outcome_for_a_final_range(self):
        for outcome in ("regressed", "blocked"):
            with self.subTest(outcome=outcome):
                result = self._result()
                result["performance_scope_decisions"] = [
                    decision
                    for decision in result["performance_scope_decisions"]
                    if not (
                        decision["round_id"] == "S3-FINAL"
                        and decision["range_id"] == "range-beneficial"
                    )
                ]
                if outcome == "regressed":
                    result["performance_scope_decisions"].append(
                        self._decision(
                            round_id="S3-FINAL",
                            range_id="range-beneficial",
                            classification="regressed",
                            action="prune",
                            status="applied",
                            candidate_fingerprint="candidate-final-fp",
                        )
                    )
                else:
                    result["unresolved_performance_ranges"].append(
                        self._decision(
                            round_id="S3-FINAL",
                            range_id="range-beneficial",
                            classification="regressed",
                            action="block",
                            status="blocked",
                            candidate_fingerprint="candidate-final-fp",
                        )
                    )

                validation = experiment_ledger.validate_experiment_result(result)

                self.assertFalse(validation["valid"])
                self.assertTrue(
                    any("final range" in error for error in validation["errors"]),
                    validation["errors"],
                )

    def test_final_applied_keep_decisions_are_not_promoted(self):
        result = self._result()

        ledger = experiment_ledger.merge_experiment_result({}, result)
        ledger = self._mark_artifact_validated(ledger, "S3")

        evidence = ledger["conditional_performance_evidence"].values()
        self.assertEqual(len(list(evidence)), 1)
        self.assertFalse(any(item["round_id"] == "S3-FINAL" for item in evidence))
        self.assertEqual(
            {
                (item["evidence_kind"], item["candidate_config_fingerprint"])
                for item in evidence
            },
            {
                ("prune", "candidate-base-fp"),
            },
        )

    def test_automatic_aot_proposed_classifications_are_not_promoted(self):
        result = self._automatic_result()
        result["rounds"][0]["performance_decision_range_ids"] = [
            "range-beneficial",
            "range-neutral",
            "range-regressed",
        ]
        result["performance_scope_decisions"] = [
            self._decision(
                round_id="S1",
                range_id="range-beneficial",
                classification="beneficial",
                action="keep",
                status="proposed",
                candidate_fingerprint="candidate-auto-fp",
            ),
            self._decision(
                round_id="S1",
                range_id="range-neutral",
                classification="neutral",
                action="prune",
                status="proposed",
                candidate_fingerprint="candidate-auto-fp",
            ),
            self._decision(
                round_id="S1",
                range_id="range-regressed",
                classification="regressed",
                action="prune",
                status="proposed",
                candidate_fingerprint="candidate-auto-fp",
            ),
        ]

        ledger = experiment_ledger.merge_experiment_result({}, result)

        self.assertEqual(ledger["conditional_performance_evidence"], {})
        self.assertEqual(
            len(ledger["experiments"]["S1"]["performance_scope_decisions"]), 3
        )

    def test_automatic_aot_retains_negative_unresolved_without_promotion(self):
        for classification, action, status in (
            ("insufficient_evidence", "reprofile", "proposed"),
            ("regressed", "block", "blocked"),
        ):
            with self.subTest(tuple=(classification, action, status)):
                result = self._automatic_result()
                result["unresolved_performance_ranges"] = [
                    self._decision(
                        round_id="S1",
                        range_id="range-auto",
                        classification=classification,
                        action=action,
                        status=status,
                        candidate_fingerprint="candidate-auto-fp",
                    )
                ]

                validation = experiment_ledger.validate_experiment_result(result)
                ledger = experiment_ledger.merge_experiment_result({}, result)

                self.assertTrue(validation["valid"], validation["errors"])
                self.assertEqual(ledger["conditional_performance_evidence"], {})
                self.assertEqual(
                    len(ledger["experiments"]["S1"]["unresolved_performance_ranges"]),
                    1,
                )

    def test_automatic_aot_optimization_descendant_preserves_scope_kind(self):
        serialized = json.dumps(self._result())
        for old, new in (
            ("S3-BASE", "S1-AUTO"),
            ("S3-P1", "S1-P1"),
            ("S3-FINAL", "S1-FINAL"),
            ("S3", "S1"),
        ):
            serialized = serialized.replace(old, new)
        result = json.loads(serialized)
        result["rounds"][0]["round_kind"] = "automatic_aot"
        for round_data in result["rounds"]:
            round_data["scope_kind"] = "automatic_aot"

        validation = experiment_ledger.validate_experiment_result(result)

        self.assertTrue(validation["valid"], validation["errors"])
        promoted = experiment_ledger._promoted_conditional_records(result)
        self.assertTrue(
            any(
                evidence_kind == "prune" and decision["round_id"] == "S1-AUTO"
                for evidence_kind, decision in promoted
            )
        )

        result["rounds"][1]["scope_kind"] = "manual"
        validation = experiment_ledger.validate_experiment_result(result)
        self.assertFalse(validation["valid"])
        self.assertTrue(
            any(
                "preserve the starter scope_kind" in error
                for error in validation["errors"]
            ),
            validation["errors"],
        )

    def test_rejects_identifiers_with_surrounding_whitespace(self):
        mutations = (
            lambda result: result.update(
                {"experiment_id": " S3 ", "parent_experiment_id": " S3 "}
            ),
            lambda result: (
                result.update({"child_agent_id": " agent-7 "}),
                [
                    item.update({"child_agent_id": " agent-7 "})
                    for item in result["rounds"]
                ],
            ),
            lambda result: result["profiling_analysis_agent_ids"].update(
                {"S3-BASE": " prof-agent-1 "}
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                result = self._result()
                mutate(result)

                validation = experiment_ledger.validate_experiment_result(result)

                self.assertFalse(validation["valid"])
                self.assertTrue(
                    any("whitespace" in error for error in validation["errors"]),
                    validation["errors"],
                )

    def test_rejects_agent_id_whitespace_variant_within_and_across_results(self):
        result = self._result()
        result["profiling_analysis_agent_ids"]["S3-P1"] = " prof-agent-1 "

        validation = experiment_ledger.validate_experiment_result(result)

        self.assertFalse(validation["valid"])

        ledger = experiment_ledger.merge_experiment_result({}, self._result())
        second = self._result()
        second["experiment_id"] = "S4"
        second["parent_experiment_id"] = "S4"
        self._freshen_profiling_evidence(second, "S4")
        second["profiling_analysis_agent_ids"]["S3-BASE"] = " prof-agent-1 "
        with self.assertRaisesRegex(ValueError, "whitespace|fresh"):
            experiment_ledger.merge_experiment_result(ledger, second)

    def test_round_requires_complete_profiling_input_binding(self):
        mutations = (
            lambda round_data: round_data.pop("baseline_profile"),
            lambda round_data: round_data.pop("baseline_profile_fingerprint"),
            lambda round_data: round_data.pop("candidate_profile_fingerprint"),
            lambda round_data: round_data.update(
                {"candidate_profile": "/tmp/profile.json"}
            ),
            lambda round_data: round_data.update(
                {"baseline_profile": "../profile.json"}
            ),
            lambda round_data: round_data.update({"declared_change_set": {}}),
            lambda round_data: round_data.update(
                {
                    "declared_change_set": {
                        "allowed_json_pointers": [],
                        "only_change_zh": "english only",
                    }
                }
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                result = self._result()
                mutate(result["rounds"][0])

                validation = experiment_ledger.validate_experiment_result(result)

                self.assertFalse(validation["valid"])

    def test_decision_requires_exact_source_round_binding(self):
        mutations = (
            lambda decision: decision.pop("baseline_profile"),
            lambda decision: decision.update(
                {"candidate_profile": "other/profile.json"}
            ),
            lambda decision: decision.update(
                {
                    "declared_change_set": {
                        "allowed_json_pointers": [],
                        "only_change_zh": "无变更",
                    }
                }
            ),
            lambda decision: decision.update(
                {"source_profiling_analysis_result": "other/analysis.json"}
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                result = self._result()
                mutate(result["performance_scope_decisions"][0])

                validation = experiment_ledger.validate_experiment_result(result)

                self.assertFalse(validation["valid"])
                self.assertTrue(
                    any("source round" in error for error in validation["errors"])
                )

    def test_conditional_evidence_retains_complete_profiling_binding(self):
        result = self._result()

        ledger = experiment_ledger.merge_experiment_result({}, result)

        for evidence in ledger["conditional_performance_evidence"].values():
            self.assertIn("baseline_profile", evidence)
            self.assertIn("baseline_profile_fingerprint", evidence)
            self.assertIn("candidate_profile", evidence)
            self.assertIn("candidate_profile_fingerprint", evidence)
            self.assertIn("declared_change_set", evidence)
            self.assertIn("source_profiling_analysis_result", evidence)
            self.assertIn("profiling_analysis_result", evidence)

    def test_rejects_illegal_decision_tuples(self):
        illegal = (
            ("neutral", "block", "verified"),
            ("regressed", "reprofile", "applied"),
            ("beneficial", "keep", "blocked"),
            ("insufficient_evidence", "block", "verified"),
        )
        for classification, action, status in illegal:
            with self.subTest(tuple=(classification, action, status)):
                result = self._result()
                decision = result["performance_scope_decisions"][0]
                decision.update(
                    {
                        "classification": classification,
                        "action": action,
                        "status": status,
                    }
                )

                validation = experiment_ledger.validate_experiment_result(result)

                self.assertFalse(validation["valid"])
                self.assertTrue(any("tuple" in error for error in validation["errors"]))

    def _automatic_result(self):
        result = self._result()
        result["experiment_id"] = "S1"
        result["parent_experiment_id"] = "S1"
        result["child_agent_id"] = "agent-1"
        result["rounds"] = [self._round("S1", "automatic_aot", "candidate-auto-fp", [])]
        result["rounds"][0]["child_agent_id"] = "agent-1"
        result["rounds"][0]["scope_kind"] = "automatic_aot"
        result["profiling_analysis_agent_ids"] = {"S1": "prof-agent-auto"}
        result["profiling_analysis_result"] = {
            "S1": "S1/profiling-analysis/profiling-analysis-result.json"
        }
        result["performance_scope_decisions"] = []
        result["unresolved_performance_ranges"] = []
        return result


class StrategyTest(unittest.TestCase):
    @staticmethod
    def _build_strategy(profiling_by_name, *, source_range_optimization=None):
        analysis_paths = {}
        for name, analyses in profiling_by_name.items():
            if isinstance(analyses, list):
                analysis_paths[name] = [
                    Path(f"/unit-test/{name}/{analysis['round_id']}.json")
                    for analysis in analyses
                ]
            else:
                analysis_paths[name] = Path(
                    f"/unit-test/{name}/profiling-analysis-result.json"
                )
        with mock.patch.object(
            recommend_sk_strategy,
            "_run_read_only_reanalysis",
            side_effect=lambda name, analysis, path: json.loads(json.dumps(analysis)),
        ):
            return recommend_sk_strategy.build_strategy(
                profiling_by_name,
                analysis_paths=analysis_paths,
                source_range_optimization=source_range_optimization,
            )

    @staticmethod
    def _build_source_range_strategy(profiling_by_name):
        return StrategyTest._build_strategy(
            profiling_by_name,
            source_range_optimization=set(profiling_by_name),
        )

    @staticmethod
    def _seal_analysis_content(analysis):
        decisions = analysis.get("per_sk_decisions")
        if analysis.get("schema_version") == "1.2" and isinstance(decisions, list):
            exact_projected = sum(
                item.get("mapping_confidence") == "exact_projected_trace"
                for item in decisions
                if isinstance(item, dict)
            )
            ambiguous = sum(
                item.get("mapping_confidence") == "ambiguous"
                for item in decisions
                if isinstance(item, dict)
            )
            distribution = {}
            blocker_counts = {}
            bound = 0
            for item in decisions:
                if not isinstance(item, dict):
                    continue
                if item.get("candidate_binding_status") == "bound":
                    bound += 1
                child_count = item.get("child_count")
                if isinstance(child_count, int) and not isinstance(child_count, bool):
                    bucket = "5+" if child_count >= 5 else str(child_count)
                    distribution[bucket] = distribution.get(bucket, 0) + 1
                for blocker in set(item.get("mapping_blockers") or ()):
                    blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
            analysis["mapping_coverage"] = {
                "total_sk_ids": len(decisions),
                "bound_sk_ids": bound,
                "exact_projected_trace_sk_ids": exact_projected,
                "ambiguous_sk_ids": ambiguous,
                "unmapped_sk_ids": len(decisions) - exact_projected - ambiguous,
                "filtered_by_child_count": 0,
                "child_count_distribution": distribution,
                "blocker_counts": blocker_counts,
            }
        analysis.pop("analysis_content_fingerprint", None)
        analysis["analysis_content_fingerprint"] = hashlib.sha256(
            json.dumps(
                analysis,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=analysis.get("schema_version") == "1.2",
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        return analysis

    @staticmethod
    def _rebind_analysis_identity(analysis):
        identity = {
            field: analysis[field]
            for field in (
                "experiment_id",
                "round_id",
                "analysis_agent_id",
                "candidate_name",
                "source_revision",
            )
        }
        analysis["analysis_id"] = (
            "analysis-"
            + hashlib.sha256(
                json.dumps(
                    identity,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=analysis.get("schema_version") == "1.2",
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
        )
        return StrategyTest._seal_analysis_content(analysis)

    @staticmethod
    def _analysis(*, recommended_experiments=None, round_id="S4-BASE"):
        scope_actions = [
            {
                "sk_id": "sk-beneficial",
                "range_id": "range-beneficial",
                "classification": "beneficial",
                "action": "keep",
                "source_scope": "decoder.layer.0.beneficial",
                "boundary": {"start_op": "A", "end_op": "A"},
            },
            {
                "sk_id": "sk-neutral",
                "range_id": "range-neutral",
                "classification": "neutral",
                "action": "prune",
                "source_scope": "decoder.layer.0.neutral",
                "boundary": {"start_op": "B", "end_op": "C"},
            },
            {
                "sk_id": "sk-regressed",
                "range_id": "range-regressed",
                "classification": "regressed",
                "action": "prune",
                "source_scope": "decoder.layer.0.regressed",
                "boundary": {"start_op": "D", "end_op": "E"},
            },
            {
                "sk_id": "sk-insufficient",
                "range_id": "range-insufficient",
                "classification": "insufficient_evidence",
                "action": "reprofile",
                "source_scope": "decoder.layer.0.unknown",
                "boundary": {"start_op": "F", "end_op": "G"},
            },
        ]
        for index, action in enumerate(scope_actions):
            sequence = [
                action["boundary"]["start_op"],
                action["boundary"]["end_op"],
            ]
            if sequence[0] == sequence[1]:
                sequence = sequence[:1]
            if action["action"] == "prune":
                start_offset = 10 if index == 1 else 12
                action["boundary"].update(
                    {
                        "source_file": "models/demo.py",
                        "start_offset": start_offset,
                        "end_offset": start_offset + 5,
                    }
                )
            action["ordered_child_op_sequence"] = sequence
            action["interval_unproven"] = "source_file" not in action["boundary"]
        identity = {
            "experiment_id": "S4",
            "round_id": round_id,
            "analysis_agent_id": f"profiling-agent-{round_id}",
            "candidate_name": "S4",
            "source_revision": "4cc7ccb",
        }
        analysis = {
            "schema_version": "1.2",
            **identity,
            "baseline_profile_fingerprint": "1" * 64,
            "candidate_profile_fingerprint": "2" * 64,
            "baseline_config_fingerprint": "3" * 64,
            "candidate_config_fingerprint": "4" * 64,
            "baseline_workload_fingerprint": "5" * 64,
            "candidate_workload_fingerprint": "5" * 64,
            "workload_fingerprint": "5" * 64,
            "baseline_control_fingerprint": "6" * 64,
            "candidate_control_fingerprint": "6" * 64,
            "control_fingerprint": "6" * 64,
            "declared_change_set": {
                "allowed_json_pointers": ["/superkernel/scope"],
                "only_change_zh": "只修改本轮 SuperKernel 框定范围。",
            },
            "thresholds": {
                "min_relative_change_pct": 3.0,
                "min_absolute_change_us": 1.0,
                "min_occurrences": 3,
            },
            "inputs": {
                "baseline_profile": "../BASE/prof/kernel_details.csv",
                "candidate_profile": "../S4/prof/kernel_details.csv",
                "sk_meta": "../S4/sk_meta",
                "baseline_config": "../BASE/config.json",
                "candidate_config": "../S4/config.json",
                "baseline_workload": "../BASE/workload.json",
                "candidate_workload": "../S4/workload.json",
                "declared_change_set": "../S4/declared-change.json",
                "sk_prof": None,
                "environment_evidence": "../S4/environment.json",
                "source_scope_map": "../S4/scope-map.json",
                "baseline_collection_manifest": None,
                "profile_collection_manifest": None,
            },
            "per_sk_decisions": [
                {
                    **json.loads(json.dumps(action)),
                    "device_id": 0,
                    "model_id": "48",
                    "raw_sk_id": index + 1,
                    "child_count": len(action["ordered_child_op_sequence"]),
                    "mapping_method": "source_scope_map",
                    "mapping_confidence": "exact",
                    "mapping_blockers": [],
                    "candidate_binding_status": None,
                    "evidence_errors": ["fixture_metrics_omitted"],
                    "identity": {
                        "model_id": "48",
                        "source_scope": action["source_scope"],
                        "boundary": {
                            "start_op": action["boundary"]["start_op"],
                            "end_op": action["boundary"]["end_op"],
                        },
                        "ordered_child_op_sequence": list(
                            action["ordered_child_op_sequence"]
                        ),
                    },
                }
                for index, action in enumerate(scope_actions)
            ],
            "scope_actions": scope_actions,
            "source_scope_mapping": {
                "protocol": "source_scope_map_v2",
                "status": "exact",
                "blockers": [],
                "source_revision": "4cc7ccb",
                "source_revision_role": "stable_marker",
                "stable_marker_revision": "4cc7ccb",
            },
            "association_protocol": {
                "protocol": "kernel_projection_trace_v2",
                "status": "not_requested",
                "manifest_set_fingerprint": None,
                "manifest_content_fingerprints": {
                    "baseline_collection_manifest": None,
                    "profile_collection_manifest": None,
                },
                "blockers": ["structural_collection_manifests_missing"],
            },
            "candidate_binding_evidence": {"summary": None, "occurrences": []},
            "canonical_graph_fingerprints": {},
            "stream_role_mapping": {},
            "graph_alignment_proof": {},
            "mapping_coverage": {
                "total_sk_ids": 4,
                "bound_sk_ids": 0,
                "exact_projected_trace_sk_ids": 0,
                "ambiguous_sk_ids": 0,
                "unmapped_sk_ids": 4,
                "filtered_by_child_count": 0,
                "child_count_distribution": {"1": 1, "2": 3},
                "blocker_counts": {},
            },
            "diagnostic_hypotheses": [],
            "recommended_experiments": recommended_experiments
            if recommended_experiments is not None
            else [
                {
                    "experiment_id": "invalid-option-regressed-auto-op-parallel",
                    "range_id": "range-regressed",
                    "only_change": {"option": "auto_op_parallel", "value": 1},
                    "only_change_zh": "只修改选项 auto_op_parallel 为 1。",
                    "option": "auto_op_parallel",
                    "value": 1,
                    "accepted_evidence": {
                        "source": (
                            "environment_evidence.options.optimize_options."
                            "auto_op_parallel.accepted_values"
                        ),
                        "accepted_value": 1,
                    },
                    "expected_signal_zh": "目标范围恢复 Cube/Vector 重叠且区间耗时改善。",
                    "lifecycle_gates": [
                        "correctness",
                        "profile_vs_baseline_remeasurement",
                        "clean_performance_profile",
                        "read_only_reanalysis",
                    ],
                }
            ],
            "blockers": [],
            "next_agent_guidance_zh": "仅按 schema 1.2 的逐 SK 证据生成下一轮。",
        }
        return StrategyTest._rebind_analysis_identity(analysis)

    @staticmethod
    def _stable_p_analysis(*, recommended_experiments=None):
        analysis = StrategyTest._analysis(
            recommended_experiments=recommended_experiments,
            round_id="S4-P1",
        )
        analysis["scope_actions"] = [
            action
            for action in analysis["scope_actions"]
            if action["action"] != "prune"
        ]
        analysis["per_sk_decisions"] = [
            decision
            for decision in analysis["per_sk_decisions"]
            if decision["action"] != "prune"
        ]
        return StrategyTest._seal_analysis_content(analysis)

    @staticmethod
    def _structural_auto_analysis():
        fixture_path = (
            Path(__file__).resolve().parents[2]
            / "superkernel-fusion-performance-analysis"
            / "tests"
            / "test_fusion_performance.py"
        )
        spec = importlib.util.spec_from_file_location(
            "structural_strategy_fixture", fixture_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load structural fixture: {fixture_path}")
        fixture_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture_module)
        analysis = fixture_module._signed_exact_projected_trace_report()
        analysis.update(
            {
                "experiment_id": "S1",
                "round_id": "S1-AUTO",
                "analysis_agent_id": "profiling-agent-S1-AUTO",
                "candidate_name": "S1",
                "source_scope_mapping": {
                    "protocol": "source_scope_map_v2",
                    "status": "not_requested",
                    "blockers": ["source_scope_map_missing"],
                },
            }
        )
        for field in ("per_sk_decisions", "scope_actions"):
            analysis[field][0]["classification"] = "regressed"
            analysis[field][0]["action"] = "block"
        analysis["per_sk_decisions"][0]["action_blocker"] = (
            "prune_requires_exact_source_scope_boundary_mapping"
        )
        analysis["recommended_experiments"] = []
        return StrategyTest._rebind_analysis_identity(analysis)

    @staticmethod
    def _completed_history(recommended_experiments):
        return [
            StrategyTest._analysis(recommended_experiments=recommended_experiments),
            StrategyTest._stable_p_analysis(recommended_experiments=[]),
        ]

    @staticmethod
    def _whole_scope_ready_auto_analysis():
        analysis = StrategyTest._analysis(
            recommended_experiments=[], round_id="S1-AUTO"
        )
        analysis.update(
            {
                "experiment_id": "S1",
                "candidate_name": "S1",
                "analysis_agent_id": "profiling-agent-S1-AUTO",
            }
        )
        for field in ("per_sk_decisions", "scope_actions"):
            analysis[field] = [
                item
                for item in analysis[field]
                if item["classification"] != "insufficient_evidence"
            ]
        analysis["source_scope_mapping"].update(
            {
                "source_revision_role": "stable_source",
            }
        )
        analysis["source_scope_mapping"].pop("stable_marker_revision", None)
        return StrategyTest._rebind_analysis_identity(analysis)

    def test_build_strategy_requires_analysis_paths_before_runnable_plan(self):
        with self.assertRaisesRegex(ValueError, "analysis_paths.*只读重分析"):
            recommend_sk_strategy.build_strategy({"S4": self._analysis()})

    @classmethod
    def _write_real_analysis_fixture(
        cls,
        root,
        *,
        range_count=1,
        round_id="S4-BASE",
        candidate_ranges=None,
        candidate_durations=None,
        dcci_recommendation=False,
        shared_baseline=None,
    ):
        if range_count not in (1, 2):
            raise ValueError("range_count must be 1 or 2")
        available_ranges = ("A", "B")[:range_count]
        if candidate_ranges is None:
            candidate_ranges = available_ranges
        candidate_ranges = tuple(candidate_ranges)
        if not candidate_ranges or not set(candidate_ranges) <= set(available_ranges):
            raise ValueError("candidate_ranges must be a non-empty available subset")
        candidate_durations = {"A": 12, "B": 6, **(candidate_durations or {})}
        analysis_dir = root / "profiling-analysis"
        analysis_dir.mkdir(parents=True)
        baseline_profile = (
            Path(shared_baseline["baseline_profile"])
            if shared_baseline is not None
            else root / "BASE" / "profile.csv"
        )
        candidate_profile = root / "S4" / "profile.csv"
        baseline_profile.parent.mkdir(exist_ok=True)
        candidate_profile.parent.mkdir()
        csv_fields = [
            "Step Id",
            "Device_id",
            "Model ID",
            "Task ID",
            "Stream ID",
            "Name",
            "Type",
            "OP State",
            "Accelerator Core",
            "Start Time(us)",
            "Duration(us)",
            "Wait Time(us)",
            "Block Num",
            "Mix Block Num",
            "aic_scalar_time(us)",
            "aic_scalar_ratio",
            "aiv_scalar_time(us)",
            "aiv_scalar_ratio",
        ]

        def write_profile(path, rows):
            with path.open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=csv_fields)
                writer.writeheader()
                for task_id, name, op_type, duration in rows:
                    for occurrence, start in enumerate((0, 20, 40)):
                        scalar_ratio = (
                            0.3
                            if dcci_recommendation
                            and task_id == 10
                            and path == baseline_profile
                            else 0
                        )
                        writer.writerow(
                            {
                                "Step Id": occurrence,
                                "Device_id": 0,
                                "Model ID": "48",
                                "Task ID": task_id,
                                "Stream ID": 1,
                                "Name": name,
                                "Type": op_type,
                                "OP State": "static",
                                "Accelerator Core": "AI_VECTOR_CORE",
                                "Start Time(us)": start,
                                "Duration(us)": duration,
                                "Wait Time(us)": 0,
                                "Block Num": 1,
                                "Mix Block Num": 0,
                                "aic_scalar_time(us)": 0,
                                "aic_scalar_ratio": 0,
                                "aiv_scalar_time(us)": duration * scalar_ratio,
                                "aiv_scalar_ratio": scalar_ratio,
                            }
                        )

        child_name = "static_kernel_A_hash"
        sk_name = (
            "sk_1_decoder.layer.0_start_static_kernel_A_hash_end_static_kernel_A_hash"
        )
        baseline_rows = [(10, child_name, "A", 8)]
        candidate_rows = []
        if "A" in candidate_ranges:
            candidate_rows.append(
                (110, sk_name, "SuperKernel", candidate_durations["A"])
            )
        if range_count == 2:
            baseline_rows.append((20, "static_kernel_B_hash", "B", 8))
            if "B" in candidate_ranges:
                candidate_rows.append(
                    (
                        120,
                        "sk_2_decoder.layer.1_start_static_kernel_B_hash_"
                        "end_static_kernel_B_hash",
                        "SuperKernel",
                        candidate_durations["B"],
                    )
                )
        if shared_baseline is None:
            write_profile(baseline_profile, baseline_rows)
        write_profile(candidate_profile, candidate_rows)

        sk_meta = root / "S4" / "sk_meta" / "model_48"
        sk_meta.mkdir(parents=True)
        metadata_text = ""
        if "A" in candidate_ranges:
            metadata_text += (
                f"SK Function: {sk_name}, scope id: 1, Node Count: 1\n"
                "[nodeId:10, streamId:1] - "
                "KernelInfos{funcName:static_kernel_A_hash, kernelType:AIV_ONLY, "
                "numBlocks:1, cubeNum:0, vecNum:1, isScheModeOn:0}\n"
            )
        if "B" in candidate_ranges:
            metadata_text += (
                "SK Function: sk_2_decoder.layer.1_start_static_kernel_B_hash_"
                "end_static_kernel_B_hash, scope id: 2, Node Count: 1\n"
                "[nodeId:20, streamId:1] - "
                "KernelInfos{funcName:static_kernel_B_hash, kernelType:AIV_ONLY, "
                "numBlocks:1, cubeNum:0, vecNum:1, isScheModeOn:0}\n"
            )
        (sk_meta / "sk_fused_nodes.log").write_text(metadata_text)
        replay_evidence = root / "S4" / "fusion-replay.json"
        baseline_config = (
            Path(shared_baseline["baseline_config"])
            if shared_baseline is not None
            else root / "BASE" / "config.json"
        )
        candidate_config = root / "S4" / "config.json"
        runtime_evidence = {"dcci_state": "enabled"} if dcci_recommendation else None
        baseline_config_data = {
            "runtime": {"rank_size": 8},
            "superkernel": {"enabled": False, "scope": []},
        }
        if runtime_evidence is not None:
            baseline_config_data["runtime_evidence"] = runtime_evidence
        if shared_baseline is None:
            baseline_config.write_text(json.dumps(baseline_config_data))
        candidate_config_data = {
            "runtime": {"rank_size": 8},
            "superkernel": {
                "enabled": True,
                "scope": [
                    *(["decoder.layer.0"] if "A" in candidate_ranges else []),
                    *(["decoder.layer.1"] if "B" in candidate_ranges else []),
                ],
            },
        }
        if runtime_evidence is not None:
            candidate_config_data["runtime_evidence"] = runtime_evidence
        candidate_config.write_text(json.dumps(candidate_config_data))
        verify_config = root / "S4" / "verify-config.json"
        verify_config.write_text(json.dumps(candidate_config_data))
        verify_meta = root / "S4" / "verify-sk_meta" / "model_48"
        verify_meta.mkdir(parents=True)
        (verify_meta / "sk_fused_nodes.log").write_text(metadata_text)
        replay_evidence.write_text(
            json.dumps(
                analyze_sk_meta.compare_fusion_replay(
                    sk_meta.parent,
                    verify_meta.parent,
                    compat_config_manifest=candidate_config,
                    verify_config_manifest=verify_config,
                    candidate_name="S4",
                    round_id=round_id,
                    source_revision="2d3de05",
                    artifact_base=replay_evidence.parent,
                )
            )
        )
        workload = {
            "model": "demo",
            "input": {"tokens": 128},
            "batch": 1,
            "rank_size": 8,
            "mode": "prefill_decode",
            "warmup": 1,
            "iterations": 10,
            "runtime": {"dtype": "float16"},
        }
        baseline_workload = (
            Path(shared_baseline["baseline_workload"])
            if shared_baseline is not None
            else root / "BASE" / "workload.json"
        )
        candidate_workload = root / "S4" / "workload.json"
        if shared_baseline is None:
            baseline_workload.write_text(json.dumps(workload))
        candidate_workload.write_text(json.dumps(workload))
        declared_change_set = root / "S4" / "declared-change.json"
        declared_change_set.write_text(
            json.dumps(
                {
                    "allowed_json_pointers": [
                        "/superkernel/enabled",
                        "/superkernel/scope",
                    ],
                    "only_change_zh": "只启用 SuperKernel 并应用本轮 scope。",
                }
            )
        )
        sk_prof = root / "S4" / "sk_prof_device_0.json"
        sk_prof.write_text(
            json.dumps(
                {
                    "traceEvents": (
                        [
                            {
                                "name": child_name,
                                "ph": "X",
                                "ts": 0,
                                "dur": 12,
                                "tid": 1,
                                "args": {"kernelType": "AIV_ONLY"},
                            }
                        ]
                        if "A" in candidate_ranges
                        else []
                    )
                    + (
                        [
                            {
                                "name": "static_kernel_B_hash",
                                "ph": "X",
                                "ts": 20,
                                "dur": 6,
                                "tid": 1,
                                "args": {"kernelType": "AIV_ONLY"},
                            }
                        ]
                        if "B" in candidate_ranges
                        else []
                    )
                }
            )
        )
        environment_evidence = root / "S4" / "environment.json"
        environment = {"accepted_options": {"auto_op_parallel": [0, 1]}}
        if dcci_recommendation:
            environment.update(
                {
                    "runtime_evidence": {"dcci_state": "enabled"},
                    "options": {
                        "optimize_options": {
                            option: {"accepted_values": [["^static_kernel_A_hash$"]]}
                            for option in (
                                "dcci_before_kernel_start",
                                "dcci_after_kernel_end",
                            )
                        }
                    },
                }
            )
        environment_evidence.write_text(json.dumps(environment))
        source_scope_map = root / "S4" / "source-scope-map.json"
        source_scope_map.write_text(
            json.dumps(
                {
                    "task_ranges": [
                        {
                            "layer": 0,
                            "model_id": "48",
                            "start_task_id": 10,
                            "end_task_id": 10,
                            "source_scope": "decoder.layer.0",
                            "ordered_child_op_sequence": ["A"],
                            "boundary": {
                                "start_op": "A",
                                "end_op": "A",
                                "source_file": "model.py",
                                "start_offset": 100,
                                "end_offset": 200,
                            },
                        },
                        *(
                            [
                                {
                                    "layer": 1,
                                    "model_id": "48",
                                    "start_task_id": 20,
                                    "end_task_id": 20,
                                    "source_scope": "decoder.layer.1",
                                    "ordered_child_op_sequence": ["B"],
                                    "boundary": {
                                        "start_op": "B",
                                        "end_op": "B",
                                        "source_file": "model.py",
                                        "start_offset": 200,
                                        "end_offset": 300,
                                    },
                                }
                            ]
                            if range_count == 2
                            else []
                        ),
                    ]
                }
            )
        )
        artifacts = {
            "baseline_profile": baseline_profile,
            "candidate_profile": candidate_profile,
            "sk_meta": sk_meta.parent,
            "baseline_config": baseline_config,
            "candidate_config": candidate_config,
            "baseline_workload": baseline_workload,
            "candidate_workload": candidate_workload,
            "declared_change_set": declared_change_set,
            "sk_prof": sk_prof,
            "environment_evidence": environment_evidence,
            "source_scope_map": source_scope_map,
        }
        analysis_path = analysis_dir / "profiling-analysis-result.json"
        analyzer = (
            Path(__file__).resolve().parents[2]
            / "superkernel-fusion-performance-analysis"
            / "scripts"
            / "analyze_fusion_performance.py"
        )
        command = [sys.executable, str(analyzer)]
        for field, path in artifacts.items():
            command.extend([f"--{field.replace('_', '-')}", str(path)])
        command.extend(
            [
                "--candidate-name",
                "S4",
                "--experiment-id",
                "S4",
                "--round-id",
                round_id,
                "--analysis-agent-id",
                f"profiling-agent-{round_id}",
                "--source-revision",
                "2d3de05",
                "--min-relative-change-pct",
                "3",
                "--min-absolute-change-us",
                "1",
                "--min-occurrences",
                "3",
                "--json-out",
                str(analysis_path),
            ]
        )
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise AssertionError(
                f"真实 analyzer fixture 生成失败：{completed.stderr or completed.stdout}"
            )
        return analysis_path, artifacts

    @unittest.skip(
        "legacy task-range exact fixture; source_scope_map_v2 flow is covered in sibling tests"
    )
    def test_cli_accepts_ordered_duplicate_name_real_base_p_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path, base_artifacts = self._write_real_analysis_fixture(
                root / "base",
                range_count=2,
                dcci_recommendation=True,
            )
            p_path, _ = self._write_real_analysis_fixture(
                root / "p1",
                range_count=2,
                round_id="S4-P1",
                candidate_ranges=("B",),
                dcci_recommendation=True,
                shared_baseline=base_artifacts,
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = recommend_sk_strategy.main(
                    [
                        "--profiling-analysis",
                        f"S4={base_path}",
                        "--profiling-analysis",
                        f"S4={p_path}",
                    ]
                )

            report = json.loads(stdout.getvalue())
            matrix = report["candidates"]["S4"]["next_candidate_matrix"]
            self.assertEqual(exit_code, 0)
            self.assertEqual(matrix, [])
            self.assertEqual({item["comparison_to"] for item in matrix}, {"S4-P1"})
            self.assertEqual(
                {item["source_analysis_round_id"] for item in matrix},
                {"S4-BASE"},
            )
            self.assertEqual(
                {item["source_profiling_analysis_result"] for item in matrix},
                {str(base_path)},
            )

    def test_structural_auto_regression_recommends_source_mapping_completion(self):
        analysis = self._structural_auto_analysis()

        strategy = self._build_strategy({"S1": analysis})

        matrix = strategy["candidates"]["S1"]["next_candidate_matrix"]
        followups = [
            item for item in matrix if item["round_kind"] == "source_mapping_completion"
        ]
        whole_scope = [
            item
            for item in matrix
            if item["round_kind"] == "whole_scope_clean_validation"
        ]
        self.assertEqual(len(followups), 1)
        self.assertEqual(len(whole_scope), 0)
        followup = followups[0]
        self.assertEqual(followup["round_kind"], "source_mapping_completion")
        self.assertEqual(followup["target_range_ids"], ["range-contract"])
        self.assertEqual(followup["graph_occurrence_fingerprints"], ["6" * 64])
        for forbidden in ("source_scope", "source_file", "start_offset", "end_offset"):
            self.assertNotIn(forbidden, followup)
        self.assertEqual(followup["source_revision_role"], "stable_source")
        self.assertEqual(
            strategy["candidates"]["S1"]["scope_strategy"]["source_mapping_status"],
            "required",
        )

    def test_exact_source_map_defaults_to_whole_scope_without_source_range_branch(self):
        analysis = self._whole_scope_ready_auto_analysis()

        strategy = self._build_strategy({"S1": analysis})

        candidate = strategy["candidates"]["S1"]
        round_kinds = {
            item["round_kind"] for item in candidate["next_candidate_matrix"]
        }
        self.assertNotIn("performance_prune", round_kinds)
        self.assertIn("whole_scope_clean_validation", round_kinds)
        self.assertEqual(
            candidate["scope_strategy"]["source_range_optimization_status"],
            "not_requested",
        )
        self.assertFalse(
            any(
                blocker["kind"].startswith("prune_")
                or blocker["kind"].startswith("ambiguous_prune")
                for blocker in candidate["blockers"]
            )
        )

    def test_explicit_source_range_authorization_enters_auto_p1(self):
        analysis = self._whole_scope_ready_auto_analysis()

        strategy = self._build_strategy(
            {"S1": analysis}, source_range_optimization={"S1"}
        )

        candidate = strategy["candidates"]["S1"]
        matrix = candidate["next_candidate_matrix"]
        prune_rounds = [
            item for item in matrix if item["round_kind"] == "performance_prune"
        ]
        self.assertEqual(len(prune_rounds), 1)
        self.assertEqual(prune_rounds[0]["id"], "S1-P1")
        self.assertEqual(
            prune_rounds[0]["target_range_ids"],
            ["range-neutral"],
        )
        self.assertFalse(
            any(item["round_kind"] == "source_mapping_completion" for item in matrix)
        )
        self.assertEqual(
            candidate["scope_strategy"]["source_range_optimization_status"],
            "requested",
        )

    def test_source_range_authorization_api_fails_closed(self):
        analysis = self._whole_scope_ready_auto_analysis()
        invalid_values = (
            "S1",
            1,
            [""],
            ["S1", "S1"],
            {"S2"},
        )

        for source_range_optimization in invalid_values:
            with self.subTest(source_range_optimization=source_range_optimization):
                with self.assertRaisesRegex(
                    (TypeError, ValueError),
                    "source_range_optimization|源码范围优化|候选",
                ):
                    self._build_strategy(
                        {"S1": analysis},
                        source_range_optimization=source_range_optimization,
                    )

    def test_cli_requires_per_candidate_source_range_authorization(self):
        analysis = self._whole_scope_ready_auto_analysis()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "analysis.json"
            source.write_text(json.dumps(analysis))

            def invoke(extra_args):
                stdout = io.StringIO()
                with (
                    mock.patch.object(
                        recommend_sk_strategy,
                        "_run_read_only_reanalysis",
                        side_effect=lambda name, report, path: json.loads(
                            json.dumps(report)
                        ),
                    ),
                    redirect_stdout(stdout),
                ):
                    exit_code = recommend_sk_strategy.main(
                        ["--profiling-analysis", f"S1={source}", *extra_args]
                    )
                return exit_code, json.loads(stdout.getvalue())

            default_exit, default_report = invoke([])
            requested_exit, requested_report = invoke(
                ["--source-range-optimization", "S1"]
            )

            self.assertEqual(default_exit, 0)
            self.assertFalse(
                any(
                    item["round_kind"] == "performance_prune"
                    for item in default_report["candidates"]["S1"][
                        "next_candidate_matrix"
                    ]
                )
            )
            self.assertEqual(requested_exit, 0)
            self.assertTrue(
                any(
                    item["id"] == "S1-P1"
                    for item in requested_report["candidates"]["S1"][
                        "next_candidate_matrix"
                    ]
                )
            )

            for invalid_args in (
                ["--source-range-optimization", "S2"],
                [
                    "--source-range-optimization",
                    "S1",
                    "--source-range-optimization",
                    "S1",
                ],
                ["--source-range-optimization", ""],
            ):
                with self.subTest(invalid_args=invalid_args):
                    stderr = io.StringIO()
                    with (
                        mock.patch.object(
                            recommend_sk_strategy,
                            "_run_read_only_reanalysis",
                            side_effect=lambda name, report, path: json.loads(
                                json.dumps(report)
                            ),
                        ),
                        redirect_stderr(stderr),
                        self.assertRaises(SystemExit),
                    ):
                        recommend_sk_strategy.main(
                            [
                                "--profiling-analysis",
                                f"S1={source}",
                                *invalid_args,
                            ]
                        )
                    self.assertNotIn("Traceback", stderr.getvalue())

    def test_p_history_requires_source_range_authorization_via_api(self):
        base = self._analysis()
        p1 = self._analysis(round_id="S4-P1")

        with self.assertRaisesRegex(
            ValueError,
            "P/FINAL profiling 历史要求.*source_range_optimization 显式授权",
        ):
            self._build_strategy({"S4": [base, p1]})

    def test_cli_rejects_p_history_without_source_range_authorization(self):
        base = self._analysis()
        p1 = self._analysis(round_id="S4-P1")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path = root / "base.json"
            p1_path = root / "p1.json"
            base_path.write_text(json.dumps(base))
            p1_path.write_text(json.dumps(p1))
            stderr = io.StringIO()

            with (
                mock.patch.object(
                    recommend_sk_strategy,
                    "_run_read_only_reanalysis",
                    side_effect=lambda name, report, path: json.loads(
                        json.dumps(report)
                    ),
                ),
                redirect_stderr(stderr),
                self.assertRaises(SystemExit),
            ):
                recommend_sk_strategy.main(
                    [
                        "--profiling-analysis",
                        f"S4={base_path}",
                        "--profiling-analysis",
                        f"S4={p1_path}",
                    ]
                )

            self.assertRegex(
                stderr.getvalue(),
                "P/FINAL profiling 历史要求.*source_range_optimization 显式授权",
            )
            self.assertNotIn("Traceback", stderr.getvalue())

    def test_manual_winner_requires_source_mapping_before_whole_scope(self):
        analysis = self._analysis(recommended_experiments=[])
        analysis["source_scope_mapping"] = {
            "protocol": "source_scope_map_v2",
            "status": "not_requested",
            "blockers": ["source_scope_map_missing"],
        }
        analysis["per_sk_decisions"] = [
            item
            for item in analysis["per_sk_decisions"]
            if item["classification"] != "insufficient_evidence"
        ]
        analysis["scope_actions"] = [
            item
            for item in analysis["scope_actions"]
            if item["classification"] != "insufficient_evidence"
        ]
        for field in ("per_sk_decisions", "scope_actions"):
            for item in analysis[field]:
                if item["classification"] in {"neutral", "regressed"}:
                    item["action"] = "block"
                    item["action_blocker"] = (
                        "prune_requires_exact_source_scope_boundary_mapping"
                    )
                    item["boundary"].pop("source_file", None)
                    item["boundary"].pop("start_offset", None)
                    item["boundary"].pop("end_offset", None)
                    item["interval_unproven"] = True
        self._seal_analysis_content(analysis)

        candidate = self._build_strategy({"S4": analysis})["candidates"]["S4"]
        whole_scope = [
            item
            for item in candidate["next_candidate_matrix"]
            if item["round_kind"] == "whole_scope_clean_validation"
        ]
        source_mapping = [
            item
            for item in candidate["next_candidate_matrix"]
            if item["round_kind"] == "source_mapping_completion"
        ]

        self.assertEqual(whole_scope, [])
        self.assertEqual(len(source_mapping), 1)
        self.assertEqual(
            source_mapping[0]["target_range_ids"],
            [
                "range-beneficial",
                "range-neutral",
                "range-regressed",
            ],
        )
        self.assertEqual(source_mapping[0]["source_revision_role"], "stable_marker")
        self.assertEqual(
            source_mapping[0]["source_edit_policy"], "stable_marker_calibration"
        )

    def test_completed_source_mapping_skips_unmapped_ranges_and_allows_whole_scope(
        self,
    ):
        analysis = self._structural_auto_analysis()
        analysis["source_scope_mapping"] = {
            "protocol": "source_scope_map_v2",
            "status": "exact",
            "blockers": [],
            "source_revision": analysis["source_revision"],
            "source_revision_role": "stable_source",
        }
        analysis["inputs"]["source_scope_map"] = "../SMAP/source-scope-map-v2.json"
        self._seal_analysis_content(analysis)

        candidate = self._build_strategy({"S1": analysis})["candidates"]["S1"]

        self.assertFalse(
            any(
                item["round_kind"] == "source_mapping_completion"
                for item in candidate["next_candidate_matrix"]
            )
        )
        self.assertTrue(
            any(
                item["round_kind"] == "whole_scope_clean_validation"
                for item in candidate["next_candidate_matrix"]
            )
        )
        self.assertEqual(
            candidate["scope_strategy"]["source_mapping_skipped_range_ids"],
            ["range-contract"],
        )

    def test_insufficient_evidence_blocks_whole_scope_clean_validation(self):
        candidate = self._build_strategy({"S4": self._analysis()})["candidates"]["S4"]

        self.assertFalse(
            any(
                item["round_kind"] == "whole_scope_clean_validation"
                for item in candidate["next_candidate_matrix"]
            )
        )

    def test_structural_auto_requires_unique_projected_trace_proof(self):
        analysis = self._structural_auto_analysis()
        proof = analysis["graph_alignment_proof"]["device:0/model:48/sk:7"]
        proof["alternative_solution_count_by_step"]["2"] = 1
        self._seal_analysis_content(analysis)

        with self.assertRaisesRegex(ValueError, "结构证据|kernel projection"):
            self._build_strategy({"S1": analysis})

    def test_structural_auto_requires_complete_schema_1_2_binding_contract(self):
        analysis = self._structural_auto_analysis()
        analysis["association_protocol"]["status"] = "blocked"
        analysis["association_protocol"]["blockers"] = [
            "profile_same_process_metadata_missing"
        ]
        self._seal_analysis_content(analysis)

        with self.assertRaisesRegex(ValueError, "schema 1.2|association_protocol"):
            self._build_strategy({"S1": analysis})

    def test_schema_1_2_non_exact_mapping_must_remain_insufficient(self):
        analysis = self._structural_auto_analysis()
        for field in ("per_sk_decisions", "scope_actions"):
            analysis[field][0]["mapping_confidence"] = "ambiguous"
            analysis[field][0]["mapping_blockers"] = ["structural_mapping_not_unique"]
        analysis["mapping_coverage"].update(
            {
                "exact_projected_trace_sk_ids": 0,
                "ambiguous_sk_ids": 1,
                "blocker_counts": {"structural_mapping_not_unique": 1},
            }
        )
        self._seal_analysis_content(analysis)

        with self.assertRaisesRegex(ValueError, "insufficient_evidence"):
            self._build_strategy({"S1": analysis})

    def test_cli_reanalysis_rejects_coordinated_decision_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis_path, artifacts = self._write_real_analysis_fixture(root)
            input_files = [path for path in artifacts.values() if path.is_file()] + [
                analysis_path
            ]
            before = {path: path.read_bytes() for path in input_files}
            stdout = io.StringIO()
            previous_cwd = Path.cwd()
            try:
                os.chdir(root / "S4")
                with redirect_stdout(stdout):
                    self.assertEqual(
                        recommend_sk_strategy.main(
                            ["--profiling-analysis", f"S4={analysis_path}"]
                        ),
                        0,
                    )
            finally:
                os.chdir(previous_cwd)
            strategy = json.loads(stdout.getvalue())
            self.assertIn("candidates", strategy)
            self.assertNotIn("per_sk_decisions", strategy)
            self.assertEqual(
                {path: path.read_bytes() for path in input_files},
                before,
            )

            tampered = json.loads(analysis_path.read_text())
            for field in ("per_sk_decisions", "scope_actions"):
                tampered[field][0]["classification"] = "beneficial"
                tampered[field][0]["action"] = "keep"
            self._seal_analysis_content(tampered)
            analysis_path.write_text(json.dumps(tampered))

            stderr = io.StringIO()
            stdout = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    recommend_sk_strategy.main(
                        ["--profiling-analysis", f"S4={analysis_path}"]
                    )
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("非精确关联", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertEqual(stdout.getvalue(), "")

    def test_cli_reanalysis_failures_are_controlled_and_do_not_leak_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis_path, _ = self._write_real_analysis_fixture(root)
            arguments = ["--profiling-analysis", f"S4={analysis_path}"]

            failures = []
            failures.append(
                (
                    "missing_sibling",
                    mock.patch.object(
                        recommend_sk_strategy,
                        "_analyzer_script_path",
                        return_value=root / "missing-analyzer.py",
                    ),
                    "重分析器不存在",
                )
            )
            failures.append(
                (
                    "process_os_error",
                    mock.patch.object(
                        recommend_sk_strategy.subprocess,
                        "run",
                        side_effect=OSError("exec failed"),
                    ),
                    "无法启动只读重分析",
                )
            )
            failures.append(
                (
                    "process_nonzero",
                    mock.patch.object(
                        recommend_sk_strategy.subprocess,
                        "run",
                        return_value=subprocess.CompletedProcess(
                            [],
                            7,
                            stdout="ANALYZER_STDOUT_SECRET",
                            stderr=(
                                "Traceback (most recent call last):\n"
                                '  File "analyzer.py", line 1\n'
                                "RuntimeError: analyzer failed"
                            ),
                        ),
                    ),
                    "只读重分析失败",
                )
            )

            def write_malformed_output(command, **kwargs):
                Path(command[-1]).write_text("{malformed")
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout="ANALYZER_STDOUT_SECRET",
                    stderr="",
                )

            failures.append(
                (
                    "malformed_json",
                    mock.patch.object(
                        recommend_sk_strategy.subprocess,
                        "run",
                        side_effect=write_malformed_output,
                    ),
                    "只读重分析输出无效",
                )
            )

            for label, patcher, expected in failures:
                with self.subTest(failure=label), patcher:
                    stdout = io.StringIO()
                    stderr = io.StringIO()
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        with self.assertRaises(SystemExit) as raised:
                            recommend_sk_strategy.main(arguments)
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn(expected, stderr.getvalue())
                    self.assertNotIn("Traceback", stderr.getvalue())
                    self.assertNotIn("ANALYZER_STDOUT_SECRET", stderr.getvalue())
                    self.assertEqual(stdout.getvalue(), "")

    def test_reanalysis_compares_every_semantic_field_canonically(self):
        required = {
            "analysis_id",
            "experiment_id",
            "round_id",
            "analysis_agent_id",
            "candidate_name",
            "source_revision",
            *recommend_sk_strategy.FINGERPRINT_FIELDS,
            "declared_change_set",
            "thresholds",
            "coverage",
            "per_sk_decisions",
            "scope_actions",
            "diagnostic_hypotheses",
            "recommended_experiments",
            "blockers",
            "environment_evidence_loaded",
            "next_agent_guidance_zh",
        }
        self.assertLessEqual(
            required,
            set(recommend_sk_strategy.REANALYSIS_SEMANTIC_FIELDS),
        )
        original = {
            field: {"first": 1, "second": field}
            for field in recommend_sk_strategy.REANALYSIS_SEMANTIC_FIELDS
        }
        regenerated = json.loads(json.dumps(original))
        regenerated["thresholds"] = {
            "second": "thresholds",
            "first": 1,
        }
        recommend_sk_strategy._validate_reanalysis_semantics(
            "S4", original, regenerated
        )

        for field in recommend_sk_strategy.REANALYSIS_SEMANTIC_FIELDS:
            with self.subTest(field=field):
                changed = json.loads(json.dumps(original))
                changed[field] = {"changed": True}
                with self.assertRaisesRegex(ValueError, re.escape(field)):
                    recommend_sk_strategy._validate_reanalysis_semantics(
                        "S4", original, changed
                    )

        original.update(
            {
                "inputs": {"candidate_profile": "old/path.csv"},
                "outputs": {"markdown": "old/report.md"},
                "analysis_content_fingerprint": "1" * 64,
            }
        )
        regenerated.update(
            {
                "inputs": {"candidate_profile": "temp/path.csv"},
                "outputs": {"markdown": None},
                "analysis_content_fingerprint": "2" * 64,
            }
        )
        recommend_sk_strategy._validate_reanalysis_semantics(
            "S4", original, regenerated
        )

    def test_history_with_fresh_p_prune_emits_only_next_p(self):
        base = self._analysis()
        p1 = self._analysis(round_id="S4-P1")

        matrix = self._build_source_range_strategy({"S4": [base, p1]})["candidates"][
            "S4"
        ]["next_candidate_matrix"]

        self.assertEqual([item["id"] for item in matrix], ["S4-P2"])
        self.assertEqual(matrix[0]["round_kind"], "performance_prune")

    def test_history_requires_shared_identity_and_contiguous_rounds(self):
        base = self._analysis()
        p2 = self._stable_p_analysis(recommended_experiments=[])
        p2["round_id"] = "S4-P2"
        p2["analysis_agent_id"] = "profiling-agent-S4-P2"
        self._rebind_analysis_identity(p2)
        with self.assertRaisesRegex(ValueError, "P 轮历史.*连续"):
            self._build_source_range_strategy({"S4": [base, p2]})

        for field in (
            "experiment_id",
            "source_revision",
            "workload_fingerprint",
            "control_fingerprint",
        ):
            with self.subTest(field=field):
                p1 = self._stable_p_analysis(recommended_experiments=[])
                p1[field] = "changed"
                if field == "workload_fingerprint":
                    p1["baseline_workload_fingerprint"] = "a" * 64
                    p1["candidate_workload_fingerprint"] = "a" * 64
                    p1[field] = "a" * 64
                elif field == "control_fingerprint":
                    p1["baseline_control_fingerprint"] = "b" * 64
                    p1["candidate_control_fingerprint"] = "b" * 64
                    p1[field] = "b" * 64
                self._rebind_analysis_identity(p1)
                with self.assertRaisesRegex(ValueError, field):
                    self._build_source_range_strategy({"S4": [base, p1]})

    def test_history_binds_all_frozen_baseline_fingerprints(self):
        mutations = {
            "baseline_profile_fingerprint": ("baseline_profile_fingerprint",),
            "baseline_config_fingerprint": ("baseline_config_fingerprint",),
            "baseline_workload_fingerprint": (
                "baseline_workload_fingerprint",
                "candidate_workload_fingerprint",
                "workload_fingerprint",
            ),
            "baseline_control_fingerprint": (
                "baseline_control_fingerprint",
                "candidate_control_fingerprint",
                "control_fingerprint",
            ),
        }
        for expected_field, fields in mutations.items():
            with self.subTest(field=expected_field):
                base = self._analysis()
                p1 = self._stable_p_analysis(recommended_experiments=[])
                for field in fields:
                    p1[field] = "a" * 64
                self._seal_analysis_content(p1)

                with self.assertRaisesRegex(ValueError, expected_field):
                    self._build_source_range_strategy({"S4": [base, p1]})

    def test_history_requires_a_fresh_analysis_agent_for_every_round(self):
        base = self._analysis()
        p1 = self._stable_p_analysis(recommended_experiments=[])
        p1["analysis_agent_id"] = base["analysis_agent_id"]
        self._rebind_analysis_identity(p1)

        with self.assertRaisesRegex(ValueError, "analysis_agent_id.*唯一"):
            self._build_source_range_strategy({"S4": [base, p1]})

    def test_prune_batch_requires_pairwise_machine_proven_non_overlap(self):
        cases = {
            "disjoint": (("model.py", 0, 10), ("model.py", 20, 30), 2),
            "adjacent": (("model.py", 0, 10), ("model.py", 10, 20), 2),
            "overlap": (("model.py", 0, 15), ("model.py", 10, 20), 1),
            "contained": (("model.py", 0, 30), ("model.py", 10, 20), 1),
            "same_boundary": (("model.py", 0, 10), ("model.py", 0, 10), 1),
            "cross_file": (("a.py", 0, 10), ("b.py", 0, 10), 1),
        }
        for label, (left, right, expected_count) in cases.items():
            with self.subTest(label=label):
                analysis = self._analysis(recommended_experiments=[])
                prune_items = [
                    item
                    for item in analysis["scope_actions"]
                    if item["action"] == "prune"
                ]
                decision_by_range = {
                    item["range_id"]: item for item in analysis["per_sk_decisions"]
                }
                for item, interval in zip(prune_items, (left, right)):
                    boundary = {
                        **item["boundary"],
                        "source_file": interval[0],
                        "start_offset": interval[1],
                        "end_offset": interval[2],
                    }
                    item["boundary"] = boundary
                    decision_by_range[item["range_id"]]["boundary"] = dict(boundary)
                self._seal_analysis_content(analysis)

                candidate = self._build_source_range_strategy({"S4": analysis})[
                    "candidates"
                ]["S4"]
                matrix = candidate["next_candidate_matrix"]

                self.assertEqual(len(matrix), 1)
                self.assertEqual(len(matrix[0]["target_range_ids"]), expected_count)
                if label == "cross_file":
                    self.assertTrue(
                        any(
                            blocker["kind"] == "prune_batch_unproven"
                            and "同一 source_file" in blocker["required_action_zh"]
                            for blocker in candidate["blockers"]
                        )
                    )

    def test_scope_actions_require_source_scope_and_boundary(self):
        for field in ("source_scope", "boundary"):
            with self.subTest(field=field):
                analysis = self._analysis()
                analysis["scope_actions"][1].pop(field)
                self._seal_analysis_content(analysis)
                with self.assertRaisesRegex(ValueError, field):
                    self._build_strategy({"S4": analysis})

    def test_dcci_recommendation_is_diagnostic_only_after_stage_o(self):
        dcci = {
            "experiment_id": "invalid-option-regressed-dcci",
            "range_id": "range-regressed",
            "only_change": {
                "option": "dcci_before_kernel_start",
                "value": [".*GroupedMatmul.*"],
            },
            "only_change_zh": "只修改选项 dcci_before_kernel_start 的目标正则。",
            "option": "dcci_before_kernel_start",
            "value": [".*GroupedMatmul.*"],
            "accepted_evidence": {
                "runtime_state": {
                    "state": "enabled",
                    "sources": [
                        {
                            "source": "candidate_config.runtime_evidence.dcci_state",
                            "state": "enabled",
                        }
                    ],
                },
                "option_acceptance": {
                    "source": (
                        "environment_evidence.options.optimize_options."
                        "dcci_before_kernel_start.accepted_values"
                    ),
                    "accepted_value": [".*GroupedMatmul.*"],
                },
            },
            "expected_signal_zh": "目标范围区间耗时改善。",
            "lifecycle_gates": ["correctness", "read_only_reanalysis"],
        }
        report = self._build_source_range_strategy(
            {"S4": self._completed_history([dcci])}
        )
        option_trials = [
            item
            for item in report["candidates"]["S4"]["next_candidate_matrix"]
            if item.get("options")
        ]
        self.assertEqual(option_trials, [])
        self.assertTrue(
            any(
                item["kind"] == "post_base_option_recommendations_ignored"
                for item in report["candidates"]["S4"]["blockers"]
            )
        )

    def test_malformed_or_conflicting_recommendation_becomes_blocker(self):
        fixtures = []
        missing_chinese = self._analysis()["recommended_experiments"][0]
        missing_chinese.pop("only_change_zh")
        fixtures.append(missing_chinese)
        conflict = self._analysis()["recommended_experiments"][0]
        conflict["only_change"]["value"] = 0
        fixtures.append(conflict)
        two_changes = self._analysis()["recommended_experiments"][0]
        two_changes["only_change"] = {
            "option": "auto_op_parallel",
            "value": 1,
            "other": True,
        }
        fixtures.append(two_changes)

        for recommendation in fixtures:
            with self.subTest(recommendation=recommendation):
                candidate = self._build_source_range_strategy(
                    {"S4": self._completed_history([recommendation])}
                )["candidates"]["S4"]
                self.assertFalse(
                    any(
                        item.get("options")
                        for item in candidate["next_candidate_matrix"]
                    )
                )
                self.assertTrue(candidate["blockers"])
                self.assertTrue(
                    all("required_action_zh" in item for item in candidate["blockers"])
                )

    def test_scope_action_conflict_is_rejected_instead_of_repaired(self):
        analysis = self._analysis()
        analysis["scope_actions"][1]["action"] = "keep"
        self._seal_analysis_content(analysis)
        with self.assertRaisesRegex(ValueError, "scope_actions.*per_sk_decisions"):
            self._build_strategy({"S4": analysis})

    def test_analysis_requires_complete_identity_fingerprints_and_inputs(self):
        required = (
            "analysis_id",
            "analysis_content_fingerprint",
            "experiment_id",
            "round_id",
            "analysis_agent_id",
            "source_revision",
            "baseline_profile_fingerprint",
            "candidate_profile_fingerprint",
            "baseline_config_fingerprint",
            "candidate_config_fingerprint",
            "baseline_workload_fingerprint",
            "candidate_workload_fingerprint",
            "workload_fingerprint",
            "baseline_control_fingerprint",
            "candidate_control_fingerprint",
            "control_fingerprint",
            "declared_change_set",
            "inputs",
            "per_sk_decisions",
        )
        for field in required:
            with self.subTest(field=field):
                analysis = self._analysis()
                analysis.pop(field)
                with self.assertRaisesRegex(ValueError, field):
                    self._build_strategy({"S4": analysis})

        analysis = self._analysis()
        analysis["analysis_id"] = "analysis-forged"
        with self.assertRaisesRegex(ValueError, "analysis_id"):
            self._build_strategy({"S4": analysis})

        analysis = self._analysis()
        analysis["inputs"]["candidate_profile"] = None
        with self.assertRaisesRegex(ValueError, "inputs.candidate_profile"):
            self._build_strategy({"S4": analysis})

    def test_scope_actions_must_exactly_match_unique_per_sk_decisions(self):
        mutations = []
        forged = self._analysis()
        forged["scope_actions"][1]["range_id"] = "range-forged"
        self._seal_analysis_content(forged)
        mutations.append(forged)
        mismatched = self._analysis()
        mismatched["per_sk_decisions"][1]["classification"] = "beneficial"
        mismatched["per_sk_decisions"][1]["action"] = "keep"
        self._seal_analysis_content(mismatched)
        mutations.append(mismatched)
        duplicate_sk = self._analysis()
        duplicate_sk["per_sk_decisions"][2]["sk_id"] = "sk-neutral"
        duplicate_sk["scope_actions"][2]["sk_id"] = "sk-neutral"
        self._seal_analysis_content(duplicate_sk)
        mutations.append(duplicate_sk)

        for analysis in mutations:
            with self.subTest(analysis=analysis):
                with self.assertRaisesRegex(
                    ValueError, "scope_actions|per_sk_decisions"
                ):
                    self._build_strategy({"S4": analysis})

        minimal_forgery = {
            "schema_version": "1.2",
            "candidate_name": "S4",
            "scope_actions": self._analysis()["scope_actions"],
            "recommended_experiments": [],
        }
        with self.assertRaisesRegex(ValueError, "analysis_id"):
            self._build_strategy({"S4": minimal_forgery})

    def test_non_dcci_evidence_requires_exact_producer_source(self):
        for source in (None, "environment accepted values", "wrong.accepted_values"):
            with self.subTest(source=source):
                recommendation = self._analysis()["recommended_experiments"][0]
                recommendation["accepted_evidence"]["source"] = source
                candidate = self._build_source_range_strategy(
                    {"S4": self._completed_history([recommendation])}
                )["candidates"]["S4"]
                self.assertFalse(
                    any(
                        item.get("options")
                        for item in candidate["next_candidate_matrix"]
                    )
                )
                self.assertTrue(candidate["blockers"])

    def test_round_ids_advance_without_self_reference(self):
        fixtures = (
            ("S4-BASE", "S4-P1"),
            ("S4-P1", "S4-P2"),
            ("S4-P9", "S4-P10"),
        )
        for source, prune_id in fixtures:
            with self.subTest(source=source):
                matrix = self._build_source_range_strategy(
                    {"S4": self._analysis(round_id=source)}
                )["candidates"]["S4"]["next_candidate_matrix"]
                self.assertEqual([item["id"] for item in matrix], [prune_id])
                self.assertEqual(matrix[0]["comparison_to"], source)
                self.assertNotEqual(matrix[0]["id"], source)

        analysis = self._analysis(round_id="S4-P3")
        analysis["scope_actions"] = [
            action
            for action in analysis["scope_actions"]
            if action["classification"] in {"beneficial", "regressed"}
        ]
        analysis["per_sk_decisions"] = [
            decision
            for decision in analysis["per_sk_decisions"]
            if decision["classification"] in {"beneficial", "regressed"}
        ]
        self._seal_analysis_content(analysis)
        matrix = self._build_source_range_strategy({"S4": analysis})["candidates"][
            "S4"
        ]["next_candidate_matrix"]
        self.assertEqual([item["id"] for item in matrix], ["S4-P4"])

        for invalid in ("S4", "S4-P0", "Other-BASE", "S4-X1", "S4-FINAL-more"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "round_id"):
                    self._build_source_range_strategy(
                        {"S4": self._analysis(round_id=invalid)}
                    )

    def test_conflicting_or_duplicate_recommendations_block_all_related_entries(self):
        original = self._analysis()["recommended_experiments"][0]
        fixtures = []
        fixtures.append(
            [json.loads(json.dumps(original)), json.loads(json.dumps(original))]
        )

        same_source = json.loads(json.dumps(original))
        same_source["value"] = 0
        same_source["only_change"]["value"] = 0
        same_source["accepted_evidence"]["accepted_value"] = 0
        fixtures.append([original, same_source])

        conflicting_value = json.loads(json.dumps(same_source))
        conflicting_value["experiment_id"] = "different-source-id"
        fixtures.append([original, conflicting_value])

        semantic_duplicate = json.loads(json.dumps(original))
        semantic_duplicate["experiment_id"] = "different-source-id"
        fixtures.append([original, semantic_duplicate])

        malformed_same_source = json.loads(json.dumps(original))
        malformed_same_source["only_change"]["other"] = True
        fixtures.append([original, malformed_same_source])

        malformed_conflicting_value = json.loads(json.dumps(conflicting_value))
        malformed_conflicting_value["only_change"].pop("option")
        fixtures.append([original, malformed_conflicting_value])

        for recommendations in fixtures:
            with self.subTest(recommendations=recommendations):
                candidate = self._build_source_range_strategy(
                    {"S4": self._completed_history(recommendations)}
                )["candidates"]["S4"]
                self.assertFalse(
                    any(
                        item.get("options")
                        for item in candidate["next_candidate_matrix"]
                    )
                )
                self.assertTrue(
                    any(
                        item["kind"] == "post_base_option_recommendations_ignored"
                        for item in candidate["blockers"]
                    )
                )

    def test_blockers_and_json_values_are_strictly_valid(self):
        invalid_blockers = (None, {}, [""], [1], [{}])
        for blockers in invalid_blockers:
            with self.subTest(blockers=blockers):
                analysis = self._analysis()
                analysis["blockers"] = blockers
                self._seal_analysis_content(analysis)
                with self.assertRaisesRegex(ValueError, "blockers"):
                    self._build_strategy({"S4": analysis})

        analysis = self._analysis()
        analysis["recommended_experiments"][0]["value"] = float("nan")
        with self.assertRaisesRegex(ValueError, "有限|JSON"):
            self._build_strategy({"S4": analysis})

    def test_cli_rejects_nonstandard_json_and_path_conflicts_without_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "analysis.json"
            source.write_text(json.dumps(self._analysis()))
            bad = root / "bad.json"
            bad.write_text('{"schema_version": NaN}')

            fixtures = (
                ["--profiling-analysis", f"S4={bad}"],
                [
                    "--profiling-analysis",
                    f"S4={source}",
                    "--json-out",
                    str(source),
                ],
                [
                    "--profiling-analysis",
                    f"S4={source}",
                    "--json-out",
                    str(root / "same.out"),
                    "--markdown-out",
                    str(root / "same.out"),
                ],
                [
                    "--profiling-analysis",
                    f"S4={root / 'missing.json'}",
                ],
            )
            for argv in fixtures:
                with self.subTest(argv=argv):
                    stderr = io.StringIO()
                    with redirect_stderr(stderr), self.assertRaises(SystemExit):
                        recommend_sk_strategy.main(argv)
                    self.assertNotIn("Traceback", stderr.getvalue())

            self.assertEqual(
                json.loads(source.read_text())["analysis_id"],
                self._analysis()["analysis_id"],
            )

    def test_deep_json_is_rejected_without_recursion_traceback(self):
        analysis = self._analysis()
        nested = []
        cursor = nested
        for _ in range(1200):
            child = []
            cursor.append(child)
            cursor = child
        analysis["blockers"] = [nested]

        with self.assertRaisesRegex(ValueError, "嵌套|递归"):
            self._build_strategy({"S4": analysis})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deep.json"
            path.write_text(
                '{"schema_version":"1.0","nested":'
                + "[" * 1200
                + "0"
                + "]" * 1200
                + "}"
            )
            stderr = io.StringIO()

            with redirect_stderr(stderr), self.assertRaises(SystemExit):
                recommend_sk_strategy.main(["--profiling-analysis", f"S4={path}"])

            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertRegex(stderr.getvalue(), "嵌套|递归")

    def test_output_write_is_transactional_and_reports_errors_without_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "analysis.json"
            source.write_text(json.dumps(self._analysis()))
            json_out = root / "strategy.json"
            markdown_out = root / "strategy.md"
            json_out.write_text("old-json\n")
            markdown_out.write_text("old-markdown\n")
            real_replace = recommend_sk_strategy.os.replace

            def fail_json_replace(source_path, target_path):
                if Path(target_path) == json_out:
                    raise OSError("simulated replace failure")
                return real_replace(source_path, target_path)

            stderr = io.StringIO()
            with (
                mock.patch.object(
                    recommend_sk_strategy.os, "replace", side_effect=fail_json_replace
                ),
                redirect_stderr(stderr),
                self.assertRaises(SystemExit),
            ):
                recommend_sk_strategy.main(
                    [
                        "--profiling-analysis",
                        f"S4={source}",
                        "--json-out",
                        str(json_out),
                        "--markdown-out",
                        str(markdown_out),
                    ]
                )
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertEqual(json_out.read_text(), "old-json\n")
            self.assertEqual(markdown_out.read_text(), "old-markdown\n")
            self.assertFalse(list(root.glob(".*.tmp")))

    def test_old_or_missing_schema_has_chinese_migration_error(self):
        for analysis in ({"fusion_performance": {}}, {"schema_version": "0.9"}):
            with self.subTest(analysis=analysis):
                with self.assertRaisesRegex(ValueError, "只支持.*schema_version=1.2"):
                    self._build_strategy({"S4": analysis})

    def test_input_name_must_match_report_candidate(self):
        analysis = self._analysis()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            second = root / "second.json"
            first.write_text(json.dumps(analysis))
            second.write_text(json.dumps(analysis))

            stderr = io.StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit):
                recommend_sk_strategy.main(
                    [
                        "--profiling-analysis",
                        f"Wrong={first}",
                    ]
                )
            self.assertIn("candidate_name", stderr.getvalue())

            duplicate_candidate = json.loads(json.dumps(analysis))
            duplicate_candidate["candidate_name"] = "S4"
            second.write_text(json.dumps(duplicate_candidate))
            stderr = io.StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit):
                recommend_sk_strategy.main(
                    [
                        "--profiling-analysis",
                        f"S4={first}",
                        "--profiling-analysis",
                        f"S5={second}",
                    ]
                )
            self.assertIn("candidate_name", stderr.getvalue())

    def test_removed_profiler_alias_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current, _ = self._write_real_analysis_fixture(root)
            stderr = io.StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit):
                recommend_sk_strategy.main(["--profiler-json", str(current)])
            self.assertIn("unrecognized arguments", stderr.getvalue())

    def test_markdown_is_chinese_and_escapes_dynamic_values(self):
        analysis = self._analysis()
        analysis["candidate_name"] = "S4|候选<script>&\n# 标题"
        analysis["round_id"] = f"{analysis['candidate_name']}-BASE"
        analysis["analysis_agent_id"] = "profiling-agent-html"
        self._rebind_analysis_identity(analysis)
        report = self._build_source_range_strategy(
            {analysis["candidate_name"]: analysis}
        )
        markdown = recommend_sk_strategy._markdown(report)

        self.assertIn("# SuperKernel 性能范围实验计划", markdown)
        self.assertIn("性能裁剪", markdown)
        self.assertNotIn("挽救实验", markdown)
        self.assertIn(r"S4\|候选&lt;script&gt;&amp; # 标题", markdown)
        self.assertNotIn("<script>", markdown)
        self.assertNotIn(">=5", markdown)

    def test_markdown_is_stable_across_candidate_and_recommendation_input_order(self):
        first = self._analysis()
        second_recommendation = json.loads(
            json.dumps(first["recommended_experiments"][0])
        )
        second_recommendation.update(
            {
                "experiment_id": "invalid-option-neutral-aggressive",
                "range_id": "range-neutral",
                "option": "aggressive",
                "value": True,
                "only_change": {"option": "aggressive", "value": True},
                "only_change_zh": "只修改选项 aggressive 为 true。",
                "accepted_evidence": {
                    "source": (
                        "environment_evidence.options.optimize_options."
                        "aggressive.accepted_values"
                    ),
                    "accepted_value": True,
                },
            }
        )
        first["recommended_experiments"].append(second_recommendation)
        self._seal_analysis_content(first)

        second = self._analysis()
        second["candidate_name"] = "S5"
        second["experiment_id"] = "S5"
        second["round_id"] = "S5-BASE"
        second["analysis_agent_id"] = "profiling-agent-S5-BASE"
        self._rebind_analysis_identity(second)

        left = self._build_source_range_strategy({"S5": second, "S4": first})
        reordered_first = json.loads(json.dumps(first))
        reordered_first["recommended_experiments"].reverse()
        self._seal_analysis_content(reordered_first)
        right = self._build_source_range_strategy({"S4": reordered_first, "S5": second})

        self.assertEqual(
            recommend_sk_strategy._markdown(left),
            recommend_sk_strategy._markdown(right),
        )

    def test_analysis_content_fingerprint_rejects_coordinated_tampering(self):
        mutations = []

        decisions = self._analysis()
        for field in ("per_sk_decisions", "scope_actions"):
            decisions[field][2]["classification"] = "beneficial"
            decisions[field][2]["action"] = "keep"
        mutations.append(decisions)

        scope = self._analysis()
        scope["scope_actions"][1]["range_id"] = "range-forged"
        mutations.append(scope)

        fingerprints = self._analysis()
        fingerprints["baseline_profile_fingerprint"] = "f" * 64
        mutations.append(fingerprints)

        for analysis in mutations:
            with self.subTest(analysis=analysis):
                with self.assertRaisesRegex(ValueError, "analysis_content_fingerprint"):
                    self._build_strategy({"S4": analysis})

    def test_decision_order_is_canonical_for_prune_plan_and_markdown(self):
        first = self._analysis()
        second = json.loads(json.dumps(first))
        second["per_sk_decisions"].reverse()
        second["scope_actions"].reverse()
        self._seal_analysis_content(second)

        left = self._build_source_range_strategy({"S4": first})
        right = self._build_source_range_strategy({"S4": second})

        self.assertEqual(left, right)
        self.assertEqual(
            recommend_sk_strategy._markdown(left),
            recommend_sk_strategy._markdown(right),
        )

    def test_cli_revalidates_original_analysis_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis_path, artifacts = self._write_real_analysis_fixture(root)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    recommend_sk_strategy.main(
                        ["--profiling-analysis", f"S4={analysis_path}"]
                    ),
                    0,
                )
            config_before = artifacts["candidate_config"].read_text()
            stderr = io.StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                recommend_sk_strategy.main(
                    [
                        "--profiling-analysis",
                        f"S4={analysis_path}",
                        "--json-out",
                        str(artifacts["candidate_config"]),
                    ]
                )
            self.assertEqual(raised.exception.code, 2)
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertEqual(artifacts["candidate_config"].read_text(), config_before)

        mutations = (
            "profile_changed",
            "config_changed",
            "workload_missing",
            "declared_changed",
        )
        for mutation in mutations:
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                analysis_path, artifacts = self._write_real_analysis_fixture(root)
                if mutation == "profile_changed":
                    artifacts["candidate_profile"].write_bytes(b"changed\n")
                elif mutation == "config_changed":
                    artifacts["candidate_config"].write_text('{"changed":true}')
                elif mutation == "workload_missing":
                    artifacts["baseline_workload"].unlink()
                else:
                    artifacts["declared_change_set"].write_text(
                        '{"allowed_json_pointers":[]}'
                    )

                stderr = io.StringIO()
                with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                    recommend_sk_strategy.main(
                        ["--profiling-analysis", f"S4={analysis_path}"]
                    )
                self.assertEqual(raised.exception.code, 2)
                self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
