#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Summarize multi-rank inference logs and compare SuperKernel candidates."""

import argparse
import hashlib
import json
import re
import statistics
from decimal import Decimal
from pathlib import Path

import yaml

from analyze_sk_meta import validate_replay_report


TIMING = re.compile(r"Inference time \((decode|prefill)\): ([0-9.]+) ms")
RANK_LOG = re.compile(r"log_(\d+)\.log$")
S0_REQUIRED_RUNS = 5
S0_MAX_SPREAD_PCT = 5.0
CANDIDATE_MIN_RUNS = 3
OPTION_TRIAL_MIN_IMPROVEMENT_PCT = 0.0
PROMOTION_MODES = {"range_optimized", "whole_scope"}
SCREENING_MATRIX_SCHEMA = "superkernel-screening-matrix-v2"
REQUIRED_SCREENING_STRATEGIES = {
    "automatic_aot",
    "broad_decode",
    "per_block",
    "semantic_segment",
}
SETTLED_SCREENING_STATUSES = {"executed", "blocked", "skipped"}
FORBIDDEN_EARLY_STOP_BLOCKER_CODES = {
    "eligible_candidate_exists",
    "winner_already_found",
    "early_stop_after_eligible",
    "deferred_until_no_eligible",
}
STAGE_A_EMPTY_OPTION_FIELDS = (
    "super_kernel_optimize_options",
    "super_kernel_debug_options",
)


def _canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value):
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _required_text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"screening matrix {field} must be a non-empty string")
    return value.strip()


def validate_screening_matrix(matrix, candidate_names):
    """Fail closed unless every planned Stage-A strategy is explicitly settled."""
    if not isinstance(matrix, dict):
        raise ValueError("screening matrix must be a JSON object")
    if matrix.get("schema_version") != SCREENING_MATRIX_SCHEMA:
        raise ValueError(
            "screening matrix schema_version must be " + SCREENING_MATRIX_SCHEMA
        )
    if matrix.get("candidate_set_frozen_before_execution") is not True:
        raise ValueError(
            "screening matrix must declare candidate_set_frozen_before_execution=true"
        )
    entries = matrix.get("candidates")
    if not isinstance(entries, list) or not entries:
        raise ValueError("screening matrix candidates must be a non-empty list")

    ids = set()
    covered_strategies = set()
    executed_ids = set()
    settled_nonexecuted = []
    for index, entry in enumerate(entries):
        prefix = f"candidates[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"screening matrix {prefix} must be an object")
        candidate_id = _required_text(entry.get("id"), f"{prefix}.id")
        if candidate_id in ids:
            raise ValueError(
                f"screening matrix has duplicate candidate id: {candidate_id}"
            )
        ids.add(candidate_id)
        strategy_kind = _required_text(
            entry.get("strategy_kind"), f"{prefix}.strategy_kind"
        )
        covered_strategies.add(strategy_kind)
        only_change = _required_text(entry.get("only_change"), f"{prefix}.only_change")
        status = _required_text(entry.get("status"), f"{prefix}.status")
        if status not in SETTLED_SCREENING_STATUSES:
            choices = ", ".join(sorted(SETTLED_SCREENING_STATUSES))
            raise ValueError(
                f"screening matrix {candidate_id} is not settled; "
                f"status must be one of: {choices}"
            )

        normalized_entry = {
            "id": candidate_id,
            "strategy_kind": strategy_kind,
            "only_change": only_change,
            "status": status,
        }
        if status == "executed":
            if entry.get("blocker") not in (None, {}):
                raise ValueError(
                    f"screening matrix executed candidate {candidate_id} must not have a blocker"
                )
            executed_ids.add(candidate_id)
        else:
            blocker = entry.get("blocker")
            if not isinstance(blocker, dict):
                raise ValueError(
                    f"screening matrix {status} candidate {candidate_id} requires blocker evidence"
                )
            code = _required_text(blocker.get("code"), f"{prefix}.blocker.code")
            reason_zh = _required_text(
                blocker.get("reason_zh"), f"{prefix}.blocker.reason_zh"
            )
            evidence = _required_text(
                blocker.get("evidence"), f"{prefix}.blocker.evidence"
            )
            reason_lower = reason_zh.lower()
            eligibility_early_stop = (
                "eligible" in reason_lower
                or "winner" in reason_lower
                or "已有候选" in reason_zh
                or "已有胜出" in reason_zh
                or "已有赢家" in reason_zh
            )
            if code in FORBIDDEN_EARLY_STOP_BLOCKER_CODES or eligibility_early_stop:
                raise ValueError(
                    f"screening matrix {candidate_id} uses an illegal early-stop blocker; "
                    "an existing eligible candidate or winner cannot skip a planned strategy"
                )
            normalized_entry["blocker"] = {
                "code": code,
                "reason_zh": reason_zh,
                "evidence": evidence,
            }
            settled_nonexecuted.append(normalized_entry)
    missing_strategies = sorted(REQUIRED_SCREENING_STRATEGIES - covered_strategies)
    if missing_strategies:
        raise ValueError(
            "screening matrix missing required strategy kinds: "
            + ", ".join(missing_strategies)
        )

    actual_names = set(candidate_names)
    missing_candidates = sorted(executed_ids - actual_names)
    extra_candidates = sorted(actual_names - executed_ids)
    if missing_candidates or extra_candidates:
        details = []
        if missing_candidates:
            details.append(
                "missing executed candidates: " + ", ".join(missing_candidates)
            )
        if extra_candidates:
            details.append("undeclared candidates: " + ", ".join(extra_candidates))
        raise ValueError(
            "screening candidate set mismatch (" + "; ".join(details) + ")"
        )

    return {
        "schema_version": SCREENING_MATRIX_SCHEMA,
        "content_fingerprint": _sha256_json(matrix),
        "candidate_set_frozen_before_execution": True,
        "required_strategy_kinds": sorted(REQUIRED_SCREENING_STRATEGIES),
        "covered_strategy_kinds": sorted(covered_strategies),
        "planned_candidate_ids": sorted(ids),
        "executed_candidate_ids": sorted(executed_ids),
        "nonexecuted_candidates": sorted(
            settled_nonexecuted, key=lambda item: item["id"]
        ),
        "settlement_complete": True,
    }


def validate_stage_a_config(config_path):
    """Require explicit empty option maps for an SK screening candidate."""
    config_path = Path(config_path)
    try:
        raw = config_path.read_bytes()
    except OSError as error:
        raise ValueError(
            f"Stage-A config cannot be read: {config_path}: {error}"
        ) from error
    try:
        config = yaml.safe_load(raw)
    except yaml.YAMLError as error:
        raise ValueError(
            f"Stage-A config is not valid YAML: {config_path}: {error}"
        ) from error
    if not isinstance(config, dict):
        raise ValueError(f"Stage-A config must be a YAML mapping: {config_path}")
    model_config = config.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError(
            f"Stage-A config model_config must be a mapping: {config_path}"
        )
    custom_params = model_config.get("custom_params")
    if not isinstance(custom_params, dict):
        raise ValueError(
            f"Stage-A config model_config.custom_params must be a mapping: {config_path}"
        )

    for field in STAGE_A_EMPTY_OPTION_FIELDS:
        if field not in custom_params:
            raise ValueError(
                f"Stage-A config must explicitly define {field}: {{}}: {config_path}"
            )
        value = custom_params[field]
        if not isinstance(value, dict) or value:
            raise ValueError(
                f"Stage-A config {field} must be an explicitly empty mapping: {config_path}"
            )

    return {
        "path": str(config_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "required_empty_fields": list(STAGE_A_EMPTY_OPTION_FIELDS),
        "explicitly_empty": True,
    }


def validate_stage_a_candidate_configs(candidates):
    """Revalidate every archived clean-run config before Stage-A selection."""
    validated = {}
    for name, candidate_path in candidates.items():
        candidate_path = Path(candidate_path)
        run_paths = sorted(
            item for item in candidate_path.glob("run-*") if item.is_dir()
        )
        if not run_paths:
            raise ValueError(f"{candidate_path}: no run-* directories found")
        run_configs = []
        for run_path in run_paths:
            config_path = run_path / "config.yaml"
            if not config_path.is_file():
                raise ValueError(
                    f"Stage-A clean run must archive config.yaml: {run_path}"
                )
            run_configs.append(validate_stage_a_config(config_path))
        validated[name] = {
            "candidate_path": str(candidate_path),
            "run_count": len(run_configs),
            "run_configs": run_configs,
            "explicitly_empty": True,
        }
    return {
        "policy": "explicit_empty_option_maps",
        "required_empty_fields": list(STAGE_A_EMPTY_OPTION_FIELDS),
        "candidates": validated,
        "all_explicitly_empty": True,
    }


def parse_log(path):
    timings = {
        "decode_ms": [],
        "prefill_ms": [],
        "decode_exact_ms": [],
        "prefill_exact_ms": [],
    }
    for phase, value in TIMING.findall(Path(path).read_text(errors="replace")):
        exact_value = Decimal(value)
        timings[f"{phase}_ms"].append(float(exact_value))
        timings[f"{phase}_exact_ms"].append(exact_value)
    return timings


def _quantile(values, probability):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _statistics(values):
    if not values:
        raise ValueError("cannot summarize an empty sample set")
    return {
        "count": len(values),
        "mean_ms": statistics.mean(values),
        "p50_ms": statistics.median(values),
        "p90_ms": _quantile(values, 0.9),
        "stddev_ms": statistics.pstdev(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "samples_ms": list(values),
    }


def summarize_run(path, warmup=8, expected_ranks=8):
    path = Path(path)
    rank_logs = []
    for log_path in path.glob("log_*.log"):
        match = RANK_LOG.search(log_path.name)
        if match:
            rank_logs.append((int(match.group(1)), log_path))
    rank_logs.sort()
    if len(rank_logs) != expected_ranks:
        raise ValueError(
            f"{path}: expected {expected_ranks} rank logs, found {len(rank_logs)}"
        )

    ranks = {}
    all_decode = []
    all_prefill = []
    for rank, log_path in rank_logs:
        parsed = parse_log(log_path)
        decode = parsed["decode_ms"][warmup:]
        decode_exact = parsed["decode_exact_ms"][warmup:]
        if not decode:
            raise ValueError(
                f"{log_path}: no decode samples remain after warmup={warmup}"
            )
        ranks[str(rank)] = _statistics(decode)
        ranks[str(rank)]["mean_exact_ms"] = str(
            sum(decode_exact, Decimal("0")) / len(decode_exact)
        )
        all_decode.extend(decode)
        all_prefill.extend(parsed["prefill_ms"])

    worst_rank = max(ranks, key=lambda rank: ranks[rank]["mean_ms"])
    return {
        "path": str(path),
        "rank_count": len(ranks),
        "warmup": warmup,
        "ranks": ranks,
        "decode": _statistics(all_decode),
        "prefill": _statistics(all_prefill) if all_prefill else None,
        "worst_rank": int(worst_rank),
        "worst_rank_mean_ms": ranks[worst_rank]["mean_ms"],
        "worst_rank_mean_exact_ms": ranks[worst_rank]["mean_exact_ms"],
    }


def summarize_candidate(path, warmup=8, expected_ranks=8):
    path = Path(path)
    run_paths = sorted(item for item in path.glob("run-*") if item.is_dir())
    if not run_paths:
        raise ValueError(f"{path}: no run-* directories found")
    runs = [
        summarize_run(run_path, warmup=warmup, expected_ranks=expected_ranks)
        for run_path in run_paths
    ]
    worst_rank_means = [run["worst_rank_mean_ms"] for run in runs]
    worst_rank_exact_means = [run["worst_rank_mean_exact_ms"] for run in runs]
    all_decode = [
        sample
        for run in runs
        for rank in run["ranks"].values()
        for sample in rank["samples_ms"]
    ]
    return {
        "path": str(path),
        "run_count": len(runs),
        "runs": runs,
        "decode": _statistics(all_decode),
        "run_worst_rank_means_ms": worst_rank_means,
        "run_worst_rank_means_exact_ms": worst_rank_exact_means,
        "median_run_worst_rank_mean_ms": statistics.median(worst_rank_means),
        "worst_rank_mean_ms": statistics.mean(worst_rank_means),
    }


def _relative_change_pct(new_value, old_value):
    if old_value == 0:
        return None
    return (new_value - old_value) / old_value * 100


def _improvement_pct(baseline_value, candidate_value):
    if baseline_value == 0:
        return None
    return (baseline_value - candidate_value) / baseline_value * 100


def evaluate_baseline_stability(baseline_summary):
    values = [
        float(value) for value in baseline_summary.get("run_worst_rank_means_ms", [])
    ]
    exact_values = baseline_summary.get("run_worst_rank_means_exact_ms")
    if exact_values is None:
        decimal_values = [Decimal(str(value)) for value in values]
    else:
        try:
            decimal_values = [Decimal(value) for value in exact_values]
        except (TypeError, ValueError, ArithmeticError):
            decimal_values = []
    spread_ms = (
        float(max(decimal_values) - min(decimal_values)) if decimal_values else None
    )
    mean_ms = (
        sum(decimal_values, Decimal("0")) / len(decimal_values)
        if decimal_values
        else None
    )
    spread_pct = (
        float((max(decimal_values) - min(decimal_values)) / mean_ms * Decimal("100"))
        if decimal_values and mean_ms != 0
        else None
    )
    checks = {
        "required_run_count": len(values) == S0_REQUIRED_RUNS,
        "max_spread_pct": (spread_pct is not None and spread_pct <= S0_MAX_SPREAD_PCT),
    }
    return {
        "stable": all(checks.values()),
        "checks": checks,
        "required_run_count": S0_REQUIRED_RUNS,
        "max_spread_pct": S0_MAX_SPREAD_PCT,
        "run_worst_rank_means_ms": values,
        "spread_ms": spread_ms,
        "mean_ms": float(mean_ms) if mean_ms is not None else None,
        "spread_pct": spread_pct,
    }


def _lte_or_false(value, threshold):
    return value is not None and value <= threshold


def evaluate_promotion(
    baseline_summary,
    candidate_summary,
    *,
    min_improvement_pct=2.0,
    allow_p90_regression_pct=0.0,
    allow_stddev_regression_pct=0.0,
    replay_evidence=None,
    candidate_name=None,
    promotion_mode="range_optimized",
    require_strict_positive_mean_gain=False,
):
    if promotion_mode not in PROMOTION_MODES:
        raise ValueError(f"invalid promotion mode: {promotion_mode}")
    authoritative_stability = evaluate_baseline_stability(baseline_summary)
    baseline_stable = authoritative_stability["stable"]
    if promotion_mode == "whole_scope":
        replay_validation = {
            "valid": True,
            "required": False,
            "status": "not_required",
            "failure_reason": None,
        }
    else:
        replay_validation = validate_replay_report(
            replay_evidence, expected_candidate=candidate_name
        )
    mean_improvement = _improvement_pct(
        baseline_summary["worst_rank_mean_ms"],
        candidate_summary["worst_rank_mean_ms"],
    )
    median_run_improvement = _improvement_pct(
        baseline_summary["median_run_worst_rank_mean_ms"],
        candidate_summary["median_run_worst_rank_mean_ms"],
    )
    p90_regression = _relative_change_pct(
        candidate_summary["decode"]["p90_ms"],
        baseline_summary["decode"]["p90_ms"],
    )
    stddev_regression = _relative_change_pct(
        candidate_summary["decode"]["stddev_ms"],
        baseline_summary["decode"]["stddev_ms"],
    )
    mean_improvement_passed = mean_improvement is not None and (
        mean_improvement > 0.0
        if require_strict_positive_mean_gain
        else mean_improvement >= min_improvement_pct
    )
    checks = {
        "baseline_stable": baseline_stable,
        "baseline_min_runs": baseline_summary["run_count"] == S0_REQUIRED_RUNS,
        "candidate_min_runs": candidate_summary["run_count"] >= CANDIDATE_MIN_RUNS,
        "deep_fusion_replay": replay_validation["valid"],
        "mean_improvement": mean_improvement_passed,
        "median_run_direction": (
            median_run_improvement is not None and median_run_improvement > 0
        ),
        "p90_no_material_regression": _lte_or_false(
            p90_regression, allow_p90_regression_pct
        ),
        "stddev_no_material_regression": _lte_or_false(
            stddev_regression, allow_stddev_regression_pct
        ),
    }
    return {
        "promoted": all(checks.values()),
        "promotion_mode": promotion_mode,
        "checks": checks,
        "baseline_required_runs": S0_REQUIRED_RUNS,
        "candidate_min_runs": CANDIDATE_MIN_RUNS,
        "min_improvement_pct": min_improvement_pct,
        "mean_improvement_rule": (
            "strictly_positive"
            if require_strict_positive_mean_gain
            else "at_least_threshold"
        ),
        "allow_p90_regression_pct": allow_p90_regression_pct,
        "allow_stddev_regression_pct": allow_stddev_regression_pct,
        "mean_improvement_pct": mean_improvement,
        "median_run_improvement_pct": median_run_improvement,
        "p90_regression_pct": p90_regression,
        "stddev_regression_pct": stddev_regression,
        "deep_fusion_replay": replay_validation,
    }


def compare_candidates(
    baseline,
    candidates,
    warmup=8,
    expected_ranks=8,
    min_improvement_pct=2.0,
    allow_p90_regression_pct=0.0,
    allow_stddev_regression_pct=0.0,
    replay_evidence=None,
    promotion_modes=None,
    selection_only=False,
    option_trial=False,
    screening_matrix=None,
):
    if selection_only and option_trial:
        raise ValueError("selection-only and option-trial are mutually exclusive")
    if selection_only and (replay_evidence or promotion_modes):
        raise ValueError(
            "selection-only screening does not accept replay evidence or promotion modes"
        )
    if option_trial and (replay_evidence or promotion_modes):
        raise ValueError(
            "option-trial does not accept replay evidence or promotion modes"
        )
    if option_trial and len(candidates) != 1:
        raise ValueError("option-trial requires exactly one candidate")
    effective_min_improvement_pct = (
        OPTION_TRIAL_MIN_IMPROVEMENT_PCT if option_trial else min_improvement_pct
    )
    if selection_only:
        decision_stage = "candidate_screening"
    elif option_trial:
        decision_stage = "winner_option_trial"
    else:
        decision_stage = "final_promotion"
    screening_matrix_validation = None
    stage_a_option_validation = None
    if selection_only:
        if screening_matrix is None:
            raise ValueError("selection-only screening requires --screening-matrix")
        screening_matrix_validation = validate_screening_matrix(
            screening_matrix, candidates
        )
        stage_a_option_validation = validate_stage_a_candidate_configs(candidates)
    elif screening_matrix is not None:
        raise ValueError("screening matrix is accepted only with selection-only")
    baseline_summary = summarize_candidate(
        baseline, warmup=warmup, expected_ranks=expected_ranks
    )
    baseline_stability = evaluate_baseline_stability(baseline_summary)
    result = {
        "baseline": baseline_summary,
        "baseline_stability": baseline_stability,
        "candidates": {},
    }
    if screening_matrix_validation is not None:
        result["screening_matrix"] = screening_matrix_validation
        result["stage_a_option_controls"] = stage_a_option_validation
    if not candidates:
        result["selection"] = {
            "stage": decision_stage,
            "best_superkernel_candidate": None,
            "recommended": "baseline",
            "selected_for_deep_analysis": None,
            "baseline_retained": True,
            "promoted_candidates": [],
            "screening_eligible_candidates": [],
            "ranked_eligible_candidates": [],
            "final_promotion_decision": (
                "deferred_until_winner_profiling"
                if selection_only
                else "baseline_retained"
            ),
        }
        return result
    if replay_evidence is None:
        replay_evidence = {}
    if not isinstance(replay_evidence, dict):
        raise ValueError("replay evidence must be a NAME=PATH mapping")
    candidate_names = set(candidates)
    if promotion_modes is None:
        promotion_modes = {}
    if not isinstance(promotion_modes, dict):
        raise ValueError("promotion modes must be a NAME=MODE mapping")
    extra_mode_names = sorted(set(promotion_modes) - candidate_names)
    invalid_modes = sorted(
        f"{name}={mode}"
        for name, mode in promotion_modes.items()
        if mode not in PROMOTION_MODES
    )
    if extra_mode_names or invalid_modes:
        details = []
        if extra_mode_names:
            details.append("unknown candidates: " + ", ".join(extra_mode_names))
        if invalid_modes:
            details.append("invalid modes: " + ", ".join(invalid_modes))
        raise ValueError("promotion mode mapping invalid (" + "; ".join(details) + ")")
    effective_modes = {
        name: (
            "whole_scope"
            if selection_only or option_trial
            else promotion_modes.get(name, "range_optimized")
        )
        for name in candidates
    }
    replay_required_names = {
        name for name, mode in effective_modes.items() if mode == "range_optimized"
    }
    evidence_names = set(replay_evidence)
    missing_names = sorted(replay_required_names - evidence_names)
    extra_names = sorted(evidence_names - replay_required_names)
    if missing_names or extra_names:
        details = []
        if missing_names:
            details.append("missing: " + ", ".join(missing_names))
        if extra_names:
            details.append("extra: " + ", ".join(extra_names))
        raise ValueError(
            "replay evidence mapping mismatch (" + "; ".join(details) + ")"
        )
    replay_validations = {
        name: validate_replay_report(replay_evidence[name], expected_candidate=name)
        for name in replay_required_names
    }
    invalid_replays = [
        f"{name}: {validation['failure_reason']}"
        for name, validation in replay_validations.items()
        if not validation["valid"]
    ]
    if invalid_replays:
        raise ValueError("replay evidence invalid (" + "; ".join(invalid_replays) + ")")
    baseline_mean = baseline_summary["worst_rank_mean_ms"]
    for name, path in candidates.items():
        summary = summarize_candidate(
            path, warmup=warmup, expected_ranks=expected_ranks
        )
        summary["improvement_pct"] = (
            (baseline_mean - summary["worst_rank_mean_ms"]) / baseline_mean * 100
        )
        promotion_mode = effective_modes[name]
        replay_validation = replay_validations.get(
            name,
            {
                "valid": True,
                "required": False,
                "status": "not_required",
                "failure_reason": None,
            },
        )
        evaluation = evaluate_promotion(
            baseline_summary,
            summary,
            min_improvement_pct=effective_min_improvement_pct,
            allow_p90_regression_pct=allow_p90_regression_pct,
            allow_stddev_regression_pct=allow_stddev_regression_pct,
            replay_evidence=replay_evidence.get(name),
            candidate_name=name,
            promotion_mode=promotion_mode,
            require_strict_positive_mean_gain=option_trial,
        )
        if selection_only:
            evaluation["eligible"] = evaluation.pop("promoted")
            evaluation.pop("promotion_mode")
            evaluation.pop("deep_fusion_replay")
            evaluation["checks"].pop("deep_fusion_replay")
            summary["screening_eligibility"] = evaluation
        elif option_trial:
            evaluation["accepted"] = evaluation.pop("promoted")
            evaluation.pop("promotion_mode")
            evaluation.pop("deep_fusion_replay")
            evaluation["checks"].pop("deep_fusion_replay")
            summary["option_trial_evaluation"] = evaluation
        else:
            summary["deep_fusion_replay"] = replay_validation
            summary["promotion_mode"] = promotion_mode
            summary["promotion"] = evaluation
        result["candidates"][name] = summary
    if selection_only:
        eligible = [
            (name, summary)
            for name, summary in result["candidates"].items()
            if summary["screening_eligibility"]["eligible"]
        ]
        promoted = []
    elif option_trial:
        eligible = [
            (name, summary)
            for name, summary in result["candidates"].items()
            if summary["option_trial_evaluation"]["accepted"]
        ]
        promoted = []
    else:
        promoted = [
            (name, summary)
            for name, summary in result["candidates"].items()
            if summary["promotion"]["promoted"]
        ]
        eligible = promoted
    best = None
    if result["candidates"]:
        best = min(
            result["candidates"].items(),
            key=lambda item: (item[1]["worst_rank_mean_ms"], item[0]),
        )
    recommended = "baseline"
    ranked_eligible = sorted(
        eligible,
        key=lambda item: (item[1]["worst_rank_mean_ms"], item[0]),
    )
    if eligible:
        recommended = ranked_eligible[0][0]
    ranked_candidates = sorted(
        result["candidates"],
        key=lambda name: (
            result["candidates"][name]["worst_rank_mean_ms"],
            name,
        ),
    )
    selection = {
        "stage": decision_stage,
        "best_superkernel_candidate": best[0] if best else None,
        "recommended": recommended,
        "selected_for_deep_analysis": recommended
        if selection_only and eligible
        else None,
        "baseline_retained": recommended == "baseline",
        "promoted_candidates": [name for name, _ in promoted],
        "screening_eligible_candidates": [name for name, _ in eligible]
        if selection_only
        else [],
        "ranked_eligible_candidates": (
            [name for name, _ in ranked_eligible] if selection_only else []
        ),
        "ranked_candidates": ranked_candidates,
        "final_promotion_decision": (
            "deferred_until_winner_profiling"
            if selection_only and eligible
            else (
                "not_applicable_option_trial"
                if option_trial
                else ("baseline_retained" if recommended == "baseline" else "promoted")
            )
        ),
    }
    if option_trial:
        selection["option_trial_accepted"] = bool(eligible)
        selection["retained_incumbent"] = recommended
    result["selection"] = selection
    return result


def _candidate_arg(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must use NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("candidate must use NAME=PATH")
    return name, Path(path)


def _promotion_mode_arg(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("promotion mode must use NAME=MODE")
    name, mode = value.split("=", 1)
    if not name or mode not in PROMOTION_MODES:
        choices = ", ".join(sorted(PROMOTION_MODES))
        raise argparse.ArgumentTypeError(
            f"promotion mode must use NAME=MODE where MODE is one of: {choices}"
        )
    return name, mode


def _candidate_mapping(items, option, candidate_names=None):
    mapping = {}
    for name, path in items:
        if name in mapping:
            raise ValueError(f"{option} has duplicate name: {name}")
        mapping[name] = path
    if candidate_names is not None:
        names = set(candidate_names)
        missing = sorted(names - set(mapping))
        extra = sorted(set(mapping) - names)
        if missing or extra:
            details = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if extra:
                details.append("extra: " + ", ".join(extra))
            raise ValueError(f"{option} mapping mismatch (" + "; ".join(details) + ")")
    return mapping


def _print_table(report):
    selection_only = report["selection"].get("stage") == "candidate_screening"
    option_trial = report["selection"].get("stage") == "winner_option_trial"
    decision_label = (
        "screening-eligible"
        if selection_only
        else "option-accepted"
        if option_trial
        else "promoted"
    )
    print(
        "candidate runs worst-rank-mean-ms p50-ms p90-ms stddev-ms "
        f"improvement {decision_label} "
        "baseline-stable baseline-spread-pct"
    )
    baseline = report["baseline"]
    baseline_stability = report["baseline_stability"]
    print(
        f"baseline {baseline['run_count']} {baseline['worst_rank_mean_ms']:.4f} "
        f"{baseline['decode']['p50_ms']:.4f} {baseline['decode']['p90_ms']:.4f} "
        f"{baseline['decode']['stddev_ms']:.4f} - - "
        f"{baseline_stability['stable']} {baseline_stability['spread_pct']}"
    )
    for name, summary in report["candidates"].items():
        decision = (
            summary["screening_eligibility"]["eligible"]
            if selection_only
            else (
                summary["option_trial_evaluation"]["accepted"]
                if option_trial
                else summary["promotion"]["promoted"]
            )
        )
        print(
            f"{name} {summary['run_count']} {summary['worst_rank_mean_ms']:.4f} "
            f"{summary['decode']['p50_ms']:.4f} {summary['decode']['p90_ms']:.4f} "
            f"{summary['decode']['stddev_ms']:.4f} {summary['improvement_pct']:.2f}% "
            f"{decision} - -"
        )
    selection = report["selection"]
    print(f"best-superkernel-candidate {selection['best_superkernel_candidate']}")
    print(f"recommended {selection['recommended']}")
    if selection_only:
        print(f"selected-for-deep-analysis {selection['selected_for_deep_analysis']}")
    if option_trial:
        print(f"option-trial-accepted {selection['option_trial_accepted']}")
        print(f"retained-incumbent {selection['retained_incumbent']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", action="append", default=[], type=_candidate_arg)
    parser.add_argument(
        "--replay-evidence",
        action="append",
        default=[],
        type=_candidate_arg,
        help="NAME=deep-fusion replay-report JSON for the same candidate",
    )
    parser.add_argument(
        "--promotion-mode",
        action="append",
        default=[],
        type=_promotion_mode_arg,
        help=(
            "NAME=range_optimized|whole_scope; range_optimized is the default and "
            "requires replay evidence"
        ),
    )
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help=(
            "rank clean S candidates and select one for later profiling; this does "
            "not make a final promotion decision"
        ),
    )
    parser.add_argument(
        "--option-trial",
        action="store_true",
        help=(
            "compare one winner option trial against a stable five-run incumbent; "
            "this is an incremental option decision, not final promotion"
        ),
    )
    parser.add_argument(
        "--screening-matrix",
        type=Path,
        help=(
            "superkernel-screening-matrix-v2 JSON; required with --selection-only "
            "and rejected for final promotion"
        ),
    )
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--expected-ranks", type=int, default=8)
    parser.add_argument(
        "--min-improvement-pct",
        type=float,
        default=2.0,
        help=(
            "minimum mean improvement for screening/final promotion; option trials "
            "always require strictly positive gain without a fixed percentage threshold"
        ),
    )
    parser.add_argument("--allow-p90-regression-pct", type=float, default=0.0)
    parser.add_argument("--allow-stddev-regression-pct", type=float, default=0.0)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    try:
        candidates = _candidate_mapping(args.candidate, "--candidate")
        promotion_modes = dict(args.promotion_mode)
        if len(promotion_modes) != len(args.promotion_mode):
            raise ValueError("--promotion-mode has duplicate candidate name")
        replay_evidence = _candidate_mapping(args.replay_evidence, "--replay-evidence")
        if args.selection_only and args.option_trial:
            raise ValueError(
                "--selection-only and --option-trial are mutually exclusive"
            )
        if args.selection_only and args.screening_matrix is None:
            raise ValueError("--selection-only requires --screening-matrix")
        if not args.selection_only and args.screening_matrix is not None:
            raise ValueError("--screening-matrix requires --selection-only")
        screening_matrix = None
        if args.screening_matrix is not None:
            try:
                screening_matrix = json.loads(args.screening_matrix.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"cannot load --screening-matrix {args.screening_matrix}: {error}"
                ) from error
    except ValueError as error:
        parser.error(str(error))

    try:
        report = compare_candidates(
            args.baseline,
            candidates,
            warmup=args.warmup,
            expected_ranks=args.expected_ranks,
            min_improvement_pct=args.min_improvement_pct,
            allow_p90_regression_pct=args.allow_p90_regression_pct,
            allow_stddev_regression_pct=args.allow_stddev_regression_pct,
            replay_evidence=replay_evidence,
            promotion_modes=promotion_modes,
            selection_only=args.selection_only,
            option_trial=args.option_trial,
            screening_matrix=screening_matrix,
        )
    except ValueError as error:
        parser.error(str(error))
    _print_table(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
