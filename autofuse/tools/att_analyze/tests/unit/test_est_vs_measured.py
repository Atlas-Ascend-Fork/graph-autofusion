#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Behavioral regression tests for the offline ATT pipe comparison CLI."""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[5]
SCRIPT = REPO / ".claude/skills/att-perf-selection-analysis/scripts/est_vs_measured.py"


def final_line(
    op="Fusion_0",
    source="runtime",
    mode="default",
    result=0,
    group=0,
    case=2,
    key=9,
    subcase="",
    estimate=200,
    framed=False,
):
    payload = '{"tile_m":64}'
    fields = (
        f'schema=1 source="{source}" selection_mode="{mode}" operator="{op}" '
        f"graph=0 result={result} group={group} case_id={case} tiling_key={key} "
        f'score=1 sub_case_tag="{subcase}" template="Template{case}" repr_kind="full_json" '
        f"pipe_estimates={json.dumps(json.dumps({'AIV_MTE2': estimate, 'M3': None, 'V': 0}))}"
    )
    if not framed:
        return f"[ATT][FINAL_TILING] {fields} tiling_repr={json.dumps(payload)}\n"
    metadata = f'id="frame" chunks=1 len={len(payload)} hash_alg=sha256 hash={hashlib.sha256(payload.encode()).hexdigest()}'
    return (
        f"[ATT][FINAL_TILING_BEGIN] {fields} {metadata}\n"
        f'[ATT][FINAL_TILING_CHUNK] id="frame" seq=0 data={json.dumps(payload)}\n'
        f"[ATT][FINAL_TILING_END] {metadata}\n"
    )


def profile_row(op="Fusion_0", total=1000, blocks=10, ratio="50%"):
    return {
        "Op Name": op,
        "aiv_total_cycles": total,
        "Block Num": blocks,
        "aiv_mte2_ratio": ratio,
        "aiv_vec_ratio": 0,
    }


def run_cli(tmp_path, logs, profiles, extra=()):
    args = [sys.executable, str(SCRIPT)]
    for i, content in enumerate(logs):
        path = tmp_path / f"run{i}.log"
        path.write_text(content, encoding="utf-8")
        args += ["--plog", str(path)]
    for i, rows in enumerate(profiles):
        path = tmp_path / f"op_summary{i}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        args += ["--op-summary", str(path)]
    output = tmp_path / "comparison.csv"
    completed = subprocess.run(
        args + list(extra) + ["-o", str(output)], capture_output=True, text=True
    )
    rows = []
    if output.exists():
        with output.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    return completed, rows


def test_runs_keep_final_estimates_and_complete_identity_separate(tmp_path):
    default = final_line(estimate=200)
    pgo = final_line(
        source="pgo", mode="pgo", result=1, case=3, key=17, subcase="tail", estimate=600
    )
    result, rows = run_cli(
        tmp_path, [default, pgo], [[profile_row()], [profile_row(total=2000)]]
    )
    assert result.returncode == 0, result.stderr
    mte2 = [r for r in rows if r["Pipe"] == "AIV_MTE2"]
    assert [(r["Run"], r["Estimate"], r["Measured"], r["Ratio"]) for r in mte2] == [
        ("1", "200", "50.0", "4.0"),
        ("2", "600", "100.0", "6.0"),
    ]
    assert mte2[1]["Result"] == "1"
    assert mte2[1]["Case"] == "3"
    assert mte2[1]["TilingKey"] == "17"
    assert mte2[1]["SubCaseTag"] == "tail"
    assert mte2[1]["FinalSource"] == "pgo"
    assert mte2[1]["SelectionMode"] == "pgo"
    assert mte2[1]["LogEvidence"] == f"{tmp_path}/run1.log:1"
    assert mte2[1]["ProfileEvidence"] == f"{tmp_path}/op_summary1.csv:2"


def test_candidates_are_never_assigned_final_measurements(tmp_path):
    content = (
        "[INFO] [Fusion_0] [PROF]The value of AIV_MTE2 is 1.2e+3 in graph1_result2_g3_R0.\n"
        "[Fusion_0][PROF]The objective value of the tiling data is 4e2 in graph0_result0_g0_2.\n"
        + final_line()
    )
    result, rows = run_cli(tmp_path, [content], [[profile_row()]])
    assert result.returncode == 0, result.stderr
    candidates = [r for r in rows if r["RecordKind"] == "candidate"]
    assert len(candidates) == 2
    assert candidates[0]["CaseLabel"] == "R0"
    assert candidates[0]["Graph"] == "1"
    assert candidates[0]["Estimate"] == "1200.0"
    assert candidates[1]["Pipe"] == "Objective"
    assert all(
        r["Status"] == "candidate_only" and r["Measured"] == "" for r in candidates
    )


def test_no_final_does_not_infer_dispatch(tmp_path):
    content = "[Fusion_0][PROF]The value of AIV_MTE2 is 10 in graph0_result0_g0_0.\n"
    result, rows = run_cli(tmp_path, [content], [[profile_row()]])
    assert result.returncode == 0
    assert len(rows) == 1
    assert rows[0]["Measured"] == ""
    assert "0 项可比较" in result.stdout


@pytest.mark.parametrize(
    "filters,expected_ops",
    [
        (("Fusion_A",), {"Fusion_A"}),
        (("_A", "_B"), {"Fusion_A", "Fusion_B"}),
        (("fusion",), set()),
        (("NoSuchOperator",), set()),
    ],
)
def test_operator_filters_apply_to_candidates_and_finals(
    tmp_path, filters, expected_ops
):
    ops = ("Fusion_A", "Fusion_B", "Other_C")
    log = "".join(
        f"[{op}][PROF]The value of AIV_MTE2 is 100 in graph0_result0_g0_2.\n"
        + final_line(op=op)
        for op in ops
    )
    extra = [arg for value in filters for arg in ("--op-filter", value)]
    result, rows = run_cli(tmp_path, [log], [[profile_row(op=op) for op in ops]], extra)
    assert result.returncode == 0, result.stderr
    assert {r["Operator"] for r in rows} == expected_ops
    assert len(rows) == 4 * len(expected_ops)
    if expected_ops:
        assert {r["RecordKind"] for r in rows} == {"candidate", "final"}
        compared = [
            r for r in rows if r["RecordKind"] == "final" and r["Pipe"] == "AIV_MTE2"
        ]
        assert len(compared) == len(expected_ops)
        assert all(r["Status"] == "ok" and r["Ratio"] == "4.0" for r in compared)
    else:
        assert "0 项可比较" in result.stdout
        assert "证据不足以比较估值与实测" in result.stderr
        with (tmp_path / "comparison.csv").open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            reader = csv.DictReader(stream)
            assert "Operator" in reader.fieldnames
            assert list(reader) == []


def test_nonfinite_candidate_estimate_is_not_assigned_measurements(tmp_path):
    log = (
        "[Fusion_0][PROF]The value of AIV_MTE2 is 1e309 in graph0_result0_g0_2.\n"
        + final_line()
    )
    result, rows = run_cli(tmp_path, [log], [[profile_row()]])
    assert result.returncode == 0, result.stderr
    candidate = next(r for r in rows if r["RecordKind"] == "candidate")
    assert candidate["Status"] == "invalid_candidate_estimate"
    assert candidate["Measured"] == candidate["Ratio"] == ""
    final = next(
        r for r in rows if r["RecordKind"] == "final" and r["Pipe"] == "AIV_MTE2"
    )
    assert final["Status"] == "ok" and final["Ratio"] == "4.0"


@pytest.mark.parametrize(
    "source,mode", [("candidate", "default"), ("runtime", "pgo"), ("pgo", "default")]
)
def test_unsupported_final_source_is_not_assigned_measurements(tmp_path, source, mode):
    result, rows = run_cli(
        tmp_path, [final_line(source=source, mode=mode)], [[profile_row()]]
    )
    assert result.returncode == 0, result.stderr
    assert len(rows) == 3
    assert all(r["Status"] == "unsupported_final_source" for r in rows)
    assert all(r["Measured"] == r["Ratio"] == "" for r in rows)
    assert all(r["FinalSource"] == source and r["SelectionMode"] == mode for r in rows)


@pytest.mark.parametrize(
    "suffix", [final_line(group=1), final_line(), final_line(subcase="another")]
)
def test_multigroup_repeated_or_different_subcase_finals_need_call_mapping(
    tmp_path, suffix
):
    result, rows = run_cli(tmp_path, [final_line() + suffix], [[profile_row()]])
    assert result.returncode == 0
    assert len(rows) == 6
    assert all(r["Measured"] == "" and r["Ratio"] == "" for r in rows)
    assert rows[0]["Status"] == "ambiguous_final"


def test_duplicate_profile_rows_are_not_combined_into_synthetic_maxima(tmp_path):
    result, rows = run_cli(
        tmp_path, [final_line()], [[profile_row(), profile_row(total=2000)]]
    )
    assert result.returncode == 0
    assert all(r["Status"] == "ambiguous_profile" and r["Measured"] == "" for r in rows)


def test_prefix_names_require_explicit_mapping(tmp_path):
    log = final_line(op="autofuse_9_X")
    profiles = [
        [profile_row(op="autofuse_90_X"), profile_row(op="autofuse_9__x_suffix")]
    ]
    result, rows = run_cli(tmp_path, [log], profiles)
    assert result.returncode == 0
    assert rows[0]["Status"] == "missing_profile"
    mapping = tmp_path / "map.json"
    mapping.write_text(
        json.dumps({"autofuse_9_X": "autofuse_9__x_suffix"}), encoding="utf-8"
    )
    (tmp_path / "comparison.csv").unlink()
    result, rows = run_cli(tmp_path, [log], profiles, ["--op-map", str(mapping)])
    assert result.returncode == 0
    assert rows[0]["Status"] == "ok"
    assert rows[0]["ProfileEvidence"].endswith(":3")


def test_normalized_name_collision_does_not_reuse_one_profile_row(tmp_path):
    result, rows = run_cli(
        tmp_path,
        [final_line(op="Fusion_0") + final_line(op="Fusion0")],
        [[profile_row()]],
    )
    assert result.returncode == 0
    assert all(r["Status"] == "ambiguous_profile" and r["Measured"] == "" for r in rows)


@pytest.mark.parametrize("framed", [False, True])
def test_final_framing_and_null_zero_semantics(tmp_path, framed):
    result, rows = run_cli(
        tmp_path, [final_line(framed=framed)], [[profile_row(ratio="0.5")]]
    )
    assert result.returncode == 0, result.stderr
    assert rows[0]["Ratio"] == "4.0"
    assert rows[1]["Estimate"] == ""
    assert rows[1]["Measured"] == ""
    assert rows[1]["Status"] == "invalid_or_missing_metric"
    assert rows[2]["Estimate"] == "0"
    assert rows[2]["Measured"] == "0.0"
    assert rows[2]["Ratio"] == ""
    assert rows[2]["Status"] == "zero_measured"


def test_incomplete_framing_cannot_be_compared(tmp_path):
    log = final_line(framed=True).split("[ATT][FINAL_TILING_END]")[0]
    result, rows = run_cli(tmp_path, [log], [[profile_row()]])
    assert result.returncode == 0
    assert all(
        r["Status"] == "incomplete_final_tiling" and r["Measured"] == "" for r in rows
    )


def test_malformed_final_without_operator_is_reported_without_crashing(tmp_path):
    log = final_line().replace('operator="Fusion_0" ', "")
    result, rows = run_cli(tmp_path, [log], [[profile_row()]])
    assert result.returncode == 0, result.stderr
    assert all(
        r["Status"] == "invalid_final_tiling" and r["Measured"] == "" for r in rows
    )


@pytest.mark.parametrize(
    "groups,expected",
    [
        ({"0": {"case_id": 2, "tiling_key": 9}}, "ok"),
        ({"0": {"case_id": 2, "tiling_key": 17}}, "conflicting_final_summary"),
        (
            {
                "0": {"case_id": 2, "tiling_key": 9},
                "1": {"case_id": 0, "tiling_key": 0},
            },
            "ambiguous_final",
        ),
    ],
)
def test_summary_does_not_hide_missing_groups_or_conflicting_identity(
    tmp_path, groups, expected
):
    summary = (
        '[ATT][FINAL_TILING_SUMMARY] schema=1 source="runtime" selection_mode="default" '
        f'operator="Fusion_0" graph=0 result=0 groups={json.dumps(json.dumps(groups))}\n'
    )
    result, rows = run_cli(tmp_path, [final_line() + summary], [[profile_row()]])
    assert result.returncode == 0, result.stderr
    assert rows[0]["Status"] == expected
    assert rows[0]["FinalSummaryEvidence"] == f"{tmp_path}/run0.log:2"
    assert bool(rows[0]["Ratio"]) == (expected == "ok")


def test_missing_final_estimate_is_not_filled_from_candidate(tmp_path):
    content = (
        "[Fusion_0][PROF]The value of AIV_MTE2 is 999 in graph0_result0_g0_2.\n"
        + final_line(estimate=None)
    )
    result, rows = run_cli(tmp_path, [content], [[profile_row()]])
    assert result.returncode == 0
    final = next(
        r for r in rows if r["RecordKind"] == "final" and r["Pipe"] == "AIV_MTE2"
    )
    assert final["Estimate"] == ""
    assert final["Measured"] == "50.0"
    assert final["Ratio"] == ""
    assert final["Status"] == "missing_estimate"


@pytest.mark.parametrize("estimate", [-1, True])
def test_invalid_final_estimate_cannot_produce_ratio(tmp_path, estimate):
    result, rows = run_cli(tmp_path, [final_line(estimate=estimate)], [[profile_row()]])
    assert result.returncode == 0
    assert rows[0]["Status"] == "invalid_estimate"
    assert rows[0]["Ratio"] == ""


@pytest.mark.parametrize(
    "patch",
    [
        {"aiv_total_cycles": "nan"},
        {"aiv_total_cycles": "inf"},
        {"Block Num": 0},
        {"Block Num": "1.5"},
        {"aiv_mte2_ratio": 50},
        {"aiv_mte2_ratio": ""},
        {"aiv_mte2_ratio": -1},
    ],
)
def test_invalid_metrics_are_not_zero_filled(tmp_path, patch):
    profile = profile_row()
    profile.update(patch)
    result, rows = run_cli(tmp_path, [final_line()], [[profile]])
    assert result.returncode == 0
    assert rows[0]["Status"] == "invalid_or_missing_metric"
    assert rows[0]["Measured"] == ""
    assert rows[0]["Ratio"] == ""


def test_cli_rejects_unpaired_inputs_and_existing_output(tmp_path):
    result, _ = run_cli(tmp_path, [final_line(), final_line()], [[profile_row()]])
    assert result.returncode == 2
    assert "一一成对" in result.stderr
    output = tmp_path / "comparison.csv"
    output.write_text("preserve", encoding="utf-8")
    result, _ = run_cli(tmp_path, [final_line()], [[profile_row()]])
    assert result.returncode == 2
    assert output.read_text(encoding="utf-8") == "preserve"


def test_missing_required_profile_columns_fail_explicitly(tmp_path):
    result, _ = run_cli(tmp_path, [final_line()], [[{"Op Name": "Fusion_0"}]])
    assert result.returncode == 2
    assert "缺少列" in result.stderr
