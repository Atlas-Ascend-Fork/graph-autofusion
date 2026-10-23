#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import csv
import subprocess
import sys
from pathlib import Path

import pytest

from commands.template_topn import build_rows


LOG_TEXT = """[paired-visit-start] BC04 0 static
[ATT][TEMPLATE] stage=topn_output output[0] group=0 semantic_index=0 semantic_name_count=1 semantic_name="Broadcast/NonTail+LoadToNDDMA" name_status=unique
[ATT][TEMPLATE] stage=topn_output output[1] group=0 semantic_index=0 semantic_name_count=1 semantic_name="Broadcast/Adaptive+NDDMA.Slice" name_status=unique
[INFO] output[0]: perf=1.250000 is_default=1 block_dim=32 repr_len=98
[INFO] output[1]: perf=2.500000 is_default=0 block_dim=32 repr_len=98
[paired-visit-end] BC04 0 static
[paired-visit-start] RD09 0 static
[ATT][TEMPLATE] stage=topn_output output[0] group=0 semantic_index=0 semantic_name_count=1 semantic_name="Reduce/RCore@Phase1" name_status=unique
[ATT][TEMPLATE] stage=topn_output output[0] group=1 semantic_index=0 semantic_name_count=1 semantic_name="Reduce/RCore@Phase2" name_status=unique
[ATT][TEMPLATE] stage=att_return group=0 semantic_index=0 semantic_name_count=1 semantic_name="Reduce/Common#AR|ReuseSrc|MultiRound"
[paired-visit-end] RD09 0 static
"""


def test_preserves_semantic_name_as_logged():
    name = "Reduce/Common/Norm/AR/NoReuseSrc/MultiRound"
    rows = build_rows(
        '[ATT][TEMPLATE] stage=topn_output output[0] group=0 semantic_name="'
        + name
        + '"'
    )
    assert rows[0]["top1_template_names"] == name


def test_build_rows_from_log_only():
    rows = build_rows(LOG_TEXT)

    assert [row["case_id"] for row in rows] == ["BC04", "RD09"]
    assert rows[0]["top1_template_names"] == "Broadcast/NonTail+LoadToNDDMA"
    assert rows[0]["topn_candidate_count"] == "2"
    assert rows[0]["top1_modeled_perf"] == "1.250000"
    assert rows[0]["top1_is_default"] == "1"
    assert rows[0]["topn_modeled_perf_values"] == "0:1.250000;1:2.500000"
    assert rows[0]["measured_task_duration_us"] == ""
    assert rows[0]["association_status"] == "unconfirmed"
    assert rows[1]["top1_template_names"] == "Reduce/RCore@Phase1;Reduce/RCore@Phase2"
    assert rows[1]["top1_group_count"] == "2"
    assert rows[1]["top1_modeled_perf"] == ""
    assert rows[1]["performance_source"] == "autofuse_log"


def test_build_rows_separates_topn_calls_in_one_visit():
    log_text = """
[paired-visit-start] CASE 0 static
GenerateTopnSolutions enter: topn=2 input_configs.size=1 operator=kernel_a
[ATT][TEMPLATE] stage=topn_output output[0] group=0 semantic_name="A/One" name_status=unique
[INFO] output[0]: perf=1.0 is_default=1
[ATT][TEMPLATE] stage=topn_output output[1] group=0 semantic_name="A/Two" name_status=unique
[INFO] output[1]: perf=2.0 is_default=0
GenerateTopnSolutions enter: topn=2 input_configs.size=1 operator=kernel_b
[ATT][TEMPLATE] stage=topn_output output[0] group=0 semantic_name="B/One" name_status=unique
[INFO] output[0]: perf=9.0 is_default=1
[ATT][TEMPLATE] stage=topn_output output[1] group=0 semantic_name="B/Two" name_status=unique
[INFO] output[1]: perf=8.0 is_default=0
[paired-visit-end] CASE 0 static
"""
    rows = build_rows(log_text)

    assert len(rows) == 2
    assert rows[0]["top1_template_names"] == "A/One"
    assert rows[0]["top1_modeled_perf"] == "1.0"
    assert rows[0]["association_status"] == "confirmed"
    assert rows[0]["topn_modeled_perf_values"] == "0:1.0;1:2.0"
    assert rows[1]["top1_template_names"] == "B/One"
    assert rows[1]["top1_modeled_perf"] == "9.0"
    assert rows[1]["association_status"] == "confirmed"
    assert rows[1]["topn_modeled_perf_values"] == "0:9.0;1:8.0"


def test_cli_reads_log_and_writes_csv_and_markdown(tmp_path):
    log_path = tmp_path / "worker.log"
    log_path.write_text(LOG_TEXT, encoding="utf-8")
    output = tmp_path / "report"
    entrypoint = Path(__file__).parents[2] / "src" / "att.py"

    result = subprocess.run(
        [
            sys.executable,
            str(entrypoint),
            "topn-templates",
            str(log_path),
            "-o",
            str(output),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert (output / "topn_templates.csv").is_file()
    markdown = (output / "topn_templates.md").read_text(encoding="utf-8")
    assert "`Broadcast/NonTail+LoadToNDDMA`" in markdown
    assert "Measured Task Duration is not inferred" in markdown
    assert "output[0]" in markdown


def test_cli_reports_missing_log(tmp_path, capsys):
    from att import build_parser
    from commands.template_topn import run

    args = build_parser().parse_args(
        ["topn-templates", str(tmp_path / "missing.log"), "-o", str(tmp_path / "out")]
    )
    args.stderr = sys.stderr

    assert run(args) == 1
    assert "No such file" in capsys.readouterr().err


def test_build_rows_separates_prefix_only_operators():
    rows = build_rows("""[paired-visit-start] CASE 0 static
EmitFinalTilingLines:[kernel_a][ATT][TEMPLATE] stage=topn_output output[0] group=0 semantic_name="Pointwise/Common"
EmitFinalTilingLines:[kernel_b][ATT][TEMPLATE] stage=topn_output output[0] group=0 semantic_name="Reduce/Common#AR"
""")
    assert len(rows) == 2
    assert [row["operator_name"] for row in rows] == ["kernel_a", "kernel_b"]
    assert all(row["association_status"] == "unconfirmed" for row in rows)


def test_build_rows_isolates_interleaved_threads():
    prefix = "[INFO] GE(123,python):2026-10-21-02:12:02.364.186 [/tmp/tiling.h:584]"
    lines = ["[paired-visit-start] CASE 0 static"]
    for tid, name in ((11, "kernel_a"), (22, "kernel_b")):
        lines.append(f"{prefix}{tid} GenerateTopnSolutions enter: operator={name}")
    for tid, name, perf in ((11, "kernel_a", "1.0"), (22, "kernel_b", "10.0")):
        lines.append(f"{prefix}{tid} output[0]: perf={perf} is_default=1")
        lines.append(
            f"{prefix}{tid} EmitFinalTilingLines:[{name}][ATT][TEMPLATE] "
            'stage=topn_output output[0] group=0 semantic_name="Pointwise/Common"'
        )
    rows = build_rows("\n".join(lines))
    assert len(rows) == 2
    assert [row["operator_name"] for row in rows] == ["kernel_a", "kernel_b"]
    assert [row["top1_modeled_perf"] for row in rows] == ["1.0", "10.0"]
    assert all(row["association_status"] == "confirmed" for row in rows)


def test_build_rows_exports_repeated_operator_call_identity():
    lines = ["[paired-visit-start] CASE 0 static"]
    for index in range(256):
        lines.append("GenerateTopnSolutions enter: operator=kernel_a")
        lines.append(
            "[ATT][TEMPLATE] stage=topn_output output[0] group=0 "
            'semantic_name="Pointwise/Common"'
        )
        lines.append(
            "[ATT][TEMPLATE] stage=topn_output output[0] group=0 "
            'fields family="Pointwise" strategy="Common"'
        )
    rows = build_rows("\n".join(lines))
    assert len(rows) == 256
    assert [row["call_index"] for row in rows] == [str(i) for i in range(1, 257)]
    assert all(row["operator_name"] == "kernel_a" for row in rows)


def test_cli_exports_operator_and_call_identity(tmp_path):
    log_path = tmp_path / "worker.log"
    log_path.write_text(
        """[paired-visit-start] CASE 0 static
GenerateTopnSolutions enter: operator=kernel_a
[ATT][TEMPLATE] stage=topn_output output[0] group=0 semantic_name="Pointwise/Common"
GenerateTopnSolutions enter: operator=kernel_a
[ATT][TEMPLATE] stage=topn_output output[0] group=0 semantic_name="Reduce/Common@Norm#AR"
""",
        encoding="utf-8",
    )
    output = tmp_path / "report"
    entrypoint = Path(__file__).parents[2] / "src" / "att.py"
    result = subprocess.run(
        [
            sys.executable,
            str(entrypoint),
            "topn-templates",
            str(log_path),
            "-o",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    with (output / "topn_templates.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [(row["operator_name"], row["call_index"]) for row in rows] == [
        ("kernel_a", "1"),
        ("kernel_a", "2"),
    ]
    assert rows[1]["top1_template_names"] == "Reduce/Common@Norm#AR"


def test_prefix_mismatch_does_not_reuse_previous_call_performance():
    rows = build_rows("""[paired-visit-start] CASE 0 static
GenerateTopnSolutions enter: operator=kernel_a
EmitFinalTilingLines:[kernel_a][ATT][TEMPLATE] stage=topn_output output[0] group=0 semantic_name="A/One"
output[0]: perf=1.0 is_default=1
EmitFinalTilingLines:[kernel_b][ATT][TEMPLATE] stage=topn_output output[0] group=0 semantic_name="B/One"
output[0]: perf=9.0 is_default=1
""")
    assert len(rows) == 2
    assert rows[0]["top1_modeled_perf"] == "1.0"
    assert rows[1]["association_status"] == "unconfirmed"
    assert rows[1]["top1_modeled_perf"] == ""


def test_build_rows_keeps_explicit_unavailable_name():
    rows = build_rows(
        "[paired-visit-start] CASE 0 static\n"
        "[ATT][TEMPLATE] stage=topn_output output[0] group=0 "
        'semantic_name="" name_status=unavailable\n'
    )
    assert len(rows) == 1
    assert rows[0]["top1_name_status"] == "unavailable"
    assert rows[0]["topn_candidate_count"] == "1"


@pytest.mark.parametrize(
    "identities,expected",
    [
        ([(0, 0, 0, 0), (1, 0, 0, 0)], "2"),
        ([(0, 0, 0, 0), (0, 1, 0, 0)], "2"),
        ([(0, 0, 0, 0), (0, 0, 0, 1)], "1"),
        ([(0, 0, 0, 0), (0, 0, 1, 0)], "2"),
    ],
)
def test_build_rows_counts_complete_group_identity(identities, expected):
    lines = [
        "[paired-visit-start] CASE 0 static",
        "GenerateTopnSolutions enter: operator=kernel_a",
    ]
    for graph, result, group, semantic_index in identities:
        prefix = (
            "EmitFinalTilingLines:[kernel_a][ATT][TEMPLATE] stage=topn_output "
            f"output[0] group={group} semantic_index={semantic_index}"
        )
        lines.extend(
            [
                f'{prefix} semantic_name="Pointwise/Common"',
                f'{prefix} fields family="Pointwise"',
                f"{prefix} detail graph={graph} result={result}",
                f"{prefix} detail tiling_key=0",
            ]
        )
    rows = build_rows("\n".join(lines))
    assert len(rows) == 1
    assert rows[0]["top1_group_count"] == expected


def test_build_rows_isolates_split_details_and_candidates():
    prefix = "[ATT][TEMPLATE] stage=topn_output"
    rows = build_rows(f"""[paired-visit-start] CASE 0 static
GenerateTopnSolutions enter: operator=kernel_a
{prefix} output[0] group=0 semantic_index=0 semantic_name="A/One"
{prefix} output[1] group=0 semantic_index=0 semantic_name="B/One"
{prefix} output[1] group=0 semantic_index=0 detail graph=8 result=9
{prefix} output[0] group=0 semantic_index=0 detail graph=0
{prefix} output[0] group=0 semantic_index=0 detail result=0
{prefix} output[0] group=0 semantic_index=0 semantic_name="A/Two"
{prefix} output[0] group=0 semantic_index=0 detail graph=1
{prefix} output[0] group=0 semantic_index=0 detail result=0
""")
    assert rows[0]["top1_group_count"] == "2"
    assert rows[0]["topn_candidate_count"] == "2"


def test_build_rows_isolates_detail_threads_and_calls():
    log_prefix = "[INFO] GE(123,python):2026-10-21 [/tmp/tiling.h:584]"
    lines = ["[paired-visit-start] CASE 0 static"]
    for call in range(2):
        for tid in (11, 22):
            lines.append(
                f"{log_prefix}{tid} GenerateTopnSolutions enter: operator=kernel_a"
            )
        for graph in (0, 1):
            for tid in (11, 22):
                prefix = (
                    f"{log_prefix}{tid} EmitFinalTilingLines:[kernel_a]"
                    "[ATT][TEMPLATE] stage=topn_output output[0] group=0 semantic_index=0"
                )
                lines.append(f'{prefix} semantic_name="Pointwise/Common"')
            for tid in (11, 22):
                prefix = (
                    f"{log_prefix}{tid} EmitFinalTilingLines:[kernel_a]"
                    "[ATT][TEMPLATE] stage=topn_output output[0] group=0 semantic_index=0"
                )
                lines.append(f"{prefix} detail graph={graph} result={call}")
    rows = build_rows("\n".join(lines))
    assert len(rows) == 4
    assert [row["call_index"] for row in rows] == ["1", "2", "3", "4"]
    assert all(row["top1_group_count"] == "2" for row in rows)


def test_build_rows_missing_detail_does_not_inherit_group_identity():
    prefix = "[ATT][TEMPLATE] stage=topn_output output[0] group=0 semantic_index=0"
    rows = build_rows(f"""[paired-visit-start] CASE 0 static
GenerateTopnSolutions enter: operator=kernel_a
{prefix} semantic_name="A/One"
{prefix} detail graph=0 result=0
{prefix} semantic_name="A/Two"
{prefix} detail graph=1 result=0
{prefix} semantic_name="A/Unknown"
""")
    # An incomplete log can only establish a lower bound from local group IDs.
    assert rows[0]["top1_group_count"] == "1"


def test_cli_exports_complete_group_count(tmp_path):
    from att import build_parser
    from commands.template_topn import run

    prefix = "[ATT][TEMPLATE] stage=topn_output output[0] group=0 semantic_index=0"
    log_path = tmp_path / "worker.log"
    log_path.write_text(
        f"""[paired-visit-start] CASE 0 static
GenerateTopnSolutions enter: operator=kernel_a
{prefix} semantic_name="Pointwise/Common"
{prefix} detail graph=0 result=0
{prefix} semantic_name="Reduce/Common@Norm#AR"
{prefix} detail graph=1 result=0
""",
        encoding="utf-8",
    )
    output = tmp_path / "report"
    args = build_parser().parse_args(
        ["topn-templates", str(log_path), "-o", str(output)]
    )
    assert run(args) == 0
    with (output / "topn_templates.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["top1_group_count"] == "2"
    assert "local group IDs" in (output / "topn_templates.md").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize("candidate_count", [1, 2, 128])
def test_build_rows_counts_summarized_core_without_counting_detail_summaries(
    candidate_count,
):
    lines = [
        "[paired-visit-start] CASE 0 static",
        "GenerateTopnSolutions enter: operator=kernel_a",
    ]
    for index in range(candidate_count):
        prefix = (
            f"[ATT][TEMPLATE] stage=topn_output output[{index}] group=0 "
            "semantic_index=0 semantic_name_count=1"
        )
        lines.extend(
            [
                f"{prefix} line_hash=0123456789abcdef line_len=917",
                f"{prefix} fields line_hash=abcdef0123456789 line_len=1024",
                f"{prefix} detail graph=0 result=0 tiling_key={index}",
                f"{prefix} detail line_hash=abcdef0123456789 line_len=1024",
            ]
        )
    rows = build_rows("\n".join(lines))
    assert len(rows) == 1
    assert rows[0]["topn_candidate_count"] == str(candidate_count)
    assert rows[0]["top1_template_names"] == "Unknown"
    assert rows[0]["top1_name_status"] == "unavailable"
    assert rows[0]["top1_group_count"] == "1"


def test_build_rows_ignores_incomplete_core_summary():
    prefix = "[ATT][TEMPLATE] stage=topn_output output[0] group=0"
    assert build_rows(f"{prefix} line_hash=0123456789abcdef") == []
    assert build_rows(f"{prefix} line_len=917") == []


def test_cli_keeps_summarized_candidate_with_valid_name_candidate(tmp_path):
    from att import build_parser
    from commands.template_topn import run

    prefix = "[ATT][TEMPLATE] stage=topn_output"
    log_path = tmp_path / "worker.log"
    log_path.write_text(
        f"""[paired-visit-start] CASE 0 static
GenerateTopnSolutions enter: operator=kernel_a
{prefix} output[0] group=0 semantic_index=0 line_hash=0123456789abcdef line_len=917
{prefix} output[0] group=0 semantic_index=0 detail graph=0 result=0 tiling_key=0
{prefix} output[1] group=0 semantic_index=0 semantic_name="Pointwise/Common"
{prefix} output[1] group=0 semantic_index=0 detail graph=0 result=0 tiling_key=1
""",
        encoding="utf-8",
    )
    output = tmp_path / "report"
    args = build_parser().parse_args(
        ["topn-templates", str(log_path), "-o", str(output)]
    )
    assert run(args) == 0
    with (output / "topn_templates.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["topn_candidate_count"] == "2"
    assert rows[0]["top1_template_names"] == "Unknown"
    assert rows[0]["top1_name_status"] == "unavailable"
    assert rows[0]["top1_group_count"] == "1"
    assert rows[0]["topn_candidate_template_names"] == "Unknown;Pointwise/Common"
