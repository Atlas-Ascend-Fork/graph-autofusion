# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""``op_summary.csv`` parser tests, test plan §4.8."""

import logging

import pytest

from multistream.algorithm.profiling import (
    _calculate_core_counts,
    _parse_row,
    match_profile,
    parse,
)


@pytest.mark.parametrize(
    "task_type,block,mix,expected",
    [
        ("AI_CORE", 3, 7, (3, 0)),
        ("AI_VECTOR_CORE", 3, 7, (0, 3)),
        ("MIX_AIC", 3, 7, (3, 7)),
        ("MIX_AIV", 3, 7, (7, 3)),
        ("UNKNOWN", 3, 7, (0, 0)),
        ("", 0, 0, (0, 0)),
    ],
)
def test_core_count_conversion(task_type, block, mix, expected):
    assert _calculate_core_counts(task_type, block, mix) == expected


def test_parse_normal_rows_and_column_order_independence(op_summary_csv):
    path = op_summary_csv(
        [
            {
                "Op Name": "a",
                "Task Type": "MIX_AIV",
                "Task Duration(us)": "2.5",
                "Block Num": "4",
                "Mix Block Num": "8",
            }
        ]
    )
    profiles = parse(path)
    assert profiles["a"].execution_time == pytest.approx(2.5)
    assert (profiles["a"].cube_block_num, profiles["a"].vec_block_num) == (8, 4)

    reordered = op_summary_csv(
        [["2", "a", "0", "1.0", "AI_CORE"]],
        headers=(
            "Block Num",
            "Op Name",
            "Mix Block Num",
            "Task Duration(us)",
            "Task Type",
        ),
        filename="reordered.csv",
    )
    assert parse(reordered)["a"].cube_block_num == 2


def _row(**overrides):
    row = {
        "Op Name": "x",
        "Task Type": "AI_CORE",
        "Task Duration(us)": "1",
        "Block Num": "1",
        "Mix Block Num": "0",
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize(
    "row,message",
    [
        (
            _row(**{"Op Name": "", "Task Duration(us)": "bad"}),
            "Failed to parse Task Duration: bad",
        ),
        (_row(**{"Task Duration(us)": "nan"}), "Failed to parse Task Duration: nan"),
        (_row(**{"Task Duration(us)": "inf"}), "Failed to parse Task Duration: inf"),
        (
            _row(**{"Task Duration(us)": "1e1000"}),
            "Failed to parse Task Duration: 1e1000",
        ),
        (_row(**{"Task Duration(us)": "1e39"}), "Failed to parse Task Duration: 1e39"),
        (_row(**{"Block Num": "-1"}), "Failed to parse Block Num or Mix Block Num"),
        (
            _row(**{"Mix Block Num": "abc"}),
            "Failed to parse Block Num or Mix Block Num",
        ),
        # csv.DictReader fills the missing tail of a short row with None.
        (
            _row(**{"Block Num": None, "Mix Block Num": None}),
            "Invalid column index 3 for field size 3",
        ),
        ({"Op Name": "x"}, "Required column not found: Task Type"),
    ],
)
def test_parse_row_skips_malformed_values(row, message, caplog):
    caplog.set_level(logging.WARNING, logger="multistream")
    assert _parse_row(row) is None
    assert message in [record.getMessage() for record in caplog.records]


def test_parse_reports_short_csv_row_like_cpp_parser(tmp_path, caplog):
    path = tmp_path / "short.csv"
    path.write_text(
        "Op Name,Task Type,Task Duration(us),Block Num,Mix Block Num\n"
        "x,AI_CORE\n"
        "y,AI_CORE,1,1,0\n",
        encoding="utf-8",
    )
    caplog.set_level(logging.WARNING, logger="multistream")
    assert list(parse(path)) == ["y"]
    assert "Invalid column index 2 for field size 2" in [
        record.getMessage() for record in caplog.records
    ]


def test_missing_required_columns_and_bad_files_are_value_errors(
    tmp_path, op_summary_csv
):
    missing = op_summary_csv(
        [], headers=("Op Name", "Task Type"), filename="missing.csv"
    )
    with pytest.raises(ValueError, match="Task Duration"):
        parse(missing)
    with pytest.raises(ValueError):
        parse(tmp_path / "does-not-exist.csv")
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ValueError):
        parse(directory)
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        parse(empty)


def test_file_level_errors_are_logged_where_found(tmp_path, op_summary_csv, caplog):
    caplog.set_level(logging.ERROR, logger="multistream")

    def errors_for(path):
        caplog.clear()
        with pytest.raises(ValueError):
            parse(path)
        return {
            (record.levelname, record.filename, record.getMessage())
            for record in caplog.records
        }

    missing = op_summary_csv(
        [], headers=("Op Name", "Task Type", "Block Num"), filename="missing.csv"
    )
    assert errors_for(missing) == {
        ("ERROR", "profiling.py", "Required column not found: Task Duration(us)"),
        ("ERROR", "profiling.py", "Required column not found: Mix Block Num"),
    }
    absent = tmp_path / "does-not-exist.csv"
    assert errors_for(absent) == {
        (
            "ERROR",
            "profiling.py",
            f"Failed to get real path for profiling file: {absent}",
        )
    }
    directory = tmp_path / "directory"
    directory.mkdir()
    assert errors_for(directory) == {
        ("ERROR", "profiling.py", f"Failed to open profiling file: {directory}")
    }
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    assert errors_for(empty) == {
        ("ERROR", "profiling.py", f"Failed to read header line from: {empty}")
    }


def test_malformed_rows_are_skipped_while_valid_rows_remain(op_summary_csv):
    path = op_summary_csv(
        [
            ["bad", "AI_CORE", "not-a-number", "1", "0"],
            ["ok", "AI_CORE", "1.5", "1", "0"],
        ],
        filename="mixed.csv",
    )
    assert list(parse(path)) == ["ok"]


def test_header_only_or_all_malformed_file_returns_empty_profiles(op_summary_csv):
    header_only = op_summary_csv([], filename="header_only.csv")
    assert parse(header_only) == {}
    all_bad = op_summary_csv(
        [["bad-a", "AI_CORE", "bad", "1", "0"], ["bad-b", "AI_CORE", "nan", "1", "0"]],
        filename="all_bad.csv",
    )
    assert parse(all_bad) == {}


def test_uint32_boundary_and_extra_columns_are_supported(op_summary_csv):
    path = op_summary_csv(
        [["x", "MIX_AIC", "1", str((1 << 32) - 1), "0", "ignored"]],
        headers=(
            "Op Name",
            "Task Type",
            "Task Duration(us)",
            "Block Num",
            "Mix Block Num",
            "Extra",
        ),
        filename="boundary.csv",
    )
    profile = parse(path)["x"]
    assert profile.cube_block_num == (1 << 32) - 1


def test_duplicate_profile_name_uses_the_later_row(op_summary_csv):
    path = op_summary_csv(
        [["dup", "AI_CORE", "1", "1", "0"], ["dup", "AI_CORE", "9", "3", "0"]],
        filename="duplicate.csv",
    )
    profile = parse(path)["dup"]
    assert profile.execution_time == pytest.approx(9)
    assert profile.cube_block_num == 3


def test_profile_name_matching_prefers_full_name_then_scope():
    first = _parse_row(
        {
            "Op Name": "a/b/c",
            "Task Type": "AI_CORE",
            "Task Duration(us)": "1",
            "Block Num": "1",
            "Mix Block Num": "0",
        }
    )
    second = _parse_row(
        {
            "Op Name": "c",
            "Task Type": "AI_CORE",
            "Task Duration(us)": "2",
            "Block Num": "2",
            "Mix Block Num": "0",
        }
    )
    profiles = {first.op_name: first, second.op_name: second}
    assert match_profile(profiles, "a/b/c") is first
    assert match_profile(profiles, "x/c") is second
    assert match_profile(profiles, "x/d") is None


@pytest.mark.parametrize("path", [b"x.csv", 1, object()])
def test_parse_rejects_non_text_paths(path):
    with pytest.raises(TypeError):
        parse(path)


def test_parse_logs_record_count(op_summary_csv, caplog):
    path = op_summary_csv(
        [["a", "AI_CORE", "1", "1", "0"], ["b", "AI_CORE", "2", "1", "0"]],
        filename="log.csv",
    )
    caplog.set_level(logging.INFO, logger="multistream")
    parse(path)
    assert any(
        "Parsed 2 profiling records from" in record.getMessage()
        for record in caplog.records
    )
