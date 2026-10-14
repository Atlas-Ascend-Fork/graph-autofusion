#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from compare_csv import CSVComparator, ComparisonReporter


def comparison(tmp_path, first, second):
    paths = [tmp_path / "first.csv", tmp_path / "second.csv"]
    for path, value in zip(paths, [first, second]):
        path.write_text(
            f"Operator,Graph,Result,Group,Case,Result Perf\nAdd,0,0,0,1,{value}\n",
            encoding="utf-8",
        )
    return CSVComparator().compare_csv_files(*(str(path) for path in paths))


def test_report_write_error_keeps_original_exception_and_restores_stdout(
    tmp_path, monkeypatch, capsys
):
    import builtins

    reporter = ComparisonReporter(comparison(tmp_path, 1, 2))
    output = tmp_path / "report.txt"
    original_open = builtins.open
    original_stdout = sys.stdout

    class FailingFile:
        def __enter__(self):
            self.file = original_open(output, "w", encoding="utf-8")
            return self

        def write(self, text):
            if text.startswith("CSV"):
                raise OSError("simulated disk full during write")
            return self.file.write(text)

        def __exit__(self, *args):
            self.file.close()

    def open_with_write_failure(path, *args, **kwargs):
        return (
            FailingFile()
            if str(path) == str(output)
            else original_open(path, *args, **kwargs)
        )

    monkeypatch.setattr(builtins, "open", open_with_write_failure)
    try:
        with pytest.raises(OSError, match="disk full"):
            reporter.save_text_report(str(output))
        assert sys.stdout is original_stdout
        print("write error can still be reported")
    finally:
        # Restore for pytest even when the original implementation fails the assertion.
        sys.stdout = original_stdout
    assert "write error can still be reported" in capsys.readouterr().out


def test_successful_text_report_preserves_contents_and_stdout(tmp_path, capsys):
    output = tmp_path / "report.txt"
    original_stdout = sys.stdout
    ComparisonReporter(comparison(tmp_path, 1, 2)).save_text_report(str(output))
    assert sys.stdout is original_stdout
    assert "CSV文件对比分析报告" in output.read_text(encoding="utf-8")
    assert "对比报告已保存到" in capsys.readouterr().out
