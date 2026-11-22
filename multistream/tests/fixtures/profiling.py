# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Profiling file fixtures for multistream tests."""

import csv
from pathlib import Path

import pytest


@pytest.fixture
def op_summary_csv(tmp_path):
    """Write rows to an ``op_summary.csv`` and return its path.

    Rows may be mappings or sequences.  Passing ``headers`` allows malformed
    header tests to use the same helper as normal parser tests.
    """

    default_headers = (
        "Op Name",
        "Task Type",
        "Task Duration(us)",
        "Block Num",
        "Mix Block Num",
    )

    def write(rows=(), headers=default_headers, filename="op_summary.csv"):
        path = Path(tmp_path) / filename
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            for row in rows:
                if isinstance(row, dict):
                    writer.writerow([row.get(header, "") for header in headers])
                else:
                    writer.writerow(row)
        return path

    return write
