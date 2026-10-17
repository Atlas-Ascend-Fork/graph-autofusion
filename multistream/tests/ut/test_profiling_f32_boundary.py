# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Boundary test for float32 conversion in profiling rows."""

import logging

import multistream.algorithm.profiling as profiling


def test_parse_row_discards_a_nonfinite_value_returned_by_f32(monkeypatch, caplog):
    """A converter must not smuggle an infinity into a stored profile."""

    monkeypatch.setattr(profiling, "_f32", lambda _value: float("inf"))
    caplog.set_level(logging.WARNING, logger="multistream")
    row = {
        "Op Name": "op",
        "Task Type": "AI_CORE",
        "Task Duration(us)": "1.0",
        "Block Num": "1",
        "Mix Block Num": "0",
    }
    assert profiling._parse_row(row) is None
    assert any(record.levelno == logging.WARNING for record in caplog.records)
