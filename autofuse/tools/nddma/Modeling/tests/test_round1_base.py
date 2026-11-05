# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "Data_collection/round1/scripts/fit.py"
spec = importlib.util.spec_from_file_location("round1_fit", SCRIPT)
fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)


class BaseFitTests(unittest.TestCase):
    def test_average_throughput_not_slope(self):
        rows = [
            dict(block_dim=k, bytes_per_core=b, actual=slope * b + 100)
            for k, slope in [(4, 0.1), (8, 0.2)]
            for b in (100, 1000)
        ]
        result = fit.fit_branch(rows, "gt2")
        self.assertAlmostEqual(result["T"], 7.5)
        self.assertAlmostEqual(result["H"], 100)

    def test_merge_groups_and_repeats_by_bytes_using_mean(self):
        base = dict(
            dtype="int8_t",
            block_dim="2",
            dim="1",
            fit_role="fit",
            input_stride="1",
            output_stride="1",
            repeat="2",
            logical_total_bytes="200",
            bytes_per_core="999",
        )
        rows = [
            dict(
                base,
                token=str(i),
                sensitivity_key=(
                    "group_a_stage1_linear_fit"
                    if i < 2
                    else "group_c_t_alpha_block_model"
                ),
                nddma_mte2_cycles_per_block=str(y * 2),
            )
            for i, y in enumerate((10, 10, 100))
        ]
        rows.append(dict(rows[0], fit_role="validation"))
        result = fit.aggregate_rows(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["actual"], 40)
        self.assertEqual(result[0]["bytes_per_core"], 100)

    def test_each_core_has_equal_parameter_weight(self):
        rows = [
            dict(block_dim=k, bytes_per_core=b, actual=b / t + h)
            for k, t, h, bs in [
                (4, 10, 100, (10, 20)),
                (8, 20, 300, (10, 20, 40, 80, 160)),
            ]
            for b in bs
        ]
        result = fit.fit_branch(rows, "gt2")
        self.assertAlmostEqual(result["T"], 15)
        self.assertAlmostEqual(result["H"], 200)


if __name__ == "__main__":
    unittest.main()
