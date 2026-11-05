# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run reference parity with HW_NDDMA_MODELING=/path/to/HW_GE_ATT/NDDMA/Modeling."""

import importlib.util
import os
from pathlib import Path
import sys
import unittest
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


collect = load(
    "v2_noncontig_collect", ROOT / "Data_collection/round2/scripts/collect.py"
)
fit = load("v2_noncontig_fit", ROOT / "Data_collection/round2/scripts/fit.py")


@unittest.skipUnless(
    os.environ.get("HW_NDDMA_MODELING"), "HW reference path not supplied"
)
class HWParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hw = Path(os.environ["HW_NDDMA_MODELING"])
        cls.plan = load(
            "plan_common", hw / "Data_collection/round4/scripts/plan_common.py"
        )
        cls.multi_plan = load(
            "hw_multi_plan",
            hw / "Data_collection/round4/scripts/plan_multicore_common.py",
        )
        cls.single = load(
            "analyze_round4_1d_staged", hw / "Ana/round4/analyze_round4_1d_staged.py"
        )
        cls.multi = load(
            "hw_multi_fit", hw / "Ana/round4/analyze_round4_1d_multicore.py"
        )

    def test_complete_grid(self):
        reference = (
            self.plan.build_round4_rows()
            + self.multi_plan.build_round4_multicore_rows()
        )
        actual = collect.build_rows(200, 3)

        def key(r):
            return (
                r["group_id"],
                r["dtype"],
                int(r["block_dim"]),
                int(r["output_dims"]),
                int(r["input_stride"]),
                int(r["output_stride"]),
            )

        self.assertEqual(Counter(map(key, reference)), Counter(map(key, actual)))
        print("Aligned grid:", dict(Counter(r["group_id"] for r in actual)))

    def test_fitted_parameters_and_excluded_samples(self):
        raw = collect.build_rows(200, 3)
        base = {d: dict(T_1=20.0, H_1=100.0, T_2=10.0, H_2=200.0) for d in fit.DTYPES}
        for r in raw:
            b = float(r["bytes_per_core"])
            k = int(r["block_dim"])
            s = min(int(r["input_stride"]) * fit.DTYPE_SIZES[r["dtype"]], 128)
            g = min(1, int(r["output_stride"]) - 1)
            correction = (2 + 0.01 * b) * s + (3 + 0.1 * s + (0.02 + 0.001 * s) * b) * g
            rho = 1 if k <= 2 else 1.2 + 0.001 * s + g * (0.3 + 0.002 * s)
            # Structured noise exposes stage weighting differences.
            y = (
                fit.predict_base(base, r["dtype"], k, b)
                + rho * correction
                + (b % 17) * 0.13
            )
            r["actual_y"] = str(y)
            r["data_split"] = r["group_id"]
        model = fit.fit_model(fit.aggregate(raw), {"parameters": base})
        rows = [
            r
            for r in raw
            if r["group_id"] in ("A", "B", "C")
            and float(r["logical_total_bytes"]) < 50000
        ]
        for d in fit.DTYPES:
            self.single.ROUND2_SECOND_MODEL[d] = dict(
                alpha_le2=100.0, t_le2_bytes_per_cycle=20.0
            )
        a = self.single.fit_a_sub128_input_stride(rows)
        q = {
            str(i): self.single.fit_is_os2_adjusted_bytes(rows, a, i)
            for i in self.single.ADJUSTED_INPUT_STRIDE_VALUES
        }
        bc = self.single.fit_adjusted_coefficients_vs_input_stride(q)
        prepared = []
        for d in fit.DTYPES:
            ap = a["dtype_models"][d]
            bp = bc["dtype_models"][d]
            params = model["parameters"][d]
            np.testing.assert_allclose(
                list(params["N_G"].values()),
                [ap["c0"], ap["c1_per_byte"]],
                rtol=1e-10,
                atol=1e-10,
            )
            np.testing.assert_allclose(
                list(params["N_GU"].values()),
                [
                    bp["a0"],
                    bp["a1_per_stride"],
                    bp["b0_per_byte"],
                    bp["b1_per_stride_byte"],
                ],
                rtol=1e-10,
                atol=1e-10,
            )
        for r in fit.aggregate(raw):
            if r["group"] not in ("G", "H"):
                continue
            ng, ngu, s = fit.predict_correction(
                model["parameters"],
                r["dtype"],
                r["bytes_per_core"],
                r["input_stride"],
                r["output_stride"],
            )
            prepared.append(
                dict(
                    row=dict(
                        data_split=r["group"],
                        input_stride=str(int(r["input_stride"])),
                        output_stride=str(int(r["output_stride"])),
                    ),
                    dtype=r["dtype"],
                    block_dim=r["block_dim"],
                    s=s,
                    correction=ng + ngu,
                    actual=r["actual"],
                    base=fit.predict_base(
                        base, r["dtype"], r["block_dim"], r["bytes_per_core"]
                    ),
                )
            )
        rho = self.multi.fit_low_stride_linear_delta1_model(prepared)
        for d in fit.DTYPES:
            np.testing.assert_allclose(
                list(model["parameters"][d]["rho"].values()),
                [rho["dtype_models"][d][f"c_{i}"] for i in range(1, 5)],
                rtol=1e-10,
                atol=1e-10,
            )
        # Out-of-scope samples must not affect any fitted parameter.
        for r in raw:
            byte_stride = int(r["input_stride"]) * fit.DTYPE_SIZES[r["dtype"]]
            excluded = (
                r["group_id"] in ("B", "F", "I", "J")
                or (
                    r["group_id"] in ("A", "C")
                    and (
                        float(r["logical_total_bytes"]) >= 50000
                        or byte_stride >= 128
                        or (r["group_id"] == "C" and int(r["output_stride"]) != 2)
                    )
                )
                or (
                    r["group_id"] in ("G", "H")
                    and (int(r["block_dim"]) <= 2 or byte_stride > 128)
                )
            )
            if excluded:
                r["actual_y"] = str(float(r["actual_y"]) + 100000)
        updated = fit.fit_model(fit.aggregate(raw), {"parameters": base})
        for d in fit.DTYPES:
            for term in ("N_G", "N_GU", "rho"):
                np.testing.assert_allclose(
                    list(model["parameters"][d][term].values()),
                    list(updated["parameters"][d][term].values()),
                    rtol=1e-10,
                    atol=1e-10,
                )


if __name__ == "__main__":
    unittest.main()
