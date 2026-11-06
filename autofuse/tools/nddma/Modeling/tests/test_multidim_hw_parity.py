# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Reference parity: set HW_NDDMA_MODELING to the HW modeling directory."""

from collections import Counter
import importlib.util
from itertools import product
import os
from pathlib import Path
import sys
import tempfile
import json
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


r3c = load("r3_collect_test", ROOT / "Data_collection/round3/scripts/collect.py")
r4c = load("r4_collect_test", ROOT / "Data_collection/round4/scripts/collect.py")
r3f = load("r3_fit_test", ROOT / "Data_collection/round3/scripts/fit.py")
r4f = load("r4_fit_test", ROOT / "Data_collection/round4/scripts/fit.py")


def fixture_model():
    return {
        "parameters": {
            d: {
                "base": dict(T_1=20.0, H_1=100.0, T_2=10.0, H_2=200.0),
                "N_G": dict(a1=2.0, a2=0.01),
                "N_GU": dict(b1=3.0, b2=0.1, b3=0.02, b4=0.001),
                "rho": dict(c1=1.2, c2=0.001, c3=0.3, c4=0.002),
            }
            for d in r3f.DTYPE_SIZES
        }
    }


def canonical(model):
    result = {}
    for d, p in model["parameters"].items():
        result[d] = dict(p["base"])
        for src, names in [
            ("N_G", ("a1", "a2")),
            ("N_GU", ("b1", "b2", "b3", "b4")),
            ("rho", ("c1", "c2", "c3", "c4")),
        ]:
            result[d].update({f"{n[0]}_{n[1]}": p[src][n] for n in names})
    return {"unified_dtype_parameters": result}


class MultidimTests(unittest.TestCase):
    def test_base_is_per_core_without_extra_scale(self):
        m = fixture_model()
        self.assertEqual(r3f.one_d_base(m, "int8_t", 1000, 8), 300.0)
        self.assertEqual(r4f.one_d_base(m, "int8_t", 1000, 8), 300.0)
        row = dict(
            dim="3",
            dtype="int8_t",
            block_dim="8",
            output_dims="2x4x8",
            input_stride="32x8x1",
            output_stride="32x8x1",
            actual_y="100",
            total_bytes="64",
        )
        result = r3f.fit_model([row], m, "fixture")
        point = result["predictions"][0]
        self.assertEqual(point["n_base_cycles"], 206.4)
        self.assertEqual(
            point["predicted_cycles"], point["n_base_cycles"] + point["t1_cycles"]
        )
        row["actual_y"] = "1000000"
        second = r3f.fit_model([row], m, "fixture")
        self.assertEqual(
            point["predicted_cycles"], second["predictions"][0]["predicted_cycles"]
        )
        self.assertEqual(result["parameters"], m["parameters"])

    def test_ng2_excludes_multicore_and_other_groups(self):
        rows = r4c.build_rows(3)
        for row in rows:
            row["actual_y"] = "1000"
        result = r4f.aggregate(rows)
        self.assertEqual(len(result), len(rows))
        extras = [
            dict(rows[0], config_id="validation", group_id="M", data_split="M"),
            dict(rows[0], config_id="multicore", block_dim="8"),
        ]
        self.assertEqual(len(r4f.aggregate(rows + extras)), len(result))

    def test_old_g_only_data_rejected(self):
        rows = [r for r in r4c.build_rows(3) if r["group_id"] == "G"]
        for row in rows:
            row["actual_y"] = "1000"
        with self.assertRaisesRegex(ValueError, "requires D/F/G"):
            r4f.fit_model(rows, fixture_model(), "fixture")

    def test_prediction_tools_share_base_and_correction(self):
        tools = ROOT / "Data_collection/common/tools"
        sys.path.insert(0, str(tools))
        unified = load("predict_nddma_ad_hoc", tools / "predict_nddma_ad_hoc.py")
        special = load(
            "special_predict_test", tools / "predict_nddma_2d_transpose_ad_hoc.py"
        )
        model = fixture_model()
        bases = {d: p["base"] for d, p in model["parameters"].items()}
        with (
            patch.object(unified, "ONE_DIMENSIONAL_PARAMETERS", bases),
            patch.object(unified, "CORRECTION_PARAMETERS", model["parameters"]),
            patch.object(special, "ROUND4_ONE_DIMENSIONAL_PARAMETERS", bases),
            patch.object(special, "ROUND4_CORRECTION_PARAMETERS", model["parameters"]),
        ):
            for d in bases:
                for k in (1, 2, 8, 56):
                    b, c = unified.one_dimensional_base_and_t1(d, 1024, 8, 1, k)
                    self.assertAlmostEqual(b, r3f.one_d_base(model, d, 1024, k))
                    self.assertAlmostEqual(
                        c, r3f.one_d_correction(model, d, 1024, 8, 1, k)
                    )
                    self.assertAlmostEqual(b, special.one_d_base(d, 1024, k))
                    self.assertAlmostEqual(c, special.one_d_g1(d, 1024, 8, k))


@unittest.skipUnless(
    os.environ.get("HW_NDDMA_MODELING"), "HW reference path not supplied"
)
class HWMultidimParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hw = Path(os.environ["HW_NDDMA_MODELING"])
        cls.plan5 = load(
            "reference_plan5",
            hw / "Data_collection/round5/scripts/plan_round5_common.py",
        )
        cls.config6 = load("reference_config6", hw / "Ana/round6/round6_3d_config.py")
        cls.config7 = load("reference_config7", hw / "Ana/round7/round7_4d_config.py")
        load("plan_common", hw / "Data_collection/round4/scripts/plan_common.py")
        cls.plan2 = load(
            "reference_plan2", hw / "Data_collection/round4/scripts/plan_2d_common.py"
        )
        cls.ref = load(
            "reference_multidim", hw / "Ana/round7/analyze_round7_baseline.py"
        )
        cls.model = fixture_model()
        cls.temp = tempfile.TemporaryDirectory()
        path = Path(cls.temp.name) / "model.json"
        path.write_text(json.dumps(canonical(cls.model)))
        cls.ref.one_d.load_merged_one_dimensional_formula(path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_multidim_grid(self):
        actual = r3c.build_rows(3)

        def key(d, dims, gm, ub):
            return (d, tuple(dims), tuple(gm), tuple(ub))

        def from_row(r):
            return key(
                r["dtype"],
                *(
                    tuple(map(int, r[f].split("x")))
                    for f in ("output_dims", "input_stride", "output_stride")
                ),
            )

        expected = Counter(from_row(r) for r in self.plan2.build_round4_2d_rows())
        p = self.plan5
        for d in p.DTYPE_SIZE:
            for k, m, n, is2, is1 in product(
                p.THREE_D_K_VALUES,
                p.THREE_D_M_VALUES,
                p.THREE_D_N_VALUES,
                p.THREE_D_IS2_VALUES,
                p.THREE_D_IS1_VALUES,
            ):
                if p.is_safe_three_d(d, k, m, n, is2, is1):
                    expected[key(d, (k, m, n), (1, is2, is1), (m * n, n, 1))] += 1
            for cfg in self.config6.iter_configs():
                dims = cfg["shape"]
                gm = cfg["gm_stride"]
                ub = cfg["ub_stride"]
                if r3c.safe(d, dims, gm, ub):
                    expected[key(d, dims, gm, ub)] += 1
            for cfg in self.config7.iter_configs():
                dims = tuple(reversed(tuple(cfg[f"l{i}"] for i in range(cfg["dim"]))))
                gm = cfg["input_stride"]
                ub = cfg["output_stride"]
                if r3c.safe(d, dims, gm, ub):
                    expected[key(d, dims, gm, ub)] += 1
        self.assertEqual(expected, Counter(from_row(r) for r in actual))
        self.assertEqual(len({r["config_id"] for r in actual}), len(actual))
        print("Multidim grid:", dict(Counter(r["source_round"] for r in actual)))

    def test_ng2_grid_and_fit(self):
        p = self.plan5
        expected = Counter()
        for d, m, n in product(p.DTYPE_SIZE, p.TRANSPOSE_M_VALUES, p.N_VALUES):
            for group, strides in [
                ("D", p.TRANSPOSE_INPUT_STRIDE_VALUES),
                ("F", p.HIGH_STRIDE_INCREMENTAL_VALUES),
            ]:
                for i in strides:
                    if i >= m and p.is_safe_strided_transpose(d, m, n, i):
                        expected[(group, d, m, n, 1, i)] += 1
            for outer, i in product(
                p.GM_OUTER_STRIDE_VALUES, p.TRANSPOSE_INPUT_STRIDE_VALUES
            ):
                if outer * m <= i and p.is_safe_gm_outer_stride(d, m, n, outer, i):
                    expected[("G", d, m, n, outer, i)] += 1
        rows = r4c.build_rows(3)
        actual = Counter(
            (
                r["group_id"],
                r["dtype"],
                int(r["m"]),
                int(r["n"]),
                int(r["is2"]),
                int(r["is1"]),
            )
            for r in rows
        )
        self.assertEqual(expected, actual)
        points = []
        for r in rows:
            d = r["dtype"]
            m = int(r["m"])
            n = int(r["n"])
            i = int(r["is1"])
            outer = int(r["is2"])
            size = p.DTYPE_SIZE[d]
            base = r4f.one_d_base(self.model, d, m * n * size, 1)
            correction = r4f.one_d_gm_correction(self.model, d, n * size, i, 1)
            value = (
                base
                + correction * (1 + 0.02 * m + 0.1 * outer + 0.003 * m * outer)
                + (n % 13) * 0.3
            )
            r["actual_y"] = str(value)
            points.append(
                dict(
                    dtype=d,
                    m=m,
                    n=n,
                    is1=i,
                    is2=outer,
                    actual_2d_cycles=value,
                    config_id=r["config_id"],
                    data_group=r["group_id"],
                )
            )
        ref = self.ref.one_d.fit_ng2_models(points)
        new = r4f.fit_model(rows, self.model, "fixture")
        for d in p.DTYPE_SIZE:
            np.testing.assert_allclose(
                list(new["parameters"]["N_G2"][d].values()),
                [
                    ref[d]["parameters"][name]
                    for name in ("g10", "g11_per_m", "g00", "g01_per_m")
                ],
                rtol=1e-9,
                atol=1e-9,
            )
        print("N_G2 grid:", dict(Counter(r["group_id"] for r in rows)))

    def test_multidim_unscaled_predictions(self):
        for dims in ((2, 3), (2, 3, 4), (2, 3, 4, 5), (2, 3, 4, 5, 6)):
            for dtype in r3f.DTYPE_SIZES:
                for core in (1, 2, 8, 56):
                    for gm, ub in [
                        (
                            tuple(range(1, len(dims) + 1)),
                            tuple(range(2, len(dims) + 2)),
                        ),
                        (r3c.continuous_stride(tuple(reversed(dims))),) * 2,
                    ]:
                        row = dict(
                            dim=str(len(dims)),
                            dtype=dtype,
                            block_dim=str(core),
                            output_dims="x".join(map(str, dims)),
                            input_stride="x".join(map(str, gm)),
                            output_stride="x".join(map(str, ub)),
                        )
                        base, terms = r3f.inherited_terms(self.model, row)
                        total = int(np.prod(dims))
                        ref_base, _ = self.ref.one_dimensional_base_and_t1(
                            dtype, total * r3f.DTYPE_SIZES[dtype], gm[-1], ub[-1], core
                        )
                        ref_terms = self.ref.n1_expansion_terms(
                            dtype,
                            total,
                            tuple(reversed(dims)),
                            tuple(reversed(gm)),
                            tuple(reversed(ub)),
                            core,
                        )
                        self.assertAlmostEqual(base, ref_base)
                        np.testing.assert_allclose(
                            [t["t1_cycles"] for t in terms],
                            [t["t1_cycles"] for t in ref_terms],
                            rtol=1e-12,
                        )


if __name__ == "__main__":
    unittest.main()
