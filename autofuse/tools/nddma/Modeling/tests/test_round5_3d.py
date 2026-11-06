# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""3D refit recovery, inherited Base invariance and complete workflow checks."""

from collections import Counter
from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "Data_collection/round5/scripts"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fit = load("round5_fit_test", SCRIPTS / "fit.py")
collect = load("round5_collect_test", SCRIPTS / "collect.py")


def fixture():
    truth = {"parameters": {}}
    for dtype in fit.common.DTYPE_SIZES:
        truth["parameters"][dtype] = {
            "base": {"T_1": 20.0, "H_1": 100.0, "T_2": 10.0, "H_2": 200.0},
            "N_G": {"a1": 2.0, "a2": 0.01},
            "N_GU": {"b1": 3.0, "b2": 0.1, "b3": 0.02, "b4": 0.001},
            "rho": {"c1": 1.2, "c2": 0.001, "c3": 0.3, "c4": 0.002},
        }
    prior = deepcopy(truth)
    for params in prior["parameters"].values():
        for section in ("N_G", "N_GU", "rho"):
            for key in params[section]:
                params[section][key] *= 1.7
    rng = random.Random(531)
    rows = []
    for dtype in fit.common.DTYPE_SIZES:
        for index in range(80):
            dims = tuple(rng.choice((2, 4, 8, 16, 32)) for _ in range(3))
            gm = tuple(rng.choice((1, 2, 4, 16, 64, 128, 512)) for _ in range(3))
            ub = tuple(rng.choice((1, 2, 4, 8, 32, 128)) for _ in range(3))
            if index % 3 == 0:
                ub = (dims[1] * dims[2], dims[2], 1)
            for core in (1, 2, 8, 32):
                row = dict(
                    dim="3",
                    dtype=dtype,
                    block_dim=str(core),
                    config_id=f"{dtype}_{index}",  # Same token across cores deliberately.
                    output_dims="x".join(map(str, dims)),
                    input_stride="x".join(map(str, gm)),
                    output_stride="x".join(map(str, ub)),
                    source_round="synthetic",
                )
                base, terms = fit.common.inherited_terms(truth, row)
                row["actual_y"] = str(base + sum(term["t1_cycles"] for term in terms))
                rows.append(row)
    return prior, truth, rows


class Round5Tests(unittest.TestCase):
    def test_recovers_all_corrections_and_preserves_base(self):
        prior, truth, rows = fixture()
        unchanged = deepcopy(prior)
        result = fit.fit_model(rows, prior, "fixture")
        self.assertEqual(prior, unchanged)
        for dtype, params in result["parameters"].items():
            self.assertEqual(params["base"], prior["parameters"][dtype]["base"])
            for section in ("N_G", "N_GU", "rho"):
                for name, value in params[section].items():
                    self.assertAlmostEqual(
                        value, truth["parameters"][dtype][section][name], places=8
                    )
            self.assertLess(result["metrics"][dtype]["3"]["mape_percent"], 1e-8)
            self.assertGreater(
                result["baseline_metrics"][dtype]["3"]["mape_percent"], 1.0
            )
        self.assertEqual(result["formula"]["dimensions"], [3])
        for point in result["predictions"]:
            self.assertAlmostEqual(
                point["predicted_cycles"], point["n_base_cycles"] + point["t1_cycles"]
            )

    def test_aggregate_excludes_other_dimensions_and_keeps_core_counts(self):
        _, _, rows = fixture()
        sample = rows[:4]
        repeated = dict(sample[0], actual_y=str(float(sample[0]["actual_y"]) + 20))
        other_dim = dict(sample[0], dim="2", actual_y="invalid")
        actual = fit.aggregate(sample + [repeated, other_dim])
        self.assertEqual(len(actual), 4)
        self.assertEqual({row["block_dim"] for row in actual}, {"1", "2", "8", "32"})
        self.assertAlmostEqual(
            float(actual[0]["actual_y"]), float(sample[0]["actual_y"]) + 10.0
        )
        raw = dict(sample[0])
        raw.pop("actual_y")
        raw.update(repeat="200", mte2_cycles=str(float(sample[0]["actual_y"]) * 200))
        self.assertAlmostEqual(
            float(fit.aggregate([raw])[0]["actual_y"]), float(sample[0]["actual_y"])
        )

    def test_missing_multicore_and_rank_deficiency_are_errors(self):
        prior, _, rows = fixture()
        with self.assertRaisesRegex(ValueError, "need both"):
            fit.fit_model(
                [row for row in rows if row["block_dim"] == "1"], prior, "fixture"
            )
        with self.assertRaisesRegex(ValueError, "rank-deficient"):
            fit.solve([[1.0, 2.0]] * 8, [10.0] * 8, "fixture")

    def test_collection_grid_and_identifiability(self):
        rows = collect.build_rows(3)
        self.assertEqual(len(rows), 178661)
        self.assertEqual(
            Counter(row["source_round"] for row in rows),
            {"round5_j": 18743 * 7, "round6": 6780 * 7},
        )
        self.assertEqual(len({row["token"] for row in rows}), len(rows))
        designs = {dtype: [] for dtype in fit.common.DTYPE_SIZES}
        rho_designs = {dtype: [] for dtype in fit.common.DTYPE_SIZES}
        prior, _, _ = fixture()
        for row in rows:
            self.assertEqual(row["dim"], "3")
            self.assertEqual(row["fit_role"], "train")
            self.assertEqual(
                int(row["logical_total_bytes"]),
                int(row["bytes_per_core"]) * int(row["block_dim"]),
            )
            if row["block_dim"] == "1":
                _, terms = fit.common.inherited_terms(prior, row, use_core_rho=False)
                designs[row["dtype"]].append(
                    fit.correction_features(terms, row["dtype"])
                )
                rho_designs[row["dtype"]].append(fit.rho_features(terms, row["dtype"]))
        for dtype, design in designs.items():
            matrix = np.asarray(design)
            self.assertEqual(
                np.linalg.matrix_rank(matrix / np.max(np.abs(matrix), axis=0)), 6, dtype
            )
            matrix = np.asarray(rho_designs[dtype])
            self.assertEqual(
                np.linalg.matrix_rank(matrix / np.max(np.abs(matrix), axis=0)), 4, dtype
            )

    def test_fit_and_draw_cli_with_existing_csv(self):
        prior, truth, rows = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "round2.json"
            source.write_text(json.dumps(prior), encoding="utf-8")
            data = root / "measurements.csv"
            with data.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            for action in ("fit", "draw"):
                command = [
                    sys.executable,
                    str(SCRIPTS / "e2e.py"),
                    action,
                    f"--model-output-dir={root / 'output'}",
                ]
                if action == "fit":
                    command += [f"--round2-model={source}", f"--measurement-csv={data}"]
                result = subprocess.run(command, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output = root / "output"
            model = json.loads(
                (output / fit.MODEL_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(
                model["parameters"]["int8_t"]["base"],
                truth["parameters"]["int8_t"]["base"],
            )
            self.assertTrue((output / fit.PREDICTIONS_FILENAME).is_file())
            self.assertEqual(len(list((output / "figures").glob("*.svg"))), 32)
            row = rows[0]
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "predict.py"),
                    row["output_dims"],
                    row["input_stride"],
                    row["output_stride"],
                    row["block_dim"],
                    row["dtype"],
                    f"--model={output / fit.MODEL_FILENAME}",
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertAlmostEqual(
                json.loads(result.stdout)["predicted_cycles"],
                float(row["actual_y"]),
                places=7,
            )


if __name__ == "__main__":
    unittest.main()
