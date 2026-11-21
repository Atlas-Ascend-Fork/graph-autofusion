#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Predict a 3D configuration with a fitted Round5 model."""

import argparse
import importlib.util
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "round5_predict_fit", SCRIPT_DIR / "fit.py"
)
fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dims", help="API order, e.g. 4x8x16")
    parser.add_argument("input_stride")
    parser.add_argument("output_stride")
    parser.add_argument("block_dim", type=int)
    parser.add_argument("dtype", choices=tuple(fit.common.DTYPE_SIZES))
    parser.add_argument(
        "--model", default=str(fit.DEFAULT_ANA_DIR / fit.MODEL_FILENAME)
    )
    args = parser.parse_args()
    if args.block_dim < 1:
        parser.error("block_dim must be positive")
    row = {
        "dim": "3",
        "dtype": args.dtype,
        "block_dim": str(args.block_dim),
        "output_dims": args.output_dims,
        "input_stride": args.input_stride,
        "output_stride": args.output_stride,
        "actual_y": "1",
    }
    row = fit.aggregate([row])[0]
    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    if model.get("model") != "NDDMA2_ROUND5_3D_SPECIALIZED":
        parser.error("--model must be a fitted Round5 3D specialization")
    base, terms = fit.common.inherited_terms(model, row)
    correction = sum(float(term["t1_cycles"]) for term in terms)
    print(
        json.dumps(
            {
                "n_base_cycles": base,
                "correction_cycles": correction,
                "predicted_cycles": base + correction,
                "axis_terms": terms,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
