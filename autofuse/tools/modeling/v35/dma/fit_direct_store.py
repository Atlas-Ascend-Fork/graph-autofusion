# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Fit the direct-Store MTE2 residual without using holdout/known candidates."""

import argparse
import json
from pathlib import Path


def isolated_cost(row):
    """Existing FP32 packed NG2 large-inner formula, unchanged by this fit."""
    shared = max(int(row["block_dim"]) - 32, 0) / 24
    return (
        194.04230121257388
        + (4640.0150678912751 + 189.36469434493978 * shared)
        * int(row["bytes_per_call"])
        / 65536
    )


def in_domain(row):
    m, n = map(int, row["dims"].split("x"))
    return (
        row["dtype"] == "float32"
        and int(row["dtype_size"]) == 4
        and 2 <= m <= 8
        and 512 <= n <= 16384
        and n % 8 == 0
        and 8192 <= int(row["bytes_per_call"]) == m * n * 4 <= 131072
        and row["gm_or_input_strides"] == f"1x{m}"
        and row["ub_or_output_strides"] == f"{n}x1"
    )


def fit(rows):
    train = [row for row in rows if row["role"] == "train" and in_domain(row)]
    if not train:
        raise ValueError("No training samples")
    best = None
    # A single positive hinge, fitted against relative MTE2 error. Neither
    # independent holdout nor known target candidates enter model selection.
    upper = max(int(row["bytes_per_call"]) * int(row["block_dim"]) for row in train)
    for threshold in range(0, upper + 1, 16384):
        x = [
            max(0, int(r["bytes_per_call"]) * int(r["block_dim"]) - threshold)
            / r["cycles_per_api"]
            for r in train
        ]
        y = [1 - isolated_cost(r) / r["cycles_per_api"] for r in train]
        norm = sum(v * v for v in x)
        coefficient = max(0, sum(a * b for a, b in zip(x, y)) / norm) if norm else 0
        loss = sum((a * coefficient - b) ** 2 for a, b in zip(x, y))
        if best is None or loss < best[0]:
            best = (loss, threshold, coefficient)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--plot", type=Path, help="Optional before/after figure (requires matplotlib)"
    )
    args = parser.parse_args()
    if args.plot is not None and args.plot.exists():
        raise FileExistsError(args.plot)
    rows = json.loads(args.samples.read_text())
    for row in rows:
        if (
            row["role"] not in {"train", "holdout", "known"}
            or row["cycles_per_api"] <= 0
        ):
            raise ValueError("Invalid sample role or cycles")
    loss, threshold, coefficient = fit(rows)
    predictions = []
    for row in rows:
        old = isolated_cost(row)
        applies = in_domain(row)
        predicted = old + (coefficient if applies else 0) * max(
            0, int(row["bytes_per_call"]) * int(row["block_dim"]) - threshold
        )
        predictions.append(
            {
                "id": row["factor_id"],
                "role": row["role"],
                "applies": applies,
                "measured": row["cycles_per_api"],
                "isolated_model": old,
                "predicted": predicted,
                "relative_error": predicted / row["cycles_per_api"] - 1,
            }
        )
    with args.output.open("x") as stream:
        json.dump(
            {
                "train_only": True,
                "threshold": threshold,
                "coefficient": coefficient,
                "loss": loss,
                "predictions": predictions,
            },
            stream,
            indent=2,
        )

    if args.plot is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        for role, marker in [("train", "o"), ("holdout", "s"), ("known", "^")]:
            selected = [r for r in predictions if r["applies"] and r["role"] == role]
            for axis, field in zip(axes, ["isolated_model", "predicted"]):
                axis.scatter(
                    [r["measured"] for r in selected],
                    [r[field] for r in selected],
                    label=role,
                    marker=marker,
                )
        upper = (
            max(max(r["measured"], r["predicted"]) for r in predictions if r["applies"])
            * 1.1
        )
        for axis, title in zip(
            axes, ["Isolated API model", "Direct-store effective MTE2"]
        ):
            axis.plot([0, upper], [0, upper], "k--", linewidth=1)
            axis.set(
                xlabel="Measured cycles/call",
                ylabel="Estimated cycles/call",
                title=title,
            )
            axis.grid(alpha=0.2)
            axis.legend()
        fig.savefig(args.plot)
        plt.close(fig)


if __name__ == "__main__":
    main()
