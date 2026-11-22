#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


from __future__ import annotations

import argparse
import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MODELING_DIR = SCRIPT_DIR.parents[2]
DEFAULT_INPUT_DIR = MODELING_DIR / "Ana" / "round3"
DEFAULT_OUTPUT_DIR = MODELING_DIR / "Ana" / "round3" / "figures"
PREDICTIONS_FILENAME = "round3_multidim_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw Round3 multidimensional diagnostics."
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def draw_svg(
    rows: list[dict[str, str]],
    output: Path,
    title: str,
    residual: bool,
    prediction_field: str = "predicted_cycles",
) -> None:
    width, height = 1120, 620
    left, top, right, bottom = 90, 36, 250 if residual else 30, 72
    plot_w, plot_h = width - left - right, height - top - bottom
    xs = [float(row["data_volume_bytes"]) for row in rows]
    if residual:
        ys = [
            (float(row[prediction_field]) - float(row["actual_cycles"]))
            / float(row["actual_cycles"])
            for row in rows
            if float(row["actual_cycles"]) != 0.0
        ]
        ylabel = "(predicted - actual) / actual"
    else:
        ys = [float(row["actual_cycles"]) for row in rows] + [
            float(row[prediction_field]) for row in rows
        ]
        ylabel = "cycles"
    if not ys:
        return
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad = max((y_max - y_min) * 0.08, 0.01 if residual else 1.0)
    y_min -= pad
    y_max += pad

    def px(value: float) -> float:
        return left + (value - x_min) / max(1.0, x_max - x_min) * plot_w

    def py(value: float) -> float:
        return top + (y_max - value) / max(1.0, y_max - y_min) * plot_h

    colors = {
        "continuous": "#2563eb",
        "transpose_outer": "#dc2626",
        "transpose_outer_ub_gap": "#ea580c",
        "transpose_dim2": "#16a34a",
        "transpose_dim1_dim3": "#9333ea",
        "transpose_3d": "#0891b2",
        "gm_noncontiguous": "#ca8a04",
        "transpose_dim1_dim4": "#be123c",
        "transpose_dim3_dim4": "#15803d",
        "transpose_dim1_dim5": "#7c3aed",
        "transpose_dim1_dim2": "#0f766e",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        "<style>text{font-family:Arial,sans-serif;font-size:13px;fill:#1f2937}"
        ".axis{stroke:#374151}.actual{fill:#111827}.predicted{fill:#fff;stroke:#2563eb;"
        "stroke-width:2}.error{fill-opacity:.75}</style>",
        f'<text x="{width / 2}" y="23" text-anchor="middle">{title}</text>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
        f'<text x="{left + plot_w / 2}" y="{height - 20}" text-anchor="middle">data volume bytes</text>',
        f'<text x="18" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 18 {top + plot_h / 2})">{ylabel}</text>',
        f'<text x="{left}" y="{top + plot_h + 22}">{x_min:.0f}</text>',
        f'<text x="{left + plot_w}" y="{top + plot_h + 22}" text-anchor="end">{x_max:.0f}</text>',
        f'<text x="{left - 8}" y="{py(y_min) + 4}" text-anchor="end">{y_min:.4g}</text>',
        f'<text x="{left - 8}" y="{py(y_max) + 4}" text-anchor="end">{y_max:.4g}</text>',
    ]
    if residual and y_min < 0 < y_max:
        parts.append(
            f'<line x1="{left}" y1="{py(0):.2f}" x2="{left + plot_w}" y2="{py(0):.2f}" stroke="#9ca3af" stroke-dasharray="6 5"/>'
        )
    for row in rows:
        color = colors.get(row["layout"], "#4b5563")
        x = px(float(row["data_volume_bytes"]))
        if residual:
            actual = float(row["actual_cycles"])
            if actual == 0.0:
                continue
            relative_error = (float(row[prediction_field]) - actual) / actual
            parts.append(
                f'<circle class="error" cx="{x:.2f}" cy="{py(relative_error):.2f}" '
                f'r="3.5" fill="{color}"><title>{row["layout"]}; bytes={float(row["data_volume_bytes"]):.0f}; rel_error={relative_error:.5g}</title></circle>'
            )
        else:
            parts.append(
                f'<circle class="actual" cx="{x:.2f}" cy="{py(float(row["actual_cycles"])):.2f}" r="3"/>'
            )
            parts.append(
                f'<circle class="predicted" cx="{x:.2f}" cy="{py(float(row[prediction_field])):.2f}" r="3"/>'
            )
    if residual:
        legend_x = left + plot_w + 30
        legend_y = top + 18
        layouts = sorted({row["layout"] for row in rows})
        parts.append(
            f'<text x="{legend_x}" y="{legend_y}" font-family="sans-serif" font-size="14" font-weight="bold">layout</text>'
        )
        for index, layout in enumerate(layouts):
            color = colors.get(layout, "#4b5563")
            y = legend_y + 26 + index * 24
            parts.extend(
                [
                    f'<circle cx="{legend_x + 8}" cy="{y - 5}" r="4" fill="{color}"/>',
                    f'<text x="{legend_x + 22}" y="{y}" font-family="sans-serif" font-size="13">{layout}</text>',
                ]
            )
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def draw_comparison_svg(rows: list[dict[str, str]], output: Path, title: str) -> None:
    width, height = 1220, 620
    left, top, right, bottom = 90, 36, 300, 72
    plot_w, plot_h = width - left - right, height - top - bottom
    points = []
    for row in rows:
        actual = float(row["actual_cycles"])
        if actual == 0.0:
            continue
        points.extend(
            [
                (
                    float(row["data_volume_bytes"]),
                    (float(row["predicted_cycles"]) - actual) / actual,
                    "rho",
                ),
                (
                    float(row["data_volume_bytes"]),
                    (float(row["baseline_predicted_cycles"]) - actual) / actual,
                    "baseline",
                ),
            ]
        )
    if not points:
        return
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad = max((y_max - y_min) * 0.08, 0.01)
    y_min -= pad
    y_max += pad

    def px(value: float) -> float:
        return left + (value - x_min) / max(1.0, x_max - x_min) * plot_w

    def py(value: float) -> float:
        return top + (y_max - value) / max(1.0, y_max - y_min) * plot_h

    colors = {"rho": "#2563eb", "baseline": "#dc2626"}
    labels = {"rho": "with core-related rho", "baseline": "without core-related rho"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        "<style>text{font-family:Arial,sans-serif;font-size:13px}.axis{stroke:#374151}</style>",
        f'<text x="{width / 2}" y="23" text-anchor="middle">{title}</text>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
        f'<text x="{left + plot_w / 2}" y="{height - 20}" text-anchor="middle">data volume bytes</text>',
        f'<text x="18" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 18 {top + plot_h / 2})">(predicted - actual) / actual</text>',
        f'<text x="{left - 8}" y="{py(y_min) + 4}" text-anchor="end">{y_min:.4g}</text>',
        f'<text x="{left - 8}" y="{py(y_max) + 4}" text-anchor="end">{y_max:.4g}</text>',
    ]
    if y_min < 0 < y_max:
        parts.append(
            f'<line x1="{left}" y1="{py(0):.2f}" x2="{left + plot_w}" y2="{py(0):.2f}" stroke="#9ca3af" stroke-dasharray="6 5"/>'
        )
    for x_value, y_value, kind in points:
        color = colors[kind]
        parts.append(
            f'<circle cx="{px(x_value):.2f}" cy="{py(y_value):.2f}" r="3.5" fill="{color}" fill-opacity=".72"><title>{labels[kind]}; bytes={x_value:.0f}; rel_error={y_value:.5g}</title></circle>'
        )
    legend_x = left + plot_w + 30
    legend_y = top + 22
    for index, kind in enumerate(("rho", "baseline")):
        y = legend_y + index * 30
        parts.extend(
            [
                f'<circle cx="{legend_x + 8}" cy="{y - 5}" r="4" fill="{colors[kind]}"/>',
                f'<text x="{legend_x + 22}" y="{y}">{labels[kind]}</text>',
            ]
        )
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir else DEFAULT_OUTPUT_DIR
    )
    with (input_dir / PREDICTIONS_FILENAME).open(
        newline="", encoding="utf-8"
    ) as file_obj:
        rows = list(csv.DictReader(file_obj))
    for dtype in sorted({row["dtype"] for row in rows}):
        for dim in sorted(
            {row["dim"] for row in rows if row["dtype"] == dtype}, key=int
        ):
            selected = [
                row for row in rows if row["dtype"] == dtype and row["dim"] == dim
            ]
            stem = f"round3_d{dim}_{dtype}"
            legacy_actual_vs_predicted = output_dir / f"{stem}_actual_vs_predicted.svg"
            if legacy_actual_vs_predicted.exists():
                legacy_actual_vs_predicted.unlink()
            draw_svg(
                selected,
                output_dir / f"{stem}_residual.svg",
                f"Round3 {dim}D {dtype}: residual",
                True,
            )
            draw_svg(
                selected,
                output_dir / f"{stem}_baseline_residual.svg",
                f"Round3 {dim}D {dtype}: residual without core-related rho",
                True,
                prediction_field="baseline_predicted_cycles",
            )
            draw_comparison_svg(
                selected,
                output_dir / f"{stem}_rho_comparison.svg",
                f"Round3 {dim}D {dtype}: core-related rho comparison",
            )
    print(f"[INFO] wrote plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
