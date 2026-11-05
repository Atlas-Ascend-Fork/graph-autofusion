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
DEFAULT_INPUT_DIR = MODELING_DIR / "Ana" / "round4"
DEFAULT_OUTPUT_DIR = MODELING_DIR / "Ana" / "round4" / "figures"
PREDICTIONS_FILENAME = "round4_2d_ub_contiguous_ng2_predictions.csv"
PALETTE = (
    "#b91c1c",
    "#0369a1",
    "#15803d",
    "#7c3aed",
    "#c2410c",
    "#0f766e",
    "#be185d",
    "#4338ca",
    "#4d7c0f",
    "#a16207",
    "#0e7490",
    "#6d28d9",
    "#374151",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw Round4 Round5 N_G2 2D UB-contiguous diagnostics."
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def draw(rows: list[dict[str, str]], output: Path, title: str, residual: bool) -> None:
    width, height = 920, 580
    left, top, right, bottom = 90, 36, 30, 72
    plot_w, plot_h = width - left - right, height - top - bottom
    xs = [float(row["logical_total_bytes"]) for row in rows]
    if residual:
        ys = [
            (float(row["predicted_cycles"]) - float(row["actual_cycles"]))
            / float(row["actual_cycles"])
            for row in rows
            if float(row["actual_cycles"]) != 0.0
        ]
        ylabel = "(predicted - actual) / actual"
    else:
        ys = [float(row["actual_cycles"]) for row in rows] + [
            float(row["predicted_cycles"]) for row in rows
        ]
        ylabel = "cycles"
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad = max((y_max - y_min) * 0.08, 0.01 if residual else 1.0)
    y_min -= pad
    y_max += pad

    def px(value: float) -> float:
        return left + (value - x_min) / max(1.0, x_max - x_min) * plot_w

    def py(value: float) -> float:
        return top + (y_max - value) / max(1.0, y_max - y_min) * plot_h

    def color(block_dim: int) -> str:
        palette = {
            2: "#2563eb",
            4: "#16a34a",
            8: "#ca8a04",
            32: "#dc2626",
            64: "#7c3aed",
        }
        return palette.get(block_dim, "#4b5563")

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        "<style>text{font-family:Arial,sans-serif;font-size:13px;fill:#1f2937}"
        ".axis{stroke:#374151}.actual{fill:#111827}.predicted{fill:#fff;stroke:#2563eb;"
        "stroke-width:2}.error{fill-opacity:.75}</style>",
        f'<text x="{width / 2}" y="23" text-anchor="middle">{title}</text>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
        f'<text x="{left + plot_w / 2}" y="{height - 20}" text-anchor="middle">logical total bytes</text>',
        f'<text x="18" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 18 {top + plot_h / 2})">{ylabel}</text>',
        f'<text x="{left}" y="{top + plot_h + 22}">{x_min:.0f}</text>',
        f'<text x="{left + plot_w}" y="{top + plot_h + 22}" text-anchor="end">{x_max:.0f}</text>',
        f'<text x="{left - 8}" y="{py(y_min) + 4}" text-anchor="end">{y_min:.4g}</text>',
        f'<text x="{left - 8}" y="{py(y_max) + 4}" text-anchor="end">{y_max:.4g}</text>',
    ]
    if residual and y_min < 0 < y_max:
        parts.append(
            f'<line x1="{left}" y1="{py(0):.2f}" x2="{left + plot_w}" y2="{py(0):.2f}" stroke="#9ca3af"/>'
        )
    for row in rows:
        block_dim = int(row["block_dim"])
        x = px(float(row["logical_total_bytes"]))
        if residual:
            actual = float(row["actual_cycles"])
            if actual == 0.0:
                continue
            relative_error = (float(row["predicted_cycles"]) - actual) / actual
            parts.append(
                f'<circle class="error" cx="{x:.2f}" cy="{py(relative_error):.2f}" '
                f'r="3" fill="{color(block_dim)}"><title>k={block_dim}; rel_error={relative_error:.5g}</title></circle>'
            )
        else:
            parts.append(
                f'<circle class="actual" cx="{x:.2f}" cy="{py(float(row["actual_cycles"])):.2f}" r="3"/>'
            )
            parts.append(
                f'<circle class="predicted" cx="{x:.2f}" cy="{py(float(row["predicted_cycles"])):.2f}" r="3"/>'
            )
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def observed_ng2_vs_m(rows: list[dict[str, str]], output: Path, title: str) -> None:
    selected = [row for row in rows if abs(float(row["n_g1_cycles"])) > 1e-12]
    if not selected:
        return

    width, height = 1120, 660
    left, top, right, bottom = 95, 42, 260, 82
    plot_w, plot_h = width - left - right, height - top - bottom
    xs = [float(row["m"]) for row in selected]
    ys = [
        (float(row["actual_cycles"]) - float(row["n_base_cycles"]))
        / float(row["n_g1_cycles"])
        for row in selected
    ]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    y_pad = max((y_max - y_min) * 0.08, 0.05)
    y_min -= y_pad
    y_max += y_pad
    is_values = sorted({int(float(row["is2"])) for row in selected})
    colors = {
        value: PALETTE[index % len(PALETTE)] for index, value in enumerate(is_values)
    }

    def px(value: float) -> float:
        return left + (value - x_min) / max(1.0, x_max - x_min) * plot_w

    def py(value: float) -> float:
        return top + (y_max - value) / max(1.0, y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<defs><clipPath id="ng2-clip"><rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}"/></clipPath></defs>',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left + plot_w / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{title}</text>',
        f'<text x="22" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 22 {top + plot_h / 2})" font-family="sans-serif" font-size="15">(N2-N_base)/N_G1</text>',
        f'<text x="{left + plot_w / 2}" y="{height - 22}" text-anchor="middle" font-family="sans-serif" font-size="15">M</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#555"/>',
    ]
    for index in range(6):
        fraction = index / 5
        x_value = x_min + fraction * (x_max - x_min)
        y_value = y_min + fraction * (y_max - y_min)
        x = px(x_value)
        y = py(y_value)
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#f3f4f6"/>',
                f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 6}" stroke="#555"/>',
                f'<text x="{x:.2f}" y="{top + plot_h + 28}" text-anchor="middle" font-family="sans-serif" font-size="13">{x_value:.5g}</text>',
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>',
                f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#555"/>',
                f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="13">{y_value:.5g}</text>',
            ]
        )

    parts.append('<g clip-path="url(#ng2-clip)">')
    for is2 in is_values:
        color = colors[is2]
        is_rows = [row for row in selected if int(float(row["is2"])) == is2]
        medians = []
        for m in sorted({int(float(row["m"])) for row in is_rows}):
            values = sorted(
                (float(row["actual_cycles"]) - float(row["n_base_cycles"]))
                / float(row["n_g1_cycles"])
                for row in is_rows
                if int(float(row["m"])) == m
            )
            mid = len(values) // 2
            median = (
                values[mid]
                if len(values) % 2
                else (values[mid - 1] + values[mid]) / 2.0
            )
            medians.append((m, median))
        if len(medians) > 1:
            points = " ".join(f"{px(m):.2f},{py(value):.2f}" for m, value in medians)
            parts.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2"/>'
            )
        for row in is_rows:
            observed = (
                float(row["actual_cycles"]) - float(row["n_base_cycles"])
            ) / float(row["n_g1_cycles"])
            parts.append(
                f'<circle cx="{px(float(row["m"])):.2f}" cy="{py(observed):.2f}" r="3" fill="{color}" fill-opacity="0.45"><title>is2={is2}; M={row["m"]}; N={row["n"]}; k={row["block_dim"]}; observed={observed:.5g}</title></circle>'
            )
        for m, observed in medians:
            parts.append(
                f'<circle cx="{px(m):.2f}" cy="{py(observed):.2f}" r="4" fill="{color}"><title>is2={is2}; M={m}; median observed={observed:.5g}</title></circle>'
            )
    parts.append("</g>")

    legend_x = left + plot_w + 30
    legend_y = top + 18
    parts.append(
        f'<text x="{legend_x}" y="{legend_y}" font-family="sans-serif" font-size="14" font-weight="bold">is2</text>'
    )
    for index, is2 in enumerate(is_values):
        y = legend_y + 26 + index * 24
        color = colors[is2]
        parts.extend(
            [
                f'<line x1="{legend_x}" y1="{y - 5}" x2="{legend_x + 24}" y2="{y - 5}" stroke="{color}" stroke-width="2.4"/>',
                f'<text x="{legend_x + 32}" y="{y}" font-family="sans-serif" font-size="13">{is2}</text>',
            ]
        )
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def error_vs_bytes(rows: list[dict[str, str]], output: Path, title: str) -> None:
    width, height = 1120, 660
    left, top, right, bottom = 95, 42, 260, 82
    plot_w, plot_h = width - left - right, height - top - bottom
    xs = [float(row["bytes_per_core"]) for row in rows]
    ys = [
        (float(row["predicted_cycles"]) - float(row["actual_cycles"]))
        / float(row["actual_cycles"])
        for row in rows
        if float(row["actual_cycles"]) != 0.0
    ]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    y_pad = max((y_max - y_min) * 0.08, 0.01)
    y_min -= y_pad
    y_max += y_pad
    block_dims = sorted({int(float(row["block_dim"])) for row in rows})
    colors = {
        value: PALETTE[index % len(PALETTE)] for index, value in enumerate(block_dims)
    }

    def px(value: float) -> float:
        return left + (value - x_min) / max(1.0, x_max - x_min) * plot_w

    def py(value: float) -> float:
        return top + (y_max - value) / max(1.0, y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<defs><clipPath id="error-bytes-clip"><rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}"/></clipPath></defs>',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left + plot_w / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{title}</text>',
        f'<text x="22" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 22 {top + plot_h / 2})" font-family="sans-serif" font-size="15">(predicted - actual) / actual</text>',
        f'<text x="{left + plot_w / 2}" y="{height - 22}" text-anchor="middle" font-family="sans-serif" font-size="15">bytes_per_core</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#555"/>',
    ]
    for index in range(6):
        fraction = index / 5
        x_value = x_min + fraction * (x_max - x_min)
        y_value = y_min + fraction * (y_max - y_min)
        x = px(x_value)
        y = py(y_value)
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#f3f4f6"/>',
                f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 6}" stroke="#555"/>',
                f'<text x="{x:.2f}" y="{top + plot_h + 28}" text-anchor="middle" font-family="sans-serif" font-size="13">{x_value:.5g}</text>',
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>',
                f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#555"/>',
                f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="13">{y_value:.5g}</text>',
            ]
        )
    if y_min < 0 < y_max:
        parts.append(
            f'<line x1="{left}" y1="{py(0):.2f}" x2="{left + plot_w}" y2="{py(0):.2f}" stroke="#9ca3af" stroke-dasharray="6 5"/>'
        )
    parts.append('<g clip-path="url(#error-bytes-clip)">')
    for row in rows:
        actual = float(row["actual_cycles"])
        if actual == 0.0:
            continue
        block_dim = int(float(row["block_dim"]))
        color = colors[block_dim]
        relative_error = (float(row["predicted_cycles"]) - actual) / actual
        parts.append(
            f'<circle cx="{px(float(row["bytes_per_core"])):.2f}" cy="{py(relative_error):.2f}" r="3.5" fill="{color}" fill-opacity="0.58"><title>k={block_dim}; bytes={float(row["bytes_per_core"]):.0f}; M={row["m"]}; is2={row["is2"]}; rel_error={relative_error:.5g}</title></circle>'
        )
    parts.append("</g>")

    legend_x = left + plot_w + 30
    legend_y = top + 18
    parts.append(
        f'<text x="{legend_x}" y="{legend_y}" font-family="sans-serif" font-size="14" font-weight="bold">block_dim</text>'
    )
    for index, block_dim in enumerate(block_dims):
        y = legend_y + 26 + index * 24
        color = colors[block_dim]
        parts.extend(
            [
                f'<circle cx="{legend_x + 8}" cy="{y - 5}" r="4" fill="{color}" fill-opacity="0.75"/>',
                f'<text x="{legend_x + 22}" y="{y}" font-family="sans-serif" font-size="13">{block_dim}</text>',
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
        selected = [row for row in rows if row["dtype"] == dtype]
        draw(
            selected,
            output_dir / f"round4_2d_ub_contiguous_ng2_{dtype}_actual_vs_predicted.svg",
            f"Round4 Round5 N_G2 2D UB-contiguous {dtype}: actual vs predicted",
            False,
        )
        draw(
            selected,
            output_dir / f"round4_2d_ub_contiguous_ng2_{dtype}_residual.svg",
            f"Round4 Round5 N_G2 2D UB-contiguous {dtype}: residual",
            True,
        )
        observed_ng2_vs_m(
            selected,
            output_dir / f"round4_2d_ub_contiguous_ng2_{dtype}_observed_ng2_vs_m.svg",
            f"Round4 Round5 N_G2 2D UB-contiguous {dtype}: observed N_G2 vs M",
        )
        error_vs_bytes(
            selected,
            output_dir / f"round4_2d_ub_contiguous_ng2_{dtype}_error_vs_bytes.svg",
            f"Round4 Round5 N_G2 2D UB-contiguous {dtype}: relative error vs bytes",
        )
    print(f"[INFO] wrote plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
