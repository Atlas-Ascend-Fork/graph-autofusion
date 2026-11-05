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
DEFAULT_INPUT_DIR = MODELING_DIR / "Ana" / "round2"
DEFAULT_OUTPUT_DIR = MODELING_DIR / "Ana" / "round2" / "figures"
PREDICTIONS_FILENAME = "round2_1d_noncontiguous_predictions.csv"
DTYPES = ("int8_t", "int16_t", "int32_t", "int64_t")
DTYPE_SIZES = {"int8_t": 1, "int16_t": 2, "int32_t": 4, "int64_t": 8}
INPUT_STRIDE_LABEL_MIN = {"int8_t": 128, "int16_t": 64, "int32_t": 256, "int64_t": 128}
INPUT_STRIDE_EXTRA_LABELS = {"int32_t": {32}, "int64_t": {16}}
RHO_PLOT_INPUT_STRIDES = (1, 2, 4, 16, 64, 128, 256)
RHO_PLOT_COLORS = {
    1: "#b91c1c",
    2: "#0369a1",
    4: "#15803d",
    16: "#7c3aed",
    64: "#c2410c",
    128: "#0f766e",
    256: "#be185d",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw Round2 1D model diagnostics.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def residual_relative_svg(dtype: str, rows: list[dict[str, str]], output: Path) -> None:
    selected = [row for row in rows if row["dtype"] == dtype]
    if not selected:
        return

    width, height = 1220, 700
    left, top, right, bottom = 105, 50, 310, 92
    plot_w, plot_h = width - left - right, height - top - bottom
    xs = [float(row["bytes_per_core"]) for row in selected]
    ys = [
        (float(row["predicted_cycles"]) - float(row["actual_cycles"]))
        / float(row["actual_cycles"])
        for row in selected
        if float(row["actual_cycles"]) != 0.0
    ]
    if not ys:
        return
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    y_pad = max((y_max - y_min) * 0.08, 0.01)
    y_min -= y_pad
    y_max += y_pad
    stride_keys = sorted(
        {
            (int(float(row["input_stride"])), int(float(row["output_stride"])))
            for row in selected
        }
    )
    palette = (
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
    stride_colors = {
        key: palette[index % len(palette)] for index, key in enumerate(stride_keys)
    }

    def px(value: float) -> float:
        return left + (value - x_min) / max(1.0, x_max - x_min) * plot_w

    def py(value: float) -> float:
        return top + (y_max - value) / max(1.0, y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<defs><clipPath id="relative-residual-clip"><rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}"/></clipPath></defs>',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left + plot_w / 2}" y="29" text-anchor="middle" font-family="sans-serif" font-size="20">{dtype}: relative prediction error vs bytes</text>',
        f'<text x="24" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 24 {top + plot_h / 2})" font-family="sans-serif" font-size="18" font-weight="600">(predicted - actual) / actual</text>',
        f'<text x="{left + plot_w / 2}" y="{height - 24}" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="600">bytes_per_core</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#555"/>',
    ]
    for index in range(6):
        fraction = index / 5
        x_value = x_min + fraction * (x_max - x_min)
        x = px(x_value)
        y_value = y_min + fraction * (y_max - y_min)
        y = py(y_value)
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#f3f4f6"/>',
                f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 6}" stroke="#555"/>',
                f'<text x="{x:.2f}" y="{top + plot_h + 30}" text-anchor="middle" font-family="sans-serif" font-size="17">{x_value:.5g}</text>',
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>',
                f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#555"/>',
                f'<text x="{left - 12}" y="{y + 6:.2f}" text-anchor="end" font-family="sans-serif" font-size="17">{y_value:.4g}</text>',
            ]
        )
    if y_min < 0.0 < y_max:
        parts.append(
            f'<line x1="{left}" y1="{py(0.0):.2f}" x2="{left + plot_w}" y2="{py(0.0):.2f}" stroke="#9ca3af" stroke-dasharray="6 5"/>'
        )

    parts.append('<g clip-path="url(#relative-residual-clip)">')
    for row in selected:
        actual = float(row["actual_cycles"])
        if actual == 0.0:
            continue
        predicted = float(row["predicted_cycles"])
        error = (predicted - actual) / actual
        input_stride = int(float(row["input_stride"]))
        output_stride = int(float(row["output_stride"]))
        block_dim = int(float(row["block_dim"]))
        color = stride_colors[(input_stride, output_stride)]
        radius = min(7.0, 2.5 + block_dim**0.35)
        parts.append(
            f'<circle cx="{px(float(row["bytes_per_core"])):.2f}" cy="{py(error):.2f}" r="{radius:.2f}" fill="{color}" fill-opacity="0.42"><title>k={block_dim}; is={input_stride}; os={output_stride}; bytes={float(row["bytes_per_core"]):.0f}; rel_error={error:.5g}</title></circle>'
        )
    parts.append("</g>")

    legend_x = left + plot_w + 35
    legend_y = top + 15
    parts.append(
        f'<text x="{legend_x}" y="{legend_y}" font-family="sans-serif" font-size="16" font-weight="bold">input/output stride</text>'
    )
    for index, key in enumerate(stride_keys[:18]):
        input_stride, output_stride = key
        color = stride_colors[key]
        column, row_index = divmod(index, 9)
        x = legend_x + column * 132
        y = legend_y + 28 + row_index * 27
        parts.extend(
            [
                f'<circle cx="{x + 10}" cy="{y - 5}" r="4" fill="{color}" fill-opacity="0.65"/>',
                f'<text x="{x + 24}" y="{y}" font-family="sans-serif" font-size="14">is={input_stride}, os={output_stride}</text>',
            ]
        )
    if len(stride_keys) > 18:
        parts.append(
            f'<text x="{legend_x}" y="{legend_y + 28 + 9 * 27}" font-family="sans-serif" font-size="14">+{len(stride_keys) - 18} stride groups</text>'
        )
    parts.extend(
        [
            f'<text x="{legend_x}" y="{height - 68}" font-family="sans-serif" font-size="14" font-weight="bold">point size</text>',
            f'<circle cx="{legend_x + 11}" cy="{height - 42}" r="3.5" fill="#6b7280" fill-opacity="0.45"/>',
            f'<text x="{legend_x + 28}" y="{height - 38}" font-family="sans-serif" font-size="13">smaller core count</text>',
            f'<circle cx="{legend_x + 11}" cy="{height - 19}" r="7" fill="#6b7280" fill-opacity="0.45"/>',
            f'<text x="{legend_x + 28}" y="{height - 15}" font-family="sans-serif" font-size="13">larger core count</text>',
        ]
    )
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def output_dim(dtype: str, row: dict[str, str]) -> int:
    return int(round(float(row["bytes_per_core"]) / DTYPE_SIZES[dtype]))


def input_stride_vs_cycles_svg(
    dtype: str,
    rows: list[dict[str, str]],
    output: Path,
) -> None:
    selected = [
        row
        for row in rows
        if row["dtype"] == dtype
        and row["group"] == "A"
        and int(float(row["block_dim"])) == 1
        and int(float(row["output_stride"])) == 1
    ]
    if not selected:
        return

    curves: dict[int, list[dict[str, str]]] = {}
    for row in selected:
        curves.setdefault(output_dim(dtype, row), []).append(row)
    for curve_rows in curves.values():
        curve_rows.sort(key=lambda row: float(row["input_stride"]))

    width, height = 1220, 700
    left, top, right, bottom = 105, 50, 310, 92
    plot_w, plot_h = width - left - right, height - top - bottom
    strides = sorted({float(row["input_stride"]) for row in selected})
    cycles = [float(row["actual_cycles"]) for row in selected]
    x_min, x_max = min(strides), max(strides)
    y_min, y_max = min(cycles), max(cycles)
    y_pad = max((y_max - y_min) * 0.05, 1.0)
    y_min = max(0.0, y_min - y_pad)
    y_max += y_pad
    palette = (
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

    def px(value: float) -> float:
        return left + (value - x_min) / max(1.0, x_max - x_min) * plot_w

    def py(value: float) -> float:
        return top + (y_max - value) / max(1.0, y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<defs><clipPath id="sweep-clip"><rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}"/></clipPath></defs>',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left + plot_w / 2}" y="29" text-anchor="middle" font-family="sans-serif" font-size="20">{dtype}: input stride vs actual cycles</text>',
        f'<text x="24" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 24 {top + plot_h / 2})" font-family="sans-serif" font-size="18" font-weight="600">actual cycles</text>',
        f'<text x="{left + plot_w / 2}" y="{height - 24}" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="600">input stride</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#555"/>',
    ]
    min_labeled_stride = INPUT_STRIDE_LABEL_MIN[dtype]
    extra_labeled_strides = INPUT_STRIDE_EXTRA_LABELS.get(dtype, set())
    for stride in strides:
        x = px(stride)
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#f3f4f6"/>',
                f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 6}" stroke="#555"/>',
            ]
        )
        if stride >= min_labeled_stride or int(stride) in extra_labeled_strides:
            parts.append(
                f'<text class="x-stride-label" x="{x:.2f}" y="{top + plot_h + 30}" text-anchor="middle" font-family="sans-serif" font-size="17">{stride:.0f}</text>'
            )
    for index in range(6):
        fraction = index / 5
        y_value = y_min + fraction * (y_max - y_min)
        y = py(y_value)
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>',
                f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#555"/>',
                f'<text x="{left - 12}" y="{y + 6:.2f}" text-anchor="end" font-family="sans-serif" font-size="17">{y_value:.5g}</text>',
            ]
        )

    parts.append('<g clip-path="url(#sweep-clip)">')
    for curve_index, shape in enumerate(sorted(curves)):
        color = palette[curve_index % len(palette)]
        curve_rows = curves[shape]
        points = " ".join(
            f"{px(float(row['input_stride'])):.2f},{py(float(row['actual_cycles'])):.2f}"
            for row in curve_rows
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.8"/>'
        )
        for row in curve_rows:
            parts.append(
                f'<circle cx="{px(float(row["input_stride"])):.2f}" cy="{py(float(row["actual_cycles"])):.2f}" r="3" fill="{color}"><title>output_dim={shape}; input_stride={float(row["input_stride"]):.0f}; actual={float(row["actual_cycles"]):.4g}</title></circle>'
            )
    parts.append("</g>")

    legend_x = left + plot_w + 35
    legend_y = top + 15
    parts.append(
        f'<text x="{legend_x}" y="{legend_y}" font-family="sans-serif" font-size="16" font-weight="bold">output_dim</text>'
    )
    for curve_index, shape in enumerate(sorted(curves)):
        color = palette[curve_index % len(palette)]
        column, row_index = divmod(curve_index, 12)
        x = legend_x + column * 112
        y = legend_y + 28 + row_index * 27
        parts.extend(
            [
                f'<line x1="{x}" y1="{y - 5}" x2="{x + 24}" y2="{y - 5}" stroke="{color}" stroke-width="2.5"/>',
                f'<text x="{x + 32}" y="{y}" font-family="sans-serif" font-size="14">{shape}</text>',
            ]
        )
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def output_stride_vs_cycles_svg(
    dtype: str,
    rows: list[dict[str, str]],
    output: Path,
) -> None:
    selected = [
        row
        for row in rows
        if row["dtype"] == dtype
        and row["group"] == "B"
        and int(float(row["block_dim"])) == 1
        and int(float(row["input_stride"])) == 1
        and int(float(row["output_stride"])) > 1
    ]
    if not selected:
        return

    curves: dict[float, list[dict[str, str]]] = {}
    for row in selected:
        curves.setdefault(float(row["logical_total_bytes"]), []).append(row)
    for curve_rows in curves.values():
        curve_rows.sort(key=lambda row: float(row["output_stride"]))

    width, height = 1220, 700
    left, top, right, bottom = 105, 50, 310, 92
    plot_w, plot_h = width - left - right, height - top - bottom
    strides = sorted({1.0} | {float(row["output_stride"]) for row in selected})
    cycles = [float(row["actual_cycles"]) for row in selected]
    base_cycles_by_bytes = {
        logical_bytes: float(curve_rows[0]["base_cycles"])
        for logical_bytes, curve_rows in curves.items()
    }
    cycles.extend(base_cycles_by_bytes.values())
    x_min, x_max = min(strides), max(strides)
    y_min, y_max = min(cycles), max(cycles)
    y_pad = max((y_max - y_min) * 0.05, 1.0)
    y_min = max(0.0, y_min - y_pad)
    y_max += y_pad
    palette = (
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

    def px(value: float) -> float:
        return left + (value - x_min) / max(1.0, x_max - x_min) * plot_w

    def py(value: float) -> float:
        return top + (y_max - value) / max(1.0, y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<defs><clipPath id="sweep-clip"><rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}"/></clipPath></defs>',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left + plot_w / 2}" y="29" text-anchor="middle" font-family="sans-serif" font-size="20">{dtype}: input stride=1, output stride vs actual cycles</text>',
        f'<text x="24" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 24 {top + plot_h / 2})" font-family="sans-serif" font-size="18" font-weight="600">actual cycles</text>',
        f'<text x="{left + plot_w / 2}" y="{height - 24}" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="600">output stride</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#555"/>',
    ]
    for stride in strides:
        x = px(stride)
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#f3f4f6"/>',
                f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 6}" stroke="#555"/>',
                f'<text class="x-stride-label" x="{x:.2f}" y="{top + plot_h + 30}" text-anchor="middle" font-family="sans-serif" font-size="17">{stride:.0f}</text>',
            ]
        )
    for index in range(6):
        fraction = index / 5
        y_value = y_min + fraction * (y_max - y_min)
        y = py(y_value)
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>',
                f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#555"/>',
                f'<text x="{left - 12}" y="{y + 6:.2f}" text-anchor="end" font-family="sans-serif" font-size="17">{y_value:.5g}</text>',
            ]
        )

    parts.append('<g clip-path="url(#sweep-clip)">')
    for curve_index, logical_bytes in enumerate(sorted(curves)):
        color = palette[curve_index % len(palette)]
        curve_rows = curves[logical_bytes]
        points = [f"{px(1.0):.2f},{py(base_cycles_by_bytes[logical_bytes]):.2f}"]
        points.extend(
            f"{px(float(row['output_stride'])):.2f},{py(float(row['actual_cycles'])):.2f}"
            for row in curve_rows
        )
        parts.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="1.8"/>'
        )
        parts.append(
            f'<circle cx="{px(1.0):.2f}" cy="{py(base_cycles_by_bytes[logical_bytes]):.2f}" r="4" fill="white" stroke="{color}" stroke-width="2"><title>output_dim={output_dim(dtype, curve_rows[0])}; output_stride=1; base={base_cycles_by_bytes[logical_bytes]:.4g}</title></circle>'
        )
        for row in curve_rows:
            parts.append(
                f'<circle cx="{px(float(row["output_stride"])):.2f}" cy="{py(float(row["actual_cycles"])):.2f}" r="3" fill="{color}"><title>output_dim={output_dim(dtype, row)}; output_stride={float(row["output_stride"]):.0f}; actual={float(row["actual_cycles"]):.4g}</title></circle>'
            )
    parts.append("</g>")

    parts.extend(
        [
            f'<circle cx="{left + 16}" cy="{top + 22}" r="4" fill="white" stroke="#374151" stroke-width="2"/>',
            f'<text x="{left + 30}" y="{top + 27}" font-family="sans-serif" font-size="15">os=1: base anchor; os&gt;=2: measured</text>',
        ]
    )
    legend_x = left + plot_w + 35
    legend_y = top + 15
    parts.append(
        f'<text x="{legend_x}" y="{legend_y}" font-family="sans-serif" font-size="16" font-weight="bold">logical bytes</text>'
    )
    for curve_index, logical_bytes in enumerate(sorted(curves)):
        color = palette[curve_index % len(palette)]
        column, row_index = divmod(curve_index, 12)
        x = legend_x + column * 136
        y = legend_y + 28 + row_index * 27
        shape = output_dim(dtype, curves[logical_bytes][0])
        parts.extend(
            [
                f'<line x1="{x}" y1="{y - 5}" x2="{x + 24}" y2="{y - 5}" stroke="{color}" stroke-width="2.5"/>',
                f'<circle cx="{x + 12}" cy="{y - 5}" r="3" fill="{color}"/>',
                f'<text x="{x + 32}" y="{y}" font-family="sans-serif" font-size="14">{logical_bytes:.0f} B (d={shape})</text>',
            ]
        )
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def os2_output_dim_vs_cycles_svg(
    dtype: str,
    rows: list[dict[str, str]],
    output: Path,
) -> None:
    selected = [
        row
        for row in rows
        if row["dtype"] == dtype
        and row["group"] in {"B", "C"}
        and int(float(row["block_dim"])) == 1
        and int(float(row["output_stride"])) == 2
    ]
    if not selected:
        return

    curves: dict[float, list[dict[str, str]]] = {}
    for row in selected:
        curves.setdefault(float(row["input_stride"]), []).append(row)
    for curve_rows in curves.values():
        curve_rows.sort(key=lambda row: output_dim(dtype, row))

    width, height = 1220, 700
    left, top, right, bottom = 105, 50, 310, 92
    plot_w, plot_h = width - left - right, height - top - bottom
    dims = sorted({output_dim(dtype, row) for row in selected})
    cycles = [float(row["actual_cycles"]) for row in selected]
    x_min, x_max = min(dims), max(dims)
    y_min, y_max = min(cycles), max(cycles)
    y_pad = max((y_max - y_min) * 0.05, 1.0)
    y_min = max(0.0, y_min - y_pad)
    y_max += y_pad
    palette = (
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

    def px(value: float) -> float:
        return left + (value - x_min) / max(1.0, x_max - x_min) * plot_w

    def py(value: float) -> float:
        return top + (y_max - value) / max(1.0, y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<defs><clipPath id="os2-output-dim-clip"><rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}"/></clipPath></defs>',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left + plot_w / 2}" y="29" text-anchor="middle" font-family="sans-serif" font-size="20">{dtype}: output stride=2, output dim vs actual cycles</text>',
        f'<text x="24" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 24 {top + plot_h / 2})" font-family="sans-serif" font-size="18" font-weight="600">actual cycles</text>',
        f'<text x="{left + plot_w / 2}" y="{height - 24}" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="600">output_dim</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#555"/>',
    ]
    pre_512_labels = [dim for dim in dims if dim < 512]
    labeled_pre_512 = max(pre_512_labels) if pre_512_labels else None
    for dim in dims:
        x = px(dim)
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#f3f4f6"/>',
                f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 6}" stroke="#555"/>',
            ]
        )
        if dim >= 512 or dim == labeled_pre_512:
            parts.append(
                f'<text class="x-output-dim-label" x="{x:.2f}" y="{top + plot_h + 30}" text-anchor="middle" font-family="sans-serif" font-size="17">{dim}</text>'
            )
    for index in range(6):
        fraction = index / 5
        y_value = y_min + fraction * (y_max - y_min)
        y = py(y_value)
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>',
                f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#555"/>',
                f'<text x="{left - 12}" y="{y + 6:.2f}" text-anchor="end" font-family="sans-serif" font-size="17">{y_value:.5g}</text>',
            ]
        )

    parts.append('<g clip-path="url(#os2-output-dim-clip)">')
    for curve_index, input_stride in enumerate(sorted(curves)):
        color = palette[curve_index % len(palette)]
        curve_rows = curves[input_stride]
        points = " ".join(
            f"{px(output_dim(dtype, row)):.2f},{py(float(row['actual_cycles'])):.2f}"
            for row in curve_rows
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.8"/>'
        )
        for row in curve_rows:
            dim = output_dim(dtype, row)
            parts.append(
                f'<circle cx="{px(dim):.2f}" cy="{py(float(row["actual_cycles"])):.2f}" r="3" fill="{color}"><title>input_stride={input_stride:.0f}; output_dim={dim}; actual={float(row["actual_cycles"]):.4g}</title></circle>'
            )
    parts.append("</g>")

    legend_x = left + plot_w + 35
    legend_y = top + 15
    parts.append(
        f'<text x="{legend_x}" y="{legend_y}" font-family="sans-serif" font-size="16" font-weight="bold">input_stride</text>'
    )
    for curve_index, input_stride in enumerate(sorted(curves)):
        color = palette[curve_index % len(palette)]
        column, row_index = divmod(curve_index, 12)
        x = legend_x + column * 112
        y = legend_y + 28 + row_index * 27
        parts.extend(
            [
                f'<line x1="{x}" y1="{y - 5}" x2="{x + 24}" y2="{y - 5}" stroke="{color}" stroke-width="2.5"/>',
                f'<text x="{x + 32}" y="{y}" font-family="sans-serif" font-size="14">{input_stride:.0f}</text>',
            ]
        )
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def os1_output_dim_vs_cycles_svg(
    dtype: str,
    rows: list[dict[str, str]],
    output: Path,
) -> None:
    selected = [
        row
        for row in rows
        if row["dtype"] == dtype
        and (
            row["group"] == "A"
            or (row["group"] == "F" and int(float(row["input_stride"])) == 1)
        )
        and int(float(row["block_dim"])) == 1
        and int(float(row["output_stride"])) == 1
    ]
    if not selected:
        return

    curves: dict[float, list[dict[str, str]]] = {}
    for row in selected:
        curves.setdefault(float(row["input_stride"]), []).append(row)
    for curve_rows in curves.values():
        curve_rows.sort(key=lambda row: output_dim(dtype, row))

    width, height = 1220, 700
    left, top, right, bottom = 105, 50, 310, 92
    plot_w, plot_h = width - left - right, height - top - bottom
    dims = sorted({output_dim(dtype, row) for row in selected})
    cycles = [float(row["actual_cycles"]) for row in selected]
    x_min, x_max = min(dims), max(dims)
    y_min, y_max = min(cycles), max(cycles)
    y_pad = max((y_max - y_min) * 0.05, 1.0)
    y_min = max(0.0, y_min - y_pad)
    y_max += y_pad
    palette = (
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

    def px(value: float) -> float:
        return left + (value - x_min) / max(1.0, x_max - x_min) * plot_w

    def py(value: float) -> float:
        return top + (y_max - value) / max(1.0, y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<defs><clipPath id="os1-output-dim-clip"><rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}"/></clipPath></defs>',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left + plot_w / 2}" y="29" text-anchor="middle" font-family="sans-serif" font-size="20">{dtype}: output stride=1, output dim vs actual cycles</text>',
        f'<text x="24" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 24 {top + plot_h / 2})" font-family="sans-serif" font-size="18" font-weight="600">actual cycles</text>',
        f'<text x="{left + plot_w / 2}" y="{height - 24}" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="600">output_dim</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#555"/>',
    ]
    pre_512_labels = [dim for dim in dims if dim < 512]
    labeled_pre_512 = max(pre_512_labels) if pre_512_labels else None
    for dim in dims:
        x = px(dim)
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#f3f4f6"/>',
                f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 6}" stroke="#555"/>',
            ]
        )
        if dim >= 512 or dim == labeled_pre_512:
            parts.append(
                f'<text class="x-output-dim-label" x="{x:.2f}" y="{top + plot_h + 30}" text-anchor="middle" font-family="sans-serif" font-size="17">{dim}</text>'
            )
    for index in range(6):
        fraction = index / 5
        y_value = y_min + fraction * (y_max - y_min)
        y = py(y_value)
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>',
                f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#555"/>',
                f'<text x="{left - 12}" y="{y + 6:.2f}" text-anchor="end" font-family="sans-serif" font-size="17">{y_value:.5g}</text>',
            ]
        )

    parts.append('<g clip-path="url(#os1-output-dim-clip)">')
    for curve_index, input_stride in enumerate(sorted(curves)):
        color = palette[curve_index % len(palette)]
        curve_rows = curves[input_stride]
        points = " ".join(
            f"{px(output_dim(dtype, row)):.2f},{py(float(row['actual_cycles'])):.2f}"
            for row in curve_rows
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.8"/>'
        )
        for row in curve_rows:
            dim = output_dim(dtype, row)
            parts.append(
                f'<circle cx="{px(dim):.2f}" cy="{py(float(row["actual_cycles"])):.2f}" r="3" fill="{color}"><title>input_stride={input_stride:.0f}; output_dim={dim}; actual={float(row["actual_cycles"]):.4g}</title></circle>'
            )
    parts.append("</g>")

    legend_x = left + plot_w + 35
    legend_y = top + 15
    parts.append(
        f'<text x="{legend_x}" y="{legend_y}" font-family="sans-serif" font-size="16" font-weight="bold">input_stride</text>'
    )
    for curve_index, input_stride in enumerate(sorted(curves)):
        color = palette[curve_index % len(palette)]
        column, row_index = divmod(curve_index, 12)
        x = legend_x + column * 112
        y = legend_y + 28 + row_index * 27
        parts.extend(
            [
                f'<line x1="{x}" y1="{y - 5}" x2="{x + 24}" y2="{y - 5}" stroke="{color}" stroke-width="2.5"/>',
                f'<text x="{x + 32}" y="{y}" font-family="sans-serif" font-size="14">{input_stride:.0f}</text>',
            ]
        )
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def os2_rho_vs_cores_svg(
    dtype: str,
    rows: list[dict[str, str]],
    output: Path,
) -> None:
    target_strides = set(RHO_PLOT_INPUT_STRIDES)
    points_by_stride: dict[int, list[tuple[int, float]]] = {}
    for row in rows:
        if row["dtype"] != dtype or int(float(row["output_stride"])) != 2:
            continue
        input_stride = int(float(row["input_stride"]))
        if input_stride not in target_strides:
            continue
        if input_stride == 1 and row["group"] != "F":
            continue
        if input_stride != 1 and row["group"] != "H":
            continue
        denominator = float(row["N_G"]) + float(row["N_GU"])
        if abs(denominator) < 1e-12:
            continue
        block_dim = int(float(row["block_dim"]))
        ratio = (float(row["actual_cycles"]) - float(row["base_cycles"])) / denominator
        points_by_stride.setdefault(input_stride, []).append((block_dim, ratio))

    if not points_by_stride:
        return

    def median(values: list[float]) -> float:
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2.0

    median_by_stride: dict[int, list[tuple[int, float]]] = {}
    for input_stride, points in points_by_stride.items():
        values_by_core: dict[int, list[float]] = {}
        for block_dim, ratio in points:
            values_by_core.setdefault(block_dim, []).append(ratio)
        median_by_stride[input_stride] = [
            (block_dim, median(values))
            for block_dim, values in sorted(values_by_core.items())
        ]

    block_dims = sorted(
        {block_dim for points in points_by_stride.values() for block_dim, _ in points}
    )
    ratios = [ratio for points in points_by_stride.values() for _, ratio in points]
    x_min, x_max = min(block_dims), max(block_dims)
    y_min, y_max = min(ratios), max(ratios)
    y_pad = max((y_max - y_min) * 0.08, 0.05)
    y_min -= y_pad
    y_max += y_pad

    width, height = 1220, 700
    left, top, right, bottom = 105, 50, 310, 92
    plot_w, plot_h = width - left - right, height - top - bottom

    def px(value: float) -> float:
        return left + (value - x_min) / max(1.0, x_max - x_min) * plot_w

    def py(value: float) -> float:
        return top + (y_max - value) / max(1.0, y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<defs><clipPath id="os2-rho-clip"><rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}"/></clipPath></defs>',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left + plot_w / 2}" y="29" text-anchor="middle" font-family="sans-serif" font-size="20">{dtype}: os=2, rho observation vs cores</text>',
        f'<text x="24" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 24 {top + plot_h / 2})" font-family="sans-serif" font-size="18" font-weight="600">(N_act-N_base)/(N_G+N_GU)</text>',
        f'<text x="{left + plot_w / 2}" y="{height - 24}" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="600">block_dim</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#555"/>',
    ]
    for block_dim in block_dims:
        x = px(block_dim)
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#f3f4f6"/>',
                f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 6}" stroke="#555"/>',
                f'<text x="{x:.2f}" y="{top + plot_h + 30}" text-anchor="middle" font-family="sans-serif" font-size="17">{block_dim}</text>',
            ]
        )
    for index in range(6):
        fraction = index / 5
        y_value = y_min + fraction * (y_max - y_min)
        y = py(y_value)
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>',
                f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#555"/>',
                f'<text x="{left - 12}" y="{y + 6:.2f}" text-anchor="end" font-family="sans-serif" font-size="17">{y_value:.5g}</text>',
            ]
        )
    if y_min < 1.0 < y_max:
        parts.append(
            f'<line x1="{left}" y1="{py(1.0):.2f}" x2="{left + plot_w}" y2="{py(1.0):.2f}" stroke="#9ca3af" stroke-dasharray="6 5"/>'
        )

    parts.append('<g clip-path="url(#os2-rho-clip)">')
    for input_stride in RHO_PLOT_INPUT_STRIDES:
        if input_stride not in points_by_stride:
            continue
        color = RHO_PLOT_COLORS[input_stride]
        median_points = " ".join(
            f"{px(block_dim):.2f},{py(ratio):.2f}"
            for block_dim, ratio in median_by_stride[input_stride]
        )
        parts.append(
            f'<polyline points="{median_points}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        )
        for block_dim, ratio in points_by_stride[input_stride]:
            parts.append(
                f'<circle cx="{px(block_dim):.2f}" cy="{py(ratio):.2f}" r="3" fill="{color}" fill-opacity="0.35"><title>input_stride={input_stride}; block_dim={block_dim}; ratio={ratio:.5g}</title></circle>'
            )
        for block_dim, ratio in median_by_stride[input_stride]:
            parts.append(
                f'<circle cx="{px(block_dim):.2f}" cy="{py(ratio):.2f}" r="4" fill="{color}"><title>input_stride={input_stride}; block_dim={block_dim}; median ratio={ratio:.5g}</title></circle>'
            )
    parts.append("</g>")

    legend_x = left + plot_w + 35
    legend_y = top + 15
    parts.append(
        f'<text x="{legend_x}" y="{legend_y}" font-family="sans-serif" font-size="16" font-weight="bold">input_stride</text>'
    )
    legend_index = 0
    for input_stride in RHO_PLOT_INPUT_STRIDES:
        if input_stride not in points_by_stride:
            continue
        color = RHO_PLOT_COLORS[input_stride]
        x = legend_x
        y = legend_y + 28 + legend_index * 27
        legend_index += 1
        parts.extend(
            [
                f'<line x1="{x}" y1="{y - 5}" x2="{x + 24}" y2="{y - 5}" stroke="{color}" stroke-width="2.5"/>',
                f'<text x="{x + 32}" y="{y}" font-family="sans-serif" font-size="14">{input_stride}</text>',
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
    for dtype in DTYPES:
        legacy_actual_vs_predicted = (
            output_dir / f"round2_1d_noncontiguous_actual_vs_predicted_{dtype}.svg"
        )
        if legacy_actual_vs_predicted.exists():
            legacy_actual_vs_predicted.unlink()
        residual_relative_svg(
            dtype,
            rows,
            output_dir / f"round2_1d_noncontiguous_residual_{dtype}.svg",
        )
        input_stride_vs_cycles_svg(
            dtype,
            rows,
            output_dir / f"round2_1d_input_stride_vs_cycles_{dtype}.svg",
        )
        output_stride_vs_cycles_svg(
            dtype,
            rows,
            output_dir / f"round2_1d_is1_output_stride_vs_cycles_{dtype}.svg",
        )
        os1_output_dim_vs_cycles_svg(
            dtype,
            rows,
            output_dir / f"round2_1d_os1_output_dim_vs_cycles_{dtype}.svg",
        )
        os2_output_dim_vs_cycles_svg(
            dtype,
            rows,
            output_dir / f"round2_1d_os2_output_dim_vs_cycles_{dtype}.svg",
        )
        os2_rho_vs_cores_svg(
            dtype,
            rows,
            output_dir / f"round2_1d_os2_rho_vs_cores_{dtype}.svg",
        )
    print(f"[INFO] wrote plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
