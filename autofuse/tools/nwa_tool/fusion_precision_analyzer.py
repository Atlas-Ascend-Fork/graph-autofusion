#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="定位融合算子精度劣化工具")
    parser.add_argument(
        "--mode",
        type=int,
        default=1,
        choices=[1, 2],
        help="模式1：基于dump图和datadump目录批量比较融合算子输入输出；"
        "模式2：直接比较两个NPY文件。默认模式1",
    )
    parser.add_argument(
        "--af-open-graph", help="模式1：开启自动融合的 dump 图 JSON 文件路径"
    )
    parser.add_argument(
        "--af-close-graph", help="模式1：关闭自动融合的 dump 图 JSON 文件路径"
    )
    parser.add_argument(
        "--af-open-data", help="模式1：开启自动融合的 datadump NPY 目录"
    )
    parser.add_argument(
        "--af-close-data", help="模式1：关闭自动融合的 datadump NPY 目录"
    )
    parser.add_argument(
        "--compare-input",
        action="store_true",
        help="模式1：是否比较融合算子输入，默认不比较",
    )
    parser.add_argument("--npy-a", help="模式2：第一个 NPY 文件路径")
    parser.add_argument("--npy-b", help="模式2：第二个 NPY 文件路径")
    return parser.parse_args()


def load_graph(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_ops(graph_data):
    graphs = graph_data.get("graph", [])
    if not graphs:
        print("WARNING: graph array not found in graph JSON")
        return []
    return graphs[0].get("op", [])


def build_node_lookup(ops):
    node_types = {}
    node_formats = {}
    for op in ops:
        name = op.get("name", "")
        node_types[name] = op.get("type", "")
        formats = []
        for desc in op.get("output_desc", []):
            formats.append(desc.get("layout"))
        node_formats[name] = formats
    return node_types, node_formats


def parse_origin_attrs(attrs):
    origin_name = None
    origin_output_index = None
    for attr in attrs:
        key = attr.get("key", "")
        value = attr.get("value", {})
        if key == "_datadump_origin_name":
            origin_name = value.get("s")
        elif key == "_datadump_origin_output_index":
            origin_output_index = value.get("i")
    return origin_name, origin_output_index


def extract_output_mappings(op, fused_op_name):
    mappings = []
    for idx, desc in enumerate(op.get("output_desc", [])):
        origin_name, origin_output_index = parse_origin_attrs(desc.get("attr", []))
        fused_format = desc.get("layout")
        if origin_name is None or origin_output_index is None:
            print(
                f"WARNING: Fused operator {fused_op_name} output {idx} is missing "
                "_datadump_origin_name or _datadump_origin_output_index; skipping"
            )
            mappings.append(
                {
                    "fused_op_name": fused_op_name,
                    "fused_output_index": idx,
                    "origin_op_name": None,
                    "origin_output_index": None,
                    "fused_format": None,
                    "status": "NO_MAPPING",
                }
            )
            continue
        mappings.append(
            {
                "fused_op_name": fused_op_name,
                "fused_output_index": idx,
                "origin_op_name": origin_name,
                "origin_output_index": origin_output_index,
                "fused_format": fused_format,
                "status": None,
            }
        )
    return mappings


def extract_input_mappings(op, fused_op_name):
    mappings = []
    input_refs = op.get("input", [])
    input_descs = op.get("input_desc", [])
    for idx, ref in enumerate(input_refs):
        if ":" not in ref:
            print(
                f"WARNING: Reference format cannot be parsed for fused operator {fused_op_name} input {idx}: {ref}; skipping"
            )
            mappings.append(
                {
                    "fused_op_name": fused_op_name,
                    "fused_input_index": idx,
                    "source_op_name": None,
                    "source_output_index": None,
                    "fused_format": None,
                    "status": "NO_MAPPING",
                }
            )
            continue
        parts = ref.rsplit(":", 1)
        source_op_name = parts[0]
        source_output_index = int(parts[1])
        if source_output_index == -1:
            continue
        fused_format = None
        if idx < len(input_descs):
            fused_format = input_descs[idx].get("layout")
        mappings.append(
            {
                "fused_op_name": fused_op_name,
                "fused_input_index": idx,
                "source_op_name": source_op_name,
                "source_output_index": source_output_index,
                "fused_format": fused_format,
                "status": None,
            }
        )
    return mappings


def extract_fusion_mappings(graph_data):
    output_mappings = []
    input_mappings = []
    ops = get_ops(graph_data)
    for op in ops:
        if op.get("type") not in ("AscBackend", "FusedAscBackend"):
            continue
        fused_op_name = op.get("name", "")
        output_mappings.extend(extract_output_mappings(op, fused_op_name))
        input_mappings.extend(extract_input_mappings(op, fused_op_name))
    return output_mappings, input_mappings


def build_fused_output_resolver(output_mappings):
    resolver = {}
    for m in output_mappings:
        if m.get("origin_op_name") is not None:
            resolver[(m["fused_op_name"], m["fused_output_index"])] = (
                m["origin_op_name"],
                m["origin_output_index"],
            )
    return resolver


def resolve_source(mapping, fused_output_resolver, node_types):
    source_op_name = mapping["source_op_name"]
    source_output_index = mapping["source_output_index"]
    source_type = None

    if (source_op_name, source_output_index) in fused_output_resolver:
        origin_name, origin_output_index = fused_output_resolver[
            (source_op_name, source_output_index)
        ]
        source_op_name = origin_name
        source_output_index = origin_output_index
        source_type = "Fused"
    else:
        source_type = node_types.get(source_op_name, "")

    mapping["source_op_name"] = source_op_name
    mapping["source_output_index"] = source_output_index
    mapping["source_type"] = source_type
    return mapping


def resolve_origin_format(op_name, output_index, af_close_node_formats):
    formats = af_close_node_formats.get(op_name, [])
    if output_index < len(formats):
        return formats[output_index]
    return None


def normalize_op_name(name):
    return name.replace("/", "_")


def find_npy(data_dir, op_name, kind, index):
    normalized = normalize_op_name(op_name)
    pattern = os.path.join(
        glob.escape(data_dir), f"*.{glob.escape(normalized)}.*.{kind}.{index}.npy"
    )
    matches = glob.glob(pattern)
    if not matches:
        return None
    if len(matches) > 1:
        print(f"WARNING: Multiple NPY files matched; using the first: {matches}")
    return matches[0]


def load_npy(path):
    return np.load(path)


def convert_nc1hwc0_to_nhwc(arr):
    n, c1, h, w, c0 = arr.shape
    arr = arr.reshape(n, c1, h, w, c0)
    arr = np.transpose(arr, axes=(0, 2, 3, 1, 4))
    arr = arr.reshape(n, h, w, c1 * c0)
    return arr


def convert_ndc1hwc0_to_ndhwc(arr):
    n, d, c1, h, w, c0 = arr.shape
    arr = np.transpose(arr, axes=(0, 1, 3, 4, 2, 5))
    arr = arr.reshape(n, d, h, w, c1 * c0)
    return arr


SUPPORTED_FORMAT_CONVERSIONS = {
    ("NC1HWC0", "NHWC"): convert_nc1hwc0_to_nhwc,
    ("NC1HWC0", "ND"): convert_nc1hwc0_to_nhwc,
    ("NDC1HWC0", "NDHWC"): convert_ndc1hwc0_to_ndhwc,
    ("NDC1HWC0", "ND"): convert_ndc1hwc0_to_ndhwc,
}


def compute_metrics(a, b):
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    scale_a = float(np.max(np.abs(a_flat)))
    scale_b = float(np.max(np.abs(b_flat)))
    if scale_a == 0 or scale_b == 0:
        cosine = 0.0
    else:
        scaled_a = a_flat / scale_a
        scaled_b = b_flat / scale_b
        dot = np.dot(scaled_a, scaled_b)
        norm_a = np.linalg.norm(scaled_a)
        norm_b = np.linalg.norm(scaled_b)
        cosine = float(dot / (norm_a * norm_b + 1e-8 / scale_a / scale_b))
    if a.dtype.kind in "biu" and b.dtype.kind in "biu":
        abs_diff = np.fromiter(
            (abs(int(lhs) - int(rhs)) for lhs, rhs in zip(a.flat, b.flat)),
            dtype=np.float64,
            count=a.size,
        )
    else:
        abs_diff = np.abs(np.subtract(a_flat, b_flat))
    max_abs = float(np.max(abs_diff))
    denom = np.maximum(np.abs(a_flat), np.abs(b_flat)) + 1e-8
    rel_err = abs_diff / denom
    max_rel = float(np.max(rel_err))
    return cosine, max_abs, max_rel


def apply_format_conversion(fused_data, origin_data, fused_format, origin_format):
    if fused_format == origin_format:
        return fused_data, origin_data, []
    status_parts = []
    conversion_key = (fused_format, origin_format)
    reverse_key = (origin_format, fused_format)
    if conversion_key in SUPPORTED_FORMAT_CONVERSIONS:
        fused_data = SUPPORTED_FORMAT_CONVERSIONS[conversion_key](fused_data)
        status_parts.append("FORMAT_CONVERTED")
    elif reverse_key in SUPPORTED_FORMAT_CONVERSIONS:
        origin_data = SUPPORTED_FORMAT_CONVERSIONS[reverse_key](origin_data)
        status_parts.append("FORMAT_CONVERTED")
    else:
        return None, None, None
    return fused_data, origin_data, status_parts


@dataclass
class NpySource:
    npy_path: Optional[str]
    fmt: Optional[str]
    label: str


def compare_data(fused_src: NpySource, origin_src: NpySource):
    if fused_src.npy_path is None:
        print(f"WARNING: Fused-side NPY not found: {fused_src.label}")
        return None, None, None, "FILE_NOT_FOUND"
    if origin_src.npy_path is None:
        print(f"WARNING: Original-side NPY not found: {origin_src.label}")
        return None, None, None, "FILE_NOT_FOUND"

    try:
        fused_data = load_npy(fused_src.npy_path)
        origin_data = load_npy(origin_src.npy_path)
    except Exception as e:
        print(f"WARNING: NPY Load failed - {fused_src.label} / {origin_src.label}: {e}")
        return None, None, None, "NPY_LOAD_ERROR"

    status_parts = []

    try:
        fused_data, origin_data, fmt_parts = apply_format_conversion(
            fused_data, origin_data, fused_src.fmt, origin_src.fmt
        )
    except (ValueError, TypeError) as error:
        print(
            f"WARNING: Format conversion failed - {fused_src.label} / {origin_src.label}: {error}"
        )
        return None, None, None, "FORMAT_CONVERSION_ERROR"
    if fmt_parts is None:
        print(
            f"WARNING: Unsupported format conversion - {fused_src.label}: {fused_src.fmt}"
            f" vs {origin_src.label}: {origin_src.fmt}"
        )
        return None, None, None, "FORMAT_UNSUPPORTED"
    status_parts.extend(fmt_parts)

    if fused_data.dtype != origin_data.dtype:
        promoted = np.promote_types(fused_data.dtype, origin_data.dtype)
        if not (fused_data.dtype.kind in "biu" and origin_data.dtype.kind in "biu"):
            fused_data = fused_data.astype(promoted)
            origin_data = origin_data.astype(promoted)
        status_parts.append("DTYPE_CAST")

    if fused_data.shape != origin_data.shape:
        if fused_data.size == origin_data.size:
            status_parts.append("SHAPE_FLATTENED")
        else:
            print(
                f"WARNING: Shape mismatch with different element counts - {fused_src.label}: {fused_data.shape}"
                f" vs {origin_src.label}: {origin_data.shape}"
            )
            return None, None, None, "SHAPE_MISMATCH"

    try:
        cosine, max_abs, max_rel = compute_metrics(fused_data, origin_data)
    except Exception as e:
        print(
            f"WARNING: Metric calculation failed - {fused_src.label} / {origin_src.label}: {e}"
        )
        return None, None, None, "COMPUTE_ERROR"

    return cosine, max_abs, max_rel, "_".join(status_parts) if status_parts else "OK"


def build_compare_result(
    type_str, fused_op_name, fused_index, origin_op_name, origin_index
):
    return {
        "type": type_str,
        "fused_op_name": fused_op_name,
        "fused_index": fused_index,
        "origin_op_name": origin_op_name,
        "origin_index": origin_index,
        "cosine_similarity": None,
        "max_abs_error": None,
        "max_rel_error": None,
        "status": "OK",
    }


def compare_output_pair(
    mapping, af_open_data_dir, af_close_data_dir, af_close_node_formats
):
    result = build_compare_result(
        "输出",
        mapping["fused_op_name"],
        mapping["fused_output_index"],
        mapping["origin_op_name"],
        mapping["origin_output_index"],
    )

    if mapping["status"] == "NO_MAPPING":
        result["status"] = "NO_MAPPING"
        return result

    origin_format = resolve_origin_format(
        mapping["origin_op_name"], mapping["origin_output_index"], af_close_node_formats
    )

    fused_src = NpySource(
        find_npy(
            af_open_data_dir,
            mapping["fused_op_name"],
            "output",
            mapping["fused_output_index"],
        ),
        mapping.get("fused_format"),
        f"{mapping['fused_op_name']} output {mapping['fused_output_index']}",
    )
    origin_src = NpySource(
        find_npy(
            af_close_data_dir,
            mapping["origin_op_name"],
            "output",
            mapping["origin_output_index"],
        ),
        origin_format,
        f"{mapping['origin_op_name']} output {mapping['origin_output_index']}",
    )

    cosine, max_abs, max_rel, status = compare_data(fused_src, origin_src)
    result["cosine_similarity"] = cosine
    result["max_abs_error"] = max_abs
    result["max_rel_error"] = max_rel
    result["status"] = status
    return result


def compare_input_pair(
    mapping, af_open_data_dir, af_close_data_dir, af_close_node_formats
):
    result = build_compare_result(
        "输入",
        mapping["fused_op_name"],
        mapping["fused_input_index"],
        mapping["source_op_name"],
        mapping["source_output_index"],
    )

    if mapping["status"] == "NO_MAPPING":
        result["status"] = "NO_MAPPING"
        return result

    if mapping.get("source_type") in ("Constant", "Data"):
        result["status"] = "SKIPPED_CONST_DATA"
        return result

    source_format = resolve_origin_format(
        mapping["source_op_name"], mapping["source_output_index"], af_close_node_formats
    )

    fused_src = NpySource(
        find_npy(
            af_open_data_dir,
            mapping["fused_op_name"],
            "input",
            mapping["fused_input_index"],
        ),
        mapping.get("fused_format"),
        f"{mapping['fused_op_name']} input {mapping['fused_input_index']}",
    )
    origin_src = NpySource(
        find_npy(
            af_close_data_dir,
            mapping["source_op_name"],
            "output",
            mapping["source_output_index"],
        ),
        source_format,
        f"{mapping['source_op_name']} output {mapping['source_output_index']}",
    )

    cosine, max_abs, max_rel, status = compare_data(fused_src, origin_src)
    result["cosine_similarity"] = cosine
    result["max_abs_error"] = max_abs
    result["max_rel_error"] = max_rel
    result["status"] = status
    return result


def format_table(results):
    headers = [
        "类型",
        "融合算子名",
        "索引",
        "原算子名",
        "原索引",
        "余弦相似度",
        "绝对误差最大值",
        "相对误差最大值",
        "状态",
    ]
    rows = []
    for r in results:
        cs = (
            f"{r['cosine_similarity']:.10f}"
            if r["cosine_similarity"] is not None
            else "-"
        )
        mae = f"{r['max_abs_error']:.6e}" if r["max_abs_error"] is not None else "-"
        mre = f"{r['max_rel_error']:.6e}" if r["max_rel_error"] is not None else "-"
        rows.append(
            [
                r["type"],
                r["fused_op_name"],
                str(r["fused_index"]),
                r["origin_op_name"] if r["origin_op_name"] else "-",
                str(r["origin_index"]) if r["origin_index"] is not None else "-",
                cs,
                mae,
                mre,
                r["status"],
            ]
        )

    col_widths = [
        max(len(str(h)), max((len(str(row[i])) for row in rows), default=0))
        for i, h in enumerate(headers)
    ]

    def format_row(row):
        return " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))

    lines = [format_row(headers)]
    lines.append("-+-".join("-" * w for w in col_widths))
    for row in rows:
        lines.append(format_row(row))
    return "\n".join(lines)


def validate_mode1_args(args):
    required = [
        ("--af-open-graph", args.af_open_graph, True),
        ("--af-close-graph", args.af_close_graph, True),
        ("--af-open-data", args.af_open_data, False),
        ("--af-close-data", args.af_close_data, False),
    ]
    for name, value, is_file in required:
        if not value:
            print(f"ERROR: 模式1需要 {name} 参数")
            sys.exit(1)
        if is_file and not os.path.isfile(value):
            print(f"ERROR: dump 图 JSON 文件不存在: {value}")
            sys.exit(1)
        if not is_file and not os.path.isdir(value):
            print(f"ERROR: datadump 目录不存在: {value}")
            sys.exit(1)


def run_mode1(args):
    validate_mode1_args(args)

    print(f"Parsing fusion-enabled dump graph: {args.af_open_graph}")
    graph_data = load_graph(args.af_open_graph)
    ops = get_ops(graph_data)
    node_types, _ = build_node_lookup(ops)
    output_mappings, input_mappings = extract_fusion_mappings(graph_data)
    fused_output_resolver = build_fused_output_resolver(output_mappings)

    print(f"Parsing fusion-disabled dump graph: {args.af_close_graph}")
    af_close_graph_data = load_graph(args.af_close_graph)
    af_close_ops = get_ops(af_close_graph_data)
    _, af_close_node_formats = build_node_lookup(af_close_ops)

    compare_input = args.compare_input

    if compare_input:
        for m in input_mappings:
            if m["status"] is not None:
                continue
            resolve_source(m, fused_output_resolver, node_types)
        print(
            f"Found {len(output_mappings)} fused operator output mappings, "
            f"{len(input_mappings)} fused operator input mappings\n"
        )
    else:
        print(f"Found {len(output_mappings)} fused operator output mappings\n")

    results = []
    for mapping in output_mappings:
        results.append(
            compare_output_pair(
                mapping, args.af_open_data, args.af_close_data, af_close_node_formats
            )
        )
    if compare_input:
        for mapping in input_mappings:
            results.append(
                compare_input_pair(
                    mapping,
                    args.af_open_data,
                    args.af_close_data,
                    af_close_node_formats,
                )
            )

    print(format_table(results))


def run_mode2(args):
    if not args.npy_a:
        print("ERROR: 模式2需要 --npy-a 参数")
        sys.exit(1)
    if not args.npy_b:
        print("ERROR: 模式2需要 --npy-b 参数")
        sys.exit(1)
    if not os.path.isfile(args.npy_a):
        print(f"ERROR: NPY 文件不存在: {args.npy_a}")
        sys.exit(1)
    if not os.path.isfile(args.npy_b):
        print(f"ERROR: NPY 文件不存在: {args.npy_b}")
        sys.exit(1)

    src_a = NpySource(args.npy_a, None, args.npy_a)
    src_b = NpySource(args.npy_b, None, args.npy_b)
    cosine, max_abs, max_rel, status = compare_data(src_a, src_b)
    print(f"文件A: {args.npy_a}")
    print(f"文件B: {args.npy_b}")
    print(f"状态: {status}")
    if cosine is not None:
        print(f"余弦相似度: {cosine:.10f}")
        print(f"绝对误差最大值: {max_abs:.6e}")
        print(f"相对误差最大值: {max_rel:.6e}")


def main():
    args = parse_args()
    if args.mode == 1:
        run_mode1(args)
    else:
        run_mode2(args)


if __name__ == "__main__":
    main()
