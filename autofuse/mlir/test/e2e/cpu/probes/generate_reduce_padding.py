# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Compile the production padding lowering into the CPU numerical probe header."""

import argparse
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--af-opt", required=True)
parser.add_argument("--translate", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--target", required=True)
args = parser.parse_args()
functions = []
dispatch = []
for dtype, cpp in [("f32", "float"), ("f16", "half"), ("i32", "int32_t")]:
    for kind in range(7):
        for tail in [False, True]:
            name = f"padding_{dtype}_{kind}_{int(tail)}"
            tensor = f"!ascendc.local_tensor<?x{dtype}>"
            functions.append(f"""
func.func @{name}(%dst: {tensor}, %a: index, %r: index, %current: index, %inner: index, %pitch: index) {{
  ascendc.reduce_init %dst, %a, %r, %current, %inner, %pitch {{reduceType = {kind} : i32, isTailLast = {str(tail).lower()}}}
    : {tensor}, index, index, index, index, index
  return
}}
""")
            dispatch.append(f"""  if constexpr (std::is_same_v<T, {cpp}> && Kind == {kind} && Tail == {str(tail).lower()}) {{
    {name}(dst, a, r, current, inner, pitch);
  }}""")
source = "module {" + "".join(functions) + "}"
lowered = subprocess.run(
    [args.af_opt, "--convert-afir-to-ascendc", f"--target={args.target}"],
    input=source,
    text=True,
    check=True,
    stdout=subprocess.PIPE,
).stdout
assert "ascendc.reduce_init" not in lowered
code = subprocess.run(
    [args.translate, "-mlir-to-ascendc"],
    input=lowered,
    text=True,
    check=True,
    stdout=subprocess.PIPE,
).stdout
assert "ReduceInit<" not in code
# This dispatch is test-only: all padding arithmetic and instructions above come
# from the production conversion pass, not a second implementation in the probe.
wrapper = (
    """
template <typename T, int Kind, bool Tail>
void RunPadding(const AscendC::LocalTensor<T> &dst, uint64_t a, uint64_t r,
                uint64_t current, uint64_t inner, uint64_t pitch = 0) {
  if (pitch == 0) {
    constexpr uint64_t alignment = 32 / sizeof(T);
    pitch = (inner + alignment - 1) / alignment * alignment;
  }
"""
    + "\n".join(dispatch)
    + "\n}\n"
)
Path(args.output).write_text("#pragma once\n#include <type_traits>\n" + code + wrapper)
