# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Embed the ordinary axes solver source without changing its tokens or whitespace."""

import argparse
import hashlib
from pathlib import Path

BEGIN = "// AF_AXES_RESOURCE_BEGIN\n"
END = "// AF_AXES_RESOURCE_END\n"

CALLBACK_MACROS = frozenset(
    (
        "AF_AXES_BACKUP_SIZE",
        "AF_AXES_EVAL_CONSTRAINT",
        "AF_AXES_EVAL_UPPER_BOUND",
        "AF_AXES_HAS_UPPER_BOUND",
        "AF_AXES_CAL_USED_CORE_NUM",
        "AF_AXES_CAL_REAL_USED_CORE_NUM",
        "AF_AXES_GET_PERF",
        "AF_AXES_SATISFY_UB_CACHE_LINE",
    )
)


def read_resource(path):
    text = path.read_text(encoding="utf-8")
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise ValueError(f"Expected one resource region: {path}")
    _, body = text.split(BEGIN)
    body, _ = body.split(END)
    return body


def select_source(path, equal_order, explicit_eval=False):
    body = read_resource(path)
    active, stack, output = True, [], []
    for line in body.splitlines(keepends=True):
        directive = line.strip()
        if directive in ("#if AF_AXES_EQUAL_ORDER", "#if AF_AXES_EXPLICIT_EVAL"):
            condition = (
                equal_order if directive == "#if AF_AXES_EQUAL_ORDER" else explicit_eval
            )
            stack.append((active, condition, False))
            active = active and condition
        elif directive == "#else":
            if not stack or stack[-1][2]:
                raise ValueError(f"Unexpected else: {path}")
            parent, condition, _ = stack[-1]
            stack[-1] = parent, condition, True
            active = parent and not condition
        elif directive == "#endif":
            if not stack:
                raise ValueError(f"Unexpected endif: {path}")
            active = stack.pop()[0]
        elif directive.startswith(("#define ", "#undef ")):
            name = directive.split()[1].split("(")[0]
            if name not in CALLBACK_MACROS:
                raise ValueError(f"Unsupported callback macro: {path}: {directive}")
            if active:
                output.append(line)
        elif directive.startswith("#"):
            raise ValueError(f"Unsupported directive in resource: {path}: {directive}")
        elif active:
            output.append(line)
    if stack:
        raise ValueError(f"Unclosed condition: {path}")
    return "".join(output)


def generate(source_dir, output):
    files = [source_dir / f"axes_reorder_{part}.inc" for part in ("head", "body")]
    digest = hashlib.sha256()
    lines = [
        "// Generated from the ordinary axes solver source. Do not edit.\n",
        "#ifndef ATT_AXES_SOLVER_RESOURCES_H_\n#define ATT_AXES_SOLVER_RESOURCES_H_\n",
        "namespace att {\nnamespace axes_resource {\n",
    ]
    for path in files:
        digest.update(path.name.encode("utf-8") + b"\0" + path.read_bytes())
        part = "Head" if path.name.endswith("head.inc") else "Body"
        for equal_order, mode in ((False, "Normal"), (True, "Equal")):
            value = select_source(path, equal_order)
            if ')AF_AXES"' in value:
                raise ValueError("Raw string delimiter occurs in source")
            lines.append(
                f'constexpr char k{mode}{part}[] = R"AF_AXES({value})AF_AXES";\n'
            )
    for filename, symbol in (
        ("expression_helpers.inc", "ExpressionHelpers"),
        ("scalar_helpers.inc", "ScalarHelpers"),
        ("axes_ceil_div.inc", "AxesCeilDiv"),
    ):
        path = source_dir.parent / "scalar" / filename
        digest.update(path.name.encode("utf-8") + b"\0" + path.read_bytes())
        value = read_resource(path)
        if ')AF_AXES"' in value:
            raise ValueError("Raw string delimiter occurs in source")
        lines.append(f'constexpr char k{symbol}[] = R"AF_AXES({value})AF_AXES";\n')
    lines.append(f'constexpr char kSourceSha256[] = "{digest.hexdigest()}";\n')
    lines.append("constexpr unsigned kResourceVersion = 1U;\n}\n}\n#endif\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    contents = "".join(lines)
    if not output.exists() or output.read_text(encoding="utf-8") != contents:
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(contents, encoding="utf-8")
        temporary.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    generate(args.source_dir, args.output)
