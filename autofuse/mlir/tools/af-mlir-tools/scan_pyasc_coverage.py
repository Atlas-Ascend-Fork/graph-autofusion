#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import difflib
import io
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYASC = ROOT / "api" / "ascendc"


def create_stream_logger(name: str, stream) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.terminator = ""
        logger.addHandler(handler)
    return logger


OUTPUT_LOGGER = create_stream_logger(f"{__name__}.stdout", sys.stdout)
ERROR_LOGGER = create_stream_logger(f"{__name__}.stderr", sys.stderr)

FAMILIES = [
    {
        "name": "elewise",
        "patterns": [
            "include/ascir/Dialect/Asc/IR/Basic/OpVecUnary.td",
            "include/ascir/Dialect/Asc/IR/Basic/OpVecUnaryExt.td",
            "include/ascir/Dialect/Asc/IR/Autofuse/OpVecUnary.td",
            "include/ascir/Dialect/Asc/IR/Autofuse/OpVecBinary.td",
            "include/ascir/Dialect/Asc/IR/Autofuse/OpVecCmpsel.td",
            "include/ascir/Dialect/Asc/IR/Basic/OpVecBinary.td",
            "include/ascir/Dialect/Asc/IR/Basic/OpVecBinaryScalar.td",
            "include/ascir/Dialect/Asc/IR/Basic/OpVecTernaryScalar.td",
            "include/ascir/Dialect/Asc/IR/Basic/OpVecCmpsel.td",
        ],
    },
    {
        "name": "brc",
        "patterns": [
            "include/ascir/Dialect/Asc/IR/Basic/OpVecBrcb.td",
            "include/ascir/Dialect/Asc/IR/Autofuse/OpVecBroadcast.td",
        ],
    },
    {
        "name": "reduce",
        "patterns": [
            "include/ascir/Dialect/Asc/IR/Basic/OpVecReduce.td",
            "include/ascir/Dialect/Asc/IR/Autofuse/OpVecReduceND.td",
        ],
    },
    {
        "name": "gather",
        "patterns": [
            "include/ascir/Dialect/Asc/IR/Basic/OpVecGather.td",
            "include/ascir/Dialect/Asc/IR/Basic/OpVecGatherMask.td",
        ],
    },
    {
        "name": "transpose",
        "patterns": [
            "include/ascir/Dialect/Asc/IR/Basic/OpVecTranspose.td",
            "include/ascir/Dialect/Asc/IR/Basic/OpDataConversion.td",
        ],
    },
    {
        "name": "datacopy",
        "patterns": [
            "include/ascir/Dialect/Asc/IR/Basic/OpDataCopy.td",
            "include/ascir/Dialect/Asc/IR/Autofuse/OpDataCopy.td",
            "lib/Target/AscendC/Basic/DataCopy.cpp",
            "lib/Target/AscendC/Autofuse/DataCopy.cpp",
        ],
    },
    {
        "name": "cube",
        "patterns": [
            "include/ascir/Dialect/Asc/IR/Adv/Matmul.td",
            "include/ascir/Dialect/Asc/IR/Basic/OpConv2d.td",
            "include/ascir/Dialect/Asc/IR/Basic/OpFixpipe.td",
            "include/ascir/Dialect/Asc/IR/Basic/OpGemm.td",
            "include/ascir/Dialect/Asc/IR/Basic/OpMm.td",
        ],
    },
]

CSV_COLUMNS = [
    "dialect",
    "op_name",
    "td_symbol",
    "template",
    "source_file",
    "family",
    "status",
    "arguments",
    "argument_type_hints",
    "results",
    "result_type_hints",
    "param_type_lists",
]


@dataclass
class Entity:
    kind: str
    name: str
    base: str
    body: str
    source_file: str
    params: list[str] = field(default_factory=list)


@dataclass
class Schema:
    arguments: list[dict[str, Any]] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    param_type_lists: list[int] = field(default_factory=list)


@dataclass
class OpRecord:
    dialect: str
    op_name: str
    td_symbol: str
    template: str
    source_file: str
    family: str
    status: str
    arguments: list[dict[str, Any]] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    param_type_lists: list[int] = field(default_factory=list)


@dataclass
class RecordSpec:
    dialect: str
    symbol: str
    base_name: str
    call_args: list[str]
    source_file: str
    status: str
    schema: Schema


@dataclass
class ExpansionContext:
    source_file: str
    classes: dict[str, Entity]
    multiclasses: dict[str, Entity]
    class_cache: dict[str, Schema]


@dataclass
class CollectionContext:
    classes: dict[str, Entity]
    multiclasses: dict[str, Entity]
    class_cache: dict[str, Schema]
    expansion_contexts: dict[str, ExpansionContext] = field(default_factory=dict)


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//.*", "", text)


def skip_string(text: str, pos: int) -> int:
    quote = text[pos]
    pos += 1
    while pos < len(text):
        if text[pos] == "\\":
            pos += 2
            continue
        if text[pos] == quote:
            return pos + 1
        pos += 1
    return pos


def skip_balanced(text: str, pos: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    while pos < len(text):
        ch = text[pos]
        if ch in ("'", '"'):
            pos = skip_string(text, pos)
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return pos + 1
        pos += 1
    return pos


def update_delimiter_depths(
    ch: str, depths: tuple[int, int, int]
) -> tuple[int, int, int]:
    angle_depth, paren_depth, bracket_depth = depths
    if ch == "<":
        angle_depth += 1
    elif ch == ">" and angle_depth:
        angle_depth -= 1
    elif ch == "(":
        paren_depth += 1
    elif ch == ")" and paren_depth:
        paren_depth -= 1
    elif ch == "[":
        bracket_depth += 1
    elif ch == "]" and bracket_depth:
        bracket_depth -= 1
    return angle_depth, paren_depth, bracket_depth


def is_top_level(depths: tuple[int, int, int]) -> bool:
    return not any(depths)


def scan_until_top_level(text: str, pos: int, stop_chars: set[str]) -> int:
    depths = (0, 0, 0)
    while pos < len(text):
        ch = text[pos]
        if ch in ("'", '"'):
            pos = skip_string(text, pos)
            continue
        depths = update_delimiter_depths(ch, depths)
        if is_top_level(depths) and ch in stop_chars:
            return pos
        pos += 1
    return pos


def split_top_level(value: str, sep: str = ",") -> list[str]:
    parts = []
    start = 0
    pos = 0
    depths = (0, 0, 0)
    while pos < len(value):
        ch = value[pos]
        if ch in ("'", '"'):
            pos = skip_string(value, pos)
            continue
        depths = update_delimiter_depths(ch, depths)
        if ch == sep and is_top_level(depths):
            parts.append(value[start:pos].strip())
            start = pos + 1
        pos += 1
    tail = value[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def parse_params(params_text: str) -> list[str]:
    params = []
    for part in split_top_level(params_text):
        before_default = part.split("=", 1)[0].strip()
        match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", before_default)
        if match:
            params.append(match.group(1))
    return params


def skip_whitespace(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def parse_entity_params(text: str, pos: int, kind: str) -> tuple[list[str], int]:
    if kind not in {"class", "multiclass"} or pos >= len(text) or text[pos] != "<":
        return [], pos
    params_end = skip_balanced(text, pos, "<", ">")
    params_start = pos + 1
    params_content_end = params_end - 1
    return parse_params(text[params_start:params_content_end]), skip_whitespace(
        text, params_end
    )


def parse_entity_base(text: str, pos: int) -> tuple[str, int]:
    if pos < len(text) and text[pos] == ":":
        base_start = pos + 1
        pos = scan_until_top_level(text, base_start, {"{", ";"})
        return text[base_start:pos].strip(), pos
    return "", scan_until_top_level(text, pos, {"{", ";"})


def parse_entity_body(text: str, pos: int) -> tuple[str, int]:
    if pos < len(text) and text[pos] == "{":
        body_start = pos + 1
        body_end = skip_balanced(text, pos, "{", "}")
        pos = body_end + int(body_end < len(text) and text[body_end] == ";")
        body_content_end = body_end - 1
        return text[body_start:body_content_end], pos
    if pos < len(text) and text[pos] == ";":
        pos += 1
    return "", pos


def parse_entities(text: str, source_file: str) -> list[Entity]:
    entities = []
    pos = 0
    keyword = re.compile(
        r"\b(class|multiclass|defm|def)\s+(\"[^\"]*\"|[A-Za-z_][A-Za-z0-9_]*)"
    )
    while True:
        match = keyword.search(text, pos)
        if not match:
            break
        kind = match.group(1)
        name_token = match.group(2)
        name = name_token[1:-1] if name_token.startswith('"') else name_token
        pos = skip_whitespace(text, match.end())
        params, pos = parse_entity_params(text, pos, kind)
        base, pos = parse_entity_base(text, pos)
        body, pos = parse_entity_body(text, pos)

        entities.append(
            Entity(
                kind=kind,
                name=name,
                base=base,
                body=body,
                source_file=source_file,
                params=params,
            )
        )
    return entities


def parse_call(expr: str) -> tuple[str, list[str]]:
    match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)", expr)
    if not match:
        return "", []
    name = match.group(1)
    pos = match.end()
    while pos < len(expr) and expr[pos].isspace():
        pos += 1
    if pos >= len(expr) or expr[pos] != "<":
        return name, []
    end = skip_balanced(expr, pos, "<", ">")
    args_start = pos + 1
    args_end = end - 1
    return name, split_top_level(expr[args_start:args_end])


def eval_td_expr(expr: str, env: dict[str, str]) -> str:
    parts = split_top_level(expr, "#")
    result = []
    for part in parts:
        value = part.strip()
        if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
            result.append(value[1:-1])
        elif value in env:
            result.append(env[value])
        else:
            result.append(value)
    return "".join(result)


def extract_let_value(body: str, key: str) -> str:
    match = re.search(r"\blet\s+" + re.escape(key) + r"\s*=", body)
    if not match:
        return ""
    pos = match.end()
    while pos < len(body) and body[pos].isspace():
        pos += 1
    if pos < len(body) and body[pos] == "(":
        end = skip_balanced(body, pos, "(", ")")
        value_start = pos + 1
        value_end = end - 1
        return body[value_start:value_end].strip()
    if pos < len(body) and body[pos] == "[":
        end = skip_balanced(body, pos, "[", "]")
        value_start = pos + 1
        value_end = end - 1
        return body[value_start:value_end].strip()
    end = scan_until_top_level(body, pos, {";"})
    return body[pos:end].strip()


def parse_value_list(raw: str) -> list[dict[str, Any]]:
    raw = re.sub(r"^(ins|outs)\b", "", raw).strip()

    values = []
    for item in split_top_level(raw):
        if not item:
            continue
        name = ""
        constraint = item
        match = re.search(r":\$([A-Za-z_][A-Za-z0-9_]*)\s*$", item)
        if match:
            name = match.group(1)
            constraint = item[: match.start()].strip()
        type_hints = sorted(set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", constraint)))
        values.append(
            {
                "name": name,
                "constraint": constraint.strip(),
                "type_hints": type_hints,
                "raw": re.sub(r"\s+", " ", item).strip(),
            }
        )
    return values


def parse_param_type_lists(raw: str) -> list[int]:
    if not raw:
        return []
    values = []
    for item in split_top_level(raw):
        try:
            values.append(int(item.strip()))
        except ValueError:
            continue
    return values


def schema_from_body(body: str) -> Schema:
    return Schema(
        arguments=parse_value_list(extract_let_value(body, "arguments")),
        results=parse_value_list(extract_let_value(body, "results")),
        param_type_lists=parse_param_type_lists(
            extract_let_value(body, "paramTypeLists")
        ),
    )


def dialect_for_source(source_file: str) -> str:
    if "Dialect/EmitAsc/IR/" in source_file:
        return "emitasc"
    if "Dialect/Asc/IR/" in source_file:
        return "ascendc"
    return "unknown"


def family_for_source(source_file: str) -> str:
    for family in FAMILIES:
        if source_file in family["patterns"]:
            return family["name"]
    return "other"


def is_op_class(
    name: str, classes: dict[str, Entity], seen: set[str] | None = None
) -> bool:
    if name in {"AscendC_Op", "EmitAsc_Op"} or name.endswith("Op"):
        return True
    if seen is None:
        seen = set()
    if name in seen or name not in classes:
        return False
    seen.add(name)
    base_name, _ = parse_call(classes[name].base)
    return is_op_class(base_name, classes, seen) if base_name else False


def resolve_class_schema(
    name: str,
    classes: dict[str, Entity],
    cache: dict[str, Schema],
    seen: set[str] | None = None,
) -> Schema:
    if name in cache:
        return cache[name]
    if seen is None:
        seen = set()
    if name in seen or name not in classes:
        return Schema()
    seen.add(name)

    entity = classes[name]
    own = schema_from_body(entity.body)
    base_name, _ = parse_call(entity.base)
    inherited = (
        resolve_class_schema(base_name, classes, cache, seen) if base_name else Schema()
    )
    schema = Schema(
        arguments=own.arguments or inherited.arguments,
        results=own.results or inherited.results,
        param_type_lists=own.param_type_lists or inherited.param_type_lists,
    )
    cache[name] = schema
    return schema


def make_op_record(spec: RecordSpec) -> OpRecord | None:
    mnemonic = spec.call_args[0] if spec.call_args else ""
    if not mnemonic:
        return None
    return OpRecord(
        dialect=spec.dialect,
        op_name=f"{spec.dialect}.{mnemonic}",
        td_symbol=spec.symbol,
        template=spec.base_name,
        source_file=spec.source_file,
        family=family_for_source(spec.source_file),
        status=spec.status,
        arguments=spec.schema.arguments,
        results=spec.schema.results,
        param_type_lists=spec.schema.param_type_lists,
    )


def make_expanded_record(
    child: Entity,
    symbol: str,
    base_name: str,
    call_args: list[str],
    context: ExpansionContext,
) -> OpRecord | None:
    if child.kind != "def" or not is_op_class(base_name, context.classes):
        return None
    schema = schema_from_body(child.body)
    if not schema.arguments and not schema.results:
        schema = resolve_class_schema(base_name, context.classes, context.class_cache)
    return make_op_record(
        RecordSpec(
            dialect=dialect_for_source(context.source_file),
            symbol=symbol,
            base_name=base_name,
            call_args=call_args,
            source_file=context.source_file,
            status="expanded-defm",
            schema=schema,
        )
    )


def expand_multiclass(
    multiclass: Entity,
    actual_args: list[str],
    prefix: str,
    context: ExpansionContext,
    depth: int = 0,
) -> list[OpRecord]:
    if depth > 8:
        return []
    env = {
        name: actual_args[index]
        for index, name in enumerate(multiclass.params)
        if index < len(actual_args)
    }
    child_entities = parse_entities(multiclass.body, context.source_file)
    records = []
    for child in child_entities:
        base_name, base_args = parse_call(child.base)
        evaluated_args = [eval_td_expr(arg, env) for arg in base_args]
        child_name = eval_td_expr(child.name, env)
        child_prefix = prefix + child_name
        if child.kind == "defm":
            nested = context.multiclasses.get(base_name)
            if nested:
                records.extend(
                    expand_multiclass(
                        nested,
                        evaluated_args,
                        child_prefix,
                        context,
                        depth + 1,
                    )
                )
            continue
        record = make_expanded_record(
            child, child_prefix, base_name, evaluated_args, context
        )
        if record:
            records.append(record)
    return records


def collect_td_files(pyasc_root: Path) -> list[Path]:
    roots = [
        pyasc_root / "include" / "ascir" / "Dialect" / "Asc" / "IR",
        pyasc_root / "include" / "ascir" / "Dialect" / "EmitAsc" / "IR",
    ]
    files = []
    for root in roots:
        if root.exists():
            files.extend(sorted(root.rglob("*.td")))
    return files


def load_entities(pyasc_root: Path) -> list[Entity]:
    entities: list[Entity] = []
    for td_file in collect_td_files(pyasc_root):
        text = strip_comments(td_file.read_text(encoding="utf-8"))
        rel = td_file.relative_to(pyasc_root).as_posix()
        entities.extend(parse_entities(text, rel))
    return entities


def records_for_defm(
    entity: Entity,
    dialect: str,
    base_name: str,
    base_args: list[str],
    context: CollectionContext,
) -> list[OpRecord]:
    multiclass = context.multiclasses.get(base_name)
    evaluated_args = [eval_td_expr(arg, {}) for arg in base_args]
    if multiclass:
        expansion_context = context.expansion_contexts.setdefault(
            entity.source_file,
            ExpansionContext(
                entity.source_file,
                context.classes,
                context.multiclasses,
                context.class_cache,
            ),
        )
        return expand_multiclass(
            multiclass, evaluated_args, entity.name, expansion_context
        )

    record = make_op_record(
        RecordSpec(
            dialect=dialect,
            symbol=entity.name,
            base_name=base_name,
            call_args=evaluated_args,
            source_file=entity.source_file,
            status="template-instance",
            schema=Schema(),
        )
    )
    return [record] if record else []


def records_for_entity(entity: Entity, context: CollectionContext) -> list[OpRecord]:
    dialect = dialect_for_source(entity.source_file)
    if dialect == "unknown":
        return []

    base_name, base_args = parse_call(entity.base)
    if entity.kind == "defm":
        return records_for_defm(entity, dialect, base_name, base_args, context)
    if entity.kind != "def" or not is_op_class(base_name, context.classes):
        return []

    evaluated_args = [eval_td_expr(arg, {}) for arg in base_args]
    schema = schema_from_body(entity.body)
    status = "explicit-def"
    if not schema.arguments and not schema.results:
        schema = resolve_class_schema(base_name, context.classes, context.class_cache)
        status = (
            "inherited-class" if schema.arguments or schema.results else "no-signature"
        )
    record = make_op_record(
        RecordSpec(
            dialect=dialect,
            symbol=entity.name,
            base_name=base_name,
            call_args=evaluated_args,
            source_file=entity.source_file,
            status=status,
            schema=schema,
        )
    )
    return [record] if record else []


def deduplicate_ops(ops: list[OpRecord]) -> list[OpRecord]:
    seen = set()
    unique_ops = []
    for op in sorted(
        ops, key=lambda item: (item.op_name, item.td_symbol, item.source_file)
    ):
        key = (op.op_name, op.td_symbol, op.source_file)
        if key in seen:
            continue
        seen.add(key)
        unique_ops.append(op)
    return unique_ops


def summarize_families(
    pyasc_root: Path, unique_ops: list[OpRecord]
) -> list[dict[str, Any]]:
    family_summary = []
    for family in FAMILIES:
        evidence = [
            pattern for pattern in family["patterns"] if (pyasc_root / pattern).exists()
        ]
        family_ops = [op for op in unique_ops if op.family == family["name"]]
        status = "detected" if family_ops or evidence else "not-detected"
        family_summary.append(
            {
                "family": family["name"],
                "op_count": len(family_ops),
                "evidence": evidence,
                "status": status,
            }
        )
    other_count = sum(1 for op in unique_ops if op.family == "other")
    if other_count:
        family_summary.append(
            {
                "family": "other",
                "op_count": other_count,
                "evidence": ["unmapped Asc/EmitAsc TableGen op definitions"],
                "status": "detected",
            }
        )
    return family_summary


def collect_ops(pyasc_root: Path) -> tuple[list[OpRecord], list[dict[str, Any]]]:
    entities = load_entities(pyasc_root)
    classes = {entity.name: entity for entity in entities if entity.kind == "class"}
    multiclasses = {
        entity.name: entity for entity in entities if entity.kind == "multiclass"
    }
    context = CollectionContext(classes, multiclasses, {})
    ops = [
        record for entity in entities for record in records_for_entity(entity, context)
    ]
    unique_ops = deduplicate_ops(ops)
    return unique_ops, summarize_families(pyasc_root, unique_ops)


def format_io(values: list[dict[str, Any]]) -> str:
    if not values:
        return "-"
    return "<br>".join(
        f"{value['name'] or '?'}:{value['constraint']}" for value in values
    )


def write_line(stream: Any, text: str = "") -> None:
    stream.write(f"{text}\n")


def emit_markdown(
    pyasc_root: Path,
    ops: list[OpRecord],
    family_summary: list[dict[str, Any]],
    stream: Any,
) -> None:
    write_line(stream, "# PyAsc AscendC/EmitAsc op extraction")
    write_line(stream)
    write_line(stream, f"- PyAsc root: `{pyasc_root}`")
    write_line(stream, f"- Extracted ops: {len(ops)}")
    write_line(
        stream,
        "- Source: TableGen `.td` definitions with lightweight PyAsc `defm` expansion",
    )
    write_line(stream)
    write_line(stream, "## Op Family Summary")
    write_line(stream)
    write_line(stream, "| Family / bucket | Extracted ops | PyAsc evidence | Status |")
    write_line(stream, "|---|---:|---|---|")
    for family in family_summary:
        evidence_text = "<br>".join(family["evidence"]) if family["evidence"] else "-"
        write_line(
            stream,
            f"| {family['family']} | {family['op_count']} | {evidence_text} | {family['status']} |",
        )

    write_line(stream)
    write_line(stream, "## Op Inventory")
    write_line(stream)
    write_line(stream, "| Op | TD symbol | Source | Arguments | Results | Status |")
    write_line(stream, "|---|---|---|---|---|---|")
    for op in ops:
        write_line(
            stream,
            "| "
            f"{op.op_name} | {op.td_symbol} | {op.source_file} | "
            f"{format_io(op.arguments)} | {format_io(op.results)} | {op.status} |",
        )


def format_csv_values(values: list[dict[str, Any]]) -> str:
    return "; ".join(
        f"{value['name'] or '?'}:{value['constraint']}" for value in values
    )


def format_csv_hints(values: list[dict[str, Any]]) -> str:
    return "; ".join(
        f"{value['name'] or '?'}:{'/'.join(value['type_hints'])}" for value in values
    )


def emit_csv(ops: list[OpRecord], stream: Any) -> None:
    writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for op in ops:
        writer.writerow(
            {
                "dialect": op.dialect,
                "op_name": op.op_name,
                "td_symbol": op.td_symbol,
                "template": op.template,
                "source_file": op.source_file,
                "family": op.family,
                "status": op.status,
                "arguments": format_csv_values(op.arguments),
                "argument_type_hints": format_csv_hints(op.arguments),
                "results": format_csv_values(op.results),
                "result_type_hints": format_csv_hints(op.results),
                "param_type_lists": ";".join(
                    str(value) for value in op.param_type_lists
                ),
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract PyAsc AscendC/EmitAsc ops from TableGen sources."
    )
    parser.add_argument(
        "--pyasc-root",
        type=Path,
        default=DEFAULT_PYASC,
        help="Path to the PyAsc checkout.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json", "csv"],
        default="markdown",
        help="Output format.",
    )
    parser.add_argument(
        "--output", type=Path, help="Write output to this file instead of stdout."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare the generated output against --output instead of writing it; "
        "exit non-zero when the snapshot is stale.",
    )
    args = parser.parse_args()
    if args.check and args.output is None:
        parser.error("--check requires --output")
    return args


def write_output(content: str, output: Path | None) -> None:
    if output is None:
        OUTPUT_LOGGER.info("%s", content)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")


def check_output(content: str, output: Path, argv_format: str) -> int:
    """Compare freshly generated content against the committed snapshot."""
    regenerate = (
        f"python3 autofuse/mlir/tools/af-mlir-tools/scan_pyasc_coverage.py "
        f"--format {argv_format} --output {output}"
    )
    if not output.exists():
        ERROR_LOGGER.error(
            "ERROR: snapshot missing: %s\n  regenerate: %s\n", output, regenerate
        )
        return 1
    current = output.read_text(encoding="utf-8")
    if current == content:
        return 0
    diff = difflib.unified_diff(
        current.splitlines(keepends=True),
        content.splitlines(keepends=True),
        fromfile=f"{output} (committed)",
        tofile=f"{output} (regenerated)",
        n=0,
    )
    ERROR_LOGGER.error(
        "ERROR: %s is out of sync with the PyAsc TableGen sources.\n"
        "  regenerate: %s\n%s\n",
        output,
        regenerate,
        "".join(diff),
    )
    return 1


def main() -> int:
    args = parse_args()
    pyasc_root = args.pyasc_root.resolve()
    if not pyasc_root.exists():
        ERROR_LOGGER.error("ERROR: PyAsc checkout missing: %s\n", pyasc_root)
        return 1

    ops, family_summary = collect_ops(pyasc_root)

    def emit(content: str) -> int:
        if args.check:
            return check_output(content, args.output, args.format)
        write_output(content, args.output)
        return 0

    if args.format == "json":
        return emit(
            json.dumps(
                {
                    "pyasc_root": str(pyasc_root),
                    "source": "td",
                    "ops": [asdict(op) for op in ops],
                    "families": family_summary,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    if args.format == "csv":
        stream = io.StringIO()
        emit_csv(ops, stream)
        return emit(stream.getvalue())

    stream = io.StringIO()
    emit_markdown(pyasc_root, ops, family_summary, stream)
    return emit(stream.getvalue())


if __name__ == "__main__":
    raise SystemExit(main())
