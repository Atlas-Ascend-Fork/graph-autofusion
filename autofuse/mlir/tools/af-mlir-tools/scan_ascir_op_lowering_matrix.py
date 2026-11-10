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
"""AscIR op 的两平台 lowering 对照表。

从 legacy 的 `REG_ASC_IR` 注册与 `*_ascir_codegen_impl.h` 抽出每个 op 在
mem-based（`NpuArch=2201`）与 reg-based（`NpuArch=3510`）两套实现上的四项事实：

  1. AscendC API 名（`GetApiName`）
  2. 依赖的 Autofuse 扩展头（`LoadApiHeaderFiles`）
  3. 是否需要 tmp buffer（是否 override `CalcTmpBufSize`）
  4. 标量输入约束（`IsScalarInputSupported` / `...IfExchangeInputs`）

据此把每个 op 判成 D0 / D1 / D2 / D3 —— 判据见 `--help` 与输出表头。这张表是
AFIR Op 补齐 elementwise 切片（docs/design/2026-09-17-*）E0-1 的产物，决定
后续每个波次的内容与每个 op 需要几个 implementation。

脚本不做完整 C++ 语义解析：它假定 legacy 的 impl 类是「单继承 + 方法体里直接
return」的规整形态，这在今天成立。任何解析不到的项都显式标成 `?`，不静默给默认值。
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

V1_REG = ROOT / "ascir" / "generator" / "ascir_builtin_ops_v1.cpp"
V1_IMPL = ROOT / "ascir" / "generator" / "v1_ascir_codegen_impl.h"
V2_REG = ROOT / "v35" / "ascir" / "generator" / "ascir_builtin_ops_v2.cpp"
V2_IMPL = ROOT / "v35" / "ascir" / "generator" / "v2_ascir_codegen_impl.h"
MEM_API_DIR = ROOT / "ascendc" / "api"
REG_API_DIR = ROOT / "v35" / "ascendc" / "api_regbase"
# `AscendCApiRegistry` 是一个进程级单例，下面三处各自把一批「头名 -> 头文本」注册进去。
# 三批的 key 命名空间不相交（mem 是 `xxx.h`，reg-base 是 `xxx_reg_base.h`），
# 所以头名本身就足以定位内容，查表不需要知道平台。
API_REGISTER_SOURCES = (
    ROOT / "codegen" / "ascendc_api_registry.cpp",
    ROOT / "v35" / "codegen" / "ascendc_reg_base_api_register.cpp",
    ROOT / "v35" / "codegen" / "ascendc_cube_api_register.cpp",
)

MEM = "mem-based"
REG = "reg-based"

# 族划分，与 docs/design/2026-09-17-afir-op-completion-elementwise-slice-design.md 一致。
# 只用于给表加一列便于筛选；不在任何族里的 op 落 "other"。
FAMILIES: dict[str, set[str]] = {
    "elementwise": {
        # 一元
        "Abs",
        "Cast",
        "Erf",
        "Exp",
        "Gelu",
        "IsFinite",
        "IsInf",
        "Isnan",
        "Ln",
        "LogicalNot",
        "Neg",
        "Reciprocal",
        "Relu",
        "Rsqrt",
        "Sigmoid",
        "Sign",
        "Sqrt",
        "Tanh",
        # 二元
        "Add",
        "Axpy",
        "BitwiseAnd",
        "Div",
        "Eq",
        "FloorDiv",
        "Ge",
        "Gt",
        "Le",
        "LeakyRelu",
        "Lt",
        "LogicalAnd",
        "LogicalOr",
        "Maximum",
        "Minimum",
        "Mul",
        "Ne",
        "Pow",
        "Remainder",
        "Sub",
        "TrueDiv",
        # 三元
        "ClipByValue",
        "MaskedFill",
        "Select",
        "Where",
        # 广播与标量源
        "Broadcast",
        "Scalar",
        "ScalarData",
        "IndexExpr",
    },
    "reduce": {
        "All",
        "Any",
        "Max",
        "Mean",
        "Min",
        "Prod",
        "Sum",
        "ArgMax",
        "ArgMaxMultiRPhase1",
        "ArgMaxMultiRPhase2",
        "FlashSoftmax",
        "Softmax",
    },
    "concat": {"Concat"},
    "split": {"Split"},
    "gather": {"Gather", "IndirectLoad"},
    "cube": {
        "MatMul",
        "MatMulBias",
        "MatMulOffset",
        "MatMulOffsetBias",
        "BatchMatMul",
        "BatchMatMulBias",
        "BatchMatMulOffset",
        "BatchMatMulOffsetBias",
        "Conv2D",
        "Conv2DBias",
        "Conv2DOffset",
        "Conv2DOffsetBias",
        "ExtendConv2D",
        "ExtendConv2DBias",
        "ExtendConv2DBiasScale",
        "ExtendConv2DScale",
    },
    "layout": {
        "Transpose",
        "Pad",
        "RemovePad",
        "Ub2ub",
        "Nddma",
        "Arange",
        "Rand",
        "Randn",
    },
    "datamove": {"Load", "Store"},
    "structural": {"Data", "Output", "Workspace", "Nop", "Unsupported", "VectorFunc"},
}

CLASS_RE = re.compile(r"\bclass\s+(\w+)\s*(?::\s*public\s+(?:af::ascir::)?(\w+)\s*)?\{")
REG_RE = re.compile(r"REG_ASC_IR\((\w+)\)")
CODEGEN_IMPL_RE = re.compile(r"AscIrImplCreator<\s*(?:af::ascir::)?(\w*Codegen\w*)\s*>")
HEADER_STR_RE = re.compile(r'"([A-Za-z0-9_./]*\.h)"')
STRING_LITERAL_RE = re.compile(r'"([^"]*)"')
RETURN_BRACED_RE = re.compile(r"return\s*(?:std::vector<std::string>\s*)?\{")
REGISTRY_KEY_RE = re.compile(r'\{\s*"([A-Za-z0-9_./]+\.h)"\s*,')
RETURN_STR_RE = re.compile(r'return\s+"([^"]*)"\s*;')
RETURN_ANY_RE = re.compile(r"return\s+([^;]+);")
COMMENT_RE = re.compile(r"//[^\n]*")

# `.Input("x1", "T1")`：把形参名与它用的符号类型绑起来。输入的 dtype 集合要读哪个符号，
# 由这里决定——不是所有 op 的输入都共用一个符号（`select` 的掩码是 `T1`、两个数据输入
# 是 `T2`）。
INPUT_DECL_RE = re.compile(r'\.(?:Dynamic)?Input\(\s*"([^"]*)"\s*,\s*"([^"]*)"\s*\)')
# `.Inputs({...})` 是另一种写法：不给每个输入单独的符号名。空列表就是起始节点
# （`Scalar` / `Data` / `Arange`），非空的靠 `.UseFirstInputDataType()` 共用一个符号。
INPUTS_LIST_RE = re.compile(r"\.Inputs\(\s*\{([^}]*)\}\s*\)")
# `.Impl(..., {creator, creator, {{"T", TensorType{DT_...}}, {"T2", MakeT2List(kXxx)}}})`
# 里的一条 `{"符号", 取值}`。取值有两种写法，第二种要再查一张 pair 数组。
TYPE_MAP_ENTRY_RE = re.compile(
    r'\{\s*"(\w+)"\s*,\s*'
    r"(?:TensorType\s*\{([^}]*)\}"
    r"|MakeT([12])List\(\s*(\w+)\s*\)"
    r"|(TensorType::ALL\(\)))\s*\}"
)
# `TensorType::ALL()` 不展开成一长串：它说的是「不限」，逐个列出来反而看不出这一点。
ALL_DTYPES = "all"
PAIR_ARRAY_RE = re.compile(
    r"constexpr\s+std::pair<\s*ge::DataType\s*,\s*ge::DataType\s*>\s+(\w+)\s*\[\s*\]\s*=\s*\{"
)
DTYPE_RE = re.compile(r"\bDT_([A-Z0-9_]+)\b")

# `ge::DataType` 到本表里的拼写。AFIR 建模得出来的六种用与 `ElementTypes` 位一致的写法
# （f16 / f32 / i16 / i32 / ui8 / ui16），其余按同样规则短写——列在这里不代表 AFIR 收，
# 这一列记的是 legacy 收什么。
DTYPE_SPELLING = {
    "FLOAT16": "f16",
    "FLOAT": "f32",
    "INT16": "i16",
    "INT32": "i32",
    "UINT8": "ui8",
    "UINT16": "ui16",
    "BF16": "bf16",
    "INT4": "i4",
    "INT8": "i8",
    "INT64": "i64",
    "UINT32": "ui32",
    "UINT64": "ui64",
    "BOOL": "bool",
    "DOUBLE": "f64",
    "COMPLEX64": "c64",
    "COMPLEX128": "c128",
    "QINT8": "qi8",
    "QINT16": "qi16",
    "QINT32": "qi32",
    "QUINT8": "qui8",
    "QUINT16": "qui16",
}
# 输出顺序固定，跟声明顺序无关：声明顺序在两个文件里本来就不一致，跟着它走会让
# 重新生成产生无意义的 diff。
DTYPE_ORDER = tuple(DTYPE_SPELLING.values())


def spell_dtype(raw: str) -> str:
    return DTYPE_SPELLING.get(raw, raw.lower())


def sort_dtypes(names: set[str]) -> tuple[str, ...]:
    known = [name for name in DTYPE_ORDER if name in names]
    return tuple(known + sorted(names - set(known)))


METHODS = (
    "GetApiName",
    "GetApiCallName",
    "LoadApiHeaderFiles",
    "CalcTmpBufSize",
    "IsScalarInputSupported",
    "IsScalarInputSupportedIfExchangeInputs",
)


def strip_comments(text: str) -> str:
    return COMMENT_RE.sub("", text)


def match_braces(text: str, open_index: int) -> int:
    """返回与 text[open_index] 这个 '{' 配对的 '}' 的下标；找不到返回 -1。

    legacy 的 impl 头里没有裸的大括号字符串字面量，注释已经先被剥掉，所以这里
    只数括号就够。
    """
    depth = 0
    for index in range(open_index, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return -1


@dataclass
class ClassInfo:
    name: str
    base: str | None
    methods: dict[str, str] = field(default_factory=dict)


def parse_impl_classes(path: Path) -> dict[str, ClassInfo]:
    text = strip_comments(path.read_text(encoding="utf-8"))
    classes: dict[str, ClassInfo] = {}
    for match in CLASS_RE.finditer(text):
        open_index = match.end() - 1
        close_index = match_braces(text, open_index)
        if close_index < 0:
            continue
        body = text[open_index + 1 : close_index]
        info = ClassInfo(name=match.group(1), base=match.group(2))
        for method in METHODS:
            method_match = re.search(re.escape(method) + r"\s*\([^)]*\)[^{;]*\{", body)
            if method_match is None:
                continue
            method_open = method_match.end() - 1
            method_close = match_braces(body, method_open)
            if method_close < 0:
                continue
            info.methods[method] = body[method_open + 1 : method_close]
        classes[info.name] = info
    return classes


def split_top_level(text: str) -> list[str]:
    """按逗号切分，忽略括号内的逗号。"""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in text:
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    parts.append("".join(current))
    return parts


def extract_headers(body: str | None) -> tuple[str, ...]:
    """从 `LoadApiHeaderFiles` 的方法体里取头名。

    元素可以是字面量（`"abs.h"`），也可以是拼接（`std::string("cast") +
    std::string("_reg_base.h")`）——后者要把相邻字面量接起来才是真正的 key。
    多个 return 分支（按 `is_dynamic` 分流）取并集。
    """
    if body is None:
        return ()
    names: set[str] = set()
    for match in RETURN_BRACED_RE.finditer(body):
        close = match_braces(body, match.end() - 1)
        if close < 0:
            continue
        for element in split_top_level(body[match.end() : close]):
            joined = "".join(STRING_LITERAL_RE.findall(element))
            if joined.endswith(".h"):
                names.add(joined)
    return tuple(sorted(names))


def parse_registry_keys() -> set[str]:
    keys: set[str] = set()
    for path in API_REGISTER_SOURCES:
        if not path.exists():
            continue
        keys.update(
            REGISTRY_KEY_RE.findall(strip_comments(path.read_text(encoding="utf-8")))
        )
    return keys


def lookup_method(
    classes: dict[str, ClassInfo], class_name: str, method: str
) -> str | None:
    """沿单继承链找方法体；基类 `AscIrCodegen` / `AscIrCodegenV2` 不在表里，返回 None
    表示「用基类默认值」。"""
    seen: set[str] = set()
    current: str | None = class_name
    while current and current in classes and current not in seen:
        seen.add(current)
        info = classes[current]
        if method in info.methods:
            return info.methods[method]
        current = info.base
    return None


def parse_registrations(path: Path) -> dict[str, str | None]:
    text = strip_comments(path.read_text(encoding="utf-8"))
    ops: dict[str, str | None] = {}
    matches = list(REG_RE.finditer(text))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.start() : end]
        impl = CODEGEN_IMPL_RE.search(block)
        ops[match.group(1)] = impl.group(1) if impl else None
    return ops


def parse_pair_arrays(text: str) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    """`constexpr std::pair<DataType, DataType> kXxx[] = {...}` → (T1 那半, T2 那半)。

    `MakeT1List` / `MakeT2List` 取的就是每对的 `.first` / `.second`，所以这里按对拆开。
    """
    arrays: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for match in PAIR_ARRAY_RE.finditer(text):
        close = match_braces(text, match.end() - 1)
        if close < 0:
            continue
        body = text[match.end() : close]
        firsts: set[str] = set()
        seconds: set[str] = set()
        for pair in re.finditer(r"\{\s*DT_(\w+)\s*,\s*DT_(\w+)\s*\}", body):
            firsts.add(spell_dtype(pair.group(1)))
            seconds.add(spell_dtype(pair.group(2)))
        arrays[match.group(1)] = (sort_dtypes(firsts), sort_dtypes(seconds))
    return arrays


def parse_type_map(
    block: str, pair_arrays: dict[str, tuple[tuple[str, ...], tuple[str, ...]]]
) -> dict[str, tuple[str, ...]]:
    """一个 `REG_ASC_IR` 块里 `符号 → dtype 列表`。解析不出来的符号不进表。"""
    type_map: dict[str, tuple[str, ...]] = {}
    for match in TYPE_MAP_ENTRY_RE.finditer(block):
        symbol, literal, half, array_name, everything = match.groups()
        if everything is not None:
            type_map[symbol] = (ALL_DTYPES,)
            continue
        if literal is not None:
            names = {spell_dtype(name) for name in DTYPE_RE.findall(literal)}
            type_map[symbol] = sort_dtypes(names)
            continue
        pair = pair_arrays.get(array_name)
        if pair is not None:
            type_map[symbol] = pair[0] if half == "1" else pair[1]
    return type_map


# 起始节点（`.Inputs({})`）在 dtype 列上的取值：它没有输入，不是没解析出来。
NO_INPUT = "n/a"


def data_input_symbol(block: str, type_map: dict[str, tuple[str, ...]]) -> str | None:
    """这个 op 的**数据**输入用哪个符号；起始节点返回 `NO_INPUT`，判不出来返回 None。

    一元、二元的全部输入共用一个符号，取它即可。三元的选择族不是：`select` 的
    `x1` 是 ui8 掩码（`T1`），`x2` / `x3` 才是数据（`T2`）。所以按「被最多输入用到」
    取，并列时取靠后的——掩码永远只占一个输入。
    """
    inputs = INPUT_DECL_RE.findall(block)
    if inputs:
        counts: dict[str, int] = {}
        for _, symbol in inputs:
            counts[symbol] = counts.get(symbol, 0) + 1
        best = max(counts.values())
        for _, symbol in reversed(inputs):
            if counts[symbol] == best:
                return symbol
        return None
    listed = INPUTS_LIST_RE.search(block)
    if listed is None:
        return None
    if not listed.group(1).strip():
        return NO_INPUT
    # `.Inputs({"x1", "x2", "x3"})` 不给符号名。只有一个符号时它就是那个；多于一个
    # 就判不出来，记 `?` 让人去看。
    return next(iter(type_map)) if len(type_map) == 1 else None


def parse_input_dtypes(path: Path) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    """每个 op 的**数据输入** dtype 集合，以及它用的符号名。

    符号名一并返回，是因为 v2 的注册块大多只写 `.Impl(...)`、不重复写 `.Input(...)`,
    它的输入声明要借 v1 的。借的时候要借的是符号名，不是 dtype 列表——两侧的 dtype
    列表本来就不一样，这一列存在的理由就是它们不一样。
    """
    text = strip_comments(path.read_text(encoding="utf-8"))
    pair_arrays = parse_pair_arrays(text)
    dtypes: dict[str, tuple[str, ...]] = {}
    symbols: dict[str, str] = {}
    matches = list(REG_RE.finditer(text))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.start() : end]
        op = match.group(1)
        type_map = parse_type_map(block, pair_arrays)
        symbol = data_input_symbol(block, type_map)
        if symbol is not None:
            symbols[op] = symbol
            if symbol in type_map:
                dtypes[op] = type_map[symbol]
    return dtypes, symbols


def parse_type_maps(path: Path) -> dict[str, dict[str, tuple[str, ...]]]:
    """每个 op 的完整 `符号 → dtype 列表`，供跨文件借符号名时取用。"""
    text = strip_comments(path.read_text(encoding="utf-8"))
    pair_arrays = parse_pair_arrays(text)
    maps: dict[str, dict[str, tuple[str, ...]]] = {}
    matches = list(REG_RE.finditer(text))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.start() : end]
        maps[match.group(1)] = parse_type_map(block, pair_arrays)
    return maps


def summarize_scalar(body: str | None, exchange_body: str | None) -> str:
    def one(text: str | None) -> str:
        if text is None:
            return "no"
        expression = RETURN_ANY_RE.search(text)
        if expression is None:
            return "?"
        value = " ".join(expression.group(1).split())
        if value == "true":
            return "any"
        if value == "false":
            return "no"
        if "OnlySecondInputSupportScalar" in value:
            return "2nd"
        if "IsAllScalarOrUbScalar" in value:
            return "all"
        return value[:40]

    direct = one(body)
    exchange = one(exchange_body)
    return direct if exchange == "no" else f"{direct}+swap({exchange})"


@dataclass
class SideFacts:
    present: bool = False
    api_name: str = ""
    api_call: str = ""
    headers: tuple[str, ...] = ()
    needs_tmp: bool = False
    tmp_func: str = ""
    scalar: str = "no"
    # 这一侧的 `REG_ASC_IR` 收哪几种输入 dtype。三种「没有值」是三件事，不能混：
    # 空元组＝这个 op 在这一侧没注册；`("n/a",)`＝起始节点，本来就没有输入；
    # `("?",)`＝有输入但解析不出来，要人去看。
    dtypes: tuple[str, ...] = ()

    @property
    def shape_key(self) -> tuple:
        """决定「调用形态」是否相同的三元组：API 名、扩展头集合、是否要 tmp。"""
        return (self.api_name, tuple(sorted(self.headers)), self.needs_tmp)


def collect_side(
    reg_path: Path, impl_path: Path, input_symbols: dict[str, str] | None = None
) -> dict[str, SideFacts]:
    registrations = parse_registrations(reg_path)
    classes = parse_impl_classes(impl_path)
    type_maps = parse_type_maps(reg_path)
    own_symbols = parse_input_dtypes(reg_path)[1]
    facts: dict[str, SideFacts] = {}
    for op, class_name in registrations.items():
        side = SideFacts(present=True)
        # 输入符号名优先用本文件自己声明的；没声明（v2 的常见形态）就借另一侧的。
        # 借的只是「哪个符号是数据输入」，dtype 列表仍然读本文件的。
        symbol = own_symbols.get(op) or (input_symbols or {}).get(op, "")
        if symbol == NO_INPUT:
            side.dtypes = (NO_INPUT,)
        elif symbol:
            side.dtypes = type_maps.get(op, {}).get(symbol, ("?",))
        else:
            side.dtypes = ("?",)
        if class_name is None:
            facts[op] = side
            continue
        api_name = lookup_method(classes, class_name, "GetApiName")
        api_call = lookup_method(classes, class_name, "GetApiCallName")
        headers = lookup_method(classes, class_name, "LoadApiHeaderFiles")
        tmp = lookup_method(classes, class_name, "CalcTmpBufSize")
        scalar = lookup_method(classes, class_name, "IsScalarInputSupported")
        exchange = lookup_method(
            classes, class_name, "IsScalarInputSupportedIfExchangeInputs"
        )

        def first_string(body: str | None) -> str:
            if body is None:
                return ""
            found = RETURN_STR_RE.search(body)
            return found.group(1) if found else "?"

        side.api_name = first_string(api_name)
        side.api_call = first_string(api_call)
        side.headers = extract_headers(headers)
        side.needs_tmp = tmp is not None
        if tmp is not None:
            call = re.search(r"^\s*return\s+(\w+)\s*\(\s*node\s*\)\s*;\s*$", tmp)
            # 不是「一句话委托给 Calc* 辅助函数」的，就是方法体自己按条件算。
            side.tmp_func = call.group(1) if call else "custom"
        side.scalar = summarize_scalar(scalar, exchange)
        facts[op] = side
    return facts


def classify(mem: SideFacts, reg: SideFacts) -> tuple[str, str]:
    """返回 (档位, 一句话理由)。"""
    if mem.present and not reg.present:
        return "D3", "仅 mem-based"
    if reg.present and not mem.present:
        return "D3", "仅 reg-based"
    if mem.shape_key != reg.shape_key:
        reasons = []
        if mem.api_name != reg.api_name:
            reasons.append(f"API 名 `{mem.api_name or '-'}` vs `{reg.api_name or '-'}`")
        if tuple(sorted(mem.headers)) != tuple(sorted(reg.headers)):
            reasons.append(
                f"扩展头 {'/'.join(mem.headers) or '-'} vs {'/'.join(reg.headers) or '-'}"
            )
        if mem.needs_tmp != reg.needs_tmp:
            reasons.append(
                f"tmp {'要' if mem.needs_tmp else '不要'} vs {'要' if reg.needs_tmp else '不要'}"
            )
        return "D2", "；".join(reasons)
    if mem.headers:
        return "D1", f"两侧共用扩展头 {'/'.join(mem.headers)}"
    return "D0", "两侧一致且不依赖扩展头"


def family_of(op: str) -> str:
    for name, members in FAMILIES.items():
        if op in members:
            return name
    return "other"


@dataclass
class Row:
    op: str
    family: str
    tier: str
    reason: str
    mem: SideFacts
    reg: SideFacts


def build_rows() -> list[Row]:
    # `.Input(...)` 基本只写在 v1 那份注册里，v2 大多只写 `.Impl(...)`。所以符号名从
    # 两份里合起来取，两份都没有的 op 的 dtype 列记 `?`。
    mem_symbols = parse_input_dtypes(V1_REG)[1]
    reg_symbols = parse_input_dtypes(V2_REG)[1]
    shared_symbols = {op: sym for op, sym in mem_symbols.items() if sym}
    for op, sym in reg_symbols.items():
        if sym:
            shared_symbols.setdefault(op, sym)
    mem_facts = collect_side(V1_REG, V1_IMPL, shared_symbols)
    reg_facts = collect_side(V2_REG, V2_IMPL, shared_symbols)
    rows: list[Row] = []
    for op in sorted(set(mem_facts) | set(reg_facts)):
        mem = mem_facts.get(op, SideFacts())
        reg = reg_facts.get(op, SideFacts())
        tier, reason = classify(mem, reg)
        rows.append(
            Row(op=op, family=family_of(op), tier=tier, reason=reason, mem=mem, reg=reg)
        )
    return rows


def unregistered_headers(rows: list[Row]) -> list[tuple[str, str, str]]:
    """impl 引用了、但 `AscendCApiRegistry` 里没有的头名。

    legacy 查不到时是 `if (file.empty()) continue;` —— 静默跳过，不报错。所以这类
    引用只能靠扫描发现。返回 (op, 平台, 头名)。
    """
    keys = parse_registry_keys()
    missing: list[tuple[str, str, str]] = []
    for row in rows:
        for label, side in ((MEM, row.mem), (REG, row.reg)):
            for name in side.headers:
                if name not in keys:
                    missing.append((row.op, label, name))
    return missing


def cell(side: SideFacts) -> tuple[str, str, str, str, str]:
    if not side.present:
        return ("—", "—", "—", "—", "—")
    return (
        side.api_name or "—",
        "/".join(side.headers) or "—",
        (side.tmp_func or "yes") if side.needs_tmp else "—",
        side.scalar,
        "/".join(side.dtypes) or "—",
    )


def render_markdown(rows: list[Row], family_filter: str | None) -> str:
    out = io.StringIO()
    selected = [
        row for row in rows if family_filter is None or row.family == family_filter
    ]

    out.write("# AscIR op 两平台 lowering 对照表\n\n")
    out.write(
        "> 能力参考 · 由 `tools/af-mlir-tools/scan_ascir_op_lowering_matrix.py` 生成，请勿手工编辑。\n\n"
    )
    out.write("## 判据\n\n")
    out.write("按「API 名 / 扩展头 / 是否需 tmp」三项在两套实现上是否一致分档：\n\n")
    out.write(
        "`dtype` 两列是该侧 `REG_ASC_IR` 收的**输入**元素类型（选择族取的是数据输入那一个，"
        "不是 ui8 掩码）。`n/a` 是起始节点、本来就没有输入；`?` 是有输入但解析不出来，"
        "要人去看——两者不是一回事。\n\n"
    )
    out.write("| 档 | 含义 | AFIR 侧的代价 |\n|---|---|---|\n")
    out.write(
        "| **D0** | 三项一致，且不依赖 Autofuse 扩展头 | 一个 implementation，无平台相关代码 |\n"
    )
    out.write(
        "| **D1** | 三项一致，且两侧引用**同一个**扩展头 key | 一个 implementation + 一条 include |\n"
    )
    out.write(
        "| **D2** | 三项至少一项不同（含扩展头 key 不同） | 两个 implementation + 两个 stable id，Capability 查 feature |\n"
    )
    out.write("| **D3** | op 只在一侧注册 | ODS 取并集，实现只在有供给的一侧注册 |\n\n")
    out.write("mem-based = `NpuArch 2201`（v1，soc `2201`）；")
    out.write("reg-based = `NpuArch 3510`（v2，soc `3510` / `5102` / `9202`）。\n\n")
    out.write("**扩展头的 key 空间是全局且不相交的**：mem 侧是 `xxx.h`，reg-base 侧是 ")
    out.write(
        "`xxx_reg_base.h`，cube 另有一批；三者注册进同一个 `AscendCApiRegistry` 单例。"
    )
    out.write("因此「头名 → 头文本」的查表**不需要知道平台**，头名本身就是判别式。\n\n")
    out.write(
        "`tmp` 列给出 `CalcTmpBufSize` 委托的辅助函数名；`custom` 表示方法体自己按条件算"
    )
    out.write(
        "（即该 op 的 tmp 需求是条件性的）；`—` 表示不 override，即不需要 tmp。\n"
    )
    out.write(
        "`标量` 列：`no` 不支持标量输入，`2nd` 仅第二个输入可为标量，`any` 任意，"
    )
    out.write("`all` 全标量，`+swap(...)` 表示交换操作数后另有一套约束。\n\n")

    out.write("## 统计\n\n")
    out.write("| 范围 | D0 | D1 | D2 | D3 | 合计 |\n|---|---|---|---|---|---|\n")
    families = ["elementwise"] + sorted({row.family for row in rows} - {"elementwise"})
    for name in ["(全部)"] + families:
        scope = (
            rows if name == "(全部)" else [row for row in rows if row.family == name]
        )
        if not scope:
            continue
        counts = {
            tier: sum(1 for row in scope if row.tier == tier)
            for tier in ("D0", "D1", "D2", "D3")
        }
        out.write(
            f"| {name} | {counts['D0']} | {counts['D1']} | {counts['D2']} | {counts['D3']} | {len(scope)} |\n"
        )
    out.write("\n")

    title = "全部 op" if family_filter is None else f"{family_filter} 族"
    out.write(f"## 逐 op 明细（{title}）\n\n")
    out.write("| op | 族 | 档 | mem API | mem 扩展头 | mem tmp | mem 标量 | mem dtype ")
    out.write("| reg API | reg 扩展头 | reg tmp | reg 标量 | reg dtype | 判据 |\n")
    out.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for row in selected:
        mem = cell(row.mem)
        reg = cell(row.reg)
        out.write(
            f"| `{row.op}` | {row.family} | **{row.tier}** "
            f"| {mem[0]} | {mem[1]} | {mem[2]} | {mem[3]} | {mem[4]} "
            f"| {reg[0]} | {reg[1]} | {reg[2]} | {reg[3]} | {reg[4]} | {row.reason} |\n"
        )
    out.write("\n")

    missing = unregistered_headers(rows)
    out.write("## 引用了未注册头名的 op\n\n")
    if missing:
        out.write(
            "legacy 查不到头名时是 `if (file.empty()) continue;` —— **静默跳过，不报错**。"
        )
        out.write(
            "下列引用因此只能靠扫描发现，迁移时要逐条确认是笔误还是有意为之。\n\n"
        )
        out.write("| op | 平台 | 头名 |\n|---|---|---|\n")
        for op, label, name in missing:
            out.write(f"| `{op}` | {label} | `{name}` |\n")
    else:
        out.write(
            "无。所有 impl 引用的头名都能在 `AscendCApiRegistry` 的三批注册里找到。\n"
        )
    out.write("\n")

    anomalies = [
        row
        for row in rows
        if "::" in row.mem.api_name
        or "::" in row.reg.api_name
        or (row.mem.present and row.mem.api_name == "" and row.mem.api_call != "")
        or (row.reg.present and row.reg.api_name == "" and row.reg.api_call != "")
    ]
    out.write("## API 名书写异常\n\n")
    if anomalies:
        out.write(
            "`GetApiName()` 的约定是**裸名**（`Add` / `AbsExtend`），由 emitter 决定"
        )
        out.write(
            "要不要加命名空间前缀。下列 op 破了这个约定，迁移时 emitter 不能无条件拼 "
        )
        out.write("`AscendC::`。\n\n")
        out.write("| op | mem API | reg API |\n|---|---|---|\n")
        for row in anomalies:
            out.write(
                f"| `{row.op}` | `{row.mem.api_name or '-'}` | `{row.reg.api_name or '-'}` |\n"
            )
    else:
        out.write("无。\n")
    out.write("\n")

    out.write("## 生成命令\n\n```bash\n")
    out.write(
        "python3 autofuse/mlir/tools/af-mlir-tools/scan_ascir_op_lowering_matrix.py \\\n"
    )
    out.write(
        "    --format markdown --output autofuse/mlir/docs/guides/ascir_op_lowering_matrix.md\n"
    )
    out.write("```\n")
    return out.getvalue()


def render_csv(rows: list[Row], family_filter: str | None) -> str:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(
        [
            "op",
            "family",
            "tier",
            "reason",
            "mem_present",
            "mem_api_name",
            "mem_api_call",
            "mem_headers",
            "mem_tmp_func",
            "mem_scalar",
            "mem_dtypes",
            "reg_present",
            "reg_api_name",
            "reg_api_call",
            "reg_headers",
            "reg_tmp_func",
            "reg_scalar",
            "reg_dtypes",
        ]
    )
    for row in rows:
        if family_filter is not None and row.family != family_filter:
            continue
        writer.writerow(
            [
                row.op,
                row.family,
                row.tier,
                row.reason,
                int(row.mem.present),
                row.mem.api_name,
                row.mem.api_call,
                ";".join(row.mem.headers),
                row.mem.tmp_func,
                row.mem.scalar,
                ";".join(row.mem.dtypes),
                int(row.reg.present),
                row.reg.api_name,
                row.reg.api_call,
                ";".join(row.reg.headers),
                row.reg.tmp_func,
                row.reg.scalar,
                ";".join(row.reg.dtypes),
            ]
        )
    return out.getvalue()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--format", choices=("markdown", "csv"), default="markdown")
    parser.add_argument("--family", help="只输出某个族，如 elementwise")
    parser.add_argument("--output", type=Path, help="写入文件；缺省写 stdout")
    parser.add_argument(
        "--check",
        action="store_true",
        help="不写文件，只比对 --output 指向的那份是不是当前 legacy 的产物；"
        "对不上退 1。给 pre-commit 用。",
    )
    args = parser.parse_args(argv)
    if args.check and args.output is None:
        parser.error("--check 要和 --output 一起用")

    for path in (V1_REG, V1_IMPL, V2_REG, V2_IMPL, MEM_API_DIR, REG_API_DIR):
        if not path.exists():
            print(f"missing input: {path}", file=sys.stderr)
            return 1

    rows = build_rows()
    text = (
        render_markdown(rows, args.family)
        if args.format == "markdown"
        else render_csv(rows, args.family)
    )
    if args.check:
        # `newline=""`：csv 那份的行尾是 `\r\n`（`csv.writer` 的缺省），按文本模式读回来
        # 会被归一成 `\n`，于是一份没动过的表也会被判成漂移。
        current = ""
        if args.output.exists():
            with args.output.open(encoding="utf-8", newline="") as handle:
                current = handle.read()
        if current == text:
            return 0
        # ElementTypeMatrixTest 用此表与 legacy 注册信息对比。
        # 检查模式必须报告快照漂移，否则测试可能只是在比对过期数据。
        print(
            f"{args.output} 与当前 legacy 的 REG_ASC_IR 对不上。重新生成：\n"
            f"  python3 {Path(__file__).name} --format {args.format} --output {args.output}",
            file=sys.stderr,
        )
        return 1
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
