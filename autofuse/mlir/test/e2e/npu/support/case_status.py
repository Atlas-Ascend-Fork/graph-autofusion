#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# ----------------------------------------------------------------------------------------------------------------------

"""上板用例的统一成败词汇，以及 `run.py` 与各用例脚本之间的机读上报格式。

为什么要单独一个模块：统一入口要把十几个脚本的结果放在一张表里对账，而这些脚本此前各有
一套词汇——`matrix_report` 的五级、`reduce_template_matrix` 的 PASS/FAIL、`reduce_fuzz_matrix`
的 PASS/SKIP/NOKERNEL/FAIL、单用例的「抛异常即失败」。只要词汇不是同一份，跨脚本对账得到的
数字就没有意义。所以词汇表、严重度次序和上报行格式都放在这里，上报端与解析端共用，
**不允许任何一侧另起一套**。

## 严重度次序（`SEVERITY`）

次序是「清单声明的预期」与「本轮实测」能够比较的前提，它不是美观排序，每一跳都有判据：

- `PASS` —— 数值通过，未捕获已知拒收诊断。
- `PASS_FALLBACK` —— 数值通过，但捕获了已知拒收诊断。它不能证明 MLIR 全链路通过。
  当前 codegen 在 MLIR 能力不足时返回 `kNotSupported`，由调用方转 legacy；不能据这个
  等级推断成“只淘汰部分候选，剩余 MLIR 候选成功”。实际执行路径需结合 codegen 和产物核对。
- `REFUSED` —— 执行失败，捕获了已知拒收诊断；数值断言失败始终归 MISMATCH。
- `NOKERNEL` —— codegen abort，连 kernel 都没生成（`reduce_fuzz_matrix` 实测到的一族）。
  排在 REFUSED 之后：REFUSED 至少是 MLIR 主动按判据拒收，NOKERNEL 是走到生成期才崩。
- `MISMATCH` —— 数值对不上。真缺陷。
- `ERROR` —— 其它异常。真缺陷。

`SKIP` 不在这条阶梯上：它表示该格不适用（形状生不出这种形态等），既不是好也不是坏，
参与比较只会制造假信号，所以单列。

## 为什么红线是「比预期更差」而不是「不是 PASS」

今天的规则是「只有 MISMATCH / ERROR 让退出码非零」——REFUSED 永远绿。这条规则防住了
「每次跑都是红的，红就不再是信号」，代价是两类漂移完全测不出来：**悄悄变差**（PASS 退化成
PASS_FALLBACK）和**悄悄变好**（支持边界修好了，但没人更新清单，于是清单继续声称这里不支持）。
清单声明预期等级之后，判据变成「实测比预期差才红」，这两类漂移就都被记账了。

预期是**选填**的：给了就断言，没给就沿用今天的规则（只有 MISMATCH / ERROR 红）。这和 CPU 侧
`AddCpuCase.cmake` 里 `COVERS` 的结论是同一条——必填只会逼着新用例填一个没推导过的值，
那比不填更糟，因为它把「声明与实际不符还能通过」这件事重新放了进来。预期的来源只能是
实测，`run.py --calibrate` 就是用来从一轮真实的跑批里把它生成出来的。
"""

from typing import Dict, List, Optional, Sequence, Tuple

PASS = "PASS"
PASS_FALLBACK = "PASS_FALLBACK"
REFUSED = "REFUSED"
NOKERNEL = "NOKERNEL"
MISMATCH = "MISMATCH"
ERROR = "ERROR"
SKIP = "SKIP"

# 严重度：数字越大越差。SKIP 不在阶梯上，见模块注释。
SEVERITY: Dict[str, int] = {
    PASS: 0,
    PASS_FALLBACK: 1,
    REFUSED: 2,
    NOKERNEL: 3,
    MISMATCH: 4,
    ERROR: 5,
}

# 没有预期时的红线，与本模块之前各脚本的既有行为一致。
DEFECT_STATUSES = (MISMATCH, ERROR)

ALL_STATUSES: Tuple[str, ...] = (
    PASS,
    PASS_FALLBACK,
    REFUSED,
    NOKERNEL,
    MISMATCH,
    ERROR,
    SKIP,
)

# 机读上报行。制表符分隔而不是 key=value：shapes 和 reason 里都带空格、括号和逗号，
# key=value 得为它们定一套引用规则，而制表符在这些值里不可能出现。
#
# 人读的那几行（`ELEWISE_RESULT ...`、`MATRIX_SUMMARY_ROW ...`）一行都没改：历史日志、
# 评审记录和既有的 grep 都还指着它们。机读行是额外打的，两者并存。
CASE_MARKER = "AF_NPU_CASE"
SUITE_MARKER = "AF_NPU_SUITE"

_FIELD_COUNT = 7


def _sanitize(text: object) -> str:
    """制表符和换行会破坏行格式，换成空格；其余原样保留。"""
    return str(text).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def emit_case(
    suite: str,
    cell: str,
    status: str,
    tag: str = "",
    shapes: object = "",
    reason: str = "",
    group: str = "",
) -> str:
    """拼一行机读的单格结果。返回拼好的行，由调用方决定往哪打。

    `group` 带上来是为了让统一入口能按分组小计（算子族、dtype、形态），而不是只能按脚本。
    各表本来就有分组这个维度（`--group chain`、`--family`、每个 op 一组），此前只体现在
    各自的人读汇总行里，跨脚本就对不上了。
    """
    if status not in ALL_STATUSES:
        raise ValueError("unknown case status: {}".format(status))
    fields = [
        CASE_MARKER,
        _sanitize(suite),
        _sanitize(cell),
        _sanitize(group),
        status,
        _sanitize(tag),
        _sanitize(shapes),
        _sanitize(reason),
    ]
    return "\t".join(fields)


def emit_suite(suite: str, tally: Dict[str, int]) -> str:
    """拼一行机读的脚本级汇总。"""
    counts = " ".join(
        "{}={}".format(status, tally[status])
        for status in ALL_STATUSES
        if tally.get(status)
    )
    total = sum(tally.get(status, 0) for status in ALL_STATUSES)
    return "\t".join([SUITE_MARKER, _sanitize(suite), "total={}".format(total), counts])


class CaseResult(object):
    """解析出来的一格结果。"""

    __slots__ = ("suite", "cell", "group", "status", "tag", "shapes", "reason")

    def __init__(self, suite, cell, group, status, tag, shapes, reason):
        self.suite = suite
        self.cell = cell
        self.group = group
        self.status = status
        self.tag = tag
        self.shapes = shapes
        self.reason = reason

    @property
    def key(self) -> str:
        """清单里引用一格的名字。"""
        return "{}/{}".format(self.suite, self.cell) if self.cell else self.suite

    @property
    def label(self) -> str:
        return "{}({})".format(self.status, self.tag) if self.tag else self.status

    def __repr__(self) -> str:
        return "CaseResult({}, {})".format(self.key, self.label)


def parse_cases(text: str) -> List[CaseResult]:
    """从一个用例脚本的输出里捞出所有机读单格结果行。

    刻意只认整行且字段数严格相等的行：用例脚本的 stdout 里混着 inductor、profiling 和
    MLIR 诊断的大量输出，宽松匹配会把别的东西认成结果。
    """
    results = []
    for line in text.splitlines():
        if not line.startswith(CASE_MARKER + "\t"):
            continue
        fields = line.split("\t")
        if len(fields) != _FIELD_COUNT + 1:
            continue
        _, suite, cell, group, status, tag, shapes, reason = fields
        if status not in ALL_STATUSES:
            continue
        results.append(CaseResult(suite, cell, group, status, tag, shapes, reason))
    return results


def tally_of(results: Sequence[CaseResult]) -> Dict[str, int]:
    tally = {}
    for result in results:
        tally[result.status] = tally.get(result.status, 0) + 1
    return tally


def worst_of(results: Sequence[CaseResult]) -> Optional[str]:
    """一批结果里最差的那一级，SKIP 不参与。全是 SKIP 则返回 None。"""
    graded = [result.status for result in results if result.status in SEVERITY]
    if not graded:
        return None
    return max(graded, key=lambda status: SEVERITY[status])


def compare(actual: str, expected: str) -> str:
    """把实测与预期比成 `match` / `regressed` / `improved`。

    任一侧是 SKIP（不在严重度阶梯上）时只认相等，不比好坏。
    """
    if actual == expected:
        return "match"
    if actual not in SEVERITY or expected not in SEVERITY:
        return "regressed" if actual in DEFECT_STATUSES else "changed"
    if SEVERITY[actual] > SEVERITY[expected]:
        return "regressed"
    return "improved"


def is_defect(status: str) -> bool:
    return status in DEFECT_STATUSES


# ---------------------------------------------------------------------------- 判级
#
# 判级的核心事实（2026-10-16 实测得到）：下降失败的原因不在异常里。 inductor 抛的异常
# 只有包装文本 `Failed to execute graph code: ...`，真正的原因是 MLIR 诊断，由 C++ 写 fd 2。
# 所以每格都要在 fd 层面捕获 stderr，再拿捕获到的文本判级，否则所有已知支持边界都会被
# 误判成未知错误（第一轮 broadcast 表就是这样把 3 个 BindingConflict 报成了 ERROR）。

# 判据按关键字匹配，顺序有意义：先认具体的拒收族，再认兜底的 unsupported。
# 这些字符串都是 2026-10-16 在 910B 上实际抓到的原文，不是推测的。
REFUSAL_MARKERS: Tuple[Tuple[str, str], ...] = (
    ("has no supported Ascend implementation", "no-impl"),
    ("BindingConflict", "binding-conflict"),
    ("carries a broadcast tile and a full-shape tile", "mixed-tile"),
    ("InvalidWorkspace", "invalid-workspace"),
    ("crosses sibling", "scope-cross"),
    ("UnsupportedOperation", "unsupported-op"),
    ("not supported", "unsupported"),
)


def refusal_tags(text: str) -> List[str]:
    """文本里出现过的拒收族，去重且保持 REFUSAL_MARKERS 的顺序。"""
    tags = []
    for marker, tag in REFUSAL_MARKERS:
        if marker in text and tag not in tags:
            tags.append(tag)
    return tags


def classify(error: BaseException, diagnostics: str) -> Tuple[str, str]:
    """把一格的结果判成 (等级, 标签)。`error is None` 表示该格跑通了。"""
    tags = refusal_tags(diagnostics)
    if error is None:
        if tags:
            return "PASS_FALLBACK", "+".join(tags)
        return "PASS", ""

    # 已经执行到数值验证时，之前被淘汰候选的拒收诊断不能覆盖实际的数值失败。
    if isinstance(error, AssertionError):
        return "MISMATCH", "numeric"
    text = f"{type(error).__name__}: {error}"
    # 异常文本自己也可能带原因（并非所有失败都走 MLIR 诊断），两边都看。
    tags = tags or refusal_tags(text)
    if tags:
        return "REFUSED", "+".join(tags)
    return "ERROR", type(error).__name__
