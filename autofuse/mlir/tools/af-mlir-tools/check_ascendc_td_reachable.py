#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""ascendc 方言 td 文件的可达性门禁。

**没有被 `Ops.td` 传递包含的 td 文件，里面的 def 一个都不会被展开**，也就从来没有被
tablegen 检查过：写错了不报错，写对了也用不上。这件事运行期测不出来——一个没被展开的
def 不会让任何断言变红，它根本不存在。

这不是假想的风险。集成时有七个 `*Ext.td` 从来没被包含过，里面 41 个 def 一直是死文本；
接 `afir.where` 时差点把它们当成现成供给用（`where_extend` 看着像，实际建模的是另一个
overload），接 `afir.clip_by_value` 时又确认了一次（`AscendC_CompareExtendOp` 声明的
apiName 是 `Compare` 且带 `AscFunc`，与要发的全局 `CompareExtend<...>` 对不上）。
那七个文件已于 2026-09-24 随 L-17 删除，豁免表因此只剩 `Doc.td`——**表是空的才是常态**，
再往里加一行之前先想清楚为什么这个 def 不该被展开。

这条 hook 必须配 `always_run: true`：它要看的目录整个在仓库级 `exclude` 里（fork 自上游，
任何 hook 都不许改写它），而 `exclude` 无条件生效——配 `files:` 会被过滤成空，hook 静默
跳过，门禁形同虚设。

所以门禁的意义是：**新加的 td 文件必须接进 include 图**，而已经死掉的那七个要显式列在
下面的豁免表里——列在这里是为了让它们可见、可数、有来由，而不是继续隐身。补一个新的
`ascendc.*` 供给时，应当加进**已经被包含**的文件，不要往豁免表里的文件加。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DIALECT_DIR = Path("autofuse/mlir/api/ascendc/include/ascir/Dialect/Asc/IR")
ROOT_TD = "Ops.td"

# 不在 include 图里的文件。每一行都要说得出为什么还留着。
#
# `Doc.td` 是 `-gen-dialect-doc` 的入口，不进 `Ops.td`，本来就不该在图里。
KNOWN_UNREACHABLE = {
    "Doc.td",
}

INCLUDE_RE = re.compile(r'include\s+"([^"]+\.td)"')


def reachable_from(root: Path) -> set[str]:
    """Ops.td 传递包含到的文件集合（相对 DIALECT_DIR 的 posix 路径）。"""
    seen: set[str] = set()
    queue = [ROOT_TD]
    while queue:
        rel = queue.pop()
        if rel in seen:
            continue
        seen.add(rel)
        path = root / rel
        if not path.is_file():
            # 方言目录之外的 include（mlir/... 那些）不在本门禁范围内。
            continue
        for target in INCLUDE_RE.findall(path.read_text(encoding="utf-8")):
            queue.append(target)
    return seen


def main() -> int:
    root = Path.cwd() / DIALECT_DIR
    if not root.is_dir():
        # 目录缺失时必须失败。本脚本按 pre-commit 约定从仓库根运行；
        # 错误工作目录不应让检查静默跳过，类似 files 被 exclude 过滤为空的情况。
        print(
            f"[td-reachable] 找不到方言目录 {DIALECT_DIR}。本脚本必须从仓库根目录运行，"
            f"当前工作目录是 {Path.cwd()}。",
            file=sys.stderr,
        )
        return 1

    all_td = {p.relative_to(root).as_posix() for p in root.rglob("*.td")}
    reachable = reachable_from(root)
    unreachable = all_td - reachable

    failures = []

    new_dead = sorted(unreachable - KNOWN_UNREACHABLE)
    if new_dead:
        failures.append(
            "以下 td 文件没有被 Ops.td 传递包含，里面的 def 不会被展开：\n"
            + "".join(f"    {name}\n" for name in new_dead)
            + "  把它 include 进去，或者——如果它确实该是死的——加进脚本里的 "
            "KNOWN_UNREACHABLE 并写明理由。"
        )

    # 检查豁免仍适用：文件接入 include 图后应移出豁免表。
    stale = sorted(name for name in KNOWN_UNREACHABLE if name in reachable)
    if stale:
        failures.append(
            "以下文件已经在 include 图里了，请从 KNOWN_UNREACHABLE 删掉：\n"
            + "".join(f"    {name}\n" for name in stale)
        )

    missing = sorted(name for name in KNOWN_UNREACHABLE if name not in all_td)
    if missing:
        failures.append(
            "以下文件已经不存在了，请从 KNOWN_UNREACHABLE 删掉：\n"
            + "".join(f"    {name}\n" for name in missing)
        )

    if failures:
        print("[td-reachable] 门禁不通过：\n", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
