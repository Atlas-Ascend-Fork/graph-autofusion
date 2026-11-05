#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------------------

"""上板跑用例所需的环境装配，以及「环境真的搭起来了」的前置判据。

为什么要有这个模块：`AGENTS.md` §2.4.4 / §2.4.6 / §2.4.7 已经把配方写清楚了，但它是散落的
shell 片段，真正可执行的那份一直只存在于设备上的两个手写脚本（`/mnt/workspace/run_npu.sh`
与 `setup_overlay.sh`）里。后果是换一台机器、换一个 clone 就跑不起来，而且**失败不报错**：
overlay 没生效时 `import autofuse.pyautofuse` 会静默回落到 CANN 自带的那份（2 月版），
用例照样跑完、照样打 PASS，结论与本次改动无关。本模块把配方搬进仓里，并把那条判据做成断言。

三件事按这个顺序成立，缺一件后面都不算：

1. **LD_LIBRARY_PATH 要带上 build 树里每个放 `.so` 的目录，且排在安装态之前。**
   安装态的 `libgraph_base_af.so` 是 2 月的，不含本仓 `libaihac_ir.so` 需要的 `af::proto`
   符号；漏一个目录就在 import 时报 undefined symbol。这里**扫出来**而不是手抄一份路径
   列表：手抄的那份在 build 布局一变时就少一项，而少一项的表现是 import 报错，不是
   「路径不全」这种能一眼看懂的提示。
2. **overlay 要同时覆盖 Python 文件和 `pyautofuse.so`，两者必须同源同 build**（§2.4.5）。
   只覆盖一侧的症状是 `ascendc_compile.py` 缺符号、或生成的 kernel 与源码对不上。
3. **跑之前验 `autofuse.pyautofuse` 的真实 origin 在 overlay 里**（§2.4.4）。这是整套里
   唯一能把「静默回落」抓出来的判据，所以它是硬断言而不是警告。

关于 `PYTHONPATH` 的一处出入：§2.4.6 的片段写的是 `PYTHONPATH="$OVERLAY_DIR"`，而实际能
跑通的是 `"$OVERLAY_DIR/python/site-packages"`——overlay 的目录层级要模仿 site-packages
（§3.2 的原话），包名 `autofuse` 必须是 `PYTHONPATH` 里某一项的直接子目录。这里按后者。
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from support import att_selection

# overlay 根下要补的、指向真实 CANN 的符号链接。`ascendc_compile.py` 按
# `PYF_PATH/../../..` 找 CANN 根，所以 overlay 根必须能冒充 CANN 根（§3.2）。
CANN_LINK_NAMES = (
    "tools",
    "lib64",
    "opp",
    "aarch64-linux",
    "include",
    "pkg_inc",
    "compiler",
    "runtime",
    "bin",
    "conf",
)

# 跑之前要核对 origin 的模块，取自 §2.4.4。
PROBED_MODULES = (
    "torch",
    "torch_npu",
    "autofuse.pyautofuse",
    "autofuse.ascendc_compile",
    "autofuse.compile_adapter",
)


class DeviceEnvError(RuntimeError):
    """环境没搭起来。与数值缺陷严格区分：这一类的结论一律无效，不该被判级。"""


def repo_root(start: Optional[Path] = None) -> Path:
    """从本文件往上找仓根。认 `autofuse/mlir/AGENTS.md` 而不是 `.git`：
    设备上的工作副本有时是 `git clone` 的子目录或 worktree，`.git` 可能是个文件。"""
    here = (start or Path(__file__)).resolve()
    for candidate in [here] + list(here.parents):
        if (candidate / "autofuse" / "mlir" / "AGENTS.md").is_file():
            return candidate
    raise DeviceEnvError(
        "找不到仓根（向上没有 autofuse/mlir/AGENTS.md）：{}".format(here)
    )


def git_sha(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return out.stdout.decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def build_dir(repo: Path, override: Optional[str] = None) -> Path:
    """autofuse 的 build 目录。`AUTOFUSE_BUILD_DIR` 优先，与设备脚本一致。"""
    raw = override or os.environ.get("AUTOFUSE_BUILD_DIR")
    path = Path(raw).resolve() if raw else (repo / "build" / "autofuse").resolve()
    if not path.is_dir():
        raise DeviceEnvError(
            "build 目录不存在：{}\n"
            "    先在设备上编一次（afnpu build），或用 --build-dir 指过去。".format(
                path
            )
        )
    return path


def library_path_entries(build: Path) -> List[str]:
    """build 树里所有直接含 `.so` 的目录，build 根排在最前。

    排序是确定的（按路径），这样两次跑出来的 `LD_LIBRARY_PATH` 一致——不确定的顺序会让
    「换一次运行就好了/坏了」这种假象出现在符号解析上。
    """
    found = set()
    for so in build.rglob("*.so"):
        found.add(so.parent.resolve())
    for so in build.rglob("*.so.*"):
        found.add(so.parent.resolve())
    entries = sorted(str(path) for path in found)
    root = str(build.resolve())
    if root in entries:
        entries.remove(root)
    return [root] + entries


def cann_root() -> Path:
    for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        value = os.environ.get(key)
        if value and Path(value).is_dir():
            return Path(value).resolve()
    raise DeviceEnvError(
        "找不到 CANN 根：ASCEND_HOME_PATH / ASCEND_TOOLKIT_HOME 都没设或不存在。\n"
        "    先 source ~/Ascend/ascend-toolkit/set_env.sh（§2.4.4：一个 shell 只 source 一套）。"
    )


def overlay_dir(repo: Path, override: Optional[str] = None) -> Path:
    """overlay 的落点。默认按 sha 命名，与 §2.4.1 的 `/mnt/workspace/agent/<...>` 一致。

    按 sha 命名是必要的：overlay 里的 `pyautofuse.so` 必须与 `libaihac_codegen.so` 同源
    （§2.4.5），不按 sha 分开的话切分支后会悄悄复用上一个分支的那份。
    """
    raw = override or os.environ.get("AF_NPU_OVERLAY")
    if raw:
        return Path(raw).resolve()
    base = Path(os.environ.get("AF_NPU_AGENT_DIR", "/mnt/workspace/agent"))
    return (base / "overlay-{}".format(git_sha(repo))).resolve()


def ensure_overlay(
    repo: Path, build: Path, overlay: Path, rebuild: bool = False
) -> Path:
    """按 §2.4.6 搭 overlay：本仓的 Python 文件 + 本 build 的 `pyautofuse.so`。

    每次调用都无条件覆盖 `.py` 和 `pyautofuse.so`，所以**常规重编后不需要 `rebuild`**：
    overlay 不在构建系统的依赖图里，构建系统不知道它存在，这里就只能每次拷一遍。
    （设备上那个手写 `setup_overlay.sh` 是 `rm -rf` 重建，所以它必须每次全量；这里不必。）

    `rebuild=True` 先删掉再建，解决拷贝盖不住的那一种：仓里**删掉**了某个 `.py`，overlay
    里会残留一份陈旧的，而 `import` 照样能找到它。另一种是 overlay 落点被别的 sha 用过。
    """
    module = build / "compiler" / "py_module" / "pyautofuse.so"
    sources = sorted((repo / "autofuse" / "compiler" / "python").glob("*.py"))
    if not module.is_file():
        raise DeviceEnvError(
            "build 里没有 pyautofuse.so：{}\n    编译没成功或没编这个目标。".format(
                module
            )
        )
    if not sources:
        raise DeviceEnvError("仓里没有 autofuse/compiler/python/*.py，overlay 无从搭起")

    site = overlay / "python" / "site-packages" / "autofuse"
    if rebuild and overlay.exists():
        shutil.rmtree(overlay)
    site.mkdir(parents=True, exist_ok=True)
    for source in sources + [module]:
        # 已运行的进程可能仍映射旧扩展；原地截断覆盖会使其访问失效页。
        # 在同目录写完后原子替换，保留已有映射及并发导入看到的完整文件。
        with tempfile.NamedTemporaryFile(
            dir=site, prefix=source.name + ".", delete=False
        ) as staged:
            staged_path = Path(staged.name)
        try:
            shutil.copy2(source, staged_path)
            staged_path.replace(site / source.name)
        finally:
            staged_path.unlink(missing_ok=True)

    cann = cann_root()
    for name in CANN_LINK_NAMES:
        target = cann / name
        if not target.exists():
            continue
        link = overlay / name
        if link.is_symlink():
            if Path(os.readlink(str(link))) == target:
                continue
            link.unlink()
        elif link.exists():
            # overlay 根下出现同名真实目录，说明这个 overlay 不是本模块建的。
            # 直接改它可能动到别人的东西，报错让人自己看一眼。
            raise DeviceEnvError(
                "overlay 根下 {} 是真实目录而不是符号链接：{}\n"
                "    这个 overlay 不像是本模块建的，换一个 --overlay 落点。".format(
                    name, link
                )
            )
        link.symlink_to(target)
    return overlay


def environ_for_run(
    repo: Path,
    build: Path,
    overlay: Path,
    mlir_codegen: bool = True,
    extra: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """跑一个用例脚本用的环境。

    不设 `TORCH_COMPILE_DEBUG_DIR` / `TORCHINDUCTOR_CACHE_DIR` 这一类（§2.4.7）：它们由
    `NpuRunner.configure_work_dir` 按 `--work-dir` 设，一份实验一个目录。在这里再设一遍
    只会和它打架，而且会让「不同 route 不共用 cache」这条取决于谁最后赋值。
    """
    env = att_selection.environment_for_mlir_config("mlir" if mlir_codegen else "none")
    libs = library_path_entries(build)
    old_libs = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(libs + ([old_libs] if old_libs else []))

    site = str(overlay / "python" / "site-packages")
    old_py = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join([site] + ([old_py] if old_py else []))
    # 用例脚本用 `from support.xxx import ...` 导入，所以用例目录也要在路径里；
    # 子进程的 cwd 是各实验的 work-dir（§2.4.1：产物不落源码目录），不能靠 cwd 兜。
    env["PYTHONPATH"] = ":".join([env["PYTHONPATH"], str(cases_dir(repo))])

    # 默认沿用当前解释器的标准库前缀；显式选择其他解释器时可通过 AF_NPU_PYTHONHOME 覆盖。
    env.setdefault("PYTHONHOME", os.environ.get("AF_NPU_PYTHONHOME", sys.base_prefix))
    if extra:
        env.update({key: str(value) for key, value in extra.items()})
    return env


def cases_dir(repo: Path) -> Path:
    return repo / "autofuse" / "mlir" / "test" / "e2e" / "npu"


def python_bin() -> str:
    """跑用例的解释器。§2.4.4 要求显式指定，不靠 PATH。"""
    return os.environ.get("AF_NPU_PYTHON", sys.executable)


_PROBE_SCRIPT = """
import importlib.util
import sys

print("sys.executable=" + sys.executable)
for name in {modules!r}:
    try:
        spec = importlib.util.find_spec(name)
    except Exception as error:
        print(name + " IMPORT_ERROR " + type(error).__name__ + ": " + str(error))
        continue
    print(name + " " + (spec.origin if spec and spec.origin else "None"))
"""


def preflight(
    env: Dict[str, str], overlay: Path, python: Optional[str] = None
) -> Dict[str, str]:
    """跑真用例之前核对模块来源（§2.4.4），并断言 pyautofuse 来自 overlay。

    为什么是硬断言：overlay 没生效时什么都不会报错，用例会用 CANN 自带的 2 月版跑完并
    打出 PASS，而那个结论与本次改动无关。这条判据是整套里唯一能把它抓出来的。
    """
    python = python or python_bin()
    script = _PROBE_SCRIPT.format(modules=list(PROBED_MODULES))
    completed = subprocess.run(
        [python, "-c", script],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    text = completed.stdout.decode("utf-8", "replace")
    if completed.returncode != 0:
        raise DeviceEnvError(
            "模块来源探测失败（退出码 {}）：\n{}".format(completed.returncode, text)
        )

    origins = {}
    for line in text.splitlines():
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[0] in PROBED_MODULES:
            origins[parts[0]] = parts[1].strip()

    missing = [
        name for name in ("torch", "torch_npu") if origins.get(name) in (None, "None")
    ]
    if missing:
        raise DeviceEnvError("导入不到 {}：\n{}".format("/".join(missing), text))

    pyautofuse = origins.get("autofuse.pyautofuse")
    if not pyautofuse or pyautofuse == "None":
        raise DeviceEnvError(
            "找不到 autofuse.pyautofuse——环境没搭起来，结论无效：\n" + text
        )
    if str(overlay) not in pyautofuse:
        raise DeviceEnvError(
            "autofuse.pyautofuse 来自 {}，不在 overlay（{}）里。\n"
            "    这是静默回落到了 CANN 自带的旧版，本次结论与改动无关。\n"
            "    重编之后要重建 overlay：run.py --rebuild-overlay".format(
                pyautofuse, overlay
            )
        )
    # find_spec 只证明文件存在；实际导入才能发现缺少依赖或动态库 ABI 不匹配。
    imports = "import importlib\nfor name in {!r}:\n    importlib.import_module(name)\n".format(
        list(PROBED_MODULES)
    )
    loaded = subprocess.run(
        [python, "-c", imports],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if loaded.returncode != 0:
        raise DeviceEnvError(
            "模块导入失败（退出码 {}）：\n{}".format(
                loaded.returncode, loaded.stdout.decode("utf-8", "replace")
            )
        )
    return origins
