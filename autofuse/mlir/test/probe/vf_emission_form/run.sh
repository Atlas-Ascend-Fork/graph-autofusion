#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
#===- run.sh - 编并跑 VF 两种生成形态的探针 -------------------------------===#
#
# 问题与统计范围见 `VfEmissionFormProbe.cpp` 的头部。这里只说怎么跑。
#
#     bash autofuse/mlir/test/probe/vf_emission_form/run.sh            # CPU 模拟器，跑数值
#     bash autofuse/mlir/test/probe/vf_emission_form/run.sh --device   # 真实 NPU 编译器，只编
#
# 两档答的是两个问题，缺一档结论就不成立：模拟器那一份是 `g++` 编的，它收得下不等于
# ccec 收得下。`--device` 这一档还会反汇编出两种形态的向量指令直方图——这是「两种形态
# 等不等价」的判据，而不是「编过了」。
#
# 编译选项不自己攒，从 build 目录里现成的 reg 侧 AscendC UT 目标抄
# （`autofuse/tests/v35/ut/ascendc/api_regbase` 的 `test_regbase_ascendc`）：那一份是被
# 既有用例证过的组合，含 `-D__DAV_C310__ -D__NPU_ARCH__=3510` 与一长串 CANN include。
# 手攒的话漏一个宏的失败方式是架构门不命中、整段 VF 被预处理器吃掉，而探针仍然
# 编得过、跑得完——哨兵就是防这个的，但省下排查更好。
#
# 前置：build 目录里要有那个目标的 `flags.make`（任意一档配置都行，`-u`/`-s` 都会配出它）。
# 它只需要 `flags.make`，不需要先把那个目标编出来。
#
# 两个额外 include 是 `flags.make` 里没有、但 tikicpulib 自己要的：
# `tools/cpudebug/include`（`tikicpulib.h` 在这里，不在 `tools/tikicpulib/lib/include`）
# 与 `x86_64-linux/include`（`securec.h`）。链接那一串照抄 backend e2e 的 `link.txt`。
#
#===----------------------------------------------------------------------===#

# `--device` 这一档的选项抄 CANN 自己的 kernel 编译命令
# （`python/site-packages/asc_op_compile_base/asc_op_compiler/super_kernel_op_compile.py`
# 里那一长串 `-I` 与 `--cce-aicore-arch` / `--cce-aicore-only`）。两处容易踩：
#   - 要 `-x cce`。`.cpp` 后缀下 ccec 按普通 C++ 编，失败方式是 CANN 自己的头里报
#     `unknown type name '__gm__'`，看起来像头文件坏了；
#   - include 根要给到 `asc` 目录本身（头里有 `#include "impl/utils/sys_macros.h"` 这种
#     相对 `asc/` 的写法），只给 `asc/include` 不够。
#
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
FLAGS_MAKE="${BASE_DIR}/build/autofuse/tests/v35/ut/ascendc/api_regbase/CMakeFiles/test_regbase_ascendc.dir/flags.make"
CANN="${ASCEND_HOME_PATH:-${HOME}/Ascend/cann}"

if [[ ! -f "${FLAGS_MAKE}" ]]; then
  echo "ERROR: 找不到 ${FLAGS_MAKE}" >&2
  echo "       先配一遍 build（任意一档都行），例如:" >&2
  echo "       bash scripts/test/run_autofuse_test.sh -u -s -m mlir -j 6" >&2
  exit 1
fi
if [[ ! -d "${CANN}" ]]; then
  echo "ERROR: 找不到 CANN (${CANN})。先 source set_env.sh 或设 ASCEND_HOME_PATH。" >&2
  exit 1
fi

DEVICE_ONLY=0
if [[ "${1:-}" == "--device" ]]; then
  DEVICE_ONLY=1
elif [[ $# -gt 0 ]]; then
  echo "用法: run.sh [--device]" >&2
  exit 1
fi

CXX_DEFINES="$(grep -m1 '^CXX_DEFINES' "${FLAGS_MAKE}" | cut -d= -f2-)"
CXX_INCLUDES="$(grep -m1 '^CXX_INCLUDES' "${FLAGS_MAKE}" | cut -d= -f2-)"
EXTRA_INCLUDES="-I${CANN}/tools/cpudebug/include -I${CANN}/x86_64-linux/include"

SIM_LIB_DIRS=("${CANN}/lib64" "${CANN}/tools/tikicpulib/lib" "${CANN}/tools/tikicpulib/lib/Ascend950PR_9599"
              "${CANN}/tools/simulator/Ascend950PR_9599/lib")
LINK_FLAGS=()
RPATH=""
for dir in "${SIM_LIB_DIRS[@]}"; do
  LINK_FLAGS+=("-L${dir}")
  RPATH="${RPATH:+${RPATH}:}${dir}"
done
LINK_FLAGS+=("-Wl,-rpath,${RPATH}" "-Wl,-Bdynamic,--no-as-needed" "-lpem_davinci" "-lcpudebug"
             "${CANN}/tools/tikicpulib/lib/libcpudebug_cceprint.so"
             "${CANN}/tools/tikicpulib/lib/libcpudebug_npuchk.so"
             "${CANN}/tools/tikicpulib/lib/libcpudebug_stubreg.so"
             "${CANN}/lib64/libc_sec.so" "-Wl,--as-needed" "-lpthread")

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

if [[ "${DEVICE_ONLY}" -eq 1 ]]; then
  ASC="${CANN}/$(uname -m)-linux/asc"
  CCEC="${CANN}/bin/ccec"
  OBJDUMP="${CANN}/tools/ccec_compiler/bin/llvm-objdump"
  if [[ ! -x "${CCEC}" ]]; then
    echo "ERROR: 找不到 ccec (${CCEC})" >&2
    exit 1
  fi
  DEVICE_INCLUDES=()
  for dir in impl/adv_api impl/basic_api impl/c_api impl/basic_api/reg_compute impl/simt_api impl/utils              include include/adv_api include/basic_api include/aicpu_api include/c_api              include/basic_api/reg_compute include/simt_api include/utils impl .; do
    DEVICE_INCLUDES+=("-I${ASC}/${dir}")
  done
  DEVICE_INCLUDES+=("-I${ASC}/../tikcpp" "-I${ASC}/../tikcpp/tikcfw" "-I${ASC}/../tikcpp/tikcfw/impl"
                    "-I${ASC}/../tikcpp/tikcfw/interface" "-I${CANN}/include" "-I${CANN}/include/ascendc")
  status=0
  for variant in simd_vf vec_scope; do
    define=()
    [[ "${variant}" == "vec_scope" ]] && define=(-DVEC_SCOPE_VARIANT)
    echo "=== ${variant}（ccec, dav-c310）" >&2
    if ! "${CCEC}" -c -O3 -x cce "${SCRIPT_DIR}/VfEmissionFormProbe.cpp" "${DEVICE_INCLUDES[@]}"          -DVF_PROBE_DEVICE_ONLY -D__DAV_C310__ -D__NPU_ARCH__=3510 -DTILING_KEY_VAR=0 -std=c++17          --cce-aicore-arch=dav-c310 --cce-aicore-only          -mllvm -cce-aicore-addr-transform -mllvm -cce-aicore-stack-size=0x8000          "${define[@]+"${define[@]}"}" -o "${WORK_DIR}/${variant}.o"; then
      echo "${variant}: ccec 编不过 —— 这一代 arch 不收这种形态" >&2
      status=1
      continue
    fi
    # 编过了不是结论：要看编出来的是不是向量指令。按符号打直方图，`.vector.thread` 那个
    # 符号才是真正跑在向量核上的那一段（两种形态都会被外联成它）。
    if [[ -x "${OBJDUMP}" ]]; then
      # 按符号打指令直方图：`.vector.thread` 那个符号才是真正跑在向量核上的那一段
      # （两种形态都会被编译器外联成它），标量侧只剩一条 `VF` 去启动它。
      # 逐字段扫而不用 `match(s, re, arr)`：三参 match 是 gawk 扩展，mawk 上会语法错。
      "${OBJDUMP}" -d "${WORK_DIR}/${variant}.o" 2>/dev/null |
        awk '/^[0-9a-f]+ <.*>:/ { sym = $2; gsub(/[<>:]/, "", sym); next }
             # 对齐填充要剔掉：同一个机器字 41400000 在标量段被反汇编成 NOP、在
             # `.vector.thread` 段被反汇编成 `VST V0, [S72], A0, #0, P0`。不剔的话一段
             # 几十条填充会在直方图里变成几十条 VST（实测 3 条真 VST 会显示成 55 条），
             # 看起来像两种形态差很多。
             /\t41400000\t/ { next }
             sym != "" {
               for (i = 1; i <= NF; i++) {
                 if ($i ~ /^(V[A-Z]+[A-Za-z0-9_.]*|PLT[A-Za-z0-9_.]*|VF|SEND)$/) {
                   count[sym "|" $i]++
                   break
                 }
               }
             }
             END {
               for (k in count) {
                 split(k, p, "|")
                 printf "    %-44s %-14s %d\n", p[1], p[2], count[k]
               }
             }' | sort
    fi
  done
  exit "${status}"
fi

status=0
for variant in simd_vf vec_scope; do
  define=()
  [[ "${variant}" == "vec_scope" ]] && define=(-DVEC_SCOPE_VARIANT)
  echo "=== ${variant} ===" >&2
  # shellcheck disable=SC2086  # CXX_DEFINES / CXX_INCLUDES 是整串选项，要按词展开
  if ! g++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 ${CXX_DEFINES} ${CXX_INCLUDES} ${EXTRA_INCLUDES} \
       "${define[@]+"${define[@]}"}" "${SCRIPT_DIR}/VfEmissionFormProbe.cpp" \
       -o "${WORK_DIR}/probe_${variant}" "${LINK_FLAGS[@]}"; then
    # 编不过与链不上要分开读：前者才是「这种形态收不下」，后者是模拟器库路径漂了
    # （`tools/simulator/<SOC>/lib` 这一支在不同 CANN 布局下换过位置）。
    echo "${variant}: 编/链失败 —— 先看上面是 g++ 的报错（形态收不下）还是 ld 找不到库（路径漂了）" >&2
    status=1
    continue
  fi
  # **在临时目录里跑**：CPU 模拟器会在当前目录下写 `cceprint/`、`npuchk/`、`stub_reg.log`，
  # 在仓库根上跑就会把这些产物落进版本库（编码红线 G11）。
  if ! (cd "${WORK_DIR}" && LD_LIBRARY_PATH="${RPATH}" "./probe_${variant}") | tail -1; then
    status=1
  fi
done
exit "${status}"
