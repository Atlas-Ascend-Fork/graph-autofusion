#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
#===- legacy_takeover_table.sh - legacy 用例的接管表 ---------------------===#
#
# 这不是测试，是一个测量工具（与同目录的 `CapabilityBoundaryProbe.cpp` 同一性质）。它不断言
# 任何事，产物是一张以 legacy e2e 用例为行的表：
#
#     用例 → {接管 / 回落(第一条诊断) / 硬失败 / 不在范围}
#
# 要它是因为拿 legacy 套件跑一遍全绿证明不了任何事：MLIR 这一侧收不了的图会静默回落给
# legacy，回落了也是绿的。`AF_ENABLE_MLIR_CODEGEN=strict` 这一档把回落也判成失败，并给每一次
# 调用在 stderr 上打一行可机读的标签
#
#     AF_MLIR_STRICT gate=<哪道门禁拦下的> shortfall=<缺省那档会不会回落> graph=<图名>
#
# 这个脚本只是把那些标签按用例收成一张表。两列都不能由另一列推出来，所以两列都要：`gate`
# 说是哪一道门禁，`shortfall` 说缺省那一档会回落还是也失败——那正是表里「回落」与「硬失败」
# 两态的分界。
#
# 用法（先把 build 配成 legacy ST 那一档，见下）：
#
#     bash autofuse/mlir/test/probe/legacy_takeover_table.sh [-j N] [-R 正则] [--no-build]
#
# 前置条件：build 目录要处于 legacy ST 那一档配置（`RUN_TEST=1` + `ENABLE_AUTOFUSE_MLIR=ON`），
# 否则 `*_codegen` 这些目标根本不存在。配法是跑一遍
#
#     bash scripts/test/run_autofuse_test.sh -s -m backend --enable-mlir -j 6
#
# 这一步会覆盖 `mlir-test.sh` 用的那一档配置（MLIR 自己的测试目标是用 `cmake -U RUN_TEST`
# 配的，两档互斥）。量完要回到 MLIR 测试，删掉 `build/CMakeCache.txt` 再跑 `mlir-test.sh`。
# 脚本不替你改配置：它只在已经配好的 build 目录上做「编 + 跑 + 收表」，这样它跑错时不会
# 把你的 build 目录连带弄坏。
#
# 已知统计范围限制，读表前先看这一条：legacy 的 `*_codegen` 用例走的是
# `Codegen::Generate`（`MlirCallerScene::kTensorFlow`），不是设计范围里的
# `GenerateForInductor`。场景这一位会传进 ATT（`is_inductor_scene`），而 ATT 在候选准备那一道
# 门禁里面——所以 inductor 与 TF 在那一道上的差异这张表量不到。表里
# `gate=candidate-preparation` 的行要按这一条打折扣看；其余各道门禁（bridge、能力层、pattern、
# 产物装配）与场景无关。把这个统计范围统一要给 legacy 的用例样板加一个 inductor 入口开关，
# 那是功能需求 7（样板收敛）的事。
#
#===----------------------------------------------------------------------===#

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BUILD_DIR="${BASE_DIR}/build"
TESTS_BUILD_DIR="${BUILD_DIR}/autofuse/tests"

# 产生方是 `bridge/CodegenOrchestration/CodegenOrchestration.h` 的
# `kMlirCodegenStrictMarker`。全仓只有这一个字符串，改名要两处一起改。
MARKER="AF_MLIR_STRICT"

JOBS=6
FILTER_REGEX=""
DO_BUILD=1

usage() {
  cat <<'EOF'
用法: legacy_takeover_table.sh [选项]

  -j N          编译并行度（缺省 6；本机 -j 8 会被 OOM killer 杀掉）
  -R 正则       只量名字匹配该正则的用例（ctest -R 的语义）
  --no-build    跳过 make，直接跑已经编好的用例
  -h, --help    本说明
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -j) JOBS="$2"; shift 2 ;;
    -R) FILTER_REGEX="$2"; shift 2 ;;
    --no-build) DO_BUILD=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ! -d "${TESTS_BUILD_DIR}" ]]; then
  echo "ERROR: 找不到 ${TESTS_BUILD_DIR}。build 目录不在 legacy ST 那一档配置上，" >&2
  echo "       先跑: bash scripts/test/run_autofuse_test.sh -s -m backend --enable-mlir -j ${JOBS}" >&2
  exit 1
fi

# 量哪些用例：**按 ctest 自己报的 `build_backend_test1` 标签取，不按目录名猜**。那个标签是
# `backend_e2e.cmake` 给生成器那一阶段打的，mem 侧与 reg 侧两套用例都在里面；按目录名猜会把
# 被注释掉的 `add_subdirectory` 也算进来。
if [[ -n "${FILTER_REGEX}" ]]; then
  mapfile -t CASES < <(cd "${TESTS_BUILD_DIR}" && ctest -N -L build_backend_test1 -R "${FILTER_REGEX}" 2>/dev/null |
    sed -n 's/^ *Test *#[0-9]*: *//p' | sort -u)
else
  mapfile -t CASES < <(cd "${TESTS_BUILD_DIR}" && ctest -N -L build_backend_test1 2>/dev/null |
    sed -n 's/^ *Test *#[0-9]*: *//p' | sort -u)
fi

if [[ "${#CASES[@]}" -eq 0 ]]; then
  echo "ERROR: ctest 在 ${TESTS_BUILD_DIR} 里没报出任何 build_backend_test1 用例。" >&2
  exit 1
fi

echo "[takeover] 用例数: ${#CASES[@]}" >&2

if [[ "${DO_BUILD}" -eq 1 ]]; then
  echo "[takeover] 编译 ${#CASES[@]} 个生成器目标 (-j ${JOBS})" >&2
  # 目标名就是用例名：`backend_e2e.cmake` 里 `add_test(NAME ${TEST_NAME}_codegen ...)` 与可执行
  # 文件同名。一次 make 带多个目标，所以用 `make -C`——本机 `cmake --build --target` 会把多个
  # 目标名拼成一个。
  if ! make -C "${BUILD_DIR}" -j "${JOBS}" "${CASES[@]}" >/dev/null 2>&1; then
    echo "[takeover] WARNING: 有目标没编过；那些行会落在 no-mlir-call 这一桶里" >&2
  fi
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

ROWS="${WORK_DIR}/rows.tsv"
: >"${ROWS}"

# 一个用例跑一遍，**严格模式下跑，而且只跑生成器那一阶段**：第二阶段编的是生成的 kernel，
# 与「谁接管了这张图」无关。
#
# 用 `ctest -R '^name$'` 而不是直接执行二进制：`LD_LIBRARY_PATH` 那一长串在
# `CTestTestfile.cmake` 里，路径不对时失败方式是调度在 `ScheduleUtils::ClearAllSizeVar` 段错误，
# 看起来像建图建错了。
#
# 必须 `-V` 而不是 `--output-on-failure`。 后者只在用例失败时才打输出，而「接管了因此变绿」
# 的用例恰好不失败——标签一行都收不到，整片被记成 `no-mlir-call`。第一版就是这么量错的：
# `add_abs_test` 这张 UT 自己的 fixture 都显示没进过 MLIR。
for case_name in "${CASES[@]}"; do
  printf '[takeover] %s\n' "${case_name}" >&2
  log="${WORK_DIR}/${case_name}.log"
  (
    cd "${TESTS_BUILD_DIR}" &&
    AF_ENABLE_MLIR_CODEGEN=strict ctest -V -L build_backend_test1 \
      -R "^${case_name}\$" --no-tests=error
  ) >"${log}" 2>&1
  ctest_rc=$?

  # 标签行。一个用例里可能有不止一次 codegen 调用，所以全收、去重。
  mapfile -t labels < <(grep -o "${MARKER} gate=[^ ]* shortfall=[^ ]*" "${log}" | sort -u)
  first_error="$(grep -m1 -o 'error:.*' "${log}" | cut -c1-150)"

  if [[ "${#labels[@]}" -eq 0 ]] && grep -qiE "Unable to find executable|Unable to find required file" "${log}"; then
    # **二进制根本不存在**，与「没走挂钩点」是两件事，必须分开——混在一桶里会让人把
    # 「没编出来」读成「这个用例不经过 MLIR」。实测踩过：`--no-build` 跑 reg 侧那一遍，
    # 80 行全是这个，而 `takeover` 是 0，看起来像 MLIR 在 reg 侧什么都接不了。
    # 判据：那一格的 ctest 用时是 0.0x 秒，而且 ctest 自己会说 Unable to find executable。
    state="no-binary"
    gate="-"
    shortfall="-"
  elif [[ "${#labels[@]}" -eq 0 ]]; then
    # 一行标签都没有：MLIR 的挂钩点一次都没被调到。开关是开着的（本脚本自己设的），所以来路有
    #   用例走的是**没挂钩的 API** —— `TilingLib::Generate`，或直接调
    #     `GenerateKernel` / `GenerateTilingData` / `GenerateTiling`（挂钩只在
    #     `Codegen::Generate` 与 `Codegen::GenerateForInductor` 两处，这是 2026-08-26
    #     launch 设计记下的已知限制）；
    #   二进制没编出来，或用例根本没调 codegen。
    # 脚本分不开这几种，所以**单列一桶、不当成回落**——混进 `fallback` 里就再也看不见了。
    # 这一桶的每一行都要人工归因一次；归因完了它不是「状态未知」，但在表里仍然是这一桶。
    state="no-hook-call"
    gate="-"
    shortfall="-"
  else
    # 一个用例多张图时取**最差**的那一格：有一张没接管，这个用例就不算接管。
    state="takeover"; gate="none"; shortfall="no"
    for label in "${labels[@]}"; do
      lgate="${label#*gate=}"; lgate="${lgate%% *}"
      lshort="${label##*shortfall=}"
      case "${lgate}" in
        none)
          continue
          ;;
        cube-fusion)
          if [[ "${state}" == "takeover" ]]; then
            state="out-of-scope"; gate="${lgate}"; shortfall="${lshort}"
          fi
          ;;
        *)
          if [[ "${lshort}" == "yes" ]]; then
            state="fallback"
          else
            # `shortfall=unknown`（抛了异常，观察者已出作用域）也归这里：答不出来时按硬失败
            # 处置是保守方向，不会把一个真 bug 藏进「回落」那一桶。
            state="hard-failure"
          fi
          gate="${lgate}"; shortfall="${lshort}"
          ;;
      esac
    done
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${case_name}" "${state}" "${gate}" "${shortfall}" "${ctest_rc}" "${first_error}" >>"${ROWS}"
done

# 表打到 stdout，进度打到 stderr —— `> table.txt` 拿到的就是干净的表。
dashes() { printf '%*s' "$1" '' | tr ' ' '-'; }

printf '\n%-52s %-13s %-24s %-10s %s\n' "case" "state" "gate" "shortfall" "first diagnostic"
dashes 170; printf '\n'
sort "${ROWS}" | while IFS=$'\t' read -r case_name state gate shortfall rc first_error; do
  printf '%-52s %-13s %-24s %-10s %s\n' "${case_name}" "${state}" "${gate}" "${shortfall}" "${first_error}"
done

printf '\n%-34s %s\n' "state" "count"
dashes 48; printf '\n'
cut -f2 "${ROWS}" | sort | uniq -c | awk '{printf "%-34s %s\n", $2, $1}'

printf '\n%-34s %s\n' "gate" "count"
dashes 48; printf '\n'
cut -f3 "${ROWS}" | sort | uniq -c | awk '{printf "%-34s %s\n", $2, $1}'

missing="$(cut -f2 "${ROWS}" | grep -c '^no-binary$')"
unhooked="$(cut -f2 "${ROWS}" | grep -c '^no-hook-call$')"
printf '\n二进制不存在的行: %s' "${missing}"
if [[ "${missing}" -ne 0 ]]; then
  printf ' —— 量具问题不是结论：去掉 --no-build 重跑\n'
else
  printf '\n'
fi
printf '没打出标签的行: %s' "${unhooked}"
if [[ "${unhooked}" -ne 0 ]]; then
  printf ' —— 逐个归因：这些用例调的是哪个 API？归完才算不是「状态未知」\n'
else
  printf '\n'
fi
