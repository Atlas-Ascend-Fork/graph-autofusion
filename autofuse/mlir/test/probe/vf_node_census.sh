#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
#===- vf_node_census.sh - legacy 用例里有多少张图带 VectorFunc -------------===#
#
# 这不是测试，是一个测量工具（与同目录的 `legacy_takeover_table.sh` / `CapabilityBoundaryProbe.cpp`
# 同一性质）。它不断言任何事，产物是一张以 legacy e2e 用例为行的表：
#
#     用例 → 调度产物里生成的 `__simd_vf__` 函数个数
#
# 要它是因为接管表那一列答不出这个数。接管表记的是「哪一道门禁第一个拦下这张图」，
# 于是「因 `VectorFunc` 回落」只数得到 VF 是第一个拦路的那些格；含 VF 但更早被别的缺口
# （节点类型没建模、plan selection 枚举不出候选……）挡住的图，诊断记在别的 op 上，
# VF 一个字都不出现。两个数的差额正是功能需求 4 要的东西：**修掉记下来那条原因也进不来的
# 那些格**。
#
# 判据是生成的 kernel 里的 `inline __simd_vf__ void`：一个 `VectorFunc` 节点必然生成
# 一个这样的函数（`v35/codegen/vec_func_call/vec_func_call.cpp`），所以数它等价于数节点，
# 而且与 MLIR 这一侧的任何门禁都无关——这正是要的性质。
#
# 用法（先把 build 配成 legacy ST 那一档，与接管表同一档，见
# `legacy_takeover_table.sh` 头部）：
#
#     bash autofuse/mlir/test/probe/vf_node_census.sh [-R 正则] [-t 接管表路径]
#
# 统计范围，读表前先看这几条：
#
#   1. 必须把 `AF_ENABLE_MLIR_CODEGEN` 关掉（脚本自己 `env -u`）。开着的话，被 MLIR
#      接管的格子不产 legacy kernel，而严格模式下失败的格子产出来的是个空壳——两种都数不到
#      VF。关掉之后 200 格全走 legacy、全部产出 kernel，这一列才是「调度到底融不融 VF」。
#   2. 归属靠 mtime，所以一次只跑一格、不并行。 取的是这一格跑完后 e2e 树里新写的 `.cpp`。
#      并行会把产物算到别的格头上，而算错了看起来仍然像一张表。
#   3. 按用例名归属，不按目录名。 169 个用例目录注册出 200 个 `*_codegen_v2` 用例
#      （一个目录可以调多次 `backend_e2e_st_test`，sweep 那几格还是 `cmake -P` 驱动器），
#      按目录数会把多格并成一格。
#   4. 数的是生成的 VF 函数个数，不是 VF 节点个数：一张图出多个模板时每个模板各生成一份，
#      所以这一列 >1 不表示融出了多个 VF 子图。判「含不含」只看 >0。
#   5. mem 侧（`_test_codegen$`）那 38 格实测 37 格为 0，与「V1 的
#      `PlatformV1::PartitionSubFunctions` 是空实现」一致。唯一非 0 的
#      `inductor_matmul_elemwise_test`（6 个 VF 函数）**不是反例**：那个用例自己把平台设成
#      `kV35AltNpuArch`（`tests/common/att/soc_perf_test_config.h`），它跑的本来就不是 V1。
#      所以这一列在 mem 侧非 0 时，先查用例自己设了什么平台，再怀疑平台分流。
#
#===----------------------------------------------------------------------===#

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BUILD_DIR="${BASE_DIR}/build"
TESTS_BUILD_DIR="${BUILD_DIR}/autofuse/tests"
# 生成产物落在用例自己的 binary dir（`backend_e2e.cmake` 的 `WORKDIR`），两套用例各一棵树。
E2E_TREES=("${TESTS_BUILD_DIR}/v35/st/backend_e2e_v2" "${TESTS_BUILD_DIR}/st/backend_e2e")

# 产生方是 `vec_func_call.cpp` 的 `VfCall::GenVfFuncDef`。换了拼法要两处一起改。
VF_MARKER='inline __simd_vf__ void'

FILTER_REGEX='_codegen_v2$'
TAKEOVER_TABLE=""

usage() {
  cat <<'EOF'
用法: vf_node_census.sh [选项]

  -R 正则       量哪些用例（ctest -R 的语义；缺省 '_codegen_v2$'，即 reg 侧 200 格。
                mem 侧是 '_test_codegen$'，那一侧应当恒为 0）
  -t 路径       接管表（legacy_takeover_table.sh 的产物）。给了就多打一张交叉表：
                「含 VF / 不含 VF」× 接管表的状态，并把「含 VF 但首因不是 VF」的格列出来
  -h, --help    本说明
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -R) FILTER_REGEX="$2"; shift 2 ;;
    -t) TAKEOVER_TABLE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ! -d "${TESTS_BUILD_DIR}" ]]; then
  echo "ERROR: 找不到 ${TESTS_BUILD_DIR}。build 目录不在 legacy ST 那一档配置上，" >&2
  echo "       先跑: bash scripts/test/run_autofuse_test.sh -s -m backend --enable-mlir -j 6" >&2
  exit 1
fi

mapfile -t CASES < <(cd "${TESTS_BUILD_DIR}" && ctest -N -L build_backend_test1 -R "${FILTER_REGEX}" 2>/dev/null |
  sed -n 's/^ *Test *#[0-9]*: *//p' | sort -u)

if [[ "${#CASES[@]}" -eq 0 ]]; then
  echo "ERROR: ctest 在 ${TESTS_BUILD_DIR} 里没报出匹配 '${FILTER_REGEX}' 的 build_backend_test1 用例。" >&2
  exit 1
fi

echo "[vf] 用例数: ${#CASES[@]}" >&2

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT
ROWS="${WORK_DIR}/rows.tsv"
: >"${ROWS}"

TREES=()
for tree in "${E2E_TREES[@]}"; do
  [[ -d "${tree}" ]] && TREES+=("${tree}")
done

for case_name in "${CASES[@]}"; do
  printf '[vf] %s\n' "${case_name}" >&2
  log="${WORK_DIR}/log"
  stamp="${WORK_DIR}/stamp"
  touch "${stamp}"
  (
    cd "${TESTS_BUILD_DIR}" &&
    env -u AF_ENABLE_MLIR_CODEGEN ctest -V -L build_backend_test1 -R "^${case_name}\$" --no-tests=error
  ) >"${log}" 2>&1
  ctest_rc=$?

  mapfile -t fresh < <(find "${TREES[@]}" -name '*.cpp' -newer "${stamp}" -not -path '*/CMakeFiles/*' 2>/dev/null)
  vf_funcs=0
  for file in ${fresh[@]+"${fresh[@]}"}; do
    count="$(grep -c "${VF_MARKER}" "${file}")"
    vf_funcs=$((vf_funcs + count))
  done

  if [[ "${#fresh[@]}" -eq 0 ]]; then
    # 一个新文件都没写出来：legacy 自己都没跑通，或者二进制不存在。**与「不含 VF」是两件事**，
    # 混在一桶里会把「没量到」读成「调度没融 VF」。
    state="no-output"
  else
    state=$([[ "${vf_funcs}" -gt 0 ]] && echo "vf" || echo "no-vf")
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "${case_name}" "${state}" "${vf_funcs}" "${#fresh[@]}" "${ctest_rc}" >>"${ROWS}"
done

dashes() { printf '%*s' "$1" '' | tr ' ' '-'; }

printf '\n%-52s %-9s %-10s %s\n' "case" "state" "vf_funcs" "generated files"
dashes 92; printf '\n'
sort "${ROWS}" | while IFS=$'\t' read -r case_name state vf_funcs files rc; do
  printf '%-52s %-9s %-10s %s\n' "${case_name}" "${state}" "${vf_funcs}" "${files}"
done

printf '\n%-34s %s\n' "state" "count"
dashes 48; printf '\n'
cut -f2 "${ROWS}" | sort | uniq -c | awk '{printf "%-34s %s\n", $2, $1}'

no_output="$(cut -f2 "${ROWS}" | grep -c '^no-output$')"
printf '\n没有产出的行: %s' "${no_output}"
if [[ "${no_output}" -ne 0 ]]; then
  printf ' —— 量具问题不是结论：那几格 legacy 自己没跑通，先查它\n'
else
  printf '\n'
fi

if [[ -z "${TAKEOVER_TABLE}" ]]; then
  exit 0
fi
if [[ ! -f "${TAKEOVER_TABLE}" ]]; then
  echo "ERROR: 接管表不存在: ${TAKEOVER_TABLE}" >&2
  exit 1
fi

# 交叉表。接管表的行格式见 `legacy_takeover_table.sh` 的 `printf`：用例名 状态 门禁 shortfall 诊断。
awk -v table="${TAKEOVER_TABLE}" -F'\t' '
  BEGIN {
    while ((getline line < table) > 0) {
      if (line !~ /_codegen/) { continue }
      n = split(line, f, /[ \t]+/)
      if (n < 4) { continue }
      state[f[1]] = f[2]
      gate[f[1]] = f[3]
      # 诊断是第 5 列起的全部内容。**按字段拼回来，不用正则吃掉前四列**：
      # `{4}` 这种区间量词不是所有 awk 都收，而不收时它静默什么都不吃，于是诊断列里
      # 连着用例名一起打出来——看起来仍然像一张表。
      rest = ""
      for (i = 5; i <= n; i++) { rest = rest (i == 5 ? "" : " ") f[i] }
      diag[f[1]] = rest
    }
  }
  {
    if (!($1 in state)) { missing++; next }
    # 「首因是 VF」怎么认。**这条判据跟着 bridge 的诊断措辞走，换了措辞要两处一起改**
    # ——和上面 `VF_MARKER` 同一类耦合，但它漏改的后果更隐蔽：匹配不上不会报错，只会把
    # 那几格算进「首因不是 VF」，于是「只差 VF」那个数静默变成 0，而 0 看起来像个结论。
    # 2026-10-26 就栽过一次：VF 那道门禁的诊断从「不支持这个节点类型」换成了按 dtype /
    # 按子图节点分的两条，旧判据只认前者，于是 17 格读成了 0 格。
    # 今天要认的三种措辞：
    #   does not support node 'X' of type 'VectorFunc'      （节点类型整个没建模）
    #   UnsupportedOperation: VectorFunc node 'X' reads ... （dtype 白名单）
    #   UnsupportedOperation: VectorFunc subgraph node ...  （子图里的节点没建模）
    first_is_vf = (diag[$1] ~ /of type .VectorFunc./ || diag[$1] ~ /UnsupportedOperation: VectorFunc /)
    key = ($2 == "vf" ? "含 VF" : ($2 == "no-vf" ? "无 VF" : "没产出")) "\t" state[$1] \
          "\t" (first_is_vf ? "首因是 VF" : "首因不是 VF")
    cross[key]++
    if ($2 == "vf" && !first_is_vf) {
      # **诊断打全，不截断。** 原先按 90 字硬切，于是最后一个词常被切掉词尾——那半个词
      # 既读不出是截断，还会被 pre-commit 的 codespell 当成拼写错，把归档这张表的提交拦下
      # （2026-10-26 栽过一次）。这一列的读法本来就是「归因靠它」，截了反而要回去查原表。
      printf "%-52s %-13s %-24s %s\n", $1, state[$1], gate[$1], diag[$1] > "/dev/stderr"
    }
  }
  END {
    printf "\n%-10s %-14s %-14s %s\n", "VF", "接管表状态", "首因", "count"
    for (k in cross) { split(k, p, "\t"); printf "%-10s %-14s %-14s %d\n", p[1], p[2], p[3], cross[k] }
    if (missing) { printf "\n接管表里查不到的用例: %d —— 两张表不是同一轮量的\n", missing }
  }
' "${ROWS}" 2>"${WORK_DIR}/vf_blocked_earlier.txt"

printf '\n含 VF 但首因不是 VF 的格（修掉记下来那条原因也进不来）:\n'
dashes 140; printf '\n'
sort "${WORK_DIR}/vf_blocked_earlier.txt"
printf '小计: %s\n' "$(wc -l <"${WORK_DIR}/vf_blocked_earlier.txt")"
