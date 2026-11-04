#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, and you can use and redistribute it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License"). You should have received a copy
# of the License along with this program. If not, see <https://gitcode.com/cann/graph-autofusion/blob/master/LICENSE>.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

BASEPATH=$(cd "$(dirname "$0")/../../.." && pwd)
AUTOFUSE_PATH="${BASEPATH}/autofuse"
BUILD_PATH="${BASEPATH}/build"
AUTOFUSE_BUILD_PATH="${BUILD_PATH}/autofuse"
NPU_CASE_DIR="${AUTOFUSE_PATH}/mlir/test/e2e/npu"
INSTALL_DIR="${BUILD_PATH}/dev-install"
THREAD_NUM=40
MODE="mlir"
STEPS=0
DEVICE="npu:0"
SELECTED_CASES=()

usage() {
  cat <<USAGE
Usage:
  bash autofuse/mlir/test/run_autofuse_npu_e2e.sh [options]

Options:
  -j, --jobs <N>       Build parallelism, default 8.
      --case <NAME>    Run one case by basename, for example load_add_store.
                       Repeat the option to run multiple cases. Default: all cases.
      --mode <MODE>    NpuRunner backend mode: mlir, legacy, or both. Default: mlir.
      --steps <N>      Profiling steps. Use 0 to disable profiling. Default: 0.
      --device <DEV>   Torch NPU device, default npu:0.
      --list           List available NPU E2E cases.
  -h, --help           Show this help.

Environment:
  LLVM_BUILD_DIR       LLVM install tree used by the Autofuse MLIR build.
  ASCEND_HOME_PATH     Active CANN installation, normally exported by set_env.sh.
USAGE
}

checkopts() {
  while (($# > 0)); do
    case "$1" in
      -h | --help)
        usage
        exit 0
        ;;
      -j | --jobs)
        THREAD_NUM="$2"
        shift 2
        ;;
      --case)
        SELECTED_CASES+=("$2")
        shift 2
        ;;
      --mode)
        MODE="$2"
        shift 2
        ;;
      --steps)
        STEPS="$2"
        shift 2
        ;;
      --device)
        DEVICE="$2"
        shift 2
        ;;
      --list)
        local case_path
        for case_path in "${NPU_CASE_DIR}"/*.py; do
          [[ -f "${case_path}" ]] && basename "${case_path}"
        done
        exit 0
        ;;
      --)
        shift
        break
        ;;
      *)
        echo "Undefined option: $1" >&2
        usage
        exit 1
        ;;
    esac
  done

  if [[ -z "${LLVM_BUILD_DIR:-}" ]]; then
    echo "ERROR: LLVM_BUILD_DIR is required." >&2
    exit 1
  fi
  if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ERROR: ASCEND_HOME_PATH is required. Source the active CANN set_env.sh first." >&2
    exit 1
  fi
  case "${MODE}" in
    mlir | legacy | both) ;;
    *)
      echo "ERROR: invalid mode '${MODE}'; expected mlir, legacy, or both." >&2
      exit 1
      ;;
  esac
}

configure_build() {
  if ! bash "${AUTOFUSE_PATH}/mlir/scripts/check_llvm_dependency_resolution.sh"; then
    echo "check_llvm_dependency_resolution failed." >&2
    return 1
  fi

  mkdir -p "${BUILD_PATH}"
  cmake -U RUN_TEST -S "${BASEPATH}" -B "${BUILD_PATH}" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_C_COMPILER="${CC:-gcc}" \
    -D CMAKE_CXX_COMPILER="${CXX:-g++}" \
    -D ASCEND_3RD_LIB_PATH="${BASEPATH}/output/third_party" \
    -D CANN_3RD_LIB_PATH="${BASEPATH}/output/third_party" \
    -D ASCEND_INSTALL_PATH="${ASCEND_HOME_PATH}" \
    -D ENABLE_OPEN_SRC=True \
    -D ENABLE_SYMENGINE=True \
    -D ENABLE_AUTOFUSE_MLIR_TESTS=ON \
    -D ENABLE_AUTOFUSE_MLIR=ON
}

build_pyautofuse() {
  cmake --build "${BUILD_PATH}" --target pyautofuse -j "${THREAD_NUM}"
}

install_autofuse_runtime() {
  local install_inputs=(
    "${AUTOFUSE_BUILD_PATH}/compiler/py_module/pyautofuse.so"
    "${AUTOFUSE_PATH}/compiler/python/"*.py
    "${AUTOFUSE_BUILD_PATH}/libaihac_codegen.so"
    "${AUTOFUSE_BUILD_PATH}/graph_metadef/graph/ascendc_ir/libaihac_ir.so"
    "${AUTOFUSE_BUILD_PATH}/graph_metadef/graph/ascendc_ir/generator/libaihac_ir_register.so"
    "${AUTOFUSE_BUILD_PATH}/graph_metadef/graph/libgraph_af.so"
    "${AUTOFUSE_BUILD_PATH}/graph_metadef/graph/libgraph_base_af.so"
    "${AUTOFUSE_BUILD_PATH}/graph_metadef/graph/expression/libaihac_symbolizer_af.so"
  )
  local install_fingerprint
  install_fingerprint=$(sha256sum "${install_inputs[@]}" | sha256sum | cut -d' ' -f1)
  local install_stamp="${INSTALL_DIR}/.autofuse-install-fingerprint"

  if [[ ! -f "${install_stamp}" || "$(cat "${install_stamp}")" != "${install_fingerprint}" ]]; then
    rm -rf "${INSTALL_DIR}"
    cmake --install "${AUTOFUSE_BUILD_PATH}" --prefix "${INSTALL_DIR}"
    printf '%s\n' "${install_fingerprint}" > "${install_stamp}"
  else
    echo "Autofuse install tree is up to date; skip cmake --install."
  fi

  local install_lib_dir="${INSTALL_DIR}/$(uname -m)-linux/lib64"
  export LD_LIBRARY_PATH="${install_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export PYTHONPATH="${INSTALL_DIR}/python/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
}

verify_python_runtime() {
  python3 - <<'PY'
import os

import autofuse
import autofuse.ascendc_compile
import autofuse.pyautofuse

expected_prefix = os.environ["AUTOFUSE_NPU_INSTALL_DIR"] + "/python/site-packages/"
paths = (
    autofuse.__file__,
    autofuse.pyautofuse.__file__,
    autofuse.ascendc_compile.__file__,
)
for path in paths:
    assert path.startswith(expected_prefix), path
assert autofuse.ascendc_compile.ASCEND_PATH == os.environ["ASCEND_HOME_PATH"]
PY
}

collect_cases() {
  if ((${#SELECTED_CASES[@]} == 0)); then
    local case_path
    for case_path in "${NPU_CASE_DIR}"/*.py; do
      [[ -f "${case_path}" ]] && printf '%s\n' "${case_path}"
    done
    return
  fi

  local case_name case_path
  for case_name in "${SELECTED_CASES[@]}"; do
    case_path="${NPU_CASE_DIR}/${case_name%.py}.py"
    if [[ ! -f "${case_path}" ]]; then
      echo "ERROR: NPU E2E case not found: ${case_name}" >&2
      return 1
    fi
    printf '%s\n' "${case_path}"
  done
}

run_cases() {
  local cases
  cases=$(collect_cases) || return 1
  if [[ -z "${cases}" ]]; then
    echo "ERROR: no NPU E2E cases found." >&2
    return 1
  fi

  rm -rf "${TMPDIR:-/tmp}/.npu_kernels_${USER}"/autofused_*

  local failed_cases=()
  local npu_case
  while IFS= read -r npu_case; do
    echo "=========================================================="
    echo "Running NPU E2E: $(basename "${npu_case}")"
    echo "=========================================================="
    if ! python3 "${npu_case}" --mode "${MODE}" --steps "${STEPS}" --device "${DEVICE}"; then
      failed_cases+=("$(basename "${npu_case}")")
    fi
  done <<< "${cases}"

  if ((${#failed_cases[@]} > 0)); then
    echo "Failed NPU E2E cases:"
    printf '  - %s\n' "${failed_cases[@]}"
    return 1
  fi

  echo "All NPU E2E cases passed."
}

main() {
  cd "${BASEPATH}"
  checkopts "$@"
  export AUTOFUSE_NPU_INSTALL_DIR="${INSTALL_DIR}"
  configure_build
  build_pyautofuse
  install_autofuse_runtime
  verify_python_runtime
  run_cases
}

main "$@"
