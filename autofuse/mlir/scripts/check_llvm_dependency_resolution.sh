#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
mlir_root="${repo_root}/autofuse/mlir"
tmp_root="${TMPDIR:-/tmp}"
tmp_dir="$(mktemp -d "${tmp_root%/}/check-llvm-deps.XXXXXX")"

cleanup() {
  rm -rf -- "${tmp_dir}"
}
trap cleanup EXIT

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

assert_file_contains() {
  local file="$1"
  local pattern="$2"
  if ! grep -Fq -- "${pattern}" "${file}"; then
    echo "--- ${file} ---" >&2
    cat "${file}" >&2
    fail "expected '${file}' to contain '${pattern}'"
  fi
}

fake_llvm="${tmp_dir}/fake-llvm"
mkdir -p "${fake_llvm}/bin" \
         "${fake_llvm}/include" \
         "${fake_llvm}/lib/cmake/llvm" \
         "${fake_llvm}/lib/cmake/mlir"

cat > "${fake_llvm}/bin/llvm-config" <<'EOF'
#!/bin/bash
set -euo pipefail
case "${1:-}" in
  --version)
    echo "21.0.0git"
    ;;
  --bindir)
    cd "$(dirname "$0")"
    pwd
    ;;
  --libdir)
    cd "$(dirname "$0")/../lib"
    pwd
    ;;
  *)
    echo "unsupported llvm-config argument: ${1:-}" >&2
    exit 2
    ;;
esac
EOF
chmod +x "${fake_llvm}/bin/llvm-config"

cat > "${fake_llvm}/lib/cmake/mlir/MLIRConfig.cmake" <<'EOF'
set(MLIR_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(MLIR_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/../../../include")
set(LLVM_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/../../../include")
set(LLVM_DEFINITIONS "-DAF_FAKE_LLVM")
EOF
touch "${fake_llvm}/lib/cmake/llvm/LLVMConfig.cmake"
touch "${fake_llvm}/lib/cmake/mlir/TableGen.cmake"
touch "${fake_llvm}/lib/cmake/mlir/AddLLVM.cmake"
touch "${fake_llvm}/lib/cmake/mlir/AddMLIR.cmake"
touch "${fake_llvm}/lib/cmake/mlir/HandleLLVMOptions.cmake"

# A toolchain supplied through the environment must not become a stale cache
# entry. Reconfigure the same build directory with a different root and verify
# that the new package paths are selected.
fake_llvm_reconfigured="${tmp_dir}/fake-llvm-reconfigured"
cp -a "${fake_llvm}" "${fake_llvm_reconfigured}"

cmake_reconfigure_src="${tmp_dir}/cmake-reconfigure"
mkdir -p "${cmake_reconfigure_src}"
cat > "${cmake_reconfigure_src}/CMakeLists.txt" <<EOF
cmake_minimum_required(VERSION 3.16)
project(check_mlir_deps_reconfigure NONE)
set(ENABLE_AUTOFUSE_MLIR ON)
include("${mlir_root}/cmake/AutofuseMlirDeps.cmake")
file(WRITE "\${CMAKE_BINARY_DIR}/resolved-package-paths.txt" "\${MLIR_DIR}\n\${LLVM_DIR}\n")
EOF
cmake_reconfigure_build="${tmp_dir}/cmake-reconfigure-build"
cmake -S "${cmake_reconfigure_src}" -B "${cmake_reconfigure_build}" \
  -DLLVM_BUILD_DIR="${fake_llvm}" \
  -DMLIR_DIR="${fake_llvm}/lib/cmake/mlir" \
  -DLLVM_DIR="${fake_llvm}/lib/cmake/llvm" >/dev/null
cmake -S "${cmake_reconfigure_src}" -B "${cmake_reconfigure_build}" \
  -DLLVM_BUILD_DIR="${fake_llvm_reconfigured}" >/dev/null
assert_file_contains "${cmake_reconfigure_build}/resolved-package-paths.txt" "${fake_llvm_reconfigured}/lib/cmake/mlir"
assert_file_contains "${cmake_reconfigure_build}/resolved-package-paths.txt" "${fake_llvm_reconfigured}/lib/cmake/llvm"

cmake_off_src="${tmp_dir}/cmake-off"
mkdir -p "${cmake_off_src}"
cat > "${cmake_off_src}/CMakeLists.txt" <<EOF
cmake_minimum_required(VERSION 3.16)
project(check_mlir_deps_off NONE)
set(ENABLE_AUTOFUSE_MLIR OFF)
include("${mlir_root}/cmake/AutofuseMlirDeps.cmake")
EOF
cmake -S "${cmake_off_src}" -B "${tmp_dir}/cmake-off-build" >/dev/null

cmake_missing_src="${tmp_dir}/cmake-missing"
mkdir -p "${cmake_missing_src}"
cat > "${cmake_missing_src}/CMakeLists.txt" <<EOF
cmake_minimum_required(VERSION 3.16)
project(check_mlir_deps_missing NONE)
set(ENABLE_AUTOFUSE_MLIR ON)
include("${mlir_root}/cmake/AutofuseMlirDeps.cmake")
EOF
cmake_missing_log="${tmp_dir}/cmake-missing.log"
if cmake -S "${cmake_missing_src}" -B "${tmp_dir}/cmake-missing-build" \
    -DLLVM_BUILD_DIR="${tmp_dir}/missing-llvm" \
    -DMLIR_DIR= \
    -DLLVM_DIR= >"${cmake_missing_log}" 2>&1; then
  fail "CMake should fail when ENABLE_AUTOFUSE_MLIR=ON and MLIRConfig.cmake is missing"
fi
assert_file_contains "${cmake_missing_log}" "ENABLE_AUTOFUSE_MLIR=ON requires MLIRConfig.cmake"

cmake_valid_src="${tmp_dir}/cmake-valid"
mkdir -p "${cmake_valid_src}"
cat > "${cmake_valid_src}/CMakeLists.txt" <<EOF
cmake_minimum_required(VERSION 3.16)
project(check_mlir_deps_valid CXX)
set(ENABLE_AUTOFUSE_MLIR ON)
include("${mlir_root}/cmake/AutofuseMlirDeps.cmake")
EOF
cmake -S "${cmake_valid_src}" -B "${tmp_dir}/cmake-valid-build" \
  -DLLVM_BUILD_DIR="${fake_llvm}" \
  -DMLIR_DIR= \
  -DLLVM_DIR= >/dev/null

echo "LLVM dependency resolution check passed."
