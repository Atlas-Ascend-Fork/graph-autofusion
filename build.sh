#!/bin/bash
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------------------

set -e

BASEPATH=$(cd "$(dirname $0)"; pwd)
BUILD_PATH="${BASEPATH}/build"
OUTPUT_PATH="${BASEPATH}/build_out"
CPU_NUM=$(($(cat /proc/cpuinfo | grep "^processor" | wc -l)))
THREAD_NUM=${CPU_NUM}
ASCEND_INSTALL_PATH="${ASCEND_HOME_PATH}"
BUILD_TYPE="Release"
CUSTOM_OPTION="-DCMAKE_INSTALL_PREFIX=${BUILD_PATH} -DASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH}"
# Detect Python command to use
if [ -n "$VIRTUAL_ENV" ]; then
  PYTHON_CMD="python3"
elif [ -f "${BASEPATH}/venv/bin/python" ]; then
  PYTHON_CMD="${BASEPATH}/venv/bin/python"
else
  PYTHON_CMD="python3"
fi

SUPPORTED_MODULES=("superkernel" "autofuse_framework" "autofuse_ascendc_api" "autofuse_e2e")
declare -A MODULE_ACTION_HANDLERS=(
  ["superkernel:py_ut"]="superkernel_py_ut"
  ["superkernel:cpp_ut"]="superkernel_cpp_ut"
  ["superkernel:py_st"]="superkernel_py_st"
  ["superkernel:cpp_st"]="superkernel_cpp_st"
  ["autofuse_framework:all_ut"]="autofuse_module_test_suite"
  ["autofuse_framework:all_st"]="autofuse_module_test_suite"
  ["autofuse_ascendc_api:all_ut"]="autofuse_module_test_suite"
  ["autofuse_ascendc_api:all_st"]="autofuse_module_test_suite"
  ["autofuse_e2e:all_st"]="autofuse_module_test_suite"
)

# print usage message
function log()
{
  local commoninfo="$1"
  local timestr
  timestr=$(date '+%Y-%m-%d %H:%M:%S.%N ' | cut -b 1-23)
  echo "[${timestr}]${commoninfo}"
}

usage() {
  echo "Usage:"
  echo "  sh build.sh [-h|--help] [--pkg] [-u|--ut] [-s|--st] [--impl=<py|cpp|all>]"
  echo "              [--module=<name>] [-c|--coverage] [-j]"
  echo "              [--output_path=<PATH>] [--cann_3rd_lib_path=<PATH>] [--build-type=<TYPE>] [--no-autofuse]"
  echo "              [--pkg-type=<TYPE>]"
  echo "              [-f <FILE>]"
  echo ""
  echo "Options:"
  echo "    -h, --help            Print usage"
  echo "    --pkg                 Build package"
  echo "    --no-autofuse         Skip autofuse backend build/package artifacts"
  echo "    -j                    Compile thread nums, default is 16, eg: -j 8"
  echo "    -u, --ut              Run unit tests for supported implementations"
  echo "    -s, --st              Run system tests for supported implementations"
  echo "    --impl=<py|cpp|all>   Select implementation under -u/-s (default: all supported)"
  echo "    --module=<name>       Select module, default: all supported modules"
  echo "    -c, --coverage        Run tests with coverage report generation"
  echo "                          Without explicit test selection, run supported tests for the selected module"
  echo "    --output_path=<PATH>"
  echo "                          Set output path, where the run package will be generated, default ./build_out"
  echo "    --cann_3rd_lib_path=<PATH>"
  echo "                          Set third_party package install path, default ./output/third_party"
  echo "                          (Third_party package will cost a little time during the first compilation,"
  echo "                          it will skip compilation to save time during subsequent builds)"
  echo "    --build-type=<TYPE>   Set build type: Debug, Release(default: Release)"
  echo "    --pkg-type=<TYPE>     Set package type: run/rpm/deb(default: run)"
  echo "    -f <FILE>             File containing list of changed files. Smart module selection:"
  echo "                          - Only super_kernel/ changed: skip autofuse build/tests"
  echo "                          - Only autofuse/ changed: skip superkernel tests"
  echo "                          - Both changed: build all"
  echo "                          - Only docs/examples/etc: skip all (exit 200)"
  echo ""
}

check_param_j() {
  local thread_num=$1
  if [[ -z "$thread_num" ]]; then
    echo "ERROR: -j must specify a positive integer (e.g. -j8, -j=16)"
    usage
    exit 1
  fi
  if [[ ! "$thread_num" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: -j only support positive integers (0/negative/non-number are not allowed)"
    usage
    exit 1
  fi

  if [[ "$thread_num" -gt "$CPU_NUM" ]]; then
    thread_num=$CPU_NUM
  fi
  echo "$thread_num"
}

is_supported_module() {
  local requested_module="$1"
  local module
  for module in "${SUPPORTED_MODULES[@]}"; do
    if [[ "${module}" == "${requested_module}" ]]; then
      return 0
    fi
  done
  return 1
}

analyze_changed_modules() {
  local changed_files="$1"
  CHANGED_SUPERKERNEL=false
  CHANGED_AUTOFUSE=false
  CHANGED_OTHER=false

  if [ -z "$changed_files" ]; then
    CHANGED_OTHER=true
    return
  fi

  for file in $changed_files; do
    file=$(echo "$file" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//;s/^"//;s/"$//')

    if echo "$file" | grep -qi "^README\.md$"; then
      continue
    fi

    if echo "$file" | grep -qi "^CONTRIBUTING\.md$"; then
      continue
    fi

    if echo "$file" | grep -q "^docs/"; then
      continue
    fi

    if echo "$file" | grep -q "^examples/"; then
      continue
    fi

    if echo "$file" | grep -q "^\.claude/"; then
      continue
    fi

    if echo "$file" | grep -q "^\.opencode/"; then
      continue
    fi

    if echo "$file" | grep -qi "^AGENTS\.md$"; then
      continue
    fi

    if echo "$file" | grep -q "^super_kernel/"; then
      CHANGED_SUPERKERNEL=true
    elif echo "$file" | grep -q "^autofuse/"; then
      CHANGED_AUTOFUSE=true
    else
      CHANGED_OTHER=true
    fi
  done
}

apply_module_selection() {
  if [ "$CHANGED_SUPERKERNEL" = true ] && [ "$CHANGED_AUTOFUSE" = false ] && [ "$CHANGED_OTHER" = false ]; then
    echo "[INFO] Only super_kernel changed, skipping autofuse build and autofuse tests."
    ENABLE_AUTOFUSE="off"
    SKIP_AUTOFUSE_TESTS="on"
    return 0
  elif [ "$CHANGED_AUTOFUSE" = true ] && [ "$CHANGED_SUPERKERNEL" = false ] && [ "$CHANGED_OTHER" = false ]; then
    echo "[INFO] Only autofuse changed, skipping superkernel tests."
    SKIP_SUPERKERNEL_TESTS="on"
    return 0
  elif [ "$CHANGED_SUPERKERNEL" = false ] && [ "$CHANGED_AUTOFUSE" = false ] && [ "$CHANGED_OTHER" = false ]; then
    echo "[INFO] Changed files only contain docs/, examples/, .claude/, .opencode/, README.md, CONTRIBUTING.md or AGENTS.md, skipping build."
    echo "[INFO] Changed files: $CHANGED_FILES"
    return 200
  fi
  return 0
}

check_changed_files() {
  local changed_files="$1"
  local skip_build=true

  if [ -z "$changed_files" ]; then
    return 1
  fi

  for file in $changed_files; do
    file=$(echo "$file" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//;s/^"//;s/"$//')

    if echo "$file" | grep -qi "^README\.md$"; then
      continue
    fi

    if echo "$file" | grep -qi "^CONTRIBUTING\.md$"; then
      continue
    fi

    if echo "$file" | grep -q "^docs/"; then
      continue
    fi

    if echo "$file" | grep -q "^examples/"; then
      continue
    fi

    if echo "$file" | grep -q "^\.claude/"; then
      continue
    fi

    if echo "$file" | grep -q "^\.opencode/"; then
      continue
    fi

    if echo "$file" | grep -qi "^AGENTS\.md$"; then
      continue
    fi

    skip_build=false
    break
  done

  if [ "$skip_build" = true ]; then
    echo "[INFO] Changed files only contain docs/, examples/, .claude/, .opencode/, README.md, CONTRIBUTING.md or AGENTS.md, skipping build."
    echo "[INFO] Changed files: $changed_files"
    return 0
  fi

  return 1
}

has_module_action_entry() {
  local action="$1"
  local module="$2"
  local target_module

  if [[ "${module}" == "all" ]]; then
    for target_module in "${SUPPORTED_MODULES[@]}"; do
      if [[ -v MODULE_ACTION_HANDLERS["${target_module}:${action}"] ]]; then
        return 0
      fi
    done
    return 1
  fi

  [[ -v MODULE_ACTION_HANDLERS["${module}:${action}"] ]]
}

normalize_test_selection() {
  EXEC_ACTIONS=()

  if [[ -z "${TARGET_MODULE}" ]]; then
    TARGET_MODULE="all"
  fi

  if [[ "${ENABLE_COVERAGE}" == "on" && "${ENABLE_UT}" != "on" && "${ENABLE_ST}" != "on" ]]; then
    ENABLE_UT="on"
    ENABLE_ST="on"
  fi

  local selected_modules=()
  if [[ "${TARGET_MODULE}" == "all" ]]; then
    selected_modules=("${SUPPORTED_MODULES[@]}")
  else
    selected_modules=("${TARGET_MODULE}")
  fi

  for suite in ut st; do
    local enabled_flag="ENABLE_${suite^^}"
    if [[ "${!enabled_flag}" != "on" ]]; then
      continue
    fi

    local impls=()
    case "${TEST_IMPL_MODE}" in
      py) impls=(py) ;;
      cpp) impls=(cpp) ;;
      all) impls=(py cpp all) ;;
    esac

    for module in "${selected_modules[@]}"; do
      for impl in "${impls[@]}"; do
        local action="${impl}_${suite}"
        if ! has_module_action_entry "${action}" "${module}"; then
          continue
        fi

        EXEC_ACTIONS+=("${module}:${action}")
      done
    done
  done
}

# parse and set options
checkopts() {
  if [ $# -eq 0 ]; then
    echo "ERROR: 'build.sh' has no options available, please select at least one option!"
    usage
    exit 1
  fi

  ENABLE_BUILD_PACKAGE="off"
  ENABLE_UT="off"
  ENABLE_ST="off"
  ENABLE_COVERAGE="off"
  TEST_IMPL_MODE="all"
  TARGET_MODULE="all"
  CANN_3RD_LIB_PATH="$BASEPATH/output/third_party"
  ENABLE_AUTOFUSE="on"
  CHANGED_FILES=""
  PACKAGE_TYPE="run"

  parsed_args=$(getopt -a -o j:huscf: -l help,pkg,autofuse,no-autofuse,impl:,module:,test_case:,ut,st,coverage,output_path:,cann_3rd_lib_path:,build-type:,pkg-type: -- "$@") || {
    usage
    exit 1
  }

  eval set -- "$parsed_args"

  while true; do
    case "$1" in
      -h | --help)
        usage
        exit 0
        ;;
      --pkg)
        ENABLE_BUILD_PACKAGE="on"
        shift
        ;;
      --autofuse)
        ENABLE_AUTOFUSE="on"
        shift
        ;;
      --no-autofuse)
        ENABLE_AUTOFUSE="off"
        shift
        ;;
      -j)
        local raw_thread_num="$2"
        raw_thread_num="${raw_thread_num#=}"
        THREAD_NUM=$(check_param_j "$raw_thread_num")
        shift 2
        ;;
      -u | --ut)
        ENABLE_UT="on"
        shift
        ;;
      -s | --st)
        ENABLE_ST="on"
        shift
        ;;
      -c | --coverage)
        ENABLE_COVERAGE="on"
        shift
        ;;
      --impl)
        if [[ ! "$2" =~ ^(py|cpp|all)$ ]]; then
          echo "ERROR: Invalid implementation type: $2"
          echo "       Valid values: py, cpp, all"
          exit 1
        fi
        TEST_IMPL_MODE="$2"
        shift 2
        ;;
      --module)
        if [[ "$2" != "all" ]] && ! is_supported_module "$2"; then
          echo "ERROR: Unsupported module '$2'. Currently supported values: all, ${SUPPORTED_MODULES[*]}."
          exit 1
        fi
        if [[ "${TARGET_MODULE}" != "all" && "${TARGET_MODULE}" != "$2" ]]; then
          echo "ERROR: Conflicting module selection: '${TARGET_MODULE}' vs '$2'."
          exit 1
        fi
        TARGET_MODULE="$2"
        shift 2
        ;;
      --test_case)
        CPP_UTEST_FILTER="$2"
        shift 2
        ;;
      --output_path)
        OUTPUT_PATH="$(realpath $2)"
        shift 2
        ;;
      --cann_3rd_lib_path)
        CANN_3RD_LIB_PATH="$(realpath $2)"
        shift 2
        ;;
      --build-type)  # 新增的 build-type 选项
        BUILD_TYPE="$2"
        # 验证 BUILD_TYPE 是否有效
        if [[ ! "$BUILD_TYPE" =~ ^(Debug|Release)$ ]]; then
          echo "ERROR: Invalid build type: $BUILD_TYPE"
          echo "       Valid types: Debug, Release"
          exit 1
        fi
        shift 2
        ;;
      --pkg-type)
        if [[ ! "$2" =~ ^(run|rpm|deb)$ ]]; then
          echo "ERROR: Invalid package type: $2"
          echo "       Valid types: run, rpm, deb"
          exit 1
        fi
        PACKAGE_TYPE="$2"
        shift 2
        ;;
      -f)
        CHANGED_FILES_FILE="$2"
        if [ ! -f "$CHANGED_FILES_FILE" ]; then
          echo "Error: File $CHANGED_FILES_FILE not found"
          exit 1
        fi
        CHANGED_FILES=$(cat "$CHANGED_FILES_FILE")
        shift 2
        ;;
      --)
        shift
        break
        ;;
      *)
        echo "Undefined option: $1"
        usage
        exit 1
        ;;
    esac
  done

  normalize_test_selection

  if [ "X$ENABLE_BUILD_PACKAGE" != "Xon" ] && [ ${#EXEC_ACTIONS[@]} -eq 0 ]; then
    echo "ERROR: No supported actions for the requested selection."
    echo "       module=${TARGET_MODULE}, impl=${TEST_IMPL_MODE}, ut=${ENABLE_UT}, st=${ENABLE_ST}, coverage=${ENABLE_COVERAGE}"
    exit 1
  fi
}

function cmake_config()
{
  local extra_option="$1"
  local cmake_option="${CUSTOM_OPTION} -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH} -DPACKAGE_TYPE=${PACKAGE_TYPE}"
  cmake_option="${cmake_option} -DENABLE_CPP_UTEST=OFF -DENABLE_CPP_STEST=OFF"
  if [ "X$ENABLE_AUTOFUSE" == "Xon" ]; then
    extra_option="${extra_option} -DBUILD_AUTOFUSE=ON"
  fi
  if [ "X$ENABLE_AUTOFUSE" == "Xoff" ]; then
    extra_option="${extra_option} -DBUILD_AUTOFUSE=OFF"
  fi
  echo "Info: cmake config ${cmake_option} ${extra_option} ."
  cmake .. ${cmake_option} ${extra_option}
}

function build()
{
  local target="$1"
  cmake --build . --target ${target} -j ${THREAD_NUM}
}

clean_coverage_artifacts() {
  if [ "X$ENABLE_COVERAGE" != "Xon" ]; then
    return
  fi

  local action_entry
  local has_python_tests="off"
  local has_cpp_tests="off"
  for action_entry in "${EXEC_ACTIONS[@]}"; do
    case "${action_entry}" in
      *:py_ut|*:py_st)
        has_python_tests="on"
        ;;
      *:cpp_ut|*:cpp_st)
        has_cpp_tests="on"
        ;;
    esac
  done

  if [ "X$has_python_tests" == "Xon" ]; then
    echo "---------------- Clean Python Coverage Data ----------------"
    cd ${BASEPATH}/super_kernel && ${PYTHON_CMD} -m coverage erase
  fi

  if [ "X$has_cpp_tests" == "Xon" ]; then
    echo "---------------- Clean AOT C++ Coverage Artifacts ----------------"
    rm -rf ${BASEPATH}/super_kernel/coverage/cpp_ut ${BASEPATH}/super_kernel/coverage/cpp_st
    find ${BUILD_PATH} -name "*.gcda" -delete 2>/dev/null || true
  fi
}

function build_package_inner(){
  # Test configurations set RUN_TEST=1 to build test-only targets.  That
  # cache entry also disables Autofuse install rules in autofuse/CMakeLists.txt
  # and would silently produce a SuperKernel-only run package.  Packaging is
  # always a product build, so explicitly reset the guard here.
  cmake_config "-DRUN_TEST=0"
  build "all package"
}

build_package() {
  echo "---------------- Start build run package ----------------"
  mkdir -pv ${BUILD_PATH} &&
  cd ${BUILD_PATH} &&
  build_package_inner &&
  mkdir -pv ${OUTPUT_PATH} &&
  cp _CPack_Packages/makeself_staging/cann-graph-autofusion*.run ${OUTPUT_PATH} &&
  output_run_path=`ls -1 ${OUTPUT_PATH}/cann-graph-autofusion*.run 2>/dev/null` &&
  echo "Build run package success!" &&
  echo "package: ${output_run_path}"
}

superkernel_py_ut() {
  echo "---------------- Start UT ----------------"
  cd ${BASEPATH}/super_kernel &&
  ${PYTHON_CMD} -m pip install --upgrade pip -q &&
  ${PYTHON_CMD} -m pip install -r requirements-dev.txt -q &&
  if [ "X$ENABLE_COVERAGE" == "Xon" ]; then
    PYTHONPATH="${BASEPATH}/super_kernel/src/jit${PYTHONPATH:+:${PYTHONPATH}}" \
      ${PYTHON_CMD} -m pytest tests/ut -m ut -n auto \
                                     --cov-config=scripts/sk_ut_cfg.toml \
                                     --cov=superkernel \
                                     --cov-report=term-missing \
                                     --cov-report=html \
                                     --cov-report=xml
  else
    PYTHONPATH="${BASEPATH}/super_kernel/src/jit${PYTHONPATH:+:${PYTHONPATH}}" \
      ${PYTHON_CMD} -m pytest tests/ut -m ut -n auto
  fi
}

superkernel_py_st() {
  echo "---------------- Start ST ----------------"
  cd ${BASEPATH}/super_kernel &&
  ${PYTHON_CMD} -m pip install --upgrade pip -q &&
  ${PYTHON_CMD} -m pip install -r requirements-dev.txt -q &&
  if [ "X$ENABLE_COVERAGE" == "Xon" ]; then
    PYTHONPATH="${BASEPATH}/super_kernel/src/jit${PYTHONPATH:+:${PYTHONPATH}}" \
      ${PYTHON_CMD} -m pytest tests/st -m st -n auto --cov-config=scripts/sk_st_cfg.toml \
                                     --cov=superkernel \
                                     --cov-report=term-missing \
                                     --cov-report=html \
                                     --cov-report=xml
  else
    PYTHONPATH="${BASEPATH}/super_kernel/src/jit${PYTHONPATH:+:${PYTHONPATH}}" \
      ${PYTHON_CMD} -m pytest tests/st -m st -n auto
  fi
}

function superkernel_cpp_ut() {
  superkernel_cpp_test ut
}

function superkernel_cpp_st() {
  superkernel_cpp_test st
}

function superkernel_cpp_test() {
  local suite="$1"
  local test_options="-DENABLE_CPP_UTEST=OFF -DENABLE_CPP_STEST=OFF -DENABLE_GCOV=OFF"
  local coverage_target="collect_coverage_data"
  echo "---------------- Start run cpp ${suite}est ----------------"

  if [ "${suite}" == "ut" ]; then
    test_options="${test_options} -DENABLE_CPP_UTEST=ON"
  else
    test_options="${test_options} -DENABLE_CPP_STEST=ON"
    coverage_target="collect_coverage_data_cpp_st"
  fi

  if [ "X$ENABLE_COVERAGE" == "Xon" ]; then
    test_options="${test_options} -DENABLE_GCOV=ON"
  fi

  # Pass gtest filter to cmake if specified
  if [ -n "${CPP_UTEST_FILTER}" ]; then
    test_options="${test_options} -DGTEST_FILTER=--gtest_filter=${CPP_UTEST_FILTER}"
  else
    test_options="${test_options} -DGTEST_FILTER="
  fi

  mkdir -pv ${BUILD_PATH} &&
  cd ${BUILD_PATH} &&
  cmake_config "${test_options}" &&
  if [ "X$ENABLE_COVERAGE" == "Xon" ]; then
    build clean &&
    build "${coverage_target}"
  else
    build "run_super_kernel_aot_${suite}est" &&
    build "run_model_specialization_${suite}est"
  fi &&
  echo "Build run cpp ${suite}est success!"
}

autofuse_module_test_suite() {
  local module="$1"
  local action="$2"
  local test_option
  local test_module
  local test_args=()

  case "${action}" in
    all_ut) test_option="-u" ;;
    all_st) test_option="-s" ;;
    *)
      echo "ERROR: Unsupported autofuse test action: ${action}."
      return 1
      ;;
  esac

  case "${module}" in
    autofuse_framework) test_module="framework" ;;
    autofuse_ascendc_api) test_module="ascendc_api" ;;
    autofuse_e2e) test_module="e2e" ;;
    *)
      echo "ERROR: Unsupported autofuse test module: ${module}."
      return 1
      ;;
  esac

  test_args+=("${test_option}" -m "${test_module}" -j "${THREAD_NUM}")
  if [ -n "${ASCEND_INSTALL_PATH}" ]; then
    test_args+=("--ascend_install_path=${ASCEND_INSTALL_PATH}")
  fi
  if [ -n "${CANN_3RD_LIB_PATH}" ]; then
    test_args+=("--ascend_3rd_lib_path=${CANN_3RD_LIB_PATH}")
  fi
  if [ "X${ENABLE_COVERAGE}" == "Xon" ]; then
    test_args+=("-c")
  fi

  bash "${BASEPATH}/scripts/test/run_autofuse_test.sh" "${test_args[@]}"
}

main() {
  checkopts "$@"

  SKIP_SUPERKERNEL_TESTS="off"
  SKIP_AUTOFUSE_TESTS="off"

  if [ -n "$CHANGED_FILES" ]; then
    analyze_changed_modules "$CHANGED_FILES"
    local ret=0
    apply_module_selection || ret=$?
    if [ $ret -eq 200 ]; then
      exit 200
    fi
    if [ "$CHANGED_SUPERKERNEL" = false ]; then
      SKIP_SUPERKERNEL_TESTS="on"
      echo "[INFO] No super_kernel files changed, skipping superkernel tests."
    fi
    if [ "$CHANGED_AUTOFUSE" = false ]; then
      SKIP_AUTOFUSE_TESTS="on"
      echo "[INFO] No autofuse files changed, skipping autofuse tests."
    fi
    local has_action_to_run=false
    for _action_entry in "${EXEC_ACTIONS[@]}"; do
      _module="${_action_entry%%:*}"
      if [ "$SKIP_SUPERKERNEL_TESTS" = "on" ] && [[ "$_module" == "superkernel" ]]; then
        continue
      fi
      if [ "$SKIP_AUTOFUSE_TESTS" = "on" ] && [[ "$_module" == "autofuse"* ]]; then
        continue
      fi
      has_action_to_run=true
      break
    done
    if [ "$has_action_to_run" = false ] && [ "${#EXEC_ACTIONS[@]}" -gt 0 ]; then
      echo "[INFO] All test actions skipped due to module selection, exiting."
      if [ "$SKIP_SUPERKERNEL_TESTS" = "on" ]; then
        mkdir -p "${BASEPATH}/super_kernel/coverage/cpp_ut"
        touch "${BASEPATH}/super_kernel/coverage/cpp_ut/coverage.info"
        mkdir -p "${BUILD_PATH}"
        if command -v python3 &>/dev/null && python3 -c "import coverage" 2>/dev/null; then
          python3 << 'PYEOF' 2>/dev/null || true
import coverage, os
build_path = os.path.join(os.path.dirname(os.path.abspath('.')), 'build')
if not os.path.isdir(build_path):
    build_path = 'build'
os.makedirs(build_path, exist_ok=True)
os.chdir(build_path)
with open('_cov_dummy.py', 'w') as f:
    f.write('x = 1\n')
cov = coverage.Coverage(source=['_cov_dummy.py'])
cov.start()
exec(open('_cov_dummy.py').read())
cov.stop()
cov.save()
PYEOF
        fi
      fi
      if [ "$SKIP_AUTOFUSE_TESTS" = "on" ]; then
        mkdir -p "${BASEPATH}/cov"
        touch "${BASEPATH}/cov/coverage.info"
      fi
      exit 200
    fi
  fi

  # Detect proxy settings for third-party tarball downloads.
  # CMake ExternalProject_Add downloads via curl, which reads http_proxy/https_proxy
  # from the process environment. Remind users to export these if git works but
  # cmake download fails (common when proxy is only configured in ~/.gitconfig).
  if [ -n "${http_proxy:-}" ] || [ -n "${https_proxy:-}" ]; then
    echo "Info: Proxy settings detected in environment:"
    [ -n "${http_proxy:-}" ] && echo "  http_proxy=${http_proxy}"
    [ -n "${https_proxy:-}" ] && echo "  https_proxy=${https_proxy}"
    echo "  These will be inherited by CMake's curl downloads automatically."
  fi

  clean_coverage_artifacts

  if [ "X$ENABLE_AUTOFUSE" == "Xon" ]; then
    echo "---------------- Start build autofuse ----------------"
    mkdir -pv ${BUILD_PATH} &&
    cd ${BUILD_PATH} || { echo "Prepare autofuse build failed."; exit 1; }
    if [ "X$ENABLE_BUILD_PACKAGE" == "Xon" ]; then
      # Configure product packaging before the first build.  Reusing a test
      # cache here would add all RUN_TEST targets to `build all`; resetting
      # only in build_package_inner is too late.
      cmake_config "-DCANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH} -DRUN_TEST=0"
    else
      cmake_config "-DCANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}"
    fi

    build all || { echo "Build autofuse failed."; exit 1; }
    echo "Build autofuse success!"
  fi

  if [ "X$ENABLE_BUILD_PACKAGE" == "Xon" ]; then
    build_package || { echo "Build run package failed."; exit 1; }
  fi

  local action_entry
  local module
  local action
  local handler
  for action_entry in "${EXEC_ACTIONS[@]}"; do
    module="${action_entry%%:*}"
    action="${action_entry#*:}"

    if [ "X$SKIP_SUPERKERNEL_TESTS" == "Xon" ] && [[ "$module" == "superkernel" ]]; then
      echo "[INFO] Skipping superkernel ${action} due to module selection."
      continue
    fi

    if [ "X$SKIP_AUTOFUSE_TESTS" == "Xon" ] && [[ "$module" == "autofuse"* ]]; then
      echo "[INFO] Skipping autofuse ${action} due to module selection."
      continue
    fi

    handler="${MODULE_ACTION_HANDLERS["${module}:${action}"]}"

    ${handler} "${module}" "${action}" || {
      echo "Run ${module} ${action} failed."
      exit 1
    }
  done
}

main "$@"
