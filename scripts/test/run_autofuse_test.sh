#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -e

BASEPATH=$(cd "$(dirname $0)"; pwd)/../../
AUTOFUSE_PATH="${BASEPATH}/autofuse"
BUILD_RELATIVE_PATH="build"
BUILD_PATH="${BASEPATH}/${BUILD_RELATIVE_PATH}/"
AUTOFUSE_BUILD_PATH="${BUILD_PATH}/autofuse"
OUTPUT_PATH="${BASEPATH}/output"
METADEF_LIB_PATH=${OUTPUT_PATH}/metadef/lib/
PYTHON_LIB_PATH=${AUTOFUSE_BUILD_PATH}/tests/
PYTHON_MODULE_PATH=${AUTOFUSE_PATH}/compiler/python/
TESTS_ST_PATH="${AUTOFUSE_PATH}/tests/st/"
LOCAL_RUNTIME_LIB_PATH="${AUTOFUSE_BUILD_PATH}/ascir/generator:${AUTOFUSE_BUILD_PATH}/ascir/meta:${AUTOFUSE_BUILD_PATH}/graph_metadef/graph/ascendc_ir/generator:${AUTOFUSE_BUILD_PATH}/graph_metadef/graph/ascendc_ir:${AUTOFUSE_BUILD_PATH}/graph_metadef/graph/expression:${AUTOFUSE_BUILD_PATH}/graph_metadef/graph:${AUTOFUSE_BUILD_PATH}/tests:${AUTOFUSE_BUILD_PATH}/tests/depends/trace:${AUTOFUSE_BUILD_PATH}/tests/depends/runtime"
RUN_V35_TESTS="off"

select_compilers() {
  local selected_cc="${CC:-}"
  local selected_cxx="${CXX:-}"
  local gcc_version="${GCC_VERSION:-}"

  if [[ -n "${selected_cc}" && -z "${selected_cxx}" ]]; then
    if [[ "${selected_cc}" =~ ^gcc-([0-9]+)$ ]]; then
      selected_cxx="g++-${BASH_REMATCH[1]}"
    fi
  fi

  if [[ -n "${selected_cxx}" && -z "${selected_cc}" ]]; then
    if [[ "${selected_cxx}" =~ ^g\+\+-([0-9]+)$ ]]; then
      selected_cc="gcc-${BASH_REMATCH[1]}"
    fi
  fi

  if [[ -z "${selected_cc}" || -z "${selected_cxx}" ]]; then
    if [[ -n "${gcc_version}" ]]; then
      selected_cc="${selected_cc:-gcc-${gcc_version}}"
      selected_cxx="${selected_cxx:-g++-${gcc_version}}"
    elif [[ "${task_name}" == *ubuntu24* ]]; then
      selected_cc="${selected_cc:-gcc-14}"
      selected_cxx="${selected_cxx:-g++-14}"
    else
      selected_cc="${selected_cc:-gcc}"
      selected_cxx="${selected_cxx:-g++}"
    fi
  fi

  export CC="${selected_cc}"
  export CXX="${selected_cxx}"
  echo "Using CC=${CC}"
  echo "Using CXX=${CXX}"
}

select_compilers

C_COMPILER="${CC}"
CXX_COMPILER="${CXX}"

source ${BASEPATH}/scripts/support_multiple_versions_of_lcov.sh

# TODO(For autofuse): Remove 'export DISABLE_COMPILATION_WERROR=ON' and fix the related compilation errors.
export DISABLE_COMPILATION_WERROR=ON

set_test_ld_library_path() {
  local autofuse_lib_path="${LOCAL_RUNTIME_LIB_PATH}:${METADEF_LIB_PATH}:${ASCEND_INSTALL_LIB_PATH}"
  if [ -n "${INITIAL_LD_LIBRARY_PATH}" ]; then
    export LD_LIBRARY_PATH="${autofuse_lib_path}:${INITIAL_LD_LIBRARY_PATH}"
  else
    export LD_LIBRARY_PATH="${autofuse_lib_path}"
  fi
}

# print usage message
usage() {
  echo "Usage:"
  echo "sh run_autofuse_test.sh [-s | --st] [-u | --ut] [-j<N>] [-c | --cov]"
  echo "               [--ascend_install_path=<PATH>]"
  echo "               [--ascend_3rd_lib_path=<PATH>]"
  echo ""
  echo "Options:"
  echo "    -s, --st       Build all st"
  echo "    -u, --ut       Build all ut"
  echo "    -m, --module    Option arg, specify the model name to build st or ut, default all"
  echo "    -j<N>            Set the number of threads used for llt, default 8."
  echo "    -c, --cov        Build ut with coverage tag"
  echo "                     Please ensure that the environment has correctly installed lcov, gcov, and genhtml."
  echo "                     and the version matched gcc/g++."
  echo "    -h, --help     Print usage"
  echo "    --ascend_install_path=<PATH>"
  echo "                   Set ascend package install path, default /usr/local/Ascend/ascend-toolkit/latest"
  echo "    --ascend_3rd_lib_path=<PATH>"
  echo "                     Set ascend third_party package install path, default ./output/third_party"
  echo ""
}

# parse and set options
checkopts() {
  ENABLE_UT="off"
  ENABLE_ST="off"
  ENABLE_COV="off"
  THREAD_NUM=8
  MODEL_NAME="all"
  if [ -n "$ASCEND_INSTALL_PATH" ]; then
    ASCEND_INSTALL_PATH="$ASCEND_INSTALL_PATH"
  else
    ASCEND_INSTALL_PATH="/usr/local/Ascend/ascend-toolkit/latest"
  fi
  if [ -n "$ASCEND_3RD_LIB_PATH" ]; then
    ASCEND_3RD_LIB_PATH="$ASCEND_3RD_LIB_PATH"
  else
    ASCEND_3RD_LIB_PATH="$BASEPATH/output/third_party"
  fi
  if [ -n "$ENABLE_PKG" ]; then
    ENABLE_PKG="$ENABLE_PKG"
  else
    ENABLE_PKG="off"
  fi
  if [ -d "${AUTOFUSE_PATH}/v35" ]; then
    RUN_V35_TESTS="on"
  fi

  # Process the options
  parsed_args=$(getopt -a -o hsucj:hvm: -l help,st,ut,cov,verbose,ascend_install_path:,ascend_3rd_lib_path:,module: -- "$@") || {
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
      -j)
        THREAD_NUM="$2"
        shift 2
        ;;
      --ascend_install_path)
        ASCEND_INSTALL_PATH="$(realpath $2)"
        shift 2
        ;;
      --ascend_3rd_lib_path)
        ASCEND_3RD_LIB_PATH="$(realpath $2)"
        shift 2
        ;;
      -m | --module)
        MODEL_NAME=$2
        shift 2
        ;;
      -s | --st)
        ENABLE_ST="on"
        shift
        ;;
      -u | --ut)
        ENABLE_UT="on"
        shift
        ;;
      -c | --cov)
        ENABLE_COV="on"
        shift
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
}

mk_dir() {
  local create_dir="$1"  # the target to make
  mkdir -pv "${create_dir}"
  echo "created ${create_dir}"
}

build_att_test() {
  echo "create build directory and build Att"
  ATT_PATH="${AUTOFUSE_PATH}/att"

  cd "${ATT_PATH}"
  ./build.sh --ascend_install_path=${ASCEND_INSTALL_PATH} --ascend_3rd_lib_path=${ASCEND_3RD_LIB_PATH}

  if [ 0 -ne $? ]; then
    echo "execute command: cmake && make failed."
    return 1
  fi
  echo "Att build success!"
}

build_ascgen-dev() {
  echo "create build directory and build ascgen-dev";
  mk_dir "${BUILD_PATH}"
  cd "${BUILD_PATH}"
  g++ -v

  ASCEND_INSTALL_LIB_PATH=${ASCEND_INSTALL_PATH}/$(uname -m)-linux/lib64/

  # Set test-related flags based on UT/ST mode
  if [[ "X$ENABLE_UT" = "Xon" ]]; then
    ENABLE_TEST_FLAG="True"
    ENABLE_GE_UT_FLAG="on"
  else
    ENABLE_TEST_FLAG=""
    ENABLE_GE_UT_FLAG=""
  fi

  if [[ "X$ENABLE_ST" = "Xon" ]]; then
    ENABLE_TEST_FLAG="True"
    ENABLE_GE_ST_FLAG="on"
  else
    ENABLE_GE_ST_FLAG=""
  fi

  CMAKE_ARGS="-D CMAKE_C_COMPILER=${C_COMPILER} \
              -D CMAKE_CXX_COMPILER=${CXX_COMPILER} \
              -D ASCEND_3RD_LIB_PATH=${ASCEND_3RD_LIB_PATH} \
              -D CANN_3RD_LIB_PATH=${ASCEND_3RD_LIB_PATH} \
              -D ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH} \
              -D CMAKE_INSTALL_PREFIX=${OUTPUT_PATH} \
              -D CMAKE_POLICY_VERSION_MINIMUM=3.5 \
              -D ASCEND_INSTALL_LIB_PATH=${ASCEND_INSTALL_LIB_PATH} \
              -D ENABLE_OPEN_SRC=True \
              -D ENABLE_SYMENGINE=True \
              -D RUN_TEST=1 \
              -D TESTS_ST_PATH=${TESTS_ST_PATH} \
              -D ENABLE_PKG=${ENABLE_PKG} \
              -D ENABLE_TEST=${ENABLE_TEST_FLAG} \
              -D ENABLE_GE_UT=${ENABLE_GE_UT_FLAG} \
              -D ENABLE_GE_ST=${ENABLE_GE_ST_FLAG}"

  echo "CMAKE_ARGS is: $CMAKE_ARGS"
  export BUILD_METADEF=${BUILD_METADEF}

  rm -f CMakeCache.txt
  cmake $CMAKE_ARGS ${BASEPATH}

  make -j${THREAD_NUM} pyautofuse
  if [ $? -ne 0 ]
  then
    env
    echo "execute command: make -j autofuse pyautofuse failed."
    return 1
  fi
  mk_dir "${METADEF_LIB_PATH}"
  echo "$(date '+%F %T') cp metadef libs from ${BUILD_PATH} to ${METADEF_LIB_PATH}"
}

build_test_ascendc_api_test() {
  echo "create build directory and build ascendc_api_test";
  cd "${BUILD_PATH}"
  CMAKE_ARGS="-D CMAKE_C_COMPILER=${C_COMPILER} \
              -D CMAKE_CXX_COMPILER=${CXX_COMPILER} \
              -D ASCEND_3RD_LIB_PATH=${ASCEND_3RD_LIB_PATH} \
              -D CANN_3RD_LIB_PATH=${ASCEND_3RD_LIB_PATH} \
              -D ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH} \
              -D CMAKE_INSTALL_PREFIX=${OUTPUT_PATH} \
              -D CMAKE_POLICY_VERSION_MINIMUM=3.5 \
              -D ASCEND_INSTALL_LIB_PATH=${ASCEND_INSTALL_LIB_PATH} \
              -D ENABLE_OPEN_SRC=True \
              -D ENABLE_SYMENGINE=True \
              -D BUILD_METADEF=ON \
              -D RUN_TEST=1 \
              -D ENABLE_PKG=${ENABLE_PKG} \
              -D ENABLE_TEST=True \
              -D ENABLE_GE_UT=on \
              -D ENABLE_LLT_PKG=ON"
  cmake $CMAKE_ARGS ..
  MAKE_TEST_TARGET="test_ascendc_api
                    test_ascendc_api_v35 \
                    test_load_broadcast_store_codegen \
                    test_load_broadcast_multi_axis_store_codegen"
  echo "[UT AUTOFUSE ASCENDC API] make -j${THREAD_NUM} start"
  make -j${THREAD_NUM} $MAKE_TEST_TARGET
  if [ $? -ne 0 ]; then
    env
    echo "[UT AUTOFUSE ASCENDC API] make failed."
    return 1
  fi
  echo "[UT AUTOFUSE ASCENDC API] make success!"
  ctest --output-on-failure -j${THREAD_NUM} --test-dir ${AUTOFUSE_BUILD_PATH}/tests/ut/ascendc --no-tests=error \
        -O ${BUILD_PATH}/ctest_test_ascendc_api.log
  if [ $? -ne 0 ]; then
    env
    echo "[UT AUTOFUSE ASCENDC API] test ascendc failed."
    return 1
  fi
  ctest --output-on-failure -j${THREAD_NUM} --test-dir ${AUTOFUSE_BUILD_PATH}/tests/v35/ut/ascendc --no-tests=error \
        -O ${BUILD_PATH}/ctest_test_ascendc_api_arch35.log
  if [ $? -ne 0 ]; then
    env
    echo "[UT AUTOFUSE ASCENDC API] test ascendc_v35 failed."
    return 1
  fi
  ctest --output-on-failure -j${THREAD_NUM} --test-dir ${AUTOFUSE_BUILD_PATH}/tests/ut/e2e --no-tests=error \
          -O ${BUILD_PATH}/ctest_ut_e2e.log
  if [ $? -ne 0 ]; then
    env
    echo "[UT AUTOFUSE ASCENDC API] test e2e failed."
    return 1
  fi
  echo "[UT AUTOFUSE ASCENDC API] test success!"
}

build_test() {
  echo "$(date '+%F %T') create build directory and build ascgen-dev";
  cd "${BUILD_PATH}"

  env

  MAKE_TEST_TARGET="test_main"
  make -j${THREAD_NUM} $MAKE_TEST_TARGET

  if [ $? -ne 0 ]
  then
    env
    echo "execute command: make -j test_main failed."
    return 1
  fi

  echo "$(date '+%F %T') run test_main success!"

  ctest --output-on-failure -j${THREAD_NUM} -R '^test_main_ut_aggregate$' \
        --test-dir ${AUTOFUSE_BUILD_PATH}/tests/ut --no-tests=error
  if [ $? -ne 0 ]
  then
    echo "execute command: test_main  failed."
    return 1
  fi
  echo "$(date '+%F %T') ascgen-dev test success!"
}

build_test_ascir_st() {
  echo "$(date '+%F %T') create build directory and build test_ascir_st";
  cd "${BUILD_PATH}"

  CMAKE_ARGS="-D CMAKE_C_COMPILER=${C_COMPILER} \
              -D CMAKE_CXX_COMPILER=${CXX_COMPILER} \
              -D ASCEND_3RD_LIB_PATH=${ASCEND_3RD_LIB_PATH} \
              -D CANN_3RD_LIB_PATH=${ASCEND_3RD_LIB_PATH} \
              -D ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH} \
              -D CMAKE_INSTALL_PREFIX=${OUTPUT_PATH} \
              -D ASCEND_INSTALL_LIB_PATH=${ASCEND_INSTALL_LIB_PATH} \
              -D ENABLE_OPEN_SRC=True \
              -D BUILD_METADEF=ON \
              -D RUN_TEST=1 \
              -D ENABLE_PKG=${ENABLE_PKG} \
              -D ENABLE_TEST=True \
              -D ENABLE_GE_ST=on \
              -D ENABLE_LLT_PKG=ON"
  cmake $CMAKE_ARGS ..

  make -j${THREAD_NUM} test_ascir_st

  if [ $? -ne 0 ]
  then
    echo "execute command: make test_ascir_st failed."
    return 1
  fi

  echo "build test_ascir_st success!"

  cp ${AUTOFUSE_BUILD_PATH}/tests/st/ascir/test_ascir_st  ${OUTPUT_PATH}
  ctest --output-on-failure -j${THREAD_NUM} -L st -L ascir_st \
        --test-dir ${AUTOFUSE_BUILD_PATH}/tests --no-tests=error

  if [ $? -ne 0 ]
  then
    env
    echo "execute command: make -j test_ascir_st failed."
    return 1
  fi

  echo "$(date '+%F %T') run test_ascir_st success!"
}

build_ut_optimize() {
  echo "$(date '+%F %T') create build directory and build optimize ut";
  cd "${BUILD_PATH}"
  make -j${THREAD_NUM} optimize_ut
  if [ $? -ne 0 ]
  then
    echo "execute command: make optimize_ut failed."
    return 1
  fi

  cp ${AUTOFUSE_BUILD_PATH}/tests/ut/optimize/optimize_ut  ${OUTPUT_PATH}
  ctest --output-on-failure -j${THREAD_NUM} \
        --test-dir ${AUTOFUSE_BUILD_PATH}/tests/ut/optimize --no-tests=error

  if [ $? -ne 0 ]
  then
    echo "execute command: run optimize_ut failed."
    return 1
  fi

  echo "$(date '+%F %T') optimize_st test successfully!"
}

build_ut_common () {
  echo "$(date '+%F %T') create build directory and build test_common";
  cd "${BUILD_PATH}"
  make -j${THREAD_NUM} test_common
  if [ $? -ne 0 ]
  then
    echo "execute command: make test_common failed."
    return 1
  fi

  cp ${AUTOFUSE_BUILD_PATH}/tests/ut/common/test_common  ${OUTPUT_PATH}
  ctest --output-on-failure -j${THREAD_NUM} \
        --test-dir ${AUTOFUSE_BUILD_PATH}/tests/ut/common --no-tests=error
  if [ $? -ne 0 ]; then
    echo "execute command: run test_common failed."
    return 1
  fi
  echo "$(date '+%F %T') test_common test successfully!"
}

build_ut_autofuse_utils() {
  echo "$(date '+%F %T') create build directory and build autofuse_utils_ut";
  cd "${BUILD_PATH}"
  make -j${THREAD_NUM} autofuse_utils_ut
  if [ $? -ne 0 ]; then
    echo "execute command: make autofuse_utils_ut failed."
    return 1
  fi

  cp ${AUTOFUSE_BUILD_PATH}/tests/ut/autofuse/autofuse_utils_ut ${OUTPUT_PATH}
  RUN_TEST_CASE=${OUTPUT_PATH}/autofuse_utils_ut && ${RUN_TEST_CASE}
  if [ $? -ne 0 ]; then
    echo "execute command: run autofuse_utils_ut failed."
    return 1
  fi
  echo "$(date '+%F %T') autofuse_utils_ut test successfully!"
}

build_st_optimize() {
  echo "$(date '+%F %T') create build directory and build optimiz st";
  cd "${BUILD_PATH}"
  make -j${THREAD_NUM} optimize_st
  if [ $? -ne 0 ]
  then
    echo "execute command: make optimize_st failed."
    return 1
  fi

  ctest --output-on-failure -j${THREAD_NUM} -L st -L optimize_st --test-dir ${AUTOFUSE_BUILD_PATH}/tests --no-tests=error \
        -O ${BUILD_PATH}/optimize_st.log
  if [ $? -ne 0 ]
  then
    echo "execute command: run optimize_st failed."
    return 1
  fi

  echo "$(date '+%F %T') optimize_st test successfully!"
}

build_st_autofuse() {
  echo "$(date '+%F %T') create build directory and build autofuse st";
  cd "${BUILD_PATH}"
  make -j${THREAD_NUM} autofusion_st
  if [ $? -ne 0 ]
  then
    echo "execute command: make autofuse_st failed."
    return 1
  fi

  ctest --output-on-failure -j${THREAD_NUM} -L st -L autofusion_st --test-dir ${AUTOFUSE_BUILD_PATH}/tests --no-tests=error \
        -O ${BUILD_PATH}/autofusion_st.log
  if [ $? -ne 0 ]
  then
    echo "execute command: run autofuse_st failed."
    return 1
  fi
  echo "$(date '+%F %T') autofuse_st test successfully!"
}

build_pyautofuse() {
  echo "create build directory and build pyautofuse";
  cd "${BUILD_PATH}"
  make -j${THREAD_NUM} pyautofuse
  if [ $? -ne 0 ]
  then
    echo "execute command: make pyautofuse failed."
    return 1
  fi
}

build_st_codegen() {
  echo "create build directory and build codegen st";
  cd "${BUILD_PATH}"
  make -j${THREAD_NUM} codegen_st
  if [ $? -ne 0 ]
  then
    echo "execute command: make codegen_st failed."
    return 1
  fi

  ctest --output-on-failure -j${THREAD_NUM} -L st -L codegen_st --test-dir ${AUTOFUSE_BUILD_PATH}/tests --no-tests=error \
        -O ${BUILD_PATH}/codegen_st.log
  if [ $? -ne 0 ]
  then
    echo "execute command: run codegen_st failed."
    return 1
  fi
  echo "codegen_st test successfully!"
}

build_st_common() {
  echo "$(date '+%F %T') create build directory and build common st";
  cd "${BUILD_PATH}"
  make -j${THREAD_NUM} test_common_st
  if [ $? -ne 0 ]
  then
    echo "execute command: make common_st failed."
    return 1
  fi

  ctest --output-on-failure -j${THREAD_NUM} -L st -L test_common_st --test-dir ${AUTOFUSE_BUILD_PATH}/tests --no-tests=error \
        -O ${BUILD_PATH}/test_common_st.log
  if [ $? -ne 0 ]
  then
    echo "execute command: run common_st failed."
    return 1
  fi
  echo "$(date '+%F %T') common_st test successfully!"
}

get_coverage() {
    echo "Generating coverage statistics, please wait..."
    cd "${BASEPATH}"
    rm -rf ${BASEPATH}/cov
    mk_dir ${BASEPATH}/cov

    local lcov_ignore_params=""
    local lcov_rc_param=""
    local lcov_parallel_params=""
    local genhtml_ignore_errors=""

    if [ "$(get_lcov_major_version)" -ge 2 ] 2>/dev/null; then
        lcov_ignore_params="--ignore-errors inconsistent,negative,mismatch,empty"
        lcov_rc_param="--rc geninfo_unexecuted_blocks=1"
        genhtml_ignore_errors="--ignore-errors inconsistent,corrupt"

        lcov_parallel_params=$(get_lcov_parallel_params ${THREAD_NUMBER:-${THREAD_NUM}})
        if [ -n "$lcov_parallel_params" ]; then
            lcov_ignore_params="--ignore-errors child,inconsistent,negative,mismatch,empty"
        fi
    fi

    lcov -c \
      -d ${BUILD_RELATIVE_PATH}/ \
      ${lcov_ignore_params} \
      ${lcov_rc_param} \
      ${lcov_parallel_params} \
      -o cov/tmp.info

    echo ${ASCEND_INSTALL_PATH}
    lcov -r cov/tmp.info "${ASCEND_INSTALL_PATH}/*" '*/output/*' '*/base/metadef/*' '*/nlohmann/*' \
        "*/${BUILD_RELATIVE_PATH}/opensrc/*" "*/${BUILD_RELATIVE_PATH}/proto/*" '*/third_party/*' '/usr/*' \
        ${lcov_ignore_params} --ignore-errors unused \
        ${lcov_rc_param} \
        ${lcov_parallel_params} \
        -o cov/coverage.info
    genhtml cov/coverage.info --output-directory cov/coverage_report ${genhtml_ignore_errors}
}

codegen_e2e_st() {
  echo "$(date '+%F %T') create build directory and build codegen_e2e_st";
  mk_dir "${BUILD_PATH}"
  cd "${BUILD_PATH}"
  g++ -v

  ASCEND_INSTALL_LIB_PATH=${ASCEND_INSTALL_PATH}/$(uname -m)-linux/lib64/
  CMAKE_ARGS="-D CMAKE_C_COMPILER=${C_COMPILER} \
            -D CMAKE_CXX_COMPILER=${CXX_COMPILER} \
            -D ASCEND_3RD_LIB_PATH=${ASCEND_3RD_LIB_PATH} \
            -D CANN_3RD_LIB_PATH=${ASCEND_3RD_LIB_PATH} \
            -D ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH} \
            -D CMAKE_INSTALL_PREFIX=${OUTPUT_PATH} \
            -D ASCEND_INSTALL_LIB_PATH=${ASCEND_INSTALL_LIB_PATH} \
            -D ENABLE_OPEN_SRC=True \
            -D BUILD_METADEF=ON \
            -D ENABLE_PKG=${ENABLE_PKG} \
            -D ENABLE_TEST=True \
            -D RUN_TEST=1 \
            -D ENABLE_GE_ST=on \
            -D ENABLE_LLT_PKG=ON"
  env

  cmake $CMAKE_ARGS ../

  # st用例可执行文件的列表
  MAKE_TARGET_LIST="load_where_store_expect_code_e2e \
                    load_where_x2x3_is_ubscalar_store_expect_code_e2e \
                    load_where_x2x3_is_ubscalar_throwfor_store_expect_code_e2e \
                    constant_load_gt_store_expect_code_e2e \
                    constant_load_le_store_expect_code_e2e \
                    discrete_store_e2e \
                    discrete_load_e2e \
                    broadcast_force_merge_e2e \
                    load_abs_store_expect_code_e2e \
                    load_leakyrelu_store_expect_code_e2e \
                    load_transpose_store_expect_code_e2e \
                    load_ub_scalar_add_store_expect_code_e2e \
                    load_sub_store_expect_code_e2e \
                    broadcast_merge_axis_e2e \
                    store_scalar_e2e \
                    concat_same_tail_dim_e2e \
                    concat_small_tail_dim_e2e \
                    concat_inter_dim_e2e \
                    load_ub2ub_abs_store_expect_code_e2e \
                    concat_3d_last_dim_e2e \
                    dynamic_inputs_and_outputs_e2e \
                    load_isfinite_store_e2e \
                    load_max_min_store_e2e \
                    load_rsum_block_store_e2e \
                    load_reciprocal_store_e2e \
                    load_bitwiseand_store_e2e \
                    load_strided_slice_store_e2e \
                    load_store_expect_code_e2e \
                    schedule_multi_group_ws_reuse_output_e2e \
                    broadcast_multi_axes_e2e \
                    load_erf_store_e2e \
                    load_scalar_sub_store_e2e \
                    load_gather_split_b_t_abs_store_e2e \
                    load_gather_split_bt_abs_store_e2e \
                    load_gather_split_bt_t_abs_store_e2e \
                    load_gather_split_t_bt_abs_store_e2e \
                    store_empty_tensor_e2e \
                    load_rsum_invalid_axis_store_e2e \
                    load_gather_first_axis_split_b_t_abs_store_e2e \
                    load_logicalnot_store_e2e"
  if [[ "X$RUN_V35_TESTS" = "Xon" ]]; then
    MAKE_TARGET_LIST="${MAKE_TARGET_LIST} \
                      load_abs_store_expect_code_e2e_v2 \
                      load_scalar_abs_brc_store_expect_code_e2e_v2 \
                      load_scalar_brc_store_2d_expect_code_e2e_v2 \
                      load_scalar_clip_store_expect_code_e2e_v2 \
                      load_div_store_expect_code_e2e_v2 \
                      load_scalar_sub_store_expect_code_e2e_v2 \
                      load_scalar_div_store_expect_code_e2e_v2 \
                      load_switch_scalar_sub_store_expect_code_e2e_v2 \
                      load_nan_out_for_store_expect_code_e2e_v2"
  fi
  # 从 build_backend 移入的 >10s 长尾用例（含 inductor/pgo split_compile 链），与 ascendc_api 门禁并行。
  # ctest 仍按 build_backend_test1/2 标签 + 用例名正则选择（见下方 moved 两次 ctest 调用）。
  BACKEND_MOVED_TARGET_LIST="add_rsqrt_test_e2e \
                      compare_x2_tensor_int64_eq_test_e2e \
                      compare_x2_tensor_int64_gt_test_e2e \
                      inductor_tail_brc_tail_reduce_test_e2e \
                      inductor_topn_test_e2e \
                      load_logical_not_store_test_e2e \
                      pgo_add_abs_inductor_concat_test_e2e \
                      pgo_add_abs_inductor_test_e2e"
  BACKEND_MOVED_TARGET_LIST_V35=""
  if [[ "X$RUN_V35_TESTS" = "Xon" ]]; then
    BACKEND_MOVED_TARGET_LIST_V35="acos_bf16_test_e2e_v2 \
                        add_abs_test_e2e_v2 \
                        add_exp2_test_e2e_v2 \
                        add_exp_bf16_test_e2e_v2 \
                        add_floor_bf16_test_e2e_v2 \
                        add_floor_test_e2e_v2 \
                        atanh_bf16_test_e2e_v2 \
                        bessel_j0_store_test_e2e_v2 \
                        bessel_j1_store_test_e2e_v2 \
                        bessel_y0_store_test_e2e_v2 \
                        bessel_y1_store_test_e2e_v2 \
                        bf16_add_test_e2e_v2 \
                        bf16_nddma_add_test_e2e_v2 \
                        cast_abs_float16_float_test_e2e_v2 \
                        cast_abs_test_e2e_v2 \
                        cast_nan_test_e2e_v2 \
                        chebyshev_polynomial_t_store_test_e2e_v2 \
                        chebyshev_polynomial_u_store_test_e2e_v2 \
                        chebyshev_polynomial_v_store_test_e2e_v2 \
                        chebyshev_polynomial_w_store_test_e2e_v2 \
                        continues_brc_test_e2e_v2 \
                        digamma_bf16_test_e2e_v2 \
                        div_abs_test_e2e_v2 \
                        erfc_bf16_test_e2e_v2 \
                        floor_div_int32_store_test_e2e_v2 \
                        floordiv_abs_test_e2e_v2 \
                        frexp_store_test_e2e_v2 \
                        hermite_polynomial_h_store_test_e2e_v2 \
                        hermite_polynomial_he_store_test_e2e_v2 \
                        i0_store_test_e2e_v2 \
                        i0e_store_test_e2e_v2 \
                        i1e_store_test_e2e_v2 \
                        indirect_load_broadcast_elements_sk_test_e2e_v2 \
                        indirect_load_broadcast_index_where_simt_test_e2e_v2 \
                        indirect_load_graph_hint_embedding_slice_e2e_v2 \
                        indirect_load_rank3_axis1_torch_gather_frontend_e2e_v2 \
                        indirect_load_rank4_axis1_full_prefix_bessel_k0_sum_simd_e2e_v2 \
                        indirect_load_user_position_bias_e2e_v2 \
                        laguerre_polynomial_l_store_test_e2e_v2 \
                        legendre_polynomial_p_store_test_e2e_v2 \
                        load_airy_ai_store_test_e2e_v2 \
                        load_bitwise_and_store_test_e2e_v2 \
                        load_bitwise_not_store_test_e2e_v2 \
                        load_bitwise_or_store_test_e2e_v2 \
                        load_bitwise_xor_store_test_e2e_v2 \
                        load_erfinv_store_test_e2e_v2 \
                        load_leaky_relu_store_test_e2e_v2 \
                        load_log2_store_test_e2e_v2 \
                        load_logicaland_store_test_e2e_v2 \
                        load_logicalor_store_test_e2e_v2 \
                        load_lshift_store_test_e2e_v2 \
                        load_modified_bessel_i0_store_test_e2e_v2 \
                        load_modified_bessel_i1_store_test_e2e_v2 \
                        load_modified_bessel_k0_store_test_e2e_v2 \
                        load_modified_bessel_k1_store_test_e2e_v2 \
                        load_tanh_store_test_e2e_v2 \
                        log_ndtr_store_test_e2e_v2 \
                        logicalxor_float_test_e2e_v2 \
                        mod_test_e2e_v2 \
                        ndtr_store_test_e2e_v2 \
                        ndtri_store_test_e2e_v2 \
                        next_after_store_test_e2e_v2 \
                        poly_gamma_store_test_e2e_v2 \
                        remainder_int32_store_test_e2e_v2 \
                        scalar_div_inf_test_e2e_v2 \
                        scaled_modified_bessel_k0_store_test_e2e_v2 \
                        scaled_modified_bessel_k1_store_test_e2e_v2 \
                        shifted_chebyshev_polynomial_t_store_test_e2e_v2 \
                        shifted_chebyshev_polynomial_u_store_test_e2e_v2 \
                        shifted_chebyshev_polynomial_v_store_test_e2e_v2 \
                        shifted_chebyshev_polynomial_w_store_test_e2e_v2 \
                        signbit_store_test_e2e_v2 \
                        spherical_bessel_j0_store_test_e2e_v2 \
                        zeta_store_test_e2e_v2"
  fi
  MAKE_TARGET_LIST="${MAKE_TARGET_LIST} ${BACKEND_MOVED_TARGET_LIST} ${BACKEND_MOVED_TARGET_LIST_V35}"
  BACKEND_MOVED_CODEGEN_LIST=$(echo "${BACKEND_MOVED_TARGET_LIST} ${BACKEND_MOVED_TARGET_LIST_V35}" | sed 's/e2e/codegen/g')
  BACKEND_MOVED_TEST1_REGEX=$(build_backend_test_regex ${BACKEND_MOVED_CODEGEN_LIST})
  BACKEND_MOVED_TEST2_REGEX=$(build_backend_test_regex ${BACKEND_MOVED_TARGET_LIST} ${BACKEND_MOVED_TARGET_LIST_V35})
  MAKE_TARGET_LIST_CODEGEN=$(echo "${MAKE_TARGET_LIST}" | sed 's/e2e/codegen/g')
  echo "MAKE_TARGET_LIST_CODEGEN"
  echo $MAKE_TARGET_LIST_CODEGEN
  make -j${THREAD_NUM} $MAKE_TARGET_LIST_CODEGEN
  if [ $? -ne 0 ]
  then
    echo "execute command: make codegen_e2e_st_test1 failed."
    return 1
  fi
  echo "$(date '+%F %T') make codegen_e2e_st_test1 end"

  (cd "${AUTOFUSE_BUILD_PATH}/tests" && \
    ctest --output-on-failure -j${THREAD_NUM} -L st -L codegen_e2e_st_test1 --no-tests=error \
          -O ${BUILD_PATH}/ctest_codegen_e2e_st_test1.log)
  if [ $? -ne 0 ]; then
    echo "execute command: run codegen_e2e_st_test1 failed."
    return 1
  fi

  (cd "${AUTOFUSE_BUILD_PATH}/tests" && \
    ctest --output-on-failure -j${THREAD_NUM} -L st -L build_backend_test1 -R "${BACKEND_MOVED_TEST1_REGEX}" --no-tests=error \
          -O ${BUILD_PATH}/ctest_codegen_e2e_st_moved_test1.log)
  if [ $? -ne 0 ]; then
    echo "execute command: run moved backend codegen_e2e_st_test1 failed."
    return 1
  fi

  make -j${THREAD_NUM} $MAKE_TARGET_LIST
  if [ $? -ne 0 ]
  then
    echo "execute command: make codegen_e2e_st_test2 failed."
    return 1
  fi
  echo "$(date '+%F %T') make codegen_e2e_st_test2 end"

  (cd "${AUTOFUSE_BUILD_PATH}/tests" && \
    ctest --output-on-failure -j${THREAD_NUM} -L st -L codegen_e2e_st_test2 --no-tests=error \
          -O ${BUILD_PATH}/ctest_codegen_e2e_st_test2.log)
  if [ $? -ne 0 ]; then
    echo "execute command: run codegen_e2e_st_test2 failed."
    return 1
  fi

  (cd "${AUTOFUSE_BUILD_PATH}/tests" && \
    ctest --output-on-failure -j${THREAD_NUM} -L st -L build_backend_test2 -R "${BACKEND_MOVED_TEST2_REGEX}" --no-tests=error \
          -O ${BUILD_PATH}/ctest_codegen_e2e_st_moved_test2.log)
  if [ $? -ne 0 ]; then
    echo "execute command: run moved backend codegen_e2e_st_test2 failed."
    return 1
  fi
  echo "$(date '+%F %T') codegen_e2e_st execute success!"
}

build_backend_test_regex() {
  local test_regex=""
  local test_name
  for test_name in "$@"; do
    if [ -n "${test_regex}" ]; then
      test_regex="${test_regex}|"
    fi
    test_regex="${test_regex}${test_name}"
  done
  echo "^(${test_regex})$"
}

build_backend() {
  echo "$(date '+%F %T') create build directory and build build_backend";
  mk_dir "${BUILD_PATH}"
  cd "${BUILD_PATH}"
  g++ -v

  ASCEND_INSTALL_LIB_PATH=${ASCEND_INSTALL_PATH}/$(uname -m)-linux/lib64/
  CMAKE_ARGS="-D CMAKE_C_COMPILER=${C_COMPILER} \
            -D CMAKE_CXX_COMPILER=${CXX_COMPILER} \
            -D ASCEND_3RD_LIB_PATH=${ASCEND_3RD_LIB_PATH} \
            -D CANN_3RD_LIB_PATH=${ASCEND_3RD_LIB_PATH} \
            -D ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH} \
            -D CMAKE_INSTALL_PREFIX=${OUTPUT_PATH} \
            -D ASCEND_INSTALL_LIB_PATH=${ASCEND_INSTALL_LIB_PATH} \
            -D ENABLE_OPEN_SRC=True \
            -D BUILD_METADEF=ON \
            -D ENABLE_PKG=${ENABLE_PKG} \
            -D ENABLE_TEST=True \
            -D ENABLE_GE_ST=on \
            -D ENABLE_LLT_PKG=ON"
  env

  cmake $CMAKE_ARGS ../

  # st用例可执行文件的列表。耗时>10s的长尾用例（含 inductor/pgo split_compile 代表用例）已移至
  # e2e 模块（codegen_e2e_st，对应 ST_Test_autofuse_e2e 任务）并行执行；其余 inductor split_compile 放 nightly。
  MAKE_TARGET_LIST="add_abs_test_e2e \
                    axpy_abs_test_e2e \
                    sub_abs_test_e2e \
                    scalar_float_inf_test_e2e \
                    scalar_div_inf_test_e2e \
                    add_gelu_test_e2e \
                    compare_test_e2e \
                    compare_x2_tensor_test_e2e \
                    compare_x2_tensor_int32_test_e2e \
                    load_to_store_and_abs_test_e2e \
                    scalar_cast_add_test_e2e \
                    concat_all_aligned_test_e2e \
                    concat_to_stores_test_e2e \
                    load_unalign_pad_test_e2e \
                    brc_inline_test_e2e \
                    add_neg_test_e2e \
                    load_where_store_test_e2e \
                    isinf_maskedfill_test_e2e \
                    load_where_x2_x3_is_ubscalar_store_test_e2e \
                    load_where_x2_is_ubscalar_store_test_e2e \
                    load_where_x3_is_ubscalar_store_test_e2e \
                    truediv_abs_test_e2e \
                    load_pow_all_input_is_scalar_store_test_e2e \
                    axpy_abs_half_test_e2e \
                    pgo_add_abs_test_e2e \
                    matmul_elemwise_test_e2e \
                    inductor_matmul_elemwise_test_e2e"
  if [[ "X$RUN_V35_TESTS" = "Xon" ]]; then
    MAKE_TARGET_LIST="${MAKE_TARGET_LIST} \
                      load_loop_mode_test_e2e_v2 \
                      concat_test_e2e_v2 \
                      scalar_brc_test_e2e_v2 \
                      brc_reduce_test_e2e_v2 \
                      log1p_bfloat16_test_e2e_v2 \
                      floortoint_float_test_e2e_v2 \
                      fmod_float_test_e2e_v2 \
                      hypot_float_test_e2e_v2 \
                      lgamma_float_test_e2e_v2 \
                      log10_float_test_e2e_v2 \
                      load_brc_test_e2e_v2 \
                      add_abs_int8_scalar_test_e2e_v2 \
                      add_abs_half_scalar_test_e2e_v2 \
                      add_abs_float_scalar_test_e2e_v2 \
                      abs_brc_add_test_e2e_v2 \
                      ub_scalar_brc_abs_add_test_e2e_v2 \
                      abs_fma_bf16_test_e2e_v2 \
                      abs_fma_test_e2e_v2 \
                      floordiv_mul_le_select_test_e2e_v2 \
                      tail_brc_tail_reduce_test_e2e_v2 \
                      int32_logical_not_test_e2e_v2 \
                      int16_logical_not_test_e2e_v2 \
                      float_logical_not_test_e2e_v2 \
                      half_logical_not_test_e2e_v2 \
                      uint8_logical_not_test_e2e_v2 \
                      abs_clip_by_value_test_e2e_v2 \
                      load_gather_split_b_t_abs_store_test_e2e_v2 \
                      load_gather_tail_split_b_t_abs_store_test_e2e_v2 \
                      load_gather_one_axis_split_b_t_abs_store_test_e2e_v2 \
                      indirect_load_rank2_axis1_simd_e2e_v2 \
                      indirect_load_rank3_axis1_float_int64_gather_e2e_v2 \
                      indirect_load_mixed_rank4_axis2_simd_e2e_v2 \
                      indirect_load_broadcast_cross_boundary_simt_fallback_test_e2e_v2 \
                      indirect_load_broadcast_axis_simd_test_e2e_v2 \
                      indirect_load_broadcast_inner_adjacent_simd_test_e2e_v2 \
                      indirect_load_broadcast_continuous_simd_test_e2e_v2 \
                      indirect_load_broadcast_continuous_index_simt_test_e2e_v2 \
                      indirect_load_embedding_test_e2e_v2 \
                      indirect_load_embedding_tail_simd_e2e_v2 \
                      indirect_load_broadcast_retained_simd_test_e2e_v2 \
                      indirect_load_complex_broadcast_simd_test_e2e_v2 \
                      indirect_load_complex_broadcast_simt_test_e2e_v2 \
                      indirect_load_complex_input_broadcast_simd_test_e2e_v2 \
                      indirect_load_broadcast_cross_boundary_simt_test_e2e_v2 \
                      indirect_load_broadcast_axis_simt_test_e2e_v2 \
                      indirect_load_broadcast_reduce_simt_fallback_test_e2e_v2 \
                      indirect_load_broadcast_retained_simt_test_e2e_v2 \
                      indirect_load_both_transpose_simd_test_e2e_v2 \
                      indirect_load_both_transpose_simd_fallback_test_e2e_v2 \
                      indirect_load_both_transpose_simd_nddma_test_e2e_v2 \
                      indirect_load_both_transpose_simt_test_e2e_v2 \
                      indirect_load_broadcast_index_physical_view_simt_test_e2e_v2 \
                      indirect_load_broadcast_index_abs_simt_test_e2e_v2 \
                      indirect_load_index_binary_same_view_simd_test_e2e_v2 \
                      indirect_load_rank3_axis1_input_index_gap_simd_test_e2e_v2 \
                      indirect_load_rank3_axis1_input_index_gap_simt_test_e2e_v2 \
                      indirect_load_rank3_axis1_input_index_outer_gap_simt_test_e2e_v2 \
                      indirect_load_graph_hint_simd_repro_e2e_v2 \
                      indirect_load_embedding_reduce_simt_test_e2e_v2 \
                      indirect_load_user_masked_embedding_minimal_e2e_v2 \
                      indirect_load_user_masked_embedding_sum_full_auto_e2e_v2 \
                      indirect_load_user_embedding_sum_e2e_v2 \
                      indirect_load_user_embedding_sum_rank2_e2e_v2 \
                      indirect_load_user_dtype_sweep_e2e_v2 \
                      indirect_load_user_int64_dense_sweep_e2e_v2 \
                      indirect_load_user_dynamic_shape_sweep_e2e_v2 \
                      indirect_load_user_embedding_sum_simd_e2e_v2 \
                      indirect_load_user_layernorm_e2e_v2 \
                      indirect_load_user_embedding_exp_abs_add_simt_e2e_v2 \
                      indirect_load_user_fanout_direct_reduce_simt_e2e_v2 \
                      indirect_load_user_fanout_post_reduce_simt_e2e_v2 \
                      indirect_load_user_fanout_side_input_simt_e2e_v2 \
                      indirect_load_user_fanout_direct_stores_simd_e2e_v2 \
                      indirect_load_user_fanout_direct_stores_simt_e2e_v2 \
                      indirect_load_user_fanout_post_stores_simd_e2e_v2 \
                      indirect_load_user_fanout_post_stores_simt_e2e_v2 \
                      indirect_load_user_fanout_direct_stores_simd_e2e_v2 \
                      indirect_load_user_fanout_direct_stores_simt_e2e_v2 \
                      indirect_load_user_fanout_post_stores_simd_e2e_v2 \
                      indirect_load_user_fanout_post_stores_simt_e2e_v2 \
                      indirect_load_user_side_input_fanout_simd_e2e_v2 \
                      load_where_x2_x3_is_ubscalar_store_test_e2e_v2 \
                      gather_reduce_store_test_e2e_v2 \
                      load_where_store_test_e2e_v2 \
                      load_where_x2_is_ubscalar_store_test_e2e_v2 \
                      load_where_x3_is_ubscalar_store_test_e2e_v2 \
                      load_compare_store_test_e2e_v2 \
                      le_bool_store_test_e2e_v2 \
                      bitwise_bool_store_test_e2e_v2 \
                      load_compare_cast_sum_store_test_e2e_v2 \
                      matmul_elemwise_brc_test_e2e_v2 \
                      matmul_compare_scalar_test_e2e_v2 \
                      abs_bf16_test_e2e_v2 \
                      abs_uint8_test_e2e_v2 \
                      erf_bf16_test_e2e_v2 \
                      ceil_bf16_test_e2e_v2 \
                      cos_bf16_test_e2e_v2 \
                      load_compare_scalar_where_store_test_e2e_v2 \
                      load_compare_where_store_test_e2e_v2 \
                      binary_api_scalar_test_e2e_v2 \
                      acosh_bf16_test_e2e_v2 \
                      asin_bf16_test_e2e_v2 \
                      asinh_bf16_test_e2e_v2 \
                      atan_bf16_test_e2e_v2 \
                      scalar_cast_add_test_e2e_v2 \
                      cosh_bf16_test_e2e_v2 \
                      pow_bf16_test_e2e_v2 \
                      reciprocal_bf16_test_e2e_v2 \
                      relu_uint8_test_e2e_v2 \
                      round_bf16_test_e2e_v2 \
                      rshift_uint8_test_e2e_v2 \
                      sign_uint8_test_e2e_v2 \
                      sign_bf16_test_e2e_v2 \
                      truediv_bf16_test_e2e_v2 \
                      atan2_bf16_test_e2e_v2 \
                      ceil2int_bf16_test_e2e_v2 \
                      bool_mask_flow_test_e2e_v2 \
                      transpose_bool_test_e2e_v2 \
                      select_bool_test_e2e_v2 \
                      copysign_bf16_test_e2e_v2 \
                      erfcx_test_e2e_v2 \
                      expm1_test_e2e_v2 \
                      rand_store_test_e2e_v2 \
                      randn_store_test_e2e_v2"
  fi
  MAKE_TARGET_LIST_CODEGEN=$(echo "${MAKE_TARGET_LIST}" | sed 's/e2e/codegen/g')
  CTEST_BACKEND_TEST1_REGEX=$(build_backend_test_regex ${MAKE_TARGET_LIST_CODEGEN})
  CTEST_BACKEND_TEST2_REGEX=$(build_backend_test_regex ${MAKE_TARGET_LIST})
  echo "MAKE_TARGET_LIST_CODEGEN"
  echo $MAKE_TARGET_LIST_CODEGEN
  make -j${THREAD_NUM} $MAKE_TARGET_LIST_CODEGEN
  if [ $? -ne 0 ]
  then
    echo "execute command: make build_backend_test1 failed."
    return 1
  fi
  echo "$(date '+%F %T') make build_backend_test1 end"

  (cd "${AUTOFUSE_BUILD_PATH}/tests" && \
    ctest --output-on-failure -j${THREAD_NUM} -L st -L build_backend_test1 -R "${CTEST_BACKEND_TEST1_REGEX}" \
          --no-tests=error -O ${BUILD_PATH}/ctest_build_backend_test1.log)
  if [ $? -ne 0 ]; then
    echo "execute command: run build_backend_test1 failed."
    return 1
  fi

  make -j${THREAD_NUM} $MAKE_TARGET_LIST
  if [ $? -ne 0 ]
  then
    echo "execute command: make build_backend_test2 failed."
    return 1
  fi
  echo "$(date '+%F %T') make build_backend_test2 end"

  (cd "${AUTOFUSE_BUILD_PATH}/tests" && \
    ctest --output-on-failure -j${THREAD_NUM} -L st -L build_backend_test2 -R "${CTEST_BACKEND_TEST2_REGEX}" \
          --no-tests=error -O ${BUILD_PATH}/ctest_build_backend_test2.log)
  if [ $? -ne 0 ]; then
    echo "execute command: run build_backend_test2 failed."
    return 1
  fi
  echo "$(date '+%F %T') build_backend execute success!"
}

build_kernel_tool() {
  echo "create build directory and build codegen kernel tool";
  cp ${AUTOFUSE_PATH}/tests/st/codegen/kernel_tool ${AUTOFUSE_BUILD_PATH}/tests/st/codegen/ -r
  cd "${AUTOFUSE_BUILD_PATH}/tests/st/codegen/kernel_tool"
  make CANN_INSTALL_PATH=${ASCEND_INSTALL_PATH}
  if [ $? -ne 0 ]
  then
    echo "execute command: make kernel tool failed."
    return 1
  fi

  cp ${AUTOFUSE_BUILD_PATH}/tests/st/codegen/kernel_tool/test_kernel  ${OUTPUT_PATH}
  ctest --output-on-failure -j${THREAD_NUM} -L st -L kernel_tool_st \
        --test-dir ${AUTOFUSE_BUILD_PATH}/tests --no-tests=error
  if [ $? -ne 0 ]
  then
    echo "execute command: run kernel tool failed."
    return 1
  fi
  echo "kernel tool test successfully!"
}

run_py_module_test() {
    local test_dir="$1"
    local ascend_devlib_path="${ASCEND_INSTALL_PATH}/$(uname -m)-linux/devlib"
    local ascend_device_path="${ASCEND_INSTALL_PATH}/$(uname -m)-linux/devlib/device"
    local test_ld_library_path="${LOCAL_RUNTIME_LIB_PATH}:${METADEF_LIB_PATH}:${ASCEND_INSTALL_LIB_PATH}:${ascend_devlib_path}:${ascend_device_path}"

    if [ -n "${INITIAL_LD_LIBRARY_PATH}" ]; then
        test_ld_library_path="${test_ld_library_path}:${INITIAL_LD_LIBRARY_PATH}"
    fi

    mk_dir ${PYTHON_LIB_PATH}/autofuse || true
    cp -f "${AUTOFUSE_BUILD_PATH}/tests/pyautofuse.so" "${PYTHON_LIB_PATH}/autofuse/pyautofuse.so"
    cp "${PYTHON_MODULE_PATH}"/*.py "${PYTHON_LIB_PATH}/autofuse" || true
    export PYTHONPATH="${PYTHON_LIB_PATH}:${PYTHONPATH}"
    rm -rf "${test_dir}/__pycache__/" || true
    if ! LD_LIBRARY_PATH="${test_ld_library_path}" python3 -m pytest -s -vv "$test_dir"; then
        echo "py module test failed."
        return 1
    fi
    echo "py module test success!"
}

py_module_st() {
    build_pyautofuse || { echo "build pyautofuse test failed."; exit 1; }
    run_py_module_test "${AUTOFUSE_PATH}/tests/st/python/"
}

py_module_ut() {
    build_pyautofuse || { echo "build pyautofuse test failed."; exit 1; }
    run_py_module_test "${AUTOFUSE_PATH}/tests/ut/python/"
}

build_ut_att() {
  echo "create build directory and build att ut";

  cd "${BUILD_PATH}"
  CMAKE_ARGS="-D CMAKE_C_COMPILER=${C_COMPILER} \
              -D CMAKE_CXX_COMPILER=${CXX_COMPILER} \
              -D ASCEND_3RD_LIB_PATH=${ASCEND_3RD_LIB_PATH} \
              -D CANN_3RD_LIB_PATH=${ASCEND_3RD_LIB_PATH} \
              -D ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH} \
              -D CMAKE_INSTALL_PREFIX=${OUTPUT_PATH} \
              -D ASCEND_INSTALL_LIB_PATH=${ASCEND_INSTALL_LIB_PATH} \
              -D ENABLE_OPEN_SRC=True \
              -D ENABLE_SYMENGINE=True \
              -D BUILD_METADEF=ON \
              -D RUN_TEST=1 \
              -D RUN_ATT_TEST=1 \
              -D ENABLE_PKG=${ENABLE_PKG} \
              -D ENABLE_TEST=True \
              -D ENABLE_GE_UT=on \
              -D ENABLE_LLT_PKG=ON"
  cmake $CMAKE_ARGS ..
  make -j${THREAD_NUM} att_ut
  if [ $? -ne 0 ]
  then
    echo "execute command: make att_ut failed."
    return 1
  fi

  cp ${AUTOFUSE_BUILD_PATH}/tests/ut/att/att_ut  ${OUTPUT_PATH}
  ldd -r ${OUTPUT_PATH}/att_ut
  ctest --output-on-failure -j${THREAD_NUM} \
        --test-dir ${AUTOFUSE_BUILD_PATH}/tests/ut/att --no-tests=error
  if [ $? -ne 0 ]; then
    echo "att_ut test failed."
    return 1
  fi
  echo "att_ut test successfully!"
}

build_st_att() {
  echo "$(date '+%F %T') create build directory and build att st"
  cd "${BUILD_PATH}"

  CMAKE_ARGS="-D CMAKE_C_COMPILER=${C_COMPILER} \
              -D CMAKE_CXX_COMPILER=${CXX_COMPILER} \
              -D ASCEND_3RD_LIB_PATH=${ASCEND_3RD_LIB_PATH} \
              -D CANN_3RD_LIB_PATH=${ASCEND_3RD_LIB_PATH} \
              -D ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH} \
              -D CMAKE_INSTALL_PREFIX=${OUTPUT_PATH} \
              -D ASCEND_INSTALL_LIB_PATH=${ASCEND_INSTALL_LIB_PATH} \
              -D ENABLE_OPEN_SRC=True \
              -D ENABLE_SYMENGINE=True \
              -D BUILD_METADEF=ON \
              -D RUN_TEST=1 \
              -D RUN_ATT_TEST=1 \
              -D ENABLE_PKG=${ENABLE_PKG} \
              -D ENABLE_TEST=True \
              -D ENABLE_GE_ST=on \
              -D ENABLE_LLT_PKG=ON"
  cmake $CMAKE_ARGS ..
  make -j${THREAD_NUM} att_st
  if [ $? -ne 0 ]
  then
    echo "execute command: make att_st failed."
    return 1
  fi

  ctest --output-on-failure -j${THREAD_NUM} -L st -L att_st --test-dir ${AUTOFUSE_BUILD_PATH}/tests --no-tests=error \
        -O ${BUILD_PATH}/att_st.log

  if [ $? -ne 0 ]
  then
    echo "att_st test failed."
    return 1
  fi
  echo "$(date '+%F %T') att_st test successfully!"
}

build_ut() {
  echo "build_ut start, mode = ${MODEL_NAME}."
  case ${MODEL_NAME} in
    "att")
      build_ut_att || { echo "test ut att failed."; exit 1; }
      ;;
    "autofuse_utils")
      build_ut_autofuse_utils || { echo "test ut autofuse_utils failed."; exit 1; }
      ;;
    "optimize")
      build_ut_optimize || { echo "test ut optimize failed."; exit 1; }
      ;;
    "codegen")
      build_test || { echo "test build failed."; exit 1; }
      py_module_ut || { echo "run py module ut failed."; exit 1; }
      ;;
    "common")
      build_ut_common || { echo "test ut common failed."; exit 1; }
      ;;
    "ascendc_api")
      build_test_ascendc_api_test || { echo "failed to build and run ascendc_api ."; exit 1; }
      ;;
    "framework")
      build_ut_att || { echo "failed to build and run att ut."; exit 1; }
      build_ut_optimize || { echo "failed to build and run optimize ut."; exit 1; }
      build_ut_common || { echo "test ut common failed."; exit 1; }
      build_test || { echo "test build failed."; exit 1; }
      py_module_ut || { echo "run py module ut failed."; exit 1; }
      ;;
    "all")
      build_ut_att || { echo "failed to build and run att ut."; exit 1; }
      build_ut_optimize || { echo "failed to build and run optimize ut."; exit 1; }
      build_ut_autofuse_utils || { echo "failed to build and run autofuse_utils ut."; exit 1; }
      build_ut_common || { echo "test ut common failed."; exit 1; }
      build_test || { echo "test build failed."; exit 1; }
      build_test_ascendc_api_test || { echo "failed to build and run ascendc_api ."; exit 1; }
      py_module_ut || { echo "run py module ut failed."; exit 1; }
      ;;
    *)
      echo "输入无效，输入范围att/autofuse/optimize/codegen/all."
      ;;
  esac
}

build_st() {
  case ${MODEL_NAME} in
    "att")
      build_st_att || { echo "failed to build and run att st."; exit 1; }
      ;;
    "ascir")
      build_test_ascir_st || { echo "run ascir st failed."; exit 1; }
      ;;
    "optimize")
      build_st_optimize || { echo "run optimize st failed."; exit 1; }
      ;;
    "common")
      build_st_common || { echo "run common st failed."; exit 1; }
      ;;
    "codegen")
      build_st_codegen || { echo "run codegen st failed."; exit 1; }
      codegen_e2e_st || { echo "test build e2e st code generator failed."; exit 1; }
      py_module_st || { echo "run py module st failed."; exit 1; }
      ;;
    "tools")
      build_kernel_tool || { echo "test kernel tool failed."; exit 1; }
      ;;
    "backend")
      build_backend || { echo "run backend st failed."; exit 1; }
      ;;
    "e2e")
      codegen_e2e_st || { echo "test build e2e st code generator failed."; exit 1; }
      ;;
    "ascendc_api")
      build_test_ascir_st || { echo "run ascir st failed."; exit 1; }
      build_st_codegen || { echo "run codegen st failed."; exit 1; }
      build_backend || { echo "run backend st failed."; exit 1; }
      build_kernel_tool || { echo "test kernel tool failed."; exit 1; }
      ;;
    "framework")
      build_st_att || { echo "failed to build and run att st."; exit 1; }
      build_st_common || { echo "run common st failed."; exit 1; }
      build_st_optimize || { echo "run optimize st failed."; exit 1; }
      py_module_st || { echo "run py module st failed."; exit 1; }
      ;;
    "all")
      build_st_att || { echo "failed to build and run att st."; exit 1; }
      build_test_ascir_st || { echo "run ascir st failed."; exit 1; }
      build_st_codegen || { echo "run codegen st failed."; exit 1; }
      build_st_common || { echo "run common st failed."; exit 1; }
      build_st_optimize || { echo "run optimize st failed."; exit 1; }
      build_backend || { echo "run backend st failed."; exit 1; }
      codegen_e2e_st || { echo "test build e2e st code generator failed."; exit 1; }
      build_kernel_tool || { echo "test kernel tool failed."; exit 1; }
      py_module_st || { echo "run py module st failed."; exit 1; }
      ;;
    *)
      echo "输入无效，输入范围att/ascir/autofuse/common/optimize/codegen/all."
      ;;
  esac
}

main() {
  cd "${BASEPATH}"
  checkopts "$@"
  INITIAL_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"

  export ASCEND_CUSTOM_PATH=${ASCEND_INSTALL_PATH}
  ASCEND_INSTALL_LIB_PATH=${ASCEND_INSTALL_PATH}/$(uname -m)-linux/lib64/
  set_test_ld_library_path

  build_ascgen-dev || { echo "ascgen-dev build failed."; exit 1; }

  METADEF_SOS=$(find ${AUTOFUSE_BUILD_PATH}/graph_metadef -name "*.so" 2>/dev/null)
  for METADEF_SO in ${METADEF_SOS}
  do
    cp -rf ${METADEF_SO} ${METADEF_LIB_PATH}
    echo "cp -rf ${METADEF_SO} ${METADEF_LIB_PATH}"
  done

  if [[ "X$ENABLE_UT" = "Xon" ]]; then
    echo "---------------- ascgen-dev build finished ----------------"
    build_ut || { echo "ut build failed."; exit 1; }
  fi

  if [[ "X$ENABLE_ST" = "Xon" ]]; then
    build_st || { echo "st build failed."; exit 1; }
  fi

  echo "---------------- test finished ----------------"

  if [[ "X$ENABLE_COV" = "Xon" ]]; then
    get_coverage
  fi
}

main "$@"
