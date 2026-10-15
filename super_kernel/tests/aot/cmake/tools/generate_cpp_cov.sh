#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

CUR_DIR="$(cd "$(dirname "$0")" && pwd)"

source ${CUR_DIR}/util.sh
BASEPATH="$(cd "$(dirname "$0")/../../../../.." && pwd)"
source ${BASEPATH}/scripts/support_multiple_versions_of_lcov.sh

set -e

LCOV_VERSION=""
LCOV_COMPAT_MODE=""
LCOV_MAJOR_VERSION=0
LCOV_CAPTURE_OPTS=()
LCOV_FILTER_OPTS=()
LCOV_PARALLEL_PARAMS=""
LCOV_RC_PARAM=""
GENHTML_IGNORE_ERRORS=""
CPP_UT_COVERAGE_EXPECTED="${CPP_UT_COVERAGE_EXPECTED:-80}"

progress() {
  local _current="$1"
  local _total="$2"
  local _message="$3"
  logging "[${_current}/${_total}] ${_message}"
}

init_lcov_compat_options() {
  LCOV_VERSION="$(lcov --version 2>/dev/null | awk '{print $NF}')"
  LCOV_MAJOR_VERSION="$(printf '%s\n' "${LCOV_VERSION}" | awk -F. '{print $1}')"
  if [[ -z "${LCOV_MAJOR_VERSION}" || ! "${LCOV_MAJOR_VERSION}" =~ ^[0-9]+$ ]]; then
    LCOV_MAJOR_VERSION=0
  fi

  LCOV_CAPTURE_OPTS=()
  LCOV_FILTER_OPTS=()
  LCOV_PARALLEL_PARAMS=""
  LCOV_RC_PARAM=""
  GENHTML_IGNORE_ERRORS=""
  if (( LCOV_MAJOR_VERSION >= 2 )); then
    LCOV_COMPAT_MODE="modern"
    LCOV_PARALLEL_PARAMS=$(get_lcov_parallel_params ${THREAD_NUMBER:-${THREAD_NUM:-8}})
    LCOV_RC_PARAM=$(get_lcov_unexecuted_blocks_param)
    GENHTML_IGNORE_ERRORS=$(get_genhtml_ignore_errors)

    if [ -n "${LCOV_PARALLEL_PARAMS}" ]; then
      LCOV_CAPTURE_OPTS=(
        --quiet
        --ignore-errors child,inconsistent,negative,mismatch,empty
        ${LCOV_RC_PARAM}
        ${LCOV_PARALLEL_PARAMS}
      )
    else
      LCOV_CAPTURE_OPTS=(
        --quiet
        --ignore-errors inconsistent,negative,mismatch,empty
        ${LCOV_RC_PARAM}
      )
    fi
    LCOV_FILTER_OPTS=(
      --quiet
      --ignore-errors inconsistent,negative,mismatch,empty,unused
      ${LCOV_RC_PARAM}
      ${LCOV_PARALLEL_PARAMS}
    )
  else
    LCOV_COMPAT_MODE="legacy"
    LCOV_CAPTURE_OPTS=(
      --quiet
      --ignore-errors gcov
    )
    LCOV_FILTER_OPTS=(
      --quiet
    )
  fi

  logging "Detected lcov version: ${LCOV_VERSION:-unknown}, compatibility mode: ${LCOV_COMPAT_MODE}"
}

# using lcov to generate coverage for cpp files
generate_coverage() {
  local _source_dir="$1"
  local _coverage_file="$2"
  local _cann_pkg_path="$3"
  local _source_filter_opts=()
  if [[ -n "${_source_filter}" ]]; then
    _source_filter_opts=(--include "${_source_filter}/*")
  fi

  if [[ -z "${_source_dir}" ]]; then
    logging "directory required to find the .da files"
    exit 1
  fi

  if [[ ! -d "${_source_dir}" ]]; then
    logging "directory is not exist, please check ${_source_dir}"
    exit 1
  fi

  if [[ -z "${_coverage_file}" ]]; then
    _coverage_file="coverage.info"
    logging "using default file name to generate coverage"
  fi

  \which lcov >/dev/null 2>&1
  if [[ $? -ne 0 ]]; then
    logging "lcov is required to generate coverage data, please install"
    exit 1
  fi
  init_lcov_compat_options

  local _path_to_gen="$(dirname ${_coverage_file})"
  if [[ ! -d "${_path_to_gen}" ]]; then
    mk_dir "${_path_to_gen}"
  fi
  lcov -c -d "${_source_dir}" "${LCOV_CAPTURE_OPTS[@]}" "${_source_filter_opts[@]}" -o "${_coverage_file}"
  lcov -r "${_coverage_file}" "${_cann_pkg_path}/*" "/home/jenkins/opensource/*" "${_src}/build/*" "${_src}/build_out/*" "${_src}/output/*" "${_src}/super_kernel/tests/*" "${LCOV_FILTER_OPTS[@]}" -o "${_coverage_file}"
  if [[ -n "${_source_filter}" ]]; then
    lcov --extract "${_coverage_file}" "${_source_filter}/*" "${LCOV_FILTER_OPTS[@]}" -o "${_coverage_file}"
  fi
  logging "generated coverage file ${_coverage_file} ${_src}"
}

# filter out some unused directories or files
filter_coverage() {
  local _coverage_file="$1"
  local _filtered_file="$2"

  if [[ ! -f "${_coverage_file}" ]]; then
    logging "coverage data file required"
    exit 1
  fi

  \which lcov >/dev/null 2>&1
  if [[ $? -ne 0 ]]; then
    logging "lcov is required to generate coverage data, please install"
    exit 1
  fi
  lcov --remove "${_coverage_file}" '/usr/include/*' '/usr/local/include/*' \
         "${LCOV_FILTER_OPTS[@]}" \
         -o "${_filtered_file}"
}

# generate html report
generate_html() {
  local _filtered_file="$1"
  local _out_path="$2"

  \which genhtml >/dev/null 2>&1
  if [[ $? -ne 0 ]]; then
    logging "genhtml is required to generate coverage html report, please install"
    exit 1
  fi

  local _path_to_gen="$(dirname ${_out_path})"
  if [[ ! -d "${_out_path}" ]]; then
    mk_dir "${_out_path}"
  fi
  genhtml "${_filtered_file}" \
          --quiet \
          --legend \
          --title "graph-autofusion AOT C++ Coverage" \
          --prefix "${_src}" \
          ${GENHTML_IGNORE_ERRORS} \
          -o "${_out_path}"
}

print_summary() {
  local _filtered_file="$1"
  local _html_path="$2"

  logging "Coverage tracefile: ${_filtered_file}"
  logging "Coverage html report: ${_html_path}/index.html"
  lcov --summary "${_filtered_file}" | sed 's/^/[INFO] /'
}

print_low_coverage_files() {
  local _filtered_file="$1"
  local _threshold="$2"
  local _under_threshold

  _under_threshold="$(
    lcov --list "${_filtered_file}" | awk -F'|' -v threshold="${_threshold}" '
      /^[[:space:]]*Filename[[:space:]]*\|/ { next }
      /^=+/ { next }
      /^[[:space:]]*Total:/ { next }
      NF >= 2 {
        file = $1
        rate = $2
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", file)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", rate)
        sub(/%.*/, "", rate)
        if (file != "" && rate != "" && rate + 0 < threshold) {
          printf "%s %.1f%%\n", file, rate + 0
        }
      }
    '
  )"

  if [[ -z "${_under_threshold}" ]]; then
    logging "All files meet the expected line coverage threshold (${_threshold}%)."
    return
  fi

  logging "Files below expected line coverage threshold (${_threshold}%):"
  while IFS= read -r _line; do
    [[ -z "${_line}" ]] && continue
    logging "  ${_line}"
  done <<< "${_under_threshold}"
}


if [[ $# -lt 4 || $# -gt 6 ]]; then
  logging "Usage: $0 DIR COV_FILE OUT_PATH CANN_PATH [CAPTURE_DIR [SOURCE_FILTER]]"
  exit 0
fi

_src="$1"
_cov_file="$2"
_out="$3"
_cann_path="$4"
_capture_dir="${5:-${_src}}"
_source_filter="${6:-}"

progress 1 4 "Capturing AOT C++ coverage data"
generate_coverage "${_capture_dir}" "${_cov_file}" "${_cann_path}"
progress 2 4 "Filtering AOT C++ coverage data"
filter_coverage   "${_cov_file}" "${_cov_file}_filtered"
progress 3 4 "Generating AOT C++ coverage html report"
generate_html     "${_cov_file}_filtered" "${_out}"
progress 4 4 "Summarizing AOT C++ coverage results"
print_summary     "${_cov_file}_filtered" "${_out}"
print_low_coverage_files "${_cov_file}_filtered" "${CPP_UT_COVERAGE_EXPECTED}"
