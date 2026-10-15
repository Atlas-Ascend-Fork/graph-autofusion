#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

sk_validate_npu_arch() {
    case "$1" in
        dav-2201|dav-3510)
            return 0
            ;;
        *)
            echo "ERROR: unsupported NPU architecture '$1'; valid values: dav-2201, dav-3510" >&2
            return 2
            ;;
    esac
}

sk_parse_npu_arch() {
    local parsed_args
    NPU_ARCH=""
    parsed_args=$(getopt -a -o h -l help,npu-arch: -- "$@") || return 2
    eval set -- "${parsed_args}"
    while true; do
        case "$1" in
            -h | --help)
                echo "Usage: bash $0 --npu-arch=<dav-2201|dav-3510>"
                exit 0
                ;;
            --npu-arch)
                NPU_ARCH="$2"
                shift 2
                ;;
            --)
                shift
                break
                ;;
        esac
    done
    if [ "$#" -ne 0 ]; then
        echo "ERROR: unexpected arguments: $*" >&2
        return 2
    fi
    if [ -z "${NPU_ARCH}" ]; then
        echo "ERROR: --npu-arch is required." >&2
        return 2
    fi
    sk_validate_npu_arch "${NPU_ARCH}"
}

sk_check_driver_version_for_aot() {
    local driver_root="/usr/local/Ascend/driver"
    local version_file="${driver_root}/version.info"
    local raw_version
    local driver_version
    local major
    local minor
    local download_url="https://www.hiascend.com/hardware/firmware-drivers/old/community"

    if [ ! -d "${driver_root}" ]; then
        echo "[WARNING][AOT-SKIP] driver directory not found: ${driver_root}"
        echo "[WARNING][AOT-SKIP] install or mount an Ascend driver (>= 25.5) before running AOT examples."
        echo "[WARNING][AOT-SKIP] download: ${download_url}"
        return 1
    fi

    if [ ! -f "${version_file}" ] || [ ! -r "${version_file}" ]; then
        echo "[WARNING][AOT-SKIP] driver version is unavailable: ${version_file} is missing or unreadable."
        echo "[WARNING][AOT-SKIP] mount the host version.info into the container before running AOT examples."
        echo "[WARNING][AOT-SKIP] download: ${download_url}"
        return 1
    fi

    raw_version=$(awk -F= '
        tolower($1) ~ /^[[:space:]]*version[[:space:]]*$/ {
            value = $2
            sub(/^[[:space:]]+/, "", value)
            sub(/[[:space:]]+$/, "", value)
            print value
            exit
        }
    ' "${version_file}" 2>/dev/null || true)
    driver_version=$(printf '%s\n' "${raw_version}" | sed -nE \
        's/^[^0-9]*([0-9]+\.[0-9]+(\.[0-9]+)?).*/\1/p')

    if [ -z "${driver_version}" ]; then
        echo "[WARNING][AOT-SKIP] unable to parse driver version from ${version_file}."
        echo "[WARNING][AOT-SKIP] required driver version: >= 25.5"
        echo "[WARNING][AOT-SKIP] download: ${download_url}"
        return 1
    fi

    major="${driver_version%%.*}"
    minor="${driver_version#*.}"
    minor="${minor%%.*}"
    if [[ ! "${major}" =~ ^[0-9]+$ || ! "${minor}" =~ ^[0-9]+$ ]]; then
        echo "[WARNING][AOT-SKIP] unable to parse driver version: ${driver_version}"
        echo "[WARNING][AOT-SKIP] required driver version: >= 25.5"
        echo "[WARNING][AOT-SKIP] download: ${download_url}"
        return 1
    fi
    major=$((10#${major}))
    minor=$((10#${minor}))

    echo "[INFO][AOT-GATE] driver version: ${driver_version} (${version_file})"
    echo "[INFO][AOT-GATE] required driver version: >= 25.5"
    if ((major > 25 || (major == 25 && minor >= 5))); then
        echo "[INFO][AOT-GATE] driver requirement satisfied; running AOT examples."
        return 0
    fi

    echo "[WARNING][AOT-SKIP] driver version ${driver_version} is below the required >= 25.5."
    echo "[WARNING][AOT-SKIP] download: ${download_url}"
    return 1
}

sk_cleanup_local() {
    local output
    for output in log tmp sk_meta kernel_meta profiling static_kernel_compile_outputs \
        aclnn_static_shape_kernel_outputs .static_kernel_records.json; do
        rm -rf -- "${PWD}/${output}" || return 1
    done
    mkdir -p "${PWD}/tmp" "${PWD}/log"
}

sk_run_python_with_log() {
    local script="$1"
    local run_log="$2"
    local python_cmd="${PYTHON_CMD:-python3}"

    mkdir -p "$(dirname "${run_log}")"
    "${python_cmd}" "${script}" 2>&1 | tee "${run_log}"
    return "${PIPESTATUS[0]}"
}

sk_check_static_kernel_outputs() {
    local output_dir="$1"
    local package_requirement="${2:-optional}"
    local error_log
    local run_package

    error_log=$(find "${output_dir}" -type f -name '*_compile_error.log' -print -quit 2>/dev/null)
    run_package=$(find "${output_dir}" -type f -name '*.run' -print -quit 2>/dev/null)
    if [ -n "${error_log}" ] && [ -z "${run_package}" ]; then
        echo "ERROR: static kernel compilation failed; see ${error_log}" >&2
        return 1
    fi
    if [ "${package_requirement}" = "required" ] && [ -z "${run_package}" ]; then
        echo "ERROR: static kernel run package was not generated under ${output_dir}" >&2
        return 1
    fi
}

sk_uninstall_static_kernel_from_log() {
    local run_log="$1"
    "${PYTHON_CMD:-python3}" - "${run_log}" "${PWD}/static_kernel_compile_outputs" <<'PY'
import os
from pathlib import Path
import re
import subprocess
import sys


def cleanup(run_log, output_dir):
    if not run_log.is_file():
        return 0
    candidates = set(re.findall(r"/[^\s]+/uninstall\.sh", run_log.read_text()))
    if not candidates:
        return 0
    cann_home = os.environ.get("ASCEND_HOME_PATH")
    if not cann_home:
        print(
            "WARN: cannot validate uninstall paths without ASCEND_HOME_PATH",
            file=sys.stderr,
        )
        return 1
    install_root = Path(cann_home).resolve() / "opp/static_kernel/ai_core"
    output_root = output_dir.resolve()
    packages = {
        package.stem
        for package in output_dir.rglob("*.run")
        if package.is_file()
        and package.resolve().is_relative_to(output_root)
        and not package.is_symlink()
    }
    failed = False
    for candidate in sorted(candidates):
        script = Path(candidate)
        expected = install_root / script.parent.name / "uninstall.sh"
        if (
            script.parent.name not in packages
            or ".." in script.parts
            or script.is_symlink()
            or script.parent.is_symlink()
            or script.resolve() != expected
            or expected.resolve() != expected
        ):
            print(
                f"WARN: refusing unrelated or redirected uninstall script: {script}",
                file=sys.stderr,
            )
            failed = True
            continue
        if not script.exists():
            continue
        if not script.is_file():
            print(
                f"WARN: uninstall script is not a regular file: {script}",
                file=sys.stderr,
            )
            failed = True
            continue
        result = subprocess.run(["bash", str(script)], check=False)
        if result.returncode:
            print(f"WARN: uninstall failed: {script}", file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(cleanup(Path(sys.argv[1]), Path(sys.argv[2])))
PY
}
