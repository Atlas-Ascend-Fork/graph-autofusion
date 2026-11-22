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

usage() {
    cat <<'EOF'
Usage:
  bash mlir/docker/prepare_cann_volume.sh --volume <volume> --toolkit-url <cann-toolkit-runfile-url>
  bash mlir/docker/prepare_cann_volume.sh <cann-toolkit-runfile-url>

Environment:
  AF_CANN_VOLUME               Docker volume name, default cann-volume
  AF_CANN_PACKAGE_URL          CANN Toolkit .run URL or local file, used when no positional URL is provided
  AF_CANN_TOOLKIT_SHA256       Optional sha256 for the Toolkit .run
  AF_CANN_OPS_PACKAGE_URL      Optional CANN ops .run URL or local file
  AF_CANN_OPS_SHA256           Optional sha256 for the ops .run
  AF_CANN_VOLUME_IMAGE         Temporary installer image, default AF_MLIR_DEV_IMAGE or autofuse-mlir-dev:<arch>
  AF_CANN_ARCH_DIR             Toolkit arch dir, default inferred from host arch
  AF_CANN_TOOLKIT_INSTALL_ARGS Toolkit installer args, default "--install --quiet"
  AF_CANN_OPS_INSTALL_ARGS     Ops installer args, default "--install --quiet"
  AF_CANN_VOLUME_FORCE         Set to 1 to overwrite a non-empty volume

Options:
  --volume <name>              Override AF_CANN_VOLUME
  --toolkit-url <url-or-file>  Override AF_CANN_PACKAGE_URL
  --ops-url <url-or-file>      Override AF_CANN_OPS_PACKAGE_URL
  --image <image>              Override AF_CANN_VOLUME_IMAGE
  --force                      Overwrite a non-empty volume
  -h, --help                   Show this help
EOF
}

check_sha256() {
    local expected="$1"
    local file="$2"
    if command -v sha256sum >/dev/null 2>&1; then
        echo "${expected}  ${file}" | sha256sum -c -
    elif command -v shasum >/dev/null 2>&1; then
        echo "${expected}  ${file}" | shasum -a 256 -c -
    else
        echo "ERROR: neither sha256sum nor shasum is available for checksum verification" >&2
        exit 2
    fi
}

download_artifact() {
    local source="$1"
    local output="$2"
    case "${source}" in
        http://* | https://* | file://*)
            curl -fL --retry 3 --retry-delay 2 -o "${output}" "${source}"
            ;;
        *)
            if [ ! -f "${source}" ]; then
                echo "ERROR: artifact is not a URL or local file: ${source}" >&2
                exit 2
            fi
            cp "${source}" "${output}"
            ;;
    esac
}

require_runfile_ref() {
    local label="$1"
    local source="$2"
    local path="${source%%\?*}"
    local name
    name="$(basename "${path}")"
    if [[ "${name}" != *.run ]]; then
        echo "ERROR: ${label} must resolve to a .run file: ${source}" >&2
        exit 2
    fi
}

host_arch="$(uname -m)"
volume="${AF_CANN_VOLUME:-cann-volume}"
toolkit_url="${AF_CANN_PACKAGE_URL:-}"
ops_url="${AF_CANN_OPS_PACKAGE_URL:-}"
force="${AF_CANN_VOLUME_FORCE:-0}"
toolkit_install_args="${AF_CANN_TOOLKIT_INSTALL_ARGS:---install --quiet}"
ops_install_args="${AF_CANN_OPS_INSTALL_ARGS:---install --quiet}"

case "${host_arch}" in
    arm64 | aarch64)
        default_cann_arch_dir=aarch64-linux
        ;;
    x86_64 | amd64)
        default_cann_arch_dir=x86_64-linux
        ;;
    *)
        echo "ERROR: unsupported host arch for default CANN arch dir: ${host_arch}" >&2
        echo "Set AF_CANN_ARCH_DIR explicitly." >&2
        exit 2
        ;;
esac
case "${host_arch}" in
    arm64 | aarch64)
        image_arch=arm64
        ;;
    amd64 | x86_64)
        image_arch=amd64
        ;;
esac
installer_image="${AF_CANN_VOLUME_IMAGE:-${AF_MLIR_DEV_IMAGE:-autofuse-mlir-dev:${image_arch}}}"
cann_arch_dir="${AF_CANN_ARCH_DIR:-${default_cann_arch_dir}}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        -h | --help)
            usage
            exit 0
            ;;
        --volume)
            if [ "$#" -lt 2 ]; then
                echo "ERROR: --volume requires a value" >&2
                exit 2
            fi
            volume="$2"
            shift 2
            ;;
        --toolkit-url)
            if [ "$#" -lt 2 ]; then
                echo "ERROR: --toolkit-url requires a value" >&2
                exit 2
            fi
            toolkit_url="$2"
            shift 2
            ;;
        --ops-url)
            if [ "$#" -lt 2 ]; then
                echo "ERROR: --ops-url requires a value" >&2
                exit 2
            fi
            ops_url="$2"
            shift 2
            ;;
        --image)
            if [ "$#" -lt 2 ]; then
                echo "ERROR: --image requires a value" >&2
                exit 2
            fi
            installer_image="$2"
            shift 2
            ;;
        --force)
            force=1
            shift
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            if [ -z "${toolkit_url}" ]; then
                toolkit_url="$1"
                shift
            else
                echo "ERROR: unexpected argument: $1" >&2
                usage >&2
                exit 2
            fi
            ;;
    esac
done

if [ "$#" -gt 0 ]; then
    echo "ERROR: unexpected argument: $1" >&2
    usage >&2
    exit 2
fi

if [ -z "${volume}" ]; then
    echo "ERROR: AF_CANN_VOLUME or --volume must not be empty" >&2
    exit 2
fi

if [ -z "${toolkit_url}" ]; then
    usage >&2
    exit 2
fi

require_runfile_ref "CANN Toolkit package" "${toolkit_url}"
if [ -n "${ops_url}" ]; then
    require_runfile_ref "CANN ops package" "${ops_url}"
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker is required to prepare the CANN volume" >&2
    exit 1
fi

if ! docker volume inspect "${volume}" >/dev/null 2>&1; then
    docker volume create "${volume}" >/dev/null
fi

tmpdir="$(mktemp -d)"
cleanup() {
    rm -rf "${tmpdir}"
}
trap cleanup EXIT

mkdir -p "${tmpdir}/cann"
download_artifact "${toolkit_url}" "${tmpdir}/cann/toolkit.run"

if [ -n "${AF_CANN_TOOLKIT_SHA256:-}" ]; then
    check_sha256 "${AF_CANN_TOOLKIT_SHA256}" "${tmpdir}/cann/toolkit.run"
fi

if [ -n "${ops_url}" ]; then
    download_artifact "${ops_url}" "${tmpdir}/cann/ops.run"
    if [ -n "${AF_CANN_OPS_SHA256:-}" ]; then
        check_sha256 "${AF_CANN_OPS_SHA256}" "${tmpdir}/cann/ops.run"
    fi
fi

echo "Preparing CANN volume ${volume} with installer image ${installer_image}" >&2

docker run --rm -i \
    -v "${tmpdir}/cann:/tmp/cann:ro" \
    -v "${volume}:/opt/cann-volume" \
    -e AF_CANN_VOLUME_FORCE="${force}" \
    -e CANN_ARCH_DIR="${cann_arch_dir}" \
    -e CANN_TOOLKIT_INSTALL_ARGS="${toolkit_install_args}" \
    -e CANN_OPS_INSTALL_ARGS="${ops_install_args}" \
    "${installer_image}" \
    bash -s <<'EOF'
set -euo pipefail

install_root=/tmp/Ascend
volume_root=/opt/cann-volume

if find "${volume_root}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    if [ "${AF_CANN_VOLUME_FORCE}" != "1" ]; then
        echo "ERROR: CANN volume is not empty. Re-run with --force or AF_CANN_VOLUME_FORCE=1." >&2
        exit 2
    fi
    find "${volume_root}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
fi

rm -rf "${install_root}"
mkdir -p "${install_root}"
tmp_cann="$(mktemp -d)"
cleanup() {
    rm -rf "${tmp_cann}" "${install_root}"
}
trap cleanup EXIT

toolkit_runfile="${tmp_cann}/toolkit.run"
cp /tmp/cann/toolkit.run "${toolkit_runfile}"
chmod +x "${toolkit_runfile}"
read -r -a toolkit_install_args <<< "${CANN_TOOLKIT_INSTALL_ARGS}"
"${toolkit_runfile}" "${toolkit_install_args[@]}" --install-path="${install_root}"

if [ -f /tmp/cann/ops.run ]; then
    ops_runfile="${tmp_cann}/ops.run"
    cp /tmp/cann/ops.run "${ops_runfile}"
    chmod +x "${ops_runfile}"
    read -r -a ops_install_args <<< "${CANN_OPS_INSTALL_ARGS}"
    "${ops_runfile}" "${ops_install_args[@]}" --install-path="${install_root}"
fi

if [ ! -d "${install_root}/ascend-toolkit" ]; then
    echo "ERROR: Toolkit installer did not create ${install_root}/ascend-toolkit" >&2
    exit 1
fi

cp -a "${install_root}/ascend-toolkit/." "${volume_root}/"

ascend_home="${volume_root}/latest/${CANN_ARCH_DIR}"
if [ ! -d "${ascend_home}" ]; then
    echo "ERROR: missing Toolkit arch directory: ${ascend_home}" >&2
    exit 1
fi

ln -sfn ../share "${ascend_home}/share"
ln -sfn ../set_env.sh "${ascend_home}/set_env.sh"
mkdir -p "${ascend_home}/compiler/tikcpp"
ln -sfn ../../tikcpp/ascendc_kernel_cmake "${ascend_home}/compiler/tikcpp/ascendc_kernel_cmake"
ln -sfn . "${ascend_home}/${CANN_ARCH_DIR}"
EOF

docker run --rm \
    -v "${volume}:/opt/cann:ro" \
    -e ASCEND_HOME_PATH="/opt/cann/latest/${cann_arch_dir}" \
    "${installer_image}" \
    bash -lc 'test -d "${ASCEND_HOME_PATH}"'

echo "Prepared CANN volume: ${volume}"
echo "Use it with: export AF_CANN_VOLUME=${volume}"
