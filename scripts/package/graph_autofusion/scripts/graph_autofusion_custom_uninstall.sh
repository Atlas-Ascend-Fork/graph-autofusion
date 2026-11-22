#!/bin/bash
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------------------

curpath=$(dirname $(readlink -f "$0"))
unset PYTHONPATH

common_parse_dir=""
logfile=""
stage=""
is_quiet="n"
hetero_arch="n"

while true; do
    case "$1" in
    --install-path=*)
        pkg_install_path=$(echo "$1" | cut -d"=" -f2-)
        shift
        ;;
    --common-parse-dir=*)
        common_parse_dir=$(echo "$1" | cut -d"=" -f2-)
        shift
        ;;
    --version-dir=*)
        pkg_version_dir=$(echo "$1" | cut -d"=" -f2-)
        shift
        ;;
    --logfile=*)
        logfile=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --stage=*)
        stage=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --quiet=*)
        is_quiet=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --hetero-arch=*)
        hetero_arch=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    -*)
        shift
        ;;
    *)
        break
        ;;
    esac
done

# 写日志
log() {
    local cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
    local log_type="$1"
    local log_msg="$2"
    local log_format="[graph_autofusion] [$cur_date] [$log_type]: $log_msg"
    if [ "$log_type" = "INFO" ]; then
        echo "$log_format"
    elif [ "$log_type" = "WARNING" ]; then
        echo "$log_format"
    elif [ "$log_type" = "ERROR" ]; then
        echo "$log_format"
    elif [ "$log_type" = "DEBUG" ]; then
        echo "$log_format" 1> /dev/null
    fi
    echo "$log_format" >> "$logfile"
}

get_arch_name() {
    local pkg_dir="$1"
    local scene_file="$pkg_dir/scene.info"
    grep '^arch=' $scene_file | cut -d"=" -f2
}

remove_stub_softlink() {
    local ref_dir="$1"
    if [ ! -d "$ref_dir" ]; then
        return
    fi
    local stub_dir="$2"
    if [ ! -d "$stub_dir" ]; then
        return
    fi
    local pwdbak="$(pwd)"
    cd $stub_dir && chmod u+w . && ls -1 "$ref_dir" | xargs --no-run-if-empty rm -rf
    cd $pwdbak
}

recreate_common_stub_softlink() {
    local arch_name="$1"
    local stub_dir="$2"
    if [ ! -d "$stub_dir" ]; then
        return
    fi
    local pwdbak="$(pwd)"
    cd $stub_dir && [ -f "$arch_name/libgraph_af.so" ] && chmod u+w . && ln -s "$arch_name/libgraph_af.so" libgraph_af.so
    cd $pwdbak
}

whl_uninstall_package() {
    local _module="$1"
    local _module_apth="$2"
    local _module_dist_name="${_module//-/_}"
    # Keep each package lookup independent when local and default installs coexist.
    local PYTHONPATH=""
    local PIP_BREAK_SYSTEM_PACKAGES="${PIP_BREAK_SYSTEM_PACKAGES:-}"
    export PYTHONPATH PIP_BREAK_SYSTEM_PACKAGES
    if [ -d "${WHL_INSTALL_DIR_PATH}/${_module}" ] ||
        ls "${WHL_INSTALL_DIR_PATH}/${_module_dist_name}-"*.dist-info >/dev/null 2>&1; then
        PYTHONPATH="${_module_apth}"
        PIP_BREAK_SYSTEM_PACKAGES=1
    elif ! pip3 show "${_module}" >/dev/null 2>&1; then
        # --pylocal is optional: retain the historical default-environment lookup.
        log "WARNING" "${_module} is not exist."
        return 0
    fi

    pip3 uninstall -y "${_module}" > /dev/null 2>&1
    local ret=$?
    if [ $ret -ne 0 ]; then
        log "WARNING" "uninstall ${_module} failed, error code: $ret."
        exit 1
    else
        log "INFO" "${_module} uninstalled successfully!"
    fi
}

remove_rl_soft_link() {
    local _path="${1}"
    if [ -d "${_path}" ]; then
        rm -rf "${_path}"
    fi
}

remove_empty_dir() {
    local _path="${1}"
    if [ -d "${_path}" ]; then
        local is_empty=$(ls "${_path}" | wc -l)
        if [ "$is_empty" -ne 0 ]; then
            log "INFO" "${_path} dir is not empty."
        else
            prev_path=$(dirname "${_path}")
            chmod +w "${prev_path}" > /dev/null 2>&1
            rm -rf "${_path}" > /dev/null 2>&1
        fi
    fi
}

remove_dir() {
    local _path="${1}"
    if [ -d "${_path}" ]; then
        prev_path=$(dirname "${_path}")
        chmod +w "${prev_path}" > /dev/null 2>&1
        rm -rf "${_path}" > /dev/null 2>&1
    fi
}

remove_last_license() {
    if [ -d "$WHL_INSTALL_DIR_PATH" ]; then
        if [ -f "$WHL_INSTALL_DIR_PATH/LICENSE" ]; then
            rm -rf "$WHL_INSTALL_DIR_PATH/LICENSE"
        fi
    fi
}

WHL_INSTALL_DIR_PATH="${common_parse_dir}/python/site-packages"

custom_uninstall() {
    if [ -z "$common_parse_dir/share/info/graph_autofusion" ]; then
        log "ERROR" "ERR_NO:0x0001;ERR_DES:graph_autofusion directory is empty"
        exit 1
    fi

    if [ "$hetero_arch" != "y" ]; then
        local arch_name="$(get_arch_name $common_parse_dir/share/info/graph_autofusion)"
        local ref_dir="$common_parse_dir/share/info/graph_autofusion/lib64/stub/linux/$arch_name"
        remove_stub_softlink "$ref_dir" "$common_parse_dir/share/info/graph_autofusion/lib64/stub"
    else
        local arch_name="$(get_arch_name $common_parse_dir/share/info/graph_autofusion)"
        local ref_dir="$common_parse_dir/share/info/graph_autofusion/lib64/stub/linux/$arch_name"
        remove_stub_softlink "$ref_dir" "$common_parse_dir/share/info/graph_autofusion/lib64/stub"
    fi

    if [ "$hetero_arch" != "y" ]; then
        chmod +w -R "$curpath" 2> /dev/null
        chmod +w -R "${WHL_INSTALL_DIR_PATH}/superkernel" 2> /dev/null
        chmod +w -R "${WHL_INSTALL_DIR_PATH}/superkernel-0.1.0.dist-info" 2> /dev/null
        chmod +w -R "${WHL_INSTALL_DIR_PATH}/multistream" 2> /dev/null
        chmod +w -R "${WHL_INSTALL_DIR_PATH}"/multistream-*.dist-info 2> /dev/null

        log "INFO" "uninstall graph_autofusion python packages begin..."
        whl_uninstall_package "superkernel" "${WHL_INSTALL_DIR_PATH}"
        whl_uninstall_package "multistream" "${WHL_INSTALL_DIR_PATH}"
        log "INFO" "graph_autofusion python packages uninstalled successfully!"
    fi

    remove_empty_dir "${common_parse_dir}/share/info/graph_autofusion/python"

    if [ "$hetero_arch" != "y" ]; then
        if [ -d "${WHL_INSTALL_DIR_PATH}" ]; then
            local python_path=$(dirname "$WHL_INSTALL_DIR_PATH")
            chmod +w "${python_path}"
            remove_last_license
        fi

        remove_empty_dir "${WHL_INSTALL_DIR_PATH}"
        remove_empty_dir "${common_parse_dir}/python"
        local autofuse_python_dir="${WHL_INSTALL_DIR_PATH}/autofuse"
        remove_dir "${autofuse_python_dir}/__pycache__"
    fi

    return 0
}

custom_uninstall
if [ $? -ne 0 ]; then
    exit 1
fi
exit 0
