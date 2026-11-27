#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail
if [[ $# != 6 ]]; then
    echo "Usage: bash collect.sh <steady|transpose_pipeline|loop|stability|nddma|nddma_full_store|nddma_long|store|store_compact_loop|wide_ng2> <build-dir> <factors.csv> <mode> <device-lock> <new-run-dir>" >&2
    exit 2
fi
case "$1:$4" in
    steady:[NC]|transpose_pipeline:[NC]|loop:[NC]|stability:[NC]|nddma:[NC]|nddma_full_store|nddma_long:[NC]) ;;
    store:normal|store:bypass|store_compact_loop:normal|store_compact_loop:bypass|wide_ng2:normal|wide_ng2:bypass) ;;
    *) echo "Load mode must be N or C. Store mode must be normal or bypass." >&2; exit 2 ;;
esac
build=$(realpath -- "$2")
factors=$(realpath -- "$3")
test -f "$factors"
mode=$4
case "$1" in
    store|store_compact_loop|wide_ng2) exe="$build/att_dma_$1" ;;
    nddma_full_store) exe="$build/att_dma_load_nddma" ;;
    *) exe="$build/att_dma_load_$1" ;;
esac
test -x "$exe"
run=$(realpath -m -- "$6")
for path in "$exe" "$factors" "$run"; do
    if [[ ! "$path" =~ ^/[[:alnum:]_./-]+$ ]]; then
        echo "Use simple paths for msprof --application" >&2
        exit 2
    fi
done
mkdir -- "$run"
sha256sum -- "$exe" "$factors" > "$run/input.sha256"
exec 9>>"$5"
flock -x 9
if [[ "$1" == store || "$1" == store_compact_loop || "$1" == wide_ng2 ]]; then
    msprof --application="$exe $factors $run/launches.jsonl $mode" \
        --output="$run/profile" --aic-mode=task-based --aic-metrics=PipeUtilization >"$run/profile.log" 2>&1
    expected=$(awk 'END {print NR-1}' "$factors")
    test "$(grep -c '^CASE_PASS ' "$run/profile.log")" -eq "$expected"
else
    consumer=tail_store
    if [[ "$1" == nddma_full_store ]]; then
        consumer=span_store
    elif [[ "$1" == transpose_pipeline ]]; then
        consumer=transpose_store
    fi
    "$exe" "$factors" "$run/verify.jsonl" verify "$consumer" "$mode" >"$run/verify.log" 2>&1
    expected=$(awk 'END {print NR-1}' "$factors")
    test "$(grep -c '^PASS ' "$run/verify.log")" -eq "$expected"
    msprof --application="$exe $factors $run/launches.jsonl profile $consumer $mode" \
        --output="$run/profile" --aic-mode=task-based --aic-metrics=PipeUtilization >"$run/profile.log" 2>&1
    test "$(grep -c '^PASS ' "$run/profile.log")" -eq "$expected"
fi
