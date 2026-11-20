#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail
if [[ $# != 2 ]]; then
    echo "Usage: bash collect.sh <build-dir> <new-run-dir>" >&2
    exit 2
fi
src=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build=$(cd -- "$1" && pwd)
mkdir -- "$2"
run=$(cd -- "$2" && pwd)
# msprof parses --application as a command line. Require simple paths.
for dir in "$build" "$run"; do
    if [[ ! "$dir" =~ ^[[:alnum:]_./-]+$ ]]; then
        echo "Use build/run paths containing only letters, digits, _, ., / or -" >&2
        exit 2
    fi
done
for bench in bridge head; do
    test -x "$build/$bench"
    mkdir "$run/$bench"
    for round in p1 p2 p3 p4; do
        order=forward
        if [[ "$round" == p2 || "$round" == p3 ]]; then
            order=reverse
        fi
        batch="$run/$bench/$round"
        msprof --output="$batch" --application="$build/$bench $batch.jsonl $order" \
            --aic-mode=task-based --aic-metrics=PipeUtilization > "$batch.log" 2>&1
    done
done
python3 "$src/calibrate.py" collect "$run" > "$run/samples.csv"
