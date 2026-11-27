# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


from pathlib import Path
from statistics import covariance, median, variance
import csv
import collections
import re
import json

p = Path.cwd()
rr = [
    x
    for x in csv.DictReader(next((p / "profile-v2").rglob("op_summary*.csv")).open())
    if "att_vf_dependency" in x["Op Name"]
]
lines = [
    line
    for line in (p / "profile-v2.log").read_text().splitlines()
    if re.fullmatch(r"\d+,\d+,\d+,\d+,\d+", line)
]
assert len(rr) == len(lines) == 1400, (len(rr), len(lines))
q = collections.defaultdict(lambda: collections.defaultdict(list))
for row, line in zip(rr, lines):
    _, mode, reps, calls, sample = map(int, line.split(","))
    if sample > 0:
        freq = float(row["aiv_total_cycles"]) / float(row["aiv_time(us)"])
        q[(reps, mode)][calls].append(freq * float(row["aiv_vec_time(us)"]))
by = collections.defaultdict(dict)
for (reps, mode), times in sorted(q.items()):
    xx = sorted(times)
    yy = [median(times[k]) for k in xx]
    by[mode][reps] = covariance(xx, yy) / variance(xx)


def fit(z):
    keys = [k for k in z if k not in (7, 16, 40, 64, 96)]
    a = covariance(keys, [z[k] for k in keys]) / variance(keys)
    b = median(z[x] - a * x for x in keys)
    err = {x: 100 * abs((a * x + b) / y - 1) for x, y in z.items() if x not in keys}
    return {
        "cycles_per_repeat": a,
        "intercept": b,
        "max_holdout_error_pct": max(err.values()),
    }


for mode, data in by.items():
    print(mode, fit(data))
    if mode:
        print("delta", fit({x: y - by[0][x] for x, y in data.items()}))
print("cycle slope at 64 repeats", {k: v[64] for k, v in by.items()})
(p / "fit.json").write_text(
    json.dumps(
        {"measured_cycles": by, "linear_fits": {k: fit(v) for k, v in by.items()}},
        indent=2,
    )
)
