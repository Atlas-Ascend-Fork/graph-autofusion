# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Process-level determinism checks independent of Python hash randomisation."""

from __future__ import annotations

import json
import os
import subprocess
import sys


_PROGRAM = r"""
import json
import multistream
from multistream.algorithm.stream_allocator import _path_cover_routes
from common.runtime_stub import install
# Keep the package intact and replace only the Runtime ABI boundary.
install()
graph = multistream.DAGGraph("process-determinism")
nodes = [graph.add_node("n" + str(index), "Op") for index in range(10)]
for src, dst, src_port in ((0, 3, 1), (0, 1, 0), (0, 2, 2), (1, 4, 0),
                           (2, 5, 0), (3, 6, 0), (4, 7, 0), (5, 7, 1),
                           (6, 8, 0), (7, 9, 0), (8, 9, 1)):
    graph.add_edge(nodes[src], src_port, nodes[dst], 0)
nodes[4].set_stream_label("serial-a")
nodes[8].set_stream_label("serial-b")
for index, node in enumerate(nodes):
    if not node.get_stream_label():
        node.set_node_cost(multistream.NodeCost(execution_time=1.0 + index / 10.0,
                                             cube_block_num=index % 3,
                                             vec_block_num=(index + 1) % 2))
result = {"routes": _path_cover_routes(graph), "plans": {}}
for strategy in multistream.MergeStrategy:
    plan = multistream.plan(graph, max_stream_num=4, base_stream_id=7, merge_strategy=strategy)
    result["plans"][strategy.name] = {"ids": [plan.get_stream_id(node) for node in nodes],
                                       "required": plan.get_required_stream_count()}
print(json.dumps(result, sort_keys=True, separators=(",", ":")))
"""


def _run_with_hash_seed(seed: int, cwd):
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(seed)
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    return subprocess.run(
        [sys.executable, "-c", _PROGRAM],
        capture_output=True,
        text=True,
        env=environment,
        cwd=cwd,
        check=True,
    )


def test_routes_and_stream_plans_are_stable_across_python_hash_seeds(tmp_path):
    first = _run_with_hash_seed(1, tmp_path)
    second = _run_with_hash_seed(987654, tmp_path)
    assert json.loads(first.stdout) == json.loads(second.stdout)
