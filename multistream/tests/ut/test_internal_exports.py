# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Keep declared internal exports importable.

The translated modules retain a few names used by existing GE-side tooling;
an incorrect ``__all__`` is particularly painful because it fails only when a
consumer uses a star import.
"""

import importlib


def test_declared_internal_exports_are_real_symbols():
    for module_name in (
        "multistream.algorithm.merger",
        "multistream.algorithm.profiling",
        "multistream.algorithm.stream_allocator",
        "multistream.algorithm.weighted_merger",
    ):
        module = importlib.import_module(module_name)
        assert all(hasattr(module, name) for name in module.__all__)


def test_star_imports_from_internal_modules_do_not_fail():
    namespaces = {}
    for module_name in (
        "multistream.algorithm.merger",
        "multistream.algorithm.profiling",
        "multistream.algorithm.stream_allocator",
        "multistream.algorithm.weighted_merger",
    ):
        namespace = {}
        exec(f"from {module_name} import *", namespace)
        namespaces[module_name] = namespace
    assert set(namespaces["multistream.algorithm.profiling"]) >= {
        "_Profile",
        "parse",
        "match_profile",
    }
    assert set(namespaces["multistream.algorithm.stream_allocator"]) >= {
        "_StreamAllocConfig",
        "by_path_cover",
        "_is_dag",
        "_split_serial_graph",
    }
    assert set(namespaces["multistream.algorithm.merger"]) >= {
        "StreamMergeOptions",
        "_Merger",
    }
    assert set(namespaces["multistream.algorithm.weighted_merger"]) >= {
        "WeightedStreamMergeOptions",
        "_WeightedMerger",
    }
