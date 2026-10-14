# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Pure Python multistream planning package (MiniDAG model).

The package needs no third-party Python package and ships no extension of
its own.  Calling :func:`plan` does need the host's CANN Runtime shared
library, which is loaded lazily when planning queries the hardware
core counts.
"""

from .algorithm.planning import plan
from .minidag.graph import DAGGraph
from .minidag.node import DAGNode
from .minidag.plan import StreamPlan

# Constants are intentionally exported as attributes but excluded from
# ``__all__``; the public API list contains the seven documented symbols.
from .minidag.types import (  # noqa: F401
    INVALID_STREAM_ID,
    MAX_STREAM_NUM,
    CoreType,
    MergeStrategy,
    NodeCost,
)

__all__ = [
    "DAGGraph",
    "DAGNode",
    "NodeCost",
    "CoreType",
    "StreamPlan",
    "MergeStrategy",
    "plan",
]
