# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 级联拓扑 T4（`load -> brc -> brc -> store`）。**一格**，而且形状必须带两个 1：
#
#   `[1,1,8] -> [4,4,8]`   两根广播轴，一趟一根。只有一根的话第二趟是空集、图退化成 T1，
#                          这一格就什么都没多测。
#
# 探针在 f16 上报不支持（回落 legacy），所以这里只有 f32；五个形状格里只有这一格有两根
# 广播轴，所以也只有这一格（2026-10-12 实测，见 Case.cpp 文件头）。
add_mlir_e2e_cpu_case(load_broadcast_chain_store_1x1x8
    OUTPUT_SHAPE "[4, 4, 8]"
    INPUT_SHAPES "[1, 1, 8]"
)
