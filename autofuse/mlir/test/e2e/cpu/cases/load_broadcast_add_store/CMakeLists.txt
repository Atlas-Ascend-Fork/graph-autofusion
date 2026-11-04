# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 级联拓扑 T3（`add(load(full), brc(load(small)))`）。选定探针实测 ok 的形态 B：
#
#   `[8,1] + [8,128] -> [8,128]`   最内轴是广播轴。`INPUT_SHAPES` 第一个是广播源、
#                                  第二个与输出同形，次序与 `Data` 的 index 对应，调换会让
#                                  `BindBroadcast` 把广播轴定反。
#
# 当前 T3 拓扑表为 6 ok / 2 FAILED / 2 unsupported；详见 Case.cpp 文件头，不要把未过格
# 写成本用例的通过前提。
add_mlir_e2e_cpu_case(load_broadcast_add_store_8x1
    OUTPUT_SHAPE "[8, 128]"
    INPUT_SHAPES "[8, 1]" "[8, 128]"
)
