# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 覆盖 T4：load -> brc -> brc -> store，源形状需含两根广播轴。
#
# [1,1,8] -> [4,4,8]：每次广播扩展一根轴，避免第二次广播退化为空操作。
#
# 采用探针已验证的 f32 双广播轴形状。2026-10-12 的 f16 探针为不支持，
# 因此本注册不声明 f16 覆盖。
add_mlir_e2e_cpu_case(load_broadcast_chain_store_1x1x8
    OUTPUT_SHAPE "[4, 4, 8]"
    INPUT_SHAPES "[1, 1, 8]"
)
