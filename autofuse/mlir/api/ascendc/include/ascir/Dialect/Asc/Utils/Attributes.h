/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_DIALECT_ASC_UTILS_ATTRIBUTES_H
#define ASCIR_DIALECT_ASC_UTILS_ATTRIBUTES_H
#define LITERAL constexpr const char*

namespace mlir {
namespace ascendc {

namespace attr {
LITERAL aicore = "ascendc.aicore";
LITERAL compilationArch = "asc.compilation_arch";
LITERAL emitAsUnsigned = "ascendc.emit_as_unsigned";
LITERAL global = "ascendc.global";
LITERAL enableDebug = "asc.enable_debug";
LITERAL kernelType = "asc.kernel_type";
LITERAL matmulCubeOnly = "asc.matmul_cube_only";
LITERAL memoryConsumed = "asc.memory_consumed";
LITERAL socVersion = "asc.soc_version";
LITERAL vfVecLen = "asc.vf_vec_len";

LITERAL kernelCube = "cube";
LITERAL kernelMixed = "mixed";
LITERAL kernelVector = "vector";
} // namespace attr

} // namespace ascendc
} // namespace mlir

#undef LITERAL
#endif // ASCIR_DIALECT_ASC_UTILS_ATTRIBUTES_H
