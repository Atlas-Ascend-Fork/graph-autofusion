/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_BASIC_OP_VEC_BINARY_TD
#define ASC_BASIC_OP_VEC_BINARY_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

defm Add : BinaryTemplateL0123Op<"add", "Add", "operator+">;
defm AddDeqRelu : BinaryCastL012Op<"add_deq_relu", "AddDeqRelu">;
defm AddRelu : BinaryTemplateL012Op<"add_relu", "AddRelu">;
defm And : BinaryTemplateL012Op<"and", "And">;
defm Div : BinaryTemplateL0123Op<"div", "Div", "operator/">;
def FusedAbsSubL2Op : BinaryL2Op<"fused_abs_sub_l2", "FusedAbsSub">;
def FusedExpSubL2Op : BinaryL2Op<"fused_exp_sub_l2", "FusedExpSub">;
defm FusedMulAdd : BinaryTemplateL012Op<"fused_mul_add", "FusedMulAdd">;
defm FusedMulAddRelu : BinaryTemplateL012Op<"fused_mul_add_relu", "FusedMulAddRelu">;
defm Max : BinaryTemplateL012Op<"max", "Max">;
defm Min : BinaryTemplateL012Op<"min", "Min">;
defm Mul : BinaryTemplateL0123Op<"mul", "Mul", "operator*">;
defm MulAddDst : BinaryCastL012Op<"mul_add_dst", "MulAddDst">;
defm Or : BinaryTemplateL012Op<"or", "Or">;
def PreluL2Op : BinaryL2Op<"prelu_l2", "Prelu">;
defm Sub : BinaryTemplateL0123Op<"sub", "Sub", "operator-">;
defm SubRelu : BinaryTemplateL012Op<"sub_relu", "SubRelu">;

#endif //ASC_BASIC_OP_VEC_BINARY_TD
