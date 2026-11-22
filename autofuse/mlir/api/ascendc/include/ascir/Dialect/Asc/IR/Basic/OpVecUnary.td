/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_BASIC_OP_VEC_UNARY_TD
#define ASC_BASIC_OP_VEC_UNARY_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

defm Abs : UnaryL012Op<"abs", "Abs">;
defm Exp : UnaryL012Op<"exp", "Exp">;
defm Ln : UnaryL012Op<"ln", "Ln">;
def NegL2Op : UnaryL2Op<"neg_l2", "Neg">;
defm Not : UnaryL012Op<"not", "Not">;
defm Reciprocal : UnaryL012Op<"reciprocal", "Reciprocal">;
defm Relu : UnaryL012Op<"relu", "Relu">;
defm Rsqrt : UnaryL012Op<"rsqrt", "Rsqrt">;
defm Sqrt : UnaryL012Op<"sqrt", "Sqrt">;

#endif //ASC_BASIC_OP_VEC_UNARY_TD
