/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_BASIC_OP_VEC_BINARY_SCALAR_TD
#define ASC_BASIC_OP_VEC_BINARY_SCALAR_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

defm Adds : VecScalarL012Op<"adds", "Adds">;
defm Divs : VecScalarL012Op<"divs", "Divs">;
defm LeakyRelu : VecScalarL012Op<"leaky_relu", "LeakyRelu">;
defm Maxs : VecScalarL012Op<"maxs", "Maxs">;
defm Mins : VecScalarL012Op<"mins", "Mins">;
defm Muls : VecScalarL012Op<"muls", "Muls">;
defm ShiftLeft : VecScalarL012Op<"shift_left", "ShiftLeft">;
defm ShiftRight : VecScalarL012Op<"shift_right", "ShiftRight">;
defm Subs : VecScalarL012Op<"subs", "Subs">;

#endif //ASC_BASIC_OP_VEC_BINARY_SCALAR_TD
