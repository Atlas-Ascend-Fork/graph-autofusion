// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

//===- AFIROps.td - AFIR operation definitions -------------*- tablegen -*-===//

#ifndef AFIR_OPS
#define AFIR_OPS

include "AFIRDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "AFIROpBase.td"

//===----------------------------------------------------------------------===//
// Basic Operations
//===----------------------------------------------------------------------===//

include "Ops/Data.td"
include "Ops/Elementwise.td"
include "Ops/Broadcast.td"
include "Ops/Cast.td"
include "Ops/Condition.td"
include "Ops/SymbolicEvalOp.td"
include "Ops/Execution.td"
include "Ops/Synchronization.td"
include "Ops/VectorFunc.td"
include "Ops/Tiling.td"

//===----------------------------------------------------------------------===//
// Advanced Operations
//===----------------------------------------------------------------------===//

include "Ops/Concat.td"
include "Ops/Layout.td"
include "Ops/Reduce.td"

#endif // AFIR_OPS
