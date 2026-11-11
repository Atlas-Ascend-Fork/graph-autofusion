/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AUTOFUSE_MLIR_DIALECT_AFIR_IR_PLANMETADATA_H
#define AUTOFUSE_MLIR_DIALECT_AFIR_IR_PLANMETADATA_H

#include "dialect/AFIR/IR/AFIRAttributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::afir {

// Module kind: which of the two AFIR module shapes this is. kPlanModuleKind and
// kEntryModuleKind are defined in AFIRAttributes.h.
//
// 候选身份由 afplan 管理，见 dialect/AFPlan/IR/AFPlan.h 的 getCandidate。
// module kind 决定生成的 module 形态，即使只有一个候选也需要，因此属于 afir 公开信息
// （设计文档 2.3）。
bool isPlanModule(ModuleOp module);
bool isEntryModule(ModuleOp module);

}  // namespace mlir::afir

#endif  // AUTOFUSE_MLIR_DIALECT_AFIR_IR_PLANMETADATA_H
