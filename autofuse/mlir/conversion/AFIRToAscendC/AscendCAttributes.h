/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_ASCENDCATTRIBUTES_H
#define AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_ASCENDCATTRIBUTES_H

#include "dialect/AFIR/IR/AFIR.h"
#include "target/Ascend/AscendTargetProfile.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::ascend::kernel {

// AscendC lowering products.
inline constexpr llvm::StringLiteral kAscendCFusedMainAttrName = "ascendc.fused_main";
inline constexpr llvm::StringLiteral kAscendCKernelBodyTemplatesRelocatedAttrName =
    "ascendc.kernel_body_templates_relocated";
inline constexpr llvm::StringLiteral kAscendCNeedSyncAllAttrName = "ascendc.need_sync_all";
inline constexpr llvm::StringLiteral kAscendCPlanEntriesAttrName = "ascendc.plan_entries";
inline constexpr llvm::StringLiteral kAscendCTilingSchemaAttrName = "ascendc.tiling_schema";
inline constexpr llvm::StringLiteral kAscendCPlanEntryTilingFieldKey = "tiling_fields";
inline constexpr llvm::StringLiteral kAscendCNumInputsAttrName = "ascendc.num_inputs";
inline constexpr llvm::StringLiteral kAscendCExecutionUnitAttrName = "ascendc.unit";
inline constexpr llvm::StringLiteral kAscendCExecutionModelAttrName = "ascendc.execution_model";
inline constexpr llvm::StringLiteral kAscendCInstructionFormAttrName = "ascendc.instruction_form";

// Map the AFIR dialect's memory-space vocabulary onto the target profile's.
// The two enums describe the same physical spaces but belong to layers that
// must not depend on each other, so the translation lives here in conversion.
// Returns failure for a space the target model does not represent.
FailureOr<ascend::MemorySpace> toTargetMemorySpace(afir::Space space);

// Read the memory space of one value from the descriptor at its definition
// (`afir.buffer_desc`). 迁移前这个查询要先选对数组属性再按下标取，选错一侧就会
// 静默拿到另一端的空间；现在它只需要值本身（设计文档 5.1 步 8）。Returns failure
// when the value carries no descriptor, or names a space the target model does
// not represent.
FailureOr<ascend::MemorySpace> getValueMemorySpace(Value value);

FailureOr<ascend::ExecutionSpec> getExecutionSpec(Operation *op);
LogicalResult setExecutionSpec(Operation *op, const ascend::ExecutionSpec &spec);

FailureOr<bool> hasAscendCAiCoreAttr(Operation *op);
void setAscendCAiCore(Operation *op);

FailureOr<IntegerAttr> getAscendCNumInputs(Operation *op);
void setAscendCNumInputs(Operation *op, unsigned inputCount);

FailureOr<bool> hasAscendCGlobalAttr(Operation *op);
void setAscendCGlobal(Operation *op, bool enabled);

}  // namespace mlir::ascend::kernel

#endif  // AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_ASCENDCATTRIBUTES_H
