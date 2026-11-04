/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_DIALECT_AFIR_TRANSFORMS_PASSES_H
#define AUTOFUSE_MLIR_DIALECT_AFIR_TRANSFORMS_PASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace afir {

std::unique_ptr<Pass> createAFIRSimplifySymbolicPass();
std::unique_ptr<Pass> createAFIRVerifySymbolicSafetyPass();
std::unique_ptr<Pass> createAFIRLowerSymbolicPass();
std::unique_ptr<Pass> createAFIRVerifyNoSymbolicEvalPass();
void registerAFIRSymbolicPasses();

// 尺寸的 SSA 化：把函数里每一个运行期尺寸符号物化成一个 `afir.tiling_param` 值。
// 排在 `AFIRKernelBodySignature` 之后、循环生成之前；细节见 TilingParams.h。
std::unique_ptr<Pass> createAFIRMaterializeTilingParamsPass();
void registerAFIRMaterializeTilingParamsPass();

// 缓存复用周期（`afir.sched` 的 `exec_condition`）→ 循环体里的 `scf.if`。排在资源物化与
// 同步之后、实现选择之前；理由见 ExecConditionPass.cpp 顶部那段。
std::unique_ptr<Pass> createAFIRExecConditionPass();
void registerAFIRExecConditionPass();

// 竞争检查（设计文档 5.1 步 8b）。覆盖在 UT 里，这个 pass 只是 af-opt 上的入口。
std::unique_ptr<Pass> createAFIRRaceCheckPass();
void registerAFIRRaceCheckPass();

// 循环生成后分析 Reduce scratch 使用作用域，不改变物理资源生命周期。
std::unique_ptr<Pass> createAFIRScopeReduceScratchPass();
void registerAFIRScopeReduceScratchPass();

}  // namespace afir
}  // namespace mlir

#endif  // AUTOFUSE_MLIR_DIALECT_AFIR_TRANSFORMS_PASSES_H
