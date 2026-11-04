/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Passes.h - AFIR to AscendC conversion passes -------------*- C++ -*-===//
//
// Single declaration point for every pass in this conversion. Adding a pass
// means adding it here and to registerAFIRToAscendCPasses() in Passes.cpp,
// not creating another per-pass header.
//
// Naming: passes that resolve their own target/registry from af-opt's
// --target/--target-profile flags (see AFIRSessionScopedPass.h) carry the
// `Standalone` suffix; the unsuffixed factory of the same pass takes those
// dependencies as arguments.
//
//===----------------------------------------------------------------------===//

#ifndef AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_PASSES_H
#define AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_PASSES_H

#include <memory>

namespace mlir {
class Pass;
}  // namespace mlir

namespace mlir::ascend {
class TargetProfile;
namespace instruction {
class ImplementationRegistry;
}  // namespace instruction
}  // namespace mlir::ascend

namespace mlir::ascend::kernel {

//===----------------------------------------------------------------------===//
// Per-stage passes, declared in pipeline execution order
//===----------------------------------------------------------------------===//

// Prepare the function boundary consumed by later AFIR lowering passes for
// functions carrying afir.function_role.
// Tensor inputs become GM memref inputs, tensor results become GM memref output
// arguments, and workspace/TilingData arguments are appended. The AFIR body
// remains tensor based and is connected to the new boundary with bufferization
// marker operations. Canonical buffer form instead preserves the external
// input/output destinations and requires GM descriptors for every argument.
// Materialize before creating callers. A complete existing ABI is validated
// and retained; any failure leaves the input module unchanged.
std::unique_ptr<Pass> createAFIRKernelBodySignaturePass();
void registerAFIRKernelBodySignaturePass();

// AFIR loop generation bound to the AscendC TilingData ABI: supplies
// dialect/AFIR/Transforms' target-neutral AFIRLoopGeneratePass with the
// symbol bindings materialized from the canonical TilingData argument.
std::unique_ptr<Pass> createAscendCLoopGeneratePass();
void registerAscendCLoopGeneratePass();

std::unique_ptr<Pass> createAFIRMemoryAllocatePass();
void registerAFIRMemoryAllocatePass();

// Standalone entry point for ascend::instruction::createSelectImplementationPass:
// resolves the frozen ImplementationRegistry/SelectionPolicyRegistry for the
// af-opt-selected target from within the pass itself.
std::unique_ptr<Pass> createAFIRSelectImplementationStandalonePass();
void registerAFIRSelectImplementationStandalonePass();

// 相位 C.5：`afir.vector_func` -> `emitasc.vec_scope { emitasc.vf_for { 寄存器算子 } }`。
// 必须跑在实现选择与 `ConvertAFIRToAscendC` **之后**（region 外的算子要先降完，region 内
// 的算子由它自己按 `regOpMappings()` 那张表降），也必须在 host 产物那一相位**之前**。
// 这一期收不了的 VF 形态在它入口报能力不足、整图干净回落。
std::unique_ptr<Pass> createAFIRVectorFuncToRegBasePass();
void registerAFIRVectorFuncToRegBasePass();

std::unique_ptr<Pass> createConvertAFIRToAscendCPass(
    std::shared_ptr<const ascend::instruction::ImplementationRegistry> registry,
    std::shared_ptr<const ascend::TargetProfile> target);
// Standalone counterpart of createConvertAFIRToAscendCPass: resolves the frozen
// ImplementationRegistry/TargetProfile for the af-opt-selected target itself.
std::unique_ptr<Pass> createConvertAFIRToAscendCStandalonePass();
void registerConvertAFIRToAscendCStandalonePass();

std::unique_ptr<Pass> createGenTilingDataPass();
void registerGenTilingDataPass();

std::unique_ptr<Pass> createGenKernelEntryPass();
void registerGenKernelEntryPass();

//===----------------------------------------------------------------------===//
// Composite pipelines
//===----------------------------------------------------------------------===//

// Standalone, session-scoped entry points for addAFIRToAscendCPipeline and
// addPlanCandidateLoweringPipeline (declared in AFIRToAscendCPipeline.h):
// resolve the frozen session for the af-opt-selected target from within the
// pass itself, so the pipelines can be registered and driven like any other
// standard pass.
std::unique_ptr<Pass> createAFIRToAscendCPipelineStandalonePass();
void registerAFIRToAscendCPipelineStandalonePass();

std::unique_ptr<Pass> createAFIRPlanCandidateLoweringPipelineStandalonePass();
void registerAFIRPlanCandidateLoweringPipelineStandalonePass();

//===----------------------------------------------------------------------===//
// Internal preconditions and diagnostics
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createAFIRPipelineSyncPass();
void registerAFIRPipelineSyncPass();

// 设计文档 5.2 的四条完备性判据，逐条可选（`--afir-verify-kernel-complete=criteria=c1,c3`）。
// 判据的定义与收窄方式见 dialect/AFIR/Transforms/VerifyKernelComplete.h；默认只跑 C3，理由见 pass 实现处。
std::unique_ptr<Pass> createAFIRVerifyKernelCompletePass();
void registerAFIRVerifyKernelCompletePass();

// Registers every pass above, in the order they appear in the real pipeline.
void registerAFIRToAscendCPasses();

}  // namespace mlir::ascend::kernel

#endif  // AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_PASSES_H
