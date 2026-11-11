/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- AscendCLoopGenerate.cpp - AscendC loop-generation adapter ------------===//

#include "conversion/AFIRToAscendC/AFIRToAscendCPasses.h"

#include "dialect/AFIR/Transforms/LoopGenerate.h"
#include "conversion/AFIRToAscendC/AscendCKernelABI.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;
using namespace mlir::afir;

namespace mlir::ascend::kernel {
namespace {

SymbolBindingProvider createAscendCSymbolBindingProvider() {
  SymbolBindingProvider provider;
  provider.materialize = materializeAscendCTilingDataBindings;
  provider.registerDialects = [](DialectRegistry &registry) { registry.insert<emitasc::EmitAscDialect>(); };
  return provider;
}

}  // namespace

std::unique_ptr<Pass> createAscendCLoopGeneratePass() {
  return createAFIRLoopGeneratePass(createAscendCSymbolBindingProvider());
}

void registerAscendCLoopGeneratePass() {
  // Register this stage as a pass, consistent with the other lowering stages.
  // createAFIRLoopGeneratePass requires a SymbolBindingProvider, so PassRegistration<T>
  // cannot default-construct it. registerPass uses an allocator and reads the pass's CLI metadata.
  registerPass([] { return createAscendCLoopGeneratePass(); });
}

}  // namespace mlir::ascend::kernel
