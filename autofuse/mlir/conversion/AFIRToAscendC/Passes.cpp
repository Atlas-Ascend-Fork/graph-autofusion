/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/AFIRToAscendC/AFIRToAscendCPasses.h"

namespace mlir::ascend::kernel {

void registerAFIRToAscendCPasses() {
  // Registration order mirrors the real execution order of the AFIR-to-AscendC
  // pipeline (AFIRToAscendCPipeline.cpp's addAFIRToAscendCPipeline and
  // addPlanCandidateLoweringPipeline): per-stage passes first, then the two
  // public pipelines that compose them, followed by internal-precondition and
  // diagnostic passes.
  registerAFIRKernelBodySignaturePass();
  registerAscendCLoopGeneratePass();
  registerAFIRMemoryAllocatePass();
  registerAFIRSelectImplementationStandalonePass();
  registerConvertAFIRToAscendCStandalonePass();
  registerAFIRVectorFuncToRegBasePass();
  registerGenTilingDataPass();
  registerGenKernelEntryPass();
  registerAFIRToAscendCPipelineStandalonePass();
  registerAFIRPlanCandidateLoweringPipelineStandalonePass();
  registerAFIRPipelineSyncPass();
  registerAFIRVerifyKernelCompletePass();
}

}  // namespace mlir::ascend::kernel
