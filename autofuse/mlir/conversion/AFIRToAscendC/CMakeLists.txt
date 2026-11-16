# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

add_mlir_library(AFIRToAscendC
  Passes.cpp
  AFIRSessionScopedPass.cpp
  AFIRKernelBodySignature.cpp
  AscendCLoopGenerate.cpp
  AFIRMemoryAllocate.cpp
  MemoryAllocate/CanonicalResources.cpp
  MemoryAllocate/CanonicalMaterialization.cpp
  MemoryAllocate/CanonicalRewrite.cpp
  AFIRSelectImplementation.cpp
  AFIRToAscendC.cpp
  AFIRVectorFuncToRegBase.cpp
  GenTilingData.cpp
  GenKernelEntry.cpp
  AFIRToAscendCPipeline.cpp
  AFIRPipelineSync.cpp
  AFIRScalarReadSync.cpp
  AFIRVerifyKernelCompletePass.cpp
  LoweringStage.cpp
  AscendCAttributes.cpp
  AscendCKernelABI.cpp
  TilingSchema.cpp
  VectorizedLoops.cpp
  Implementations/BinaryElementwise.cpp
  Implementations/Broadcast.cpp
  Implementations/ConstBlockTensor.cpp
  Implementations/ElementwiseApiFamily.cpp
  Implementations/ElementwiseLocalView.cpp
  Implementations/DataMovement.cpp
  Implementations/Execution.cpp
  Implementations/Nddma.cpp
  Implementations/Pad.cpp
  Implementations/RemovePad.cpp
  Implementations/Reduce.cpp
  Implementations/ReducePadding.cpp
  Implementations/ReduceGeometry.cpp
  Implementations/Synchronization.cpp
  Implementations/TernaryElementwise.cpp
  Implementations/TilingParam.cpp
  Implementations/Transpose.cpp
  Implementations/UnaryElementwise.cpp
  RegisterAscendCImplementations.cpp

  LINK_LIBS PUBLIC
  AFIRAnalysis
  AFIRDialect
  AFPlanDialect
  AFIRTransforms
  AscendInstructionCapability
  AscendTargetProfile
  MLIRArithDialect
  MLIRAsc
  MLIRBufferizationDialect
  MLIRControlFlowDialect
  MLIREmitAsc
  MLIREmitCDialect
  MLIRFuncDialect
  MLIRIR
  MLIRIndexDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransformUtils
)

target_include_directories(AFIRToAscendC PUBLIC
  ${MLIR_CODE_ROOT_DIR}
)
target_link_libraries(AFIRToAscendC PRIVATE
  ascendc_api_headers
)
