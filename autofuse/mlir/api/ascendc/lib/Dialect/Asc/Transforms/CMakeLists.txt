# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

add_mlir_dialect_library(MLIRAscTransforms
  DeclarePyStructPass.cpp
  DefineCubeOnlyPass.cpp
  DetectEnableDebugPass.cpp
  DetectKernelTypePass.cpp
  EraseSync.cpp
  GenerateBoilerplatePass.cpp
  HoistQueBind.cpp
  HoistTensorAllocation.cpp
  InputOutputTensor.cpp
  InsertQueSync.cpp
  LegalizeKernelArgs.cpp
  MaterializeTensor.cpp
  Noop.cpp
  PrivatizeFunc.cpp
  UnifyPipe.cpp
  VerifySync.cpp

  ADDITIONAL_HEADER_DIRS
  ${ASCIR_INCLUDE_DIR}/ascir/Dialect/Asc

  DEPENDS
  MLIRAscPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRAsc
  MLIRAscUtils
)
