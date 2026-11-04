# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

add_afir_gtest(AFIRToAscendCTest
  SOURCES
    ${MLIR_CODE_ROOT_DIR}/test/ut/conversion/AFIRToAscendCPipelineTest.cpp
    ${MLIR_CODE_ROOT_DIR}/test/ut/conversion/AscendCAttributesTest.cpp
    ${MLIR_CODE_ROOT_DIR}/test/ut/conversion/DeviceTilingSchemaTest.cpp
    ${MLIR_CODE_ROOT_DIR}/test/ut/conversion/KernelBodySignatureTest.cpp
    ${MLIR_CODE_ROOT_DIR}/test/ut/conversion/BinaryElementwiseL2CapabilityTest.cpp
    ${MLIR_CODE_ROOT_DIR}/test/ut/conversion/ElementTypeMatrixTest.cpp
    ${MLIR_CODE_ROOT_DIR}/test/ut/conversion/ElementwiseApiFamilyTest.cpp
    ${MLIR_CODE_ROOT_DIR}/test/ut/conversion/SynchronizationTranslationTest.cpp
    ${MLIR_CODE_ROOT_DIR}/test/ut/conversion/VectorizedLoopsTest.cpp
  LIBS
    AFIRToAscendC
    AfirBuilder
    AfirKernelSource
    AscendInstructionCapability
    AscendTargetProfile
    MLIRFuncDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRParser
    MLIRSCFDialect
    MLIRPass
    MLIRTransformUtils
    ascendc_api_headers
  COMMENT "Running AFIR to AscendC conversion unit tests"
)

# `ElementTypeMatrixTest` 拿注册处的元素类型集合与 legacy 抽出来的对照表对账，所以要知道
# 那张表在哪。路径从 CMake 传进去，不在用例里按相对路径猜——用例的工作目录是构建目录。
target_compile_definitions(AFIRToAscendCTest PRIVATE
  AF_MLIR_OP_LOWERING_MATRIX_CSV="${MLIR_CODE_ROOT_DIR}/docs/guides/ascir_op_lowering_matrix.csv"
)
