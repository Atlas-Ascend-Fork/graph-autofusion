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

# 将 legacy 对照表路径传给 ElementTypeMatrixTest，用于核对注册的元素类型。
# 测试运行于构建目录，不按当前工作目录猜测数据路径。
target_compile_definitions(AFIRToAscendCTest PRIVATE
  AF_MLIR_OP_LOWERING_MATRIX_CSV="${MLIR_CODE_ROOT_DIR}/docs/guides/ascir_op_lowering_matrix.csv"
)
