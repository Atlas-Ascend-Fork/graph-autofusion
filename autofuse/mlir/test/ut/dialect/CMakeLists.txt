# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

add_afir_gtest(AFIRDialectTest
  SOURCES
    ${MLIR_CODE_ROOT_DIR}/test/ut/dialect/AFIRElementwiseContractTest.cpp
    ${MLIR_CODE_ROOT_DIR}/test/ut/dialect/AFIRLayeringTest.cpp
    ${MLIR_CODE_ROOT_DIR}/test/ut/dialect/AFIRLayoutTest.cpp
    ${MLIR_CODE_ROOT_DIR}/test/ut/dialect/AFIRMemorySpaceTest.cpp
    ${MLIR_CODE_ROOT_DIR}/test/ut/dialect/AFIRRaceCheckTest.cpp
    ${MLIR_CODE_ROOT_DIR}/test/ut/dialect/AFIRRoleInterfaceTest.cpp
    ${MLIR_CODE_ROOT_DIR}/test/ut/dialect/AFIRSynchronizationTest.cpp
    ${MLIR_CODE_ROOT_DIR}/test/ut/dialect/AFPlanAttrsTest.cpp
    ${MLIR_CODE_ROOT_DIR}/test/ut/dialect/CanonicalResourceFactsTest.cpp
    ${MLIR_CODE_ROOT_DIR}/test/ut/dialect/PlanMetadataTest.cpp
    ${MLIR_CODE_ROOT_DIR}/test/ut/dialect/SymbolicBuilderTest.cpp
    ${MLIR_CODE_ROOT_DIR}/test/ut/dialect/VectorSegmentationTest.cpp
    ${MLIR_CODE_ROOT_DIR}/test/ut/dialect/ReducePlanTest.cpp
    ${MLIR_CODE_ROOT_DIR}/test/ut/dialect/TransferPlanTest.cpp
  LIBS
    AFIRAnalysis
    AFIRDialect
    AFPlanDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRParser
    MLIRSCFDialect
  COMMENT "Running AFIR dialect unit tests"
)
