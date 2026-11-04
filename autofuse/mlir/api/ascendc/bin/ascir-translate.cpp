/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Target/Asc/Translation.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

int main(int argc, char** argv)
{
    registerAllTranslations();

    TranslateFromMLIRRegistration reg(
        "mlir-to-ascendc", "translate from mlir to Ascend C",
        [](Operation* op, raw_ostream& output) { return translateToAscendC(op, output); },
        [](DialectRegistry& registry) {
            registry.insert<affine::AffineDialect, bufferization::BufferizationDialect>();
            registry.insert<
                //
                arith::ArithDialect, ascendc::AscendCDialect, cf::ControlFlowDialect, DLTIDialect,
                emitasc::EmitAscDialect, emitc::EmitCDialect, func::FuncDialect, LLVM::LLVMDialect, math::MathDialect,
                memref::MemRefDialect, scf::SCFDialect
                //
                >();
            ascendc::registerExternalModels(registry);
            emitasc::registerExternalModels(registry);
        });

    return failed(mlirTranslateMain(argc, argv, "AscIR translation tool"));
}
