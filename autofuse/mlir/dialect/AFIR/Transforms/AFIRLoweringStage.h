/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- LoweringStage.h - 下降阶段的结构判据（AFIR 侧）------------*- C++ -*-===//
//
// 根据 IR 结构判断资源物化是否完成，不使用 pass 执行标记（设计文档 D5）。
// 本文件的判据只检查 AFIR destination 类型，不依赖后端方言。
// 需要识别 AscendC 资源和同步算子的判据放在 conversion/AFIRToAscendC/AFIRToAscendCLoweringStage.h，
// 其中也说明了结构判据的设计依据和边界。
//
//===----------------------------------------------------------------------===//

#ifndef AUTOFUSE_MLIR_DIALECT_AFIR_TRANSFORMS_LOWERINGSTAGE_H
#define AUTOFUSE_MLIR_DIALECT_AFIR_TRANSFORMS_LOWERINGSTAGE_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::afir {

// 是否仍有 AFIR destination-style 算子的 destination 为 memref。
// 资源物化会将这些 destination 替换为目标 buffer 类型。
//
// 此判据与 hasMaterializedAscendCResources 不互补：没有片上资源的函数两者均为假。
// 这种函数也没有需要同步的算子，流水同步应成功返回，不能报资源未物化。
bool hasPendingResourceMaterialization(func::FuncOp func);

}  // namespace mlir::afir

#endif  // AUTOFUSE_MLIR_DIALECT_AFIR_TRANSFORMS_LOWERINGSTAGE_H
