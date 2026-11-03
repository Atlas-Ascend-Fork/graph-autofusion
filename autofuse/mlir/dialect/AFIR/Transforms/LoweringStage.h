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
// 「某个下降阶段是否已经发生」的判据，取 IR 自身的结构而不是「哪个 pass 跑过」的标记
// 属性（设计文档 D5）。理由与边界见 conversion/AFIRToAscendC/LoweringStage.h——那里放的
// 是同一族里判据要认后端算子的那两个。
//
// 本文件只放判据是纯 AFIR 事实的那一个。分开是因为它们的依赖不同：「资源还欠债吗」只问
// AFIR 算子的落点类型，而「物化过没有」按定义就要认 `ascendc` 方言。
//
//===----------------------------------------------------------------------===//

#ifndef AUTOFUSE_MLIR_DIALECT_AFIR_TRANSFORMS_LOWERINGSTAGE_H
#define AUTOFUSE_MLIR_DIALECT_AFIR_TRANSFORMS_LOWERINGSTAGE_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::afir {

// 资源物化是否**还欠这个函数的债**。
//
// 判据是「还有 AFIR destination-style 算子的落点是 memref」——物化会把这些落点换成目标
// 的 buffer 类型，所以它们的存在就等于「还没跑」。
//
// 它与 `hasMaterializedAscendCResources` 不是互补关系，这正是要分开两个函数的原因：
// 一个没有任何片上资源的函数两者**同时为假**——没物化出东西，也没有待物化的东西。
// 那种函数没有可同步的算子，流水同步对它无事可做，应当成功返回而不是报「资源未物化」。
bool hasPendingResourceMaterialization(func::FuncOp func);

}  // namespace mlir::afir

#endif  // AUTOFUSE_MLIR_DIALECT_AFIR_TRANSFORMS_LOWERINGSTAGE_H
