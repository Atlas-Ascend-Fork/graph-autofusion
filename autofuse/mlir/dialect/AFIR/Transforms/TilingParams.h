/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_DIALECT_AFIR_TRANSFORMS_TILINGPARAMS_H
#define AUTOFUSE_MLIR_DIALECT_AFIR_TRANSFORMS_TILINGPARAMS_H

#include "dialect/AFIR/IR/AFIR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::afir {

// 此接口按函数收集尺寸符号。单符号物化使用 SymbolicBuilder.h 的
// getOrCreateTilingParam / materializeRuntimeSize；它们不依赖 Func 方言，供 bridge 和 conversion 共用。

// 将运行期尺寸符号预先物化为 afir.tiling_param，固定序言顺序并集中诊断无效表达式。
// 此步骤可选；循环生成和资源物化也能通过 getOrCreateTilingParam 补建参数，
// afir_memory_allocate_buffer_form.mlir 覆盖省略本步骤的路径。
//
// 收集 asc_graph 的 size_var，以及 buffer_desc 的 size_bytes_expr 自由符号，
// 例如 q0_size、b5_size。目标 TilingData 字段是否存在由下降阶段校验，AFIR 层不依赖目标 ABI。
LogicalResult materializeTilingParams(func::FuncOp func);

}  // namespace mlir::afir

#endif  // AUTOFUSE_MLIR_DIALECT_AFIR_TRANSFORMS_TILINGPARAMS_H
