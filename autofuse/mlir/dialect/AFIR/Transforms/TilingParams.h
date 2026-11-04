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

// 单个符号的物化在 `dialect/AFIR/IR/SymbolicBuilder.h`
// （`getOrCreateTilingParam` / `materializeRuntimeSize`）：它不需要 Func 方言，住在
// 方言层，bridge 与 conversion 都用得上。本文件只管「一个函数的全部符号」这一步。

// 把函数里所有运行期尺寸符号一次性物化成 `afir.tiling_param`。
//
// **这一步是可选的，不是后面几步的前置。** `getOrCreateTilingParam` 是 get-or-create，
// 循环生成与资源物化在缺的时候会自己建（`test/lit/demo/afir_memory_allocate_buffer_form.mlir`
// 的 RUN 行就不跑这一步，照样 CHECK 得出 `afir.tiling_param`）。也就是说同一个 op 有两个
// 活的产生点，两个都被用例走到。
//
// 提前跑它买到的是两件事，都不是「让后面能跑」：尺寸序言的顺序与「谁先问」无关地固定
// 下来；坏的尺寸表达式在一处一次性报出来，而不是等某个消费者恰好第一个碰到它。
//
// 符号来源两处，与资源物化阶段收集的是同一个集合：`#afir.asc_graph` 的 `size_var`
// （轴尺寸），以及每一个 `#afir.buffer_desc` 的 `size_bytes_expr` 的自由符号
// （缓冲字节数，典型值是 `q0_size` / `b5_size`）。
//
// 不校验名字在目标 ABI 里存在——AFIR 层不知道也不该知道 TilingData 的字段表，那是
// 下降阶段的事（见 `afir.tiling_param` 的 ODS description）。
LogicalResult materializeTilingParams(func::FuncOp func);

}  // namespace mlir::afir

#endif  // AUTOFUSE_MLIR_DIALECT_AFIR_TRANSFORMS_TILINGPARAMS_H
