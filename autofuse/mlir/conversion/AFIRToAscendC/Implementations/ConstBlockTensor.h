/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- ConstBlockTensor.h - kernel-scoped pre-filled constant blocks -*- C++ -*-===//
//
// 有几个 Autofuse 包装除了 tmp buffer，还要一块**预先填好某个常量**的 block 张量：
// `RsqrtExtend(dst, src, blk_tensor_with_value_1, tmp_buf, size)` 和
// `LogicalNot(dst, src, blk_tensor_with_value_1, tmp_buf, size)`。本文件负责造这一块。
//
// **它不是第二块 tmp buffer。** 两者没有一处相同：
//
//   - tmp buffer 由调度分配，内容未初始化，就是一张草稿纸。它的尺寸是 ATT 按
//     `max(同 id) ⊔ Σ(同 reuse 组)` 聚合出来的 `b{id}_size`；AFIR 只带得动一个 id，
//     造不出这块存储，也没有谁往里写初值。
//   - 这一块是一个 block（32 字节）的**常量**。包装把它当作 `Div` / `Sub` / `Min` 的
//     左操作数，配合 `src0BlkStride = 0` 和 `src0RepStride = 0` 反复读同一个 block，
//     所以一个 block 就够用，和 tile 多大无关。它的元素类型由**包装的形参**写死
//     （`RsqrtExtend` 收 `LocalTensor<float>`，`LogicalNot` 收 `LocalTensor<half>`），
//     和 dst / src 的 dtype 没有关系。
//
// 因此它既不参与 AFIR 操作数的三段划分，也不经过 `MemoryAllocate`：它在 AscGraph 上
// 本来就不存在。legacy 是在 codegen 里查一张按节点类型索引的静态表现造的
// （`common_utils.cpp` 的 `GetApiExtractDupSet` 和 `api_extract_dup_map`）。本文件做的
// 是同一件事，只是挪到了 lowering，并且**唯一的真源是 pattern 的模板实参**，不再另立
// 一张表。
//
// **UB 预算 ATT 已经算过了**：`CalcExtraTmpBufForAscGraph` 把这些 block 记进
// `tuning_space_->builtin_tmp_buffer`，按 (值, dtype) 去重，每块 32 字节。所以这里也
// 必须按 (值, dtype) 在整个 kernel 内共用一块。一个 op 造一块会超预算，而超出去的表现
// 是运行期 UB 越界，不是编译报错。
//
//===----------------------------------------------------------------------===//

#ifndef AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_IMPLEMENTATIONS_CONSTBLOCKTENSOR_H_
#define AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_IMPLEMENTATIONS_CONSTBLOCKTENSOR_H_

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::ascend::kernel::elementwise {

/// 认领标记。打在 `ascendc.tbuf.get_tensor` 上，取值是那个常量本身
/// （`FloatAttr`，类型就是这块张量的元素类型），所以一条属性同时说清「值是几」与
/// 「什么 dtype」——去重查的就是它。
constexpr llvm::StringLiteral kConstBlockAttrName = "ascendc.const_block";

/// 取这个 kernel 里 (value, elementType) 那一块常量张量；没有就在序言里现造一块。
///
/// 造出来的是四条：`ascendc.tbuf` + `pipe.init_buffer(32)` + `tbuf.get_tensor` +
/// `duplicate_l2`，末尾补一条 `PipeBarrier<PIPE_V>`——`Duplicate` 与随后读它的那个
/// 包装都在 V 上，中间没有屏障就是一次 RAW 竞争。位置在入口块的 `ascendc.pipe` 之后；
/// 函数里还没有 pipe 就先建一个（只喂 `!ascendc.local_tensor` 参数的 lit 用例走的就是
/// 这一支——那种函数里 `MemoryAllocate` 一个资源都没建，自然也没有 pipe）。
///
/// 失败只有一种：`op` 不在 `func.func` 里。
FailureOr<Value> getOrCreateConstBlockTensor(ConversionPatternRewriter &rewriter, Operation *op, Type elementType,
                                             const llvm::APFloat &value);

}  // namespace mlir::ascend::kernel::elementwise

#endif  // AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_IMPLEMENTATIONS_CONSTBLOCKTENSOR_H_
