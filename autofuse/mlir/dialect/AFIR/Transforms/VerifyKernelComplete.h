/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_DIALECT_AFIR_TRANSFORMS_VERIFYKERNELCOMPLETE_H
#define AUTOFUSE_MLIR_DIALECT_AFIR_TRANSFORMS_VERIFYKERNELCOMPLETE_H

#include <cstdint>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::afir {

// C4 要问「这个 Op 有没有目标实现」，而实现注册表是**目标层**的东西
// （target/Ascend/Capability/ImplementationRegistry.h）。判据本身与目标无关，所以这里
// 收一个谓词而不是注册表：AFIR 层不链接任何一个具体目标，查询由调用方注入。
//
// `targetName` 只进诊断（"no <target> implementation is registered for ..."）——把目标
// 名硬编码在 AFIR 层就等于承认这条判据只对一个后端成立。
struct ImplementationQuery {
  llvm::StringRef targetName;
  llvm::function_ref<bool(Operation *)> contains;
};

// 设计文档 5.2 的四条完备性判据。
//
// 把完备性从文档表格变成可运行的检查——否则「完备」永远是主观判断，也无法在 PR 里当
// 门禁。每条判据都被收窄成一个**对单个 kernel 函数可判定**的命题：判据本身谈的是
// 表达能力（「任意 SSA buffer 都……」「……可只用 AFIR 表示」），而表达能力的检验方式
// 就是「我们关心的每个 kernel 都通过」。
enum class CompletenessCriterion : uint8_t {
  // C1 任意 SSA buffer 的来源、地址空间、shape/stride、所有权和释放点可追踪。
  //
  // 收窄为：函数里每个 memref 值都有确定的来源（函数参数或定义 Op）、类型上带
  // `#afir.space<*>`（或已物化的 AscendC 地址空间）、是 ranked，且每个
  // `memref.alloc` 都有对应的 `memref.dealloc`。最后一条是 ADR-10 选
  // `alloc`+`dealloc` 而不是 `alloca` 的**唯一**理由——释放点要可追踪。
  BufferTraceability,
  // C2 每个异步操作的完成条件可表示，所有 consumer 的 wait 可验证。
  //
  // 收窄为：竞争检查器在这个函数上零发现。两者是同一个命题的两种说法——「所有
  // consumer 的 wait 可验证」成立，当且仅当没有哪一对访问缺 happens-before。
  // 这也是验收 15 升级后的判据（设计文档 5.6）。
  AsyncCompletion,
  // C3 Vector / Cube / Scalar / MTE 数据流与控制流可只用 AFIR 表示。
  //
  // 收窄为：函数体里只出现 AFIR 与标准方言的 Op。后端方言（`ascendc` / `emitasc` /
  // `emitc`）一旦出现在 AFIR 层，就说明有一段语义 AFIR 表达不了、只能靠提前下降到
  // 后端绕过去——那正是 C3 不成立的形态。
  AfirOnlyRepresentation,
  // C4 可通过确定的 lowering 生成目标代码，unsupported 编译期返回原因。
  //
  // 收窄为：函数里每个 AFIR Op 在实现注册表里都能查到至少一个实现。查不到时今天的
  // 失败是 `applyFullConversion` 的一句泛化的 "failed to legalize"，既不说是哪条
  // 判据不成立，也不说缺的是什么——本检查把它变成一条指名道姓的诊断，且在下降开始
  // **之前**就报出来。
  DeterministicLowering,
};

llvm::StringRef getCompletenessCriterionName(CompletenessCriterion criterion);

// 在一个 kernel 函数上检查一条判据。不满足时把原因报成挂在具体 Op / 值上的诊断，
// 并返回 failure；判据是「这个函数满足吗」，因此一个函数可以同时违反多条。
//
// `implementations` 只被 C4 使用；谓词为空时 C4 报「缺少可用的实现查询」而不是静默通过。
LogicalResult verifyCompletenessCriterion(func::FuncOp func, CompletenessCriterion criterion,
                                          ImplementationQuery implementations);

}  // namespace mlir::afir

#endif  // AUTOFUSE_MLIR_DIALECT_AFIR_TRANSFORMS_VERIFYKERNELCOMPLETE_H
