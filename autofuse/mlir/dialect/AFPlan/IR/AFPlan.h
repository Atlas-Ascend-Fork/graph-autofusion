/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- AFPlan.h - afplan dialect public header -------------------*- C++ -*-===//
//
// `afplan` 承载候选身份与候选间分派（设计文档 2.2 / 2.3 / ADR-5）：**这是哪个候选、
// 凭什么选它、运行时如何在候选间分派**。它是仅属性方言，没有 Op 也没有 Type。
//
// 归属判别：当候选集中只有一个候选，这条信息还影响生成的 kernel 代码吗？是则属
// `afir`（公开面），否则属 `afplan`（内部）。因此 `tiling_key` 留在 `afir`
// （它仍决定 TilingData 结构与 entry 分派），而 (subgraph, plan) 归 `afplan`。
//
// 依赖方向单向：`afplan` 的属性挂在 `afir` 的容器上，但 `afir` 的语义不依赖它。
//
//===----------------------------------------------------------------------===//

#ifndef AUTOFUSE_MLIR_DIALECT_AFPLAN_IR_AFPLAN_H
#define AUTOFUSE_MLIR_DIALECT_AFPLAN_IR_AFPLAN_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

#include "AFPlan/IR/AFPlanDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "AFPlan/IR/AFPlanAttrs.h.inc"

namespace mlir::afplan {

// Discardable attribute names. They are part of the serialized IR contract and
// are read outside ODS, so keep the spellings centralized.
inline constexpr llvm::StringLiteral kAFPlanAttrPrefix = "afplan.";
// module 级：这是哪个候选。
// Module retains original ATT coordinates after compile-time selection. Keep
// graph selectors in its aggregate ABI even when only one plan survives.
inline constexpr llvm::StringLiteral kStaticSelectionAttrName = "afplan.static_selection";
inline constexpr llvm::StringLiteral kCandidateAttrName = "afplan.candidate";
// func 级：候选内部的哪个 (stage, template) 变体。
inline constexpr llvm::StringLiteral kVariantAttrName = "afplan.variant";
// module 级：分派表的形状。
inline constexpr llvm::StringLiteral kStagesAttrName = "afplan.stages";
// 打分函数名。消费者在 MLIR 之外（ATT 打分），仓内无读点——归属判据见设计文档 2.3
// 「三个只写不读的属性」。保留在 afplan 而不是删除，因为它确有仓外消费者。
inline constexpr llvm::StringLiteral kScoreFuncAttrName = "afplan.score_func";

// Accessors. 返回空属性表示「不存在或类型不符」——两者对调用方是同一件事：这个容器
// 没有可用的候选身份。取到非空即意味着已经过 ODS verifier，字段无需再次校验。
inline CandidateAttr getCandidate(Operation *op) {
  return op ? op->getAttrOfType<CandidateAttr>(kCandidateAttrName) : CandidateAttr();
}

// 聚合入口的 plan 表把候选身份放在字典项里，与挂在 module 上的是同一个属性。
inline CandidateAttr getCandidate(DictionaryAttr entry) {
  return entry ? entry.getAs<CandidateAttr>(kCandidateAttrName) : CandidateAttr();
}

inline VariantAttr getVariant(Operation *op) {
  return op ? op->getAttrOfType<VariantAttr>(kVariantAttrName) : VariantAttr();
}

inline StagesAttr getStages(Operation *op) {
  return op ? op->getAttrOfType<StagesAttr>(kStagesAttrName) : StagesAttr();
}

}  // namespace mlir::afplan

#endif  // AUTOFUSE_MLIR_DIALECT_AFPLAN_IR_AFPLAN_H
