/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dialect/AFPlan/IR/AFPlan.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "AFPlan/IR/AFPlanDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "AFPlan/IR/AFPlanAttrs.cpp.inc"

void mlir::afplan::AFPlanDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "AFPlan/IR/AFPlanAttrs.cpp.inc"
      >();
}

namespace {
// 名字 -> 该名字必须承载的属性类型。表为空即表示该前缀下没有合法名字。
bool hasExpectedType(llvm::StringRef name, mlir::Attribute value) {
  if (name == mlir::afplan::kStaticSelectionAttrName) {
    return mlir::isa<mlir::UnitAttr>(value);
  }
  if (name == mlir::afplan::kCandidateAttrName) {
    return mlir::isa<mlir::afplan::CandidateAttr>(value);
  }
  if (name == mlir::afplan::kVariantAttrName) {
    return mlir::isa<mlir::afplan::VariantAttr>(value);
  }
  if (name == mlir::afplan::kStagesAttrName) {
    return mlir::isa<mlir::afplan::StagesAttr>(value);
  }
  if (name == mlir::afplan::kScoreFuncAttrName) {
    return mlir::isa<mlir::StringAttr>(value);
  }
  return false;
}

bool isKnownName(llvm::StringRef name) {
  static constexpr llvm::StringLiteral known[] = {mlir::afplan::kStaticSelectionAttrName,
                                                  mlir::afplan::kCandidateAttrName, mlir::afplan::kVariantAttrName,
                                                  mlir::afplan::kStagesAttrName, mlir::afplan::kScoreFuncAttrName};
  return llvm::is_contained(known, name);
}
}  // namespace

mlir::LogicalResult mlir::afplan::AFPlanDialect::verifyOperationAttribute(mlir::Operation *op,
                                                                          mlir::NamedAttribute attribute) {
  llvm::StringRef name = attribute.getName().getValue();
  if (!name.starts_with(kAFPlanAttrPrefix)) {
    return mlir::success();
  }
  if (!isKnownName(name)) {
    return op->emitError() << "unknown afplan attribute '" << name << "'";
  }
  if (!hasExpectedType(name, attribute.getValue())) {
    return op->emitError() << "afplan attribute '" << name << "' has an unexpected value type";
  }
  return mlir::success();
}

// 候选编号是数组下标语义，负数没有对应的候选。此前这条校验散在
// `PlanMetadata::validatePlanMetadata`、`TilingProgramSpec`、`GenAggregate*` 各处，
// 每个读点自己判一次；现在在属性构造时一次判掉。
mlir::LogicalResult mlir::afplan::CandidateAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                                                        int64_t subgraph, int64_t plan) {
  if (subgraph < 0) {
    return emitError() << "afplan.candidate subgraph must be non-negative";
  }
  if (plan < 0) {
    return emitError() << "afplan.candidate plan must be non-negative";
  }
  return mlir::success();
}

mlir::LogicalResult mlir::afplan::VariantAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                                                      int64_t stage, int64_t templateId) {
  if (stage < 0) {
    return emitError() << "afplan.variant stage must be non-negative";
  }
  if (templateId < 0) {
    return emitError() << "afplan.variant template must be non-negative";
  }
  return mlir::success();
}

// 分派表每个 stage 至少要有一个 template，否则该 stage 无从分派。
mlir::LogicalResult mlir::afplan::StagesAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                                                     llvm::ArrayRef<int64_t> templateCounts) {
  if (templateCounts.empty()) {
    return emitError() << "afplan.stages requires at least one stage";
  }
  for (int64_t count : templateCounts) {
    if (count <= 0) {
      return emitError() << "afplan.stages template count must be positive";
    }
  }
  return mlir::success();
}
