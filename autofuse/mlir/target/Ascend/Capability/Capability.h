/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_TARGET_ASCEND_CAPABILITY_CAPABILITY_H
#define AUTOFUSE_MLIR_TARGET_ASCEND_CAPABILITY_CAPABILITY_H

#include "target/Ascend/AscendTargetProfile.h"

#include <cstdint>
#include <optional>
#include <string>
#include <variant>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"

namespace mlir {
class Operation;
}

namespace mlir::ascend::instruction {

enum class ImplementationKind {
  AscendC,
  CCEIntrinsic,
  PTOISA,
};

struct ImplementationId {
  std::string value;
};

bool operator==(const ImplementationId &lhs, const ImplementationId &rhs);
bool operator!=(const ImplementationId &lhs, const ImplementationId &rhs);

struct ImplementationKey {
  OperationName sourceOp;
  ImplementationKind kind;
  ImplementationId id;
};

bool operator==(const ImplementationKey &lhs, const ImplementationKey &rhs);
bool operator!=(const ImplementationKey &lhs, const ImplementationKey &rhs);

enum class SupportState {
  Supported,
  Unsupported,
  Undetermined,
};

enum class CapabilityKind {
  ScalarInput,
  BroadcastInline,
  Inplace,
};

enum class CapabilityReason {
  None,
  QueryNotImplemented,
  UnsupportedType,
  UnsupportedShape,
  UnsupportedTarget,
  TargetFeatureNotModeled,
  TargetFeatureUnavailable,
  MissingInformation,
  ResourceExpressionError,
};

enum class ValuePositionKind {
  Operand,
  Result,
};

struct ValuePosition {
  ValuePositionKind kind;
  uint32_t index;
};

struct ScalarOperandConstraint {
  llvm::SmallVector<uint32_t, 2> allowedOperandIndices;
  bool allowsAllScalar;
};

struct TypeConstraint {
  ValuePosition value;
  llvm::SmallVector<Type, 4> allowedTypes;
};

struct RankConstraint {
  ValuePosition value;
  int64_t minimumRank;
  int64_t maximumRank;
};

// Describes parameter-free pairwise shape relations.
//
// Parameterized or N-ary shape semantics are validated by the operation verifier or capability provider and use
// dedicated typed constraints when they must be exposed to a consumer.
enum class ShapeRelation {
  Equal,
  BroadcastCompatible,
};

struct ShapeConstraint {
  ValuePosition lhs;
  ValuePosition rhs;
  ShapeRelation relation;
};

struct AlignmentConstraint {
  ValuePosition value;
  uint64_t alignmentBytes;
};

struct TargetFeatureConstraint {
  TargetFeature feature;
};

enum class AliasAccessMode {
  ReadOnlyInput,
  ReadWriteInput,
};

struct AliasConstraint {
  uint32_t inputOperandIndex;
  uint32_t outputResultIndex;
  AliasAccessMode accessMode;
};

struct ScalarInputQuery {
  llvm::SmallVector<uint32_t, 2> scalarOperandIndices;
  bool allScalar = false;
};

struct BroadcastInlineQuery {
  uint32_t sourceOperandIndex = 0;
  uint32_t destinationOperandIndex = 0;
  int64_t axis = -1;
  llvm::SmallVector<int64_t, 4> sourceStrides;
  llvm::SmallVector<int64_t, 4> destinationStrides;
  bool hasDynamicShape = false;
};

struct InplaceQuery {
  uint32_t inputOperandIndex = 0;
  uint32_t outputResultIndex = 0;
  AliasAccessMode accessMode = AliasAccessMode::ReadWriteInput;
};

using CapabilityQueryDetail = std::variant<ScalarInputQuery, BroadcastInlineQuery, InplaceQuery>;

struct CapabilityQueryRequest {
  CapabilityKind kind;
  Operation *op;
  std::optional<ImplementationKey> implementation;
  CapabilityQueryDetail detail;
};

using CapabilityConstraint = std::variant<ScalarOperandConstraint, TypeConstraint, RankConstraint, ShapeConstraint,
                                          AlignmentConstraint, TargetFeatureConstraint, AliasConstraint>;

class CapabilitySupport {
 public:
  static CapabilitySupport supported(llvm::SmallVector<CapabilityConstraint, 4> constraints = {});
  static CapabilitySupport unsupported(CapabilityReason reason,
                                       llvm::SmallVector<CapabilityConstraint, 4> constraints = {});
  static CapabilitySupport undetermined(CapabilityReason reason,
                                        llvm::SmallVector<CapabilityConstraint, 4> constraints = {});

  SupportState getState() const;
  CapabilityReason getReason() const;
  llvm::ArrayRef<CapabilityConstraint> getConstraints() const;

 private:
  CapabilitySupport(SupportState state, CapabilityReason reason,
                    llvm::SmallVector<CapabilityConstraint, 4> constraints);

  SupportState state_;
  CapabilityReason reason_;
  llvm::SmallVector<CapabilityConstraint, 4> constraints_;
};

struct QueryContext {
  const TargetProfile &target;
};

struct Candidate {
  ImplementationKey key;
  CapabilitySupport support;
};

class Candidates {
 public:
  Candidates() = default;
  explicit Candidates(llvm::SmallVector<Candidate, 2> candidates);

  llvm::ArrayRef<Candidate> getAll() const;
  llvm::SmallVector<Candidate, 2> getSupported() const;
  bool hasUndetermined() const;

 private:
  llvm::SmallVector<Candidate, 2> candidates_;
};

}  // namespace mlir::ascend::instruction

#endif  // AUTOFUSE_MLIR_TARGET_ASCEND_CAPABILITY_CAPABILITY_H
