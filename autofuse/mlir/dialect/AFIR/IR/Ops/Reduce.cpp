/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Reduce.cpp - AFIR reduce op implementation ---------------*- C++ -*-===//
//
// Ported from Ascend-MLIR-gsr: reduce verifiers. The seven reduce ops share one
// ODS class (`AFIR_ReduceOp`), so their verifiers and interface methods share one
// set of templates here.
//
//===----------------------------------------------------------------------===//

#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/IR/ReduceScratch.h"
#include "dialect/AFIR/IR/AFIRVerifierUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::afir;
using MemoryEffect = SideEffects::EffectInstance<MemoryEffects::Effect>;

namespace mlir::afir {
FailureOr<ReduceScratchIndices> getReduceScratchIndices(Operation *op, unsigned scratchCount, std::string *diagnostic) {
  auto fail = [&](llvm::StringRef reason) -> FailureOr<ReduceScratchIndices> {
    if (diagnostic) {
      *diagnostic = reason.str();
    }
    return failure();
  };
  auto roles = op->getAttrOfType<ArrayAttr>(kReduceScratchRolesAttr);
  if (!roles || roles.size() != scratchCount) {
    return fail("afir.scratch_roles must contain one role per scratch operand");
  }
  std::optional<unsigned> api, round;
  for (auto [index, attr] : llvm::enumerate(roles)) {
    auto role = dyn_cast<StringAttr>(attr);
    if (!role) {
      return fail("scratch roles must be strings");
    }
    std::optional<unsigned> *slot = nullptr;
    if (role.getValue() == "api_scratch") {
      slot = &api;
    } else if (role.getValue() == "round_result") {
      slot = &round;
    } else {
      return fail("unknown reduce scratch role");
    }
    if (slot->has_value()) {
      return fail("duplicate reduce scratch role");
    }
    *slot = index;
  }
  if (!api) {
    return fail("reduce requires an api_scratch operand");
  }
  return ReduceScratchIndices{*api, round};
}
}  // namespace mlir::afir

namespace {

LogicalResult inferReduceShape(RankedTensorType inputType, int64_t axis, SmallVectorImpl<int64_t> &inferredShape) {
  if (axis < 0 || axis >= inputType.getRank()) {
    return failure();
  }
  inferredShape.assign(inputType.getShape().begin(), inputType.getShape().end());
  inferredShape.erase(inferredShape.begin() + axis);
  return success();
}

LogicalResult verifyReduceOp(Operation *op) {
  auto types = getUnaryShapedTypes(op);
  if (failed(types)) return failure();
  auto inputType = mlir::dyn_cast<RankedTensorType>(types->input);
  if (!inputType) {
    return success();
  }

  SmallVector<int64_t> inferredShape;
  auto axis = op->getAttrOfType<IntegerAttr>("axis");
  if (!axis || failed(inferReduceShape(inputType, axis.getInt(), inferredShape))) {
    return op->emitOpError("axis must be in the range [0, input rank)");
  }
  if (types->result.hasRank() && failed(verifyCompatibleShape(inferredShape, types->result.getShape()))) {
    return op->emitOpError("result shape must match the inferred reduce shape");
  }
  return success();
}

// buffer 形态保留被归约的维，归约了哪几维由 descriptor 决定，所以这里只查结构：
// 同秩、同元素类型、结果即 destination，scratch 都是一维 ui8。
//
// 七个归约 op 的 ODS 形态由 `AFIR_ReduceOp` 一个类生成，结构检查因此也只有一份。
template <typename OpT>
LogicalResult verifyBufferizedReduce(OpT op) {
  if (op.getAxisAttr()) {
    return op.emitOpError("buffer form takes its reduced dimensions from tensor descriptors and cannot carry axis");
  }
  if (op.getResult().getType() != op.getDestination().getType()) {
    return op.emitOpError("buffer result and destination must have the same type");
  }
  if (llvm::any_of(op->getOperandTypes(), [](Type type) { return isa<TensorType>(type); }) ||
      isa<TensorType>(op.getResult().getType())) {
    return op.emitOpError("buffer form expects non-tensor input, scratch, destination, and result");
  }
  if (!op.getScratch().empty() || op->hasAttr(kReduceScratchRolesAttr)) {
    std::string reason;
    if (failed(getReduceScratchIndices(op, op.getScratch().size(), &reason))) {
      return op.emitOpError(reason);
    }
  }
  auto inputType = dyn_cast<MemRefType>(op.getInput().getType());
  auto destinationType = dyn_cast<MemRefType>(op.getDestination().getType());
  if (!inputType || !destinationType) {
    // 资源物化之后操作数换成 AscendC 的 `LocalTensor`，而 `afir.sum` 要到相位 C 才下降，
    // 因此它会以非 memref 形态存在一段时间。memref 专属的结构检查此时不再适用——与
    // `verifyBufferizedAdd` 的处理一致（Elementwise.cpp）。descriptor 一致性仍然要查，
    // `verifyCanonicalBufferDescriptors` 自己会在非 memref 形态下放行。
    return verifyCanonicalBufferDescriptors(op);
  }
  if (inputType.getElementType() != destinationType.getElementType()) {
    return op.emitOpError("reduce input and destination must have the same element type");
  }
  if (inputType.getRank() != destinationType.getRank()) {
    return op.emitOpError("buffer form keeps reduced dimensions, so input and destination must have the same rank");
  }
  for (Value scratch : op.getScratch()) {
    // 同上：物化之后不再是 memref，只在 memref 形态下查秩。
    if (auto scratchType = dyn_cast<MemRefType>(scratch.getType()); scratchType && scratchType.getRank() != 1) {
      return op.emitOpError("scratch buffers must be rank-1 memrefs");
    }
  }
  return verifyCanonicalBufferDescriptors(op);
}

template <typename OpT>
LogicalResult verifyReduce(OpT op) {
  if (op.isBufferizedForm()) {
    return verifyBufferizedReduce(op);
  }
  if (!op.getScratch().empty() || op->hasAttr(kReduceScratchRolesAttr)) {
    return op.emitOpError("tensor form cannot carry scratch buffers or afir.scratch_roles");
  }
  return verifyReduceOp(op.getOperation());
}

template <typename OpT>
void getReduceEffects(OpT op, SmallVectorImpl<MemoryEffect> &effects) {
  if (!op.isBufferizedForm()) {
    return;
  }
  Operation *operation = op.getOperation();
  FailureOr<OperandPartition> partition = getOperandPartition(operation);
  if (failed(partition)) {
    return;
  }
  const unsigned destinationIndex = partition->inputCount;
  effects.emplace_back(MemoryEffects::Read::get(), &operation->getOpOperand(0), SideEffects::DefaultResource::get());
  for (unsigned index = destinationIndex + partition->destinationCount; index < operation->getNumOperands(); ++index) {
    effects.emplace_back(MemoryEffects::Read::get(), &operation->getOpOperand(index),
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), &operation->getOpOperand(index),
                         SideEffects::DefaultResource::get());
  }
  // destination 既读又写：沿被归约的 tile 外轴分段累加时，本轮要把上一轮的结果读出来
  // 合并（legacy 的 `Add(y, tmp, y)`）。只标 Write 等于声称"调用前的内容无关紧要"，
  // 依赖 effect 的分析会据此复用或丢弃这块存储，那是静默算错而不是报错。
  effects.emplace_back(MemoryEffects::Read::get(), &operation->getOpOperand(destinationIndex),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &operation->getOpOperand(destinationIndex),
                       SideEffects::DefaultResource::get());
}

template <typename OpT>
FailureOr<unsigned> getReduceTiedResultIndex(OpT op, unsigned destinationIndex) {
  if (destinationIndex != 0 || !op.getDestination() || op.getOperation()->getNumResults() != 1) {
    return failure();
  }
  return 0U;
}

}  // namespace

// 七个 op 的接口实现逐字相同，各自转发进上面那一套模板。ODS 不能替 op 生成这些方法体，
// 所以宏是这里唯一避免七份复制的手段——与 `Elementwise.cpp` 对二元算子族的做法一致。
#define AFIR_REDUCE_OP_IMPL(OP)                                                           \
  LogicalResult OP::verify() {                                                            \
    return verifyReduce(*this);                                                           \
  }                                                                                       \
  FailureOr<Role> OP::getRole() {                                                         \
    return getCanonicalTargetRole(getOperation());                                        \
  }                                                                                       \
  unsigned OP::getNumDestinations() {                                                     \
    return getDestination() ? 1 : 0;                                                      \
  }                                                                                       \
  unsigned OP::getNumTmpBuffers() {                                                       \
    return getScratch().size();                                                           \
  }                                                                                       \
  FailureOr<unsigned> OP::getTiedResultIndex(unsigned destinationIndex) {                 \
    return getReduceTiedResultIndex(*this, destinationIndex);                             \
  }                                                                                       \
  void OP::getEffects(SmallVectorImpl<MemoryEffect> &effects) {                           \
    getReduceEffects(*this, effects);                                                     \
  }                                                                                       \
  Speculation::Speculatability OP::getSpeculatability() {                                 \
    return isBufferizedForm() ? Speculation::NotSpeculatable : Speculation::Speculatable; \
  }

AFIR_REDUCE_OP_IMPL(MaxOp)
AFIR_REDUCE_OP_IMPL(MinOp)
AFIR_REDUCE_OP_IMPL(SumOp)
AFIR_REDUCE_OP_IMPL(MeanOp)
AFIR_REDUCE_OP_IMPL(ProdOp)
AFIR_REDUCE_OP_IMPL(AnyOp)
AFIR_REDUCE_OP_IMPL(AllOp)

#undef AFIR_REDUCE_OP_IMPL
