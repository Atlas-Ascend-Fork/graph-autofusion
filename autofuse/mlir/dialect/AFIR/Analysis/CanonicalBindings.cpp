/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dialect/AFIR/Analysis/CanonicalBindings.h"

#include "dialect/AFIR/Analysis/ResourceIds.h"

#include <algorithm>
#include <limits>
#include <string>
#include <utility>

#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/IR/AFIRVerifierUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::afir {
namespace {

// 定义在下面的公开区。

// 收集期的可变状态。它只有分析字段——`resourceBuilder` / `createdOps` 那一批属于物化器，
// 留在 conversion 侧（CanonicalMemoryAllocator.h）。这条分界就是这一相能搬过来的理由。
class CanonicalBindingCollector {
 public:
  explicit CanonicalBindingCollector(func::FuncOp func) : func(func), context(func.getContext()) {}

  FailureOr<CanonicalBindings> collect() {
    if (failed(validateWorkspacePlan()) || failed(collectCanonicalBindings())) {
      return failure();
    }
    return CanonicalBindings{std::move(values), std::move(valueOrder)};
  }

 private:
  LogicalResult validateWorkspacePlan();
  FailureOr<ArrayAttr> getDescriptorArray(Operation *owner, StringRef name, unsigned expected, bool required);
  static Space resolveValueSpace(Value value, BufferDescAttr buffer);
  static bool hasCompatibleGmTensorFacts(const CanonicalValueInfo &existing, TensorDescAttr tensor);
  FailureOr<StorageView> makeStorageView(Value value, TensorDescAttr tensor, BufferDescAttr buffer, Operation *owner,
                                         unsigned index);
  LogicalResult recordValue(Value value, TensorDescAttr tensor, BufferDescAttr buffer, Operation *owner,
                            unsigned index);
  LogicalResult recordArray(Operation *owner, ValueRange valueRange, StringRef tensorName, bool required);
  LogicalResult recordExternalStoreDestinations(Operation *operation, ValueRange destinations);
  LogicalResult recordTmpBuffers(Operation *operation, unsigned firstIndex, ValueRange tmpBuffers);
  LogicalResult collectFunctionBindings();
  LogicalResult collectOperationBinding(Operation *operation);
  LogicalResult collectOperationBindings();
  LogicalResult collectCanonicalBindings();

  func::FuncOp func;
  MLIRContext *context;
  // `walk` 的回调不能返回 failure，所以错误经这个成员带出来。
  LogicalResult result = success();
  llvm::DenseMap<Value, CanonicalValueInfo> values;
  SmallVector<Value> valueOrder;
  unsigned nextOrdinal = 0;
};

LogicalResult CanonicalBindingCollector::validateWorkspacePlan() {
  auto workspacePlan = func->getAttrOfType<ArrayAttr>(kWorkspacePlanAttrName);
  if (!workspacePlan) {
    return success();
  }
  std::set<int64_t> tensorIds;
  for (auto [index, attribute] : llvm::enumerate(workspacePlan)) {
    auto descriptor = dyn_cast<BufferDescAttr>(attribute);
    if (!descriptor || descriptor.getBufferId() < 0 || !descriptor.getSizeBytesExpr() ||
        descriptor.getLifeTimeAxisId() != -1) {
      return func.emitError() << kInvalidDescriptorDiagnostic << ": workspace_plan element " << index
                              << " requires buffer_id, size_bytes_expr and function lifetime (-1)";
    }
    if (!tensorIds.insert(descriptor.getBufferId()).second) {
      return func.emitError() << kInvalidBindingDiagnostic << ": workspace_plan contains duplicate buffer_id "
                              << descriptor.getBufferId();
    }
  }
  return success();
}

FailureOr<ArrayAttr> CanonicalBindingCollector::getDescriptorArray(Operation *owner, StringRef name, unsigned expected,
                                                                   bool required) {
  auto array = owner->getAttrOfType<ArrayAttr>(name);
  if (!array) {
    if (required || expected != 0) {
      owner->emitError() << kInvalidDescriptorDiagnostic << ": missing '" << name << "'";
      return failure();
    }
    return ArrayAttr::get(context, {});
  }
  if (array.size() != expected) {
    owner->emitError() << kDescriptorCountMismatchDiagnostic << ": '" << name << "' has " << array.size()
                       << " entries, expected " << expected;
    return failure();
  }
  return array;
}

Space CanonicalBindingCollector::resolveValueSpace(Value value, BufferDescAttr buffer) {
  if (std::optional<Space> space = getMemorySpace(value.getType())) {
    return *space;
  }
  return buffer.getSpace().getValue();
}

// 同一个 SSA 值被多处绑定时，GM 上的 tensor 视图允许不同：GM 值在不同使用点可以按
// 不同的轴/步长解释，而片上资源不行。空间取自 `existing.view.resource.space`（已由
// 类型决定），调用点随后还会要求两份 descriptor 完全相等。
bool CanonicalBindingCollector::hasCompatibleGmTensorFacts(const CanonicalValueInfo &existing, TensorDescAttr tensor) {
  return existing.view.resource.space == Space::GM && existing.tensor.getAxis() == tensor.getAxis() &&
         existing.tensor.getRepeats() == tensor.getRepeats() && existing.tensor.getStrides() == tensor.getStrides() &&
         existing.tensor.getVectorizedAxis() == tensor.getVectorizedAxis() &&
         existing.tensor.getVectorizedStrides() == tensor.getVectorizedStrides();
}

FailureOr<StorageView> CanonicalBindingCollector::makeStorageView(Value value, TensorDescAttr tensor,
                                                                  BufferDescAttr buffer, Operation *sourceOwner,
                                                                  unsigned sourceIndex) {
  auto type = cast<MemRefType>(value.getType());
  SmallVector<int64_t> strides;
  int64_t offset = ShapedType::kDynamic;
  if (failed(type.getStridesAndOffset(strides, offset))) {
    return emitError(value.getLoc()) << kInvalidBindingDiagnostic
                                     << ": canonical memref layout cannot be represented as a StorageView";
  }
  for (int64_t size : type.getShape()) {
    if (size == 0 || size < ShapedType::kDynamic) {
      return emitError(value.getLoc()) << kInvalidBindingDiagnostic
                                       << ": canonical StorageView has an invalid zero/negative dimension";
    }
  }
  StorageView view;
  view.resource = {resolveValueSpace(value, buffer), false, buffer.getBufferId()};
  view.sourceOwner = sourceOwner;
  view.sourceIndex = sourceIndex;
  view.axis.assign(tensor.getAxis().begin(), tensor.getAxis().end());
  if (offset != ShapedType::kDynamic) {
    if (!isa<IntegerType, FloatType>(type.getElementType())) {
      return emitError(value.getLoc()) << kInvalidBindingDiagnostic
                                       << ": StorageView offset requires a scalar integer or float element type";
    }
    int64_t elementBytes = type.getElementType().getIntOrFloatBitWidth() / 8;
    if (elementBytes <= 0 || offset > std::numeric_limits<int64_t>::max() / elementBytes) {
      return emitError(value.getLoc()) << kInvalidBindingDiagnostic << ": StorageView byte offset overflows int64_t";
    }
    offset *= elementBytes;
  }
  view.offset = offset;
  view.resource.queue = isQueueBinding(view, buffer);
  view.sizes.assign(type.getShape().begin(), type.getShape().end());
  view.strides = std::move(strides);
  return view;
}

LogicalResult CanonicalBindingCollector::recordValue(Value value, TensorDescAttr tensor, BufferDescAttr buffer,
                                                     Operation *sourceOwner, unsigned sourceIndex) {
  if (!value || !isa<MemRefType>(value.getType())) {
    return emitError(value ? value.getLoc() : func.getLoc())
           << kInvalidDescriptorDiagnostic << ": canonical descriptor must bind a memref value";
  }
  auto existing = values.find(value);
  if (existing != values.end()) {
    bool sameTensorFacts = existing->second.tensor == tensor || hasCompatibleGmTensorFacts(existing->second, tensor);
    if (!sameTensorFacts || existing->second.buffer != buffer) {
      return emitError(value.getLoc()) << kInvalidBindingDiagnostic
                                       << ": conflicting canonical descriptors for one SSA value";
    }
    if (getTransferDirection(existing->second.view.sourceOwner) == TransferDirection::None &&
        getTransferDirection(sourceOwner) != TransferDirection::None) {
      existing->second.view.sourceOwner = sourceOwner;
      existing->second.view.sourceIndex = sourceIndex;
    }
    return success();
  }
  FailureOr<StorageView> view = makeStorageView(value, tensor, buffer, sourceOwner, sourceIndex);
  if (failed(view)) {
    return failure();
  }
  CanonicalValueInfo info;
  info.value = value;
  info.tensor = tensor;
  info.buffer = buffer;
  info.view = *view;
  info.ordinal = nextOrdinal++;
  values.try_emplace(value, info);
  valueOrder.push_back(value);
  return success();
}

LogicalResult CanonicalBindingCollector::recordArray(Operation *owner, ValueRange valueRange, StringRef tensorName,
                                                     bool required) {
  FailureOr<ArrayAttr> tensors = getDescriptorArray(owner, tensorName, valueRange.size(), required);
  if (failed(tensors)) {
    return failure();
  }
  // 标量输入没有可绑定的存储：它不是一块 buffer，也就没有 `afir.buffer_desc`。它在
  // 描述子数组里仍然占一格（否则下标就跟操作数错开），但不参与资源绑定。
  //
  // 只对 `afir.operand_descs` 放行——目的地和结果不可能是标量，那条兜底不该一并撤掉。
  // 函数参数那一侧（`afir.arg_descs`）不走本函数，见 `collectFunctionBindings`；
  // 两处说的是同一件事，**改一处就要改另一处**（`AFIRVerifierUtils.cpp` 的
  // `verifyDescriptorList` 上那段注释同理）。
  const bool allowsScalars = tensorName == kOperandDescsAttrName;
  for (auto [index, value] : llvm::enumerate(valueRange)) {
    auto tensor = dyn_cast<TensorDescAttr>((*tensors)[index]);
    if (!tensor) {
      owner->emitError() << kDescriptorTypeMismatchDiagnostic << ": canonical descriptor at index " << index;
      return failure();
    }
    if (allowsScalars && !isa<ShapedType>(value.getType())) {
      continue;
    }
    FailureOr<BufferDescAttr> buffer = requireBufferDescAttr(value, owner);
    if (failed(buffer) || failed(recordValue(value, tensor, *buffer, owner, index))) {
      return failure();
    }
  }
  return success();
}

// Store 写回的是函数参数。它的 tensor 视图取函数的 `arg_descs`——参数在函数边界上的
// 轴解释是函数的事实，不是 Store 这个使用点的事实；buffer descriptor 则和其它值一样
// 由 `recordArray` 顺着定义点（arg attr）取回，不再有第二份拷贝。
LogicalResult CanonicalBindingCollector::recordExternalStoreDestinations(Operation *operation,
                                                                         ValueRange destinations) {
  if (!isa<StoreOp>(operation) || destinations.empty() ||
      !llvm::all_of(destinations, [](Value value) { return isa<BlockArgument>(value); })) {
    return recordArray(operation, destinations, kDestinationDescsAttrName, !destinations.empty());
  }

  auto function = operation->getParentOfType<func::FuncOp>();
  auto argTensors = function ? function->getAttrOfType<ArrayAttr>(kArgDescsAttrName) : ArrayAttr();
  if (!function || !argTensors) {
    return operation->emitError() << kInvalidDescriptorDiagnostic
                                  << ": external store destination is missing function argument descriptors";
  }

  for (Value destination : destinations) {
    auto blockArgument = cast<BlockArgument>(destination);
    if (blockArgument.getArgNumber() >= argTensors.size()) {
      return operation->emitError() << kDescriptorCountMismatchDiagnostic
                                    << ": external store destination argument is outside descriptor arrays";
    }
    auto tensor = dyn_cast<TensorDescAttr>(argTensors[blockArgument.getArgNumber()]);
    if (!tensor) {
      return operation->emitError() << kDescriptorTypeMismatchDiagnostic
                                    << ": external store destination argument descriptor has the wrong type";
    }
    FailureOr<BufferDescAttr> buffer = requireBufferDescAttr(destination, operation);
    if (failed(buffer) || failed(recordValue(destination, tensor, *buffer, operation, blockArgument.getArgNumber()))) {
      return failure();
    }
  }
  return success();
}

// tmp buffer 操作数在 IR 里没有 tensor descriptor：它不描述计算里的任何值，只是被调
// API 要的一段字节。但它仍要占一块片上存储，所以这里给它合成一份退化的视图——无轴、
// 无向量化——让它和别的 UB buffer 走同一条资源分配路径。字节数不看这份视图，而是
// 从它自己 alloc 上的 `size_from_resource` 取。
LogicalResult CanonicalBindingCollector::recordTmpBuffers(Operation *operation, unsigned firstIndex,
                                                          ValueRange tmpBuffers) {
  if (tmpBuffers.empty()) {
    return success();
  }
  auto tensor = TensorDescAttr::get(context, /*axis=*/{}, /*repeats=*/{}, /*strides=*/{}, /*vectorized_axis=*/{},
                                    /*vectorized_strides=*/{}, RoleAttr::get(context, Role::Compute));
  for (auto [offset, value] : llvm::enumerate(tmpBuffers)) {
    FailureOr<BufferDescAttr> buffer = requireBufferDescAttr(value, operation);
    if (failed(buffer) || failed(recordValue(value, tensor, *buffer, operation, firstIndex + offset))) {
      return failure();
    }
  }
  return success();
}

LogicalResult CanonicalBindingCollector::collectFunctionBindings() {
  auto argTensors = func->getAttrOfType<ArrayAttr>(kArgDescsAttrName);
  // 数据参数个数 = 参数总数 − 声明了 ABI 角色的参数个数。此前这里按形状反推——「倒数
  // 第二个是 memref 且倒数第一个是 tiling 结构体就减 2」——那是 A3 之前的做法：一个碰巧
  // 同型的普通参数排在倒数第二位就会被当成 workspace，而 tiling 那一半还依赖结构体的
  // 字面名字。身份由参数自己声明，规则与判据见 `getCanonicalDataArgumentCount`。
  FailureOr<unsigned> dataArguments = getCanonicalDataArgumentCount(func.getOperation(), func.getNumArguments());
  if (failed(dataArguments)) {
    return failure();
  }
  unsigned dataArgumentCount = *dataArguments;
  if (!argTensors || argTensors.size() != dataArgumentCount) {
    return func.emitError() << kDescriptorCountMismatchDiagnostic
                            << ": canonical function arguments require descriptor arrays aligned with ABI";
  }
  for (auto [index, argument] : llvm::enumerate(func.getArguments().take_front(dataArgumentCount))) {
    auto tensor = dyn_cast<TensorDescAttr>(argTensors[index]);
    if (!tensor) {
      return failure();
    }
    // 按值传的标量 kernel 参数（`ScalarData`）没有可绑定的存储：它不是一块 buffer，也就
    // 没有 `afir.buffer_desc`。它在描述子数组里仍然占一格（否则下标就跟参数错开），但
    // 不参与资源绑定——与 `recordArray` 里标量操作数那一条完全同形。
    if (!isa<ShapedType>(argument.getType())) {
      continue;
    }
    FailureOr<BufferDescAttr> buffer = requireBufferDescAttr(argument, func.getOperation());
    if (failed(buffer) || failed(recordValue(argument, tensor, *buffer, func.getOperation(), index))) {
      return failure();
    }
  }
  return success();
}

LogicalResult CanonicalBindingCollector::collectOperationBinding(Operation *operation) {
  if (auto roleInterface = dyn_cast<AFIRRoleInterface>(operation); roleInterface && failed(roleInterface.getRole())) {
    return failure();
  }
  auto destinationStyle = dyn_cast<AFIRDestinationStyleInterface>(operation);
  FailureOr<OperandPartition> partition = getOperandPartition(operation);
  if (failed(partition)) {
    operation->emitError() << kInvalidDescriptorDiagnostic << ": invalid destination count";
    return failure();
  }
  // tmp buffer 操作数不绑定任何东西：它没有 tensor descriptor，buffer descriptor 挂在
  // 自己的 alloc 上，`partition` 已经把它从两段里摘掉了。
  OperandRange destinationValues = partition->destinations(operation);
  if (failed(recordArray(operation, partition->inputs(operation), kOperandDescsAttrName, partition->inputCount != 0)) ||
      failed(recordExternalStoreDestinations(operation, destinationValues)) ||
      failed(recordArray(operation, operation->getResults(), kResultDescsAttrName, operation->getNumResults() != 0)) ||
      failed(
          recordTmpBuffers(operation, partition->inputCount + partition->destinationCount,
                           operation->getOperands().drop_front(partition->inputCount + partition->destinationCount)))) {
    return failure();
  }
  for (unsigned index = 0; index < partition->destinationCount; ++index) {
    FailureOr<unsigned> tied = failure();
    if (destinationStyle) {
      tied = destinationStyle.getTiedResultIndex(index);
    }
    if (failed(tied)) {
      continue;
    }
    if (*tied >= operation->getNumResults()) {
      operation->emitError() << kInvalidDescriptorDiagnostic << ": tied result index out of range";
      return failure();
    }
    auto resultValue = values.find(operation->getResult(*tied));
    auto destinationValue = values.find(destinationValues[index]);
    if (resultValue == values.end() || destinationValue == values.end()) {
      return failure();
    }
    if (resultValue->second.tensor != destinationValue->second.tensor ||
        resultValue->second.buffer != destinationValue->second.buffer) {
      operation->emitError() << kInvalidBindingDiagnostic << ": tied result and destination descriptors differ";
      return failure();
    }
    resultValue->second.alias = destinationValues[index];
  }
  return success();
}

LogicalResult CanonicalBindingCollector::collectOperationBindings() {
  func.walk([&](Operation *operation) {
    // **region 内不进这一层。** `afir.vector_func` 的 region 里是寄存器上的数据流：那些值
    // 没有片上存储，也就没有 buffer descriptor 可绑。不挡住的话这里会按「有 canonical
    // 属性」把它们收进来，再去要一份不存在的 descriptor（`requireBufferDescAttr`）。
    // 算子自己（边界是真的 UB 张量）照常参与绑定，只是不往里走。
    if (failed(result) || isInsideVectorFunc(operation) || !isAfirOperation(operation) ||
        isa<SymbolicEvalOp>(operation) || !hasCanonicalAttrs(operation)) {
      return;
    }
    result = collectOperationBinding(operation);
  });
  return result;
}

LogicalResult CanonicalBindingCollector::collectCanonicalBindings() {
  if (failed(collectFunctionBindings())) {
    return failure();
  }
  return collectOperationBindings();
}

}  // namespace

bool hasCanonicalAttrs(Operation *operation) {
  return operation->hasAttr(kOperandDescsAttrName) || operation->hasAttr(kResultDescsAttrName) ||
         operation->hasAttr(kDestinationDescsAttrName);
}

TransferDirection getTransferDirection(Operation *owner) {
  if (!owner) {
    return TransferDirection::None;
  }
  if (owner->hasTrait<OpTrait::AFIRTransferIn>()) {
    return TransferDirection::In;
  }
  if (owner->hasTrait<OpTrait::AFIRTransferOut>()) {
    return TransferDirection::Out;
  }
  return TransferDirection::None;
}

bool isQueueBinding(const StorageView &view, BufferDescAttr buffer) {
  return view.resource.space == Space::UB && isCanonicalQueueResource(buffer);
}

FailureOr<CanonicalBindings> collectCanonicalBindings(func::FuncOp func) {
  return CanonicalBindingCollector(func).collect();
}

}  // namespace mlir::afir
