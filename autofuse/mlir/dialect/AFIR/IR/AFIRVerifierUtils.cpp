/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- AFIRVerifierUtils.cpp - Shared AFIR verifier helpers -----*- C++ -*-===//

#include "dialect/AFIR/IR/AFIRVerifierUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

using namespace mlir;
using namespace mlir::afir;

namespace {

FailureOr<Role> getDescriptorRole(Operation *op, StringRef descriptorName) {
  auto descriptors = op->getAttrOfType<ArrayAttr>(descriptorName);
  if (!descriptors || descriptors.empty()) {
    return failure();
  }
  std::optional<Role> role;
  for (Attribute attr : descriptors) {
    auto descriptor = dyn_cast<TensorDescAttr>(attr);
    if (!descriptor) {
      return op->emitOpError() << kDescriptorTypeMismatchDiagnostic << ": '" << descriptorName
                               << "' target descriptor must be a TensorDescAttr";
    }
    Role current = descriptor.getRole().getValue();
    if (role && *role != current) {
      return op->emitOpError() << kInvalidDescriptorDiagnostic << ": target descriptors must have one common role";
    }
    role = current;
  }
  return *role;
}

// 迁移期的等价性门禁（设计文档 3.1 步 3）。
//
// 地址空间的目标载体是值的类型（memref memory space），`#afir.buffer_desc` 的 `space`
// 字段是被替换的旧编码。二者并存期间必须逐点一致，否则就是 D2 描述的「同一事实两份
// 拷贝」——消费端读哪一份取决于它恰好拿到的是值还是 descriptor，发散时不报错。
//
// 删除条件：`BufferDescAttr` 的 `space` 字段随步 9（Schedule 迁移为 `transform.afir`）
// 一并删除——届时 tensor 形态不再携带调度结果，space 只存在于 buffer 形态的类型上，
// 本函数连同两处并存的容忍一起消失。详见设计文档 5.1「步 3 实施结论」。
//
// 两种情况下不校验，都是迁移期的形态，步 3 要求它们可继续解析，使这一步可单独回退：
//   - 类型没有携带 memory space：bridge 尚未改造前产出的 memref 就是这样；
//   - memory space 不是 `#afir.space<*>`：`AFIRKernelBodySignature` 目前会把 kernel
//     参数的地址空间直接换成 AscendC 的 CCE-C 魔数，而它在生产流水线里排在 AFIR
//     op 全部消失之前（`addPlanCandidateLoweringPipeline`），因此这种形态会真实出现
//     在带 descriptor 的 AFIR op 上。I4 把该魔数收为单点映射、AFIR 侧全程用
//     `#afir.space` 之后，这条容忍要收紧为拒绝。
LogicalResult verifyMemorySpaceAgreement(Operation *op, llvm::StringRef tensorName, size_t index, Type type,
                                         BufferDescAttr bufferDescriptor) {
  auto spaceAttr = getMemorySpaceAttr(type);
  if (!spaceAttr) {
    return success();
  }
  SpaceAttr descriptorSpace = bufferDescriptor.getSpace();
  if (descriptorSpace && descriptorSpace != spaceAttr) {
    return op->emitOpError() << kInvalidDescriptorDiagnostic << ": '" << tensorName << "' element " << index
                             << " memref memory space " << spaceAttr << " disagrees with descriptor space "
                             << descriptorSpace;
  }
  return success();
}

// tensor descriptor 描述使用点的轴解释，同一 GM 值在不同使用点可以不同。
// buffer descriptor 描述存储本身，仅保存在定义点，通过 getBufferDescAttr 沿 SSA 查询。
// operand_descs 和 arg_descs 允许标量，分别对应 op 标量输入和 ScalarData 按值参数；
// destination 和结果仍要求存储描述，不能同时放宽。
LogicalResult verifyDescriptorList(Operation *op, llvm::StringRef tensorName, ValueRange values,
                                   bool allowsScalars = false) {
  if (values.empty()) {
    return success();
  }

  auto tensorDescriptors = op->getAttrOfType<ArrayAttr>(tensorName);
  if (!tensorDescriptors) {
    return op->emitOpError() << kInvalidDescriptorDiagnostic << ": buffer form requires '" << tensorName << "'";
  }
  if (tensorDescriptors.size() != values.size()) {
    return op->emitOpError() << kDescriptorCountMismatchDiagnostic << ": '" << tensorName << "' must contain "
                             << values.size() << " element";
  }

  for (auto [index, value] : llvm::enumerate(values)) {
    auto tensorDescriptor = dyn_cast<TensorDescAttr>(tensorDescriptors[index]);
    if (!tensorDescriptor) {
      return op->emitOpError() << kDescriptorTypeMismatchDiagnostic << ": descriptor at '" << tensorName << "' element "
                               << index << " has an unexpected attribute type";
    }
    Type type = value.getType();
    auto shapedType = dyn_cast<ShapedType>(type);
    // 标量操作数占一个位置但没有可描述的事实：没有轴、没有缓冲、没有地址空间。它仍然
    // 要在列表里占一格，否则描述子下标就跟操作数下标错开了——错开之后每一条检查都还
    // 在跑，只是查的是别的操作数。
    if (allowsScalars && !shapedType) {
      continue;
    }
    if (!shapedType || !shapedType.hasRank()) {
      return op->emitOpError() << kInvalidDescriptorDiagnostic << ": '" << tensorName << "' element " << index
                               << " requires a ranked shaped value";
    }
    ArrayRef<int64_t> axis = tensorDescriptor.getAxis();
    if (!axis.empty() && axis.size() != static_cast<size_t>(shapedType.getRank())) {
      return op->emitOpError() << kInvalidDescriptorDiagnostic << ": '" << tensorName << "' element " << index
                               << " axis count " << axis.size() << " must match shaped rank " << shapedType.getRank();
    }
    BufferDescAttr bufferDescriptor = getBufferDescAttr(value);
    if (!bufferDescriptor) {
      return op->emitOpError() << kInvalidDescriptorDiagnostic << ": '" << tensorName << "' element " << index
                               << " has no '" << kBufferDescAttrName << "' at its definition";
    }
    if (failed(verifyMemorySpaceAgreement(op, tensorName, index, type, bufferDescriptor))) {
      return failure();
    }
  }
  return success();
}

}  // namespace

FailureOr<Role> mlir::afir::getCanonicalTargetRole(Operation *op) {
  std::optional<Role> destinationRole;
  std::optional<Role> resultRole;
  if (op->hasAttr(kDestinationDescsAttrName)) {
    FailureOr<Role> role = getDescriptorRole(op, kDestinationDescsAttrName);
    if (failed(role)) {
      return failure();
    }
    destinationRole = *role;
  }
  if (op->hasAttr(kResultDescsAttrName)) {
    FailureOr<Role> role = getDescriptorRole(op, kResultDescsAttrName);
    if (failed(role)) {
      return failure();
    }
    resultRole = *role;
  }
  if (!destinationRole && !resultRole) {
    return op->emitOpError() << kInvalidDescriptorDiagnostic
                             << ": canonical target role requires destination_descs or result_descs";
  }
  if (destinationRole && resultRole && *destinationRole != *resultRole) {
    return op->emitOpError() << kInvalidDescriptorDiagnostic << ": destination and result target roles must match";
  }
  return destinationRole ? *destinationRole : *resultRole;
}

// 独立计算 arg_descs 应覆盖的数据参数个数，不能从描述符数组长度反推，
// 否则长度校验恒为真，缺少描述符也无法发现。
// 不再另存 afir.num_inputs：物化追加的参数已通过 workspace / tiling 角色声明身份
// （设计文档 D2、D5）。物化前全部参数为数据参数；物化后排除 ABI 角色参数，
// buffer 形态和 tensor 形态均可按同一规则计算。
// tensor 输出追加为参数时，AFIRKernelBodySignature 必须将 result_descs 合并到
// arg_descs，使其覆盖全部数据参数，而不只是原来的输入。
FailureOr<unsigned> mlir::afir::getCanonicalDataArgumentCount(Operation *owner, unsigned argumentCount) {
  auto function = dyn_cast<FunctionOpInterface>(owner);
  if (!function) {
    return argumentCount;
  }
  unsigned abiArgumentCount = 0;
  for (unsigned index = 0; index < function.getNumArguments() && index < argumentCount; ++index) {
    KernelArgAttr argument = getKernelArgAttr(function, index);
    // `workspace_offset` 与 workspace / tiling 同类：ABI 追加出来的标量参数，不描述任何
    // 张量，因此不在 `afir.arg_descs` 的覆盖范围里。
    if (argument && (argument.getRole() == KernelArgRole::Workspace || argument.getRole() == KernelArgRole::Tiling ||
                     argument.getRole() == KernelArgRole::WorkspaceOffset)) {
      ++abiArgumentCount;
    }
  }
  if (abiArgumentCount > argumentCount) {
    owner->emitError() << kInvalidDescriptorDiagnostic << ": more ABI arguments than arguments";
    return failure();
  }
  return argumentCount - abiArgumentCount;
}

LogicalResult verifyCanonicalFunctionArguments(Operation *op) {
  auto function = op->getParentOfType<func::FuncOp>();
  if (!function) {
    return op->emitOpError() << kInvalidDescriptorDiagnostic << ": buffer form requires a func.func owner";
  }
  FailureOr<unsigned> canonicalInputCount = getCanonicalDataArgumentCount(function, function.getNumArguments());
  if (failed(canonicalInputCount)) {
    return failure();
  }
  // 标量 kernel 参数（`ScalarData`）占一格描述子但没有缓冲，与标量操作数同形。
  return verifyDescriptorList(function.getOperation(), kArgDescsAttrName,
                              function.getArguments().take_front(*canonicalInputCount),
                              /*allowsScalars=*/true);
}

LogicalResult mlir::afir::verifyCanonicalBufferDescriptors(Operation *op) {
  bool hasMemRef = false;
  bool hasNonMemRef = false;
  // 这条查的是「半个 buffer 形态」：一部分操作数已经是 memref、另一部分还是遗留的
  // `!ascendc.local_tensor`。**标量操作数不在其中**——它压根不是一块缓冲，两种形态下
  // 都是同一个值。把它算进来的话，`afir.add %tile, %scalar` 的 buffer 形态整个建不
  // 起来，而它没有任何不一致。
  auto classifyType = [&](Type type) {
    if (!isa<ShapedType>(type)) {
      return;
    }
    hasMemRef |= isa<MemRefType>(type);
    hasNonMemRef |= !isa<MemRefType>(type);
  };
  llvm::for_each(op->getOperandTypes(), classifyType);
  llvm::for_each(op->getResultTypes(), classifyType);
  if (!hasMemRef) {
    return success();
  }
  if (hasNonMemRef) {
    return op->emitOpError() << kInvalidDescriptorDiagnostic
                             << ": canonical buffer form cannot mix memref and legacy shaped types";
  }

  if (!isa<AFIRDestinationStyleInterface>(op)) {
    return op->emitOpError() << kInvalidDescriptorDiagnostic << ": buffer form requires AFIRDestinationStyleInterface";
  }
  FailureOr<OperandPartition> partition = getOperandPartition(op);
  if (failed(partition)) {
    return op->emitOpError() << kInvalidDescriptorDiagnostic << ": destination count exceeds operand count";
  }

  if (failed(verifyDescriptorList(op, kOperandDescsAttrName, partition->inputs(op), /*allowsScalars=*/true)) ||
      failed(verifyDescriptorList(op, kResultDescsAttrName, op->getResults())) ||
      failed(verifyDescriptorList(op, kDestinationDescsAttrName, partition->destinations(op)))) {
    return failure();
  }

  return verifyCanonicalFunctionArguments(op);
}
