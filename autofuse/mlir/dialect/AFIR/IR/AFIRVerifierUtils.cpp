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

// 迁移期的等价性闸门（设计文档 3.1 步 3）。
//
// 地址空间的目标载体是值的类型（memref memory space），`#afir.buffer_desc` 的 `space`
// 字段是被替换的旧编码。二者并存期间必须逐点一致，否则就是 D2 描述的「同一事实两份
// 拷贝」——消费端读哪一份取决于它恰好拿到的是值还是 descriptor，发散时不报错。
//
// **删除条件**：`BufferDescAttr` 的 `space` 字段随步 9（Schedule 迁移为 `transform.afir`）
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

// tensor descriptor 仍按使用点存放——同一个 GM 值在不同使用点可以有不同的轴解释，
// 那是使用点的事实。buffer descriptor 不同：它是**值**的事实，现在只在定义点存一份，
// 由 `getBufferDescAttr` 顺着 SSA 取（设计文档 5.1 步 8）。
// `allowsScalars` 对 `afir.operand_descs` 与 `afir.arg_descs` 两处为真：标量只可能出现在
// 输入里——op 的标量操作数（`afir.add %tile, %scalar`），以及主机按值传进来的标量 kernel
// 参数（AscGraph 的 `ScalarData`）。**目的地与结果不能是标量**，那条兜底不该一并撤掉。
//
// 这一层按**类型**放行（非 `ShapedType`），与操作数那条完全同形。「这个参数的角色是不是
// `scalar_input`」是 ABI 的事实，由 `verifyCanonicalKernelArguments` 判，它在那里把角色与
// 类型绑成一条双向不变量；本函数只管描述子数组与值的下标对得上。
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

// `afir.arg_descs` 应当覆盖的参数个数 —— 即**数据参数**的个数。
//
// 此前这个数由一个独立的函数属性 `afir.num_inputs` 携带。它的名字不实（buffer 形态下
// 它等于 inputs + outputs），真正的语义是「ABI 物化之前有几个参数」，而物化之后这件事
// 已经写在 IR 里了：`workspace` / `tiling` 这两个角色就是物化追加出来的参数（设计文档
// D2 / D5）。于是数据参数个数 = 参数总数 − 这两个角色的参数个数，不需要另存计数。
//
// **期望值必须独立于 `afir.arg_descs` 自己的长度**，否则消费端拿它去和同一个数组比长度
// 就退化成恒真，「少写描述符」再也没有任何东西能发现。这正是本函数第一版的缺陷：它在
// 物化后直接返回 `descriptors.size()`，于是给任意参数挂一个
// `#afir.kernel_arg<role = workspace>` 就能关掉整条闸门。
//
// 三种形态下这条规则都成立，且都有牙齿：
//   物化前            无 ABI 角色参数 → 全部参数都是数据参数；
//   物化后 buffer 形态 [数据…, workspace, tiling]        → 总数 − 2；
//   物化后 tensor 形态 [输入…, 追加的输出…, workspace, tiling] → 总数 − 2。
//     后者要求 `AFIRKernelBodySignature` 在追加输出参数时把 `afir.result_descs` 并入
//     `afir.arg_descs`，否则数组只覆盖输入，与本规则算出的期望值对不上。
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
