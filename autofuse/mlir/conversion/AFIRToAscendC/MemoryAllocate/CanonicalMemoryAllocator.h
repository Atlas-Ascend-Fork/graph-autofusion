/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- CanonicalMemoryAllocator.h - AFIR memory allocation ------*- C++ -*-===//
//
// Internal to the AFIRMemoryAllocate pass. The allocator's phases are split
// across TUs (CanonicalBindings / CanonicalResources / CanonicalMaterialization
// / CanonicalRewrite) because one 1100-line file made the phase order that
// run() encodes impossible to follow. This header is the shared declaration
// those TUs implement; it is not part of the conversion's public surface.
//
//===----------------------------------------------------------------------===//

#ifndef AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_MEMORYALLOCATE_CANONICALMEMORYALLOCATOR_H
#define AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_MEMORYALLOCATE_CANONICALMEMORYALLOCATOR_H

#include <map>
#include <memory>
#include <tuple>
#include <utility>

#include "dialect/AFIR/Analysis/CanonicalBindings.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/IR/SymbolicBuilder.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

namespace mlir::ascend::kernel::detail {

// 数据结构与绑定收集住在 AFIR 层（判据全部来自 descriptor 契约，与后端无关）。
// 本头文件只剩物化器那一半——它生成 `ascendc` 算子，所以留在 conversion。
using afir::CanonicalBindings;
using afir::CanonicalGroupKey;
using afir::CanonicalResourceKey;
using afir::CanonicalValueInfo;
using afir::StorageView;
using afir::TransferDirection;

struct CanonicalResourceInfo {
  CanonicalResourceKey key;
  afir::BufferDescAttr descriptor;
  Value resource;
  Value raw;
  SmallVector<Value> values;
  TransferDirection queueDirection = TransferDirection::None;
  unsigned queueOrdinal = 0;
};

class CanonicalFunctionMemoryAllocator {
 public:
  explicit CanonicalFunctionMemoryAllocator(func::FuncOp func) : func(func), context(func.getContext()) {}

  LogicalResult run();

 private:
  // 值的地址空间从它的类型读（设计文档 3.1 步 3）。`StorageView::resource.space`
  // 因此成为本 pass 内唯一的空间事实，其余阶段一律读它而不再回到 descriptor。
  //
  // 类型尚未携带 memory space 时回落到 descriptor：这是迁移期兼容，供未改造的生产者
  // 与手写 fixture 使用。两者都存在时必然一致——`verifyCanonicalBufferDescriptors`
  // 已把分歧挡在 pass 入口之外，所以这里不需要再做交叉校验。

  LogicalResult validateResourceProperties(const CanonicalValueInfo &info, bool queue);

  LogicalResult registerResource(const CanonicalValueInfo &info, CanonicalResourceKey key, bool queue);

  LogicalResult validateResource(const CanonicalValueInfo &info);

  LogicalResult validateCanonicalBindings();

  // 把一个尺寸表达式里的自由符号并进 symbols。
  LogicalResult collectExpressionSymbols(afir::ExprAttr expression, llvm::StringSet<> &symbols);

  // 本函数的资源尺寸一共引用了哪些 tiling 符号：AscGraph 的 size_var，加上每个值自己的
  // `size_bytes_expr` / `size_from_resource`。
  LogicalResult collectRequiredSizeSymbols(llvm::StringSet<> &requiredSymbols);

  // 把这些符号逐个绑到 TilingData 的同名字段上；缺任何一个字段即报错。
  LogicalResult bindSizeSymbolsToTiling(const llvm::StringSet<> &requiredSymbols);

  LogicalResult collectSymbolicBindings();

  FailureOr<Value> materializeExpression(afir::ExprAttr expression, Location loc);

  // `size_from_resource` 解析成哪个字段（`q<id>_size` / `b<id>_size`）只由 descriptor
  // 决定：id 是它的 `buffer_id`，队列身份是 `isCanonicalQueueResource`。
  FailureOr<Value> materializeResourceSize(const afir::BufferDescAttr &descriptor, Location loc);

  FailureOr<Value> alignResourceSize(Value size, int64_t alignment, Location loc);

  Operation *firstAfirOperation();
  Operation *firstMaterializationAnchor();

  ascendc::TPosition getPosition(const CanonicalResourceInfo &resource) const;

  LogicalResult createQueueResource(Value pipe, CanonicalResourceInfo &resource);

  LogicalResult createBufferResource(Value pipe, CanonicalResourceInfo &resource);

  LogicalResult createCanonicalResources();

  Operation *valueAnchor(Value value);

  static bool isInLoopScope(Operation *operation, int64_t axisId);

  Value resolveAlias(Value value) const;

  static bool isContiguousView(MemRefType type);

  bool isWorkspaceValue(const CanonicalValueInfo &info) const;
  FailureOr<Value> materializeWorkspaceBase(OpBuilder &builder, const CanonicalValueInfo &info, MemRefType type);
  LogicalResult materializeGlobalValue(CanonicalValueInfo &info, MemRefType type);

  LogicalResult collectMaterializationGroup(CanonicalValueInfo &info,
                                            std::map<CanonicalGroupKey, SmallVector<CanonicalValueInfo *>> &groups);

  // 同一队列中复用一次 AllocTensor 的资源组，对应 legacy reuse_from 链。
  // 当前组是无需入队和归还的 VecCalc 时，下一组直接使用同一块存储，只需 PIPE_V 屏障。
  // 整条链只分配和归还一次，见 AllocOutputs 的 VecCalc / HandleVecOutAlloc 分支。
  struct QueueEpoch {
    Value raw;
    Operation *anchor = nullptr;
    SmallVector<CanonicalValueInfo *> members;
    // 上一组是搬运成员吗。是的话它会被 EnQue / DeQue / 归还，链到此为止，下一组必须
    // 自己再 alloc 一次。
    bool transferred = false;
  };

  FailureOr<Value> createLocalRaw(OpBuilder &builder, const CanonicalGroupKey &key, CanonicalResourceInfo &resource,
                                  Operation *anchor, ascendc::LocalTensorType rawType,
                                  const SmallVector<CanonicalValueInfo *> &members);

  // 走查这一组落到的所有绑定，别名一并算进来——`outs` 与它 tie 的结果是同一块
  // 存储，而搬运方向、末次读都记在结果那一侧。下面两个判据共用这一条闭包。
  void forEachGroupValue(const SmallVector<CanonicalValueInfo *> &members,
                         llvm::function_ref<void(const CanonicalValueInfo &)> callback) const;

  // 这一组会被搬运（EnQue / DeQue）吗。
  bool isTransferGroup(const SmallVector<CanonicalValueInfo *> &members) const;

  // 归还这条链：在它全部成员的末次使用之后生成一条 free_tensor。
  LogicalResult closeQueueEpoch(const CanonicalResourceKey &key, QueueEpoch &epoch);

  // 带溢出断言的加法 / 乘法：除结果外还生成一条 `cf.assert`，把「进位为零」「高位为零」
  // 钉在 IR 里，而不是留给运行期悄悄回绕。
  FailureOr<Value> materializeCheckedAdd(Value lhs, Value rhs, StringRef message, Location loc);
  FailureOr<Value> materializeCheckedMul(Value lhs, Value rhs, StringRef message, Location loc);

  FailureOr<int64_t> queueMemberElementBytes(const CanonicalValueInfo &member, Location loc);

  // 一条向量化轴贡献的元素间隔 (repeat - 1) * stride。stride 字面量为 0 的轴不贡献，
  // 返回空 Value。
  FailureOr<Value> materializeVectorizedAxisSpan(const CanonicalValueInfo &member, size_t index, Value one,
                                                 Location loc);

  FailureOr<Value> materializeQueueMemberElementCount(const CanonicalValueInfo &member, Location loc);

  FailureOr<Value> materializeLocalMemberSize(const CanonicalValueInfo &member, Location loc, bool queue);

  // 正在被逐个成员切分的那块裸存储：整块 local tensor 与它的类型，切到哪个字节偏移
  // （每切一个成员往后推），以及它来自队列资源还是普通 buffer 资源。
  struct LocalSlab {
    Value raw;
    ascendc::LocalTensorType rawType;
    Value offset;
    bool queue;
  };

  LogicalResult materializeLocalMember(OpBuilder &builder, LocalSlab &slab, CanonicalValueInfo &member);

  // 这些队列成员在 `block` 里最后一次被读到的地方。
  Operation *lastQueueGroupUse(const SmallVector<CanonicalValueInfo *> &members, Block *block) const;

  LogicalResult materializeLocalGroup(const CanonicalGroupKey &key, SmallVector<CanonicalValueInfo *> &members);

  LogicalResult materializeCanonicalValues();

  Value materializedValue(Value value) const;

  SmallVector<Operation *> collectCanonicalOperations();

  LogicalResult rewriteCanonicalOperation(Operation *operation);

  void eraseCanonicalOperations(ArrayRef<Operation *> operations);

  LogicalResult rewriteCanonicalOperations();

  LogicalResult lowerSymbolicExpressions();

  // 成功路径没有对应的 `commit()`：此前它只写 `afir.memory_allocated` /
  // `afir.ascendc_materialized` 两个标记属性，验收 8 删除标记之后函数体为空。
  // 「物化过没有」现在直接问 IR，见 conversion/AFIRToAscendC/AFIRToAscendCLoweringStage.h。
  void rollback();

  func::FuncOp func;
  MLIRContext *context;
  std::unique_ptr<OpBuilder> resourceBuilder;
  // 绑定由 AFIR 层收集好之后交进来（dialect/AFIR/Analysis/CanonicalBindings.h）。
  // 本类只做物化：读这张表，生成 `ascendc` 算子，把 `alias` / `materialized` 回填进去。
  llvm::DenseMap<Value, CanonicalValueInfo> values;
  SmallVector<Value> valueOrder;
  std::map<CanonicalResourceKey, CanonicalResourceInfo> resources;
  std::map<CanonicalResourceKey, QueueEpoch> queueEpochs;
  afir::SymbolBindingMap symbolicBindings;
  SmallVector<Operation *> createdOps;
  SmallVector<Operation *> newOperations;
  SmallVector<std::pair<Value, Value>> replacements;
  Operation *materializationAnchor = nullptr;
  unsigned nextOrdinal = 0;
  LogicalResult result = success();
};

}  // namespace mlir::ascend::kernel::detail

#endif  // AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_MEMORYALLOCATE_CANONICALMEMORYALLOCATOR_H
