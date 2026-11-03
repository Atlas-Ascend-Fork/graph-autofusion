/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- CanonicalBindings.h - canonical 形态的绑定收集 -----------*- C++ -*-===//
//
// 「这个 kernel 函数声明了哪些 canonical 存储，它们彼此一致吗」。
//
// 输入是 `#afir.tensor_desc` / `#afir.buffer_desc` 与它们挂在哪个值上，输出是一张
// 值 -> 绑定信息的表。判据全部来自 AFIR 自己的 descriptor 契约，与任何后端无关——所以
// 这一相住在 AFIR 层，尽管今天唯一的消费者是 AscendC 的资源物化。
//
// **本层只填分析字段。** `CanonicalValueInfo::alias` 与 `materialized` 由下游的物化器
// 回填；把它们留在同一个结构体里而不是另起一张并行表，是因为它们描述的是同一个值，
// 拆成两张表之后「哪个 alias 对应哪条绑定」会变成第二处需要维护的对应关系（D2）。
// `Value` 是核心 MLIR 类型，留着它们不会把后端拉进这一层。
//
//===----------------------------------------------------------------------===//

#ifndef AUTOFUSE_MLIR_DIALECT_AFIR_ANALYSIS_CANONICALBINDINGS_H
#define AUTOFUSE_MLIR_DIALECT_AFIR_ANALYSIS_CANONICALBINDINGS_H

#include <cstdint>
#include <map>
#include <tuple>

#include "dialect/AFIR/IR/AFIR.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

namespace mlir::afir {

struct CanonicalResourceKey {
  Space space = Space::GM;
  bool queue = false;
  int64_t bufferId = -1;

  bool operator<(const CanonicalResourceKey &other) const {
    return std::tie(space, queue, bufferId) < std::tie(other.space, other.queue, other.bufferId);
  }
};

enum class TransferDirection {
  None,
  In,
  Out,
};

struct CanonicalGroupKey {
  CanonicalResourceKey resource;
  int64_t reuseId = -1;
  int64_t lifeTimeAxisId = -1;
  // 同一组里的成员是**并排**放的：物化时依次累加偏移。tmp buffer 不是这个关系——调度给
  // 两个节点同一个 buffer id，说的是「先后复用同一块」，而 ATT 的聚合口径正是跨节点
  // 取 max（`buf_occupy_expr.cpp`），整块只按一个节点的需要算过一次。并排放会让第二块
  // 整个落在资源之外，而且没有任何断言拦得住（偏移检查只查算术溢出）。
  //
  // 所以 tmp buffer 成员各自成组：这里填它自己的序号，组与组之间互不相邻，各自从 0 开始。
  // 非 tmp buffer 一律 -1，行为与加这个字段之前逐字相同。
  int64_t exclusiveMemberOrdinal = -1;

  bool operator<(const CanonicalGroupKey &other) const {
    return std::tie(resource, reuseId, lifeTimeAxisId, exclusiveMemberOrdinal) <
           std::tie(other.resource, other.reuseId, other.lifeTimeAxisId, other.exclusiveMemberOrdinal);
  }
};

struct StorageView {
  CanonicalResourceKey resource;
  Operation *sourceOwner = nullptr;
  unsigned sourceIndex = 0;
  SmallVector<int64_t> axis;
  int64_t offset = ShapedType::kDynamic;
  SmallVector<int64_t> sizes;
  SmallVector<int64_t> strides;
};

struct CanonicalValueInfo {
  Value value;
  TensorDescAttr tensor;
  BufferDescAttr buffer;
  StorageView view;
  Value alias;
  Value materialized;
  unsigned ordinal = 0;
};

// 一次收集的结果：值的绑定表加一个稳定的遍历顺序。
//
// `valueOrder` 不是可有可无的：`values` 是 DenseMap，遍历顺序由指针决定，而下游要按它
// 建资源、发算子。少了它，同一份输入两次编译会得到顺序不同的 IR。
struct CanonicalBindings {
  llvm::DenseMap<Value, CanonicalValueInfo> values;
  SmallVector<Value> valueOrder;
};

// 这个 Op 带 canonical descriptor 吗。判据只问 AFIR 的契约，收集侧与物化侧都要用。
// （「它是不是 AFIR 的算子」问的是方言归属，不是绑定的事，住在 AFIR.h。）
bool hasCanonicalAttrs(Operation *operation);

// 这个 Op 是搬入、搬出，还是都不是。物化侧要按它给队列定位置（VECIN / VECOUT），
// 所以是自由函数而不是物化器的私有 helper。队列身份不看它——见 `isQueueBinding`。
TransferDirection getTransferDirection(Operation *owner);

// 这个绑定点上的存储是不是队列。判据就是 `isCanonicalQueueResource`（descriptor 的
// `slot_count` / `queue_depth`），这里只多一条：非 UB 的空间不可能是队列。
//
// 不看 `view.sourceOwner` 的搬运方向。一个值可以在多个点被绑定，`recordValue` 只在
// 后来遇到搬运 owner 时改 `sourceOwner`，而 `view.resource.queue` 是建 StorageView 时
// 定的、不会重算——判据掺进 owner 就意味着同一块存储的队列身份取决于哪个绑定点先被
// 收集到。字段名（`q<id>_size` / `b<id>_size`）正是按这个身份拼的，两处答案不一样时
// kernel 侧取的字段与 host 侧填的字段就对不上。
bool isQueueBinding(const StorageView &view, BufferDescAttr buffer);

// 收集并校验一个函数的 canonical 绑定。失败时诊断已挂在具体的 Op / 值上。
FailureOr<CanonicalBindings> collectCanonicalBindings(func::FuncOp func);

}  // namespace mlir::afir

#endif  // AUTOFUSE_MLIR_DIALECT_AFIR_ANALYSIS_CANONICALBINDINGS_H
