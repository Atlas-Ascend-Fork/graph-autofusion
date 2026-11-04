/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- ResourceIds.h - canonical 片上资源的身份集合 -------------*- C++ -*-===//
//
// 一个 kernel 函数用到哪些队列 id、哪些普通缓冲 id。判据全部来自 `#afir.buffer_desc`
// 与 `afir.workspace_plan`，与任何后端无关——所以住在 AFIR 层，尽管今天唯一的消费者是
// AscendC 的 ABI 物化（它拿这两个集合去拼 `q{id}_size` / `b{id}_size` 字段）。
//
// `buffer_id` 不只是资源身份，它同时是 kernel 的 host 侧 ABI（设计文档 2.3 的判别：
// 单候选时它仍决定生成物的 ABI -> 归 `afir`），这是它没有随步 8 一起删除的原因。
//
// 「哪些 descriptor 字段在同一个资源上必须一致」也住在这里（`canonicalResourceFactsAgree`）：
// 它与上面两件是同一类判据——只看 `#afir.buffer_desc`，与后端无关，而且两侧各查一遍。
//
//===----------------------------------------------------------------------===//

#ifndef AUTOFUSE_MLIR_DIALECT_AFIR_ANALYSIS_RESOURCEIDS_H
#define AUTOFUSE_MLIR_DIALECT_AFIR_ANALYSIS_RESOURCEIDS_H

#include <cstdint>
#include <limits>
#include <numeric>
#include <set>
#include <string>
#include <tuple>

#include "common/tiling_schema/canonical_resource_names.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::afir {

// 一个片上资源的 canonical 尺寸字段名——`#afir.buffer_desc` 的 `size_from_resource`
// 引用的就是它，host 侧 tiling 往这个字段里填聚合后的 slot 字节数。
//
// 这个名字是与 ATT 的 ABI，两侧必须逐字相同，所以拼法只此一处：`AFIRKernelBodySignature`
// 按它往 TilingData 里追加字段，memory allocate 按它把 `afir.tiling_param` 绑回来。
// 此前两处各拼一遍（一处还是手写字符串拼接），对不上时唯一的症状是很晚才发的
// 「TilingData is missing canonical size expression field」。
//
// 另一半在 `common/tiling_schema/canonical_resource_names.h`，与 legacy Codegen 共用。
inline std::string makeCanonicalResourceSizeName(bool queue, int64_t bufferId) {
  return queue ? mlir::common::MakeQueueSizeName(bufferId) : mlir::common::MakeBufferSizeName(bufferId);
}

// 收集函数用到的 canonical 片上资源 id，按「是不是队列」分成两组。判据见
// `isCanonicalQueueResource`。
//
// GM 跳过（它不是片上资源）；GM / UB 之外的地址空间报错——L1 / L0A / L0B / L0C 要等
// Cube 路线的资源模型，静默忽略会得到一个少了几个字段的 TilingData。
//
// 只看值上的 descriptor。`afir.workspace_plan` 不参与：那是 host 分配的 GM 暂存，
// verifier 已要求它的每个条目都是 GM；片上的临时量是带 descriptor 的 `memref.alloc`，
// 走这条路进来。
LogicalResult collectCanonicalResourceIds(func::FuncOp func, std::set<int64_t> &queueIds, std::set<int64_t> &bufferIds);

// 这块存储是不是队列：只看 descriptor 自己的 `slot_count` / `queue_depth`。
//
// 「队列」在 AFIR 这一层的含义就是「要按 slot 轮转」，而这件事只有 descriptor 说了算——
// bridge 对队列必设 `queue_depth` / `slot_count ≥ 1`（ScheduleToAfir 的 `BindingDomain`
// 分支），两者都缺省的就是一块普通 UB buffer。TQue 还是 TBuf 由物化侧据此选，VECIN /
// VECOUT 才要回头看搬运方向（`getTransferDirection`）。
//
// 判据必须只依赖 descriptor：`size_from_resource` 要用它决定字段名是 `q<id>_size` 还是
// `b<id>_size`，而这个名字 host 侧也要拼一遍。掺进「谁在读写这个值」之后，同一块存储在
// 不同 pass 里就可能得到不同答案——建字段的一侧（本文件）看全部 user，取符号的一侧
// （`isQueueBinding`）看绑定点的 owner，漂了就是 kernel 读的字段与 host 填的字段对不上。
// 现在两边都退化成同一个表达式，`isQueueBinding` 直接转调这里。
bool isCanonicalQueueResource(BufferDescAttr buffer);

// 同一个 canonical 资源上的多个张量，descriptor 里必须逐字一致的那几个字段。
//
// **这张清单只此一处。** 两侧各查一遍：bridge 在 DTO 上查
// （`ScheduleToAfir` 的 `CheckResourceCompatibility`，判成 `BindingConflict`），conversion
// 在聚合 descriptor 时再查一遍（`CanonicalResources` 的 `registerResource`，判成
// "conflicting descriptors"）。此前两处各写一份字段列表，分开写就会分开漂，而且漂了
// 之后的症状是静默的——松的那一侧照样把图放过去。
//
// 不在清单里的三类字段，各有各的理由：
//
//   - `slot_count`：每个张量说自己要几格，队列的格数是这些数的**最大值**
//     （legacy `codegen_kernel.cpp` 对 `que_buf_num_value` 取 max）。聚合在
//     `registerResource` 里做；
//   - `alignment_bytes`：它不是资源的身份，而是**某个张量对这块存储字节数的要求**
//     （-1 = 没有要求），所以多个张量共用一块存储时，这块存储要同时满足它们——聚合取
//     最小公倍数，同样在 `registerResource` 里做。要求它们相等会拒掉合法调度：
//     tmp buffer 与普通计算缓冲**同在一个 id 空间**（见 `ScheduleToAfir` 里那段注释），
//     前者由 bridge 固定带 32、后者是 -1，于是「tmp buffer 与某个中间结果落到同一个
//     `b<id>`」这种调度就只差这一个字段。单输入单输出算子有两个消费者时就是这个形状
//     （探针 T5c / T5e / T5，2026-10-12）。聚合之后这块存储只会更大不会更小
//     （`ceil32(max(a, b)) >= ceil32(a)`），所以任何一个成员都装得下；
//   - `reuse_id`：legacy 的 `que->share_group[reuse_id]` 本来就允许一条队列上并存多个
//     复用组，所以它是资源身份的一部分而不是要比相等的字段；
//   - **张量的 dtype**：队列就是一块字节，legacy 的 `oneBufSize` 取成员的 max，所以
//     一条队列上混 dtype 是合法调度——ui8 掩码与两边的 f32 一起搬入就是这个形状
//     （`load_mask_select_store`）。片上资源的字节数由 ATT 填进 `q<id>_size` /
//     `b<id>_size`（`size_from_resource`），descriptor 上没有一处依赖 dtype。
struct CanonicalResourceFacts {
  Space space = Space::GM;
  int64_t queueDepth = -1;
  int64_t lifeTimeAxisId = std::numeric_limits<int64_t>::min();
  // 与 `size_from_resource` 互斥的那一半；取不到时是空串，两侧都按空串比。
  llvm::StringRef sizeBytesExpr;
  bool sizeFromResource = false;
};

inline bool canonicalResourceFactsAgree(const CanonicalResourceFacts &first, const CanonicalResourceFacts &second) {
  return std::tie(first.space, first.queueDepth, first.lifeTimeAxisId, first.sizeBytesExpr, first.sizeFromResource) ==
         std::tie(second.space, second.queueDepth, second.lifeTimeAxisId, second.sizeBytesExpr,
                  second.sizeFromResource);
}

// 两个张量各自对同一块存储字节数的对齐要求合成一条，理由见上面那份清单里
// `alignment_bytes` 那一条。-1 是「没有要求」，按 1 参与，所以两边都没有要求时结果仍是 -1。
//
// 取最小公倍数而不是 max：要求的含义是「字节数必须是 N 的倍数」，48 与 32 的公共解是 96，
// 取 48 会让 32 那一条落空。**今天分不出来**——`alignment_bytes` 实际取值只有 -1 与 32
// （32 是 bridge 的 `kTmpBufferAlignmentBytes`），两种聚合同解，所以这条判据的证据在
// UT 里（`CanonicalResourceFactsTest`），不在 lit 或 E2E 上。
inline int64_t aggregateAlignmentBytes(int64_t first, int64_t second) {
  if (first == -1 && second == -1) {
    return -1;
  }
  return std::lcm(first == -1 ? 1 : first, second == -1 ? 1 : second);
}

inline CanonicalResourceFacts getCanonicalResourceFacts(BufferDescAttr buffer) {
  CanonicalResourceFacts facts;
  facts.space = buffer.getSpace().getValue();
  facts.queueDepth = buffer.getQueueDepth();
  facts.lifeTimeAxisId = buffer.getLifeTimeAxisId();
  if (ExprAttr expression = buffer.getSizeBytesExpr()) {
    facts.sizeBytesExpr = expression.getValue().getValue();
  }
  facts.sizeFromResource = buffer.getSizeFromResource();
  return facts;
}

}  // namespace mlir::afir

#endif  // AUTOFUSE_MLIR_DIALECT_AFIR_ANALYSIS_RESOURCEIDS_H
