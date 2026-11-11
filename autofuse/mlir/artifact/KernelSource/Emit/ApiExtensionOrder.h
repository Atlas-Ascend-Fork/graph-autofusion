/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AUTOFUSE_MLIR_ARTIFACT_KERNELSOURCE_EMIT_APIEXTENSIONORDER_H
#define AUTOFUSE_MLIR_ARTIFACT_KERNELSOURCE_EMIT_APIEXTENSIONORDER_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <map>
#include <set>
#include <utility>

namespace mlir::ascend::kernel {

// Order Autofuse AscendC API extension headers for emission.
//
// 扩展头以文本内联，彼此不 include，但可能相互调用。例如 where.h 调用的 Duplicate
// 定义在 duplicate.h 中，因此需要按依赖顺序生成。
//
// 每个 op 的声明列表中，相邻的 headers[i] 和 headers[i + 1] 构成一条顺序约束。
// resolve 对所有头做拓扑排序，就绪节点按字典序选择：
//   - 合并跨 op 的约束。airy_ai 将 trigonometric_function_utils 列为首项，bessel_j0
//     则要求 bessel_j_utils 在它之前；按首次出现顺序会依赖 plan 的遍历顺序，
//     单纯按字典序排序又会忽略声明的依赖。
//   - 无依赖的头按名称排序，输出不受图中 op 次序影响。
//   - 若所有边都沿字典序递增，输出与字典序完全一致：剩余的最小节点不可能还有入边。
//     2026-10-21 前接入的 op 均满足此条件，原有 include 块保持不变。
//
// 无法排序说明约束成环。单个无重名声明列表不会成环，但跨 op 合并后可能成环，
// 因此每条边保留来源 op，用于报告诊断。
class ApiExtensionOrder {
 public:
  // Records the header order declared by declaringOp. Different operations may
  // declare the same header or ordering pair without conflict.
  void declare(mlir::Operation *declaringOp, llvm::ArrayRef<llvm::StringRef> headers);

  bool empty() const {
    return nodes.empty();
  }

  bool contains(llvm::StringRef header) const {
    return nodes.count(header) != 0;
  }

  // Returns the emission order. On conflicting constraints, reports a diagnostic
  // on a declaring operation and returns failure.
  mlir::FailureOr<llvm::SmallVector<llvm::StringRef>> resolve() const;

 private:
  using Edge = std::pair<llvm::StringRef, llvm::StringRef>;

  mlir::LogicalResult reportCycle(const std::map<llvm::StringRef, size_t> &indegrees) const;

  // 遍历次序影响输出，使用有序容器按头名确定次序。
  std::set<llvm::StringRef> nodes;
  std::map<llvm::StringRef, std::set<llvm::StringRef>> successors;
  std::map<llvm::StringRef, std::set<llvm::StringRef>> predecessors;
  std::map<Edge, mlir::Operation *> edgeOrigins;
};

}  // namespace mlir::ascend::kernel

#endif  // AUTOFUSE_MLIR_ARTIFACT_KERNELSOURCE_EMIT_APIEXTENSIONORDER_H
