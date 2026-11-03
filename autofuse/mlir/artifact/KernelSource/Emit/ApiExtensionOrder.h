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

// The order Autofuse AscendC API extension headers have to be emitted in.
//
// 一个 op 可以声明多个扩展头，而且它们之间有先后：一个包装头可以调用另一个包装头里的函数
// （`where.h` 调 `Duplicate`，而 `Duplicate` 只有 `duplicate.h` 定义），这些头是当作文本内联
// 进来的、彼此不 include，所以谁在前谁在后完全由发出的 include 次序说了算。
//
// 这里收的是**约束**而不是名字集合：声明列表里每一对相邻项都是一条边（`headers[i]` 必须排在
// `headers[i + 1]` 之前），`resolve` 对全体头做拓扑排，就绪集合里取字典序最小的出队。于是：
//   * 跨 op 的约束有保证。这是先前按字典序发射保不住的那一格——`airy_ai` 把
//     `trigonometric_function_utils` 放在列表首项，而 `bessel_j0` 要 `bessel_j_utils` 排在它
//     前面；按首见次序发射（legacy 的做法）时谁在前取决于两个 op 哪个先被 walk 到，也就是
//     取决于 plan，而按字典序发射则根本不看声明。
//   * 输出确定。同层由字典序定序，所以同一个 module 每次排出来的次序一样，且与 op 在图里的
//     次序无关。
//   * 所有边都是正字典序时，输出**精确等于**字典序：剩余节点里字典序最小的那个不可能有来自
//     剩余节点的入边（入边只能来自比它更小的点，而那些已经出队了）。2026-10-21 之前接通的那些
//     op 的边全是正字典序，所以它们的 include 块一个字节都不变。
//
// 排不出来只有一种情况：成环。环不是单个 op 的性质（一个 op 的列表除非有重名，否则永远不
// 成环），而是跨 op 的，所以边要记下来源 op，诊断才报得到某个具体的 op 上。
class ApiExtensionOrder {
 public:
  // Records the constraint that `headers` be emitted in the order given, as
  // declared by `declaringOp`. Declaring the same header or the same pair from
  // several operations is how the common case looks and is not a conflict.
  void declare(mlir::Operation *declaringOp, llvm::ArrayRef<llvm::StringRef> headers);

  bool empty() const {
    return nodes.empty();
  }

  bool contains(llvm::StringRef header) const {
    return nodes.count(header) != 0;
  }

  // The emission order, or failure with a diagnostic already reported on one of
  // the declaring operations when the declarations cannot all be honored.
  mlir::FailureOr<llvm::SmallVector<llvm::StringRef>> resolve() const;

 private:
  using Edge = std::pair<llvm::StringRef, llvm::StringRef>;

  mlir::LogicalResult reportCycle(const std::map<llvm::StringRef, size_t> &indegrees) const;

  // 都用有序容器：遍历次序进了输出，所以它必须只由头名决定。
  std::set<llvm::StringRef> nodes;
  std::map<llvm::StringRef, std::set<llvm::StringRef>> successors;
  std::map<llvm::StringRef, std::set<llvm::StringRef>> predecessors;
  std::map<Edge, mlir::Operation *> edgeOrigins;
};

}  // namespace mlir::ascend::kernel

#endif  // AUTOFUSE_MLIR_ARTIFACT_KERNELSOURCE_EMIT_APIEXTENSIONORDER_H
