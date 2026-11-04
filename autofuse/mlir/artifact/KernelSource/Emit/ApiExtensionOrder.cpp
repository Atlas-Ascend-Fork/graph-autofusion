/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "artifact/KernelSource/Emit/ApiExtensionOrder.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <string>

namespace mlir::ascend::kernel {

void ApiExtensionOrder::declare(mlir::Operation *declaringOp, llvm::ArrayRef<llvm::StringRef> headers) {
  for (llvm::StringRef header : headers) {
    nodes.insert(header);
    // `resolve` 的 `at` 要对每个节点都取得到，没有任何边的孤立节点也算。
    (void)successors[header];
    (void)predecessors[header];
  }
  for (size_t index = 0; index + 1 < headers.size(); ++index) {
    successors[headers[index]].insert(headers[index + 1]);
    predecessors[headers[index + 1]].insert(headers[index]);
    // 同一条边常被多个 op 声明，诊断只需要其中一个来源，留第一个。
    edgeOrigins.try_emplace(Edge{headers[index], headers[index + 1]}, declaringOp);
  }
}

mlir::FailureOr<llvm::SmallVector<llvm::StringRef>> ApiExtensionOrder::resolve() const {
  std::map<llvm::StringRef, size_t> indegrees;
  // 就绪集合按字典序：同层之间的次序由它定，于是输出与 op 的 walk 次序无关。
  std::set<llvm::StringRef> ready;
  for (llvm::StringRef node : nodes) {
    const size_t indegree = predecessors.at(node).size();
    indegrees[node] = indegree;
    if (indegree == 0) {
      ready.insert(node);
    }
  }

  llvm::SmallVector<llvm::StringRef> order;
  while (!ready.empty()) {
    const llvm::StringRef node = *ready.begin();
    ready.erase(ready.begin());
    order.push_back(node);
    for (llvm::StringRef successor : successors.at(node)) {
      if (--indegrees[successor] == 0) {
        ready.insert(successor);
      }
    }
  }
  if (order.size() == nodes.size()) {
    return order;
  }
  return reportCycle(indegrees);
}

// 没出队的节点都还有来自剩余节点的入边——入边计数没归零就是这个意思。所以沿**前驱**一直走
// 不会走到尽头，必然回到走过的某个节点，截下从它起的那一段就是一个环。沿后继走不行：剩余
// 节点可以没有后继（它是被前驱卡住的，不是卡住别人的）。
mlir::LogicalResult ApiExtensionOrder::reportCycle(const std::map<llvm::StringRef, size_t> &indegrees) const {
  auto isRemaining = [&indegrees](llvm::StringRef node) { return indegrees.at(node) != 0; };
  // 从字典序最小的剩余节点起步，这样同一份声明每次找到的是同一个环。
  llvm::StringRef node = *llvm::find_if(nodes, isRemaining);
  llvm::SmallVector<llvm::StringRef> walk;
  std::set<llvm::StringRef> seen;
  while (seen.insert(node).second) {
    walk.push_back(node);
    node = *llvm::find_if(predecessors.at(node), isRemaining);
  }

  // `walk` 是沿前驱记下的，所以它是反着的；`node` 是重复出现的那个点，环就是 `walk` 里从它
  // 起的那一段。按前进方向打完整路径，最后一项与第一项相同，读的人不用自己把环接上。
  llvm::SmallVector<llvm::StringRef> cycle(llvm::find(walk, node), walk.end());
  std::reverse(cycle.begin(), cycle.end());
  // 反转把起点转掉了，而环是旋转不变的，所以固定从环上字典序最小的那个头起：报出来的形式
  // 就不取决于走的时候从哪个节点进去。
  std::rotate(cycle.begin(), std::min_element(cycle.begin(), cycle.end()), cycle.end());
  std::string path;
  llvm::raw_string_ostream stream(path);
  for (llvm::StringRef header : cycle) {
    stream << "'" << header << "' -> ";
  }
  stream << "'" << cycle.front() << "'";

  // 环上第一条边：长度为 1 的环是自环，来自同一个列表里紧挨着的重名。
  const Edge reported = cycle.size() == 1 ? Edge{cycle.front(), cycle.front()} : Edge{cycle[0], cycle[1]};
  return edgeOrigins.at(reported)->emitError(
             "API extension headers cannot be ordered for emission: the declared order is cyclic (")
         << path << ")";
}

}  // namespace mlir::ascend::kernel
