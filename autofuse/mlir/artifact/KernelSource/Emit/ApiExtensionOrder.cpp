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
    // 为孤立节点也建立邻接表项，供 resolve 使用 at 查询。
    (void)successors[header];
    (void)predecessors[header];
  }
  for (size_t index = 0; index + 1 < headers.size(); ++index) {
    successors[headers[index]].insert(headers[index + 1]);
    predecessors[headers[index + 1]].insert(headers[index]);
    // 同一条边可能由多个 op 声明，保留首次声明用于诊断。
    edgeOrigins.try_emplace(Edge{headers[index], headers[index + 1]}, declaringOp);
  }
}

mlir::FailureOr<llvm::SmallVector<llvm::StringRef>> ApiExtensionOrder::resolve() const {
  std::map<llvm::StringRef, size_t> indegrees;
  // 就绪节点按字典序处理，使输出不依赖 op 的遍历次序。
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

// 剩余节点入度均非零，沿剩余前驱遍历必然遇到重复节点，由此提取一个环。
// 不能沿后继查找：剩余节点可能只有前驱而没有后继。
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

  // node 是重复节点，walk 中从它开始的部分构成环。
  // walk 沿前驱记录，反转后按正向输出，并在末尾重复起点以显示完整环。
  llvm::SmallVector<llvm::StringRef> cycle(llvm::find(walk, node), walk.end());
  std::reverse(cycle.begin(), cycle.end());
  // 将字典序最小的头移到起点，使诊断不依赖进入环的位置。
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
