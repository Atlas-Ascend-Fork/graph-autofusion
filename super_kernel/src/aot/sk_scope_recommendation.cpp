/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sk_scope_recommendation.h"

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <map>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "sk_common.h"
#include "sk_graph.h"
#include "sk_node.h"

namespace {

using Bits = std::vector<uint64_t>;

struct BitsHash {
  size_t operator()(const Bits &bits) const {
    size_t value = 0;
    for (const uint64_t word : bits) {
      value ^= static_cast<size_t>(word) + 0x9e3779b9U + (value << 6U) + (value >> 2U);
    }
    return value;
  }
};

void SetBit(Bits &bits, size_t index) {
  bits[index / 64U] |= (uint64_t{1} << (index % 64U));
}

bool HasBit(const Bits &bits, size_t index) {
  return (bits[index / 64U] & (uint64_t{1} << (index % 64U))) != 0;
}

void OrBits(Bits &target, const Bits &source) {
  for (size_t i = 0; i < target.size(); ++i) {
    target[i] |= source[i];
  }
}

template <typename Fn>
void ForEachBit(const Bits &bits, Fn fn) {
  for (size_t word = 0; word < bits.size(); ++word) {
    uint64_t value = bits[word];
    while (value != 0) {
      const unsigned offset = static_cast<unsigned>(__builtin_ctzll(value));
      fn(word * 64U + offset);
      value &= value - 1U;
    }
  }
}

// Compute maximum flow for the weighted antichain reduction.
class Dinic {
 public:
  explicit Dinic(size_t size) : edges_(size), level_(size), cursor_(size) {}

  void AddEdge(size_t from, size_t to, int64_t capacity) {
    Edge forward{to, edges_[to].size(), capacity};
    Edge reverse{from, edges_[from].size(), 0};
    edges_[from].push_back(forward);
    edges_[to].push_back(reverse);
  }

  int64_t MaxFlow(size_t source, size_t sink) {
    int64_t flow = 0;
    while (BuildLevels(source, sink)) {
      std::fill(cursor_.begin(), cursor_.end(), 0);
      while (int64_t pushed = Push(source, sink, std::numeric_limits<int64_t>::max())) {
        flow += pushed;
      }
    }
    return flow;
  }

 private:
  struct Edge {
    size_t to;
    size_t reverseEdgeIndex;
    int64_t capacity;
  };

  bool BuildLevels(size_t source, size_t sink) {
    std::fill(level_.begin(), level_.end(), -1);
    std::queue<size_t> queue;
    level_[source] = 0;
    queue.push(source);
    while (!queue.empty()) {
      const size_t node = queue.front();
      queue.pop();
      for (const auto &edge : edges_[node]) {
        if (edge.capacity > 0 && level_[edge.to] < 0) {
          level_[edge.to] = level_[node] + 1;
          queue.push(edge.to);
        }
      }
    }
    return level_[sink] >= 0;
  }

  int64_t Push(size_t node, size_t sink, int64_t amount) {
    if (node == sink) {
      return amount;
    }
    for (size_t &index = cursor_[node]; index < edges_[node].size(); ++index) {
      Edge &edge = edges_[node][index];
      if (edge.capacity <= 0 || level_[edge.to] != level_[node] + 1) {
        continue;
      }
      const int64_t pushed = Push(edge.to, sink, std::min(amount, edge.capacity));
      if (pushed > 0) {
        edge.capacity -= pushed;
        edges_[edge.to][edge.reverseEdgeIndex].capacity += pushed;
        return pushed;
      }
    }
    return 0;
  }

  std::vector<std::vector<Edge>> edges_;
  std::vector<int64_t> level_;
  std::vector<size_t> cursor_;
};

struct DependencyVertex {
  uint32_t streamIdxInGraph;
  uint32_t maxCube = 0;
  uint32_t maxVector = 0;
  std::vector<size_t> successors;
};

// Compute peak external resource demand as a maximum weighted antichain using a minimum cut.
int64_t MaxAntichainWeight(const Bits &external, const std::vector<DependencyVertex> &vertices,
                           uint32_t DependencyVertex::*resource, const std::vector<Bits> &descendants) {
  std::vector<size_t> members;
  ForEachBit(external, [&](size_t member) {
    if (vertices[member].*resource != 0) {
      members.push_back(member);
    }
  });
  if (members.empty()) {
    return 0;
  }
  const size_t count = members.size();
  const size_t source = count * 2U;
  const size_t sink = source + 1U;
  Dinic flow(sink + 1U);
  int64_t totalWeight = 0;
  for (size_t i = 0; i < count; ++i) {
    totalWeight += vertices[members[i]].*resource;
  }
  const int64_t infinity = totalWeight + 1;
  for (size_t i = 0; i < count; ++i) {
    const int64_t weight = vertices[members[i]].*resource;
    flow.AddEdge(source, i, weight);
    flow.AddEdge(count + i, sink, weight);
    for (size_t j = 0; j < count; ++j) {
      if (HasBit(descendants[members[i]], members[j])) {
        flow.AddEdge(i, count + j, infinity);
      }
    }
  }
  return totalWeight - flow.MaxFlow(source, sink);
}

struct ExternalResourceProfile {
  int64_t lowerCube = 0;
  int64_t lowerVector = 0;
  int64_t upperCube = 0;
  int64_t upperVector = 0;
  int64_t exactCube = -1;
  int64_t exactVector = -1;
};

// Check resource bounds first, then compute and cache the exact demand only when needed.
bool FitsResourceBudget(const Bits &external, const std::vector<DependencyVertex> &vertices,
                        uint32_t DependencyVertex::*resource, const std::vector<Bits> &descendants, int64_t lower,
                        int64_t upper, int64_t budget, int64_t &exact) {
  if (lower > budget) {
    return false;
  }
  if (upper <= budget) {
    return true;
  }
  if (exact < 0) {
    exact = MaxAntichainWeight(external, vertices, resource, descendants);
  }
  return exact <= budget;
}

using StreamNodes = std::map<uint32_t, std::vector<SuperKernelBaseNode *>>;

struct RecommendationSegment {
  size_t dependencyVertexId;
  size_t beginNodeOffset;
  size_t endNodeOffset;
};

struct AnalysisGraph {
  std::vector<DependencyVertex> vertices;
  std::vector<RecommendationSegment> segments;
};

// Track one scope-eligible segment while its stream is being traversed.
class PendingRecommendationSegment {
 public:
  void Close(size_t endNodeOffset, std::vector<RecommendationSegment> &segments) {
    if (hasKernel_) {
      segment_.endNodeOffset = endNodeOffset;
      segments.push_back(segment_);
    }
    segmentOpen_ = false;
    hasKernel_ = false;
  }

  void Append(const SuperKernelBaseNode &node, size_t dependencyVertexId, size_t nodeOffset, bool hasNamedScope,
              std::vector<RecommendationSegment> &segments) {
    const auto flags = node.GetScopeBitFlags();
    if (hasNamedScope && (flags.none() || (segmentOpen_ && flags != scopeFlags_))) {
      Close(nodeOffset, segments);
    }
    if (hasNamedScope && flags.none()) {
      return;
    }
    if (!segmentOpen_) {
      segment_ = RecommendationSegment{dependencyVertexId, nodeOffset, nodeOffset};
      scopeFlags_ = flags;
      segmentOpen_ = true;
    }
    hasKernel_ = hasKernel_ || node.GetNodeType() == SkNodeType::NODE_KERNEL;
  }

 private:
  RecommendationSegment segment_{};
  std::bitset<MAX_SCOPE_NUM> scopeFlags_;
  bool segmentOpen_ = false;
  bool hasKernel_ = false;
};

// Only complete same-stream notify/wait groups can remain inside a dependency interval.
std::unordered_set<uint64_t> CollectSameStreamSynchronizationNodes(const StreamNodes &streams,
                                                                   const SuperKernelGraph &graph) {
  std::unordered_set<uint64_t> sameStreamNodes;
  for (const auto &entry : streams) {
    for (const auto *node : entry.second) {
      if (node->GetNodeType() != SkNodeType::NODE_NOTIFY) {
        continue;
      }
      const auto waitIds = node->GetCorrespondingWaitNodeIds();
      if (waitIds.empty() || !std::all_of(waitIds.begin(), waitIds.end(), [&](uint64_t waitId) {
            return graph.GetNodeById(waitId)->GetStreamIdxInGraph() == entry.first;
          })) {
        continue;
      }
      sameStreamNodes.insert(node->GetNodeId());
      sameStreamNodes.insert(waitIds.begin(), waitIds.end());
    }
  }
  return sameStreamNodes;
}

// Connect synchronization endpoints after every stream has supplied its compressed vertex mapping.
void AddSynchronizationEdges(const StreamNodes &streams, const std::unordered_map<uint64_t, size_t> &syncVertexByNodeId,
                             std::vector<DependencyVertex> &vertices) {
  for (const auto &entry : streams) {
    for (const auto *node : entry.second) {
      if (node->GetNodeType() != SkNodeType::NODE_NOTIFY) {
        continue;
      }
      const size_t source = syncVertexByNodeId.at(node->GetNodeId());
      for (const uint64_t waitId : node->GetCorrespondingWaitNodeIds()) {
        const auto wait = syncVertexByNodeId.find(waitId);
        // Same-stream synchronization endpoints may belong to the same compressed interval.
        if (wait != syncVertexByNodeId.end() && source != wait->second) {
          vertices[source].successors.push_back(wait->second);
        }
      }
    }
  }
}

// Compress stream intervals around synchronization boundaries and record scope-eligible recommendation ranges.
AnalysisGraph BuildDependencyGraphAndSegments(const StreamNodes &streams, const SuperKernelGraph &graph,
                                              bool hasNamedScope) {
  const auto sameStreamNodes = CollectSameStreamSynchronizationNodes(streams, graph);
  AnalysisGraph result;
  std::unordered_map<uint64_t, size_t> syncVertexByNodeId;
  for (const auto &entry : streams) {
    const auto &nodes = entry.second;
    PendingRecommendationSegment segment;
    bool previousBoundary = false;
    for (size_t i = 0; i < nodes.size(); ++i) {
      const auto *node = nodes[i];
      const auto type = node->GetNodeType();
      const bool synchronization = type == SkNodeType::NODE_NOTIFY || type == SkNodeType::NODE_WAIT;
      const bool boundary = synchronization && sameStreamNodes.count(node->GetNodeId()) == 0;
      if (i == 0 || boundary || previousBoundary) {
        const size_t vertex = result.vertices.size();
        result.vertices.emplace_back();
        result.vertices.back().streamIdxInGraph = entry.first;
        if (i != 0) {
          result.vertices[vertex - 1U].successors.push_back(vertex);
        }
      }
      previousBoundary = boundary;
      const size_t vertex = result.vertices.size() - 1U;
      if (synchronization) {
        syncVertexByNodeId.emplace(node->GetNodeId(), vertex);
      }
      if (type == SkNodeType::NODE_KERNEL) {
        auto &value = result.vertices[vertex];
        value.maxCube = std::max(value.maxCube, node->GetCubeNum());
        value.maxVector = std::max(value.maxVector, node->GetVecNum());
      }
      if (boundary) {
        segment.Close(i, result.segments);
        continue;
      }
      segment.Append(*node, vertex, i, hasNamedScope, result.segments);
    }
    segment.Close(nodes.size(), result.segments);
  }
  AddSynchronizationEdges(streams, syncVertexByNodeId, result.vertices);
  return result;
}

// Build transitive reachability in reverse topological order, retaining paths through synchronization vertices.
std::vector<Bits> ComputeDescendants(const std::vector<DependencyVertex> &vertices) {
  std::vector<size_t> indegree(vertices.size(), 0);
  for (const auto &vertex : vertices) {
    for (const size_t child : vertex.successors) {
      ++indegree[child];
    }
  }
  std::queue<size_t> ready;
  for (size_t i = 0; i < indegree.size(); ++i) {
    if (indegree[i] == 0) {
      ready.push(i);
    }
  }
  std::vector<size_t> topological;
  topological.reserve(vertices.size());
  while (!ready.empty()) {
    const size_t current = ready.front();
    ready.pop();
    topological.push_back(current);
    for (const size_t child : vertices[current].successors) {
      if (--indegree[child] == 0) {
        ready.push(child);
      }
    }
  }
  const size_t wordCount = (vertices.size() + 63U) / 64U;
  std::vector<Bits> descendants(vertices.size(), Bits(wordCount, 0));
  for (auto iter = topological.rbegin(); iter != topological.rend(); ++iter) {
    const size_t current = *iter;
    for (const size_t child : vertices[current].successors) {
      SetBit(descendants[current], child);
      OrBits(descendants[current], descendants[child]);
    }
  }
  return descendants;
}

// Find resource-bearing vertices that can overlap the segment's dependency interval.
Bits FindExternalVertices(const RecommendationSegment &segment, const std::vector<DependencyVertex> &vertices,
                          const std::vector<Bits> &descendants) {
  const uint32_t streamIdxInGraph = vertices[segment.dependencyVertexId].streamIdxInGraph;
  Bits external((vertices.size() + 63U) / 64U, 0);
  // No cross-stream edge enters or leaves inside an interval, so every run shares this external set.
  for (size_t other = 0; other < vertices.size(); ++other) {
    if (vertices[other].streamIdxInGraph != streamIdxInGraph &&
        (vertices[other].maxCube != 0 || vertices[other].maxVector != 0) &&
        !HasBit(descendants[segment.dependencyVertexId], other) &&
        !HasBit(descendants[other], segment.dependencyVertexId)) {
      SetBit(external, other);
    }
  }
  return external;
}

// Bound each resource by the largest single vertex and the sum of per-stream maxima.
ExternalResourceProfile BuildExternalResourceProfile(const Bits &external,
                                                     const std::vector<DependencyVertex> &vertices) {
  ExternalResourceProfile value;
  std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> streamMaxima;
  ForEachBit(external, [&](size_t bit) {
    const auto &vertex = vertices[bit];
    value.lowerCube = std::max(value.lowerCube, static_cast<int64_t>(vertex.maxCube));
    value.lowerVector = std::max(value.lowerVector, static_cast<int64_t>(vertex.maxVector));
    auto &[maxCube, maxVector] = streamMaxima[vertex.streamIdxInGraph];
    maxCube = std::max(maxCube, vertex.maxCube);
    maxVector = std::max(maxVector, vertex.maxVector);
  });
  for (const auto &entry : streamMaxima) {
    const auto &[maxCube, maxVector] = entry.second;
    value.upperCube += maxCube;
    value.upperVector += maxVector;
  }
  return value;
}

struct SegmentResourceBudget {
  const Bits &external;
  const std::vector<DependencyVertex> &vertices;
  const std::vector<Bits> &descendants;
  ExternalResourceProfile &profile;
  int64_t cubeCapacity;
  int64_t vectorCapacity;

  bool Fits(uint32_t cube, uint32_t vector) const {
    return FitsResourceBudget(external, vertices, &DependencyVertex::maxCube, descendants, profile.lowerCube,
                              profile.upperCube, cubeCapacity - cube, profile.exactCube) &&
           FitsResourceBudget(external, vertices, &DependencyVertex::maxVector, descendants, profile.lowerVector,
                              profile.upperVector, vectorCapacity - vector, profile.exactVector);
  }
};

void AddCandidateRange(const std::vector<SuperKernelBaseNode *> &nodes, size_t beginNodeOffset, size_t endNodeOffset,
                       std::unordered_map<uint64_t, uint64_t> &candidateByNode, uint64_t &nextCandidateId) {
  for (size_t i = beginNodeOffset; i < endNodeOffset; ++i) {
    candidateByNode.emplace(nodes[i]->GetNodeId(), nextCandidateId);
  }
  ++nextCandidateId;
}

// Fall back to adjacent kernel runs with equal C/V demand when the whole segment cannot fit.
void RecommendEqualCoreRuns(const std::vector<SuperKernelBaseNode *> &nodes, const RecommendationSegment &segment,
                            const SegmentResourceBudget &budget,
                            std::unordered_map<uint64_t, uint64_t> &candidateByNode, uint64_t &nextCandidateId) {
  size_t runStart = segment.beginNodeOffset;
  size_t runLast = segment.beginNodeOffset;
  size_t runKernelCount = 0;
  auto closeRun = [&]() {
    if (runKernelCount >= 2U && budget.Fits(nodes[runStart]->GetCubeNum(), nodes[runStart]->GetVecNum())) {
      AddCandidateRange(nodes, runStart, runLast + 1U, candidateByNode, nextCandidateId);
    }
    runKernelCount = 0;
  };
  for (size_t i = segment.beginNodeOffset; i < segment.endNodeOffset; ++i) {
    if (nodes[i]->GetNodeType() != SkNodeType::NODE_KERNEL) {
      continue;
    }
    if (runKernelCount != 0 && (nodes[i]->GetCubeNum() != nodes[runStart]->GetCubeNum() ||
                                nodes[i]->GetVecNum() != nodes[runStart]->GetVecNum())) {
      closeRun();
    }
    if (runKernelCount == 0) {
      runStart = i;
    }
    runLast = i;
    ++runKernelCount;
  }
  closeRun();
}

// Recommend whole segments when resources fit; otherwise try adjacent kernel runs with equal C/V demand.
uint64_t GenerateCandidateRanges(const StreamNodes &streams, const AnalysisGraph &graph,
                                 const std::vector<Bits> &descendants, int64_t cubeCapacity, int64_t vectorCapacity,
                                 std::unordered_map<uint64_t, uint64_t> &candidateByNode) {
  const auto &vertices = graph.vertices;
  std::unordered_map<Bits, ExternalResourceProfile, BitsHash> externalResourceProfileCache;
  uint64_t nextCandidateId = 0;
  for (const auto &segment : graph.segments) {
    Bits external = FindExternalVertices(segment, vertices, descendants);
    auto profile = externalResourceProfileCache.find(external);
    if (profile == externalResourceProfileCache.end()) {
      const auto value = BuildExternalResourceProfile(external, vertices);
      profile = externalResourceProfileCache.emplace(std::move(external), value).first;
    }
    const SegmentResourceBudget budget{profile->first,  vertices,     descendants,
                                       profile->second, cubeCapacity, vectorCapacity};
    const auto &nodes = streams.at(vertices[segment.dependencyVertexId].streamIdxInGraph);
    uint32_t ownCube = 0;
    uint32_t ownVector = 0;
    for (size_t i = segment.beginNodeOffset; i < segment.endNodeOffset; ++i) {
      if (nodes[i]->GetNodeType() == SkNodeType::NODE_KERNEL) {
        ownCube = std::max(ownCube, nodes[i]->GetCubeNum());
        ownVector = std::max(ownVector, nodes[i]->GetVecNum());
      }
    }
    if (budget.Fits(ownCube, ownVector)) {
      AddCandidateRange(nodes, segment.beginNodeOffset, segment.endNodeOffset, candidateByNode, nextCandidateId);
      continue;
    }
    RecommendEqualCoreRuns(nodes, segment, budget, candidateByNode, nextCandidateId);
  }
  return nextCandidateId;
}

}  // namespace

// Analyze graph dependencies and device capacity to assign candidate IDs without modifying the input graph.
bool ScopeRecommendation::Analyze(const SuperKernelGraph &graph) {
  candidateByNode_.clear();

  StreamNodes streams;
  bool hasNamedScope = false;
  bool hasKernel = false;
  for (const uint64_t headId : graph.GetHeadNodes()) {
    uint64_t nodeId = headId;
    while (nodeId != INVALID_TASK_ID) {
      SuperKernelBaseNode *node = graph.GetNodeById(nodeId);
      if (node == nullptr) {
        SK_LOGE("[ScopeRecommendation] node %lu not found while traversing graph", nodeId);
        return false;
      }
      streams[node->GetStreamIdxInGraph()].push_back(node);
      hasNamedScope = hasNamedScope || (node->IsScopeBegin() && !node->GetScopeName().empty());
      hasKernel = hasKernel || node->GetNodeType() == SkNodeType::NODE_KERNEL;
      nodeId = node->GetNextNodeId();
    }
  }
  if (streams.empty()) {
    SK_LOGI("[ScopeRecommendation] graph has no nodes");
    return true;
  }

  int64_t cubeCapacity = 0;
  int64_t vectorCapacity = 0;
  if (GetDeviceCoreNums(cubeCapacity, vectorCapacity) != ACL_SUCCESS) {
    SK_LOGE("[ScopeRecommendation] failed to get valid C/V device capacity");
    return false;
  }
  if (!hasKernel) {
    SK_LOGI("[ScopeRecommendation] graph has no KERNEL nodes");
    return true;
  }

  const auto analysisGraph = BuildDependencyGraphAndSegments(streams, graph, hasNamedScope);
  const auto descendants = ComputeDescendants(analysisGraph.vertices);
  const uint64_t candidateCount =
      GenerateCandidateRanges(streams, analysisGraph, descendants, cubeCapacity, vectorCapacity, candidateByNode_);
  SK_LOGI(
      "[ScopeRecommendation] selected %zu candidate ranges from %zu stream segments "
      "(userScopes=%d, capacity C/V=%ld/%ld)",
      candidateCount, analysisGraph.segments.size(), hasNamedScope, cubeCapacity, vectorCapacity);
  return true;
}
