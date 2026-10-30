/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sk_lock_detector.h
 * \brief
 */

#ifndef __SK_LOCK_DETECTOR_H__
#define __SK_LOCK_DETECTOR_H__

#include <vector>
#include <deque>
#include <algorithm>
#include <map>
#include <tuple>
#include <unordered_map>
#include <set>
#include "sk_types.h"
#include "sk_node.h"
#include "sk_graph.h"
#include "sk_scope_info.h"
#include "acl/acl.h"

class SuperKernelOptionsManager;

/*!
 * \enum DeadlockFailReason
 * \brief Detailed failure reasons for deadlock detection
 */
enum class DeadlockFailReason : uint8_t {
  NOT_FIND_DEADLOCK = 0,
  KERNEL_INSUFFICIENT_CORES,  ///< Kernel node requires more cores than available
  NOTIFY_NOT_IN_GRAPH,        ///< Wait node's notify not found in graph
  NOTIFY_AFTER_SK_RANGE,      ///< Notify node is after SK range
  NOTIFY_INVALID,             ///< Deadlock detected in wait node's pre-path
  NOTIFY_INSUFFICIENT_CORES,  ///< Notify node in other SK that requires more cores than available
  FIRST_WAIT,                 ///< First wait node in the graph is locked
  NO_SUPPORT_NODE,            ///< Not support node type in waiting for nodes
};
/*!
 * \brief Convert DeadlockFailReason to enum name string
 */
inline const char *to_string(DeadlockFailReason reason) {
  switch (reason) {
    case DeadlockFailReason::NOT_FIND_DEADLOCK:
      return "NOT_FIND_DEADLOCK";
    case DeadlockFailReason::KERNEL_INSUFFICIENT_CORES:
      return "KERNEL_INSUFFICIENT_CORES";
    case DeadlockFailReason::NOTIFY_INSUFFICIENT_CORES:
      return "NOTIFY_INSUFFICIENT_CORES";
    case DeadlockFailReason::NOTIFY_NOT_IN_GRAPH:
      return "NOTIFY_NOT_IN_GRAPH";
    case DeadlockFailReason::NOTIFY_AFTER_SK_RANGE:
      return "NOTIFY_AFTER_SK_RANGE";
    case DeadlockFailReason::NOTIFY_INVALID:
      return "NOTIFY_INVALID";
    case DeadlockFailReason::FIRST_WAIT:
      return "FIRST_WAIT";
    case DeadlockFailReason::NO_SUPPORT_NODE:
      return "NO_SUPPORT_NODE";
    default:
      return "UNKNOWN_DEADLOCK_REASON";
  }
}

/**
 * @class LockDetector
 * @brief 死锁检测器，用于检测SuperKernel融合时是否会引入死锁
 *
 * 该类通过分析节点的核心使用情况和依赖关系，判断是否可以将节点融合到SuperKernel中。
 */
class LockDetector {
 public:
  /**
   * @brief 构造函数，传入图对象
   * @param graph 图对象引用
   */
  explicit LockDetector(SuperKernelGraph &graph) {
    Init(graph);
  }

  LockDetector(SuperKernelGraph &graph, const SuperKernelOptionsManager &opts) : opts_(&opts) {
    Init(graph);
  }

  ~LockDetector() {
    if (graph_ != nullptr) {
      Reset();
    }
  }

  static aclError GetDeviceCores();

  /**
   * @brief 重置检测器状态
   */
  void Reset();

  void SetScopeCoreInfo(const ScopeCoreInfo &scopeCoreInfo) {
    const uint32_t cubeNum = scopeCoreInfo.GetCubeNum();
    const uint32_t vectorNum = scopeCoreInfo.GetVectorNum();
    scopeCoreInfoChanged_ = cubeNum != superKernelCubeNum || vectorNum != superKernelVecNum;
    superKernelCubeNum = cubeNum;
    superKernelVecNum = vectorNum;
  }

  void SetCurrentScopeId(uint16_t scopeId) {
    currentScopeId_ = scopeId;
  }

  void SetScopeInfos(const std::vector<SuperKernelScopeInfo> &scopes);
  void UpdateScopeInfo(const SuperKernelScopeInfo &scope);
  void RemoveScopeInfo(uint16_t scopeId);

  /**
   * @brief 检查多个scope并行运行时是否存在资源死锁
   * @param scopes 待检查的scope集合
   * @param riskWaitId 输出检测到风险的Wait节点ID；无风险时为INVALID_TASK_ID
   * @return true表示检测过程正常完成，false表示检测过程失败
   */
  bool CheckParallelScopeResources(const std::vector<SuperKernelScopeInfo> &scopes, uint64_t &riskWaitId);

  /**
   * @brief 获取最近一次检测到的死锁原因
   */
  DeadlockFailReason GetDeadlockReason() const {
    return deadlockReason_;
  }

  /**
   * @brief 判断节点是否可融合到SuperKernel中
   * @param curNode 待检测的节点
   * @return true 可融合，false 不可融合（会导致死锁）
   */
  bool IsFusible(SuperKernelBaseNode &curNode);

  /**
   * @brief 设置scope中Notify节点的expand number
   * @param scope Scope信息
   * @param scopeCoreInfo 当前scope用于死锁检测的核信息
   *
   * 该函数使用死锁检测过程中的ScopeCoreInfo，将vec/cube num设置到所有Notify节点上。
   */
  void SetNotifyNodesExpandNumForScope(SuperKernelScopeInfo &scope, const ScopeCoreInfo &scopeCoreInfo);

  /**
   * @brief 重置scope中Notify节点的expand number为0
   * @param scope Scope信息
   *
   * 该函数将scope中所有Notify节点的expandVecNum和expandCubeNum重置为0，
   * 用于在死锁检测Pass结束时复原节点状态，使得Pass可重入。
   */
  void ResetNotifyExpandNumForScope(SuperKernelScopeInfo &scope);

 private:
  // 一个scope在单条流上的可并行节点范围。
  struct ConcurrentNodeRange {
    uint64_t firstNodeId = INVALID_TASK_ID;
    uint64_t lastNodeId = INVALID_TASK_ID;
  };
  using ScopeConcurrentRanges = std::unordered_map<uint16_t, std::unordered_map<uint32_t, ConcurrentNodeRange>>;

  // 保存scope的核数、流范围、成员节点及是否包含Wait等运行时信息。
  struct ScopeRuntimeInfo {
    ScopeCoreInfo coreInfo;
    std::unordered_set<uint32_t> streamIds;
    std::unordered_map<uint32_t, uint64_t> headNodeIds;
    std::unordered_map<uint32_t, uint64_t> tailNodeIds;
    std::vector<SuperKernelBaseNode *> memberNodes;  // scope中的节点
    bool hasWait = false;                            // scope是否包含Wait节点
  };

  // 单scope死锁搜索中已经展开过的scope集合。
  struct DeadlockSearchState {
    std::unordered_set<uint16_t> expandedScopeIds;
  };

  // 一个scope或解除链当前占用的Cube/Vector核数。
  struct ResourceDemand {
    uint64_t cube = 0;
    uint64_t vec = 0;
  };

  // 多scope资源搜索过程中的待处理节点、常驻scope和资源占用状态。
  struct ResourceSearchState {
    std::vector<uint64_t> pending;
    std::unordered_set<uint64_t> visited;
    std::unordered_set<uint16_t> residentScopes;
    std::unordered_set<uint16_t> expandedWaitScopes;
    std::unordered_map<uint32_t, uint16_t> occupiedStreams;
    std::unordered_map<uint32_t, ResourceDemand> externalByStream;
    std::unordered_map<uint16_t, ResourceDemand> externalScopes;
    ResourceDemand residentDemand;
  };

  struct ConcurrentRangeState {
    uint16_t candidateScopeId = INVALID_SCOPE_ID;
    const std::vector<SuperKernelBaseNode *> *candidateNodes = nullptr;
    std::unordered_map<uint32_t, ConcurrentNodeRange> ranges;
  };

  static constexpr uint32_t CANDIDATE_CONCURRENT_SCOPE = static_cast<uint32_t>(INVALID_SCOPE_ID) + 1U;

  struct ConcurrentVertex {
    enum class Kind : uint8_t { NODE, EVENT_NODE, SCOPE };
    Kind kind = Kind::NODE;
    uint32_t scopeId = INVALID_SCOPE_ID;
    uint64_t nodeId = INVALID_TASK_ID;

    bool operator<(const ConcurrentVertex &other) const {
      return std::tie(kind, scopeId, nodeId) < std::tie(other.kind, other.scopeId, other.nodeId);
    }
  };

  void Init(SuperKernelGraph &graph);

  std::pair<uint64_t, uint64_t> GetAvailableCores(bool isSuperKernel) const;

  bool IsInSKStream(const SuperKernelBaseNode &node) const;

  void UpdateNodeInfo(const SuperKernelBaseNode &node);

  void UpdateSKRangeInStream(const SuperKernelBaseNode &curNode);
  void UpdateSKRangeInStream(uint32_t streamId, const SuperKernelBaseNode &headNode,
                             const SuperKernelBaseNode &tailNode);

  bool IsBeforeSKRange(const SuperKernelBaseNode &curNode);

  bool IsAfterSKRange(const SuperKernelBaseNode &curNode);

  bool HasIntersection(const std::unordered_set<uint32_t> &lhsStreams,
                       const std::unordered_set<uint32_t> &rhsStreams) const;

  // 检查当前节点及其依赖链是否形成死锁。
  bool HasDeadlock(SuperKernelBaseNode *curNode, DeadlockSearchState &state);
  // 沿当前节点的前驱依赖查找死锁。
  bool HasDeadlockPredecessor(SuperKernelBaseNode *curNode, DeadlockSearchState &state);
  // 检查一个scope内部的节点依赖是否形成死锁。
  bool CheckScopeDeadlock(const ScopeRuntimeInfo &scopeInfo, DeadlockSearchState &state);

  // 分别处理前驱链中的kernel、Wait和Notify节点。
  bool CheckKernelNodeDeadlock(SuperKernelBaseNode *preNode, DeadlockSearchState &state);

  bool CheckWaitNodeDeadlock(SuperKernelBaseNode *preNode, DeadlockSearchState &state);

  bool CheckNotifyNodeDeadlock(SuperKernelBaseNode *preNode, DeadlockSearchState &state);

  bool GetWaitNodeFusibleStatus(SuperKernelBaseNode &curNode);

  bool CheckNotifyInSKStream(SuperKernelBaseNode &curNode, SuperKernelBaseNode &notifyNode);

  bool GetFusibleStatus(SuperKernelBaseNode &curNode);
  bool ShouldBypassValueWaitDeadlock(const SuperKernelBaseNode &curNode) const;

  bool HasEnoughCores(const SuperKernelBaseNode *curNode, bool isSuperKernel);
  bool HasEnoughDeviceCores(uint64_t skCubeNum, uint64_t skVecNum, uint64_t depCubeNum, uint64_t depVecNum) const;
  // 计算候选scope在每条流上的可并行节点范围。
  std::unordered_map<uint32_t, ConcurrentNodeRange> CalculateScopeConcurrentRanges(
      const std::vector<SuperKernelBaseNode *> &candidate, uint16_t candidateScopeId) const;
  void InitializeConcurrentRangeState(const std::vector<SuperKernelBaseNode *> &candidate, uint16_t candidateScopeId,
                                      ConcurrentRangeState &rangeContext) const;
  void CalculateLeftConcurrentBoundary(ConcurrentRangeState &rangeContext) const;
  void CalculateRightConcurrentBoundary(ConcurrentRangeState &rangeContext) const;
  void TraverseConcurrentBoundary(ConcurrentRangeState &rangeContext, bool left) const;
  void TightenConcurrentBoundary(ConcurrentRangeState &rangeContext, uint32_t streamId,
                                 const SuperKernelBaseNode *boundary, bool left) const;
  void TightenScopeConcurrentBoundary(ConcurrentRangeState &rangeContext, uint32_t scopeId,
                                      uint32_t entryStreamId, bool left, bool eventBoundary) const;
  void UpdateConcurrentBoundary(ConcurrentRangeState &rangeContext, const ConcurrentVertex &vertex, bool left) const;
  void QueueConcurrentBoundarySeeds(const ConcurrentRangeState &rangeContext, bool left,
                                    std::deque<ConcurrentVertex> &pending) const;
  void AppendScopeVertices(const ConcurrentRangeState &rangeContext, const ConcurrentVertex &vertex, bool left,
                            std::deque<ConcurrentVertex> &pending) const;
  void AppendConcurrentStreamVertex(const ConcurrentRangeState &rangeContext, const SuperKernelBaseNode &node,
                                     bool left, const ConcurrentVertex &vertex,
                                     std::deque<ConcurrentVertex> &pending) const;
  void AppendForwardVertices(const ConcurrentRangeState &rangeContext, const ConcurrentVertex &vertex,
                             std::deque<ConcurrentVertex> &pending) const;
  void AppendReverseVertices(const ConcurrentRangeState &rangeContext, const ConcurrentVertex &vertex,
                             std::deque<ConcurrentVertex> &pending) const;
  uint32_t GetTrackedScopeId(const ConcurrentRangeState &rangeContext, const SuperKernelBaseNode &node) const;
  SuperKernelBaseNode *GetScopeBoundary(const ConcurrentRangeState &rangeContext, uint32_t scopeId, uint32_t streamId,
                                        bool first) const;
  // 从一个Wait展开依赖链，并检查外部资源是否会超出设备容量。
  bool CheckWaitResourceBranches(const SuperKernelBaseNode &waitNode, uint16_t rootScopeId,
                                 const ResourceDemand &rootDemand,
                                 const std::vector<SuperKernelBaseNode *> &rootNodes,
                                 const std::unordered_set<uint16_t> &initialResidents,
                                 const ScopeConcurrentRanges *precomputedRanges = nullptr);
  std::vector<const SuperKernelScopeInfo *> CollectWaitScopeCandidates(
      const std::vector<SuperKernelScopeInfo> &scopes) const;
  ScopeConcurrentRanges BuildScopeConcurrentRanges() const;
  bool CheckSelectedScopeResourceBranches(const std::vector<const SuperKernelScopeInfo *> &selected,
                                          const ScopeConcurrentRanges &ranges, uint64_t &riskWaitId);
  void PrepareResourceSearch(const SuperKernelBaseNode &waitNode, uint16_t rootScopeId,
                             const ResourceDemand &rootDemand, const std::vector<SuperKernelBaseNode *> &rootNodes,
                             const std::unordered_set<uint16_t> &initialResidents, ScopeConcurrentRanges &ranges,
                             ResourceSearchState &state, const ScopeConcurrentRanges *precomputedRanges);
  bool CanStartWaitScope(uint16_t scopeId, const ScopeRuntimeInfo &scope, uint16_t rootScopeId,
                         const std::vector<SuperKernelBaseNode *> &rootNodes, const ScopeConcurrentRanges &ranges,
                         const ResourceSearchState &state) const;
  void QueueStartedScopeBranch(uint16_t scopeId, const ScopeRuntimeInfo &scope, const ResourceSearchState &state,
                               std::vector<ResourceSearchState> &branches) const;
  void ExpandUnstartedScope(uint16_t scopeId, const ScopeRuntimeInfo &scope, ResourceSearchState &state) const;
  bool ProcessResourceSearchNode(uint64_t nodeId, uint16_t rootScopeId,
                                 const std::vector<SuperKernelBaseNode *> &rootNodes,
                                 const ScopeConcurrentRanges &ranges, ResourceSearchState &state,
                                 std::vector<ResourceSearchState> &branches);
  // 判断两个scope是否存在同一时刻并行运行的可能。
  bool CanRunTogether(uint16_t firstScopeId, uint16_t secondScopeId,
                      const ScopeConcurrentRanges &ranges) const;
  // 判断节点是否位于指定并行范围之前。
  bool IsBeforeConcurrentRange(const SuperKernelBaseNode &node,
                               const std::unordered_map<uint32_t, ConcurrentNodeRange> &range) const;
  // 将scope中的Wait对应Notify加入资源搜索队列。
  void QueueScopeWaitNotifies(uint16_t scopeId, ResourceSearchState &state) const;
  // 判断当前资源搜索状态是否已经超出设备资源限制。
  bool ResourceSearchHasRisk(const ResourceSearchState &state) const;
  bool IsSerializedWithCandidateScope(const SuperKernelBaseNode &waitNode, uint64_t candidateWaitNodeId,
                                      const ScopeRuntimeInfo &candidateScope) const;
  bool IsNodeBeforeCandidateScope(const SuperKernelBaseNode &node, const ScopeRuntimeInfo &candidateScope) const;
  bool IsNodeAfterCandidateScope(const SuperKernelBaseNode &node, const ScopeRuntimeInfo &candidateScope) const;
  void MergeCandidateScopeRange(const ScopeRuntimeInfo &candidateScope);
  void RollbackVisitedState(std::vector<uint64_t> &visitedNodes);

  std::vector<uint64_t> nodes;  // visited nodes
  std::vector<SuperKernelBaseNode *> candidateScopeNodes_;  // 当前候选scope中的节点
  std::unordered_map<uint32_t, ConcurrentNodeRange> activeConcurrentRanges_;  // 当前生效的流并行范围
  std::vector<uint64_t> tempVisitedNodes;  // temporary visited nodes for HasDeadlock
  uint32_t depOpCubeNum;                   // visited op cube num outside superkernel
  uint32_t depOpVecNum;                    // visited op vec num outside superkernel
  uint32_t superKernelCubeNum;
  uint32_t superKernelVecNum;
  uint16_t currentScopeId_ = INVALID_SCOPE_ID;
  std::unordered_map<uint16_t, ScopeRuntimeInfo> scopeInfos_;
  bool scopeCoreInfoChanged_ = false;  // Whether the current candidate changed physical core requirements
  static int64_t deviceRealCubeNum;
  static int64_t deviceRealVecNum;
  std::unordered_set<uint32_t> skStreamIds;
  uint32_t nodeNum;
  uint32_t kernelNodeNum;
  std::unordered_map<uint32_t, std::pair<uint64_t, uint64_t>> skRangeInStream;
  std::unordered_map<uint32_t, uint64_t> parallelScopeHeadNodeIds_;
  std::unordered_map<uint32_t, uint64_t> parallelScopeTailNodeIds_;
  SuperKernelGraph *graph_;  // 存储graph指针，用于析构时调用Reset
  const SuperKernelOptionsManager *opts_ = nullptr;
  DeadlockFailReason deadlockReason_ = DeadlockFailReason::NOT_FIND_DEADLOCK;  // 当前检测到的死锁原因
};

#endif  // __SK_LOCK_DETECTOR_H__
