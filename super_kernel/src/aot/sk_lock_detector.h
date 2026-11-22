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
#include <unordered_set>
#include <set>
#include <string>
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

struct ScopeRuntimeInfo {
  static ScopeRuntimeInfo FromScope(const SuperKernelScopeInfo &scope);

  ScopeCoreInfo coreInfo;
  std::unordered_set<uint32_t> streamIds;
  std::unordered_map<uint32_t, uint64_t> headNodeIds;  // 流ID -> scope在该流的首节点ID。
  std::unordered_map<uint32_t, uint64_t> tailNodeIds;  // 流ID -> scope在该流的尾节点ID。
  std::vector<SuperKernelBaseNode *> memberNodes;
  bool hasWait = false;
};

struct ResourceConflictInfo {
  std::vector<uint64_t> parallelNodeIds;
  std::vector<uint16_t> parallelScopeIds;

  std::string Format() const;
};

class ScopeConcurrentRangeAnalyzer {
 public:
  struct ConcurrentNodeRange {
    uint64_t firstNodeId = INVALID_TASK_ID;
    uint64_t lastNodeId = INVALID_TASK_ID;
  };
  // scope ID -> (流ID -> 该scope在该流上的可并行节点范围)。
  using ScopeConcurrentRanges = std::unordered_map<uint16_t, std::unordered_map<uint32_t, ConcurrentNodeRange>>;

  ScopeConcurrentRangeAnalyzer(SuperKernelGraph &graph,
                               const std::unordered_map<uint16_t, ScopeRuntimeInfo> &scopeInfos)
      : graph_(&graph), scopeInfos_(scopeInfos) {}

  std::unordered_map<uint32_t, ConcurrentNodeRange> CalculateScopeConcurrentRanges(
      const std::vector<SuperKernelBaseNode *> &candidate, uint16_t candidateScopeId) const;
  bool IsBeforeConcurrentRange(const SuperKernelBaseNode &node,
                               const std::unordered_map<uint32_t, ConcurrentNodeRange> &range) const;
  bool CanRunTogether(uint16_t firstScopeId, uint16_t secondScopeId, const ScopeConcurrentRanges &ranges) const;
  bool HasIntersection(const std::unordered_set<uint32_t> &lhsStreams,
                       const std::unordered_set<uint32_t> &rhsStreams) const;

 private:
  struct ConcurrentRangeState {
    uint16_t candidateScopeId = INVALID_SCOPE_ID;
    // 仅在本次计算期间借用候选节点。
    const std::vector<SuperKernelBaseNode *> *candidateNodes = nullptr;
    std::unordered_map<uint32_t, ConcurrentNodeRange> candidateBoundaries;  // 流ID -> 候选scope的固定首尾节点。
    std::unordered_map<uint32_t, ConcurrentNodeRange> concurrentRangesByStream;  // 流ID -> 沿依赖收紧后的可并行范围。
  };

  // 候选节点可能尚未归入scope，遍历时用独立ID与已有scope区分。
  static constexpr uint32_t CANDIDATE_CONCURRENT_SCOPE = static_cast<uint32_t>(INVALID_SCOPE_ID) + 1U;

  struct ConcurrentVertex {
    // EVENT_NODE表示跨Notify/Wait依赖到达的节点；SCOPE表示跨流scope整体。
    enum class Kind : uint8_t { NODE, EVENT_NODE, SCOPE };
    Kind kind = Kind::NODE;
    uint32_t activeScopeId = INVALID_SCOPE_ID;  // 遍历顶点关联的scope，离开时为INVALID_SCOPE_ID。
    uint64_t vertexNodeId = INVALID_TASK_ID;    // SCOPE顶点不对应单个节点。

    bool operator<(const ConcurrentVertex &other) const {
      return std::tie(kind, activeScopeId, vertexNodeId) <
             std::tie(other.kind, other.activeScopeId, other.vertexNodeId);
    }
  };

  // 计算入口依次沿左右依赖收紧区间。
  void CalculateLeftConcurrentBoundary(ConcurrentRangeState &rangeContext) const;
  void CalculateRightConcurrentBoundary(ConcurrentRangeState &rangeContext) const;

  // 查找scope边界并初始化候选节点在各流上的区间。
  SuperKernelBaseNode *GetScopeBoundary(const ConcurrentRangeState &rangeContext, uint32_t scopeId, uint32_t streamId,
                                        bool first) const;
  uint32_t GetTrackedScopeId(const ConcurrentRangeState &rangeContext, const SuperKernelBaseNode &node) const;
  void InitializeConcurrentRangeState(const std::vector<SuperKernelBaseNode *> &candidate, uint16_t candidateScopeId,
                                      ConcurrentRangeState &rangeContext) const;

  // 根据跨流scope和Notify/Wait依赖收紧区间。
  void TightenConcurrentBoundary(ConcurrentRangeState &rangeContext, uint32_t streamId,
                                 const SuperKernelBaseNode *boundary, bool left) const;
  void TightenScopeConcurrentBoundary(ConcurrentRangeState &rangeContext, uint32_t scopeId, uint32_t entryStreamId,
                                      bool left, bool eventBoundary) const;
  void UpdateConcurrentBoundary(ConcurrentRangeState &rangeContext, const ConcurrentVertex &vertex, bool left) const;

  // 在流内及Notify/Wait边之间扩展遍历顶点。
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
  void TraverseConcurrentBoundary(ConcurrentRangeState &rangeContext, bool left) const;

  SuperKernelGraph *graph_;
  const std::unordered_map<uint16_t, ScopeRuntimeInfo> &scopeInfos_;  // scope ID -> 运行时信息；由调用方维护。
};

class WaitResourceDependencyAnalyzer {
 public:
  using ScopeConcurrentRanges = ScopeConcurrentRangeAnalyzer::ScopeConcurrentRanges;

  // 独立的cube/vec核数需求，可用于累计多个scope或记录外部依赖上界。
  struct CoreDemand {
    uint64_t cubeCoreNum = 0;
    uint64_t vectorCoreNum = 0;
  };

  WaitResourceDependencyAnalyzer(SuperKernelGraph &graph, const SuperKernelOptionsManager *opts,
                                 const std::unordered_map<uint16_t, ScopeRuntimeInfo> &scopeInfos,
                                 const ScopeConcurrentRangeAnalyzer &ranges, const int64_t &deviceCubeNum,
                                 const int64_t &deviceVecNum)
      : graph_(&graph),
        opts_(opts),
        scopeInfos_(scopeInfos),
        ranges_(ranges),
        deviceRealCubeNum(deviceCubeNum),
        deviceRealVecNum(deviceVecNum) {}

  bool CheckWaitResourceBranches(const SuperKernelBaseNode &waitNode, uint16_t rootScopeId,
                                 const CoreDemand &rootDemand, const std::vector<SuperKernelBaseNode *> &rootNodes,
                                 const std::unordered_set<uint16_t> &initialResidents,
                                 const ScopeConcurrentRanges *precomputedRanges = nullptr,
                                 ResourceConflictInfo *conflictInfo = nullptr);
  bool HasEnoughDeviceCores(uint64_t skCubeNum, uint64_t skVecNum, uint64_t depCubeNum, uint64_t depVecNum) const;
  bool ShouldBypassValueWaitDeadlock(const SuperKernelBaseNode &curNode) const;

 private:
  struct ResourceSearchState {
    std::vector<uint64_t> pendingNodeIds;                    // 待沿前驱或Wait对应Notify追踪的节点ID。
    std::unordered_set<uint64_t> visited;                    // 当前分支已处理的节点ID。
    std::unordered_set<uint16_t> residentScopes;             // 假定已经启动并占用核资源的scope。
    std::unordered_set<uint16_t> expandedWaitScopes;         // 已将内部Wait的外部Notify加入搜索的scope。
    std::unordered_map<uint32_t, uint16_t> occupiedStreams;  // 流ID到占用该流的常驻scope ID。
    std::unordered_map<uint32_t, CoreDemand> externalByStream;        // 未归入scope的外部流需求上界。
    std::unordered_map<uint32_t, uint64_t> cubeSourceNodeByStream;    // 流ID -> cube需求上界的节点ID。
    std::unordered_map<uint32_t, uint64_t> vectorSourceNodeByStream;  // 流ID -> vector需求上界的节点ID。
    std::unordered_map<uint16_t, CoreDemand> externalScopes;          // 尚未启动的外部scope需求。
    CoreDemand residentDemand;                                        // 当前分支所有常驻scope的核数之和。
  };

  // WAIT资源依赖搜索、分支展开与核需求判定。
  void QueueScopeWaitNotifies(uint16_t scopeId, ResourceSearchState &state) const;
  bool ResourceSearchHasRisk(const ResourceSearchState &state, uint16_t rootScopeId,
                             ResourceConflictInfo *conflictInfo) const;
  void PrepareResourceSearch(const SuperKernelBaseNode &waitNode, uint16_t rootScopeId, const CoreDemand &rootDemand,
                             const std::vector<SuperKernelBaseNode *> &rootNodes,
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

  SuperKernelGraph *graph_;
  const SuperKernelOptionsManager *opts_;  // 可为空；用于判断无配对Value Wait的绕过策略。
  const std::unordered_map<uint16_t, ScopeRuntimeInfo> &scopeInfos_;  // 与并行区间分析共享调用方索引。
  const ScopeConcurrentRangeAnalyzer &ranges_;  // 复用同一份scope索引上的并行区间判定。
  const int64_t &deviceRealCubeNum;  // 引用设备核数缓存，避免构造后缓存更新导致数值过期。
  const int64_t &deviceRealVecNum;
};

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
  explicit LockDetector(SuperKernelGraph &graph)
      : concurrentRanges_(graph, scopeInfos_),
        waitResources_(graph, nullptr, scopeInfos_, concurrentRanges_, deviceRealCubeNum, deviceRealVecNum) {
    Init(graph);
  }

  LockDetector(SuperKernelGraph &graph, const SuperKernelOptionsManager &opts)
      : concurrentRanges_(graph, scopeInfos_),
        waitResources_(graph, &opts, scopeInfos_, concurrentRanges_, deviceRealCubeNum, deviceRealVecNum),
        opts_(&opts) {
    Init(graph);
  }

  ~LockDetector() {
    if (graph_ != nullptr) {
      Reset();
    }
  }

  static aclError GetDeviceCores();
  static const int64_t &GetDeviceCubeNum() {
    return deviceRealCubeNum;
  }
  static const int64_t &GetDeviceVecNum() {
    return deviceRealVecNum;
  }

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
   * @brief 获取最近一次检测到的死锁原因
   */
  DeadlockFailReason GetDeadlockReason() const {
    return deadlockReason_;
  }
  const ResourceConflictInfo &GetResourceConflictInfo() const {
    return resourceConflictInfo_;
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

  // 前置检测与scope组合检测共用的并行区间及资源搜索入口。
  // 一个scope在单条流上的可并行节点范围。
  using ConcurrentNodeRange = ScopeConcurrentRangeAnalyzer::ConcurrentNodeRange;
  using ScopeConcurrentRanges = ScopeConcurrentRangeAnalyzer::ScopeConcurrentRanges;
  using CoreDemand = WaitResourceDependencyAnalyzer::CoreDemand;

  std::unordered_map<uint32_t, ConcurrentNodeRange> CalculateScopeConcurrentRanges(
      const std::vector<SuperKernelBaseNode *> &candidate, uint16_t candidateScopeId) const;
  bool CanRunTogether(uint16_t firstScopeId, uint16_t secondScopeId, const ScopeConcurrentRanges &ranges) const;
  bool CheckWaitResourceBranches(const SuperKernelBaseNode &waitNode, uint16_t rootScopeId,
                                 const CoreDemand &rootDemand, const std::vector<SuperKernelBaseNode *> &rootNodes,
                                 const std::unordered_set<uint16_t> &initialResidents,
                                 const ScopeConcurrentRanges *precomputedRanges = nullptr,
                                 ResourceConflictInfo *conflictInfo = nullptr);
  bool HasEnoughDeviceCores(uint64_t skCubeNum, uint64_t skVecNum, uint64_t depCubeNum, uint64_t depVecNum) const;
  bool ShouldBypassValueWaitDeadlock(const SuperKernelBaseNode &curNode) const;

 private:
  // 保存scope的核数、流范围、成员节点及是否包含Wait等运行时信息。
  using ScopeRuntimeInfo = ::ScopeRuntimeInfo;

  void Init(SuperKernelGraph &graph);
  std::pair<uint64_t, uint64_t> GetAvailableCores(bool isSuperKernel) const;
  bool IsInSKStream(const SuperKernelBaseNode &node) const;
  void UpdateNodeInfo(const SuperKernelBaseNode &node);

  // 单scope前驱依赖死锁检查。
  bool HasDeadlock(SuperKernelBaseNode *curNode, std::unordered_set<uint16_t> &expandedScopeIds);
  bool CheckScopeDeadlock(const ScopeRuntimeInfo &scopeInfo, std::unordered_set<uint16_t> &expandedScopeIds);
  bool HasDeadlockPredecessor(SuperKernelBaseNode *curNode, std::unordered_set<uint16_t> &expandedScopeIds);
  bool CheckKernelNodeDeadlock(SuperKernelBaseNode *preNode, std::unordered_set<uint16_t> &expandedScopeIds);
  bool CheckWaitNodeDeadlock(SuperKernelBaseNode *preNode, std::unordered_set<uint16_t> &expandedScopeIds);
  bool CheckNotifyNodeDeadlock(SuperKernelBaseNode *preNode, std::unordered_set<uint16_t> &expandedScopeIds);

  // 核需求、已访问节点与当前scope的流范围。
  bool HasEnoughCores(const SuperKernelBaseNode *curNode, bool isSuperKernel);
  void RecordResourceConflict(uint32_t externalCubeNum, uint32_t externalVecNum, uint64_t cubeSourceNodeId,
                              uint64_t vectorSourceNodeId);
  void RollbackVisitedState(std::vector<uint64_t> &visitedNodes);
  void UpdateSKRangeInStream(const SuperKernelBaseNode &curNode);
  void UpdateSKRangeInStream(uint32_t streamId, const SuperKernelBaseNode &headNode,
                             const SuperKernelBaseNode &tailNode);
  bool IsBeforeSKRange(const SuperKernelBaseNode &curNode);
  bool IsAfterSKRange(const SuperKernelBaseNode &curNode);

  // 按节点类型分派融合判定，并处理Wait与Notify的依赖。
  bool GetFusibleStatus(SuperKernelBaseNode &curNode);
  bool GetWaitNodeFusibleStatus(SuperKernelBaseNode &curNode);
  bool CheckNotifyInSKStream(SuperKernelBaseNode &curNode, SuperKernelBaseNode &notifyNode);

  std::vector<uint64_t> nodes;                                                // visited nodes
  std::vector<SuperKernelBaseNode *> candidateScopeNodes_;                    // 当前候选scope中的节点
  std::unordered_map<uint32_t, ConcurrentNodeRange> activeConcurrentRanges_;  // 当前生效的流并行范围
  std::vector<uint64_t> tempVisitedNodes;                                     // temporary visited nodes for HasDeadlock
  uint32_t depOpCubeNum;                                                      // visited op cube num outside superkernel
  uint32_t depOpVecNum;                                                       // visited op vec num outside superkernel
  uint64_t depOpCubeSourceNodeId_ = INVALID_TASK_ID;
  uint64_t depOpVectorSourceNodeId_ = INVALID_TASK_ID;
  ResourceConflictInfo resourceConflictInfo_;
  uint32_t superKernelCubeNum;
  uint32_t superKernelVecNum;
  uint16_t currentScopeId_ = INVALID_SCOPE_ID;
  std::unordered_map<uint16_t, ScopeRuntimeInfo> scopeInfos_;
  ScopeConcurrentRangeAnalyzer concurrentRanges_;
  WaitResourceDependencyAnalyzer waitResources_;
  bool scopeCoreInfoChanged_ = false;  // Whether the current candidate changed physical core requirements
  static int64_t deviceRealCubeNum;
  static int64_t deviceRealVecNum;
  std::unordered_set<uint32_t> skStreamIds;
  uint32_t nodeNum;
  uint32_t kernelNodeNum;
  std::unordered_map<uint32_t, std::pair<uint64_t, uint64_t>> skRangeInStream;
  SuperKernelGraph *graph_;  // 存储graph指针，用于析构时调用Reset
  const SuperKernelOptionsManager *opts_ = nullptr;
  DeadlockFailReason deadlockReason_ = DeadlockFailReason::NOT_FIND_DEADLOCK;  // 当前检测到的死锁原因
};

class ScopeDeadlockDetector {
 public:
  ScopeDeadlockDetector(SuperKernelGraph &graph, const SuperKernelOptionsManager &opts)
      : concurrentRanges_(graph, scopeInfos_),
        waitResources_(graph, &opts, scopeInfos_, concurrentRanges_, LockDetector::GetDeviceCubeNum(),
                       LockDetector::GetDeviceVecNum()) {
    LockDetector::GetDeviceCores();
  }

  // 返回风险Wait所属scope，并通过riskWaitId输出对应节点；超限组合的新加入scope通过overCapacityScopeIds输出。
  const SuperKernelScopeInfo *CheckParallelScopeResources(const std::vector<SuperKernelScopeInfo> &scopes,
                                                          uint64_t &riskWaitId,
                                                          ResourceConflictInfo *conflictInfo = nullptr,
                                                          std::vector<uint16_t> *overCapacityScopeIds = nullptr);

 private:
  using ScopeConcurrentRanges = ScopeConcurrentRangeAnalyzer::ScopeConcurrentRanges;

  static std::string FormatScopeIds(const std::vector<const SuperKernelScopeInfo *> &combination);
  std::vector<const SuperKernelScopeInfo *> CollectWaitScopeCandidates(
      const std::vector<SuperKernelScopeInfo> &scopes) const;
  ScopeConcurrentRanges BuildScopeConcurrentRanges(const std::vector<SuperKernelScopeInfo> &scopes) const;
  bool CheckSelectedScopeResourceBranches(const std::vector<const SuperKernelScopeInfo *> &selected,
                                          const ScopeConcurrentRanges &ranges, uint64_t &riskWaitId,
                                          const SuperKernelScopeInfo *&riskScope, ResourceConflictInfo *conflictInfo);

  std::unordered_map<uint16_t, ScopeRuntimeInfo> scopeInfos_;
  ScopeConcurrentRangeAnalyzer concurrentRanges_;
  WaitResourceDependencyAnalyzer waitResources_;
};

#endif  // __SK_LOCK_DETECTOR_H__
