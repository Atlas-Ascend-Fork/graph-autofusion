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
#include <algorithm>
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
  struct ScopeRuntimeInfo {
    ScopeCoreInfo coreInfo;
    std::unordered_set<uint32_t> streamIds;
    std::unordered_map<uint32_t, uint64_t> headNodeIds;
    std::unordered_map<uint32_t, uint64_t> tailNodeIds;
  };

  struct FusedNotifyWaitInfo {
    uint64_t waitId;
    std::unordered_set<uint32_t> predecessorStreamIds;
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

  bool HasDeadlock(SuperKernelBaseNode *curNode);

  bool CheckKernelNodeDeadlock(SuperKernelBaseNode *preNode);

  bool CheckWaitNodeDeadlock(SuperKernelBaseNode *preNode);

  bool CheckNotifyNodeDeadlock(SuperKernelBaseNode *preNode);

  bool GetWaitNodeFusibleStatus(SuperKernelBaseNode &curNode);

  bool CheckNotifyInSKStream(SuperKernelBaseNode &curNode, SuperKernelBaseNode &notifyNode);

  bool GetFusibleStatus(SuperKernelBaseNode &curNode);
  bool ShouldBypassValueWaitDeadlock(const SuperKernelBaseNode &curNode) const;

  bool HasEnoughCores(const SuperKernelBaseNode *curNode, bool isSuperKernel);
  bool HasEnoughDeviceCores(uint64_t skCubeNum, uint64_t skVecNum, uint64_t depCubeNum, uint64_t depVecNum) const;
  bool CheckNotifyWaitScopeCombinations(const SuperKernelBaseNode &waitNode, const SuperKernelBaseNode &notifyNode,
                                        uint32_t skCubeNum, uint32_t skVecNum, uint32_t depCubeNum, uint32_t depVecNum,
                                        const std::unordered_set<uint32_t> &predecessorStreamIds);
  bool CheckFusedWaitNotifies(uint32_t skCubeNum, uint32_t skVecNum, uint32_t depCubeNum, uint32_t depVecNum);
  bool IsSerializedWithCandidateScope(const SuperKernelBaseNode &waitNode, uint64_t candidateWaitNodeId,
                                      const ScopeRuntimeInfo &candidateScope) const;
  bool IsNodeBeforeCandidateScope(const SuperKernelBaseNode &node, const ScopeRuntimeInfo &candidateScope) const;
  bool IsNodeAfterCandidateScope(const SuperKernelBaseNode &node, const ScopeRuntimeInfo &candidateScope) const;
  void MergeCandidateScopeRange(const ScopeRuntimeInfo &candidateScope);
  void RollbackVisitedState(std::vector<uint64_t> &visitedNodes);

  std::vector<uint64_t> nodes;             // visited nodes
  std::vector<uint64_t> tempVisitedNodes;  // temporary visited nodes for HasDeadlock
  uint32_t depOpCubeNum;                   // visited op cube num outside superkernel
  uint32_t depOpVecNum;                    // visited op vec num outside superkernel
  uint32_t superKernelCubeNum;
  uint32_t superKernelVecNum;
  uint16_t currentScopeId_ = INVALID_SCOPE_ID;
  std::unordered_map<uint16_t, ScopeRuntimeInfo> scopeInfos_;
  std::unordered_map<uint64_t, FusedNotifyWaitInfo> fusedNotifyWaitInfos_;
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
