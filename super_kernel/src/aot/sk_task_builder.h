/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __SK_TASK_BUILDER_H__
#define __SK_TASK_BUILDER_H__

#include "sk_node.h"
#include "sk_options_manager.h"
#include "sk_scope_info.h"
#include "sk_types.h"

#include <vector>
#include <map>
#include <string>
#include <unordered_map>

// Forward declaration
class SuperKernelGraph;

// ========== Queue type enum (indicates which queue executes the task) ==========

enum class SkQueueType : uint8_t {
  AIC,      // AIC queue only
  AIV,      // AIV queue only
  MIX_1_1,  // Execute on both queues (MIX_AIC_1_1)
  MIX_1_2,  // Execute on both queues (MIX_AIC_1_2)
  UNKNOWN,  // Unknown/invalid type (used for debug or fallback paths)
};

inline const char *to_string(SkQueueType type) {
  switch (type) {
    case SkQueueType::AIC:
      return "AIC";
    case SkQueueType::AIV:
      return "AIV";
    case SkQueueType::MIX_1_1:
      return "MIX_1_1";
    case SkQueueType::MIX_1_2:
      return "MIX_1_2";
    case SkQueueType::UNKNOWN:
      return "UNKNOWN";
    default:
      return "UNKNOWN";
  }
}

// ========== Node-oriented sync metadata (aligned with Python behavior) ==========

// Sync direction type (corresponds to Python labels such as "cub:vec", "vec:cub")
enum class SyncDirection : uint8_t {
  NONE = 0,    // No sync
  CUB_TO_CUB,  // AIC -> AIC
  VEC_TO_VEC,  // AIV -> AIV
  CUB_TO_VEC,  // AIC -> AIV
  VEC_TO_CUB,  // AIV -> AIC
  MIX_TO_MIX,  // Bidirectional sync (MIX -> MIX)
  ALL_SYNC,    // Full-core sync
};

inline const char *to_string(SyncDirection dir) {
  switch (dir) {
    case SyncDirection::NONE:
      return "NONE";
    case SyncDirection::CUB_TO_CUB:
      return "CUB_TO_CUB";
    case SyncDirection::VEC_TO_VEC:
      return "VEC_TO_VEC";
    case SyncDirection::CUB_TO_VEC:
      return "CUB_TO_VEC";
    case SyncDirection::VEC_TO_CUB:
      return "VEC_TO_CUB";
    case SyncDirection::MIX_TO_MIX:
      return "MIX_TO_MIX";
    case SyncDirection::ALL_SYNC:
      return "ALL_SYNC";
    default:
      return "UNKNOWN";
  }
}

struct EarlyStartInfo {
  uint32_t funcEarlyStartConfig = 0U;
  SuperKernelBaseNode *relatedSetNode = nullptr;
  SuperKernelBaseNode *relatedWaitNode = nullptr;
  SuperKernelBaseNode *nextAicRelatedNode = nullptr;
  SuperKernelBaseNode *nextAivRelatedNode = nullptr;
  uint32_t syncEarlyStartConfig = 0U;
  void ApplyFuncMask(SkEarlyStartMask mask) {
    funcEarlyStartConfig |= static_cast<uint32_t>(mask);
  }
  bool CheckFuncMask(SkEarlyStartMask mask) const {
    return (funcEarlyStartConfig & static_cast<uint32_t>(mask)) != 0;
  }
  void ApplySyncMask(SkEarlyStartMask mask) {
    syncEarlyStartConfig |= static_cast<uint32_t>(mask);
  }
  bool CheckSyncMask(SkEarlyStartMask mask) const {
    return (syncEarlyStartConfig & static_cast<uint32_t>(mask)) != 0;
  }
};

// Per-task sync metadata
struct TaskSyncInfo {
  SkQueueType queueType;  // Task execution queue: AIC/AIV/MIX

  // AIC queue sync metadata (cub_op_list)
  std::map<size_t, SyncDirection> cubSendInfo;  // Target tasks receiving sync from this task
  std::map<size_t, SyncDirection> cubRecvInfo;  // Source tasks sending sync to this task

  // AIV queue sync metadata (vec_op_list)
  std::map<size_t, SyncDirection> vecSendInfo;  // Target tasks receiving sync from this task
  std::map<size_t, SyncDirection> vecRecvInfo;  // Source tasks sending sync to this task

  // Cross-core sync direction: 0=CUBE(CUB_TO_CUB), 1=VEC(VEC_TO_VEC)
  std::map<size_t, SyncDirection> crossSyncInfo;
  EarlyStartInfo earlyStartInfo{};
  TaskSyncInfo() : queueType(SkQueueType::UNKNOWN) {}
};

// SkBuildResult - contains both launch info and task queue JSON
struct SkBuildResult {
  SkLaunchInfo launchInfo;
  Json taskQueueJson;
};

class SkTaskBuilder {
 public:
  SkTaskBuilder(SuperKernelOptionsManager &opts, const SuperKernelGraph &graph) : opts(opts), graph_(graph) {}

  SkBuildResult Build(std::string skFuncName, const std::vector<SuperKernelBaseNode *> &tasks,
                      const std::vector<SuperKernelBaseNode *> &customTasks, uint16_t scopeId,
                      const ScopeCoreInfo &scopeCoreInfo);

 private:
  SuperKernelOptionsManager &opts;
  const SuperKernelGraph &graph_;  // Graph reference

  // Sync metadata storage: each task maintains its own send/recv maps
  std::vector<TaskSyncInfo> taskSyncInfos_;
  std::unordered_map<uint64_t, size_t> nodeIdToIndex_;
  std::unordered_map<size_t, uint64_t> indexToNodeId_;
  bool aicAvailable_ = false;
  bool aivAvailable_ = false;

  // Task insertion helpers, separated by task type
  std::pair<int, int> GetPreFetchCnt(const ResolvedFunctionInfo &resolved) const;
  bool AddSyncTask(SkTask &skTask, size_t nodeIndex, SkCoreSyncType syncType, uint8_t earlyStartConfig = 0U,
                   uint32_t skipCoreCount = 0U, SkKernelType relatedType = SkKernelType::DEFAULT);
  bool AddEventTask(SkTask &skTask, SuperKernelBaseNode *node, size_t nodeIndex, SkTaskType taskType);
  bool AddFuncTask(SkTask &skTask, SuperKernelBaseNode *node, SkDfxInfo *dfxInfo, size_t nodeIndex, int addrIndex,
                   int binCount, SkTaskType taskType, uint32_t numBlocks);

  bool DispatchFuncTask(SkTask &skTaskCube, SkTask &skTaskVec, SuperKernelBaseNode *node, SkDfxInfo *dfxInfo,
                        size_t nodeIndex, int binCount, SkTaskType taskType, SkQueueType queueType,
                        const ScopeCoreInfo &scopeCoreInfo);
  bool DispatchEventTask(SkTask &skTaskCube, SkTask &skTaskVec, SuperKernelBaseNode *node, size_t nodeIndex,
                         SkTaskType taskType, SkQueueType queueType);

  bool DispatchSyncTasks(SkTask &skTaskCube, SkTask &skTaskVec, size_t nodeIndex,
                         const std::map<size_t, SyncDirection> &syncInfo, bool isSend, SkQueueType queueType);
  bool DispatchSyncTasks(SkTask &skTaskCube, SkTask &skTaskVec, size_t nodeIndex, const EarlyStartInfo &earlyStartInfo,
                         bool isSend, SkQueueType queueType, const ScopeCoreInfo &scopeCoreInfo);

  // ========== Graph-topology-based sync extraction ==========

  // Initialize taskSyncInfos_
  bool InitTaskSyncInfos(const std::vector<SuperKernelBaseNode *> &tasks);

  // Precompute sync relations (based on graph topology)
  bool PrecomputeSyncRelationsFromGraph(const std::vector<SuperKernelBaseNode *> &tasks);

  // Precompute sync relations by splitting MIX kernels into standalone groups.
  bool PrecomputeSyncRelationsByMixGroups(const std::vector<SuperKernelBaseNode *> &tasks);
  bool SplitTasksByMixGroups(const std::vector<SuperKernelBaseNode *> &tasks,
                             std::vector<std::vector<SuperKernelBaseNode *>> &splitTasks, bool &hasMixKernel) const;
  bool InitSyncInfoSnapshotForMixGroups(const std::vector<SuperKernelBaseNode *> &tasks,
                                        std::vector<TaskSyncInfo> &taskSyncInfosOrigin);
  bool ProcessSyncRelationSplitGroup(const std::vector<SuperKernelBaseNode *> &curSplitTasks, size_t groupIndex,
                                     size_t groupOffset, bool hasNextGroup,
                                     const std::vector<TaskSyncInfo> &taskSyncInfosOrigin,
                                     std::vector<TaskSyncInfo> &mergedTaskSyncInfos);
  bool RebaseTaskSyncInfo(TaskSyncInfo &syncInfo, size_t offset) const;
  void AddBoundaryAllSync(const std::vector<SuperKernelBaseNode *> &curSplitTasks, size_t groupIndex,
                          size_t groupOffset);
  bool IsMixKernelTask(const SuperKernelBaseNode *task) const;

  // Extract intra-stream sync relations (based on GetNextNodeId)
  void ExtractIntraStreamSync(const std::vector<SuperKernelBaseNode *> &tasks);

  // Extract inter-stream sync relations (event-based)
  bool ExtractInterStreamSync(const std::vector<SuperKernelBaseNode *> &tasks);

  // ========== Core sync insertion (aligned with Python insert_sync_event) ==========

  // Insert sync event: preOp -> currOp
  void InsertSyncEvent(size_t preIdx, size_t currIdx);

  // ========== Optimization methods ==========
  void OptimizeSyncRelations(const std::vector<SuperKernelBaseNode *> &tasks);
  void RemoveCrossedLineSync();
  void RemoveMultiSendSync();
  void RemoveMultiRecvSync();
  void RemoveRedundantCrossSync(const std::vector<SuperKernelBaseNode *> &tasks);
  // ========== Early-start-specific sync methods ==========
  bool ApplyEarlyStartSyncPass(const std::vector<SuperKernelBaseNode *> &tasks);

  // Helper: determine whether crossed sync can be removed
  bool JudgeRemoveCrossSync(size_t sendIdx, size_t recvIdx, bool isCubToVec);

  // Helper: remove sync metadata
  void RemoveSyncInfo(size_t sendIdx, size_t recvIdx, bool isRemoveRecv, SyncDirection dirToRemove);

  // Print sync metadata (debug only)
  void PrintSyncInfo(const char *stage) const;

  SkHostEntryInfo GenEntryInfo(SkTask &skTaskCube, SkTask &skTaskVec, const ScopeCoreInfo &scopeCoreInfo,
                               bool useSimtEntry = false);
  bool CalculateSkScheMode(const std::vector<SuperKernelBaseNode *> &tasks, const SkTask &aicTask,
                           const SkTask &aivTask) const;
  DeviceArgsPtr GenEntryArgs(const SkTask &skTaskCube, const SkTask &skTaskVec, const SkDfxInfo *dfxInfos,
                             uint32_t dfxCount, const SkEventConfig *eventConfig = nullptr);

  // DFX info update helpers
  bool UpdateDfxInfo(SkDfxInfo *dfxInfo, const KernelInfos &kernelInfo, const ResolvedFunctionInfo &resolved,
                     int binIndex, int addrIndex);
  // Helper to process core function size (AIC/AIV)
  bool ProcessCoreFuncSize(SkDfxInfo *dfxInfo, aclrtBinHandle binHdl, const void *binHostAddr, uint32_t binHostSize,
                           const ResolvedFunctionInfo &resolved, int coreIndex, int binIndex, const char *coreName);
};
#endif  // __SK_TASK_BUILDER_H__
