/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sk_task_lowering.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <unordered_map>
#include <utility>

#include "sk_graph.h"
#include "sk_log.h"
#include "sk_node.h"
#include "sk_options_manager.h"
#include "sk_queue_type_resolver.h"
#include "sk_scope_info.h"

namespace {
constexpr uint64_t kDebugOptionDisableDcci = 0x1;
constexpr uint64_t kDebugOptionDcciBeforeKernelStart = 0x4;
constexpr uint64_t kDebugOptionDcciAfterKernelEnd = 0x8;
constexpr uint64_t kDebugOptionCrossCoreSyncCheck = 0x10;
constexpr uint64_t kDebugOptionEnableDcciAfterFunc = 0x20;
SkTaskKind ToBuildTaskKind(SkNodeType nodeType) {
  switch (nodeType) {
    case SkNodeType::NODE_KERNEL:
      return SkTaskKind::KERNEL;
    case SkNodeType::NODE_NOTIFY:
      return SkTaskKind::NOTIFY;
    case SkNodeType::NODE_WAIT:
      return SkTaskKind::WAIT;
    case SkNodeType::NODE_RESET:
      return SkTaskKind::RESET;
    default:
      return SkTaskKind::KERNEL;
  }
}

bool IsSupportedTaskKind(SkNodeType nodeType) {
  return nodeType == SkNodeType::NODE_KERNEL || nodeType == SkNodeType::NODE_NOTIFY ||
         nodeType == SkNodeType::NODE_WAIT || nodeType == SkNodeType::NODE_RESET;
}

bool MatchKernelOption(const SuperKernelOptionsManager &options, const std::vector<std::string> &kernelList,
                       const std::string &functionName) {
  return !kernelList.empty() && !functionName.empty() && options.MatchKernelNameInList(kernelList, functionName);
}

uint64_t BuildDebugOptions(const SuperKernelOptionsManager &options, const KernelInfos &kernelInfo,
                           const std::vector<std::string> &disableDcciList,
                           const std::vector<std::string> &dcciBeforeList,
                           const std::vector<std::string> &dcciAfterList, bool crossCoreSyncCheck) {
  const bool disableByCap = kernelInfo.capBits.disableDcci;
  const bool disableByOption = MatchKernelOption(options, disableDcciList, kernelInfo.funcName);
  const bool dcciBefore = MatchKernelOption(options, dcciBeforeList, kernelInfo.funcName);
  bool dcciAfter = MatchKernelOption(options, dcciAfterList, kernelInfo.funcName);
  if (disableByOption && !disableByCap) {
    SK_LOGW(
        "DCCI disable option and SK_BIND mismatch for %s: dcci_disable_on_kernel=true, "
        "but SK_BIND disable_dcci=0. The sub-kernel binary may be reused from stale cache.",
        kernelInfo.funcName.c_str());
  }
  if (!disableByCap && disableByOption && !dcciBefore && !dcciAfter) {
    dcciAfter = true;
    SK_LOGI(
        "Apply dcci_after_kernel_end fallback for %s: SK_BIND disable_dcci=0, "
        "dcci_disable_on_kernel option matched, before/after dcci options not matched",
        kernelInfo.funcName.c_str());
  }
  uint64_t result = 0;
  if (disableByCap || disableByOption) {
    result |= kDebugOptionDisableDcci;
  }
  if (dcciBefore) {
    result |= kDebugOptionDcciBeforeKernelStart;
  }
  if (dcciAfter) {
    result |= kDebugOptionDcciAfterKernelEnd;
  }
  if (crossCoreSyncCheck) {
    result |= kDebugOptionCrossCoreSyncCheck;
  }
  if ((result & kDebugOptionDcciAfterKernelEnd) != 0 || (result & kDebugOptionDisableDcci) == 0) {
    result |= kDebugOptionEnableDcciAfterFunc;
  }
  return result;
}

std::vector<std::string> GetStringListOption(const SuperKernelOptionsManager &options, aclskOptionType type) {
  const auto *option = options.GetOption(type);
  return option == nullptr ? std::vector<std::string>{} : option->GetStringListValue();
}

uint32_t GetIntOption(const SuperKernelOptionsManager &options, aclskOptionType type, uint32_t defaultValue) {
  const auto *option = options.GetOption(type);
  return option == nullptr ? defaultValue : option->GetIntValue();
}

std::array<uint64_t, kSkCoreTypeCount> EffectivePrefetch(const ResolvedFunctionInfo &resolved, uint32_t preloadMode) {
  std::array<uint64_t, kSkCoreTypeCount> prefetchCount = {std::min(resolved.prefetchCnt[0], SK_AIC_MAX_PRELOAD_COUNT),
                                                          std::min(resolved.prefetchCnt[1], SK_AIV_MAX_PRELOAD_COUNT)};
  switch (preloadMode) {
    case ACLSK_PRELOAD_MAX:
      return {SK_AIC_MAX_PRELOAD_COUNT, SK_AIV_MAX_PRELOAD_COUNT};
    case ACLSK_PRELOAD_DISABLED:
      return {0, 0};
    case ACLSK_PRELOAD_MIN:
      return {1, 1};
    case ACLSK_PRELOAD_PER_FUNC_HALF_AIV:
      prefetchCount[1] = prefetchCount[1] / 2U + prefetchCount[1] % 2U;
      break;
    case ACLSK_PRELOAD_PER_FUNC_MIN_AIV:
      prefetchCount[1] = std::min(prefetchCount[1], uint64_t{1});
      break;
    case ACLSK_PRELOAD_PER_FUNC:
    default:
      break;
  }
  return prefetchCount;
}

bool ValidateCoreRequirement(const SuperKernelBaseNode &node, const ScopeCoreInfo &coreInfo) {
  const uint32_t cubeNum = node.GetCubeNum();
  const uint32_t vectorNum = node.GetVecNum();
  if (node.GetKernelType() == SkKernelType::MIX_AIC_1_2 && coreInfo.type != SkKernelType::MIX_AIC_1_2) {
    return false;
  }
  if (!node.RequiresExactCoreMatch()) {
    return cubeNum <= coreInfo.GetCubeNum() && vectorNum <= coreInfo.GetVectorNum();
  }
  if (cubeNum > coreInfo.GetCubeNum() || vectorNum > coreInfo.GetVectorNum()) {
    return false;
  }
  return cubeNum == 0 && vectorNum != 0 ? vectorNum == coreInfo.GetVectorNum() : cubeNum == coreInfo.GetCubeNum();
}

bool LowerPayload(const SuperKernelBaseNode &node, const SuperKernelOptionsManager &options, SkTaskDesc &result) {
  result.nodeId = node.GetNodeId();
  result.kind = ToBuildTaskKind(node.GetNodeType());
  result.debugLabel = node.Format();
  result.successorIds.assign(node.sendToNodeId.cbegin(), node.sendToNodeId.cend());
  std::sort(result.successorIds.begin(), result.successorIds.end());
  result.successorIds.erase(std::unique(result.successorIds.begin(), result.successorIds.end()),
                            result.successorIds.end());
  if (node.GetNodeType() != SkNodeType::NODE_KERNEL) {
    const SyncInfos &syncInfo = node.GetNodeInfos().syncInfos;
    result.payload = SkEventDesc{syncInfo.addrValue, syncInfo.memoryValue, syncInfo.memoryWaitFlag};
    return true;
  }

  const auto argsSize = node.GetNodeInfos().kernelInfos.devArgsSize;
  if (argsSize > std::numeric_limits<uint32_t>::max()) {
    SK_LOGE("Lower failed: kernel args size overflow, nodeId=%lu, argsSize=%zu", node.GetNodeId(), argsSize);
    return false;
  }
  const KernelInfos &kernelInfo = node.GetNodeInfos().kernelInfos;
  if (kernelInfo.resolvedNum > K_MAX_SPLIT_BIN_COUNT) {
    SK_LOGE("Lower failed: resolved function count overflow, nodeId=%lu, count=%u", node.GetNodeId(),
            kernelInfo.resolvedNum);
    return false;
  }
  const uint32_t preloadMode = GetIntOption(options, aclskOptionType::PRELOAD_CODE, ACLSK_PRELOAD_PER_FUNC);
  const uint32_t splitBinCount = GetIntOption(options, aclskOptionType::SPLIT_MODE, kSkDefaultSplitBinCount);
  if (splitBinCount == 0 || splitBinCount > K_MAX_SPLIT_BIN_COUNT) {
    SK_LOGE("Lower failed: invalid split binary count, count=%u", splitBinCount);
    return false;
  }
  const auto disableDcciList = GetStringListOption(options, aclskOptionType::DCCI_DISABLE_ON_KERNEL);
  const auto dcciBeforeList = GetStringListOption(options, aclskOptionType::DCCI_BEFORE_KERNEL_START);
  const auto dcciAfterList = GetStringListOption(options, aclskOptionType::DCCI_AFTER_KERNEL_END);
  const bool crossCoreSyncCheck = GetIntOption(options, aclskOptionType::DEBUG_CROSS_CORE_SYNC_CHECK, 0) == 1;

  SkKernelDesc kernel;
  kernel.kernelType = kernelInfo.kernelType;
  kernel.originalNumBlocks = kernelInfo.numBlocks;
  kernel.cubeNum = kernelInfo.cubeNum;
  kernel.vectorNum = kernelInfo.vecNum;
  SetBoolAttribute(kernel.attributes, SkTaskAttributeId::KERNEL_SCHE_MODE, kernelInfo.isScheModeOn);
  SetBoolAttribute(kernel.attributes, SkTaskAttributeId::KERNEL_BLOCK_DIM_SCALE_UP, kernelInfo.capBits.blockDimScaleUp);
  SetBoolAttribute(kernel.attributes, SkTaskAttributeId::KERNEL_MIX_SPLIT, kernelInfo.needMixKernelSplit);
  SetBoolAttribute(kernel.attributes, SkTaskAttributeId::EARLY_START_SET, kernelInfo.capBits.earlyStartSetFlag);
  SetBoolAttribute(kernel.attributes, SkTaskAttributeId::EARLY_START_WAIT, kernelInfo.capBits.earlyStartWaitFlag);
  SetBoolAttribute(kernel.attributes, SkTaskAttributeId::SIMT_ENABLED, kernelInfo.isSimtOp);
  if (kernelInfo.hasDynUbufSize) {
    SetUint64Attribute(kernel.attributes, SkTaskAttributeId::SIMT_DYN_UBUF_SIZE, kernelInfo.dynUbufSize);
  }
  if (kernelInfo.hasAllocUbufSize) {
    SetUint64Attribute(kernel.attributes, SkTaskAttributeId::SIMT_ALLOC_UBUF_SIZE, kernelInfo.allocUbufSize);
  }
  kernel.functionName = kernelInfo.funcName;
  kernel.deviceArgs = kernelInfo.devArgs;
  kernel.argsSize = static_cast<uint32_t>(argsSize);
  kernel.binaryHandle = kernelInfo.binHdl;
  kernel.originalFunction = kernelInfo.funcHdl;
  SetUint64Attribute(
      kernel.attributes, SkTaskAttributeId::KERNEL_EFFECTIVE_DEBUG_OPTIONS,
      BuildDebugOptions(options, kernelInfo, disableDcciList, dcciBeforeList, dcciAfterList, crossCoreSyncCheck));
  if (kernelInfo.resolvedNum != splitBinCount) {
    SK_LOGW("mismatch num between sub sk registered and sk option, funcName: %s, registered: %u,option: %u",
            kernelInfo.funcName.c_str(), kernelInfo.resolvedNum, splitBinCount);
  }
  kernel.functions.reserve(splitBinCount);
  for (uint32_t i = 0; i < splitBinCount; ++i) {
    const ResolvedFunctionInfo &resolved = kernelInfo.resolvedFuncs[i];
    SkResolvedFunctionDesc function;
    function.functionAddress = {resolved.funcAddr[0], resolved.funcAddr[1]};
    function.functionOffset = {resolved.funcOffset[0], resolved.funcOffset[1]};
    function.prefetchCount = EffectivePrefetch(resolved, preloadMode);
    kernel.functions.push_back(function);
  }
  result.payload = std::move(kernel);
  return true;
}

bool LowerTasks(const std::vector<SuperKernelBaseNode *> &nodes, const SuperKernelOptionsManager &options,
                const SuperKernelGraph &graph, std::vector<SkTaskDesc> &descs) {
  descs.clear();
  descs.reserve(nodes.size());
  std::unordered_map<uint64_t, const SuperKernelBaseNode *> nodeIdToTask;
  nodeIdToTask.reserve(nodes.size());
  for (const auto *node : nodes) {
    if (node == nullptr || node->GetNodeId() == INVALID_TASK_ID) {
      SK_LOGE("Build failed: task list contains null or invalid node");
      return false;
    }
    nodeIdToTask.emplace(node->GetNodeId(), node);
  }
  for (const auto *node : nodes) {
    if (node == nullptr || !IsSupportedTaskKind(node->GetNodeType())) {
      SK_LOGE("Lower failed: null or unsupported task node");
      return false;
    }
    SkTaskDesc desc;
    if (!LowerPayload(*node, options, desc)) {
      return false;
    }
    for (uint64_t targetId : desc.successorIds) {
      const auto target = nodeIdToTask.find(targetId);
      if (target != nodeIdToTask.end() && target->second->receiveNodeId.count(node->GetNodeId()) == 0) {
        SK_LOGW("ValidateGraphRelations: reverse relation missing for edge %lu -> %lu", node->GetNodeId(), targetId);
      }
    }
    descs.push_back(std::move(desc));
  }
  std::vector<SkQueueType> queueTypes;
  if (!SkQueueTypeResolver(graph).Resolve(nodes, queueTypes) || queueTypes.size() != descs.size()) {
    return false;
  }
  for (size_t i = 0; i < descs.size(); ++i) {
    descs[i].queueType = queueTypes[i];
  }
  return true;
}

}  // namespace

bool SkTaskLowering::Run(std::string functionName, const std::vector<SuperKernelBaseNode *> &tasks, uint16_t scopeId,
                         const ScopeCoreInfo &coreInfo, SkTaskIR &output) const {
  output = {};
  if (tasks.empty()) {
    SK_LOGE("Build failed: no task to build for super kernel");
    return false;
  }
  if (!coreInfo.IsValid()) {
    SK_LOGE("Build failed: scope %u has no final SK core result", scopeId);
    return false;
  }
  for (const auto *task : tasks) {
    if (task != nullptr && task->GetNodeType() == SkNodeType::NODE_KERNEL &&
        !ValidateCoreRequirement(*task, coreInfo)) {
      SK_LOGE(
          "kernel %lu core requirement does not match final SK result, taskCore={cube:%u,vec:%u}, "
          "scheMode=%d, blockDimScaleUp=%d, skCore={cube:%u,vec:%u}, skType=%s",
          task->GetNodeId(), task->GetCubeNum(), task->GetVecNum(), task->IsScheModeOn(),
          task->GetNodeInfos().kernelInfos.capBits.blockDimScaleUp, coreInfo.GetCubeNum(), coreInfo.GetVectorNum(),
          to_string(coreInfo.type));
      return false;
    }
  }

  SkTaskIR lowered;
  lowered.functionName = std::move(functionName);
  lowered.scopeId = scopeId;
  lowered.coreSpec = {coreInfo.type, coreInfo.numBlocks, coreInfo.GetCubeNum(), coreInfo.GetVectorNum()};
  lowered.config.splitBinCount = GetIntOption(options_, aclskOptionType::SPLIT_MODE, kSkDefaultSplitBinCount);
  lowered.config.counterSlotCount = GetSkRuntimeConfig().eventCoreNum;
  lowered.config.kernelArch = GetSkRuntimeConfig().kernelArch;
  SetBoolAttribute(lowered.config.attributes, SkTaskAttributeId::CONFIG_DEBUG_ENABLED, options_.EnableDebug());
  SetBoolAttribute(lowered.config.attributes, SkTaskAttributeId::CONFIG_DEBUG_SYNC_ALL,
                   GetIntOption(options_, aclskOptionType::DEBUG_SYNC_ALL, 0) != 0);
  SetBoolAttribute(lowered.config.attributes, SkTaskAttributeId::CONFIG_STREAM_FUSION_ENABLED,
                   GetIntOption(options_, aclskOptionType::STREAM_FUSION, 1) != 0);
  SetBoolAttribute(lowered.config.attributes, SkTaskAttributeId::CONFIG_MIX_KERNEL_SPLIT_ENABLED,
                   options_.IsInnerOptionEnabled(SkInnerOptionType::MIX_KERNEL_SPLIT));
  SetBoolAttribute(
      lowered.config.attributes, SkTaskAttributeId::CONFIG_EARLY_START_ENABLED,
      GetIntOption(options_, aclskOptionType::EARLY_START, 0) == aclskEarlyStartValue::ACLSK_EARLY_START_ENABLED);
  SetBoolAttribute(lowered.config.attributes, SkTaskAttributeId::CONFIG_SIMT_ENABLED,
                   options_.IsInnerOptionEnabled(SkInnerOptionType::SIMT_OP_SUPPORT));
  const char *profiling = std::getenv("ASCEND_PROF_SK_ON");
  SetBoolAttribute(lowered.config.attributes, SkTaskAttributeId::CONFIG_PROFILING_ENABLED,
                   profiling != nullptr && std::string(profiling) != "0");
  const bool crossCoreSyncCheckEnabled = GetIntOption(options_, aclskOptionType::DEBUG_CROSS_CORE_SYNC_CHECK, 0) == 1;
  SetBoolAttribute(lowered.config.attributes, SkTaskAttributeId::CONFIG_CROSS_CORE_SYNC_CHECK_ENABLED,
                   crossCoreSyncCheckEnabled);
  SetBoolAttribute(lowered.config.attributes, SkTaskAttributeId::CONFIG_OP_TRACE_ENABLED,
                   GetIntOption(options_, aclskOptionType::DEBUG_OP_EXEC_TRACE, 0) == 1 || crossCoreSyncCheckEnabled);

  if (!LowerTasks(tasks, options_, graph_, lowered.tasks)) {
    return false;
  }
  output = std::move(lowered);
  return true;
}
