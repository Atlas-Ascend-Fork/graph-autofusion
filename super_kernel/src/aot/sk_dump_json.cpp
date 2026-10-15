/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sk_dump_json.cpp
 * \brief Implementation of SuperKernel graph JSON dump utilities
 */

#include "sk_dump_json.h"
#include "sk_common.h"
#include "sk_graph.h"
#include "sk_lock_detector.h"
#include "sk_log.h"
#include "sk_node.h"
#include "sk_options_manager.h"
#include "sk_scope_info.h"
#include "sk_scope_split.h"
#include "sk_types.h"

#include <acl/acl_rt.h>
#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// For resolved function info
#include "runtime/rt_external_kernel.h"
#include "runtime/kernel.h"

SkBindMap InitSuperKernelBindMap(aclrtBinHandle binHdl);

namespace {

struct SkInfoForDump {
  uint16_t scopeId = 0;
  std::string scopeName;
  std::string startKernelFuncName;
  std::string endKernelFuncName;
};

using SkInfoForDumpMap = std::unordered_map<uint64_t, SkInfoForDump>;

const char *ScopeBreakReasonDetail(ScopeBreakReason reason) {
  switch (reason) {
    case ScopeBreakReason::UNFUSIBLE_NODE:
      return "There exists unfusible node in scope";
    case ScopeBreakReason::DEADLOCK_DETECTED:
      return "There exists deadlock in scope";
    case ScopeBreakReason::SYNCALL_OP_DROP:
      return "The core count of syncall operators in scope should be greater than or equal to other operators";
    case ScopeBreakReason::DEBUG_PER_OP_MAX_CORE:
      return "Per-Op debug mode: each operator is an independent scope";
    default:
      return "";
  }
}

/**
 * @brief Fill symbol info for a single core type (AIC/AIV) in resolved function info
 */
static void FillSymbolInfoForCore(aclrtBinHandle binHdl, uint64_t skFuncOffset, uint32_t coreIdx,
                                  ResolvedFunctionInfo &info) {
  void *binHostAddr = nullptr;
  uint32_t binHostSize = 0;
  if (rtGetBinBuffer(binHdl, RT_BIN_HOST_ADDR, &binHostAddr, &binHostSize) == 0) {
    std::string symbolName;
    uint64_t funcSize = 0;
    std::string symbolBind;
    if (GetFuncSymbolInfo(binHdl, static_cast<const char *>(binHostAddr), binHostSize, skFuncOffset, symbolName,
                          funcSize, symbolBind)) {
      info.prefetchCnt[coreIdx] = AlignUpAndClamp(funcSize, coreIdx);
      info.symbolBind[coreIdx] = symbolBind;
    }
  }
}

/**
 * @brief Get resolved function info from kernel function handle
 */
void GetResolvedFuncsForDump(aclrtFuncHandle funcHandle, aclrtBinHandle binHdl, ResolvedFunctionInfo resolvedFuncs[],
                             uint32_t &resolvedNum) {
  resolvedNum = 0;
  if (binHdl == nullptr || funcHandle == nullptr) {
    return;
  }

  // Get bind map and function address
  SkBindMap bindMap = InitSuperKernelBindMap(binHdl);
  if (bindMap.empty()) {
    return;
  }

  void *addr[2] = {nullptr, nullptr};
  if (aclrtGetFunctionAddr(funcHandle, addr, addr + 1) != ACL_SUCCESS) {
    return;
  }

  // Get binary device address and calculate offsets
  void *binDevAddr = nullptr;
  size_t binDevSize = 0;
  if (aclrtBinaryGetDevAddress(binHdl, &binDevAddr, &binDevSize) != ACL_SUCCESS) {
    return;
  }

  auto aicItor = bindMap.find((uint64_t)addr[0] - (uint64_t)binDevAddr);
  auto aivItor = bindMap.find((uint64_t)addr[1] - (uint64_t)binDevAddr);

  // Process each split bin
  for (size_t i = 0; i < K_MAX_SPLIT_BIN_COUNT; ++i) {
    ResolvedFunctionInfo &info = resolvedFuncs[i];
    uint32_t validFuncNum = 0;

    if (aicItor != bindMap.end()) {
      uint64_t skFuncOffset = aicItor->second.sknlFuncs[i];
      info.funcAddr[0] = skFuncOffset + (uint64_t)binDevAddr;
      info.funcOffset[0] = skFuncOffset;
      FillSymbolInfoForCore(binHdl, skFuncOffset, 0, info);
      validFuncNum += (i == 0 || skFuncOffset != aicItor->second.sknlFuncs[0]) ? 1 : 0;
    }

    if (aivItor != bindMap.end()) {
      uint64_t skFuncOffset = aivItor->second.sknlFuncs[i];
      info.funcAddr[1] = skFuncOffset + (uint64_t)binDevAddr;
      info.funcOffset[1] = skFuncOffset;
      FillSymbolInfoForCore(binHdl, skFuncOffset, 1, info);
      validFuncNum += (i == 0 || skFuncOffset != aivItor->second.sknlFuncs[0]) ? 1 : 0;
    }

    if (validFuncNum > 0) {
      resolvedFuncs[resolvedNum++] = info;
    }
  }
}

/**
 * @brief Convert pointer to hex string
 */
std::string PtrToHexString(const void *ptr) {
  std::stringstream hexStream;
  hexStream << "0x" << std::hex << reinterpret_cast<uintptr_t>(ptr);
  return hexStream.str();
}

/**
 * @brief Convert unsigned integer to hex string
 */
std::string UintToHexString(uint64_t value) {
  std::stringstream hexStream;
  hexStream << "0x" << std::hex << value;
  return hexStream.str();
}

/**
 * @brief Dump SkTask (AIC/AIV task queue) to JSON
 * @param task SkTask reference
 * @param queueName Queue name ("AIC" or "AIV")
 * @return JSON object representing the task queue
 */
Json SkTaskToJson(const SkTask &task) {
  Json taskJson;
  taskJson["numBlocks"] = task.numBlocks;
  taskJson["funcCnt"] = task.funcCnt;

  const TaskQue *taskQue = task.GetTaskQue();
  if (taskQue != nullptr) {
    taskJson["taskQue"]["taskCnt"] = taskQue->taskCnt;

    Json taskInfos = Json::array();
    for (uint32_t i = 0; i < taskQue->taskCnt; ++i) {
      const TaskInfo &singleTaskInfo = taskQue->taskInfos[i];
      Json taskInfo;
      taskInfo["nodeIndex"] = singleTaskInfo.index;  // 原始任务索引
      taskInfo["type"] = to_string(singleTaskInfo.type);
      taskInfo["numBlocks"] = singleTaskInfo.numBlocks;
      taskInfo["entryCnt"] = singleTaskInfo.entryCnt;
      std::stringstream hexStream;
      hexStream << "0x" << std::hex << singleTaskInfo.args;
      taskInfo["args"] = hexStream.str();
      taskInfo["debugOptions"] = singleTaskInfo.debugOptions;

      const bool isEventTask = singleTaskInfo.type == SkTaskType::TYPE_EVENT_NOTIFY ||
                               singleTaskInfo.type == SkTaskType::TYPE_EVENT_WAIT ||
                               singleTaskInfo.type == SkTaskType::TYPE_EVENT_RESET;
      if (isEventTask) {
        taskInfo["eventValue"] = UintToHexString(GetEventTaskValue(singleTaskInfo));
      }
      if (singleTaskInfo.type == SkTaskType::TYPE_EVENT_WAIT) {
        taskInfo["waitFlag"] = GetEventTaskFlag(singleTaskInfo);
      }

      Json entries = Json::array();
      for (uint32_t j = 0; j < singleTaskInfo.entryCnt && j < 4; ++j) {
        std::stringstream hexStream;
        hexStream << "0x" << std::hex << singleTaskInfo.entry[j];
        entries.push_back(hexStream.str());
      }
      taskInfo["entries"] = entries;

      taskInfos.push_back(taskInfo);
    }
    taskJson["taskQue"]["taskInfos"] = taskInfos;
  }

  return taskJson;
}

/**
 * @brief Add basic kernel parameters to JSON
 */
static void AddBasicKernelParams(Json &kernelParamsJson, const aclmdlRIKernelTaskParams &kernelParams) {
  char funcName[MAX_FUNC_NAME_LEN] = {0};
  if (aclrtGetFunctionName(kernelParams.funcHandle, sizeof(funcName), funcName) == ACL_SUCCESS) {
    kernelParamsJson["funcName"] = std::string(funcName);
  }
  kernelParamsJson["funcHandle"] = PtrToHexString(kernelParams.funcHandle);
  kernelParamsJson["numBlocks"] = kernelParams.numBlocks;
  kernelParamsJson["devargs"] = PtrToHexString(kernelParams.args);
  kernelParamsJson["argsSize"] = kernelParams.argsSize;
  kernelParamsJson["isHostArgs"] = kernelParams.isHostArgs;
  kernelParamsJson["launchKernelCfg"] = PtrToHexString(kernelParams.cfg);

  // Get binary handle
  aclrtBinHandle binHdl = nullptr;
  if (aclrtFunctionGetBinary(kernelParams.funcHandle, &binHdl) == ACL_SUCCESS) {
    kernelParamsJson["binHandle"] = PtrToHexString(binHdl);
  }

  int64_t kernelType = 0;
  int64_t taskRatioVal = 0;
  CHECK_ACL(aclrtGetFunctionAttribute(kernelParams.funcHandle, ACL_FUNC_ATTR_KERNEL_TYPE, &kernelType));
  CHECK_ACL(aclrtGetFunctionAttribute(kernelParams.funcHandle, ACL_FUNC_ATTR_KERNEL_RATIO, &taskRatioVal));

  const int16_t *taskRatioInt16 = reinterpret_cast<const int16_t *>(&taskRatioVal);
  uint32_t skTaskRatio[2] = {static_cast<uint32_t>(taskRatioInt16[1]), static_cast<uint32_t>(taskRatioInt16[0])};

  kernelParamsJson["kernelTypeInt"] = static_cast<int>(kernelType);
  kernelParamsJson["kernelType"] = GetKernelTypeString(static_cast<uint32_t>(kernelType), skTaskRatio);
  kernelParamsJson["taskRatio"] = Json::array({skTaskRatio[0], skTaskRatio[1]});
}

/**
 * @brief Add resolved functions info to JSON
 */
static void AddResolvedFuncsToJson(Json &kernelParamsJson, aclrtFuncHandle funcHandle, aclrtBinHandle binHdl) {
  ResolvedFunctionInfo dumpResolvedFuncs[K_MAX_SPLIT_BIN_COUNT];
  uint32_t resolvedNum = 0;
  GetResolvedFuncsForDump(funcHandle, binHdl, dumpResolvedFuncs, resolvedNum);
  kernelParamsJson["resolvedNum"] = resolvedNum;

  if (resolvedNum == 0) {
    return;
  }

  Json resolvedFuncsJson = Json::array();
  for (uint32_t i = 0; i < resolvedNum && i < K_MAX_SPLIT_BIN_COUNT; ++i) {
    Json rfJson;
    rfJson["funcAddr"][0] = UintToHexString(dumpResolvedFuncs[i].funcAddr[0]);
    rfJson["funcAddr"][1] = UintToHexString(dumpResolvedFuncs[i].funcAddr[1]);
    rfJson["funcOffset"][0] = UintToHexString(dumpResolvedFuncs[i].funcOffset[0]);
    rfJson["funcOffset"][1] = UintToHexString(dumpResolvedFuncs[i].funcOffset[1]);
    rfJson["prefetchCnt"][0] = dumpResolvedFuncs[i].prefetchCnt[0];
    rfJson["prefetchCnt"][1] = dumpResolvedFuncs[i].prefetchCnt[1];
    rfJson["symbolBind"][0] = dumpResolvedFuncs[i].symbolBind[0];
    rfJson["symbolBind"][1] = dumpResolvedFuncs[i].symbolBind[1];
    resolvedFuncsJson.push_back(rfJson);
  }
  kernelParamsJson["resolvedFuncs"] = resolvedFuncsJson;
}

/**
 * @brief Add kernel task parameters to JSON (nested format)
 */
void AddKernelTaskParams(Json &taskJson, const aclmdlRIKernelTaskParams &kernelParams,
                         const aclmdlRITaskParams &taskParams, const aclmdlRITask &task) {
  Json kernelParamsJson;
  AddBasicKernelParams(kernelParamsJson, kernelParams);

  // Common task params
  kernelParamsJson["opInfoPtr"] = PtrToHexString(taskParams.opInfoPtr);
  kernelParamsJson["opInfoSize"] = taskParams.opInfoSize;
  kernelParamsJson["taskGrp"] = PtrToHexString(taskParams.taskGrp);

  // Get resolved functions info
  aclrtBinHandle binHdl = nullptr;
  if (aclrtFunctionGetBinary(kernelParams.funcHandle, &binHdl) == ACL_SUCCESS) {
    AddResolvedFuncsToJson(kernelParamsJson, kernelParams.funcHandle, binHdl);
  }

  taskJson["kernelParams"] = kernelParamsJson;
}

/**
 * @brief Add event record task parameters to JSON (nested format)
 */
void AddEventRecordParams(Json &taskJson, const aclmdlRIEventRecordTaskParams &eventParams) {
  Json eventParamsJson;
  eventParamsJson["eventId"] = PtrToHexString(eventParams.event);
  eventParamsJson["eventFlag"] = eventParams.eventFlag;
  eventParamsJson["recordFlag"] = eventParams.recordFlag;
  taskJson["eventRecordParams"] = eventParamsJson;
}

/**
 * @brief Add event wait task parameters to JSON (nested format)
 */
void AddEventWaitParams(Json &taskJson, const aclmdlRIEventWaitTaskParams &eventParams) {
  Json eventParamsJson;
  eventParamsJson["event"] = PtrToHexString(eventParams.event);
  eventParamsJson["eventFlag"] = eventParams.eventFlag;
  eventParamsJson["waitFlag"] = eventParams.waitFlag;
  taskJson["eventWaitParams"] = eventParamsJson;
}

/**
 * @brief Add event reset task parameters to JSON (nested format)
 */
void AddEventResetParams(Json &taskJson, const aclmdlRIEventResetTaskParams &eventParams) {
  Json eventParamsJson;
  eventParamsJson["event"] = PtrToHexString(eventParams.event);
  eventParamsJson["eventFlag"] = eventParams.eventFlag;
  eventParamsJson["resetFlag"] = eventParams.resetFlag;
  taskJson["eventResetParams"] = eventParamsJson;
}

/**
 * @brief Add value write task parameters to JSON (nested format)
 */
void AddValueWriteParams(Json &taskJson, const aclmdlRIValueWriteTaskParams &valueParams) {
  Json valueParamsJson;
  valueParamsJson["devAddr"] = PtrToHexString(valueParams.devAddr);
  valueParamsJson["value"] = UintToHexString(valueParams.value);
  taskJson["valueWriteParams"] = valueParamsJson;
}

/**
 * @brief Add value wait task parameters to JSON (nested format)
 */
void AddValueWaitParams(Json &taskJson, const aclmdlRIValueWaitTaskParams &valueParams) {
  Json valueParamsJson;
  valueParamsJson["devAddr"] = PtrToHexString(valueParams.devAddr);
  valueParamsJson["value"] = UintToHexString(valueParams.value);
  valueParamsJson["flag"] = valueParams.flag;
  taskJson["valueWaitParams"] = valueParamsJson;
}

/**
 * @brief Add task parameters to JSON based on task type
 */
void AddTaskParamsToJson(Json &taskJson, const aclmdlRITaskParams &params, const aclmdlRITask &task) {
  switch (params.type) {
    case ACL_MODEL_RI_TASK_KERNEL:
      AddKernelTaskParams(taskJson, params.kernelTaskParams, params, task);
      break;
    case ACL_MODEL_RI_TASK_EVENT_RECORD:
      AddEventRecordParams(taskJson, params.eventRecordTaskParams);
      break;
    case ACL_MODEL_RI_TASK_EVENT_WAIT:
      AddEventWaitParams(taskJson, params.eventWaitTaskParams);
      break;
    case ACL_MODEL_RI_TASK_EVENT_RESET:
      AddEventResetParams(taskJson, params.eventResetTaskParams);
      break;
    case ACL_MODEL_RI_TASK_VALUE_WRITE:
      AddValueWriteParams(taskJson, params.valueWriteTaskParams);
      break;
    case ACL_MODEL_RI_TASK_VALUE_WAIT:
      AddValueWaitParams(taskJson, params.valueWaitTaskParams);
      break;
    default:
      break;
  }
}

/**
 * @brief Convert a single task to JSON object
 */
Json TaskToJson(uint32_t taskIdx, int32_t streamId, aclmdlRITaskType taskType, const aclmdlRITaskParams *params,
                const aclmdlRITask &task) {
  Json taskJson;
  // Get the real taskId (nodeId) from modelRI using aclmdlRITaskGetSeqId
  uint32_t seqId = 0;
  aclError ret = aclmdlRITaskGetSeqId(task, &seqId);
  if (ret == ACL_SUCCESS) {
    taskJson["taskId"] = static_cast<uint64_t>(seqId);
  }
  // taskIdx is the index within the stream (from enumeration)
  taskJson["streamId"] = streamId;
  taskJson["taskType"] = TaskTypeToString(taskType);
  taskJson["taskTypeInt"] = static_cast<int>(taskType);

  // Add type-specific parameters (nested in separate fields)
  if (params != nullptr) {
    AddTaskParamsToJson(taskJson, *params, task);
  }

  return taskJson;
}

/**
 * @brief Print nodeIds in batches to avoid exceeding maxLineLength
 */
void PrintNodeIdsInBatches(const std::vector<uint64_t> &nodeIds, const char *prefix = "    nodeIds: [") {
  constexpr size_t BATCH_SIZE = 100;
  constexpr size_t MAX_LINE_LEN = 4096;
  const std::string suffix = "]";

  std::string linePrefix = prefix;
  std::string currentBatch;

  for (size_t i = 0; i < nodeIds.size(); ++i) {
    std::string nodeIdStr = (currentBatch.empty() ? "" : ", ") + std::to_string(nodeIds[i]);
    size_t estimatedLen = linePrefix.length() + currentBatch.length() + nodeIdStr.length() + suffix.length();

    if (estimatedLen > MAX_LINE_LEN && !currentBatch.empty()) {
      SK_LOGI("%s%s%s", linePrefix.c_str(), currentBatch.c_str(), suffix.c_str());
      currentBatch.clear();
      linePrefix = "    continued: ";
    }
    currentBatch += nodeIdStr;
  }

  if (!currentBatch.empty()) {
    SK_LOGI("%s%s%s", linePrefix.c_str(), currentBatch.c_str(), suffix.c_str());
  }
}

/**
 * @brief Collect kernel node IDs from a node list, skipping scope nodes
 */
std::unordered_set<uint64_t> CollectKernelIdsFromNodes(const SuperKernelBaseNode *const *nodes, size_t nodeCount) {
  std::unordered_set<uint64_t> kernelIds;
  for (size_t i = 0; i < nodeCount; ++i) {
    const auto *node = nodes[i];
    if (node == nullptr || node->IsScopeBegin() || node->IsScopeEnd() || node->IsScopePlaceholder()) {
      continue;
    }
    if (node->GetNodeType() == SkNodeType::NODE_KERNEL) {
      kernelIds.insert(node->GetNodeId());
    }
  }
  return kernelIds;
}

/**
 * @brief Collect kernel node IDs from a scope
 */
std::unordered_set<uint64_t> CollectKernelIdsFromScope(const SuperKernelScopeInfo &scopeInfo) {
  const auto &nodes = scopeInfo.GetNodes();
  return CollectKernelIdsFromNodes(nodes.data(), nodes.size());
}

/**
 * @brief Build original kernel sets grouped by scopeBitFlags
 */
std::unordered_map<std::string, std::unordered_set<uint64_t>> BuildOriginalKernelSets(
    const SuperKernelGraph &graph, const std::vector<OriginalScopeInfo> &originalScopes) {
  std::unordered_map<std::string, std::unordered_set<uint64_t>> originalKernelSets;

  for (const auto &origScope : originalScopes) {
    std::unordered_set<uint64_t> kernelIds;
    for (uint64_t nodeId : origScope.nodeIds) {
      SuperKernelBaseNode *node = graph.GetNodeById(nodeId);
      if (node == nullptr || node->IsScopeBegin() || node->IsScopeEnd() || node->IsScopePlaceholder()) {
        continue;
      }
      if (node->GetNodeType() == SkNodeType::NODE_KERNEL) {
        kernelIds.insert(nodeId);
      }
    }
    originalKernelSets[graph.BitsetToString(origScope.GetScopeBitFlags())] = std::move(kernelIds);
  }
  return originalKernelSets;
}

/**
 * @brief Check if fused scope kernel set matches original scope
 */
bool IsKernelSetMatch(const SuperKernelScopeInfo &scope,
                      const std::unordered_map<std::string, std::unordered_set<uint64_t>> &originalKernelSets,
                      const SuperKernelGraph &graph) {
  const auto &scopeBitFlags = scope.GetScopeBitFlags();
  std::string bitFlagStr = graph.BitsetToString(scopeBitFlags);

  auto it = originalKernelSets.find(bitFlagStr);
  if (it == originalKernelSets.end()) {
    return false;
  }

  std::unordered_set<uint64_t> fusedKernelIds = CollectKernelIdsFromScope(scope);
  return it->second == fusedKernelIds;
}

/**
 * @brief Print fused node IDs with line length limit
 */
void PrintFusedNodeIds(const std::vector<uint64_t> &fusedNodeIds) {
  constexpr size_t MAX_LINE_LEN = 4096;
  std::string prefix = "    fusedNodeIds=[";
  std::string suffix = "];";
  std::string currentLine;

  for (size_t i = 0; i < fusedNodeIds.size(); ++i) {
    std::string nodeIdStr = (currentLine.empty() ? "" : ", ") + std::to_string(fusedNodeIds[i]);

    if (currentLine.length() + nodeIdStr.length() + suffix.length() > MAX_LINE_LEN) {
      SK_LOGI("%s%s%s", prefix.c_str(), currentLine.c_str(), suffix.c_str());
      currentLine.clear();
      prefix = "    continued: ";
    }
    currentLine += nodeIdStr;
  }

  if (!currentLine.empty()) {
    SK_LOGI("%s%s%s", prefix.c_str(), currentLine.c_str(), suffix.c_str());
  }
}

/**
 * @brief Collect fused node IDs and check if scope has kernel
 */
void CollectFusedNodes(const ScopeExtInfo &extInfo, std::vector<uint64_t> &fusedNodeIds, bool &hasKernel) {
  fusedNodeIds.clear();
  hasKernel = false;

  for (const auto *node : extInfo.filteredNodes) {
    if (node == nullptr || node->IsScopeBegin() || node->IsScopeEnd() || node->IsScopePlaceholder()) {
      continue;
    }
    fusedNodeIds.push_back(node->GetNodeId());
    if (node->GetNodeType() == SkNodeType::NODE_KERNEL) {
      hasKernel = true;
    }
  }
}

SkInfoForDumpMap BuildSkInfoForDumpMap(const std::vector<SuperKernelScopeInfo> &scopeInfos) {
  SkInfoForDumpMap skInfos;
  for (const auto &scopeInfo : scopeInfos) {
    const auto &extInfo = scopeInfo.GetExtInfo();
    if (extInfo.processStatus != ScopeProcessStatus::SUCCESS || extInfo.skMainNodeId == INVALID_TASK_ID) {
      continue;
    }

    const SuperKernelBaseNode *startKernelNode = nullptr;
    const SuperKernelBaseNode *endKernelNode = nullptr;
    for (const auto *node : extInfo.filteredNodes) {
      if (node == nullptr || node->GetNodeType() != SkNodeType::NODE_KERNEL) {
        continue;
      }
      if (startKernelNode == nullptr) {
        startKernelNode = node;
      }
      endKernelNode = node;
    }
    if (startKernelNode == nullptr || endKernelNode == nullptr) {
      continue;
    }

    SkInfoForDump skInfo;
    skInfo.scopeId = scopeInfo.GetScopeId();
    skInfo.scopeName = extInfo.scopeName;
    skInfo.startKernelFuncName = startKernelNode->GetNodeInfos().kernelInfos.funcName;
    skInfo.endKernelFuncName = endKernelNode->GetNodeInfos().kernelInfos.funcName;
    skInfos[extInfo.skMainNodeId] = skInfo;
  }
  return skInfos;
}

Json SkInfoForDumpToJson(const SkInfoForDump &skInfo) {
  Json skInfoJson;
  skInfoJson["scopeId"] = skInfo.scopeId;
  skInfoJson["scopeName"] = skInfo.scopeName.empty() ? Json() : Json(skInfo.scopeName);
  skInfoJson["startKernelFuncName"] = skInfo.startKernelFuncName;
  skInfoJson["endKernelFuncName"] = skInfo.endKernelFuncName;
  return skInfoJson;
}

void InjectSkInfos(Json &graphJson, const SkInfoForDumpMap &skInfos) {
  if (skInfos.empty() || !graphJson.contains("streams") || !graphJson["streams"].is_array()) {
    return;
  }

  for (auto &streamJson : graphJson["streams"]) {
    if (!streamJson.contains("nodes") || !streamJson["nodes"].is_array()) {
      continue;
    }

    for (auto &nodeJson : streamJson["nodes"]) {
      if (!nodeJson.contains("nodeId") || !nodeJson["nodeId"].is_number_unsigned()) {
        continue;
      }
      auto it = skInfos.find(nodeJson["nodeId"].get<uint64_t>());
      if (it != skInfos.end()) {
        nodeJson["skinfos"] = SkInfoForDumpToJson(it->second);
      }
    }
  }
}

void InjectSkInfos(Json &graphJson, const std::vector<SuperKernelScopeInfo> &scopeInfos) {
  InjectSkInfos(graphJson, BuildSkInfoForDumpMap(scopeInfos));
}

bool DumpGraphJsonToFile(Json graphJson, const SuperKernelOptionsManager &opts, const std::string &metaDir,
                         const std::string &filename, const std::vector<SuperKernelScopeInfo> *scopeInfos) {
  if (scopeInfos != nullptr) {
    InjectSkInfos(graphJson, *scopeInfos);
  }
  graphJson["options"] = opts.ToJson();

  std::string fullFilename = filename + ".json";
  std::string testJsonPath = metaDir + "/" + fullFilename;
  std::ofstream testJsonFile(testJsonPath);
  if (!testJsonFile.is_open()) {
    SK_LOGE("Failed to open %s for writing: %s", fullFilename.c_str(), testJsonPath.c_str());
    return false;
  }

  testJsonFile << graphJson.dump(2);  // Pretty print with 2-space indentation
  testJsonFile.close();
  SK_LOGI("Successfully dumped graph to %s: %s", fullFilename.c_str(), testJsonPath.c_str());
  return true;
}

}  // anonymous namespace

void PrintFusedScopes(const SuperKernelGraph &graph, const std::vector<SuperKernelScopeInfo> &processedScopeInfos,
                      bool debugPerOpMaxCoreEnabled) {
  // Build original scope kernel node sets
  const auto &originalScopes = graph.GetOriginalScopeInfos();
  auto originalKernelSets = BuildOriginalKernelSets(graph, originalScopes);

  // Print scopes after fusion
  SK_LOGI("[PrintFusedScopes] scopes after fusion: DEBUG_PER_OP_MAX_CORE=%s",
          debugPerOpMaxCoreEnabled ? "enabled" : "disabled");
  for (const auto &scopeInfo : processedScopeInfos) {
    const auto &extInfo = scopeInfo.GetExtInfo();

    std::vector<uint64_t> fusedNodeIds;
    bool hasKernel = false;
    CollectFusedNodes(extInfo, fusedNodeIds, hasKernel);

    // Skip scope without kernel
    if (!hasKernel) {
      continue;
    }

    const auto &breakInfo = scopeInfo.GetBreakInfo();
    std::string scopeNames = ScopeSplitPass::GetScopeNamesFromBitFlags(scopeInfo.GetScopeBitFlags(), graph);

    // Line 0: scopeId and scopeBitFlag
    SK_LOGI("  scopeId=%u, scopeBitFlag=%s", scopeInfo.GetScopeId(),
            graph.BitsetToString(scopeInfo.GetScopeBitFlags()).c_str());

    // Line 1: fusedNodeIds
    PrintFusedNodeIds(fusedNodeIds);

    // Line 2: scopeStatus
    if (extInfo.processStatus == ScopeProcessStatus::SUCCESS) {
      SK_LOGI("    scopeStatus=SUCCESS");
    } else {
      SK_LOGI("    scopeStatus=%s, scopeStatusDetail=%s", to_string(extInfo.processStatus),
              ScopeProcessStatusDetail(extInfo.processStatus));
    }

    // Line 3: breakReason (if kernel set differs from original scope)
    if (!debugPerOpMaxCoreEnabled && breakInfo.GetReason() != ScopeBreakReason::NONE &&
        !IsKernelSetMatch(scopeInfo, originalKernelSets, graph)) {
      SK_LOGI("    breakReason=[%s], breakReasonDetail=%s, scopeName=[%s]", breakInfo.Format().c_str(),
              ScopeBreakReasonDetail(breakInfo.GetReason()), scopeNames.c_str());
    }
  }
}

void PrintOriginalScopes(const SuperKernelGraph &graph) {
  const auto &originalScopeInfos = graph.GetOriginalScopeInfos();
  SK_LOGI("original scopes before fusion:");
  for (const auto &origScope : originalScopeInfos) {
    std::string scopeNames = ScopeSplitPass::GetScopeNamesFromBitFlags(origScope.GetScopeBitFlags(), graph);
    SK_LOGI("  scopeId=%u, scopeFlags=%s, scopeNames=[%s], totalNodes=%zu", origScope.scopeId,
            graph.BitsetToString(origScope.GetScopeBitFlags()).c_str(), scopeNames.c_str(), origScope.nodeIds.size());
    PrintNodeIdsInBatches(origScope.nodeIds);
  }
}

Json SkTaskToQueueJson(const SkTask &aicTask, const SkTask &aivTask, uint16_t scopeId) {
  Json taskQueueJson;
  taskQueueJson["scopeId"] = scopeId;
  taskQueueJson["taskQueues"]["aic"] = SkTaskToJson(aicTask);
  taskQueueJson["taskQueues"]["aiv"] = SkTaskToJson(aivTask);
  return taskQueueJson;
}

bool DumpAllTaskQueuesToJson(const SuperKernelGraph &graph,
                             const std::unordered_map<std::string, Json> &taskQueueJsons) {
  if (!sk::logger::FileLogger::Instance().IsEnabled()) {
    return true;  // Kernel meta save is disabled, skip
  }

  std::string metaDir = CreateSkMetaDirectory(graph.GetModelId());
  if (metaDir.empty()) {
    SK_LOGE("Failed to create sk_meta directory for task queue JSON dump");
    return false;
  }
  std::string jsonPath = metaDir + "/sk_task_queue.json";

  try {
    Json rootJson;
    rootJson["version"] = "1.0";
    rootJson["description"] = "SuperKernel Task Queue Information";
    rootJson["modelId"] = graph.GetModelId();
    rootJson["scopeCount"] = taskQueueJsons.size();

    Json scopesArray = Json::array();
    for (const auto &[scopeId, taskQueueJson] : taskQueueJsons) {
      scopesArray.push_back(taskQueueJson);
    }
    rootJson["scopes"] = scopesArray;

    std::ofstream outFile(jsonPath);
    if (!outFile.is_open()) {
      SK_LOGE("Failed to open task queue JSON file for writing: %s", jsonPath.c_str());
      return false;
    }

    outFile << rootJson.dump(2);
    outFile.close();

    SK_LOGI("Successfully dumped %zu task queues to JSON file: %s", taskQueueJsons.size(), jsonPath.c_str());
    return true;

  } catch (const std::exception &e) {
    SK_LOGE("Exception while dumping task queues to JSON: %s", e.what());
    return false;
  }
}

bool DumpGraphJson(aclmdlRI model, const std::string &modelId, const SuperKernelOptionsManager &opts,
                   const std::string &metaDir, const std::string &filename,
                   const std::vector<SuperKernelScopeInfo> *scopeInfos) {
  if (!sk::logger::FileLogger::Instance().IsEnabled()) {
    return true;  // Kernel meta save is disabled, skip
  }

  if (filename.empty()) {
    SK_LOGE("DumpGraphJson failed: filename is empty");
    return false;
  }

  SK_LOGI("Start creating temp graph for %s dump...", filename.c_str());
  SuperKernelGraph tempGraph(model, opts, modelId);
  if (!tempGraph.InitFromModelRI()) {
    SK_LOGE("Failed to init temp graph for %s dump", filename.c_str());
    return false;
  }
  SK_LOGI("End creating temp graph for %s dump", filename.c_str());

  return DumpGraphJsonToFile(tempGraph.ToJson(), opts, metaDir, filename, scopeInfos);
}

bool DumpGraphJson(const SuperKernelGraph &graph, const SuperKernelOptionsManager &opts, const std::string &metaDir,
                   const std::string &filename, const std::vector<SuperKernelScopeInfo> *scopeInfos) {
  if (!sk::logger::FileLogger::Instance().IsEnabled()) {
    return true;  // Kernel meta save is disabled, skip
  }

  if (filename.empty()) {
    SK_LOGE("DumpGraphJson failed: filename is empty");
    return false;
  }

  return DumpGraphJsonToFile(graph.ToJson(), opts, metaDir, filename, scopeInfos);
}
