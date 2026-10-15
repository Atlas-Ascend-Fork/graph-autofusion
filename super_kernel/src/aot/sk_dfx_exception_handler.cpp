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
 * \file sk_dfx_exception_handler.cpp
 * \brief SuperKernel DFX, handle sk aicore error
 */

#include <cstdarg>
#include <cstring>
#include <set>
#include <map>
#include <unordered_set>
#include "sk_dfx_exception_handler.h"
#include "sk_log.h"
#include "sk_types.h"
#include "sk_common.h"
#include "sk_event_recorder.h"
#include "runtime/rt_external_kernel.h"
#include "runtime/kernel.h"

// COND register index in errReg[] for different architectures
// DAV_2201: errReg[20] is low 32 bits, errReg[21] is high 32 bits
constexpr uint32_t SK_COND_ERR_REG_LOW_IDX_DAV2201 = 20;
constexpr uint32_t SK_COND_ERR_REG_HIGH_IDX_DAV2201 = 21;
// DAV_3510: errReg[32] is low 32 bits, errReg[33] is high 32 bits
constexpr uint32_t SK_COND_ERR_REG_LOW_IDX_DAV3510 = 32;
constexpr uint32_t SK_COND_ERR_REG_HIGH_IDX_DAV3510 = 33;

SuperKernelExceptionHandler::SuperKernelExceptionHandler()
    : aicoreNums(GetSkRuntimeConfig().eventCoreNum),
      skDeviceEntryArgsDev(nullptr),
      skDeviceEntryArgsPtrLen(0),
      skDeviceEntryArgsHost(nullptr),
      skHeaderInfoHost(nullptr),
      aicTaskQueDevPtr(nullptr),
      aivTaskQueDevPtr(nullptr),
      aicTaskCnt(0),
      aivTaskCnt(0),
      hasOpTrace_(false),
      errorNodeIdx_(-1) {}

void SuperKernelExceptionHandler::HandleException(aclrtExceptionInfo *exceptionInfo) {
  if (exceptionInfo == nullptr) {
    SK_DLOGE("Exception info is null");
    return;
  }

  // Check if the exception is from sk_entry operator
  if (!IsSuperKernelException(exceptionInfo)) {
    SK_DLOGD("Exception is not from sk_entry, skip handling.");
    return;
  }

  SK_DLOGD("Aclgraph superkernel aicore exception occurred, callback function.");

  std::unique_ptr<sk::logger::LogContextGuard> logContext;
  if (!ExtractSkEntryArgs(exceptionInfo, logContext)) {
    FreeResources();
    return;
  }

  if (!ExtractTaskQueue()) {
    FreeResources();
    return;
  }

  // Print info log
  ExtractAndPrintSkInfo();

  // Print error log
  if (!ParseAndPrintSubKernelSymbols(exceptionInfo)) {
    FreeResources();
    return;
  }

  // Print modelId index mapping table for debugging
  SkEventRecorder::Instance().PrintModelIdIndexMap();

  // Print op trace for all cores
  PrintAllCoreSymbols();

  FreeResources();
}

bool SuperKernelExceptionHandler::ExtractSkEntryArgs(aclrtExceptionInfo *exceptionInfo,
                                                     std::unique_ptr<sk::logger::LogContextGuard> &logContext) {
  if (!ExtractSkDeviceEntryArgsPtr(exceptionInfo)) {
    return false;
  }

  if (!CopySkDeviceEntryArgsToHost(logContext)) {
    return false;
  }

  return true;
}

bool SuperKernelExceptionHandler::CopySkDeviceEntryArgsToHost(
    std::unique_ptr<sk::logger::LogContextGuard> &logContext) {
  // Step 1: First copy SkHeaderInfo to get totalSize
  if (!ExtractSkHeaderInfo()) {
    return false;
  }

  uint16_t modelIdIdx = static_cast<uint16_t>((skHeaderInfoHost->modelIdIndexAndSkScopeId >> 32) & 0xFFFF);
  std::string modelId = SkEventRecorder::Instance().GetModelIdByIndex(modelIdIdx);
  logContext = std::make_unique<sk::logger::LogContextGuard>(modelId.empty() ? UNKNOWN_MODEL_ID : modelId);

  // Step 2: Now that we know totalSize, copy all SkDeviceEntryArgs data to host at once
  SK_LOGI("---Total SkDeviceEntryArgs size: %lu bytes", skHeaderInfoHost->totalSize);
  if (skHeaderInfoHost->totalSize > skDeviceEntryArgsPtrLen) {
    SK_LOGE("totalSize(%lu) exceeds skDeviceEntryArgsPtrLen(%u), invalid device address range",
            skHeaderInfoHost->totalSize, skDeviceEntryArgsPtrLen);
    aclrtFreeHost(skHeaderInfoHost);
    skHeaderInfoHost = nullptr;
    return false;
  }

  if (CheckError(aclrtMallocHost((void **)(&skDeviceEntryArgsHost), skHeaderInfoHost->totalSize),
                 "aclrtMallocHost for skDeviceEntryArgsHost") != ACL_SUCCESS) {
    aclrtFreeHost(skHeaderInfoHost);
    skHeaderInfoHost = nullptr;
    return false;
  }

  if (CheckError(aclrtMemcpy(skDeviceEntryArgsHost, skHeaderInfoHost->totalSize, skDeviceEntryArgsDev,
                             skHeaderInfoHost->totalSize, ACL_MEMCPY_DEVICE_TO_HOST),
                 "aclrtMemcpy for skDeviceEntryArgs") != ACL_SUCCESS) {
    aclrtFreeHost(skHeaderInfoHost);
    skHeaderInfoHost = nullptr;
    return false;
  }

  // Free temporary SkHeaderInfo, as data is now in skDeviceEntryArgsHost
  if (skHeaderInfoHost != nullptr) {
    aclrtFreeHost(skHeaderInfoHost);
    skHeaderInfoHost = nullptr;
  }

  // Update skHeaderInfoHost to point to skHeaderInfo in skDeviceEntryArgsHost
  skHeaderInfoHost = &(skDeviceEntryArgsHost->skHeader);

  // Validate all offsets in SkHeaderInfo to prevent out-of-bounds access
  if (!ValidateSkHeaderOffsets()) {
    aclrtFreeHost(skDeviceEntryArgsHost);
    skDeviceEntryArgsHost = nullptr;
    skHeaderInfoHost = nullptr;
    return false;
  }

  return true;
}

bool SuperKernelExceptionHandler::ExtractSkDeviceEntryArgsPtr(aclrtExceptionInfo *exceptionInfo) {
  auto ret = aclrtGetArgsFromExceptionInfo(exceptionInfo, &skDeviceEntryArgsDev, &skDeviceEntryArgsPtrLen);
  SK_DLOGI("---skDeviceEntryArgsDev: %p", skDeviceEntryArgsDev);
  SK_DLOGI("---skDeviceEntryArgsPtrLen: %d", skDeviceEntryArgsPtrLen);

  if (ret != ACL_SUCCESS) {
    SK_DLOGE("aclrtGetArgsFromExceptionInfo failed, ret=%d", ret);
    return false;
  }

  if (skDeviceEntryArgsPtrLen < 8 || skDeviceEntryArgsDev == nullptr) {
    SK_DLOGI("no args, callback return");
    return false;
  }
  return true;
}

bool SuperKernelExceptionHandler::ExtractSkHeaderInfo() {
  // SkHeaderInfo is at the beginning of SkDeviceEntryArgs, allocate temporary memory to copy
  SkHeaderInfo *tempHeader = nullptr;
  aclError ret = aclrtMallocHost((void **)(&tempHeader), sizeof(SkHeaderInfo));
  if (ret != ACL_SUCCESS) {
    SK_DLOGE("aclrtMallocHost for temp SkHeaderInfo failed, ret=%d", ret);
    return false;
  }

  ret = aclrtMemcpy(tempHeader, sizeof(SkHeaderInfo), skDeviceEntryArgsDev, sizeof(SkHeaderInfo),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != ACL_SUCCESS) {
    SK_DLOGE("aclrtMemcpy for SkHeaderInfo failed, ret=%d", ret);
    aclrtFreeHost(tempHeader);
    return false;
  }

  // Temporarily save SkHeaderInfo data
  skHeaderInfoHost = tempHeader;

  return true;
}

bool SuperKernelExceptionHandler::ValidateSkHeaderOffsets() {
  uint64_t totalSize = skHeaderInfoHost->totalSize;

  // Validate aicQueOffset: offset + aicQueSize must not exceed totalSize
  if (skHeaderInfoHost->aicQueOffset > 0) {
    if (skHeaderInfoHost->aicQueOffset >= totalSize) {
      SK_LOGE("aicQueOffset(%u) >= totalSize(%lu), invalid offset", skHeaderInfoHost->aicQueOffset, totalSize);
      return false;
    }
    const uint64_t aicEnd = static_cast<uint64_t>(skHeaderInfoHost->aicQueOffset) + skHeaderInfoHost->aicQueSize;
    if (aicEnd > totalSize) {
      SK_LOGE("aicQueOffset(%u) + aicQueSize(%u) exceeds totalSize(%lu), invalid offset",
              skHeaderInfoHost->aicQueOffset, skHeaderInfoHost->aicQueSize, totalSize);
      return false;
    }
  }

  // Validate aivQueOffset: offset + aivQueSize must not exceed totalSize
  if (skHeaderInfoHost->aivQueOffset > 0) {
    if (skHeaderInfoHost->aivQueOffset >= totalSize) {
      SK_LOGE("aivQueOffset(%u) >= totalSize(%lu), invalid offset", skHeaderInfoHost->aivQueOffset, totalSize);
      return false;
    }
    const uint64_t aivEnd = static_cast<uint64_t>(skHeaderInfoHost->aivQueOffset) + skHeaderInfoHost->aivQueSize;
    if (aivEnd > totalSize) {
      SK_LOGE("aivQueOffset(%u) + aivQueSize(%u) exceeds totalSize(%lu), invalid offset",
              skHeaderInfoHost->aivQueOffset, skHeaderInfoHost->aivQueSize, totalSize);
      return false;
    }
  }

  // Validate counterOffset: offset + aicoreNums * sizeof(SkCounterInfo) must not exceed totalSize
  if (skHeaderInfoHost->counterOffset > 0) {
    if (skHeaderInfoHost->counterOffset >= totalSize) {
      SK_LOGE("counterOffset(%u) >= totalSize(%lu), invalid offset", skHeaderInfoHost->counterOffset, totalSize);
      return false;
    }
    uint64_t counterEnd = static_cast<uint64_t>(skHeaderInfoHost->counterOffset) +
                          static_cast<uint64_t>(aicoreNums) * sizeof(SkCounterInfo);
    if (counterEnd > totalSize) {
      SK_LOGE("counterOffset(%u) + aicoreNums(%u) * sizeof(SkCounterInfo) exceeds totalSize(%lu), invalid offset",
              skHeaderInfoHost->counterOffset, aicoreNums, totalSize);
      return false;
    }
  }

  // Validate dfxOffset: offset + nodeCnt * sizeof(SkDfxInfo) must not exceed totalSize
  if (skHeaderInfoHost->dfxOffset > 0) {
    if (skHeaderInfoHost->dfxOffset >= totalSize) {
      SK_LOGE("dfxOffset(%u) >= totalSize(%lu), invalid offset", skHeaderInfoHost->dfxOffset, totalSize);
      return false;
    }
    uint64_t dfxEnd = static_cast<uint64_t>(skHeaderInfoHost->dfxOffset) +
                      static_cast<uint64_t>(skHeaderInfoHost->nodeCnt) * sizeof(SkDfxInfo);
    if (dfxEnd > totalSize) {
      SK_LOGE("dfxOffset(%u) + nodeCnt(%u) * sizeof(SkDfxInfo) exceeds totalSize(%lu), invalid offset",
              skHeaderInfoHost->dfxOffset, skHeaderInfoHost->nodeCnt, totalSize);
      return false;
    }
  }

  return true;
}

bool SuperKernelExceptionHandler::ExtractTaskQueue() {
  // Data is already in skDeviceEntryArgsHost, accessed via offset
  // Extract task counts
  uint8_t *dataBase = reinterpret_cast<uint8_t *>(skDeviceEntryArgsHost);

  if (skHeaderInfoHost->aicQueOffset > 0) {
    TaskQue *aicTaskQue = reinterpret_cast<TaskQue *>(dataBase + skHeaderInfoHost->aicQueOffset);
    aicTaskCnt = aicTaskQue->taskCnt;
    for (uint32_t i = 0; i < aicTaskCnt; ++i) {
      if (aicTaskQue->taskInfos[i].type == SkTaskType::TYPE_FUNC) {
        funcNodeIndices_.insert(aicTaskQue->taskInfos[i].index);
      }
    }
  }

  if (skHeaderInfoHost->aivQueOffset > 0) {
    TaskQue *aivTaskQue = reinterpret_cast<TaskQue *>(dataBase + skHeaderInfoHost->aivQueOffset);
    aivTaskCnt = aivTaskQue->taskCnt;
    for (uint32_t i = 0; i < aivTaskCnt; ++i) {
      if (aivTaskQue->taskInfos[i].type == SkTaskType::TYPE_FUNC) {
        funcNodeIndices_.insert(aivTaskQue->taskInfos[i].index);
      }
    }
  }

  return true;
}

void SuperKernelExceptionHandler::PrintSkHeaderInfo() const {
  SK_LOGI("=== SkHeaderInfo ===");
  SK_LOGI("aicQueOffset: %u", skHeaderInfoHost->aicQueOffset);
  SK_LOGI("aivQueOffset: %u", skHeaderInfoHost->aivQueOffset);
  SK_LOGI("counterOffset: %u", skHeaderInfoHost->counterOffset);
  SK_LOGI("dfxOffset: %u", skHeaderInfoHost->dfxOffset);
  SK_LOGI("nodeCnt: %u", skHeaderInfoHost->nodeCnt);
  SK_LOGI("totalSize: %lu", skHeaderInfoHost->totalSize);
  uint16_t modelIdIdx = static_cast<uint16_t>((skHeaderInfoHost->modelIdIndexAndSkScopeId >> 32) & 0xFFFF);
  uint16_t skScopeId = static_cast<uint16_t>((skHeaderInfoHost->modelIdIndexAndSkScopeId >> 16) & 0xFFFF);
  std::string modelId = SkEventRecorder::Instance().GetModelIdByIndex(modelIdIdx);
  SK_LOGI("modelIdIndexAndSkScopeId: 0x%lx (modelIdIdx=%u, skScopeId=%u, modelId=%s)",
          skHeaderInfoHost->modelIdIndexAndSkScopeId, modelIdIdx, skScopeId, modelId.c_str());
}

void SuperKernelExceptionHandler::ExtractAndPrintSkInfo() {
  PrintSkHeaderInfo();
  PrintTaskQueue();
  PrintCounterInfo();
  PrintDfxInfo();
}

void SuperKernelExceptionHandler::PrintTaskQueue() const {
  uint8_t *dataBase = reinterpret_cast<uint8_t *>(skDeviceEntryArgsHost);

  if (skHeaderInfoHost->aicQueOffset > 0) {
    TaskQue *aicTaskQue = reinterpret_cast<TaskQue *>(dataBase + skHeaderInfoHost->aicQueOffset);
    SK_LOGI("=== AIC TaskQue (offset=%u) ===", skHeaderInfoHost->aicQueOffset);
    SK_LOGI("taskCnt: %u", aicTaskCnt);
    for (uint32_t i = 0; i < aicTaskCnt; ++i) {
      const TaskInfo &task = aicTaskQue->taskInfos[i];
      SK_LOGI("  [%u] index=%u, type=%s, numBlocks=%u, args=0x%lx", i, task.index, to_string(task.type), task.numBlocks,
              task.args);
    }
  }

  if (skHeaderInfoHost->aivQueOffset > 0) {
    TaskQue *aivTaskQue = reinterpret_cast<TaskQue *>(dataBase + skHeaderInfoHost->aivQueOffset);
    SK_LOGI("=== AIV TaskQue (offset=%u) ===", skHeaderInfoHost->aivQueOffset);
    SK_LOGI("taskCnt: %u", aivTaskCnt);
    for (uint32_t i = 0; i < aivTaskCnt; ++i) {
      const TaskInfo &task = aivTaskQue->taskInfos[i];
      SK_LOGI("  [%u] index=%u, type=%s, numBlocks=%u, args=0x%lx", i, task.index, to_string(task.type), task.numBlocks,
              task.args);
    }
  }
}

void SuperKernelExceptionHandler::PrintCounterInfo() const {
  uint8_t *dataBase = reinterpret_cast<uint8_t *>(skDeviceEntryArgsHost);

  if (skHeaderInfoHost->counterOffset > 0) {
    SkCounterInfo *counterInfo = reinterpret_cast<SkCounterInfo *>(dataBase + skHeaderInfoHost->counterOffset);
    SK_LOGI("=== SkCounterInfo (offset=%u) ===", skHeaderInfoHost->counterOffset);
    for (uint32_t i = 0; i < aicoreNums; ++i) {
      SK_LOGI("  [core %u] index=%u, opState=%u", i, counterInfo[i].index, counterInfo[i].opState);
    }
  }
}

void SuperKernelExceptionHandler::PrintDfxInfo() const {
  uint8_t *dataBase = reinterpret_cast<uint8_t *>(skDeviceEntryArgsHost);

  if (skHeaderInfoHost->dfxOffset > 0) {
    SkDfxInfo *dfxInfo = reinterpret_cast<SkDfxInfo *>(dataBase + skHeaderInfoHost->dfxOffset);
    SK_LOGE("=== SkDfxInfo (offset=%u, nodeCnt=%u) ===", skHeaderInfoHost->dfxOffset, skHeaderInfoHost->nodeCnt);
    for (uint32_t i = 0; i < skHeaderInfoHost->nodeCnt; ++i) {
      SK_LOGE(
          "  [node %u] binHdl=0x%lx, funcHdlOri=0x%lx, aicSize=0x%x, aivSize=0x%x, numBlocks=%u, cubeNum=%u, vecNum=%u",
          i, dfxInfo[i].binHdl, dfxInfo[i].funcHdlOri, dfxInfo[i].aicSize, dfxInfo[i].aivSize, dfxInfo[i].numBlocks,
          dfxInfo[i].cubeNum, dfxInfo[i].vecNum);
      if (funcNodeIndices_.count(i) > 0) {
        aclrtFuncHandle funcHdl = reinterpret_cast<aclrtFuncHandle>(dfxInfo[i].funcHdlOri);
        char funcName[MAX_FUNC_NAME_LEN] = {0};
        aclError ret = aclrtGetFunctionName(funcHdl, sizeof(funcName), funcName);
        if (ret == ACL_SUCCESS) {
          SK_LOGE("    Origin function name: %s", funcName);
        } else {
          SK_LOGE("    Failed to get origin function name for node[%u], ret=%d", i, ret);
        }
      } else {
        SK_LOGE("    Node[%u] is not a FUNC type", i);
      }
      for (int j = 0; j < 4; ++j) {
        if (dfxInfo[i].entryAic[j] != 0) {
          SK_LOGE("    entryAic[%d]=0x%lx", j, dfxInfo[i].entryAic[j]);
        }
        if (dfxInfo[i].entryAiv[j] != 0) {
          SK_LOGE("    entryAiv[%d]=0x%lx", j, dfxInfo[i].entryAiv[j]);
        }
      }
    }
  }
}

bool SuperKernelExceptionHandler::GetExceptionRegInfo(const aclrtExceptionInfo &exception,
                                                      ExceptionRegInfo &exceptionRegInfo) {
  // errRegInfo memory allocated by rtGetExceptionRegInfo is managed by RTS, should not be manually freed
  rtError_t rtRet = rtGetExceptionRegInfo(&exception, &exceptionRegInfo.errRegInfo, &exceptionRegInfo.coreNum);
  if (rtRet != 0) {
    SK_LOGE("Call rtGetExceptionRegInfo for error register information failed. ret: %d", static_cast<int32_t>(rtRet));
    return false;
  }
  SK_LOGI("Get error register information. coreNum=%u", exceptionRegInfo.coreNum);
  return true;
}

uint64_t SuperKernelExceptionHandler::GetCondRegValue(const rtExceptionErrRegInfo_t &coreErrRegInfo) {
  // COND register is 64-bit
  // For DAV_2201: errReg[20] is low 32 bits, errReg[21] is high 32 bits
  // For DAV_3510: errReg[32] is low 32 bits, errReg[33] is high 32 bits
  uint32_t condRegLowIdx = SK_COND_ERR_REG_LOW_IDX_DAV2201;
  uint32_t condRegHighIdx = SK_COND_ERR_REG_HIGH_IDX_DAV2201;
  SkKernelArch arch = GetCurrentSkKernelArch();
  if (arch == SkKernelArch::DAV_3510) {
    condRegLowIdx = SK_COND_ERR_REG_LOW_IDX_DAV3510;
    condRegHighIdx = SK_COND_ERR_REG_HIGH_IDX_DAV3510;
  }
  SK_LOGE("GetCondRegValue: arch=%s, condRegLowIdx=%u, condRegHighIdx=%u", to_string(arch), condRegLowIdx,
          condRegHighIdx);
  uint64_t condValue = (static_cast<uint64_t>(coreErrRegInfo.errReg[condRegHighIdx]) << 32) |
                       static_cast<uint64_t>(coreErrRegInfo.errReg[condRegLowIdx]);
  return condValue;
}

void SuperKernelExceptionHandler::ParseAndPrintCondInfo(uint32_t coreId, rtCoreType_t coreType, uint64_t condValue) {
  // Cond value is set by sk_entry kernel via set_cond instruction.
  // Format: cond = modelIdIndexAndSkScopeId | (opState + (task->index << 8))
  //   bits [7:0]   : opState (SkOpTraceType)
  //   bits [15:8]  : task->index (sub-operator index)
  //   bits [63:16] : modelIdIndexAndSkScopeId
  // If cond is 0, the runtime/driver does not support this feature yet, skip printing.
  if (condValue == 0) {
    SK_LOGE("[Core %u] Failed to get COND register value, possible reason: driver package not upgraded", coreId);
    return;
  }

  uint8_t opState = static_cast<uint8_t>(condValue & 0xFF);
  uint32_t opIndex = static_cast<uint32_t>((condValue >> 8) & 0xFF);
  const char *coreTypeName = (coreType == RT_CORE_TYPE_AIC) ? "AIC" : "AIV";

  SK_LOGE("[Core %u] %s COND register value: 0x%lx (opState=%u, opIndex=%u)", coreId, coreTypeName, condValue, opState,
          opIndex);

  PrintCondSubKernelInfo(coreId, condValue);
}

void SuperKernelExceptionHandler::PrintCondSubKernelInfo(uint32_t coreId, uint64_t condValue) {
  uint8_t opState = static_cast<uint8_t>(condValue & 0xFF);
  uint32_t opIndex = static_cast<uint32_t>((condValue >> 8) & 0xFF);

  if (opState == static_cast<uint8_t>(SkOpTraceType::ORIGIN)) {
    SK_LOGE("[Core %u] COND: No SK entry executed yet (opState=ORIGIN).", coreId);
  } else if (opState == static_cast<uint8_t>(SkOpTraceType::SK_ENTRY_LAUNCHED)) {
    SK_LOGE("[Core %u] COND: SK entry launched, no sub-kernel executed yet. Next opIndex=%u", coreId, opIndex);
    if (opIndex < skHeaderInfoHost->nodeCnt) {
      KernelFuncName kernelFuncName = GetOrLoadKernelSymbols(opIndex);
      if (!kernelFuncName.name.empty()) {
        SK_LOGE("[Core %u] COND: Next sub-kernel function name: %s", coreId, kernelFuncName.name.c_str());
      }
    }
  } else if (opState == static_cast<uint8_t>(SkOpTraceType::OP_LAUNCHED)) {
    SK_LOGE("[Core %u] COND: Currently running sub-kernel opIndex=%u", coreId, opIndex);
    if (opIndex < skHeaderInfoHost->nodeCnt) {
      KernelFuncName kernelFuncName = GetOrLoadKernelSymbols(opIndex);
      if (!kernelFuncName.name.empty()) {
        SK_LOGE("[Core %u] COND: Current sub-kernel function name: %s", coreId, kernelFuncName.name.c_str());
      }
    }
  } else if (opState == static_cast<uint8_t>(SkOpTraceType::OP_FINISHED)) {
    SK_LOGE("[Core %u] COND: Sub-kernel opIndex=%u finished, about to run next.", coreId, opIndex);
    if (opIndex < skHeaderInfoHost->nodeCnt) {
      KernelFuncName currentKernelFuncName = GetOrLoadKernelSymbols(opIndex);
      if (!currentKernelFuncName.name.empty()) {
        SK_LOGE("[Core %u] COND: Last finished sub-kernel function name: %s", coreId,
                currentKernelFuncName.name.c_str());
      }
    }
    if (opIndex + 1 < skHeaderInfoHost->nodeCnt) {
      KernelFuncName nextKernelFuncName = GetOrLoadKernelSymbols(opIndex + 1);
      if (!nextKernelFuncName.name.empty()) {
        SK_LOGE("[Core %u] COND: Next sub-kernel function name: %s", coreId, nextKernelFuncName.name.c_str());
      }
    }
  } else if (opState == static_cast<uint8_t>(SkOpTraceType::SK_ENTRY_FINISHED)) {
    SK_LOGE("[Core %u] COND: SK entry execution completed.", coreId);
  } else {
    SK_LOGE("[Core %u] COND: Unknown opState=%u, opIndex=%u", coreId, opState, opIndex);
  }
}

void SuperKernelExceptionHandler::PrintSymbolByCoreId(uint32_t coreId, rtCoreType_t coreType, uint64_t startPC,
                                                      uint64_t currentPC, const KernelFuncName &kernelFuncName) {
  const char *coreTypeName = (coreType == RT_CORE_TYPE_AIC) ? "AIC" : "AIV";

  SK_LOGI("========== coreId=%u, coreType=%s, startPC=0x%lx, currentPC=0x%lx ==========", coreId, coreTypeName, startPC,
          currentPC);

  // Print function name
  if (!kernelFuncName.name.empty()) {
    SK_LOGE("    Function name: %s", kernelFuncName.name.c_str());
  } else {
    SK_LOGE("    No function name available");
  }
}

KernelFuncName SuperKernelExceptionHandler::GetOrLoadKernelSymbols(uint32_t opId) {
  // Check cache
  auto it = opSymbolCache.find(opId);
  if (it != opSymbolCache.end()) {
    SK_LOGI("Using cached function name for opId=%u", opId);
    return it->second;
  }

  // Cache miss, get function name from funcHdl
  if (opId >= skHeaderInfoHost->nodeCnt) {
    SK_LOGE("opId=%u exceeds nodeCnt=%u", opId, skHeaderInfoHost->nodeCnt);
    return KernelFuncName{""};
  }

  // Only TYPE_FUNC nodes have valid function handles
  if (funcNodeIndices_.count(opId) == 0) {
    SK_LOGI("opId=%u is not a FUNC type node", opId);
    return KernelFuncName{""};
  }

  // Access dfxInfo via offset from skDeviceEntryArgsHost
  uint8_t *dataBase = reinterpret_cast<uint8_t *>(skDeviceEntryArgsHost);
  SkDfxInfo *dfxInfo = reinterpret_cast<SkDfxInfo *>(dataBase + skHeaderInfoHost->dfxOffset);

  aclrtFuncHandle funcHdl = reinterpret_cast<aclrtFuncHandle>(dfxInfo[opId].funcHdlOri);
  SK_LOGI("Loading function name for opId=%u, funcHdl=%p", opId, funcHdl);

  char funcName[MAX_FUNC_NAME_LEN] = {0};
  constexpr uint32_t maxLen = sizeof(funcName);
  aclError ret = aclrtGetFunctionName(funcHdl, maxLen, funcName);
  if (ret != ACL_SUCCESS) {
    SK_LOGE("aclrtGetFunctionName failed for opId=%u, funcHdl=%p, ret=%d", opId, funcHdl, ret);
    return KernelFuncName{""};
  }

  SK_LOGI("Got function name for opId=%u: %s", opId, funcName);

  // Construct simple KernelFuncName, only containing function name
  KernelFuncName kernelFuncName{std::string(funcName)};

  // Store in cache
  opSymbolCache[opId] = kernelFuncName;
  return kernelFuncName;
}

void SuperKernelExceptionHandler::PrintMatchedNodeBasicInfo(uint32_t coreId, rtCoreType_t coreType, uint64_t startPC,
                                                            uint64_t currentPC, uint32_t nodeIdx, int entryIdx,
                                                            uint64_t entryAddr, uint64_t endAddr, uint32_t funcSize,
                                                            const SkDfxInfo &dfxNode) const {
  const char *coreTypeName = (coreType == RT_CORE_TYPE_AIC) ? "AIC" : "AIV";
  uint16_t modelIdIdx = static_cast<uint16_t>((skHeaderInfoHost->modelIdIndexAndSkScopeId >> 32) & 0xFFFF);
  uint16_t skScopeId = static_cast<uint16_t>((skHeaderInfoHost->modelIdIndexAndSkScopeId >> 16) & 0xFFFF);
  std::string modelId = SkEventRecorder::Instance().GetModelIdByIndex(modelIdIdx);

  SK_LOGE("============================================================");
  SK_LOGE("[Core %u] ModelIdIdx=%u, ModelId=%s, SkScopeId=%u", coreId, modelIdIdx, modelId.c_str(), skScopeId);
  SK_LOGE("[Core %u] CoreType: %s", coreId, coreTypeName);
  SK_LOGE("[Core %u] StartPC: 0x%lx", coreId, startPC);
  SK_LOGE("[Core %u] CurrentPC: 0x%lx", coreId, currentPC);
  SK_LOGE("[Core %u] Found in node[%u], entry[%d]", coreId, nodeIdx, entryIdx);
  SK_LOGE("[Core %u] Entry address: 0x%lx", coreId, entryAddr);
  SK_LOGE("[Core %u] End address: 0x%lx", coreId, endAddr);
  SK_LOGE("[Core %u] Function size: 0x%x (%u bytes)", coreId, funcSize, funcSize);
  SK_LOGE("[Core %u] numBlocks=%u, cubeNum=%u, vecNum=%u", coreId, dfxNode.numBlocks, dfxNode.cubeNum, dfxNode.vecNum);
}

void SuperKernelExceptionHandler::PrintFuncSymbolInfo(uint32_t coreId, rtCoreType_t coreType, uint32_t nodeIdx,
                                                      int entryIdx, const uint64_t *entries, const SkDfxInfo &dfxNode) {
  if (funcNodeIndices_.count(nodeIdx) == 0) {
    SK_LOGE("[Core %u] Node[%u] is not a FUNC type", coreId, nodeIdx);
    return;
  }

  // Print origin function name from funcHdlOri
  aclrtFuncHandle funcHdl = reinterpret_cast<aclrtFuncHandle>(dfxNode.funcHdlOri);
  char funcName[MAX_FUNC_NAME_LEN] = {0};
  aclError ret = aclrtGetFunctionName(funcHdl, sizeof(funcName), funcName);
  if (ret == ACL_SUCCESS) {
    SK_LOGE("[Core %u] Origin function name: %s", coreId, funcName);
  } else {
    SK_LOGE("Failed to get origin function name for node[%u], ret=%d", nodeIdx, ret);
  }

  // Resolve actual SK symbol name from bin using funcOffset
  aclrtBinHandle binHdl = reinterpret_cast<aclrtBinHandle>(dfxNode.binHdl);
  void *binHostAddr = nullptr;
  uint32_t binHostSize = 0;
  int rtRet = rtGetBinBuffer(binHdl, RT_BIN_HOST_ADDR, &binHostAddr, &binHostSize);
  if (rtRet != 0 || binHostAddr == nullptr) {
    SK_LOGE("[Core %u] Failed to get bin buffer for node[%u], rtRet=%d", coreId, nodeIdx, rtRet);
    return;
  }

  uint64_t funcOffset =
      (coreType == RT_CORE_TYPE_AIC) ? dfxNode.aicFuncOffset[entryIdx] : dfxNode.aivFuncOffset[entryIdx];
  // Fallback: compute offset from entry address using AIC entry[0] as reference
  if (funcOffset == 0 && dfxNode.aicFuncOffset[0] != 0 && dfxNode.entryAic[0] != 0) {
    uint64_t binDevAddr = dfxNode.entryAic[0] - dfxNode.aicFuncOffset[0];
    funcOffset = entries[entryIdx] - binDevAddr;
    SK_LOGI("[Core %u] Computed funcOffset=0x%lx for node[%u] entry[%d] (binDevAddr=0x%lx)", coreId, funcOffset,
            nodeIdx, entryIdx, binDevAddr);
  }
  if (funcOffset == 0) {
    SK_LOGE("[Core %u] funcOffset is 0 for node[%u] entry[%d], cannot resolve SK symbol name", coreId, nodeIdx,
            entryIdx);
    return;
  }

  std::string symbolName;
  uint64_t symbolSize = 0;
  std::string symbolBind;
  if (GetFuncSymbolInfo(binHdl, static_cast<const char *>(binHostAddr), binHostSize, funcOffset, symbolName, symbolSize,
                        symbolBind)) {
    SK_LOGE("[Core %u] Bound SK symbol name: %s (bind=%s, size=0x%lx)", coreId, symbolName.c_str(), symbolBind.c_str(),
            symbolSize);
  } else {
    SK_LOGE("[Core %u] Failed to get SK symbol name for node[%u] entry[%d], offset=0x%lx", coreId, nodeIdx, entryIdx,
            funcOffset);
  }
}

void SuperKernelExceptionHandler::PrintNodeDevArgs(uint32_t coreId, rtCoreType_t coreType, uint32_t nodeIdx) const {
  uint8_t *dataBase = reinterpret_cast<uint8_t *>(skDeviceEntryArgsHost);
  TaskQue *taskQue = (coreType == RT_CORE_TYPE_AIC)
                         ? reinterpret_cast<TaskQue *>(dataBase + skHeaderInfoHost->aicQueOffset)
                         : reinterpret_cast<TaskQue *>(dataBase + skHeaderInfoHost->aivQueOffset);
  uint32_t taskCnt = (coreType == RT_CORE_TYPE_AIC) ? aicTaskCnt : aivTaskCnt;
  for (uint32_t t = 0; t < taskCnt; ++t) {
    if (taskQue->taskInfos[t].index == nodeIdx && taskQue->taskInfos[t].type == SkTaskType::TYPE_FUNC) {
      SK_LOGE("[Core %u] node[%u] devArgs: 0x%lx", coreId, nodeIdx, taskQue->taskInfos[t].args);
      break;
    }
  }
}

void SuperKernelExceptionHandler::PrintNoMatchInfo(uint32_t coreId, rtCoreType_t coreType, uint64_t startPC,
                                                   uint64_t currentPC) const {
  const char *coreTypeName = (coreType == RT_CORE_TYPE_AIC) ? "AIC" : "AIV";
  uint16_t modelIdIdx = static_cast<uint16_t>((skHeaderInfoHost->modelIdIndexAndSkScopeId >> 32) & 0xFFFF);
  uint16_t skScopeId = static_cast<uint16_t>((skHeaderInfoHost->modelIdIndexAndSkScopeId >> 16) & 0xFFFF);
  std::string modelId = SkEventRecorder::Instance().GetModelIdByIndex(modelIdIdx);

  SK_LOGE("============================================================");
  SK_LOGE("[Core %u] No sub kernel matched, aicore error occurred in sk entry.", coreId);
  SK_LOGE("[Core %u] ModelIdIdx=%u, ModelId=%s, SkScopeId=%u", coreId, modelIdIdx, modelId.c_str(), skScopeId);
  SK_LOGE("[Core %u] CoreType: %s", coreId, coreTypeName);
  SK_LOGE("[Core %u] startPC: 0x%lx", coreId, startPC);
  SK_LOGE("[Core %u] CurrentPC: 0x%lx", coreId, currentPC);
  SK_LOGE("============================================================");
}

void SuperKernelExceptionHandler::IdentifyErrorNodeByPC(uint32_t coreId, rtCoreType_t coreType, uint64_t startPC,
                                                        uint64_t currentPC) {
  errorNodeIdx_ = -1;
  ErrorNodeMatchInfo matchInfo;
  if (FindErrorNodeByPC(coreType, currentPC, matchInfo)) {
    PrintMatchedNodeBasicInfo(coreId, coreType, startPC, currentPC, matchInfo.nodeIdx, matchInfo.entryIdx,
                              matchInfo.entryAddr, matchInfo.endAddr, matchInfo.funcSize, *matchInfo.dfxNode);
    PrintFuncSymbolInfo(coreId, coreType, matchInfo.nodeIdx, matchInfo.entryIdx, matchInfo.entries, *matchInfo.dfxNode);
    PrintNodeDevArgs(coreId, coreType, matchInfo.nodeIdx);
    SK_LOGE("============================================================");
    errorNodeIdx_ = matchInfo.nodeIdx;
    SK_LOGE("errorNodeIdx_ = %d", errorNodeIdx_);
    return;  // Found the error node
  }

  PrintNoMatchInfo(coreId, coreType, startPC, currentPC);
  SK_LOGE("errorNodeIdx_ = -1");
}

bool SuperKernelExceptionHandler::FindErrorNodeByPC(rtCoreType_t coreType, uint64_t currentPC,
                                                    ErrorNodeMatchInfo &matchInfo) const {
  if (skHeaderInfoHost->dfxOffset == 0 || skHeaderInfoHost->nodeCnt == 0 || currentPC == 0) {
    return false;
  }

  uint8_t *dataBase = reinterpret_cast<uint8_t *>(skDeviceEntryArgsHost);
  SkDfxInfo *dfxInfo = reinterpret_cast<SkDfxInfo *>(dataBase + skHeaderInfoHost->dfxOffset);

  // Iterate through all nodes to find which one's entry range contains currentPC
  for (uint32_t i = 0; i < skHeaderInfoHost->nodeCnt; ++i) {
    uint64_t *entries = (coreType == RT_CORE_TYPE_AIC) ? dfxInfo[i].entryAic : dfxInfo[i].entryAiv;
    uint32_t funcSize = (coreType == RT_CORE_TYPE_AIC) ? dfxInfo[i].aicSize : dfxInfo[i].aivSize;

    for (int j = 0; j < 4; ++j) {
      if (entries[j] == 0) {
        continue;  // Skip invalid entries
      }

      uint64_t entryAddr = entries[j];
      uint64_t endAddr = entryAddr + funcSize;

      // Check if currentPC falls within this entry's range [entryAddr, entryAddr + funcSize)
      if (currentPC >= entryAddr && currentPC < endAddr) {
        matchInfo.nodeIdx = i;
        matchInfo.entryIdx = j;
        matchInfo.entryAddr = entryAddr;
        matchInfo.endAddr = endAddr;
        matchInfo.funcSize = funcSize;
        matchInfo.entries = entries;
        matchInfo.dfxNode = &dfxInfo[i];
        return true;
      }
    }
  }

  return false;
}

void SuperKernelExceptionHandler::PrintCoreSymbols(uint32_t coreId, rtCoreType_t coreType, uint64_t startPC,
                                                   uint64_t currentPC) {
  if (coreId >= aicoreNums) {
    SK_LOGE("coreId=%u exceeds aicoreNums=%u", coreId, aicoreNums);
    return;
  }

  if (!hasOpTrace_) {
    return;
  }

  // Access counterInfo via offset from skDeviceEntryArgsHost
  uint8_t *dataBase = reinterpret_cast<uint8_t *>(skDeviceEntryArgsHost);
  SkCounterInfo *counterInfo = reinterpret_cast<SkCounterInfo *>(dataBase + skHeaderInfoHost->counterOffset);

  uint32_t opState = counterInfo[coreId].opState;
  uint32_t opId = counterInfo[coreId].index;
  const char *coreTypeName = (coreType == RT_CORE_TYPE_AIC) ? "AIC" : "AIV";

  SK_LOGE("[Core %u] Type=%s, opState=%u, opId=%u", coreId, coreTypeName, opState, opId);

  if (opState == static_cast<uint8_t>(SkOpTraceType::ORIGIN)) {
    // args initialized to 0, if ORIGIN means no SK has been executed yet
    SK_LOGE("[Core %u] No SK entry executed yet.", coreId);
  } else if (opState == static_cast<uint8_t>(SkOpTraceType::SK_ENTRY_LAUNCHED)) {
    // SK operator just started, no sub-kernel executed yet, print next operator symbols
    SK_LOGE("[Core %u] SK started but no sub-kernel executed yet, checking next operators", coreId);

    // Print next operator symbols (starting from first operator)
    if (opId < skHeaderInfoHost->nodeCnt) {
      SK_LOGE("[Core %u] > Next opId=%u (first sub-kernel):", coreId, opId);
      KernelFuncName nextKernelFuncName = GetOrLoadKernelSymbols(opId);
      PrintSymbolByCoreId(coreId, coreType, startPC, currentPC, nextKernelFuncName);
    } else {
      SK_LOGE("[Core %u] > No next operator (opId=%u/%u)", coreId, opId, skHeaderInfoHost->nodeCnt);
    }
  } else if (opState == static_cast<uint8_t>(SkOpTraceType::OP_LAUNCHED)) {
    // Sub-kernel is running, print current operator symbols
    SK_LOGE("[Core %u] Currently running opId=%u", coreId, opId);
    KernelFuncName kernelFuncName = GetOrLoadKernelSymbols(opId);
    PrintSymbolByCoreId(coreId, coreType, startPC, currentPC, kernelFuncName);
  } else if (opState == static_cast<uint8_t>(SkOpTraceType::OP_FINISHED)) {
    // Sub-kernel has finished execution, current opId is the last executed operator, print current and next operator
    // symbols
    SK_LOGE("[Core %u] opId=%u finished, checking current and next operators", coreId, opId);

    // Print current finished operator symbols
    SK_LOGE("[Core %u] > Last finished opId=%u:", coreId, opId);
    KernelFuncName currentKernelFuncName = GetOrLoadKernelSymbols(opId);
    PrintSymbolByCoreId(coreId, coreType, startPC, currentPC, currentKernelFuncName);

    // Print next operator symbols (not exceeding nodeCnt)
    if (opId + 1 < skHeaderInfoHost->nodeCnt) {
      SK_LOGE("[Core %u] > Next opId=%u:", coreId, opId + 1);
      KernelFuncName nextKernelFuncName = GetOrLoadKernelSymbols(opId + 1);
      PrintSymbolByCoreId(coreId, coreType, startPC, currentPC, nextKernelFuncName);
    } else {
      SK_LOGE("[Core %u] > No next operator (opId=%u/%u)", coreId, opId, skHeaderInfoHost->nodeCnt);
    }
  } else if (opState == static_cast<uint8_t>(SkOpTraceType::SK_ENTRY_FINISHED)) {
    // SK operator execution completed
    SK_LOGE("[Core %u] SK entry operator execution completed.", coreId);
  } else {
    SK_LOGE("[Core %u] Unknown opState: %u", coreId, opState);
  }
}

void SuperKernelExceptionHandler::PrintAllCoreSymbols() {
  if (!hasOpTrace_) {
    return;
  }

  SK_LOGE("==================================================");
  SK_LOGE("=== Sub-kernel running info on all %u cores ===", aicoreNums);
  SK_LOGE("==================================================");

  for (uint32_t coreId = 0; coreId < aicoreNums; coreId++) {
    // Determine core type
    rtCoreType_t coreType = CoreIsAiv(coreId) ? RT_CORE_TYPE_AIV : RT_CORE_TYPE_AIC;

    // Temporarily use default values, as real PC values cannot be obtained from counter info
    uint64_t startPC = 0;
    uint64_t currentPC = 0;

    // Call common function to print symbols
    PrintCoreSymbols(coreId, coreType, startPC, currentPC);
  }
}

bool SuperKernelExceptionHandler::ParseAndPrintSubKernelSymbols(aclrtExceptionInfo *exceptionInfo) {
  // Get exception register information
  ExceptionRegInfo exceptionRegInfo{0, nullptr};
  if (!GetExceptionRegInfo(*reinterpret_cast<const aclrtExceptionInfo *>(exceptionInfo), exceptionRegInfo)) {
    // Note: errRegInfo pointer is allocated by RTS, no need to manually free
    return false;
  }

  SK_LOGE("====================================================");
  SK_LOGE("=== Sub-kernel running info on exception cores ===");
  SK_LOGE("====================================================");
  for (uint32_t i = 0; i < exceptionRegInfo.coreNum; i++) {
    rtExceptionErrRegInfo_t coreErrRegInfo = exceptionRegInfo.errRegInfo[i];
    rtCoreType_t coreType = static_cast<rtCoreType_t>(coreErrRegInfo.coreType);
    // Always try to identify error node by PC address first
    IdentifyErrorNodeByPC(coreErrRegInfo.coreId, coreType, coreErrRegInfo.startPC, coreErrRegInfo.currentPC);

    PrintCoreSymbols(coreErrRegInfo.coreId, coreType, coreErrRegInfo.startPC, coreErrRegInfo.currentPC);

    // Parse COND register value from errReg[20](low32) and errReg[21](high32)
    uint64_t condValue = GetCondRegValue(coreErrRegInfo);
    ParseAndPrintCondInfo(coreErrRegInfo.coreId, coreType, condValue);
  }

  return true;
}

void SuperKernelExceptionHandler::FreeResources() {
  if (skDeviceEntryArgsHost != nullptr) {
    aclrtFreeHost(skDeviceEntryArgsHost);
    skDeviceEntryArgsHost = nullptr;
  }

  // skHeaderInfoHost points inside skDeviceEntryArgsHost (normal path, no separate free needed),
  // or has already been freed in error paths of CopySkDeviceEntryArgsToHost().
  skHeaderInfoHost = nullptr;
  errorNodeIdx_ = -1;
  opSymbolCache.clear();
  funcNodeIndices_.clear();
}

bool SuperKernelExceptionHandler::StartsWith(const char *source, const char *prefix) {
  if (!source || !prefix) {
    return false;
  }

  const size_t srcLen = strlen(source);
  const size_t prefixLen = strlen(prefix);
  if (prefixLen > srcLen) {
    return false;
  }
  return strncmp(source, prefix, prefixLen) == 0;
}

aclError SuperKernelExceptionHandler::CheckError(aclError ret, const char *errorMsg) {
  if (ret != ACL_SUCCESS) {
    SK_LOGE("Operation failed: %s returned error code %d", errorMsg, static_cast<int32_t>(ret));
  }
  return ret;
}

bool SuperKernelExceptionHandler::IsSuperKernelException(aclrtExceptionInfo *exceptionInfo) {
  char funcName[MAX_FUNC_NAME_LEN] = {0};

  // Get exception function handle
  aclrtFuncHandle funcHandle = nullptr;
  auto ret = aclrtGetFuncHandleFromExceptionInfo(exceptionInfo, &funcHandle);
  if (ret != ACL_SUCCESS) {
    SK_DLOGE("Failed to get func handle from exception info, ret=%d", ret);
    return false;
  }

  // Get function name
  ret = aclrtGetFunctionName(funcHandle, MAX_FUNC_NAME_LEN, funcName);
  if (ret != ACL_SUCCESS) {
    SK_DLOGE("Failed to get function name, ret=%d", ret);
    return false;
  }

  // Check if function name starts with sk_entry (e.g., sk_entry, sk_entry_aiv, sk_entry_mix11, etc.)
  if (!StartsWith(funcName, "sk_entry")) {
    SK_DLOGD("fault kernel_name '%s' does not start with 'sk_entry', skipping", funcName);
    return false;
  }

  // Check if function name contains "op_trace" for additional symbol printing
  hasOpTrace_ = (strstr(funcName, "op_trace") != nullptr);

  SK_DLOGE("Exception is from superkernel function '%s', op_trace=%s, proceeding with handling", funcName,
           hasOpTrace_ ? "true" : "false");
  return true;
}

/**
 * \brief Prepare common operations before exception dump
 * \param exceptionInfo Input: exception info
 * \param exceptionRegInfo Output: exception register info
 * \return aclError ACL_SUCCESS on success, error code on failure
 */
aclError SuperKernelExceptionHandler::PrepareExceptionDump(aclrtExceptionInfo *exceptionInfo,
                                                           ExceptionRegInfo &exceptionRegInfo,
                                                           std::unique_ptr<sk::logger::LogContextGuard> &logContext) {
  // Extract SK entry arguments from device to host and parse task queue
  if (!ExtractSkEntryArgs(exceptionInfo, logContext)) {
    FreeResources();
    SK_DLOGE("Failed to extract SK entry args");
    return ACL_ERROR_FAILURE;
  }
  if (!ExtractTaskQueue()) {
    FreeResources();
    SK_LOGE("Failed to extract SK task queue");
    return ACL_ERROR_FAILURE;
  }

  // Get exception register information (core ID, PC pointer, core type)
  exceptionRegInfo = {0, nullptr};
  if (!GetExceptionRegInfo(*exceptionInfo, exceptionRegInfo)) {
    FreeResources();
    SK_LOGE("Failed to get exception register info");
    return ACL_ERROR_FAILURE;
  }

  return ACL_SUCCESS;
}

/**
 * \brief Fill sk_entry function name into kernelDisplayName
 * \param dumpInfo Output: dump info structure
 * \param exceptionInfo Input: exception info
 * \return aclError ACL_SUCCESS on success, error code on failure
 */
aclError SuperKernelExceptionHandler::PopulateSkEntryFields(Adx::ExceptionDumpInfo &dumpInfo,
                                                            aclrtExceptionInfo *exceptionInfo) {
  char skFuncName[MAX_FUNC_NAME_LEN] = {0};
  aclrtFuncHandle funcHandle = nullptr;

  aclError ret = aclrtGetFuncHandleFromExceptionInfo(exceptionInfo, &funcHandle);
  if (ret != ACL_SUCCESS) {
    SK_LOGE("Failed to get func handle from exception info, ret=%d", ret);
    return ACL_ERROR_FAILURE;
  }

  ret = aclrtGetFunctionName(funcHandle, MAX_FUNC_NAME_LEN, skFuncName);
  if (ret != ACL_SUCCESS) {
    SK_LOGE("Failed to get function name, ret=%d", ret);
    return ACL_ERROR_FAILURE;
  }

  uint16_t modelIdIdx = static_cast<uint16_t>((skHeaderInfoHost->modelIdIndexAndSkScopeId >> 32) & 0xFFFF);
  uint16_t skScopeId = static_cast<uint16_t>((skHeaderInfoHost->modelIdIndexAndSkScopeId >> 16) & 0xFFFF);
  std::string modelId = SkEventRecorder::Instance().GetModelIdByIndex(modelIdIdx);
  const std::string skNameFromRecorder = SkEventRecorder::Instance().GetSkName(modelId, skScopeId);
  if (!skNameFromRecorder.empty()) {
    snprintf_s(dumpInfo.kernelDisplayName, Adx::MAX_KERNELNAME_LEN, Adx::MAX_KERNELNAME_LEN - 1, "%s",
               skNameFromRecorder.c_str());
  } else {
    snprintf_s(dumpInfo.kernelDisplayName, Adx::MAX_KERNELNAME_LEN, Adx::MAX_KERNELNAME_LEN - 1, "%s_scope%u",
               skFuncName, skScopeId);
  }
  return ACL_SUCCESS;
}

/**
 * \brief Fill tensor related fields (coreId, coreType, extraTensor)
 * \param dumpInfo Output: dump info structure
 * \param coreId Input: exception core ID
 * \param coreType Input: exception core type
 */
void SuperKernelExceptionHandler::PopulateTensorFields(Adx::ExceptionDumpInfo &dumpInfo, uint32_t coreId,
                                                       rtCoreType_t coreType) {
  dumpInfo.coreId = coreId;
  dumpInfo.coreType = coreType;
  dumpInfo.extraTensorNum = 1;

  Adx::TensorInfo &tensor = dumpInfo.extraTensor[0];
  tensor.type = Adx::TensorType::WORKSPACE;
  tensor.tensorSize = skHeaderInfoHost->totalSize;
  tensor.format = ACL_FORMAT_ND;
  tensor.dataType = ACL_UINT8;
  tensor.tensorAddr = reinterpret_cast<int64_t *>(skDeviceEntryArgsDev);
  tensor.addrType = Adx::AddressType::RAW;
  tensor.placement = Adx::TensorPlacement::kOnDeviceHbm;
  tensor.argsOffSet = 0;
  tensor.shape = {static_cast<int64_t>(tensor.tensorSize)};
  tensor.originShape = {static_cast<int64_t>(tensor.tensorSize)};
}

/**
 * \brief Fill sub-kernel fields (bin, kernelName, argAddr, argSize)
 * \param dumpInfo Output: dump info structure
 * \param errorNodeIdx Input: index of the error sub-kernel node
 * \param exceptionInfo exception info for getting SK entry func handle
 * \param coreId exception core id
 * \return true: filled sub-kernel info; false: no match, fields filled with SK entry info
 */
bool SuperKernelExceptionHandler::PopulateSubKernelFields(Adx::ExceptionDumpInfo &dumpInfo, int32_t errorNodeIdx,
                                                          aclrtExceptionInfo *exceptionInfo) {
  if (errorNodeIdx < 0 || static_cast<uint32_t>(errorNodeIdx) >= skHeaderInfoHost->nodeCnt) {
    SK_LOGI("No sub kernel matched, fill with SK entry fields");
    char funcName[MAX_FUNC_NAME_LEN] = {0};

    // Get SK entry func handle from exception info
    aclrtFuncHandle funcHandle = nullptr;
    aclError ret = aclrtGetFuncHandleFromExceptionInfo(exceptionInfo, &funcHandle);
    if (ret != ACL_SUCCESS || funcHandle == nullptr) {
      SK_LOGE("Failed to get SK entry func handle, ret=%d", ret);
      memset_s(dumpInfo.kernelName, sizeof(dumpInfo.kernelName), 0, sizeof(dumpInfo.kernelName));
      dumpInfo.bin = nullptr;
      dumpInfo.argAddr = nullptr;
      dumpInfo.argSize = 0;
      return false;
    }

    // Fill kernel name
    ret = aclrtGetFunctionName(funcHandle, MAX_FUNC_NAME_LEN, funcName);
    if (ret == ACL_SUCCESS) {
      SK_LOGI("Successfully got function name: %s", funcName);
      snprintf_s(dumpInfo.kernelName, Adx::MAX_KERNELNAME_LEN, Adx::MAX_KERNELNAME_LEN - 1, "%s", funcName);
    } else {
      SK_LOGW("Failed to get SK entry function name, ret=%d; continuing without function name", ret);
      memset_s(dumpInfo.kernelName, sizeof(dumpInfo.kernelName), 0, sizeof(dumpInfo.kernelName));
    }

    // Fill bin handle
    ret = aclrtFunctionGetBinary(funcHandle, &dumpInfo.bin);
    if (ret != ACL_SUCCESS) {
      SK_LOGE("Failed to get SK entry bin handle, ret=%d", ret);
      dumpInfo.bin = nullptr;
    }

    // Exception occurred within Superkernel (not in sub-kernel), no sub-kernel args available
    dumpInfo.argAddr = nullptr;
    dumpInfo.argSize = 0;
    SK_LOGI("Does not match sub kernel, exception occurred within Superkernel");
    return true;
  }

  SK_LOGI("Fill sub kernel info, errorNodeIdx=%d", errorNodeIdx);
  uint8_t *dataBase = reinterpret_cast<uint8_t *>(skDeviceEntryArgsHost);
  SkDfxInfo *dfxInfo = reinterpret_cast<SkDfxInfo *>(dataBase + skHeaderInfoHost->dfxOffset);
  SkDfxInfo &errorNode = dfxInfo[errorNodeIdx];

  dumpInfo.bin = reinterpret_cast<void *>(errorNode.binHdl);
  KernelFuncName kernelFunc = GetOrLoadKernelSymbols(errorNodeIdx);
  snprintf_s(dumpInfo.kernelName, Adx::MAX_KERNELNAME_LEN, Adx::MAX_KERNELNAME_LEN - 1, "%s", kernelFunc.name.c_str());

  uint64_t argsAddr = 0;
  uint32_t argsSize = 0;
  GetSubKernelTaskArgs(errorNodeIdx, argsAddr, argsSize);
  dumpInfo.argAddr = reinterpret_cast<void *>(argsAddr);
  dumpInfo.argSize = argsSize;
  SK_LOGI("Match sub kernel success, exception occurred within sub kernel");
  return true;
}

/**
 * \brief Populate all fields of exception dump info structure
 * \param dumpInfo Output: dump info structure
 * \param errorNodeIdx Index of the error sub-kernel node
 * \param exceptionInfo exception info
 * \param coreId exception core id
 * \param coreType exception core type (AIC/AIV)
 * \return aclError ACL_SUCCESS on success, error code on failure
 */
aclError SuperKernelExceptionHandler::PopulateDumpInfoFields(Adx::ExceptionDumpInfo &dumpInfo, int32_t errorNodeIdx,
                                                             aclrtExceptionInfo *exceptionInfo, uint32_t coreId,
                                                             rtCoreType_t coreType) {
  if (skHeaderInfoHost == nullptr || skDeviceEntryArgsDev == nullptr) {
    SK_LOGE("PopulateDumpInfoFields: null pointer");
    return ACL_ERROR_FAILURE;
  }

  aclError ret = PopulateSkEntryFields(dumpInfo, exceptionInfo);
  if (ret != ACL_SUCCESS) {
    SK_LOGE("Failed to populate SK entry fields, ret=%d", ret);
    return ret;
  }

  PopulateTensorFields(dumpInfo, coreId, coreType);
  if (PopulateSubKernelFields(dumpInfo, errorNodeIdx, exceptionInfo)) {
    SK_LOGI("PopulateSubKernelFields success");
  }

  return ACL_SUCCESS;
}

/**
 * \brief Fill exception dump information structure
 * \param dumpInfo Output: exception dump info structure to be filled
 * \return aclError ACL_SUCCESS on success, error code on failure
 */
aclError SuperKernelExceptionHandler::FillExceptionDumpInfo(Adx::ExceptionDumpInfo &dumpInfo,
                                                            aclrtExceptionInfo *exceptionInfo) {
  if (exceptionInfo == nullptr) {
    SK_DLOGE("FillExceptionDumpInfo: no exception info");
    return ACL_ERROR_INVALID_PARAM;
  }

  ExceptionRegInfo exceptionRegInfo{0, nullptr};
  std::unique_ptr<sk::logger::LogContextGuard> logContext;
  aclError ret = PrepareExceptionDump(exceptionInfo, exceptionRegInfo, logContext);
  if (ret != ACL_SUCCESS) {
    return ret;
  }

  int32_t errorNodeIdx = -1;
  for (uint32_t i = 0; i < exceptionRegInfo.coreNum; i++) {
    rtExceptionErrRegInfo_t coreErr = exceptionRegInfo.errRegInfo[i];
    errorNodeIdx_ = -1;
    ErrorNodeMatchInfo matchInfo;
    if (FindErrorNodeByPC((rtCoreType_t)coreErr.coreType, coreErr.currentPC, matchInfo)) {
      errorNodeIdx_ = matchInfo.nodeIdx;
    }
    errorNodeIdx = errorNodeIdx_;
    (void)PopulateDumpInfoFields(dumpInfo, errorNodeIdx, exceptionInfo, coreErr.coreId, (rtCoreType_t)coreErr.coreType);
  }

  SK_LOGI("FillExceptionDumpInfo final errorNodeIdx=%d", errorNodeIdx);
  FreeResources();
  return ACL_SUCCESS;
}

bool SuperKernelExceptionHandler::GetSubKernelTaskArgs(uint32_t nodeIdx, uint64_t &argsAddr, uint32_t &argsSize) {
  if (nodeIdx >= skHeaderInfoHost->nodeCnt) {
    return false;
  }
  uint8_t *dataBase = reinterpret_cast<uint8_t *>(skDeviceEntryArgsHost);

  auto GetTaskArgs = [&](uint32_t queOffset, uint32_t taskCnt) -> bool {
    if (queOffset == 0 || taskCnt == 0) {
      SK_LOGI("queOffset or taskCnt is null, skip");
      return false;
    }
    TaskQue *taskQue = reinterpret_cast<TaskQue *>(dataBase + queOffset);
    for (uint32_t i = 0; i < taskCnt; ++i) {
      const TaskInfo &task = taskQue->taskInfos[i];
      if (task.index == nodeIdx && task.type == SkTaskType::TYPE_FUNC) {
        argsAddr = task.args;
        argsSize = task.argsSize;
        return true;
      }
    }
    SK_LOGI("task not found, skip");
    return false;
  };

  if (GetTaskArgs(skHeaderInfoHost->aivQueOffset, aivTaskCnt)) {
    return true;
  }
  if (GetTaskArgs(skHeaderInfoHost->aicQueOffset, aicTaskCnt)) {
    return true;
  }
  return false;
}

uint32_t SuperKernelExceptionHandler::FillUniqueExceptionDumpInfos(aclrtExceptionInfo *exceptionInfo,
                                                                   const ExceptionRegInfo &exceptionRegInfo,
                                                                   Adx::ExceptionDumpInfo *exceptionDumpInfo,
                                                                   uint32_t exceptionDumpSize) {
  uint32_t validDumpNum = 0;
  std::unordered_set<uintptr_t> seenBinHandles;
  for (uint32_t i = 0; i < exceptionRegInfo.coreNum && validDumpNum < exceptionDumpSize; ++i) {
    rtExceptionErrRegInfo_t &coreErr = exceptionRegInfo.errRegInfo[i];
    errorNodeIdx_ = -1;
    ErrorNodeMatchInfo matchInfo;
    if (FindErrorNodeByPC((rtCoreType_t)coreErr.coreType, coreErr.currentPC, matchInfo)) {
      errorNodeIdx_ = matchInfo.nodeIdx;
    }

    Adx::ExceptionDumpInfo candidate = {};
    aclError ret =
        PopulateDumpInfoFields(candidate, errorNodeIdx_, exceptionInfo, coreErr.coreId, (rtCoreType_t)coreErr.coreType);
    if (ret != ACL_SUCCESS) {
      continue;
    }

    uintptr_t binHandle = reinterpret_cast<uintptr_t>(candidate.bin);
    if (binHandle != 0 && seenBinHandles.find(binHandle) != seenBinHandles.end()) {
      SK_LOGI("Skip duplicate ExceptionDumpInfo by binHandle=0x%lx, coreId=%u, coreType=%u", binHandle, coreErr.coreId,
              coreErr.coreType);
      continue;
    }
    if (binHandle != 0) {
      seenBinHandles.insert(binHandle);
    }

    exceptionDumpInfo[validDumpNum] = candidate;
    validDumpNum++;
  }
  return validDumpNum;
}

namespace {
static const char *TensorTypeToString(Adx::TensorType type) {
  switch (type) {
    case Adx::TensorType::INPUT:
      return "INPUT";
    case Adx::TensorType::OUTPUT:
      return "OUTPUT";
    case Adx::TensorType::WORKSPACE:
      return "WORKSPACE";
    default:
      return "UNKNOWN";
  }
}

static const char *AddressTypeToString(Adx::AddressType type) {
  switch (type) {
    case Adx::AddressType::TRADITIONAL:
      return "TRADITIONAL";
    case Adx::AddressType::NOTILING:
      return "NOTILING";
    case Adx::AddressType::RAW:
      return "RAW";
    default:
      return "UNKNOWN";
  }
}

static void PrintVector(const std::vector<int64_t> &vec, const char *name) {
  if (vec.empty()) {
    SK_LOGI("    %-18s: empty", name);
    return;
  }
  std::string vecStr;
  for (size_t i = 0; i < vec.size(); ++i) {
    vecStr += std::to_string(vec[i]);
    if (i != vec.size() - 1) vecStr += ", ";
  }
  SK_LOGI("    %-18s: [%s]", name, vecStr.c_str());
}
}  // namespace

static void PrintExceptionDumpInfoArray(Adx::ExceptionDumpInfo *dumpInfos, uint32_t dumpCount) {
  if (dumpInfos == nullptr || dumpCount == 0) {
    SK_LOGI("Adx::ExceptionDumpInfo array is empty, skip print");
    return;
  }

  SK_LOGI("==============================================================");
  SK_LOGI("======= SuperKernel Exception Dump Info Array (count=%u) ======", dumpCount);
  SK_LOGI("==============================================================");

  for (uint32_t i = 0; i < dumpCount; i++) {
    Adx::ExceptionDumpInfo &info = dumpInfos[i];
    SK_LOGI("------------------ Adx::ExceptionDumpInfo [%u] ------------------", i);
    SK_LOGI("coreId             : %u", info.coreId);
    SK_LOGI("coreType           : %s", (info.coreType == RT_CORE_TYPE_AIC) ? "AIC" : "AIV");
    SK_LOGI("argSize            : %u", info.argSize);
    SK_LOGI("argAddr            : 0x%lx", reinterpret_cast<uint64_t>(info.argAddr));
    SK_LOGI("bin handle         : 0x%lx", reinterpret_cast<uint64_t>(reinterpret_cast<void *>(info.bin)));
    SK_LOGI("kernelName         : %s", info.kernelName);
    SK_LOGI("kernelDisplayName  : %s", info.kernelDisplayName);
    SK_LOGI("extraTensorNum     : %u", info.extraTensorNum);

    for (uint32_t j = 0; j < info.extraTensorNum && j < Adx::EXCEPTION_DUMP_MAX_TENSOR_NUM; j++) {
      Adx::TensorInfo &tensor = info.extraTensor[j];
      SK_LOGI("  >>> extraTensor[%u] <<<", j);
      SK_LOGI("    type           : %s", TensorTypeToString(tensor.type));
      SK_LOGI("    tensorSize     : %zu bytes", tensor.tensorSize);
      SK_LOGI("    format         : %d", tensor.format);
      SK_LOGI("    dataType       : %d", tensor.dataType);
      SK_LOGI("    tensorAddr     : 0x%lx", reinterpret_cast<uint64_t>(tensor.tensorAddr));
      SK_LOGI("    addrType       : %s", AddressTypeToString(tensor.addrType));
      SK_LOGI("    placement      : %d", tensor.placement);
      SK_LOGI("    argsOffSet     : %u", tensor.argsOffSet);
      PrintVector(tensor.shape, "shape");
      PrintVector(tensor.originShape, "originShape");
    }
  }
  SK_LOGI("==============================================================\n");
}

void SuperKernelExceptionCallBackFunc(aclrtExceptionInfo *exceptionInfo) {
  SuperKernelExceptionHandler handler;
  handler.HandleException(exceptionInfo);
}

uint32_t SuperKernelExceptionHandler::ProcessExceptionDump(aclrtExceptionInfo *exceptionInfo,
                                                           Adx::ExceptionDumpInfo *exceptionDumpInfo,
                                                           uint32_t exceptionDumpSize, uint32_t *exceptionDumpRealSize,
                                                           Adx::ExceptionDumpMode *mode) {
  if (exceptionDumpSize < 1) {
    SK_DLOGE("ExceptionDumpCallBack: exceptionDumpSize too small");
    return ACL_ERROR_INVALID_PARAM;
  }

  *exceptionDumpRealSize = 0;
  memset_s(exceptionDumpInfo, exceptionDumpSize * sizeof(Adx::ExceptionDumpInfo), 0,
           exceptionDumpSize * sizeof(Adx::ExceptionDumpInfo));

  if (!IsSuperKernelException(exceptionInfo)) {
    SK_DLOGD("Not superkernel exception, skip dump");
    *exceptionDumpRealSize = 0;
    *mode = Adx::ExceptionDumpMode::DUMP_MODE_NONE;
    return ACL_SUCCESS;
  }
  ExceptionRegInfo exceptionRegInfo{0, nullptr};
  std::unique_ptr<sk::logger::LogContextGuard> logContext;
  aclError ret = PrepareExceptionDump(exceptionInfo, exceptionRegInfo, logContext);
  if (ret != ACL_SUCCESS) {
    return ret;
  }
  SK_LOGI("Exception is in SuperKernel");

  uint32_t validDumpNum =
      FillUniqueExceptionDumpInfos(exceptionInfo, exceptionRegInfo, exceptionDumpInfo, exceptionDumpSize);

  PrintExceptionDumpInfoArray(exceptionDumpInfo, validDumpNum);
  *exceptionDumpRealSize = validDumpNum;
  *mode = Adx::ExceptionDumpMode::DUMP_MODE_OVERWRITE;

  FreeResources();
  SK_LOGI("Adx::ExceptionDumpInfo fill success, coreNum=%u, validDump=%u", exceptionRegInfo.coreNum, validDumpNum);
  return ACL_SUCCESS;
}

bool IsValidCommonException(const rtExceptionExpandType_t exceptionTaskType) {
  return exceptionTaskType != RT_EXCEPTION_AICORE && exceptionTaskType != RT_EXCEPTION_FFTS_PLUS &&
         exceptionTaskType != RT_EXCEPTION_FUSION;
}

uint32_t ExceptionDumpInfoCallBack(void *exceptionInfo, Adx::ExceptionDumpInfo *exceptionDumpInfo,
                                   uint32_t exceptionDumpSize, uint32_t *exceptionDumpRealSize,
                                   Adx::ExceptionDumpMode *mode) {
  if (exceptionInfo == nullptr || exceptionDumpInfo == nullptr || exceptionDumpRealSize == nullptr) {
    SK_DLOGE("ExceptionDumpCallBack: invalid null params");
    return ACL_ERROR_INVALID_PARAM;
  }
  if (IsValidCommonException(static_cast<aclrtExceptionInfo *>(exceptionInfo)->expandInfo.type)) {
    SK_DLOGI("Not superkernel exception, skip dump");
    *exceptionDumpRealSize = 0;
    *mode = Adx::ExceptionDumpMode::DUMP_MODE_NONE;
    return ACL_SUCCESS;
  }
  SK_DLOGI("Start SuperKernelExceptionHandler::ProcessExceptionDump");
  SuperKernelExceptionHandler handler;
  return handler.ProcessExceptionDump(static_cast<aclrtExceptionInfo *>(exceptionInfo), exceptionDumpInfo,
                                      exceptionDumpSize, exceptionDumpRealSize, mode);
}
