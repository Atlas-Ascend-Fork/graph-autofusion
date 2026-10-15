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
 * \file sk_node.cpp
 * \brief
 */

#include <map>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <memory>
#include <mutex>
#include <limits>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <sstream>
#include <fstream>
#include <utility>
#include <cstring>

#include "sk_node.h"
#include "sk_log.h"
#include "sk_scope_info.h"
#include "sk_lock_detector.h"
#include "sk_common.h"
#include "sk_options_manager.h"
#include "runtime/rt_external_kernel.h"
#include "kernel.h"
#include "base.h"

extern "C" aclrtBinHandle AscendGetEntryBinHandle();

constexpr uint64_t INVALID_SK_BIND_VALUE = 0xffffffffffffffffULL;

namespace {
bool GetFunctionAllocUbufSize(aclrtFuncHandle funcHandle, size_t &allocUbufSize, const std::string &context) {
  uint32_t allocUbufSizeValue = 0;
  rtError_t ret = rtFunctionGetMetaInfo(funcHandle, RT_FUNCTION_TYPE_COMPILER_ALLOC_UB_SIZE, &allocUbufSizeValue,
                                        sizeof(allocUbufSizeValue));
  if (ret != RT_ERROR_NONE) {
    SK_LOGE("rtFunctionGetMetaInfo COMPILER_ALLOC_UB_SIZE failed for %s, ret=%d", context.c_str(), ret);
    return false;
  }
  allocUbufSize = static_cast<size_t>(allocUbufSizeValue);
  return true;
}

const char *FusionFailReasonDetail(FusionFailReason reason) {
  switch (reason) {
    case FusionFailReason::CAN_FUSE:
      return "node can fuse";
    case FusionFailReason::OP_UNSUPPORT:
      return "Failed to resolve SuperKernel bind map for the operator";
    case FusionFailReason::DYNAMIC_TASK_UNSUPPORT:
      return "The operator will refresh task information at runtime, but SK does not support fusing dynamically "
             "changing tasks";
    case FusionFailReason::NOT_IN_SCOPE:
      return "The user actively marked that this operator is not fused";
    case FusionFailReason::IN_UNFUSIBLE_SCOPE:
      return "This operator is not within the fusion range marked by the user";
    case FusionFailReason::EXCEED_CORE_MAX:
      return "The number of kernels required by the operator exceeds the maximum number of kernels that the device can "
             "provide";
    case FusionFailReason::RESET_TYPE_NODE:
      return "reset type node in end";
    case FusionFailReason::ISOLATED_EVENT:
      return "There is no kernel node on the stream where the current node is located, and this stream is within the "
             "scope";
    case FusionFailReason::EXIST_DEADLOCK:
      return "deadlock exists";
    case FusionFailReason::SCOPE_FUSE_PART:
      return "scope fuse failed";
    case FusionFailReason::EXTERNAL_DEPEND:
      return "event node has external dependency";
    case FusionFailReason::UNSUPPORT_EVENT_TYPE:
      return "unsupported event type";
    case FusionFailReason::MEMORY_WAIT_NODE_ONLY:
      return "No memory write exists, meaning the memory write is outside modelRI. Therefore change all waits to event "
             "semantics, but they cannot be fused.";
    case FusionFailReason::MEMORY_WRITE_NODE_ONLY:
      return "only exists memory write nodes, mask it as unfusible";
    case FusionFailReason::DEFAULT_NODE:
      return "default node uses aicpu resources, mask it as unfusible";
    case FusionFailReason::KERNEL_ATTR_GET_FAILED:
      return "Failed to get kernel attribute for SuperKernel fusion";
    case FusionFailReason::EXCEED_SCOPE_MAX:
      return "Exceeded maximum scope number limit for SuperKernel fusion";
    default:
      return "UNKNOWN_REASON";
  }
}

}  // namespace

KernelCapBits ParseKernelCapBits(uint64_t cap) {
  const auto getBit = [cap](KernelCapBitOffset offset) -> bool {
    return ((cap >> static_cast<uint8_t>(offset)) & 1ULL) != 0;
  };

  KernelCapBits bits;
  bits.earlyStartWaitFlag = getBit(KernelCapBitOffset::EARLY_START_WAIT_FLAG);
  bits.earlyStartSetFlag = getBit(KernelCapBitOffset::EARLY_START_SET_FLAG);
  bits.disableDcci = getBit(KernelCapBitOffset::DCCI);
  bits.disableScheMode = getBit(KernelCapBitOffset::DISABLE_SCHEMODE);
  bits.blockDimScaleUp = getBit(KernelCapBitOffset::BLOCKDIM_SCALE_UP);
  return bits;
}

// Implementation of FusionFailReasonInfo methods (requires complete ScopeProcessStatus/DeadlockFailReason definition)
FusionFailReasonInfo::FusionFailReasonInfo()
    : scopeProcessStatus(ScopeProcessStatus::INIT),
      deadlockFailReason(DeadlockFailReason::NOT_FIND_DEADLOCK),
      bindmapFailReason(BindmapFailReason::NONE) {}

FusionFailReasonInfo::FusionFailReasonInfo(FusionFailReason reason)
    : primary(reason),
      scopeProcessStatus(ScopeProcessStatus::INIT),
      deadlockFailReason(DeadlockFailReason::NOT_FIND_DEADLOCK),
      bindmapFailReason(BindmapFailReason::NONE) {}

FusionFailReasonInfo::FusionFailReasonInfo(ScopeProcessStatus scopeStatus)
    : primary(FusionFailReason::SCOPE_FUSE_PART),
      scopeProcessStatus(scopeStatus),
      deadlockFailReason(DeadlockFailReason::NOT_FIND_DEADLOCK),
      bindmapFailReason(BindmapFailReason::NONE) {}

FusionFailReasonInfo::FusionFailReasonInfo(DeadlockFailReason deadlockReason)
    : primary(FusionFailReason::EXIST_DEADLOCK),
      scopeProcessStatus(ScopeProcessStatus::INIT),
      deadlockFailReason(deadlockReason),
      bindmapFailReason(BindmapFailReason::NONE) {}

ScopeProcessStatus FusionFailReasonInfo::GetScopeProcessStatus() const {
  return scopeProcessStatus;
}

void FusionFailReasonInfo::SetScopeProcessStatus(ScopeProcessStatus scopeStatus) {
  scopeProcessStatus = scopeStatus;
}

const char *FusionFailReasonInfo::GetScopeDetail() const {
  return ScopeProcessStatusDetail(GetScopeProcessStatus());
}

DeadlockFailReason FusionFailReasonInfo::GetDeadlockFailReason() const {
  return deadlockFailReason;
}

void FusionFailReasonInfo::SetDeadlockFailReason(DeadlockFailReason deadlockReason) {
  deadlockFailReason = deadlockReason;
}

const char *FusionFailReasonInfo::GetDeadlockDetail() const {
  switch (GetDeadlockFailReason()) {
    case DeadlockFailReason::KERNEL_INSUFFICIENT_CORES:
      return "The wait node depends on a kernel node that requires more cores than available";
    case DeadlockFailReason::NOTIFY_INSUFFICIENT_CORES:
      return "The wait node depends on a notify node which has fused in other SK that it requires more cores";
    case DeadlockFailReason::NOTIFY_NOT_IN_GRAPH:
      return "The wait node depends on a notify node that it is not in graph";
    case DeadlockFailReason::NOTIFY_AFTER_SK_RANGE:
      return "The wait node depends on a notify node that it is after SK range";
    case DeadlockFailReason::NOTIFY_INVALID:
      return "In check deadlock of this wait node, get invalid notify node";
    case DeadlockFailReason::FIRST_WAIT:
      return "The wait node is first node in scope";
    case DeadlockFailReason::NO_SUPPORT_NODE:
      return "The wait node waiting for unsupport node type";
    default:
      return "";
  }
}

FusionFailReasonInfo::FusionFailReasonInfo(BindmapFailReason bindmapReason)
    : primary(FusionFailReason::OP_UNSUPPORT),
      scopeProcessStatus(ScopeProcessStatus::INIT),
      deadlockFailReason(DeadlockFailReason::NOT_FIND_DEADLOCK),
      bindmapFailReason(bindmapReason) {}

BindmapFailReason FusionFailReasonInfo::GetBindmapFailReason() const {
  return bindmapFailReason;
}

void FusionFailReasonInfo::SetBindmapFailReason(BindmapFailReason bindmapReason) {
  bindmapFailReason = bindmapReason;
}

const char *FusionFailReasonInfo::GetBindmapDetail() const {
  switch (GetBindmapFailReason()) {
    case BindmapFailReason::BINDMAP_INIT_EMPTY:
      return "bindmap init empty";
    case BindmapFailReason::BINHDL_NULL:
      return "binHdl is null";
    case BindmapFailReason::FUNCHDL_NULL:
      return "funcHdl is null";
    case BindmapFailReason::FUNC_NOT_FOUND:
      return "function not found in bind map";
    case BindmapFailReason::BIN_DEV_ADDR_GET_FAILED:
      return "failed to get binary device address";
    case BindmapFailReason::FUNC_ADDR_GET_FAILED:
      return "failed to get function address";
    case BindmapFailReason::BINDMAP_ENTRY_CONFLICT:
      return "bind map entry conflict";
    case BindmapFailReason::BINDMAP_CAP_INCONSISTENT:
      return "bind map cap inconsistent";
    case BindmapFailReason::BIN_HOST_ADDR_GET_FAILED:
      return "failed to get binary host address";
    default:
      return "";
  }
}

std::string FusionFailReasonToStr(const FusionFailReasonInfo &info) {
  std::string reasonKey = to_string(info.primary);
  if (info.primary == FusionFailReason::SCOPE_FUSE_PART) {
    ScopeProcessStatus scopeStatus = info.GetScopeProcessStatus();
    if (scopeStatus != ScopeProcessStatus::INIT && scopeStatus != ScopeProcessStatus::SUCCESS) {
      reasonKey += " [";
      reasonKey += to_string(scopeStatus);
      reasonKey += "]";
    }
  } else if (info.primary == FusionFailReason::EXIST_DEADLOCK) {
    DeadlockFailReason deadlockReason = info.GetDeadlockFailReason();
    if (deadlockReason != DeadlockFailReason::NOT_FIND_DEADLOCK) {
      reasonKey += " [";
      reasonKey += to_string(deadlockReason);
      reasonKey += "]";
    }
  } else if (info.primary == FusionFailReason::OP_UNSUPPORT) {
    BindmapFailReason bindmapReason = info.GetBindmapFailReason();
    if (bindmapReason != BindmapFailReason::NONE) {
      reasonKey += " [";
      reasonKey += to_string(bindmapReason);
      reasonKey += "]";
    }
  }
  return reasonKey;
}

std::string FusionFailReasonDetailToStr(const FusionFailReasonInfo &info) {
  std::string reasonDetail = FusionFailReasonDetail(info.primary);
  if (info.primary == FusionFailReason::SCOPE_FUSE_PART) {
    ScopeProcessStatus scopeStatus = info.GetScopeProcessStatus();
    if (scopeStatus != ScopeProcessStatus::INIT && scopeStatus != ScopeProcessStatus::SUCCESS) {
      const char *scopeDetailStr = info.GetScopeDetail();
      if (scopeDetailStr[0] != '\0') {
        reasonDetail += ", ";
        reasonDetail += scopeDetailStr;
      }
    }
  } else if (info.primary == FusionFailReason::EXIST_DEADLOCK) {
    DeadlockFailReason deadlockReason = info.GetDeadlockFailReason();
    if (deadlockReason != DeadlockFailReason::NOT_FIND_DEADLOCK) {
      const char *deadlockDetailStr = info.GetDeadlockDetail();
      if (deadlockDetailStr[0] != '\0') {
        reasonDetail += ", ";
        reasonDetail += deadlockDetailStr;
      }
    }
  } else if (info.primary == FusionFailReason::OP_UNSUPPORT) {
    BindmapFailReason bindmapReason = info.GetBindmapFailReason();
    if (bindmapReason != BindmapFailReason::NONE) {
      const char *bindmapDetailStr = info.GetBindmapDetail();
      if (bindmapDetailStr[0] != '\0') {
        reasonDetail += ", ";
        reasonDetail += bindmapDetailStr;
      }
    }
  }
  return reasonDetail;
}

SkBindMap InitSuperKernelBindMap(aclrtBinHandle binHdl) {
  struct __attribute__((packed)) SknlValuePayload {
    uint32_t res;
    SknlMapInfo info;
  };
  constexpr size_t payloadSize = sizeof(SknlValuePayload);

  size_t metaNum = 0;
  if (int ret = rtBinaryGetMetaNum(binHdl, RT_BINARY_TYPE_SK_INFO, &metaNum) != 0) {
    SK_LOGI("rtBinaryGetMetaNum unsuccessful, ret=%d", ret);
    return SkBindMap();
  }
  if (metaNum == 0) {
    SK_LOGI("metaNum is zero!");
    return SkBindMap();
  }

  SK_LOGI("binHdl=0x%lx, metaNum=%lu, payloadSize=%zu", (uint64_t)binHdl, metaNum, payloadSize);

  std::vector<uint8_t> dataPool(metaNum * payloadSize);
  std::vector<size_t> infoSize(metaNum, payloadSize);
  std::vector<void *> metaDataList(metaNum);

  for (size_t i = 0; i < metaNum; ++i) {
    metaDataList[i] = &dataPool[i * payloadSize];
  }

  if (int ret =
          rtBinaryGetMetaInfo(binHdl, RT_BINARY_TYPE_SK_INFO, metaNum, metaDataList.data(), infoSize.data()) != 0) {
    SK_LOGW("Failed to get SuperKernel metadata from binary, ret=%d; fusion will be skipped", ret);
    return SkBindMap();
  }

  SkBindMap bindMap;
  for (size_t i = 0; i < metaNum; ++i) {
    SknlValuePayload *payload = (SknlValuePayload *)metaDataList[i];
    SknlMapInfo localInfo;
    memcpy_s(&localInfo, sizeof(SknlMapInfo), &(payload->info), sizeof(SknlMapInfo));

    SK_LOGI(
        "[%zu] cap=%lu, globalFunc=0x%lx, skFunc[0]=0x%lx, skFunc[1]=0x%lx, "
        "skFunc[2]=0x%lx, skFunc[3]=0x%lx",
        i, localInfo.cap, (uint64_t)localInfo.globalFunc, (uint64_t)localInfo.sknlFunc[0],
        (uint64_t)localInfo.sknlFunc[1], (uint64_t)localInfo.sknlFunc[2], (uint64_t)localInfo.sknlFunc[3]);

    const auto globalFunc = (uint64_t)(localInfo.globalFunc);
    SkBindInfo bindInfo;
    bindInfo.cap = localInfo.cap;
    bindInfo.sknlFuncs = {(uint64_t)(localInfo.sknlFunc[0]), (uint64_t)(localInfo.sknlFunc[1]),
                          (uint64_t)(localInfo.sknlFunc[2]), (uint64_t)(localInfo.sknlFunc[3])};
    auto it = bindMap.find(globalFunc);
    if (it != bindMap.end() && (it->second.cap != bindInfo.cap || it->second.sknlFuncs != bindInfo.sknlFuncs)) {
      SK_LOGI("InitSuperKernelBindMap: globalFunc=0x%lx is duplicated with different value", globalFunc);
      it->second.sknlFuncs[0] = INVALID_SK_BIND_VALUE;
      continue;
    }
    bindMap[globalFunc] = bindInfo;
  }
  return bindMap;
}

namespace {
constexpr uint32_t AIV_TYPE_SIMT_VF_ONLY = 3U;
constexpr uint32_t AIV_TYPE_SIMD_SIMT_MIX_VF = 4U;

using SkAllBinMap = std::unordered_map<aclrtBinHandle, SkBindMap>;

struct CoreFuncInitContext {
  ResolvedFunctionInfo *info;
  SkBindMap::iterator bindIt;
  size_t splitIdx;
  BindmapFailReason *failReason;
};

enum class SkNodeCoreType : uint32_t {
  AIC,
  AIV,
};

constexpr int32_t ACL_FUNC_ATTR_KERNEL_SCHEMODE_PLACEHOLDER = 3;

bool HasInvalidSkBindValue(const SkBindInfo &bindInfo) {
  return bindInfo.sknlFuncs[0] == INVALID_SK_BIND_VALUE;
}

bool UpdateKernelCap(const SkBindInfo &bindInfo, bool &hasCap, uint64_t &cap) {
  if (!hasCap) {
    cap = bindInfo.cap;
    hasCap = true;
    return true;
  }
  return cap == bindInfo.cap;
}

ScheModeState ParseScheModeState(int64_t rawValue) {
  constexpr int64_t SCHE_MODE_OFF_VALUE = 0;
  constexpr int64_t SCHE_MODE_ON_VALUE = 1;
  if (rawValue == SCHE_MODE_OFF_VALUE) {
    return ScheModeState::SCHE_MODE_OFF;
  }
  if (rawValue == SCHE_MODE_ON_VALUE) {
    return ScheModeState::SCHE_MODE_ON;
  }
  SK_LOGW("Invalid schemode value: %ld, valid value is 0 or 1", rawValue);
  return ScheModeState::NONE;
}

ScheModeState GetScheModeFromFuncAttr(aclrtFuncHandle funcHandle) {
  int64_t funcAttrScheModeValue = 0;
  aclError aclRet = aclrtGetFunctionAttribute(
      funcHandle, static_cast<aclrtFuncAttribute>(ACL_FUNC_ATTR_KERNEL_SCHEMODE_PLACEHOLDER), &funcAttrScheModeValue);
  if (aclRet != ACL_SUCCESS) {
    SK_LOGE("Failed to query function attribute schemode, ret=%d", aclRet);
    return ScheModeState::NONE;
  }
  return ParseScheModeState(funcAttrScheModeValue);
}

ScheModeState GetScheModeFromKernelTask(aclmdlRITask kernelTask) {
  SK_LOGI("Query kernel task schemode begin, kernelTask=%p", kernelTask);
  aclError aclRet = ACL_SUCCESS;
  aclrtLaunchKernelAttrValue launchAttr;
  aclRet = aclmdlRIKernelTaskGetAttribute(
      kernelTask, static_cast<aclrtLaunchKernelAttrId>(ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE), &launchAttr);
  if (aclRet != ACL_SUCCESS) {
    SK_LOGE("Failed to get task launch attribute schemode, ret=%d", aclRet);
    return ScheModeState::NONE;
  }
  ScheModeState scheModeState = ScheModeState::NONE;
  scheModeState = ParseScheModeState(static_cast<int64_t>(launchAttr.schemMode));
  SK_LOGI("Query kernel task schemode end, kernelTask=%p, rawSchemMode=%ld, parsedState=%ld", kernelTask,
          static_cast<int64_t>(launchAttr.schemMode), static_cast<int64_t>(scheModeState));
  return scheModeState;
}

const SkBindMap &GetSkBindMap(aclrtBinHandle binHdl) {
  static SkAllBinMap allBinMap;
  static std::mutex allBinMapMutex;
  std::lock_guard<std::mutex> lock(allBinMapMutex);
  auto it = allBinMap.find(binHdl);
  if (it != allBinMap.end()) {
    return it->second;
  }
  return allBinMap.emplace(binHdl, InitSuperKernelBindMap(binHdl)).first->second;
}

template <SkNodeCoreType coreType>
bool InitSingleCoreFunc(const CoreFuncInitContext &ctx, aclrtBinHandle binHdl, void *binDevAddr,
                        uint32_t &validFuncNum) {
  std::string coreName = "";
  if (coreType == SkNodeCoreType::AIC) {
    coreName = "AIC";
  } else {
    coreName = "AIV";
  }
  constexpr uint32_t coreTypeId = static_cast<uint32_t>(coreType);  // 0 : aic, 1 : aiv
  uint64_t skFuncOffset = ctx.bindIt->second.sknlFuncs[ctx.splitIdx];
  ctx.info->funcAddr[coreTypeId] = skFuncOffset + (uint64_t)binDevAddr;
  ctx.info->funcOffset[coreTypeId] = skFuncOffset;  // Save offset for dfx info
  void *binHostAddr = nullptr;
  uint32_t binHostSize = 0;
  if (int ret = rtGetBinBuffer(binHdl, RT_BIN_HOST_ADDR, &binHostAddr, &binHostSize) != 0) {
    SK_LOGE("split[%zu] rtGetBinBuffer failed for %s, ret=%d", ctx.splitIdx, coreName.c_str(), ret);
    if (ctx.failReason != nullptr) {
      *ctx.failReason = BindmapFailReason::BIN_HOST_ADDR_GET_FAILED;
    }
    return false;
  }
  std::string symbolName = "";
  uint64_t funcSize = 0;
  std::string symbolBind = "";
  if (GetFuncSymbolInfo(binHdl, static_cast<const char *>(binHostAddr), binHostSize, skFuncOffset, symbolName, funcSize,
                        symbolBind)) {
    ctx.info->prefetchCnt[coreTypeId] = AlignUpAndClamp(funcSize, coreTypeId);
    ctx.info->symbolBind[coreTypeId] = symbolBind;
    SK_LOGI("split[%zu] %s symbol=%s, size=0x%lx, bind=%s", ctx.splitIdx, coreName.c_str(), symbolName.c_str(),
            funcSize, symbolBind.c_str());
  } else {
    ctx.info->prefetchCnt[coreTypeId] = coreTypeId == 0 ? 16 : 8;
    SK_LOGW("split[%zu] Failed to get %s symbol info, default prefetchCnt[%zu]=%u", ctx.splitIdx, coreName.c_str(),
            coreTypeId, ctx.info->prefetchCnt[coreTypeId]);
  }
  if (ctx.splitIdx > 0 && ctx.bindIt->second.sknlFuncs[ctx.splitIdx] == ctx.bindIt->second.sknlFuncs[0]) {
    SK_LOGI("InitSingleCoreFunc: split[%zu] %s function is not sk sub op", ctx.splitIdx, coreName.c_str());
  } else {
    validFuncNum++;
  }
  return true;
}

bool InitSingleSplitFunc(ResolvedFunctionInfo &info, size_t splitIdx, const SkBindMap &bindMap,
                         SkBindMap::iterator aicIt, SkBindMap::iterator aivIt, aclrtBinHandle binHdl, void *binDevAddr,
                         uint32_t &resolvedNum, BindmapFailReason &failReason) {
  bool res = false;
  uint32_t validFuncNum = 0;
  if (aicIt != bindMap.end()) {
    CoreFuncInitContext aicCtx = {&info, aicIt, splitIdx, &failReason};
    res |= InitSingleCoreFunc<SkNodeCoreType::AIC>(aicCtx, binHdl, binDevAddr, validFuncNum);
  }
  if (aivIt != bindMap.end()) {
    CoreFuncInitContext aivCtx = {&info, aivIt, splitIdx, &failReason};
    res |= InitSingleCoreFunc<SkNodeCoreType::AIV>(aivCtx, binHdl, binDevAddr, validFuncNum);
  }
  if (!res) {
    return false;
  }
  if (validFuncNum > 0) {
    resolvedNum++;
  }
  return true;
}

bool InitKernelResolvedFuncs(KernelInfos &kernelInfos) {
  aclrtBinHandle binHdl = kernelInfos.binHdl;
  aclrtFuncHandle oriFuncHdl = kernelInfos.funcHdl;
  if (binHdl == nullptr) {
    SK_LOGI("binHdl is null for kernel %s", kernelInfos.funcName.c_str());
    kernelInfos.bindmapFailReason = BindmapFailReason::BINHDL_NULL;
    return false;
  }
  if (oriFuncHdl == nullptr) {
    SK_LOGI("funcHdl is null for kernel %s", kernelInfos.funcName.c_str());
    kernelInfos.bindmapFailReason = BindmapFailReason::FUNCHDL_NULL;
    return false;
  }
  SkBindMap bindMap = GetSkBindMap(binHdl);
  if (bindMap.empty()) {
    SK_LOGI("bindMap is empty for kernel %s", kernelInfos.funcName.c_str());
    kernelInfos.bindmapFailReason = BindmapFailReason::BINDMAP_INIT_EMPTY;
    return false;
  }
  size_t binDevSize = 0;
  void *binDevAddr = nullptr;
  aclError aclRet = aclrtBinaryGetDevAddress(binHdl, &binDevAddr, &binDevSize);
  if (aclRet != ACL_SUCCESS) {
    SK_LOGE("Failed to get binary device address for kernel %s, ret=%d", kernelInfos.funcName.c_str(), aclRet);
    kernelInfos.bindmapFailReason = BindmapFailReason::BIN_DEV_ADDR_GET_FAILED;
    return false;
  }

  void *addr[2] = {nullptr, nullptr};  // {aic addr, aiv addr}
  aclRet = aclrtGetFunctionAddr(oriFuncHdl, addr, addr + 1);
  if (aclRet != ACL_SUCCESS) {
    SK_LOGE("Failed to get function address for kernel %s, ret=%d", kernelInfos.funcName.c_str(), aclRet);
    kernelInfos.bindmapFailReason = BindmapFailReason::FUNC_ADDR_GET_FAILED;
    return false;
  }

  uint64_t aicOffset = (uint64_t)addr[0] - (uint64_t)binDevAddr;
  uint64_t aivOffset = (uint64_t)addr[1] - (uint64_t)binDevAddr;
  SK_LOGI("funcName=%s, binDevAddr=0x%lx, binDevSize=%lu, aicAddr=0x%lx, aivAddr=0x%lx", kernelInfos.funcName.c_str(),
          (uint64_t)binDevAddr, binDevSize, (uint64_t)addr[0], (uint64_t)addr[1]);
  SK_LOGI("aicOffset=0x%lx, aivOffset=0x%lx", aicOffset, aivOffset);
  auto aicItor = bindMap.find(aicOffset);
  auto aivItor = bindMap.find(aivOffset);
  if (aicItor != bindMap.end() && HasInvalidSkBindValue(aicItor->second)) {
    SK_LOGI("Invalid sk bind map for globalFunc=0x%lx, kernel %s has duplicated entries with different values",
            aicItor->first, kernelInfos.funcName.c_str());
    kernelInfos.bindmapFailReason = BindmapFailReason::BINDMAP_ENTRY_CONFLICT;
    return false;
  }
  if (aivItor != bindMap.end() && HasInvalidSkBindValue(aivItor->second)) {
    SK_LOGI("Invalid sk bind map for globalFunc=0x%lx, kernel %s has duplicated entries with different values",
            aivItor->first, kernelInfos.funcName.c_str());
    kernelInfos.bindmapFailReason = BindmapFailReason::BINDMAP_ENTRY_CONFLICT;
    return false;
  }
  if (aicItor == bindMap.end() && aivItor == bindMap.end()) {
    SK_LOGI("Function is not found in sk bind map for kernel %s", kernelInfos.funcName.c_str());
    kernelInfos.bindmapFailReason = BindmapFailReason::FUNC_NOT_FOUND;
    return false;
  }
  bool hasCap = false;
  uint64_t cap = 0;
  if (aicItor != bindMap.end() && !UpdateKernelCap(aicItor->second, hasCap, cap)) {
    SK_LOGI("Invalid sk bind map for kernel %s, cap is inconsistent", kernelInfos.funcName.c_str());
    kernelInfos.bindmapFailReason = BindmapFailReason::BINDMAP_CAP_INCONSISTENT;
    return false;
  }
  if (aivItor != bindMap.end() && !UpdateKernelCap(aivItor->second, hasCap, cap)) {
    SK_LOGI("Invalid sk bind map for kernel %s, cap is inconsistent", kernelInfos.funcName.c_str());
    kernelInfos.bindmapFailReason = BindmapFailReason::BINDMAP_CAP_INCONSISTENT;
    return false;
  }
  kernelInfos.cap = hasCap ? cap : 0;
  const KernelCapBits capBits = ParseKernelCapBits(kernelInfos.cap);
  kernelInfos.capBits = capBits;
  SK_LOGI(
      "bindMap size=%lu, aicFound=%d, aivFound=%d, earlyStartWaitFlag=%d, "
      "earlyStartSetFlag=%d, disableDcci=%d, disableScheMode=%d, blockDimScaleUp=%d",
      bindMap.size(), aicItor != bindMap.end(), aivItor != bindMap.end(), capBits.earlyStartWaitFlag,
      capBits.earlyStartSetFlag, capBits.disableDcci, capBits.disableScheMode, capBits.blockDimScaleUp);
  if (capBits.disableScheMode) {
    const bool originScheModeOn = kernelInfos.isScheModeOn;
    kernelInfos.isScheModeOn = false;
    SK_LOGI(
        "Disable ScheMode by kernel cap, funcName=%s, cap=0x%lx, originIsScheModeOn=%d, "
        "currentIsScheModeOn=%d",
        kernelInfos.funcName.c_str(), kernelInfos.cap, originScheModeOn, kernelInfos.isScheModeOn);
  }
  kernelInfos.resolvedNum = 0;
  for (size_t i = 0; i < K_MAX_SPLIT_BIN_COUNT; ++i) {
    ResolvedFunctionInfo info{};
    BindmapFailReason failReason = BindmapFailReason::NONE;
    if (!InitSingleSplitFunc(info, i, bindMap, aicItor, aivItor, binHdl, binDevAddr, kernelInfos.resolvedNum,
                             failReason)) {
      SK_LOGW("Failed to initialize kernel function for SK node split[%zu]; fusion will be skipped", i);
      kernelInfos.bindmapFailReason = failReason;
      return false;
    }
    kernelInfos.resolvedFuncs[i] = info;
    SK_LOGI(
        "split[%zu] funcAddr[0]=0x%lx, funcAddr[1]=0x%lx, "
        "prefetchCnt[0]=0x%lx, prefetchCnt[1]=0x%lx, cap=%lu",
        i, info.funcAddr[0], info.funcAddr[1], info.prefetchCnt[0], info.prefetchCnt[1], kernelInfos.cap);
  }
  if (kernelInfos.resolvedNum == 2) {  // sk balance
    kernelInfos.resolvedFuncs[2] = kernelInfos.resolvedFuncs[0];
    kernelInfos.resolvedFuncs[3] = kernelInfos.resolvedFuncs[1];
  }
  return true;
}

SkKernelType NormalizeKernelType(uint32_t kernelType, const uint32_t taskRatio[2]) {
  switch (kernelType) {
    case ACL_KERNEL_TYPE_CUBE:
      return SkKernelType::AIC_ONLY;
    case ACL_KERNEL_TYPE_VECTOR:
      return SkKernelType::AIV_ONLY;
    case ACL_KERNEL_TYPE_MIX:
      if (taskRatio[1] == 0) {
        return SkKernelType::AIC_ONLY;
      }
      if (taskRatio[1] == 1) {
        return SkKernelType::MIX_AIC_1_1;
      }
      if (taskRatio[1] == 2) {
        return SkKernelType::MIX_AIC_1_2;
      }
      break;
    default:
      break;
  }
  return SkKernelType::DEFAULT;
}

bool IsMixKernelType(SkKernelType kernelType) {
  return kernelType == SkKernelType::MIX_AIC_1_1 || kernelType == SkKernelType::MIX_AIC_1_2;
}

constexpr char SK_SCOPE_KERNEL_SUFFIX_SEPARATOR = '_';

bool IsScopeKernelNameWithSupportedArch(const char *kernelName, const char *baseName) {
  const size_t baseLen = strlen(baseName);
  if (strncmp(kernelName, baseName, baseLen) != 0) {
    return false;
  }
  if (kernelName[baseLen] != SK_SCOPE_KERNEL_SUFFIX_SEPARATOR) {
    return false;
  }
  for (const auto arch : SK_SUPPORTED_KERNEL_ARCHS) {
    if (strcmp(kernelName + baseLen + 1, GetSkKernelArchSymbolSuffix(arch)) == 0) {
      return true;
    }
  }
  return false;
}

}  // namespace

// ============================================================================
// ToJson Helper Functions
// ============================================================================

static std::string PtrToHexString(const void *ptr) {
  std::stringstream hexStream;
  hexStream << "0x" << std::hex << reinterpret_cast<uintptr_t>(ptr);
  return hexStream.str();
}

static std::string UintToHexString(uint64_t value) {
  std::stringstream hexStream;
  hexStream << "0x" << std::hex << value;
  return hexStream.str();
}

Json KernelInfosToJson(const KernelInfos &kernelInfos) {
  Json kernelJson;
  kernelJson["funcName"] = kernelInfos.funcName;
  kernelJson["funcHandle"] = PtrToHexString(kernelInfos.funcHdl);
  kernelJson["numBlocks"] = kernelInfos.numBlocks;
  kernelJson["cap"] = UintToHexString(kernelInfos.cap);
  kernelJson["devargs"] = PtrToHexString(kernelInfos.devArgs);
  kernelJson["argsSize"] = 0;  // Will be filled by caller if available
  kernelJson["isHostArgs"] = false;
  kernelJson["launchKernelCfg"] = PtrToHexString(kernelInfos.launchKernelCfg);
  kernelJson["binHandle"] = PtrToHexString(kernelInfos.binHdl);
  kernelJson["kernelTypeInt"] = kernelInfos.kernelTypeInt;
  kernelJson["kernelType"] = to_string(kernelInfos.kernelType);
  kernelJson["needMixKernelSplit"] = kernelInfos.needMixKernelSplit;
  kernelJson["isScheModeOn"] = kernelInfos.isScheModeOn;
  kernelJson["isSimtOp"] = kernelInfos.isSimtOp;
  kernelJson["taskRatio"] = Json::array({kernelInfos.taskRatio[0], kernelInfos.taskRatio[1]});
  kernelJson["opInfoPtr"] = PtrToHexString(kernelInfos.opInfoPtr);
  kernelJson["opInfoSize"] = static_cast<uint64_t>(kernelInfos.opInfoSize);
  kernelJson["dynUbufSize"] = static_cast<uint64_t>(kernelInfos.dynUbufSize);
  kernelJson["allocUbufSize"] = static_cast<uint64_t>(kernelInfos.allocUbufSize);
  kernelJson["hasDynUbufSize"] = kernelInfos.hasDynUbufSize;
  kernelJson["hasAllocUbufSize"] = kernelInfos.hasAllocUbufSize;
  kernelJson["taskGrp"] = "0x0";
  kernelJson["resolvedNum"] = kernelInfos.resolvedNum;

  // Add resolved functions info
  Json resolvedFuncs = Json::array();
  for (size_t i = 0; i < kernelInfos.resolvedNum && i < K_MAX_SPLIT_BIN_COUNT; ++i) {
    Json rfJson;
    rfJson["funcAddr"][0] = UintToHexString(kernelInfos.resolvedFuncs[i].funcAddr[0]);
    rfJson["funcAddr"][1] = UintToHexString(kernelInfos.resolvedFuncs[i].funcAddr[1]);
    rfJson["prefetchCnt"][0] = kernelInfos.resolvedFuncs[i].prefetchCnt[0];
    rfJson["prefetchCnt"][1] = kernelInfos.resolvedFuncs[i].prefetchCnt[1];
    rfJson["funcOffset"][0] = UintToHexString(kernelInfos.resolvedFuncs[i].funcOffset[0]);
    rfJson["funcOffset"][1] = UintToHexString(kernelInfos.resolvedFuncs[i].funcOffset[1]);
    rfJson["symbolBind"][0] = kernelInfos.resolvedFuncs[i].symbolBind[0];
    rfJson["symbolBind"][1] = kernelInfos.resolvedFuncs[i].symbolBind[1];
    resolvedFuncs.push_back(rfJson);
  }
  kernelJson["resolvedFuncs"] = resolvedFuncs;

  return kernelJson;
}

Json SyncInfosToJson(const SyncInfos &syncInfos, SkNodeType nodeType) {
  Json syncJson;
  syncJson["eventId"] = UintToHexString(syncInfos.eventId);

  if (nodeType == SkNodeType::NODE_WAIT || nodeType == SkNodeType::NODE_MEMORY_WAIT) {
    syncJson["correspondingNotifyNodeId"] = syncInfos.correspondingNotifyNodeId;
  }

  syncJson["addrValue"] = PtrToHexString(syncInfos.addrValue);

  if (!syncInfos.correspondingWaitNodeIds.empty()) {
    syncJson["correspondingWaitNodeIds"] = syncInfos.correspondingWaitNodeIds;
  }
  if (!syncInfos.correspondingResetNodeIds.empty()) {
    syncJson["correspondingResetNodeIds"] = syncInfos.correspondingResetNodeIds;
  }
  if (syncInfos.memoryValue != std::numeric_limits<uint64_t>::max()) {
    syncJson["memoryValue"] = UintToHexString(syncInfos.memoryValue);
  }
  if (syncInfos.memoryWaitFlag != std::numeric_limits<uint32_t>::max()) {
    syncJson["memoryWaitFlag"] = syncInfos.memoryWaitFlag;
  }
  if (syncInfos.eventFlag != std::numeric_limits<uint32_t>::max()) {
    syncJson["eventFlag"] = UintToHexString(syncInfos.eventFlag);
  }
  if (syncInfos.eventTaskFlag != std::numeric_limits<uint32_t>::max()) {
    syncJson["eventTaskFlag"] = syncInfos.eventTaskFlag;
  }

  return syncJson;
}

Json NodeInfosToJson(const NodeInfos &nodeInfos, SkNodeType nodeType) {
  Json nodeInfosJson;
  nodeInfosJson["kernelInfos"] = KernelInfosToJson(nodeInfos.kernelInfos);
  if (nodeType != SkNodeType::NODE_KERNEL) {
    nodeInfosJson["syncInfos"] = SyncInfosToJson(nodeInfos.syncInfos, nodeType);
  }
  return nodeInfosJson;
}

/**
 * @brief Get kernel type string from kernelType value
 */
const char *GetKernelTypeString(uint32_t kernelType, const uint32_t taskRatio[2]) {
  // 直接复用你原来匿名空间里的逻辑！
  SkKernelType skType = NormalizeKernelType(kernelType, taskRatio);

  switch (skType) {
    case SkKernelType::AIC_ONLY:
      return "AIC_ONLY";
    case SkKernelType::AIV_ONLY:
      return "AIV_ONLY";
    case SkKernelType::MIX_AIC_1_1:
      return "MIX_AIC_1_1";
    case SkKernelType::MIX_AIC_1_2:
      return "MIX_AIC_1_2";
    default:
      return "DEFAULT";
  }
}

// Implementation of AlignUpAndClamp (declared in sk_node.h)
size_t AlignUpAndClamp(size_t value, size_t coreIdx) {
  constexpr size_t aicFuncMaxPrefetchCnt = 0x800 * 16;                 // 32k
  constexpr size_t aivFuncMaxPrefetchCnt = 0x800 * 8;                  // 16k
  constexpr size_t alignNum = 0x800;                                   // 2k
  size_t prefetchCntValue = (value + alignNum - 1) & ~(alignNum - 1);  // 以2k为单位向上取整
  if (coreIdx == 0 && prefetchCntValue > aicFuncMaxPrefetchCnt) {
    prefetchCntValue = aicFuncMaxPrefetchCnt;
  } else if (coreIdx == 1 && prefetchCntValue > aivFuncMaxPrefetchCnt) {
    prefetchCntValue = aivFuncMaxPrefetchCnt;
  }
  return prefetchCntValue / alignNum;
}

const char *SuperKernelBaseNode::GetUpdateTargetTypeName(aclmdlRITaskType type) const {
  switch (type) {
    case ACL_MODEL_RI_TASK_KERNEL:
      return "KERNEL";
    case ACL_MODEL_RI_TASK_VALUE_WRITE:
      return "VALUE_WRITE";
    case ACL_MODEL_RI_TASK_VALUE_WAIT:
      return "VALUE_WAIT";
    default:
      return "UNKNOWN";
  }
}

bool SuperKernelBaseNode::InitNode(const SuperKernelOptionsManager *opts) {
  (void)opts;
  if (originTask == nullptr) {
    SK_LOGE("Origin task is null for %s", Format().c_str());
    return false;
  }
  uint32_t seqId = 0;
  if (aclmdlRITaskGetSeqId(*originTask, &seqId) != ACL_SUCCESS) {
    SK_LOGE("Failed to get nodeId for %s", Format().c_str());
    return false;
  }
  nodeId = static_cast<uint64_t>(seqId);
  return true;
}

bool SuperKernelBaseNode::Update(const UpdateContext &ctx) {
  if (isUpdate) {
    SK_LOGE("Node has already been updated and cannot be updated again, %s", Format().c_str());
    return false;
  }

  isUpdate = true;
  SK_LOGI("Updating node %lu.", nodeId);
  return true;
}

void SuperKernelBaseNode::LogNodeUpdateResult(const aclmdlRITaskParams *paramsToLog) const {
  std::ostringstream oss;
  oss << "node update result: nodeId=" << nodeId;
  if (paramsToLog == nullptr) {
    oss << ", type=INVALID";
    SK_LOGI("%s", oss.str().c_str());
    return;
  }

  oss << ", type=" << GetUpdateTargetTypeName(paramsToLog->type);
  switch (paramsToLog->type) {
    case ACL_MODEL_RI_TASK_KERNEL:
      oss << ", opInfoPtr=" << paramsToLog->opInfoPtr << ", opInfoSize=" << paramsToLog->opInfoSize
          << ", funcHandle=" << paramsToLog->kernelTaskParams.funcHandle
          << ", args=" << paramsToLog->kernelTaskParams.args << ", argsSize=" << paramsToLog->kernelTaskParams.argsSize
          << ", numBlocks=" << static_cast<uint32_t>(paramsToLog->kernelTaskParams.numBlocks);
      break;
    case ACL_MODEL_RI_TASK_VALUE_WRITE:
      oss << ", addr=" << paramsToLog->valueWriteTaskParams.devAddr << ", value=0x" << std::hex
          << paramsToLog->valueWriteTaskParams.value << std::dec;
      break;
    case ACL_MODEL_RI_TASK_VALUE_WAIT:
      oss << ", addr=" << paramsToLog->valueWaitTaskParams.devAddr << ", value=0x" << std::hex
          << paramsToLog->valueWaitTaskParams.value << ", flag=0x" << paramsToLog->valueWaitTaskParams.flag << std::dec;
      break;
    default:
      break;
  }
  SK_LOGI("%s", oss.str().c_str());
}

aclError SuperKernelBaseNode::InValidateNode() {
  SK_LOGI("Invalidating node %lu for super kernel fusion.", nodeId);
  aclError aclRet = aclmdlRITaskDisable(*originTask);
  if (aclRet != ACL_SUCCESS) {
    SK_LOGE("Failed to invalidate node %s", Format().c_str());
    return aclRet;
  }
  isInvalidated = true;
  SK_LOGI("Node %lu was invalidated successfully.", nodeId);
  return ACL_SUCCESS;
}

struct JudgeTaskKernelInfo {
  bool isBegin = false;
  bool isEnd = false;
  bool isPlaceholder = false;
  bool isFuseEnable = true;
  std::unique_ptr<char[]> scopeName;
};

bool GetScopeKernelInfo(aclmdlRIKernelTaskParams params, JudgeTaskKernelInfo *info) {
  char kernelName[MAX_FUNC_NAME_LEN] = {0};
  int32_t ret = aclrtGetFunctionName(params.funcHandle, sizeof(kernelName), kernelName);
  if (ret != ACL_SUCCESS) {
    SK_LOGE("Failed to get kernel name for funcHandle, ret: %d", ret);
    return false;
  }
  info->isBegin = IsScopeKernelNameWithSupportedArch(kernelName, "sk_scope_kernel_begin");
  info->isEnd = IsScopeKernelNameWithSupportedArch(kernelName, "sk_scope_kernel_end");
  info->isPlaceholder = IsScopeKernelNameWithSupportedArch(kernelName, "sk_placeholder_kernel");
  if (!info->isBegin && !info->isEnd && !info->isPlaceholder) {
    SK_LOGD("Kernel task is not recognized as a valid scope kernel task, kernelName=%s", kernelName);
    return true;
  }
  auto parseArgsAddr = std::make_unique<ScopeKernelArgs>();
  ret = aclrtMemcpy((void *)parseArgsAddr.get(), sizeof(ScopeKernelArgs), params.args, sizeof(ScopeKernelArgs),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != ACL_SUCCESS) {
    SK_LOGE("Failed to copy kernel args from device to host, ret: %d, direction=DEVICE_TO_HOST", ret);
    return false;
  }
  parseArgsAddr->name[MAX_SCOPE_NAME_LEN - 1] = '\0';
  size_t nameLen = strlen(parseArgsAddr->name);
  info->scopeName = std::make_unique<char[]>(nameLen + 1);
  errno_t res = memcpy_s(info->scopeName.get(), nameLen + 1, parseArgsAddr->name, nameLen + 1);
  if (res != 0) {
    SK_LOGE("Failed to copy scope name '%s', memcpy_s error code: %d", parseArgsAddr->name, res);
    return false;
  }
  if (strcmp(info->scopeName.get(), DEFAULT_SK_SCOPE_NAME) == 0) {
    info->isFuseEnable = false;
  }
  SK_LOGI(
      "Success parse scope kernel task, kernelName: %s, scopeName: %s, isBegin: %d, isEnd: %d, "
      "isPlaceholder: %d, isFuseEnable: %d",
      kernelName, info->scopeName.get(), info->isBegin, info->isEnd, info->isPlaceholder, info->isFuseEnable);
  return true;
}

bool SuperKernelKernelNode::InitNode(const SuperKernelOptionsManager *opts) {
  if (!SuperKernelBaseNode::InitNode(opts)) {
    SK_LOGE("Failed to init kernel node for %s", Format().c_str());
    return false;
  }
  nodeType = SkNodeType::NODE_KERNEL;
  aclError aclRet = aclmdlRITaskGetParams(*originTask, &taskParams);
  if (aclRet != ACL_SUCCESS) {
    SK_LOGE("Failed to get kernel params for %s", Format().c_str());
    return false;
  }
  JudgeTaskKernelInfo scopeKernelInfo;
  auto &kernelParams = taskParams.kernelTaskParams;
  if (!GetScopeKernelInfo(kernelParams, &scopeKernelInfo)) {
    SK_LOGE("Failed to get scope kernel information for kernel node %lu", nodeId);
    return false;
  }
  if (scopeKernelInfo.isBegin || scopeKernelInfo.isEnd || scopeKernelInfo.isPlaceholder) {
    SK_LOGI("Kernel node %lu is a scope kernel node.", nodeId);
    isScopeNode = true;
    isFusible = scopeKernelInfo.isFuseEnable;
    isScopeBegin = scopeKernelInfo.isBegin;
    isScopeEnd = scopeKernelInfo.isEnd;
    isPlaceholder = scopeKernelInfo.isPlaceholder;
    if (isFusible && scopeKernelInfo.scopeName != nullptr) {
      char *rawPtr = scopeKernelInfo.scopeName.get();
      scopeName = std::string(rawPtr);
    }
  } else {
    SK_LOGD("Kernel node %lu is a regular kernel node.", nodeId);
  }

  int64_t kernelType = 0;
  aclRet = aclrtGetFunctionAttribute(kernelParams.funcHandle, ACL_FUNC_ATTR_KERNEL_TYPE, &kernelType);
  if (aclRet != ACL_SUCCESS) {
    SK_LOGE("Failed to get kernel type for node %s, ret=%d", Format().c_str(), aclRet);
    SetFusionFailReason(FusionFailReason::KERNEL_ATTR_GET_FAILED);
    return false;
  }

  int64_t taskRatio = 0;
  aclRet = aclrtGetFunctionAttribute(kernelParams.funcHandle, ACL_FUNC_ATTR_KERNEL_RATIO, &taskRatio);
  if (aclRet != ACL_SUCCESS) {
    SK_LOGE("Failed to get task ratio for node %s, ret=%d", Format().c_str(), aclRet);
    SetFusionFailReason(FusionFailReason::KERNEL_ATTR_GET_FAILED);
    return false;
  }

  const int16_t *taskRatioInt16 = reinterpret_cast<const int16_t *>(&taskRatio);
  uint32_t skTaskTatio[2] = {static_cast<uint32_t>(taskRatioInt16[1]), static_cast<uint32_t>(taskRatioInt16[0])};

  nodeInfos.kernelInfos.taskRatio[0] = skTaskTatio[0];
  nodeInfos.kernelInfos.taskRatio[1] = skTaskTatio[1];
  nodeInfos.kernelInfos.kernelType = NormalizeKernelType((uint32_t)(kernelType), skTaskTatio);
  nodeInfos.kernelInfos.kernelTypeInt = static_cast<uint32_t>(kernelType);
  nodeInfos.kernelInfos.numBlocks = kernelParams.numBlocks;
  nodeInfos.kernelInfos.devArgs = kernelParams.args;
  nodeInfos.kernelInfos.opInfoPtr = taskParams.opInfoPtr;
  nodeInfos.kernelInfos.opInfoSize = taskParams.opInfoSize;
  nodeInfos.kernelInfos.launchKernelCfg = kernelParams.cfg;
  nodeInfos.kernelInfos.isScheModeOn = GetScheMode();
  aclRet = aclrtFunctionGetBinary(kernelParams.funcHandle, &nodeInfos.kernelInfos.binHdl);
  if (aclRet != ACL_SUCCESS) {
    SK_LOGE("Failed to get kernel bin handle for %s, ret=%d", Format().c_str(), aclRet);
    return false;
  }
  nodeInfos.kernelInfos.funcHdl = kernelParams.funcHandle;

  // Calculate vecNum and cubeNum based on kernel type and numBlocks
  uint32_t numBlocks = kernelParams.numBlocks;
  SkKernelType kt = nodeInfos.kernelInfos.kernelType;
  if (kt == SkKernelType::AIC_ONLY || kt == SkKernelType::MIX_AIC_1_0) {
    nodeInfos.kernelInfos.cubeNum = numBlocks;
    nodeInfos.kernelInfos.vecNum = 0;
  } else if (kt == SkKernelType::AIV_ONLY || kt == SkKernelType::MIX_AIV_1_0) {
    nodeInfos.kernelInfos.cubeNum = 0;
    nodeInfos.kernelInfos.vecNum = numBlocks;
  } else if (kt == SkKernelType::MIX_AIC_1_1) {
    nodeInfos.kernelInfos.cubeNum = numBlocks;
    nodeInfos.kernelInfos.vecNum = numBlocks;
  } else if (kt == SkKernelType::MIX_AIC_1_2) {
    nodeInfos.kernelInfos.cubeNum = numBlocks;
    nodeInfos.kernelInfos.vecNum = numBlocks << 1;
  }

  char tmpFuncName[MAX_FUNC_NAME_LEN] = {0};
  aclRet = aclrtGetFunctionName(kernelParams.funcHandle, sizeof(tmpFuncName), tmpFuncName);
  if (aclRet != ACL_SUCCESS) {
    SK_LOGE("Failed to get function name for node %s, ret=%d", Format().c_str(), aclRet);
    SetFusionFailReason(FusionFailReason::KERNEL_ATTR_GET_FAILED);
    return false;
  }
  nodeInfos.kernelInfos.funcName = std::string(tmpFuncName);
  nodeInfos.kernelInfos.needMixKernelSplit = IsMixKernelType(nodeInfos.kernelInfos.kernelType);
  const auto *ubufLockIgnoreKernelOpt =
      opts == nullptr ? nullptr : opts->GetOption(aclskOptionType::UBUF_LOCK_IGNORE_KERNEL);
  if (nodeInfos.kernelInfos.needMixKernelSplit && ubufLockIgnoreKernelOpt != nullptr &&
      !nodeInfos.kernelInfos.funcName.empty() &&
      opts->JudgeUbufLockIgnoreKernel(ubufLockIgnoreKernelOpt->GetStringListValue(), nodeInfos.kernelInfos.funcName)) {
    nodeInfos.kernelInfos.needMixKernelSplit = false;
  }
  SK_LOGI("Kernel node %lu mix split flag initialized, funcName=%s, kernelType=%s, needMixKernelSplit=%d", nodeId,
          nodeInfos.kernelInfos.funcName.c_str(), to_string(nodeInfos.kernelInfos.kernelType),
          static_cast<int>(nodeInfos.kernelInfos.needMixKernelSplit));
  if (!isScopeNode && !nodeInfos.kernelInfos.funcName.empty() && nodeInfos.kernelInfos.binHdl != nullptr) {
    isFusible = InitKernelResolvedFuncs(nodeInfos.kernelInfos);
    if (!isFusible) {
      SetFusionFailReason(nodeInfos.kernelInfos.bindmapFailReason);
    }
  }

  // SIMT kernels need dynamic ubuf size propagated to the SK launch cfg.
  IdentifyAndHandleSimtKernel(opts);

  if (taskParams.taskGrp != nullptr) {
    SK_LOGI("Kernel node %lu has a non-null task group and cannot be fused in super kernel.", nodeId);
    isFusible = false;
    SetFusionFailReason(FusionFailReason::DYNAMIC_TASK_UNSUPPORT);
  }

  return true;
}

bool SuperKernelKernelNode::GetScheMode() const {
  const aclmdlRIKernelTaskParams &kernelParams = taskParams.kernelTaskParams;
  const ScheModeState funcAttrScheModeState = GetScheModeFromFuncAttr(kernelParams.funcHandle);
  const ScheModeState launchAttrScheModeState = GetScheModeFromKernelTask(*originTask);

  ScheModeState finalScheModeState = ScheModeState::SCHE_MODE_OFF;
  if (launchAttrScheModeState != ScheModeState::NONE) {
    finalScheModeState = launchAttrScheModeState;
  } else if (funcAttrScheModeState != ScheModeState::NONE) {
    finalScheModeState = funcAttrScheModeState;
  }

  SK_LOGI("schemode detect result: funcAttrState=%ld, launchAttrState=%ld, finalState=%ld",
          static_cast<int64_t>(funcAttrScheModeState), static_cast<int64_t>(launchAttrScheModeState),
          static_cast<int64_t>(finalScheModeState));
  return finalScheModeState == ScheModeState::SCHE_MODE_ON;
}

void SuperKernelKernelNode::IdentifyAndHandleSimtKernel(const SuperKernelOptionsManager *opts) {
  nodeInfos.kernelInfos.isSimtOp = false;
  nodeInfos.kernelInfos.hasDynUbufSize = false;
  nodeInfos.kernelInfos.hasAllocUbufSize = false;
  nodeInfos.kernelInfos.dynUbufSize = 0;
  nodeInfos.kernelInfos.allocUbufSize = 0;
  if (opts == nullptr || !opts->IsInnerOptionEnabled(SkInnerOptionType::SIMT_OP_SUPPORT)) {
    return;
  }
  SkKernelType kernelType = nodeInfos.kernelInfos.kernelType;
  bool hasAivSection = (kernelType == SkKernelType::AIV_ONLY || kernelType == SkKernelType::MIX_AIV_1_0 ||
                        kernelType == SkKernelType::MIX_AIC_1_1 || kernelType == SkKernelType::MIX_AIC_1_2);
  if (!hasAivSection) {
    SK_LOGI("IdentifyAndHandleSimtKernel: %s has no AIV section (kernelType=%s), skip SIMT analysis", Format().c_str(),
            to_string(kernelType));
    return;
  }
  SK_LOGI("IdentifyAndHandleSimtKernel: analyzing %s, kernelType=%s, nodeId=%lu", Format().c_str(),
          to_string(kernelType), nodeId);
  uint32_t aivType = 0;
  rtError_t ret = rtFunctionGetMetaInfo(taskParams.kernelTaskParams.funcHandle, RT_FUNCTION_TYPE_AIV_TYPE_FLAG,
                                        &aivType, sizeof(uint32_t));
  if (ret != RT_ERROR_NONE) {
    SK_LOGD("rtFunctionGetMetaInfo AIV_TYPE_FLAG failed for %s, ret=%d", Format().c_str(), ret);
    return;
  }
  bool isSimt = (aivType == AIV_TYPE_SIMT_VF_ONLY || aivType == AIV_TYPE_SIMD_SIMT_MIX_VF);
  if (isSimt) {
    nodeInfos.kernelInfos.isSimtOp = true;
    size_t dynUbufSize = 0;
    aclError aclRet = aclrtFunctionGetAvailDynUbufPerBlock(taskParams.kernelTaskParams.funcHandle, 0, &dynUbufSize);
    if (aclRet != ACL_SUCCESS) {
      SK_LOGE("Failed to get available dynamic ubuf size for %s, ret=%d", Format().c_str(), aclRet);
      isFusible = false;
      SetFusionFailReason(FusionFailReason::KERNEL_ATTR_GET_FAILED);
      return;
    }
    size_t allocUbufSize = 0;
    if (!GetFunctionAllocUbufSize(taskParams.kernelTaskParams.funcHandle, allocUbufSize, Format())) {
      isFusible = false;
      SetFusionFailReason(FusionFailReason::KERNEL_ATTR_GET_FAILED);
      return;
    }
    nodeInfos.kernelInfos.dynUbufSize = dynUbufSize;
    nodeInfos.kernelInfos.allocUbufSize = allocUbufSize;
    nodeInfos.kernelInfos.hasDynUbufSize = true;
    nodeInfos.kernelInfos.hasAllocUbufSize = true;
    SK_LOGI("%s is SIMT type, aivType=%u, dynUbufSize=%zu, allocUbufSize=%zu", Format().c_str(), aivType, dynUbufSize,
            allocUbufSize);
  }
  return;
}

bool SuperKernelKernelNode::SetupLaunchKernelCfg(const SkLaunchInfo &launchInfo) {
  launchKernelAttrs_.clear();
  size_t skEntryDynUbufSize = 0;

  if (launchInfo.isScheModeOn) {
    aclrtLaunchKernelAttr schemAttr{};
    schemAttr.id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
    schemAttr.value.schemMode = 1;
    launchKernelAttrs_.push_back(schemAttr);
  }

  if (launchInfo.useSimtEntry) {
    aclrtLaunchKernelAttr dynUbufAttr{};
    dynUbufAttr.id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;
    const size_t skMaxDcacheSize = launchInfo.skMaxDcacheSize;
    size_t skAllocUbufSize = 0;
    if (!GetFunctionAllocUbufSize(launchInfo.entryInfo.skEntryFunc, skAllocUbufSize, Format())) {
      return false;
    }
    if (skMaxDcacheSize > SK_TOTAL_UB_SIZE || skAllocUbufSize > SK_TOTAL_UB_SIZE - skMaxDcacheSize) {
      SK_LOGE(
          "invalid dyn ubuf calculation for %s, totalUbSize=%zu, skMaxDcacheSize=%zu, "
          "skAllocUbufSize=%zu",
          Format().c_str(), SK_TOTAL_UB_SIZE, skMaxDcacheSize, skAllocUbufSize);
      return false;
    }
    skEntryDynUbufSize = SK_TOTAL_UB_SIZE - skMaxDcacheSize - skAllocUbufSize;
    dynUbufAttr.value.dynUBufSize = static_cast<uint32_t>(skEntryDynUbufSize);
    launchKernelAttrs_.push_back(dynUbufAttr);
  }

  launchKernelCfg_.attrs = launchKernelAttrs_.data();
  launchKernelCfg_.numAttrs = launchKernelAttrs_.size();
  SK_LOGI("Set SK launch cfg: nodeId=%lu, isScheModeOn=%d, useSimtEntry=%d, skEntryDynUbufSize=%zu, attrCount=%zu",
          nodeId, launchInfo.isScheModeOn, launchInfo.useSimtEntry, skEntryDynUbufSize, launchKernelCfg_.numAttrs);
  return true;
}

std::string SuperKernelKernelNode::Format() const {
  std::ostringstream oss;
  oss << "[nodeId:" << nodeId << ", streamId:" << streamId << ", streamIdxInGraph:" << streamIdxInGraph
      << ", nodeIdxInStream:" << nodeIdxInStream << "] - " << nodeInfos.kernelInfos.Format();
  return oss.str();
}

std::string KernelInfos::Format() const {
  std::ostringstream oss;
  oss << "KernelInfos{funcName:" << funcName << ", kernelType:" << to_string(kernelType) << ", taskRatio:["
      << taskRatio[0] << "," << taskRatio[1] << "]"
      << ", cap:" << cap << ", numBlocks:" << numBlocks << ", cubeNum:" << cubeNum << ", vecNum:" << vecNum
      << ", isScheModeOn:" << isScheModeOn << ", needMixKernelSplit:" << needMixKernelSplit;
  if (isSimtOp) {
    oss << ", isSimtOp:" << isSimtOp;
  }
  if (hasDynUbufSize) {
    oss << ", dynUbufSize:" << dynUbufSize;
  }
  if (hasAllocUbufSize) {
    oss << ", allocUbufSize:" << allocUbufSize;
  }
  oss << ", resolvedNum:" << resolvedNum;
  if (binHdl != nullptr) {
    oss << ", binHdl:0x" << std::hex << reinterpret_cast<uintptr_t>(binHdl) << std::dec;
  }
  if (funcHdl != nullptr) {
    oss << ", funcHdl:0x" << std::hex << reinterpret_cast<uintptr_t>(funcHdl) << std::dec;
  }
  if (launchKernelCfg != nullptr) {
    oss << ", launchKernelCfg:0x" << std::hex << reinterpret_cast<uintptr_t>(launchKernelCfg) << std::dec;
  }
  if (devArgs != nullptr) {
    oss << ", devArgs:0x" << std::hex << reinterpret_cast<uintptr_t>(devArgs) << std::dec;
  }
  oss << "}";
  return oss.str();
}

bool SuperKernelKernelNode::Update(const UpdateContext &ctx) {
  if (!SuperKernelBaseNode::Update(ctx)) {
    SK_LOGE("Failed to update base node for %s", Format().c_str());
    return false;
  }
  bool hasUpdateParams = false;

  if (ctx.customParams != nullptr && ctx.customParams->type != 0) {
    // check update value
    switch (ctx.customParams->type) {
      case ACL_MODEL_RI_TASK_VALUE_WRITE:
      case ACL_MODEL_RI_TASK_VALUE_WAIT:
        if (ctx.customParams->valueWriteTaskParams.devAddr == nullptr) {
          SK_LOGE("Custom params for kernel node %s has null devAddr, invalid params.", Format().c_str());
          return false;
        }
        break;
      default:
        SK_LOGI("custom param type : %u not in check list, which will direct update, %s", ctx.customParams->type,
                Format().c_str());
        break;
    }
    updateParams = *ctx.customParams;
    // update kernel with custom params for stream sync
    aclError aclRet = aclmdlRITaskSetParams(*originTask, &updateParams);
    if (aclRet != ACL_SUCCESS) {
      SK_LOGE("Failed to set kernel with custom params for %s", Format().c_str());
      return false;
    }
    hasUpdateParams = true;
  } else if (ctx.launchInfo != nullptr && ctx.launchInfo->entryInfo.skEntryFunc != nullptr) {
    updateParams.type = ACL_MODEL_RI_TASK_KERNEL;
    updateParams.opInfoPtr = ctx.launchInfo->cacheInfo;
    updateParams.opInfoSize = ctx.launchInfo->cacheopInfoSize;
    updateParams.kernelTaskParams.args = static_cast<void *>(ctx.launchInfo->devArgs.Get());
    updateParams.kernelTaskParams.argsSize = ctx.launchInfo->devArgs.Get()->skHeader.totalSize;
    updateParams.kernelTaskParams.isHostArgs = true;
    updateParams.kernelTaskParams.funcHandle = ctx.launchInfo->entryInfo.skEntryFunc;
    updateParams.kernelTaskParams.numBlocks = ctx.launchInfo->entryInfo.numBlocks;
    if (!SetupLaunchKernelCfg(*ctx.launchInfo)) {
      SK_LOGE("Failed to setup SK launch cfg for kernel node %s", Format().c_str());
      return false;
    }
    updateParams.kernelTaskParams.cfg = launchKernelCfg_.numAttrs == 0 ? nullptr : &launchKernelCfg_;

    aclError aclRet = aclmdlRITaskSetParams(*originTask, &updateParams);

    if (aclRet != ACL_SUCCESS) {
      SK_LOGE("Failed to update kernel node %s", Format().c_str());
      return false;
    }
    hasUpdateParams = true;
  } else {
    aclError aclRet = InValidateNode();
    if (aclRet != ACL_SUCCESS) {
      return false;
    }
  }

  LogNodeUpdateResult(hasUpdateParams ? &updateParams : nullptr);
  return true;
}

bool SuperKernelMemoryNode::InitNode(const SuperKernelOptionsManager *opts) {
  if (!SuperKernelBaseNode::InitNode(opts)) {
    SK_LOGE("Failed to init memory node for %s", Format().c_str());
    return false;
  }

  aclError aclRet = aclmdlRITaskGetParams(*originTask, &taskParams);
  if (aclRet != ACL_SUCCESS) {
    SK_LOGE("Failed to get task params (aclRet=%d) for %s", aclRet, Format().c_str());
    return false;
  }

  // Handle event-based synchronization nodes
  if (rtNodeType != ACL_MODEL_RI_TASK_VALUE_WRITE && rtNodeType != ACL_MODEL_RI_TASK_VALUE_WAIT) {
    switch (rtNodeType) {
      case ACL_MODEL_RI_TASK_EVENT_RECORD: {
        const auto &eventParam = taskParams.eventRecordTaskParams;
        nodeType = SkNodeType::NODE_NOTIFY;
        nodeInfos.syncInfos.eventId = reinterpret_cast<uintptr_t>(eventParam.event);
        nodeInfos.syncInfos.eventFlag = eventParam.eventFlag;
        nodeInfos.syncInfos.eventTaskFlag = eventParam.recordFlag;
        nodeInfos.syncInfos.memoryValue = SK_DEFAULT_NOTIFY_VALUE;
        nodeInfos.syncInfos.memoryWaitFlag = SK_DEFAULT_WRITE_FLAG;
        isFusible = nodeInfos.syncInfos.eventTaskFlag == ACL_EVENT_RECORD_DEFAULT;
        break;
      }
      case ACL_MODEL_RI_TASK_EVENT_WAIT: {
        const auto &eventParam = taskParams.eventWaitTaskParams;
        nodeType = SkNodeType::NODE_WAIT;
        nodeInfos.syncInfos.eventId = reinterpret_cast<uintptr_t>(eventParam.event);
        nodeInfos.syncInfos.eventFlag = eventParam.eventFlag;
        nodeInfos.syncInfos.eventTaskFlag = eventParam.waitFlag;
        nodeInfos.syncInfos.memoryValue = SK_DEFAULT_WAIT_VALUE;
        nodeInfos.syncInfos.memoryWaitFlag = static_cast<uint32_t>(SkMemoryWaitFlag::EQ);
        isFusible = nodeInfos.syncInfos.eventTaskFlag == ACL_EVENT_WAIT_DEFAULT;
        break;
      }
      case ACL_MODEL_RI_TASK_EVENT_RESET: {
        const auto &eventParam = taskParams.eventResetTaskParams;
        nodeType = SkNodeType::NODE_RESET;
        nodeInfos.syncInfos.eventId = reinterpret_cast<uintptr_t>(eventParam.event);
        nodeInfos.syncInfos.eventFlag = eventParam.eventFlag;
        nodeInfos.syncInfos.eventTaskFlag = eventParam.resetFlag;
        nodeInfos.syncInfos.memoryValue = SK_DEFAULT_RESET_VALUE;
        nodeInfos.syncInfos.memoryWaitFlag = SK_DEFAULT_WRITE_FLAG;
        isFusible = false;
        SetFusionFailReason(FusionFailReason::RESET_TYPE_NODE);
        SK_LOGI("Event %s: is reset type, cannot be fused in super kernel", Format().c_str());
        return true;
      }
      default:
        isFusible = false;
        SK_LOGE("Unsupported event type %u for %s, which cannot be fused in super kernel.", rtNodeType,
                Format().c_str());
        SetFusionFailReason(FusionFailReason::UNSUPPORT_EVENT_TYPE);
        return false;
    }

    // Check internal (not external)
    if (!isFusible || (nodeInfos.syncInfos.eventFlag & ACL_EVENT_EXTERNAL)) {
      isFusible = false;
      SetFusionFailReason(FusionFailReason::EXTERNAL_DEPEND);
      SK_LOGI("Event %s: has external dependencies or is reset, cannot be fused in super kernel", Format().c_str());
    } else {
      SK_LOGI("Event %s: internal to ModelRI, fusible in super kernel", Format().c_str());
    }

    return true;
  }

  // Handle memory-based synchronization nodes
  if (rtNodeType == ACL_MODEL_RI_TASK_VALUE_WRITE) {
    const auto &memoryParam = taskParams.valueWriteTaskParams;
    nodeType = SkNodeType::NODE_MEMORY_WRITE;
    nodeInfos.syncInfos.eventId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(memoryParam.devAddr));
    nodeInfos.syncInfos.addrValue = memoryParam.devAddr;
    nodeInfos.syncInfos.memoryValue = memoryParam.value;
  } else {
    const auto &memoryParam = taskParams.valueWaitTaskParams;
    nodeType = SkNodeType::NODE_MEMORY_WAIT;
    nodeInfos.syncInfos.eventId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(memoryParam.devAddr));
    nodeInfos.syncInfos.addrValue = memoryParam.devAddr;
    nodeInfos.syncInfos.memoryValue = memoryParam.value;
    nodeInfos.syncInfos.memoryWaitFlag = memoryParam.flag;
  }

  if (nodeInfos.syncInfos.addrValue == nullptr) {
    SK_LOGE("Memory node %s has null device address, which is invalid for super kernel fusion.", Format().c_str());
    return false;
  }
  SK_LOGI("Memory node %s default not fusible, but it may be bypassed", Format().c_str());

  return true;
}

bool SuperKernelMemoryNode::Update(const UpdateContext &ctx) {
  if (!SuperKernelBaseNode::Update(ctx)) {
    SK_LOGE("Failed to update base node for %s", Format().c_str());
    return false;
  }
  aclmdlRITaskParams updateParams{};
  bool hasUpdateParams = false;

  if (ctx.customParams != nullptr && ctx.customParams->type != 0) {
    // check update value
    switch (ctx.customParams->type) {
      case ACL_MODEL_RI_TASK_VALUE_WRITE:
      case ACL_MODEL_RI_TASK_VALUE_WAIT:
        if (ctx.customParams->valueWriteTaskParams.devAddr == nullptr) {
          SK_LOGE("Custom params for memory node %s has null devAddr, invalid params.", Format().c_str());
          return false;
        }
        break;
      default:
        SK_LOGI("custom param type : %u not in check list, which will direct update, %s", ctx.customParams->type,
                Format().c_str());
        break;
    }
    updateParams = *ctx.customParams;
    // update memory node with custom params for stream sync
    aclError aclRet = aclmdlRITaskSetParams(*originTask, &updateParams);
    if (aclRet != ACL_SUCCESS) {
      SK_LOGE("Failed to set custom params on memory node %s", Format().c_str());
      return false;
    }
    // Sync taskParams for JSON dump
    taskParams = updateParams;
    hasUpdateParams = true;
  } else {
    aclError aclRet = InValidateNode();
    if (aclRet != ACL_SUCCESS) {
      return false;
    }
  }

  LogNodeUpdateResult(hasUpdateParams ? &updateParams : nullptr);
  return true;
}

std::string SuperKernelMemoryNode::Format() const {
  std::ostringstream oss;
  const char *eventType = nullptr;
  switch (rtNodeType) {
    case ACL_MODEL_RI_TASK_EVENT_RECORD:
      eventType = "EventNotify";
      break;
    case ACL_MODEL_RI_TASK_EVENT_WAIT:
      eventType = "EventWait";
      break;
    case ACL_MODEL_RI_TASK_EVENT_RESET:
      eventType = "EventReset";
      break;
    case ACL_MODEL_RI_TASK_VALUE_WRITE:
      oss << "[nodeId:" << nodeId << ", streamId:" << streamId << ", streamIdxInGraph:" << streamIdxInGraph
          << ", nodeIdxInStream:" << nodeIdxInStream << ", MemoryWrite(value:0x" << std::hex
          << nodeInfos.syncInfos.memoryValue << std::dec << ", eventId:0x" << std::hex << GetEventId() << std::dec
          << ")]";
      return oss.str();
    case ACL_MODEL_RI_TASK_VALUE_WAIT:
      oss << "[nodeId:" << nodeId << ", streamId:" << streamId << ", streamIdxInGraph:" << streamIdxInGraph
          << ", nodeIdxInStream:" << nodeIdxInStream << ", MemoryWait(flag:0x" << std::hex
          << nodeInfos.syncInfos.memoryWaitFlag << ", value:0x" << std::hex << nodeInfos.syncInfos.memoryValue
          << std::dec << ", eventId:0x" << std::hex << GetEventId() << std::dec << ")]";
      return oss.str();
    default:
      eventType = "Unknown";
      break;
  }
  uint64_t eventId = GetEventId();
  uint32_t eventFlag = nodeInfos.syncInfos.eventFlag;
  uint32_t eventTaskFlag = nodeInfos.syncInfos.eventTaskFlag;

  oss << "[nodeId:" << nodeId << ", streamId:" << streamId << ", streamIdxInGraph:" << streamIdxInGraph
      << ", nodeIdxInStream:" << nodeIdxInStream << ", " << eventType << "(eventId:0x" << std::hex << eventId
      << ", eventFlag:0x" << eventFlag << ", eventTaskFlag:0x" << eventTaskFlag << std::dec << ")]";

  return oss.str();
}

bool SuperKernelDefaultNode::InitNode(const SuperKernelOptionsManager *opts) {
  if (!SuperKernelBaseNode::InitNode(opts)) {
    SK_LOGE("Failed to init default node for %s", Format().c_str());
    return false;
  }
  nodeType = SkNodeType::NODE_DEFAULT;
  SK_LOGI("Default node %lu cannot be fused in super kernel.", nodeId);
  return true;
}

aclError SuperKernelDefaultNode::InValidateNode() {
  SK_LOGE("Default node %s should not be invalidated.", Format().c_str());
  return ACL_ERROR_FAILURE;
}

std::string SuperKernelDefaultNode::Format() const {
  std::ostringstream oss;
  oss << "[nodeId:" << nodeId << ", streamId:" << streamId << ", streamIdxInGraph:" << streamIdxInGraph
      << ", nodeIdxInStream:" << nodeIdxInStream << ", type: Default]";
  return oss.str();
}

static int TaskTypeToInt(aclmdlRITaskType type) {
  return static_cast<int>(type);
}

Json SuperKernelBaseNodeToJson(const SuperKernelBaseNode *node) {
  Json nodeJson;
  if (node == nullptr) {
    return nodeJson;
  }

  // Basic node info - matching sk_raw_tasks_after.json format
  nodeJson["taskId"] = node->GetNodeId();
  nodeJson["streamId"] = node->GetStreamId();

  aclmdlRITaskType taskType = node->GetTaskParams().type;

  nodeJson["taskType"] = TaskTypeToString(taskType);
  nodeJson["taskTypeInt"] = TaskTypeToInt(taskType);

  return nodeJson;
}

Json SuperKernelKernelNodeToJson(const SuperKernelKernelNode *node) {
  Json nodeJson = SuperKernelBaseNodeToJson(node);
  if (node == nullptr) {
    return nodeJson;
  }

  // Add kernel-specific params matching sk_raw_tasks_after.json format
  const auto &kernelInfos = node->GetNodeInfos().kernelInfos;
  Json kernelParams = KernelInfosToJson(kernelInfos);

  // Get argsSize and isHostArgs from taskParams
  const auto &taskParams = node->GetTaskParams();
  kernelParams["argsSize"] = taskParams.kernelTaskParams.argsSize;
  kernelParams["isHostArgs"] = taskParams.kernelTaskParams.isHostArgs;

  nodeJson["kernelParams"] = kernelParams;

  return nodeJson;
}

Json SuperKernelMemoryNodeToJson(const SuperKernelMemoryNode *node) {
  Json nodeJson = SuperKernelBaseNodeToJson(node);
  if (node == nullptr) {
    return nodeJson;
  }

  const auto &syncInfos = node->GetNodeInfos().syncInfos;
  SkNodeType nodeType = node->GetNodeType();

  // Add type-specific params matching sk_raw_tasks_after.json format
  switch (nodeType) {
    case SkNodeType::NODE_NOTIFY:
    case SkNodeType::NODE_WAIT:
    case SkNodeType::NODE_RESET: {
      Json eventParams;
      eventParams["eventId"] = PtrToHexString(reinterpret_cast<const void *>(syncInfos.eventId));
      eventParams["eventFlag"] = syncInfos.eventFlag;
      eventParams["eventTaskFlag"] = syncInfos.eventTaskFlag;
      nodeJson["eventParams"] = eventParams;
      break;
    }
    case SkNodeType::NODE_MEMORY_WRITE: {
      Json valueParams;
      valueParams["devAddr"] = PtrToHexString(syncInfos.addrValue);
      valueParams["value"] = UintToHexString(syncInfos.memoryValue);
      nodeJson["valueWriteParams"] = valueParams;
      break;
    }
    case SkNodeType::NODE_MEMORY_WAIT: {
      Json valueParams;
      valueParams["devAddr"] = PtrToHexString(syncInfos.addrValue);
      valueParams["value"] = UintToHexString(syncInfos.memoryValue);
      valueParams["flag"] = syncInfos.memoryWaitFlag;
      nodeJson["valueWaitParams"] = valueParams;
      break;
    }
    default:
      break;
  }

  return nodeJson;
}

Json SuperKernelDefaultNodeToJson(const SuperKernelDefaultNode *node) {
  return SuperKernelBaseNodeToJson(node);
}

// ============================================================================
// Kernel Binary Dump Functions
// ============================================================================

// Forward declarations
static uint32_t DumpKernelBinariesToDir(const SuperKernelGraph &graph, const std::string &kernelBinsDir);
static uint32_t DumpSkEntryBinary(const std::string &kernelBinsDir);

bool DumpKernelBinaries(const SuperKernelGraph &graph, const std::string &binPath) {
  if (!sk::logger::FileLogger::Instance().IsEnabled()) {
    return true;  // Kernel meta save is disabled, skip
  }
  SK_LOGI("Starting to dump kernel binaries to: %s", binPath.c_str());

  // binPath is the base directory, create bin_files subdirectory under it
  std::string baseDir = binPath.empty() ? "." : binPath;
  std::string kernelBinsDir = baseDir + "/bin_files";

  // Create kernel_bins directory
  if (!CreateDirectoryRecursive(kernelBinsDir)) {
    SK_LOGE("Failed to create kernel binaries directory: %s", kernelBinsDir.c_str());
    return false;
  }

  // Dump kernel binaries and SK entry binary
  uint32_t kernelCount = DumpKernelBinariesToDir(graph, kernelBinsDir);
  kernelCount += DumpSkEntryBinary(kernelBinsDir);

  SK_LOGI("Successfully dumped %u kernel binaries to directory: %s", kernelCount, kernelBinsDir.c_str());
  return true;
}

/**
 * @brief Dump a single kernel binary to file
 * @param kernelInfo Kernel information containing binary handle
 * @param kernelBinsDir Output directory path
 * @return true if dumped successfully, false otherwise
 */
bool DumpSingleKernelBinary(const KernelInfos &kernelInfo, const std::string &kernelBinsDir) {
  // Get binary buffer from runtime
  void *binHostAddr = nullptr;
  uint32_t binHostSize = 0;
  int rtRet = rtGetBinBuffer(kernelInfo.binHdl, RT_BIN_HOST_ADDR, &binHostAddr, &binHostSize);
  if (rtRet != 0 || binHostAddr == nullptr || binHostSize == 0) {
    SK_LOGW("Failed to get bin buffer for kernel %s, rtRet=%d, addr=%p, size=%u", kernelInfo.funcName.c_str(), rtRet,
            binHostAddr, binHostSize);
    return false;
  }

  // Get binary device address (code segment start address)
  void *binDevAddr = nullptr;
  size_t binDevSize = 0;
  rtRet = aclrtBinaryGetDevAddress(kernelInfo.binHdl, &binDevAddr, &binDevSize);
  if (rtRet != 0) {
    SK_LOGW("Failed to get bin dev address for kernel %s, rtRet=%d", kernelInfo.funcName.c_str(), rtRet);
    return false;
  }
  uint64_t codeSegmentAddr = reinterpret_cast<uint64_t>(binDevAddr);

  // Generate safe filename from kernel function name
  std::string kernelName = SanitizePathComponent(kernelInfo.funcName);
  std::ostringstream addrOss;
  addrOss << "0x" << std::hex << std::uppercase << codeSegmentAddr;
  std::string oFilePath = kernelBinsDir + "/" + kernelName + "_" + addrOss.str() + ".o";

  // Write binary to file
  std::ofstream outFile(oFilePath, std::ios::binary);
  if (!outFile.is_open()) {
    SK_LOGW("Failed to open file for writing: %s", oFilePath.c_str());
    return false;
  }

  outFile.write(static_cast<char *>(binHostAddr), binHostSize);
  outFile.close();

  SK_LOGI("Dumped kernel binary: %s, size=%u, codeSegAddr=%lu", oFilePath.c_str(), binHostSize, codeSegmentAddr);
  return true;
}

uint32_t DumpKernelBinariesToDir(const SuperKernelGraph &graph, const std::string &kernelBinsDir) {
  // Track unique binaries (deduplicate by binHdl address)
  std::unordered_set<uint64_t> seenBinHdls;
  uint32_t kernelCount = 0;

  std::vector<uint64_t> sortedNodeIds = graph.GetSortedNodeIds();
  for (uint64_t nodeId : sortedNodeIds) {
    const SuperKernelBaseNode *node = graph.GetNodeById(nodeId);
    if (node == nullptr) {
      SK_LOGE("Failed to get node %lu from graph", nodeId);
      continue;
    }

    // Only process KERNEL type nodes, skip scope-type kernels
    if (node->GetNodeType() != SkNodeType::NODE_KERNEL || node->IsScopeBegin() || node->IsScopeEnd() ||
        node->IsScopePlaceholder()) {
      continue;
    }

    const KernelInfos &kernelInfo = node->GetNodeInfos().kernelInfos;
    uint64_t binHdl = reinterpret_cast<uint64_t>(kernelInfo.binHdl);

    // Skip if we've already processed this binHdl
    if (binHdl == 0 || seenBinHdls.count(binHdl) > 0) {
      continue;
    }
    seenBinHdls.insert(binHdl);

    if (DumpSingleKernelBinary(kernelInfo, kernelBinsDir)) {
      kernelCount++;
    }
  }
  return kernelCount;
}

uint32_t DumpSkEntryBinary(const std::string &kernelBinsDir) {
  aclrtBinHandle entryBinHandle = AscendGetEntryBinHandle();
  if (entryBinHandle == nullptr) {
    SK_LOGI("SK entry bin handle is null, skip SK binary dump");
    return 0;
  }

  void *entryBinAddr = nullptr;
  uint32_t entryBinSize = 0;
  int rtRet = rtGetBinBuffer(entryBinHandle, RT_BIN_HOST_ADDR, &entryBinAddr, &entryBinSize);
  if (rtRet != 0 || entryBinAddr == nullptr || entryBinSize == 0) {
    SK_LOGW("Failed to get SK entry bin buffer, rtRet=%d", rtRet);
    return 0;
  }

  // Get SK entry binary device address (code segment start address)
  void *entryBinDevAddr = nullptr;
  size_t entryBinDevSize = 0;
  rtRet = aclrtBinaryGetDevAddress(entryBinHandle, &entryBinDevAddr, &entryBinDevSize);
  if (rtRet != 0) {
    SK_LOGW("Failed to get SK entry bin dev address, rtRet=%d", rtRet);
    return 0;
  }
  uint64_t codeSegmentAddr = reinterpret_cast<uint64_t>(entryBinDevAddr);

  // Save SK entry binary with code segment address in filename
  std::ostringstream skAddrOss;
  skAddrOss << "0x" << std::hex << std::uppercase << codeSegmentAddr;
  std::string skOFilePath = kernelBinsDir + "/sk_entry_" + skAddrOss.str() + ".o";

  std::ofstream skOutFile(skOFilePath, std::ios::binary);
  if (!skOutFile.is_open()) {
    SK_LOGW("Failed to open SK binary file for writing: %s", skOFilePath.c_str());
    return 0;
  }

  skOutFile.write(static_cast<char *>(entryBinAddr), entryBinSize);
  skOutFile.close();

  SK_LOGI("Dumped SK entry binary: %s, size=%u, codeSegAddr=%lu", skOFilePath.c_str(), entryBinSize, codeSegmentAddr);
  return 1;
}
