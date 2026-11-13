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
 * \file acl_spec_optimizer.cpp
 * \brief Model kernel specialization: task collection, resource ID extraction, function replacement
 *        and binary ownership. A failed replacement is reported, not undone.
 */

#include "acl_spec_optimizer.h"
#include "aclmodel_specialization.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <elf.h>
#include <limits>
#include <new>
#include <stdexcept>
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>

#include <boost/filesystem.hpp>
#include <boost/system/error_code.hpp>

#include "feature_manager.h"
#include "spec_scope_launch.h"
#include "runtime/rt_external_kernel.h"
#include "securec.h"
#include "sk_log.h"

namespace model_spec {

namespace {
namespace fs = boost::filesystem;

constexpr size_t MAX_KERNEL_ENTRY_LEN = 256;

void BindOptionPointers(const std::vector<std::string> &storage, std::vector<const char *> &pointers) {
  pointers.clear();
  pointers.reserve(storage.size());
  for (const auto &option : storage) {
    pointers.push_back(option.c_str());
  }
}

inline aclError RetainBinaries(aclmdlRI model, const std::vector<std::unique_ptr<KernelSpecRequest>> &requests) {
  return ModelSpecResourceManager::GetInstance().RetainBinariesForModel(model, requests);
}

bool BindHostBufferPlaceholders(const aclmdlRIKernelTaskParams &params, const std::vector<size_t> &offsets,
                                KernelSpecRequest &request) {
  const auto &hostBuffer = params.hostBuffer;
  if (hostBuffer.placeHolderNum == 0) {
    return true;
  }
  if (hostBuffer.placeHolderArray == nullptr || hostBuffer.placeHolderNum > offsets.size()) {
    SK_DLOGE("aclmdlRISpecOptimize invalid placeholder table: kernel=%s streamId=%d taskId=%" PRId64 " reason=%s",
             request.kernelEntry.c_str(), request.streamId, request.taskId,
             hostBuffer.placeHolderArray == nullptr ? "null table" : "count exceeds parameters");
    return false;
  }
  size_t parameterEnd = 0;
  for (size_t index = 0; index < offsets.size(); ++index) {
    parameterEnd = std::max(parameterEnd, offsets[index] + request.argumentBytes[index]);
  }
  std::vector<size_t> dataOffsets;
  dataOffsets.reserve(hostBuffer.placeHolderNum);
  for (size_t index = 0; index < hostBuffer.placeHolderNum; ++index) {
    dataOffsets.push_back(hostBuffer.placeHolderArray[index].dataOffset);
  }
  std::sort(dataOffsets.begin(), dataOffsets.end());
  dataOffsets.erase(std::unique(dataOffsets.begin(), dataOffsets.end()), dataOffsets.end());
  std::vector<bool> rebound(offsets.size(), false);
  for (size_t index = 0; index < hostBuffer.placeHolderNum; ++index) {
    const auto &placeholder = hostBuffer.placeHolderArray[index];
    const auto parameter = std::find(offsets.begin(), offsets.end(), placeholder.addrOffset);
    if (parameter == offsets.end() || placeholder.dataOffset < parameterEnd ||
        placeholder.dataOffset >= request.hostArgs.size()) {
      SK_DLOGE("aclmdlRISpecOptimize invalid placeholder: kernel=%s streamId=%d taskId=%" PRId64 " index=%zu reason=%s",
               request.kernelEntry.c_str(), request.streamId, request.taskId, index,
               parameter == offsets.end() ? "address offset not found" : "data offset out of range");
      return false;
    }
    const size_t parameterIndex = static_cast<size_t>(parameter - offsets.begin());
    if (rebound[parameterIndex] || request.argumentBytes[parameterIndex] != sizeof(uint64_t)) {
      SK_DLOGE("aclmdlRISpecOptimize invalid placeholder parameter: kernel=%s streamId=%d taskId=%" PRId64
               " index=%zu parameter=%zu reason=%s",
               request.kernelEntry.c_str(), request.streamId, request.taskId, index, parameterIndex,
               rebound[parameterIndex] ? "duplicate" : "not a pointer slot");
      return false;
    }
    // ACLRTC consumes Host data directly; the pointer slot may already contain a relocated Device address.
    request.argumentAddresses[parameterIndex] = request.hostArgs.data() + placeholder.dataOffset;
    // Bound the data by the next distinct start; the Manifest specifies the actual constant length.
    const auto nextData = std::upper_bound(dataOffsets.begin(), dataOffsets.end(), placeholder.dataOffset);
    const size_t dataEnd = nextData == dataOffsets.end() ? request.hostArgs.size() : *nextData;
    request.argumentBytes[parameterIndex] = dataEnd - placeholder.dataOffset;
    rebound[parameterIndex] = true;
  }
  return true;
}

bool GetKernelArgumentCount(const aclmdlRIKernelTaskParams &params, const KernelSpecRequest &request, size_t &count) {
  const aclError result = aclrtFunctionGetParamCount(params.funcHandle, &count);
  if (result != ACL_SUCCESS || count == 0) {
    SK_DLOGE("aclmdlRISpecOptimize parameter count unavailable: kernel=%s streamId=%d taskId=%" PRId64
             " reason=%s ret=%d",
             request.kernelEntry.c_str(), request.streamId, request.taskId,
             result != ACL_SUCCESS ? "query failed" : "zero count", result);
    return false;
  }
  return true;
}

bool GetKernelArgumentInfo(const aclmdlRIKernelTaskParams &params, const KernelSpecRequest &request, size_t index,
                           size_t &offset, size_t &size) {
  const aclError result = aclrtFunctionGetParamInfo(params.funcHandle, index, &offset, &size);
  if (result != ACL_SUCCESS || size == 0 || size > std::numeric_limits<size_t>::max() - offset) {
    SK_DLOGE("aclmdlRISpecOptimize invalid parameter metadata: kernel=%s streamId=%d taskId=%" PRId64
             " parameter=%zu reason=%s ret=%d",
             request.kernelEntry.c_str(), request.streamId, request.taskId, index,
             result != ACL_SUCCESS ? "query failed" : "invalid range", result);
    return false;
  }
  return true;
}

bool CaptureArrayArguments(const aclmdlRIKernelTaskParams &params, KernelSpecRequest &request) {
  if (params.argsArray == nullptr) {
    SK_DLOGE("aclmdlRISpecOptimize null argument array: kernel=%s streamId=%d taskId=%" PRId64,
             request.kernelEntry.c_str(), request.streamId, request.taskId);
    return false;
  }
  size_t count = 0;
  if (!GetKernelArgumentCount(params, request, count)) {
    return false;
  }
  std::vector<size_t> offsets(count);
  request.argumentBytes.resize(count);
  size_t snapshotSize = 0;
  for (size_t index = 0; index < count; ++index) {
    size_t offset = 0;
    size_t size = 0;
    if (!GetKernelArgumentInfo(params, request, index, offset, size)) {
      return false;
    }
    if (params.argsArray[index] == nullptr) {
      SK_DLOGE("aclmdlRISpecOptimize null array parameter: kernel=%s streamId=%d taskId=%" PRId64 " parameter=%zu",
               request.kernelEntry.c_str(), request.streamId, request.taskId, index);
      return false;
    }
    snapshotSize = std::max(snapshotSize, offset + size);
    offsets[index] = offset;
    request.argumentBytes[index] = size;
  }
  // Allocate once before binding addresses, and keep the snapshot alive through specialization.
  request.hostArgs.resize(snapshotSize);
  request.argumentAddresses.reserve(count);
  for (size_t index = 0; index < count; ++index) {
    const auto copyResult = memcpy_s(request.hostArgs.data() + offsets[index], snapshotSize - offsets[index],
                                     params.argsArray[index], request.argumentBytes[index]);
    if (copyResult != EOK) {
      SK_DLOGE("aclmdlRISpecOptimize array argument snapshot failed: kernel=%s streamId=%d taskId=%" PRId64
               " parameter=%zu ret=%d",
               request.kernelEntry.c_str(), request.streamId, request.taskId, index, copyResult);
      return false;
    }
    request.argumentAddresses.push_back(request.hostArgs.data() + offsets[index]);
  }
  return true;
}

bool CaptureHostBufferArguments(const aclmdlRIKernelTaskParams &params, KernelSpecRequest &request) {
  const void *buffer = params.hostBuffer.hostArgs;
  const size_t bufferSize = params.hostBuffer.hostArgsSize;
  size_t count = 0;
  if (!GetKernelArgumentCount(params, request, count)) {
    return false;
  }
  if (buffer == nullptr || count > bufferSize) {
    SK_DLOGE(
        "aclmdlRISpecOptimize invalid host argument buffer: kernel=%s streamId=%d "
        "taskId=%" PRId64 " reason=%s",
        request.kernelEntry.c_str(), request.streamId, request.taskId,
        buffer == nullptr ? "null buffer" : "buffer too small");
    return false;
  }
  std::vector<size_t> offsets(count);
  request.argumentBytes.resize(count);
  for (size_t index = 0; index < count; ++index) {
    size_t offset = 0;
    size_t size = 0;
    if (!GetKernelArgumentInfo(params, request, index, offset, size)) {
      return false;
    }
    if (offset > bufferSize || size > bufferSize - offset) {
      SK_DLOGE("aclmdlRISpecOptimize parameter exceeds host buffer: kernel=%s streamId=%d taskId=%" PRId64
               " parameter=%zu",
               request.kernelEntry.c_str(), request.streamId, request.taskId, index);
      return false;
    }
    offsets[index] = offset;
    request.argumentBytes[index] = size;
  }
  request.hostArgs.resize(bufferSize);
  const auto copyResult = memcpy_s(request.hostArgs.data(), bufferSize, buffer, bufferSize);
  if (copyResult != EOK) {
    SK_DLOGE(
        "aclmdlRISpecOptimize host argument snapshot failed: kernel=%s streamId=%d "
        "taskId=%" PRId64 " ret=%d",
        request.kernelEntry.c_str(), request.streamId, request.taskId, copyResult);
    return false;
  }
  request.argumentAddresses.reserve(count);
  for (size_t offset : offsets) {
    request.argumentAddresses.push_back(request.hostArgs.data() + offset);
  }
  return BindHostBufferPlaceholders(params, offsets, request);
}

bool CaptureKernelArguments(const aclmdlRIKernelTaskParams &params, KernelSpecRequest &request) {
  switch (params.argsType) {
    case ACL_RT_KERNEL_ARGS_ARRAY:
      return CaptureArrayArguments(params, request);
    case ACL_RT_KERNEL_ARGS_HOST_BUFFER:
      return CaptureHostBufferArguments(params, request);
    case ACL_RT_KERNEL_ARGS_DEVICE_BUFFER:
      SK_DLOGW("aclmdlRISpecOptimize DEVICE_BUFFER kernel arguments are not supported: streamId=%d taskId=%" PRId64
               " kernel=%s argsType=%u",
               request.streamId, request.taskId, request.kernelEntry.c_str(), static_cast<uint32_t>(params.argsType));
      return false;
    default:
      SK_DLOGE("aclmdlRISpecOptimize unknown argument type: streamId=%d taskId=%" PRId64 " kernel=%s argsType=%u",
               request.streamId, request.taskId, request.kernelEntry.c_str(), static_cast<uint32_t>(params.argsType));
      return false;
  }
}

}  // namespace

aclError AclSpecOptimizer::Init(aclmdlRI model) {
  if (model == nullptr) {
    SK_DLOGE("aclmdlRISpecOptimize init rejected: model is null");
    return ACL_ERROR_INVALID_PARAM;
  }
  uint32_t modelId = 0;
  aclError result = aclmdlRIGetId(model, &modelId);
  if (result != ACL_SUCCESS) {
    SK_DLOGE("aclmdlRISpecOptimize model id unavailable: model=%p ret=%d", model, result);
    return result;
  }
  result = outputDirectory_.Create(modelId);
  if (result != ACL_SUCCESS) {
    return result;
  }
  return ACL_SUCCESS;
}

aclError AclSpecOptimizer::Optimize(aclmdlRI model, const FeatureManager &featureManager) const {
  const char *apiName =
      featureManager.Get<Feature::IS_SCOPE_JIT>() ? "aclmdlRISpecOptimizeByFlag" : "aclmdlRISpecOptimize";
  SK_DLOGI("%s started: model=%p", apiName, model);
  std::vector<std::unique_ptr<KernelSpecRequest>> requests;
  std::vector<ScopeTaskRecord> scopeMarkers;
  aclError result = Collect(model, featureManager, outputDirectory_.Path(), requests, scopeMarkers);
  if (result != ACL_SUCCESS) {
    return result;
  }
  if (requests.empty()) {
    SK_DLOGW("%s has no candidate kernels: model=%p markers=%zu", apiName, model, scopeMarkers.size());
    return scopeMarkers.empty() ? ACL_SUCCESS : UpdateTasks(model, requests, scopeMarkers);
  }
  const size_t candidateCount = requests.size();
  SpecializeKernels(requests);
  result = LoadKernels(requests);
  if (result != ACL_SUCCESS) {
    return result;
  }
  result = UpdateTasks(model, requests, scopeMarkers);
  const aclError retained = RetainBinaries(model, requests);
  SK_DLOGI("%s finished: model=%p commit=%d retain=%d candidates=%zu specialized=%zu keptDynamic=%zu", apiName, model,
           result, retained, candidateCount, requests.size(), candidateCount - requests.size());
  return result != ACL_SUCCESS ? result : retained;
}

aclError AclSpecOptimizer::ReadTaskIdentity(TaskIdentity &identity) const {
  const auto &record = identity.record;
  if (record.task == nullptr) {
    SK_DLOGE("Task identity unavailable: streamId=%d taskId=%" PRId64 " reason=null task ret=%d", record.streamId,
             record.taskId, ACL_ERROR_FAILURE);
    return ACL_ERROR_FAILURE;
  }
  aclmdlRITaskType type = ACL_MODEL_RI_TASK_DEFAULT;
  aclError result = aclmdlRITaskGetType(record.task, &type);
  if (result != ACL_SUCCESS) {
    SK_DLOGE("Task identity unavailable: streamId=%d taskId=%" PRId64 " reason=task type query failed ret=%d",
             record.streamId, record.taskId, result);
    return result;
  }
  identity.params.type = type;
  if (type != ACL_MODEL_RI_TASK_KERNEL) {
    identity.record.kind = ScopeTaskKind::OTHER;
    return ACL_SUCCESS;
  }

  result = aclmdlRITaskGetParams(record.task, &identity.params);
  if (result != ACL_SUCCESS) {
    SK_DLOGE("Task identity unavailable: streamId=%d taskId=%" PRId64 " reason=task parameters query failed ret=%d",
             record.streamId, record.taskId, result);
    return result;
  }
  const auto function = identity.params.kernelTaskParams.funcHandle;
  if (function == nullptr) {
    SK_DLOGE("Task identity unavailable: streamId=%d taskId=%" PRId64 " reason=null function handle ret=%d",
             record.streamId, record.taskId, ACL_ERROR_FAILURE);
    return ACL_ERROR_FAILURE;
  }
  std::array<char, MAX_KERNEL_ENTRY_LEN> name{};
  result = aclrtGetFunctionName(function, name.size(), name.data());
  if (result != ACL_SUCCESS) {
    SK_DLOGE("Task identity unavailable: streamId=%d taskId=%" PRId64 " reason=function name query failed ret=%d",
             record.streamId, record.taskId, result);
    return result;
  }
  if (name[0] == '\0') {
    SK_DLOGE("Task identity unavailable: streamId=%d taskId=%" PRId64 " reason=empty function name ret=%d",
             record.streamId, record.taskId, ACL_ERROR_FAILURE);
    return ACL_ERROR_FAILURE;
  }
  if (std::memchr(name.data(), '\0', name.size()) == nullptr) {
    SK_DLOGE("Task identity unavailable: streamId=%d taskId=%" PRId64 " reason=unterminated function name ret=%d",
             record.streamId, record.taskId, ACL_ERROR_FAILURE);
    return ACL_ERROR_FAILURE;
  }

  identity.kernelEntry = name.data();
  if (identity.kernelEntry == SPEC_SCOPE_BEGIN_DAV_2201 || identity.kernelEntry == SPEC_SCOPE_BEGIN_DAV_3510) {
    identity.record.kind = ScopeTaskKind::BEGIN;
  } else if (identity.kernelEntry == SPEC_SCOPE_END_DAV_2201 || identity.kernelEntry == SPEC_SCOPE_END_DAV_3510) {
    identity.record.kind = ScopeTaskKind::END;
  } else {
    identity.record.kind = ScopeTaskKind::KERNEL;
  }
  return ACL_SUCCESS;
}

aclError AclSpecOptimizer::SetTaskAdmission(std::vector<TaskIdentity> &identities,
                                            const FeatureManager &featureManager) const {
  if (!featureManager.Get<Feature::IS_SCOPE_JIT>()) {
    for (auto &identity : identities) {
      identity.record.allowSpecialization = identity.record.kind == ScopeTaskKind::KERNEL;
    }
    return ACL_SUCCESS;
  }

  std::sort(identities.begin(), identities.end(), [](const TaskIdentity &left, const TaskIdentity &right) {
    return left.record.taskId < right.record.taskId;
  });
  for (size_t index = 1; index < identities.size(); ++index) {
    if (identities[index - 1].record.taskId == identities[index].record.taskId) {
      const auto &identity = identities[index];
      SK_DLOGE("aclmdlRISpecOptimizeByFlag found duplicate taskId: streamId=%d taskId=%" PRId64 " kernel=%s",
               identity.record.streamId, identity.record.taskId, identity.kernelEntry.c_str());
      return ACL_ERROR_INVALID_PARAM;
    }
  }

  std::vector<size_t> activeBegins;
  for (size_t index = 0; index < identities.size(); ++index) {
    auto &identity = identities[index];
    auto &record = identity.record;
    switch (record.kind) {
      case ScopeTaskKind::BEGIN:
        activeBegins.push_back(index);
        break;
      case ScopeTaskKind::END:
        if (activeBegins.empty()) {
          SK_DLOGE("aclmdlRISpecOptimizeByFlag invalid scope: streamId=%d taskId=%" PRId64
                   " end has no matching begin, kernel=%s",
                   record.streamId, record.taskId, identity.kernelEntry.c_str());
          return ACL_ERROR_INVALID_PARAM;
        }
        activeBegins.pop_back();
        break;
      case ScopeTaskKind::KERNEL:
      case ScopeTaskKind::OTHER:
        break;
    }
    record.allowSpecialization = record.kind == ScopeTaskKind::KERNEL && !activeBegins.empty();
    if (record.allowSpecialization) {
      SK_DLOGI("Kernel Specialization: [IN_SCOPE] streamId=%d, taskId=%" PRId64 ", type=KERNEL, kernel=%s",
               record.streamId, record.taskId, identity.kernelEntry.c_str());
    }
  }
  if (!activeBegins.empty()) {
    const auto &begin = identities[activeBegins.back()];
    SK_DLOGE("aclmdlRISpecOptimizeByFlag invalid scope: streamId=%d taskId=%" PRId64
             " begin has no matching end, depth=%zu kernel=%s",
             begin.record.streamId, begin.record.taskId, activeBegins.size(), begin.kernelEntry.c_str());
    return ACL_ERROR_INVALID_PARAM;
  }
  return ACL_SUCCESS;
}

aclError AclSpecOptimizer::Collect(aclmdlRI model, const FeatureManager &featureManager,
                                   std::string_view outputDirectory,
                                   std::vector<std::unique_ptr<KernelSpecRequest>> &requests,
                                   std::vector<ScopeTaskRecord> &scopeMarkers) const {
  uint32_t streamCount = 0;
  aclError result = aclmdlRIGetStreams(model, nullptr, &streamCount);
  if (result != ACL_SUCCESS) {
    SK_DLOGE("aclmdlRISpecOptimize stream count unavailable: model=%p ret=%d", model, result);
    return result;
  }
  if (streamCount == 0) {
    SK_DLOGW("aclmdlRISpecOptimize model has no stream: model=%p", model);
    return ACL_SUCCESS;
  }
  std::vector<aclrtStream> streams(streamCount);
  result = aclmdlRIGetStreams(model, streams.data(), &streamCount);
  if (result != ACL_SUCCESS) {
    SK_DLOGE("aclmdlRISpecOptimize streams unavailable: model=%p count=%u ret=%d", model, streamCount, result);
    return result;
  }
  if (streamCount > streams.size()) {
    SK_DLOGE("Runtime returned more streams than the allocated capacity: streamCount=%u capacity=%zu", streamCount,
             streams.size());
    return ACL_ERROR_FAILURE;
  }
  std::vector<TaskIdentity> identities;
  std::vector<std::unique_ptr<KernelSpecRequest>> collected;
  std::vector<ScopeTaskRecord> collectedMarkers;
  for (uint32_t streamIndex = 0; streamIndex < streamCount; ++streamIndex) {
    if (streams[streamIndex] == nullptr) {
      SK_DLOGE("Runtime returned a null stream handle: streamIndex=%u", streamIndex);
      return ACL_ERROR_FAILURE;
    }
    uint32_t taskCount = 0;
    result = aclmdlRIGetTasksByStream(streams[streamIndex], nullptr, &taskCount);
    if (result != ACL_SUCCESS) {
      SK_DLOGE("Failed to get the stream task count: streamIndex=%u ret=%d", streamIndex, result);
      return result;
    }
    if (taskCount == 0) {
      continue;
    }
    int32_t streamId = -1;
    result = aclrtStreamGetId(streams[streamIndex], &streamId);
    if (result != ACL_SUCCESS) {
      streamId = -1;
      SK_DLOGW("Failed to get stream ID: streamIndex=%u ret=%d; continuing specialization", streamIndex, result);
    }
    std::vector<aclmdlRITask> tasks(taskCount);
    result = aclmdlRIGetTasksByStream(streams[streamIndex], tasks.data(), &taskCount);
    if (result != ACL_SUCCESS) {
      SK_DLOGE("Failed to get the stream task list: streamId=%d streamIndex=%u capacity=%zu ret=%d", streamId,
               streamIndex, tasks.size(), result);
      return result;
    }
    if (taskCount > tasks.size()) {
      SK_DLOGE(
          "Runtime returned more tasks than the allocated capacity: streamId=%d streamIndex=%u taskCount=%u "
          "capacity=%zu",
          streamId, streamIndex, taskCount, tasks.size());
      return ACL_ERROR_FAILURE;
    }
    for (uint32_t taskIndex = 0; taskIndex < taskCount; ++taskIndex) {
      TaskIdentity identity;
      identity.record.task = tasks[taskIndex];
      identity.record.streamIndex = streamIndex;
      identity.record.taskIndex = taskIndex;
      identity.record.streamId = streamId;
      if (identity.record.task != nullptr) {
        uint32_t taskId = 0;
        result = aclmdlRITaskGetSeqId(identity.record.task, &taskId);
        if (result == ACL_SUCCESS) {
          identity.record.taskId = taskId;
        } else {
          SK_DLOGE("Failed to get task sequence ID: streamId=%d streamIndex=%u taskIndex=%u ret=%d", streamId,
                   streamIndex, taskIndex, result);
          return result;
        }
      }
      result = ReadTaskIdentity(identity);
      if (result != ACL_SUCCESS) {
        return result;
      }
      if (identity.record.kind == ScopeTaskKind::BEGIN || identity.record.kind == ScopeTaskKind::END) {
        collectedMarkers.push_back(identity.record);
      }
      identities.push_back(std::move(identity));
    }
  }
  // Validate model-wide scopes before reading resources or copying any candidate's arguments.
  result = SetTaskAdmission(identities, featureManager);
  if (result != ACL_SUCCESS) {
    return result;
  }
  for (const auto &identity : identities) {
    const auto &record = identity.record;
    if (record.kind != ScopeTaskKind::KERNEL || !record.allowSpecialization) {
      continue;
    }
    auto request = std::make_unique<KernelSpecRequest>();
    request->task = record.task;
    request->streamIndex = record.streamIndex;
    request->taskIndex = record.taskIndex;
    request->streamId = record.streamId;
    request->taskId = record.taskId;
    if (TryBuildRequest(identity.params, identity.kernelEntry, featureManager, outputDirectory, *request)) {
      collected.push_back(std::move(request));
    }
  }
  requests.swap(collected);
  scopeMarkers.swap(collectedMarkers);
  return ACL_SUCCESS;
}

bool AclSpecOptimizer::TryBuildRequest(const aclmdlRITaskParams &originalParams, std::string_view kernelEntry,
                                       const FeatureManager &featureManager, std::string_view outputDirectory,
                                       KernelSpecRequest &request) const {
  const uint32_t streamIndex = request.streamIndex;
  const uint32_t taskIndex = request.taskIndex;
  const int32_t streamId = request.streamId;
  const int64_t taskId = request.taskId;
  request.originalParams = originalParams;
  request.kernelEntry = kernelEntry;
  const auto &params = request.originalParams.kernelTaskParams;
  // Borrowed from the runtime: this is the binary the original kernel came from, so it is only read
  // here and never unloaded.
  aclrtBinHandle originalBinary = nullptr;
  void *elfBytes = nullptr;
  uint32_t elfSize = 0;
  const aclError binaryResult = aclrtFunctionGetBinary(params.funcHandle, &originalBinary);
  if (binaryResult != ACL_SUCCESS || originalBinary == nullptr) {
    SK_DLOGI("aclmdlRISpecOptimize keep dynamic task: streamId=%d taskId=%" PRId64
             " kernel=%s reason=original binary unavailable ret=%d",
             streamId, taskId, request.kernelEntry.c_str(), binaryResult);
    return false;
  }
  const rtError_t bufferResult = rtGetBinBuffer(originalBinary, RT_BIN_HOST_ADDR, &elfBytes, &elfSize);
  if (bufferResult != RT_ERROR_NONE) {
    SK_DLOGI("aclmdlRISpecOptimize keep dynamic task: streamId=%d taskId=%" PRId64
             " kernel=%s reason=original binary buffer unavailable ret=%d",
             streamId, taskId, request.kernelEntry.c_str(), bufferResult);
    return false;
  }
  if (!ReadSpecResourceId(elfBytes, elfSize, request.resourceId)) {
    SK_DLOGI("aclmdlRISpecOptimize keep dynamic task: streamId=%d taskId=%" PRId64 " kernel=%s no usable resource id",
             streamId, taskId, request.kernelEntry.c_str());
    return false;
  }
  if (!CaptureKernelArguments(params, request)) {
    SK_DLOGI("aclmdlRISpecOptimize keep dynamic task: streamId=%d taskId=%" PRId64
             " kernel=%s reason=argument capture failed",
             streamId, taskId, request.kernelEntry.c_str());
    return false;
  }
  request.basicOptionStorage = featureManager.Get<Feature::BASIC_COMPILE_OPTIONS>(request.kernelEntry);
  request.skOptionStorage = featureManager.Get<Feature::SK_COMPILE_OPTIONS>(request.kernelEntry);
  request.outputElfPath = (fs::path(std::string(outputDirectory)) /
                           ("stream_" + std::to_string(streamIndex) + "_task_" + std::to_string(taskIndex) + ".elf"))
                              .string();
  return true;
}

void AclSpecOptimizer::SpecializeKernels(std::vector<std::unique_ptr<KernelSpecRequest>> &requests) const {
  std::vector<std::unique_ptr<KernelSpecRequest>> specialized;
  specialized.reserve(requests.size());
  for (auto &request : requests) {
    request->BindApiRequest();
    const aclError result = aclrtcKernelSpecialization(&request->apiRequest, request->outputElfPath.c_str());
    if (result != ACL_SUCCESS) {
      SK_DLOGW("aclmdlRISpecOptimize specialization failed: streamId=%d taskId=%" PRId64
               " kernel=%s ret=%d; keeping the original dynamic kernel",
               request->streamId, request->taskId, request->kernelEntry.c_str(), result);
      continue;
    }
    specialized.push_back(std::move(request));
  }
  if (!requests.empty() && specialized.empty()) {
    SK_DLOGW("aclmdlRISpecOptimize specialized none of its %zu candidate kernels", requests.size());
  }
  requests.swap(specialized);
}

aclError AclSpecOptimizer::LoadKernels(std::vector<std::unique_ptr<KernelSpecRequest>> &requests) const {
  for (auto &request : requests) {
    aclrtBinHandle loadedBinary = nullptr;
    aclError result = aclrtBinaryLoadFromFile(request->outputElfPath.c_str(), nullptr, &loadedBinary);
    // Take ownership before checking, so a handle produced alongside an error is still released.
    request->binary.reset(loadedBinary);
    if (result != ACL_SUCCESS || loadedBinary == nullptr) {
      const aclError failure = result != ACL_SUCCESS ? result : ACL_ERROR_FAILURE;
      SK_DLOGE("aclmdlRISpecOptimize binary load failed: kernel=%s streamId=%d taskId=%" PRId64 " ret=%d",
               request->kernelEntry.c_str(), request->streamId, request->taskId, failure);
      return failure;
    }
    result = aclrtBinaryGetFunction(loadedBinary, request->kernelEntry.c_str(), &request->specFuncHandle);
    if (result != ACL_SUCCESS || request->specFuncHandle == nullptr) {
      const aclError failure = result != ACL_SUCCESS ? result : ACL_ERROR_FAILURE;
      SK_DLOGE("aclmdlRISpecOptimize function lookup failed: kernel=%s streamId=%d taskId=%" PRId64 " ret=%d",
               request->kernelEntry.c_str(), request->streamId, request->taskId, failure);
      return failure;
    }
  }
  return ACL_SUCCESS;
}

aclError AclSpecOptimizer::UpdateTasks(aclmdlRI model, const std::vector<std::unique_ptr<KernelSpecRequest>> &requests,
                                       const std::vector<ScopeTaskRecord> &markersToDisable) const {
  for (const auto &request : requests) {
    auto params = request->originalParams;
    params.kernelTaskParams.funcHandle = request->specFuncHandle;
    const aclError result = aclmdlRITaskSetParams(request->task, &params);
    if (result != ACL_SUCCESS) {
      // Stop at the first failure without undoing the earlier tasks: they keep their specialized
      // functions, and the caller decides what to do with the model. Their binaries must therefore
      // stay loaded, which is why the caller retains them either way.
      SK_DLOGE("aclmdlRISpecOptimize SetParams failed: streamId=%d taskId=%" PRId64 " kernel=%s ret=%d",
               request->streamId, request->taskId, request->kernelEntry.c_str(), result);
      return result;
    }
  }
  for (const auto &marker : markersToDisable) {
    const aclError result = aclmdlRITaskDisable(marker.task);
    if (result != ACL_SUCCESS) {
      SK_DLOGE("aclmdlRISpecOptimize Disable failed: streamId=%d taskId=%" PRId64 " kind=%s ret=%d", marker.streamId,
               marker.taskId, marker.kind == ScopeTaskKind::BEGIN ? "BEGIN" : "END", result);
      return result;
    }
  }
  const aclError result = aclmdlRIUpdate(model);
  if (result != ACL_SUCCESS) {
    if (requests.empty()) {
      SK_DLOGE("aclmdlRISpecOptimize aclmdlRIUpdate failed: ret=%d", result);
    } else {
      for (const auto &request : requests) {
        SK_DLOGE("aclmdlRISpecOptimize aclmdlRIUpdate failed: kernel=%s streamId=%d taskId=%" PRId64 " ret=%d",
                 request->kernelEntry.c_str(), request->streamId, request->taskId, result);
      }
    }
  }
  return result;
}

void KernelSpecRequest::BindApiRequest() {
  BindOptionPointers(basicOptionStorage, basicOptionPointers);
  BindOptionPointers(skOptionStorage, skOptionPointers);
  apiRequest.resourceId = resourceId.c_str();
  apiRequest.kernelEntry = kernelEntry.c_str();
  apiRequest.argsCount = argumentAddresses.size();
  apiRequest.argsAddr = argumentAddresses.data();
  apiRequest.argsBytes = argumentBytes.data();
  apiRequest.options = basicOptionPointers.data();
  apiRequest.optionCount = basicOptionPointers.size();
  apiRequest.skOptions = skOptionPointers.data();
  apiRequest.skOptionCount = skOptionPointers.size();
}

void KernelSpecRequest::UnloadBinary(aclrtBinHandle binary) {
  if (binary == nullptr) {
    return;
  }
  const aclError result = aclrtBinaryUnLoad(binary);
  if (result != ACL_SUCCESS) {
    SK_DLOGE("aclmdlRISpecOptimize unload binary failed: ret=%d", result);
  }
}

ModelSpecResourceManager &ModelSpecResourceManager::GetInstance() {
  static ModelSpecResourceManager instance;
  return instance;
}

aclError ModelSpecResourceManager::RetainBinariesForModel(
    aclmdlRI model, const std::vector<std::unique_ptr<KernelSpecRequest>> &requests) {
  if (model == nullptr) {
    SK_DLOGE("aclmdlRISpecOptimize retain rejected: model is null");
    return ACL_ERROR_INVALID_PARAM;
  }
  if (requests.empty()) {
    // With no binary to own there is nothing to release later, and the destroy callback exists
    // only to do that releasing. Registering one here would saddle the model with a callback
    // over an empty list.
    return ACL_SUCCESS;
  }
  aclError result = ACL_ERROR_BAD_ALLOC;
  try {
    result = TryTakeOwnership(model, requests);
  } catch (const std::bad_alloc &) {
    result = ACL_ERROR_BAD_ALLOC;
  }
  if (result != ACL_SUCCESS) {
    for (const auto &request : requests) {
      SK_DLOGE("aclmdlRISpecOptimize binary retention failed: kernel=%s streamId=%d taskId=%" PRId64 " ret=%d",
               request->kernelEntry.c_str(), request->streamId, request->taskId, result);
      request->binary.release();
    }
  }
  return result;
}

aclError ModelSpecResourceManager::TryTakeOwnership(aclmdlRI model,
                                                    const std::vector<std::unique_ptr<KernelSpecRequest>> &requests) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto entry = modelBinaries_.find(model);
  if (entry == modelBinaries_.end()) {
    std::vector<aclrtBinHandle> reserved;
    reserved.reserve(requests.size());
    entry = modelBinaries_.emplace(model, std::move(reserved)).first;
    const aclError result = aclmdlRIDestroyRegisterCallback(model, OnModelDestroy, model);
    if (result != ACL_SUCCESS) {
      modelBinaries_.erase(entry);
      SK_DLOGE("aclmdlRISpecOptimize register destroy callback failed: ret=%d", result);
      return result;
    }
  } else {
    // This model was optimized before, so only grow the list it already has.
    auto &binaries = entry->second;
    if (requests.size() > binaries.max_size() - binaries.size()) {
      SK_DLOGE("aclmdlRISpecOptimize binary storage exhausted");
      return ACL_ERROR_BAD_ALLOC;
    }
    binaries.reserve(binaries.size() + requests.size());
  }
  // The capacity is in place, so moving the handles across can no longer fail.
  for (const auto &request : requests) {
    entry->second.push_back(request->binary.get());
    request->binary.release();
  }
  return ACL_SUCCESS;
}

void ModelSpecResourceManager::OnModelDestroy(void *userData) {
  const aclmdlRI model = static_cast<aclmdlRI>(userData);
  if (model == nullptr) {
    SK_DLOGE("aclmdlRISpecOptimize destroy callback has invalid model=%p", userData);
    return;
  }
  std::vector<aclrtBinHandle> binaries;
  {
    auto &manager = GetInstance();
    std::lock_guard<std::mutex> lock(manager.mutex_);
    const auto entry = manager.modelBinaries_.find(model);
    if (entry == manager.modelBinaries_.end()) {
      SK_DLOGE("aclmdlRISpecOptimize destroy callback found no tracked binaries: model=%p", model);
      return;
    }
    binaries.swap(entry->second);
    manager.modelBinaries_.erase(entry);
  }
  for (const auto binary : binaries) {
    KernelSpecRequest::UnloadBinary(binary);
  }
}

aclError SpecOutputDirectory::Create(uint32_t modelId) {
  if (created_) {
    SK_DLOGE("aclmdlRISpecOptimize output directory already created: path=%s", path_.c_str());
    return ACL_ERROR_INVALID_PARAM;
  }
  boost::system::error_code error;
  const char *temporaryRoot = std::getenv("TMPDIR");
  auto root =
      temporaryRoot != nullptr && temporaryRoot[0] != '\0' ? fs::path(temporaryRoot) : fs::temp_directory_path(error);
  if (error) {
    SK_DLOGE("aclmdlRISpecOptimize temporary root unavailable: modelId=%u ret=%d error=%s", modelId, error.value(),
             error.message().c_str());
    return ACL_ERROR_FAILURE;
  }
  // Resolve symlinks as well, so the directory we later remove is the one we created.
  root = fs::canonical(root, error);
  if (error) {
    SK_DLOGE("aclmdlRISpecOptimize temporary root is not resolvable: modelId=%u path=%s ret=%d error=%s", modelId,
             root.c_str(), error.value(), error.message().c_str());
    return ACL_ERROR_FAILURE;
  }
  path_ = (root / ("acl_spec_optimize_" + std::to_string(getpid()) + "_" + std::to_string(syscall(SYS_gettid)) + "_" +
                   std::to_string(modelId) + "_XXXXXX"))
              .string();
  // mkdtemp rewrites the trailing XXXXXX in place, so it edits the string's own buffer.
  if (mkdtemp(path_.data()) == nullptr) {
    SK_DLOGE("aclmdlRISpecOptimize directory creation failed: modelId=%u path=%s errno=%d", modelId, path_.c_str(),
             errno);
    path_.clear();
    return ACL_ERROR_FAILURE;
  }
  created_ = true;
  return ACL_SUCCESS;
}

SpecOutputDirectory::~SpecOutputDirectory() {
  if (!created_) {
    return;
  }
  try {
    boost::system::error_code error;
    fs::remove_all(path_, error);
    if (error) {
      SK_DLOGW("aclmdlRISpecOptimize directory cleanup failed: path=%s ret=%d", path_.c_str(), error.value());
    }
  } catch (...) {
    SK_DLOGW("aclmdlRISpecOptimize directory cleanup failed: path=%s", path_.c_str());
  }
}

bool ReadSpecResourceId(const void *elfBytes, size_t elfSize, std::string &resourceId) {
  if (elfBytes == nullptr || elfSize < sizeof(Elf64_Ehdr)) {
    SK_DLOGW("aclmdlRISpecOptimize cannot read resource id: image is null or smaller than an ELF64 header");
    return false;
  }
  const auto *bytes = static_cast<const unsigned char *>(elfBytes);
  const auto containsRange = [elfSize](uint64_t offset, uint64_t length) {
    return offset <= elfSize && length <= elfSize - offset;
  };
  Elf64_Ehdr header{};
  if (memcpy_s(&header, sizeof(header), bytes, sizeof(header)) != EOK) {
    SK_DLOGW("aclmdlRISpecOptimize cannot read resource id: the ELF header could not be copied");
    return false;
  }
  if (std::memcmp(header.e_ident, ELFMAG, SELFMAG) != 0 || header.e_ident[EI_CLASS] != ELFCLASS64 ||
      header.e_ident[EI_DATA] != ELFDATA2LSB || header.e_ident[EI_VERSION] != EV_CURRENT ||
      header.e_version != EV_CURRENT || header.e_ehsize != sizeof(header) || header.e_shentsize != sizeof(Elf64_Shdr) ||
      header.e_shnum == 0 || header.e_shstrndx == SHN_UNDEF || header.e_shstrndx >= header.e_shnum ||
      header.e_shoff < sizeof(header) ||
      !containsRange(header.e_shoff, static_cast<uint64_t>(header.e_shnum) * sizeof(Elf64_Shdr))) {
    SK_DLOGW(
        "aclmdlRISpecOptimize cannot read resource id: not a little-endian ELF64 object with a usable section table");
    return false;
  }
  const auto readSection = [&](size_t index, Elf64_Shdr &section) {
    const auto *source = bytes + header.e_shoff + index * sizeof(section);
    return memcpy_s(&section, sizeof(section), source, sizeof(section)) == EOK;
  };
  Elf64_Shdr names{};
  if (!readSection(header.e_shstrndx, names)) {
    SK_DLOGW("aclmdlRISpecOptimize cannot read resource id: the section name table could not be copied");
    return false;
  }
  if (names.sh_type != SHT_STRTAB || !containsRange(names.sh_offset, names.sh_size)) {
    SK_DLOGW("aclmdlRISpecOptimize cannot read resource id: section name table is not a readable string table");
    return false;
  }
  const auto *nameData = reinterpret_cast<const char *>(bytes + names.sh_offset);
  std::string_view found;
  for (size_t index = 0; index < header.e_shnum; ++index) {
    Elf64_Shdr section{};
    if (!readSection(index, section)) {
      SK_DLOGW("aclmdlRISpecOptimize cannot read resource id: a section header could not be copied");
      return false;
    }
    if (section.sh_type == SHT_NULL || section.sh_type == SHT_NOBITS || section.sh_name >= names.sh_size) {
      continue;
    }
    const char *name = nameData + section.sh_name;
    const auto *end = static_cast<const char *>(std::memchr(name, '\0', names.sh_size - section.sh_name));
    if (end == nullptr || std::string_view(name, end - name) != SPEC_RESOURCE_ID_SECTION_NAME) {
      continue;
    }
    if ((section.sh_type != SHT_NOTE && section.sh_type != SHT_PROGBITS) ||
        !containsRange(section.sh_offset, section.sh_size)) {
      SK_DLOGW("aclmdlRISpecOptimize cannot read resource id: section .ascend.meta is not readable metadata");
      return false;
    }
    const auto *data = bytes + section.sh_offset;
    size_t offset = 0;
    while (offset < section.sh_size) {
      if (section.sh_size - offset < sizeof(uint16_t) * 2) {
        SK_DLOGW("aclmdlRISpecOptimize cannot read resource id: trailing bytes are too short for a TLV head");
        return false;
      }
      uint16_t type = 0;
      uint16_t length = 0;
      if (memcpy_s(&type, sizeof(type), data + offset, sizeof(type)) != EOK ||
          memcpy_s(&length, sizeof(length), data + offset + sizeof(type), sizeof(length)) != EOK) {
        SK_DLOGW("aclmdlRISpecOptimize cannot read resource id: a TLV head could not be copied");
        return false;
      }
      offset += sizeof(type) + sizeof(length);
      if (length > section.sh_size - offset) {
        SK_DLOGW("aclmdlRISpecOptimize cannot read resource id: a TLV value runs past the end of .ascend.meta");
        return false;
      }
      if (type == SPEC_RESOURCE_ID_TLV_TYPE) {
        if (!found.empty()) {
          SK_DLOGW("aclmdlRISpecOptimize cannot read resource id: more than one resource id entry");
          return false;
        }
        if (length != SPEC_RESOURCE_ID_TLV_LENGTH) {
          SK_DLOGW("aclmdlRISpecOptimize cannot read resource id: resource id entry has an unexpected length");
          return false;
        }
        const std::string_view value(reinterpret_cast<const char *>(data + offset), length);
        for (const char digit : value) {
          if (!((digit >= '0' && digit <= '9') || (digit >= 'a' && digit <= 'f'))) {
            SK_DLOGW("aclmdlRISpecOptimize cannot read resource id: resource id is not lowercase hexadecimal");
            return false;
          }
        }
        found = value;
      }
      offset += length;
    }
  }
  if (found.empty()) {
    SK_DLOGI("aclmdlRISpecOptimize no specialization resource id: section=%s type=%u",
             std::string(SPEC_RESOURCE_ID_SECTION_NAME).c_str(), SPEC_RESOURCE_ID_TLV_TYPE);
    return false;
  }
  resourceId.assign(found);
  return true;
}

}  // namespace model_spec
