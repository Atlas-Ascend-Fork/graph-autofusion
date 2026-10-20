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
#include <cstdlib>
#include <cstring>
#include <elf.h>
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
  std::vector<std::unique_ptr<KernelSpecRequest>> requests;
  std::vector<ScopeTaskRecord> scopeMarkers;
  aclError result = Collect(model, featureManager, outputDirectory_.Path(), requests, scopeMarkers);
  if (result != ACL_SUCCESS) {
    SK_DLOGE("aclmdlRISpecOptimize collection failed: model=%p ret=%d", model, result);
    return result;
  }
  if (requests.empty() && scopeMarkers.empty()) {
    SK_DLOGW("aclmdlRISpecOptimize finished with nothing to specialize: model=%p", model);
    return ACL_SUCCESS;
  }
  SpecializeKernels(requests);
  result = LoadKernels(requests);
  if (result != ACL_SUCCESS) {
    SK_DLOGE("aclmdlRISpecOptimize load failed: model=%p ret=%d", model, result);
    return result;
  }
  result = UpdateTasks(model, requests, scopeMarkers);
  const aclError retained = RetainBinaries(model, requests);
  SK_DLOGI("aclmdlRISpecOptimize finished: model=%p commit=%d retain=%d", model, result, retained);
  return result != ACL_SUCCESS ? result : retained;
}

aclError AclSpecOptimizer::ReadTaskIdentity(TaskIdentity &identity) const {
  const auto &record = identity.record;
  if (record.task == nullptr) {
    SK_DLOGW("Task identity unavailable: stream=%u task=%u reason=null task ret=%d", record.streamIndex,
             record.taskIndex, ACL_ERROR_FAILURE);
    return ACL_ERROR_FAILURE;
  }
  aclmdlRITaskType type = ACL_MODEL_RI_TASK_DEFAULT;
  aclError result = aclmdlRITaskGetType(record.task, &type);
  if (result != ACL_SUCCESS) {
    SK_DLOGW("Task identity unavailable: stream=%u task=%u reason=task type query failed ret=%d", record.streamIndex,
             record.taskIndex, result);
    return result;
  }
  if (type != ACL_MODEL_RI_TASK_KERNEL) {
    identity.record.kind = ScopeTaskKind::OTHER;
    return ACL_SUCCESS;
  }

  result = aclmdlRITaskGetParams(record.task, &identity.params);
  if (result != ACL_SUCCESS) {
    SK_DLOGW("Task identity unavailable: stream=%u task=%u reason=task parameters query failed ret=%d",
             record.streamIndex, record.taskIndex, result);
    return result;
  }
  if (identity.params.type != ACL_MODEL_RI_TASK_KERNEL) {
    SK_DLOGW("Task identity unavailable: stream=%u task=%u reason=parameter type mismatch ret=%d", record.streamIndex,
             record.taskIndex, ACL_ERROR_FAILURE);
    return ACL_ERROR_FAILURE;
  }
  const auto function = identity.params.kernelTaskParams.funcHandle;
  if (function == nullptr) {
    SK_DLOGW("Task identity unavailable: stream=%u task=%u reason=null function handle ret=%d", record.streamIndex,
             record.taskIndex, ACL_ERROR_FAILURE);
    return ACL_ERROR_FAILURE;
  }
  std::array<char, MAX_KERNEL_ENTRY_LEN> name{};
  result = aclrtGetFunctionName(function, name.size(), name.data());
  if (result != ACL_SUCCESS) {
    SK_DLOGW("Task identity unavailable: stream=%u task=%u reason=function name query failed ret=%d",
             record.streamIndex, record.taskIndex, result);
    return result;
  }
  if (name[0] == '\0') {
    SK_DLOGW("Task identity unavailable: stream=%u task=%u reason=empty function name ret=%d", record.streamIndex,
             record.taskIndex, ACL_ERROR_FAILURE);
    return ACL_ERROR_FAILURE;
  }
  if (std::memchr(name.data(), '\0', name.size()) == nullptr) {
    SK_DLOGW("Task identity unavailable: stream=%u task=%u reason=unterminated function name ret=%d",
             record.streamIndex, record.taskIndex, ACL_ERROR_FAILURE);
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

aclError AclSpecOptimizer::SetTaskAdmission(std::vector<TaskIdentity> &identities, bool isScopeJit) const {
  if (!isScopeJit) {
    for (auto &identity : identities) {
      identity.record.allowSpecialization = identity.record.kind == ScopeTaskKind::KERNEL;
    }
    return ACL_SUCCESS;
  }

  // Identities are appended in stream/task order. Each contiguous stream group has its own stack.
  size_t streamFirst = 0;
  while (streamFirst < identities.size()) {
    const uint32_t streamIndex = identities[streamFirst].record.streamIndex;
    std::vector<size_t> activeBegins;
    size_t index = streamFirst;
    for (; index < identities.size() && identities[index].record.streamIndex == streamIndex; ++index) {
      auto &record = identities[index].record;
      switch (record.kind) {
        case ScopeTaskKind::BEGIN:
          activeBegins.push_back(index);
          break;
        case ScopeTaskKind::END:
          if (activeBegins.empty()) {
            SK_DLOGE("aclmdlRISpecOptimizeByFlag invalid scope: stream=%u task=%u reason=End without Begin",
                     streamIndex, record.taskIndex);
            return ACL_ERROR_INVALID_PARAM;
          }
          activeBegins.pop_back();
          break;
        case ScopeTaskKind::KERNEL:
          record.allowSpecialization = !activeBegins.empty();
          if (record.allowSpecialization) {
            SK_DLOGI("JIT Scope: kernel in scope, stream=%u, task=%u, kernel=%s", streamIndex, record.taskIndex,
                     identities[index].kernelEntry.c_str());
          }
          break;
        case ScopeTaskKind::OTHER:
          break;
      }
    }
    if (!activeBegins.empty()) {
      const auto &begin = identities[activeBegins.back()].record;
      SK_DLOGE("aclmdlRISpecOptimizeByFlag invalid scope: stream=%u task=%u reason=Begin without End depth=%zu",
               streamIndex, begin.taskIndex, activeBegins.size());
      return ACL_ERROR_INVALID_PARAM;
    }
    streamFirst = index;
  }
  return ACL_SUCCESS;
}

aclError AclSpecOptimizer::Collect(aclmdlRI model, const FeatureManager &featureManager,
                                   std::string_view outputDirectory,
                                   std::vector<std::unique_ptr<KernelSpecRequest>> &requests,
                                   std::vector<ScopeTaskRecord> &scopeMarkers) const {
  if (model == nullptr || outputDirectory.empty()) {
    SK_DLOGE("aclmdlRISpecOptimize collect rejected: model=%p outputDirectoryEmpty=%d", model,
             static_cast<int>(outputDirectory.empty()));
    return ACL_ERROR_INVALID_PARAM;
  }
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
    SK_DLOGE("aclmdlRISpecOptimize invalid stream count: model=%p count=%u capacity=%zu", model, streamCount,
             streams.size());
    return ACL_ERROR_FAILURE;
  }
  std::vector<TaskIdentity> identities;
  std::vector<std::unique_ptr<KernelSpecRequest>> collected;
  std::vector<ScopeTaskRecord> collectedMarkers;
  const bool isScopeJit = featureManager.Get<Feature::IS_SCOPE_JIT>();
  for (uint32_t streamIndex = 0; streamIndex < streamCount; ++streamIndex) {
    if (streams[streamIndex] == nullptr) {
      SK_DLOGE("aclmdlRISpecOptimize null stream: model=%p stream=%u of %u", model, streamIndex, streamCount);
      return ACL_ERROR_FAILURE;
    }
    uint32_t taskCount = 0;
    result = aclmdlRIGetTasksByStream(streams[streamIndex], nullptr, &taskCount);
    if (result != ACL_SUCCESS) {
      SK_DLOGE("aclmdlRISpecOptimize task count unavailable: model=%p stream=%u ret=%d", model, streamIndex, result);
      return result;
    }
    if (taskCount == 0) {
      continue;
    }
    std::vector<aclmdlRITask> tasks(taskCount);
    result = aclmdlRIGetTasksByStream(streams[streamIndex], tasks.data(), &taskCount);
    if (result != ACL_SUCCESS) {
      SK_DLOGE("aclmdlRISpecOptimize tasks unavailable: model=%p stream=%u count=%u ret=%d", model, streamIndex,
               taskCount, result);
      return result;
    }
    for (uint32_t taskIndex = 0; taskIndex < taskCount; ++taskIndex) {
      TaskIdentity identity;
      identity.record.task = tasks[taskIndex];
      identity.record.streamIndex = streamIndex;
      identity.record.taskIndex = taskIndex;
      result = ReadTaskIdentity(identity);
      if (result != ACL_SUCCESS) {
        SK_DLOGW("ReadTaskIdentity failed; skipping task: stream=%u task=%u ret=%d", streamIndex, taskIndex, result);
        continue;
      }
      if (identity.record.kind == ScopeTaskKind::BEGIN || identity.record.kind == ScopeTaskKind::END) {
        collectedMarkers.push_back(identity.record);
      }
      identities.push_back(std::move(identity));
    }
  }
  // Validate every stream's range before reading resources or copying any candidate's arguments.
  result = SetTaskAdmission(identities, isScopeJit);
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
  request.originalParams = originalParams;
  request.kernelEntry = kernelEntry;
  const auto &params = request.originalParams.kernelTaskParams;
  // Borrowed from the runtime: this is the binary the original kernel came from, so it is only read
  // here and never unloaded.
  aclrtBinHandle originalBinary = nullptr;
  void *elfBytes = nullptr;
  uint32_t elfSize = 0;
  if (aclrtFunctionGetBinary(params.funcHandle, &originalBinary) != ACL_SUCCESS || originalBinary == nullptr ||
      rtGetBinBuffer(originalBinary, RT_BIN_HOST_ADDR, &elfBytes, &elfSize) != RT_ERROR_NONE) {
    SK_DLOGI("aclmdlRISpecOptimize keep dynamic task: stream=%u task=%u reason=kernel binary unavailable", streamIndex,
             taskIndex);
    return false;
  }
  if (!ReadSpecResourceId(elfBytes, elfSize, request.resourceId)) {
    SK_DLOGI("aclmdlRISpecOptimize keep dynamic task: stream=%u task=%u reason=no usable resource id", streamIndex,
             taskIndex);
    return false;
  }
  size_t count = 0;
  if (aclrtFunctionGetParamCount(params.funcHandle, &count) != ACL_SUCCESS || count == 0 || count > params.argsSize ||
      params.args == nullptr) {
    SK_DLOGI("aclmdlRISpecOptimize invalid arguments: stream=%u task=%u", streamIndex, taskIndex);
    return false;
  }
  // ACLRTC reads the argument values itself, so it needs host addresses. A captured kernel task
  // carries them on the device, so the copy is always D2H; a task holding host args would fail here
  // and simply keep its original function.
  request.hostArgs.resize(params.argsSize);
  if (aclrtMemcpy(request.hostArgs.data(), request.hostArgs.size(), params.args, params.argsSize,
                  ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
    SK_DLOGI("aclmdlRISpecOptimize keep dynamic task: stream=%u task=%u reason=arguments unavailable on host",
             streamIndex, taskIndex);
    return false;
  }
  request.argumentAddresses.reserve(count);
  request.argumentBytes.reserve(count);
  for (size_t index = 0; index < count; ++index) {
    size_t offset = 0;
    size_t size = 0;
    if (aclrtFunctionGetParamInfo(params.funcHandle, index, &offset, &size) != ACL_SUCCESS || size == 0 ||
        offset > params.argsSize || size > params.argsSize - offset) {
      SK_DLOGI("aclmdlRISpecOptimize invalid parameter range: stream=%u task=%u parameter=%zu", streamIndex, taskIndex,
               index);
      return false;
    }
    request.argumentAddresses.push_back(request.hostArgs.data() + offset);
    request.argumentBytes.push_back(size);
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
      SK_DLOGW("aclmdlRISpecOptimize specialization failed: stream=%u task=%u ret=%d", request->streamIndex,
               request->taskIndex, result);
      continue;
    }
    specialized.push_back(std::move(request));
  }
  if (specialized.empty()) {
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
      SK_DLOGE("aclmdlRISpecOptimize binary load failed: stream=%u task=%u ret=%d", request->streamIndex,
               request->taskIndex, result);
      return result != ACL_SUCCESS ? result : ACL_ERROR_FAILURE;
    }
    result = aclrtBinaryGetFunction(loadedBinary, request->kernelEntry.c_str(), &request->specFuncHandle);
    if (result != ACL_SUCCESS || request->specFuncHandle == nullptr) {
      SK_DLOGE("aclmdlRISpecOptimize function lookup failed: stream=%u task=%u ret=%d", request->streamIndex,
               request->taskIndex, result);
      return result != ACL_SUCCESS ? result : ACL_ERROR_FAILURE;
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
      SK_DLOGE("aclmdlRISpecOptimize SetParams failed: model=%p task=%p ret=%d", model, request->task, result);
      return result;
    }
  }
  for (const auto &marker : markersToDisable) {
    const aclError result = aclmdlRITaskDisable(marker.task);
    if (result != ACL_SUCCESS) {
      SK_DLOGE("aclmdlRISpecOptimize Disable failed: model=%p task=%p stream=%u taskIndex=%u kind=%s ret=%d", model,
               marker.task, marker.streamIndex, marker.taskIndex, marker.kind == ScopeTaskKind::BEGIN ? "BEGIN" : "END",
               result);
      return result;
    }
  }
  const aclError result = aclmdlRIUpdate(model);
  if (result != ACL_SUCCESS) {
    SK_DLOGE("aclmdlRISpecOptimize commit failed: model=%p ret=%d", model, result);
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
    SK_DLOGE("aclmdlRISpecOptimize unload binary failed: binary=%p ret=%d", binary, result);
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
    SK_DLOGE("aclmdlRISpecOptimize binary storage allocation failed: model=%p", model);
  }
  if (result != ACL_SUCCESS) {
    SK_DLOGE("aclmdlRISpecOptimize binaries leak until the process exits: model=%p ret=%d", model, result);
    for (const auto &request : requests) {
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
      SK_DLOGE("aclmdlRISpecOptimize register destroy callback failed: model=%p ret=%d", model, result);
      return result;
    }
  } else {
    // This model was optimized before, so only grow the list it already has.
    auto &binaries = entry->second;
    if (requests.size() > binaries.max_size() - binaries.size()) {
      SK_DLOGE("aclmdlRISpecOptimize binary storage exhausted: model=%p", model);
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
    SK_DLOGE("aclmdlRISpecOptimize temporary root unavailable: modelId=%u ret=%d", modelId, error.value());
    return ACL_ERROR_FAILURE;
  }
  // Resolve symlinks as well, so the directory we later remove is the one we created.
  root = fs::canonical(root, error);
  if (error) {
    SK_DLOGE("aclmdlRISpecOptimize temporary root is not resolvable: modelId=%u ret=%d", modelId, error.value());
    return ACL_ERROR_FAILURE;
  }
  path_ = (root / ("acl_spec_optimize_" + std::to_string(getpid()) + "_" + std::to_string(syscall(SYS_gettid)) + "_" +
                   std::to_string(modelId) + "_XXXXXX"))
              .string();
  // mkdtemp rewrites the trailing XXXXXX in place, so it edits the string's own buffer.
  if (mkdtemp(path_.data()) == nullptr) {
    SK_DLOGE("aclmdlRISpecOptimize directory creation failed: modelId=%u errno=%d", modelId, errno);
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
