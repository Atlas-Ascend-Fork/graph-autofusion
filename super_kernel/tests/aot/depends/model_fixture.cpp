/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "model_fixture_internal.h"
#include "ut_common_stubs.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <tuple>
#include <unordered_map>

namespace sk::test {
namespace {
struct Function {
  std::string name;
  KernelSpec spec;
  std::optional<std::vector<BinaryBinding>> bindings;
  // The fake code is never executed; offsets describe the SK binary metadata.
  std::array<unsigned char, 256> code{};
};
struct OwnedTask {
  RuntimeTask task;
  std::vector<unsigned char> args;
  std::vector<unsigned char> opInfo;
  std::vector<aclrtLaunchKernelAttr> attributes;
  aclrtLaunchKernelCfg config{};
  bool disabled = false;
  size_t setParamsCount = 0;
};
struct Stream {
  int32_t id;
  uint32_t *nextTaskId;
  std::vector<std::unique_ptr<OwnedTask>> tasks;
  std::vector<LaunchRecord> launches;
};
struct ModelState {
  uint32_t nextTaskId = 0;
  std::vector<std::unique_ptr<Stream>> streams;
  size_t updateAttempts = 0;
  size_t successfulUpdates = 0;
};
std::unordered_map<aclmdlRI, ModelState *> models;
std::unordered_map<aclrtStream, Stream *> streams;
std::unordered_map<aclmdlRITask, OwnedTask *> tasks;
// Function identities remain stable for the process, matching production's
// binary metadata cache. Reuse names instead of recycling binary addresses.
using FunctionKey = std::tuple<std::string, aclrtKernelType, uint16_t, uint16_t, uint32_t, uint64_t>;
std::map<FunctionKey, Function> functions;
std::unordered_map<void *, Function *> functionHandles;
std::map<uintptr_t, size_t> memory;
std::map<void *, size_t> allocations;
std::mutex memoryMutex;

bool ContainsMemory(const void *address, size_t size) {
  auto pos = memory.upper_bound(reinterpret_cast<uintptr_t>(address));
  if (pos == memory.begin()) {
    return false;
  }
  --pos;
  auto offset = reinterpret_cast<uintptr_t>(address) - pos->first;
  return offset <= pos->second && size <= pos->second - offset;
}

Function *GetFunction(const std::string &name, const KernelSpec &spec = {}) {
  auto &function = functions[{name, spec.type, spec.cubeRatio, spec.vectorRatio, spec.scheMode, spec.capability}];
  function.name = name;
  function.spec = spec;
  functionHandles[&function] = &function;
  return &function;
}

OwnedTask &AppendTask(Stream &stream, aclmdlRITaskType type) {
  if (*stream.nextTaskId == std::numeric_limits<uint32_t>::max()) {
    throw std::length_error("too many fixture tasks");
  }
  auto owned = std::make_unique<OwnedTask>();
  owned->task.task_id = (*stream.nextTaskId)++;
  owned->task.type = type;
  owned->task.params.type = type;
  auto &result = *owned;
  tasks[&result.task] = &result;
  stream.tasks.push_back(std::move(owned));
  return result;
}

aclmdlRITask AppendKernel(Stream &stream, const std::string &name, const void *args, size_t size,
                          const KernelSpec &spec = {}) {
  auto &owned = AppendTask(stream, ACL_MODEL_RI_TASK_KERNEL);
  owned.args.resize(size, 0);
  if (args != nullptr) {
    std::memcpy(owned.args.data(), args, size);
  }
  {
    std::lock_guard<std::mutex> lock(memoryMutex);
    memory[reinterpret_cast<uintptr_t>(owned.args.data())] = size;
  }
  auto &params = owned.task.params.kernelTaskParams;
  params.funcHandle = GetFunction(name, spec);
  params.args = owned.args.data();
  params.argsSize = size;
  params.numBlocks = spec.numBlocks;
  return &owned.task;
}

Stream &RequireStream(ModelState &model, aclrtStream handle) {
  for (auto &stream : model.streams) {
    if (stream.get() == handle) {
      return *stream;
    }
  }
  throw std::invalid_argument("stream does not belong to model");
}
}  // namespace

struct Model::Impl : ModelState {};
Model::Model() : impl_(std::make_unique<Impl>()) {
  models[Handle()] = impl_.get();
}
Model::~Model() {
  Destroy();
}
aclmdlRI Model::Handle() const {
  return impl_.get();
}
aclrtStream Model::AddStream() {
  if (!impl_) {
    throw std::logic_error("model destroyed");
  }
  if (impl_->streams.size() >= static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw std::length_error("too many fixture streams");
  }
  auto stream = std::make_unique<Stream>();
  stream->id = static_cast<int32_t>(impl_->streams.size());
  stream->nextTaskId = &impl_->nextTaskId;
  auto handle = stream.get();
  streams[handle] = handle;
  impl_->streams.push_back(std::move(stream));
  return handle;
}
aclmdlRITask Model::AddKernel(aclrtStream stream, const std::string &name, const KernelSpec &spec) {
  if (!impl_) {
    throw std::logic_error("model destroyed");
  }
  return AppendKernel(RequireStream(*impl_, stream), name, nullptr, sizeof(uint64_t), spec);
}
aclmdlRITask Model::AddEvent(aclrtStream stream, aclmdlRITaskType type, aclrtEvent event) {
  if (!impl_) {
    throw std::logic_error("model destroyed");
  }
  if (type != ACL_MODEL_RI_TASK_EVENT_RECORD && type != ACL_MODEL_RI_TASK_EVENT_WAIT &&
      type != ACL_MODEL_RI_TASK_EVENT_RESET) {
    throw std::invalid_argument("only record/wait/reset events are supported");
  }
  auto &owned = AppendTask(RequireStream(*impl_, stream), type);
  if (type == ACL_MODEL_RI_TASK_EVENT_RECORD) {
    owned.task.params.eventRecordTaskParams.event = event;
  } else if (type == ACL_MODEL_RI_TASK_EVENT_WAIT) {
    owned.task.params.eventWaitTaskParams.event = event;
  } else {
    owned.task.params.eventResetTaskParams.event = event;
  }
  return &owned.task;
}
aclmdlRITask Model::AddTask(aclrtStream stream, const aclmdlRITaskParams &params) {
  if (!impl_) {
    throw std::logic_error("model destroyed");
  }
  if (params.type == ACL_MODEL_RI_TASK_KERNEL) {
    throw std::invalid_argument("use AddKernel for owned kernel arguments and metadata");
  }
  auto &owned = AppendTask(RequireStream(*impl_, stream), params.type);
  owned.task.params = params;
  return &owned.task;
}
TaskSnapshot Model::Snapshot(aclmdlRITask handle) const {
  if (!impl_) {
    throw std::logic_error("model destroyed");
  }
  for (auto &stream : impl_->streams) {
    for (auto &owned : stream->tasks) {
      if (&owned->task != handle) {
        continue;
      }
      TaskSnapshot result;
      result.id = owned->task.task_id;
      result.type = owned->task.params.type;
      result.disabled = owned->disabled;
      result.setParamsCount = owned->setParamsCount;
      const auto &params = owned->task.params;
      if (result.type == ACL_MODEL_RI_TASK_KERNEL) {
        FunctionName(params.kernelTaskParams.funcHandle, result.function);
        result.numBlocks = params.kernelTaskParams.numBlocks;
        result.attributes = owned->attributes;
        std::lock_guard<std::mutex> lock(memoryMutex);
        if (params.kernelTaskParams.argsSize != 0 &&
            ContainsMemory(params.kernelTaskParams.args, params.kernelTaskParams.argsSize)) {
          auto ptr = static_cast<const unsigned char *>(params.kernelTaskParams.args);
          result.args.assign(ptr, ptr + params.kernelTaskParams.argsSize);
        }
      } else if (result.type == ACL_MODEL_RI_TASK_EVENT_RECORD) {
        result.syncAddress = reinterpret_cast<uintptr_t>(params.eventRecordTaskParams.event);
      } else if (result.type == ACL_MODEL_RI_TASK_EVENT_WAIT) {
        result.syncAddress = reinterpret_cast<uintptr_t>(params.eventWaitTaskParams.event);
      } else if (result.type == ACL_MODEL_RI_TASK_EVENT_RESET) {
        result.syncAddress = reinterpret_cast<uintptr_t>(params.eventResetTaskParams.event);
      } else if (result.type == ACL_MODEL_RI_TASK_VALUE_WRITE) {
        result.syncAddress = reinterpret_cast<uintptr_t>(params.valueWriteTaskParams.devAddr);
        result.syncValue = params.valueWriteTaskParams.value;
      } else if (result.type == ACL_MODEL_RI_TASK_VALUE_WAIT) {
        result.syncAddress = reinterpret_cast<uintptr_t>(params.valueWaitTaskParams.devAddr);
        result.syncValue = params.valueWaitTaskParams.value;
      }
      return result;
    }
  }
  throw std::invalid_argument("task does not belong to model");
}
std::vector<TaskSnapshot> Model::Tasks(aclrtStream handle) const {
  if (!impl_) {
    throw std::logic_error("model destroyed");
  }
  std::vector<TaskSnapshot> result;
  for (auto &task : RequireStream(*impl_, handle).tasks) {
    result.push_back(Snapshot(&task->task));
  }
  return result;
}
std::vector<LaunchRecord> Model::Launches() const {
  std::vector<LaunchRecord> result;
  if (impl_) {
    for (auto &stream : impl_->streams) {
      result.insert(result.end(), stream->launches.begin(), stream->launches.end());
    }
  }
  return result;
}
size_t Model::UpdateAttempts() const {
  return impl_ ? impl_->updateAttempts : 0;
}
size_t Model::SuccessfulUpdates() const {
  return impl_ ? impl_->successfulUpdates : 0;
}
void Model::Destroy() {
  if (!impl_) {
    return;
  }
  while (SkUtInvokeModelDestroyCallback(Handle()) == ACL_SUCCESS) {
  }
  for (auto &stream : impl_->streams) {
    for (auto &task : stream->tasks) {
      tasks.erase(&task->task);
      std::lock_guard<std::mutex> lock(memoryMutex);
      memory.erase(reinterpret_cast<uintptr_t>(task->args.data()));
    }
    streams.erase(stream.get());
  }
  models.erase(Handle());
  impl_.reset();
}

bool ModelStreams(aclmdlRI model, aclrtStream *output, uint32_t *count, aclError &result) {
  auto it = models.find(model);
  if (it == models.end()) {
    return false;
  }
  const auto &items = it->second->streams;
  result = ACL_SUCCESS;
  if (output != nullptr && *count < items.size()) {
    result = ACL_ERROR_INVALID_PARAM;
  }
  if (output != nullptr && result == ACL_SUCCESS) {
    for (size_t i = 0; i < items.size(); ++i) {
      output[i] = items[i].get();
    }
  }
  *count = static_cast<uint32_t>(items.size());
  return true;
}
bool StreamTasks(aclrtStream stream, aclmdlRITask *output, uint32_t *count, aclError &result) {
  auto it = streams.find(stream);
  if (it == streams.end()) {
    return false;
  }
  const auto &items = it->second->tasks;
  result = ACL_SUCCESS;
  if (output != nullptr && *count < items.size()) {
    result = ACL_ERROR_INVALID_PARAM;
  }
  if (output != nullptr && result == ACL_SUCCESS) {
    for (size_t i = 0; i < items.size(); ++i) {
      output[i] = &items[i]->task;
    }
  }
  *count = static_cast<uint32_t>(items.size());
  return true;
}
bool StreamId(aclrtStream stream, int32_t &id) {
  auto it = streams.find(stream);
  if (it == streams.end()) {
    return false;
  }
  id = it->second->id;
  return true;
}
void RecordUpdate(aclmdlRI model, aclError result) {
  auto it = models.find(model);
  if (it != models.end()) {
    ++it->second->updateAttempts;
    if (result == ACL_SUCCESS) {
      ++it->second->successfulUpdates;
    }
  }
}
bool ValidTaskUpdate(aclmdlRITask handle, const aclmdlRITaskParams &params) {
  if (tasks.count(handle) == 0) {
    return true;
  }
  constexpr size_t maxBytes = 1024 * 1024 * 1024;
  if (params.opInfoSize > maxBytes || (params.opInfoSize != 0 && params.opInfoPtr == nullptr)) {
    return false;
  }
  if (params.type != ACL_MODEL_RI_TASK_KERNEL) {
    return true;
  }
  const auto &kernel = params.kernelTaskParams;
  if (kernel.argsSize > maxBytes || (kernel.argsSize != 0 && kernel.args == nullptr)) {
    return false;
  }
  return kernel.cfg == nullptr || (kernel.cfg->numAttrs <= maxBytes / sizeof(aclrtLaunchKernelAttr) &&
                                   (kernel.cfg->numAttrs == 0 || kernel.cfg->attrs != nullptr));
}
void RecordTaskUpdate(aclmdlRITask handle, const aclmdlRITaskParams &params) {
  auto it = tasks.find(handle);
  if (it != tasks.end()) {
    ++it->second->setParamsCount;
    it->second->task.type = params.type;
    auto &owned = *it->second;
    if (params.opInfoSize != 0) {
      auto data = static_cast<const unsigned char *>(params.opInfoPtr);
      owned.opInfo = std::vector<unsigned char>(data, data + params.opInfoSize);
      owned.task.params.opInfoPtr = owned.opInfo.data();
    }
    if (params.type == ACL_MODEL_RI_TASK_KERNEL && params.kernelTaskParams.cfg != nullptr) {
      const auto &cfg = *params.kernelTaskParams.cfg;
      if (cfg.numAttrs != 0) {
        owned.attributes = std::vector<aclrtLaunchKernelAttr>(cfg.attrs, cfg.attrs + cfg.numAttrs);
      } else {
        owned.attributes.clear();
      }
      owned.config = {owned.attributes.data(), owned.attributes.size()};
      owned.task.params.kernelTaskParams.cfg = &owned.config;
    } else {
      owned.attributes.clear();
    }
    if (params.type == ACL_MODEL_RI_TASK_KERNEL && params.kernelTaskParams.isHostArgs) {
      const auto *data = static_cast<const unsigned char *>(params.kernelTaskParams.args);
      std::vector<unsigned char> copied;
      if (params.kernelTaskParams.argsSize != 0) {
        copied.assign(data, data + params.kernelTaskParams.argsSize);
      }
      std::lock_guard<std::mutex> lock(memoryMutex);
      memory.erase(reinterpret_cast<uintptr_t>(owned.args.data()));
      owned.args = std::move(copied);
      if (!owned.args.empty()) {
        memory[reinterpret_cast<uintptr_t>(owned.args.data())] = owned.args.size();
      }
      owned.task.params.kernelTaskParams.args = owned.args.data();
    }
  }
}
void RecordDisable(aclmdlRITask handle) {
  auto it = tasks.find(handle);
  if (it != tasks.end()) {
    it->second->disabled = true;
  }
}
void RecordLaunch(aclrtStream handle, const char *function, const char *scope, size_t size) {
  auto it = streams.find(handle);
  if (it != streams.end()) {
    it->second->launches.push_back({handle, function, scope});
    AppendKernel(*it->second, function, scope, size);
  }
}
bool FunctionName(aclrtFuncHandle handle, std::string &name) {
  auto it = functionHandles.find(handle);
  if (it == functionHandles.end()) {
    return false;
  }
  name = it->second->name;
  return true;
}
bool FunctionBinary(aclrtFuncHandle handle, aclrtBinHandle &binary) {
  if (!HasBinary(handle)) {
    return false;
  }
  binary = handle;
  return true;
}
bool FunctionAttribute(aclrtFuncHandle handle, aclrtFuncAttribute attr, int64_t &value) {
  auto it = functionHandles.find(handle);
  if (it == functionHandles.end()) {
    return false;
  }
  const auto &spec = it->second->spec;
  value = 0;
  if (attr == ACL_FUNC_ATTR_KERNEL_TYPE) {
    value = spec.type;
  } else if (attr == ACL_FUNC_ATTR_KERNEL_RATIO) {
    value = (static_cast<int64_t>(spec.cubeRatio) << 16) | spec.vectorRatio;
  } else if (attr == ACL_FUNC_ATTR_KERNEL_SCHED_MODE) {
    value = spec.scheMode;
  }
  return true;
}
aclrtFuncHandle ResolveFunction(const char *name) {
  return GetFunction(name);
}
bool HasBinary(aclrtBinHandle handle) {
  return functionHandles.count(handle) != 0;
}
bool FunctionAddress(aclrtFuncHandle handle, void **cube, void **vector) {
  auto it = functionHandles.find(handle);
  if (it == functionHandles.end()) {
    return false;
  }
  const auto &spec = it->second->spec;
  const bool mixed = spec.type == ACL_KERNEL_TYPE_AICORE || spec.type == ACL_KERNEL_TYPE_MIX;
  *cube =
      (spec.type == ACL_KERNEL_TYPE_CUBE || (mixed && spec.cubeRatio != 0)) ? it->second->code.data() + 16 : nullptr;
  *vector = (spec.type == ACL_KERNEL_TYPE_VECTOR || (mixed && spec.vectorRatio != 0))
                ? it->second->code.data() + (mixed && spec.cubeRatio != 0 ? 24 : 16)
                : nullptr;
  return true;
}
bool BinaryAddress(aclrtBinHandle handle, void **address, size_t *size) {
  auto it = functionHandles.find(handle);
  if (it == functionHandles.end()) {
    return false;
  }
  *address = it->second->code.data();
  *size = it->second->code.size();
  return true;
}
void SetKernelBindings(aclmdlRITask task, const std::vector<BinaryBinding> &bindings) {
  const auto owned = tasks.find(task);
  if (owned == tasks.end() || owned->second->task.params.type != ACL_MODEL_RI_TASK_KERNEL) {
    throw std::invalid_argument("metadata requires an owned kernel task");
  }
  const auto function = owned->second->task.params.kernelTaskParams.funcHandle;
  functionHandles.at(function)->bindings = bindings;
}
size_t BinaryMetadataCount(aclrtBinHandle handle) {
  const auto function = functionHandles.find(handle);
  if (function != functionHandles.end() && function->second->bindings) {
    return function->second->bindings->size();
  }
  void *cube = nullptr;
  void *vector = nullptr;
  if (!FunctionAddress(handle, &cube, &vector)) {
    return 0;
  }
  return static_cast<size_t>(cube != nullptr) + static_cast<size_t>(vector != nullptr);
}
bool BinaryMetadata(aclrtBinHandle handle, size_t count, void **data, size_t *sizes, int &result) {
  if (!HasBinary(handle)) {
    return false;
  }
  // RT_BINARY_TYPE_SK_INFO wire payload: reserved u32, cap u64, global
  // function offset, then four SK function offsets. All offsets stay in code.
  std::array<uint64_t, 6> values{0, 16, 32, 32, 32, 32};
  values[0] = functionHandles.at(handle)->spec.capability;
  const size_t payloadSize = sizeof(uint32_t) + sizeof(values);
  result = -1;
  if (count != BinaryMetadataCount(handle) || data == nullptr || sizes == nullptr) {
    return true;
  }
  for (size_t i = 0; i < count; ++i) {
    if (data[i] == nullptr || sizes[i] < payloadSize) {
      return true;
    }
    values[1] = 16 + i * 8;
    for (size_t j = 2; j < values.size(); ++j) {
      values[j] = 32 + i * 8;
    }
    const auto &bindings = functionHandles.at(handle)->bindings;
    if (bindings) {
      const auto &binding = bindings->at(i);
      values[0] = binding.capability;
      values[1] = binding.globalOffset;
      std::copy(binding.entryOffsets.begin(), binding.entryOffsets.end(), values.begin() + 2);
    }
    std::memset(data[i], 0, sizeof(uint32_t));
    std::memcpy(static_cast<unsigned char *>(data[i]) + sizeof(uint32_t), values.data(), sizeof(values));
    sizes[i] = payloadSize;
  }
  result = 0;
  return true;
}
void TrackAllocation(void *address, size_t size) {
  std::lock_guard<std::mutex> lock(memoryMutex);
  allocations[address] = size;
  memory[reinterpret_cast<uintptr_t>(address)] = size;
}
void ForgetAllocation(void *address) {
  std::lock_guard<std::mutex> lock(memoryMutex);
  allocations.erase(address);
  memory.erase(reinterpret_cast<uintptr_t>(address));
}
size_t OutstandingAllocations() {
  std::lock_guard<std::mutex> lock(memoryMutex);
  return allocations.size();
}
void SetRegisteredMemory(void *address, int value, size_t count) {
  std::lock_guard<std::mutex> lock(memoryMutex);
  if (ContainsMemory(address, count)) {
    std::memset(address, value, count);
  }
}
bool CopyRegisteredMemory(void *dst, const void *src, size_t count, aclrtMemcpyKind kind) {
  std::lock_guard<std::mutex> lock(memoryMutex);
  // Only dereference emulated device memory owned by this stub. Legacy UT
  // use opaque device addresses and retain their original no-copy behavior.
  const bool sourceKnown = ContainsMemory(src, count);
  const bool destinationKnown = ContainsMemory(dst, count);
  if ((kind == ACL_MEMCPY_DEVICE_TO_HOST && !sourceKnown) || (kind == ACL_MEMCPY_HOST_TO_DEVICE && !destinationKnown) ||
      (kind == ACL_MEMCPY_DEVICE_TO_DEVICE && (!sourceKnown || !destinationKnown)) ||
      (kind == ACL_MEMCPY_HOST_TO_HOST && (!sourceKnown || !destinationKnown))) {
    return false;
  }
  std::memcpy(dst, src, count);
  return true;
}
}  // namespace sk::test
