/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascendc_backend.h"

#include "acl_runtime.h"
#include "tensor_buffer.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <memory>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace device_validation {
namespace {
double DecodeFp16Value(uint16_t bits);
int64_t DecodeSignedInteger(uint64_t bits, size_t element_size);

Status Fail(RunResult *result, Status status, const char *stage, const char *error_code, const std::string &reason) {
  result->stage = stage;
  result->error_code = error_code;
  result->reason = reason;
  result->report.stage_status = "failed";
  result->report.stage = stage;
  result->report.error_code = error_code;
  result->report.reason = reason;
  return result->status = status;
}

class ProfileScope {
 public:
  ProfileScope(InProcessProfiler *profiler, std::string output_dir)
      : profiler_(profiler), output_dir_(std::move(output_dir)) {}
  Status Begin(int32_t device_id) {
    if (profiler_ == nullptr) return Status::kOk;
    const Status status = profiler_->Start(device_id, output_dir_);
    started_ = status == Status::kOk;
    return status;
  }
  void Collect(RunResult *result) {
    if (profiler_ == nullptr || !started_) return;
    started_ = false;
    if (profiler_->Stop() == Status::kOk) {
      result->report.performance.actual.profiler_collected = true;
      result->report.performance.actual.profiler_collect_dir = output_dir_;
    }
  }
  ~ProfileScope() {
    if (profiler_ != nullptr && started_) (void)profiler_->Stop();
  }

 private:
  InProcessProfiler *profiler_;
  std::string output_dir_;
  bool started_ = false;
};

size_t DtypeSize(const std::string &dtype) {
  if (dtype == "float16" || dtype == "bfloat16") return 2;
  if (dtype == "float32" || dtype == "int32" || dtype == "uint32") return 4;
  if (dtype == "int64" || dtype == "uint64") return 8;
  if (dtype == "uint8" || dtype == "int8" || dtype == "bool") return 1;
  return 0;
}

size_t ElementCount(const std::vector<int64_t> &shape) {
  size_t count = 1;
  for (const auto dim : shape) {
    if (dim <= 0 || count > std::numeric_limits<size_t>::max() / static_cast<size_t>(dim))
      throw std::invalid_argument("invalid output shape");
    count *= static_cast<size_t>(dim);
  }
  return count;
}

void DecodeFloats(TensorView &view, size_t count, const std::vector<uint8_t> &bytes, size_t element_size) {
  view.values.resize(count);
  for (size_t i = 0; i < count; ++i) {
    float value = 0.0F;
    std::copy_n(bytes.data() + i * element_size, element_size, reinterpret_cast<uint8_t *>(&value));
    view.values[i] = value;
  }
}
void DecodeFp16(TensorView &view, size_t count, const std::vector<uint8_t> &bytes) {
  view.values.resize(count);
  for (size_t i = 0; i < count; ++i) {
    uint16_t bits = 0;
    std::copy_n(bytes.data() + i * sizeof(bits), sizeof(bits), reinterpret_cast<uint8_t *>(&bits));
    view.values[i] = DecodeFp16Value(bits);
  }
}
double DecodeFp16Value(uint16_t bits) {
  const uint32_t sign = (bits >> 15) & 1U;
  const uint32_t exponent = (bits >> 10) & 0x1fU;
  const uint32_t fraction = bits & 0x3ffU;
  if (exponent == 0) return (sign ? -1.0 : 1.0) * std::ldexp(static_cast<double>(fraction), -24);
  if (exponent == 31) return fraction == 0 ? (sign ? -INFINITY : INFINITY) : NAN;
  return (sign ? -1.0 : 1.0) * std::ldexp(1.0 + fraction / 1024.0, static_cast<int>(exponent) - 15);
}
void DecodeSignedIntegers(TensorView &view, size_t count, const std::vector<uint8_t> &bytes, size_t element_size) {
  view.integer_values.resize(count);
  for (size_t i = 0; i < count; ++i) {
    uint64_t bits = 0;
    std::copy_n(bytes.data() + i * element_size, element_size, reinterpret_cast<uint8_t *>(&bits));
    view.integer_values[i] = DecodeSignedInteger(bits, element_size);
  }
}
int64_t DecodeSignedInteger(uint64_t bits, size_t element_size) {
  if (element_size == sizeof(uint64_t)) {
    int64_t value = 0;
    std::copy_n(reinterpret_cast<const uint8_t *>(&bits), sizeof(value), reinterpret_cast<uint8_t *>(&value));
    return value;
  }
  const auto bit_width = element_size * 8;
  const uint64_t value_mask = (uint64_t{1} << bit_width) - 1;
  const uint64_t sign_bit = uint64_t{1} << (bit_width - 1);
  const uint64_t extended = bits & value_mask;
  if ((extended & sign_bit) == 0) return static_cast<int64_t>(extended);
  return static_cast<int64_t>(extended | (~value_mask));
}
void DecodeUnsignedIntegers(TensorView &view, size_t count, const std::vector<uint8_t> &bytes, size_t element_size) {
  view.unsigned_values.resize(count);
  for (size_t i = 0; i < count; ++i) {
    uint64_t value = 0;
    std::copy_n(bytes.data() + i * element_size, element_size, reinterpret_cast<uint8_t *>(&value));
    view.unsigned_values[i] = value;
  }
}
TensorView DecodeOutput(const std::string &dtype, const std::vector<int64_t> &shape,
                        const std::vector<uint8_t> &bytes) {
  const auto element_size = DtypeSize(dtype);
  const auto count = ElementCount(shape);
  if (element_size == 0 || bytes.size() != count * element_size) throw std::invalid_argument("invalid output buffer");
  TensorView view;
  view.dtype = dtype;
  view.shape = shape;
  if (dtype == "float32") {
    DecodeFloats(view, count, bytes, element_size);
  } else if (dtype == "float16") {
    DecodeFp16(view, count, bytes);
  } else if (dtype == "int32" || dtype == "int64" || dtype == "int8") {
    DecodeSignedIntegers(view, count, bytes, element_size);
  } else if (dtype == "uint8" || dtype == "uint32" || dtype == "uint64" || dtype == "bool") {
    DecodeUnsignedIntegers(view, count, bytes, element_size);
  } else {
    view.raw_bits.resize(count);
    for (size_t i = 0; i < count; ++i) std::copy_n(bytes.data() + i * element_size, element_size, &view.raw_bits[i]);
  }
  return view;
}

bool IsPassed(const SupportDecision &decision) {
  for (const auto &[capability, result] : decision.capabilities) {
    if (result.result == CapabilityResult::kFailed) return false;
  }
  return true;
}

RunRequest BuildRequest(const CaseConfig &config, const BackendInfo &info, const std::string &module_path) {
  RunRequest request;
  request.config = config;
  request.backend = info.name;
  request.soc = info.profile;
  request.abi = info.abi;
  request.variant = "fused";
  request.profiler = config.performance.profiler;
  request.module_path = module_path;
  request.warmup_count = config.performance.warmup_count;
  request.kernel_count = config.performance.kernel_count;
  request.dtype = config.inputs.front().dtype;
  for (const auto &input : config.inputs) request.shapes.push_back(input.shape);
  request.dynamic_shape = false;
  request.multi_output = config.outputs.size() > 1;
  request.config.performance.warmup_count = request.warmup_count;
  request.config.performance.kernel_count = request.kernel_count;
  request.capability_config = config;
  return request;
}

Status ValidateRunRequest(const RunRequest &request, const BackendInfo &info, RunResult *result) {
  if (request.module_path.empty() || request.config.inputs.empty() || request.config.outputs.empty() ||
      request.kernel_count == 0 || request.backend != info.name || request.soc != info.profile)
    return Fail(result, Status::kInvalidArgument, "preflight", "invalid_request", "invalid backend run request");
  return Status::kOk;
}

Status ResolveRunSupport(const RunRequest &request, const BackendInfo &info, const SocCapabilities &capabilities,
                         RunResult *result) {
  try {
    auto capability_request = request;
    capability_request.abi = info.abi;
    const auto &capability_config =
        request.capability_config.inputs.empty() ? request.config : request.capability_config;
    result->report.support = ResolveSupport(capability_config, info.capabilities, capabilities, capability_request);
  } catch (const std::invalid_argument &) {
    return Fail(result, Status::kUnsupported, "preflight", "unsupported_combination",
                "undeclared case/backend/SoC/ABI/shape combination");
  }
  if (!IsPassed(result->report.support)) {
    return Fail(result, Status::kUnsupported, "preflight", "capability_unavailable",
                "preflight capability check failed");
  }
  return Status::kOk;
}

Status ValidateHostInputCount(const RunRequest &request, RunResult *result) {
  if (request.host_inputs.size() != request.config.inputs.size()) {
    return Fail(result, Status::kInvalidArgument, "tensor", "input_count", "host input count does not match case");
  }
  return Status::kOk;
}

Status InitializeRuntimeAndModule(RunRequest *request, RunResult *result, const BackendInfo &info,
                                  const SocCapabilities &capabilities, AclRuntime *runtime, KernelModule *module) {
  if (runtime->Initialize(request->device_id) != Status::kOk ||
      module->Load(request->module_path, {"GetTilingDataSize", "AutofuseTiling", request->abi,
                                          static_cast<int32_t>(request->config.inputs.size()),
                                          static_cast<int32_t>(request->config.outputs.size())}) != Status::kOk) {
    return Fail(result, Status::kRuntimeError, "init", "runtime_or_module_init",
                "runtime initialization or module loading failed");
  }
  if (request->abi != module->abi()) {
    result->report.abi = module->abi();
    return Fail(result, Status::kInvalidArgument, "preflight", "abi_mismatch", "request ABI does not match module ABI");
  }
  request->abi = module->abi();
  result->report.abi = request->abi;
  try {
    const auto &capability_config =
        request->capability_config.inputs.empty() ? request->config : request->capability_config;
    result->report.support = ResolveSupport(capability_config, info.capabilities, capabilities, *request);
  } catch (const std::invalid_argument &) {
    return Fail(result, Status::kUnsupported, "preflight", "unsupported_combination",
                "selected module ABI is not allowed by the profile");
  }
  if (!IsPassed(result->report.support)) {
    return Fail(result, Status::kUnsupported, "preflight", "capability_unavailable",
                "selected module ABI is unavailable");
  }
  return Status::kOk;
}

Status SetupInputBuffers(const RunRequest &request, AclRuntime *runtime, std::vector<TensorBuffer> *inputs,
                         RunResult *result) {
  inputs->resize(request.config.inputs.size());
  for (size_t i = 0; i < inputs->size(); ++i) {
    if ((*inputs)[i].Allocate({request.config.inputs[i].shape, request.config.inputs[i].dtype}, runtime) !=
            Status::kOk ||
        (*inputs)[i].CopyToDevice(request.host_inputs[i].data(), request.host_inputs[i].size()) != Status::kOk) {
      return Fail(result, Status::kRuntimeError, "tensor", "input_allocation_or_h2d",
                  "input allocation or H2D copy failed");
    }
  }
  return Status::kOk;
}

Status SetupOutputBuffers(const RunRequest &request, AclRuntime *runtime, std::vector<TensorBuffer> *outputs,
                          RunResult *result) {
  outputs->resize(request.config.outputs.size());
  for (size_t i = 0; i < outputs->size(); ++i) {
    if ((*outputs)[i].Allocate({request.config.outputs[i].shape, request.config.outputs[i].dtype}, runtime) !=
        Status::kOk) {
      return Fail(result, Status::kRuntimeError, "tensor", "output_allocation", "output allocation failed");
    }
  }
  return Status::kOk;
}

Status RunTilingSetup(KernelModule *module, const DeviceProfile &profile, std::vector<uint8_t> *tiling_data,
                      uint64_t *workspace_size, uint32_t *block_dim, double *tiling_us, RunResult *result) {
  size_t tiling_size = 0;
  const auto tiling_start = std::chrono::steady_clock::now();
  if (module->GetTilingDataSize(&tiling_size) != Status::kOk || tiling_size == 0 ||
      (profile.max_tiling_bytes != 0 && tiling_size > profile.max_tiling_bytes)) {
    return Fail(result, Status::kRuntimeError, "tiling", "tiling_size_limit",
                "tiling data size is invalid or exceeds profile limit");
  }
  try {
    tiling_data->resize(tiling_size);
  } catch (const std::bad_alloc &) {
    return Fail(result, Status::kRuntimeError, "tiling", "tiling_allocation", "tiling allocation failed");
  }
  if (module->RunTiling(tiling_data->data(), workspace_size, block_dim, nullptr) != Status::kOk) {
    return Fail(result, Status::kRuntimeError, "tiling", "tiling_execution", "tiling execution failed");
  }
  *tiling_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - tiling_start).count();
  return Status::kOk;
}

Status SetupWorkspace(uint64_t workspace_size, const DeviceProfile &profile, AclRuntime *runtime,
                      TensorBuffer *workspace, RunResult *result) {
  if (profile.max_workspace_bytes != 0 && workspace_size > profile.max_workspace_bytes) {
    return Fail(result, Status::kRuntimeError, "workspace", "workspace_size_limit_or_allocation",
                "workspace size exceeds profile limit or allocation failed");
  }
  if (workspace_size > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return Fail(result, Status::kRuntimeError, "workspace", "workspace_size_limit_or_allocation",
                "workspace size exceeds tensor shape limit");
  }
  if (workspace_size != 0 &&
      workspace->Allocate({{static_cast<int64_t>(workspace_size)}, "uint8"}, runtime) != Status::kOk) {
    return Fail(result, Status::kRuntimeError, "workspace", "workspace_size_limit_or_allocation",
                "workspace size exceeds profile limit or allocation failed");
  }
  return Status::kOk;
}

Status ValidateLaunchConfiguration(uint32_t block_dim, const DeviceProfile &profile, RunResult *result) {
  if (block_dim == 0 || (profile.max_block_dimension != 0 && block_dim > profile.max_block_dimension)) {
    return Fail(result, Status::kRuntimeError, "launch", "block_dimension_limit",
                "block dimension exceeds profile limit");
  }
  return Status::kOk;
}

Status RunKernelIterations(KernelModule *module, AclRuntime *runtime, uint32_t block_dim,
                           std::vector<void *> &input_ptrs, std::vector<void *> &output_ptrs, TensorBuffer *workspace,
                           std::vector<uint8_t> &tiling_data, size_t count, const char *sync_error, RunResult *result) {
  for (size_t i = 0; i < count; ++i) {
    if (module->Launch(runtime, block_dim, runtime->stream(), input_ptrs.data(),
                       static_cast<int32_t>(input_ptrs.size()), output_ptrs.data(),
                       static_cast<int32_t>(output_ptrs.size()), workspace->device_ptr(),
                       tiling_data.data()) != Status::kOk) {
      return Fail(result, Status::kRuntimeError, "launch", "launch_failed", "kernel launch failed");
    }
    if (runtime->Synchronize() != Status::kOk) {
      return Fail(result, Status::kRuntimeError, "sync", "sync_failed", sync_error);
    }
  }
  return Status::kOk;
}

Status MeasureKernelExecutions(KernelModule *module, AclRuntime *runtime, uint32_t block_dim,
                               const std::vector<TensorBuffer> &inputs, const std::vector<TensorBuffer> &outputs,
                               TensorBuffer *workspace, std::vector<uint8_t> &tiling_data, const RunRequest &request,
                               RunResult *result, std::vector<double> *samples, bool *kernel_timing) {
  std::vector<void *> input_ptrs;
  std::vector<void *> output_ptrs;
  for (const auto &input : inputs) input_ptrs.push_back(input.device_ptr());
  for (const auto &output : outputs) output_ptrs.push_back(output.device_ptr());
  // The host clock is only a wall-clock fallback. It must never be reported as
  // device kernel duration when profiler export is unavailable.
  module->EnableKernelTiming(false);
  if (RunKernelIterations(module, runtime, block_dim, input_ptrs, output_ptrs, workspace, tiling_data,
                          request.warmup_count, "warmup synchronization failed", result) != Status::kOk)
    return result->status;
  if (RunKernelIterations(module, runtime, block_dim, input_ptrs, output_ptrs, workspace, tiling_data,
                          request.kernel_count, "kernel synchronization failed", result) != Status::kOk)
    return result->status;
  module->TakeKernelTimingSamples();
  *kernel_timing = false;
  auto start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < request.kernel_count; ++i) {
    if (RunKernelIterations(module, runtime, block_dim, input_ptrs, output_ptrs, workspace, tiling_data, 1,
                            "kernel synchronization failed", result) != Status::kOk)
      return result->status;
    samples->push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    start = std::chrono::steady_clock::now();
  }
  return Status::kOk;
}

Status FetchAndDecodeOutputs(const RunRequest &request, std::vector<TensorBuffer> &outputs, RunResult *result) {
  for (size_t i = 0; i < outputs.size(); ++i) {
    std::vector<uint8_t> host_output(outputs[i].size());
    if (outputs[i].CopyToHost(host_output.data(), host_output.size()) != Status::kOk) {
      return Fail(result, Status::kRuntimeError, "D2H", "d2h_failed", "output D2H copy failed");
    }
    try {
      result->outputs.push_back(
          DecodeOutput(request.config.outputs[i].dtype, request.config.outputs[i].shape, host_output));
      result->output_bytes.push_back(std::move(host_output));
    } catch (const std::invalid_argument &error) {
      return Fail(result, Status::kRuntimeError, "verification", "output_decode", error.what());
    }
  }
  return Status::kOk;
}

Status SummarizeRunPerformance(const RunRequest &request, const std::vector<double> &samples, bool kernel_timing,
                               bool profiler_available, double tiling_us, uint64_t workspace_size, uint32_t block_dim,
                               ProfileScope *profiler_scope, RunResult *result) {
  try {
    const bool kernel_timing_used = kernel_timing && !samples.empty();
    result->report.performance = SummarizeSamplesWithRuntimeState(
        samples, request.config.performance, request.profiler, profiler_available, false,
        kernel_timing_used ? "host_clock_kernel_launch_us" : "runner_wall_clock");
  } catch (const std::invalid_argument &error) {
    return Fail(result, Status::kInvalidArgument, "verification", "timing_summary", error.what());
  }
  result->report.performance.actual.warmup_count = request.warmup_count;
  result->report.performance.actual.samples = samples;
  result->report.performance.actual.kernel_count = request.kernel_count;
  result->report.performance.actual.tiling_time_us = tiling_us;
  result->report.performance.actual.workspace_size = workspace_size;
  result->report.performance.actual.block_dimension = block_dim;
  result->report.metric = kernel_timing && !samples.empty() ? "device_kernel_duration" : "runner_wall_clock";
  result->report.performance.actual.metric = result->report.metric;
  result->report.performance.actual.unit = result->report.metric == "device_kernel_duration" ? "us" : "ms";
  profiler_scope->Collect(result);
  result->report.stage_status = "passed";
  result->stage = "execution";
  result->report.stage = result->stage;
  return result->status = Status::kOk;
}
}  // namespace

RunRequest BuildRunRequest(const CaseConfig &config, const BackendInfo &info, const std::string &module_path) {
  return BuildRequest(config, info, module_path);
}

TensorView DecodeOutputForTest(const std::string &dtype, const std::vector<int64_t> &shape,
                               const std::vector<uint8_t> &bytes) {
  return DecodeOutput(dtype, shape, bytes);
}

AscendCBackend::AscendCBackend(DeviceProfile profile, RuntimeApi *runtime_api, ModuleLoader *module_loader)
    : profile_(std::move(profile)), runtime_api_(runtime_api), module_loader_(module_loader) {
  profile_.capabilities.profiler_available = IsProfilerAvailable(profile_);
}

BackendInfo AscendCBackend::GetCapabilities() const {
  return {profile_.real_device_backend,
          profile_.capabilities.soc,
          "AutofuseLaunchV2",
          {true, true, true, true, true, false, false, true}};
}

Status AscendCBackend::Validate(const CaseConfig &config) const {
  if (config.inputs.empty() || config.outputs.empty()) return Status::kInvalidArgument;
  const auto info = GetCapabilities();
  const auto request = BuildRequest(config, info, "");
  try {
    const auto decision = ResolveSupport(config, info.capabilities, profile_.capabilities, request);
    return IsPassed(decision) ? Status::kOk : Status::kUnsupported;
  } catch (const std::invalid_argument &) {
    return Status::kUnsupported;
  }
}

Status AscendCBackend::InitializeReport(RunResult &result, const RunRequest &request, const BackendInfo &info) const {
  result.report.case_id = request.config.case_id;
  result.report.backend = info.name;
  result.report.abi = info.abi;
  result.report.requested_abi = request.abi;
  result.report.variant = request.variant;
  result.report.soc_profile = info.profile;
  result.report.performance.declared = request.config.performance;
  result.report.profile = info.profile;
  result.report.run_parameters = {
      {"device_id", request.device_id}, {"warmup", request.warmup_count}, {"repeat", request.kernel_count}};
  if (request.abi != "AutofuseLaunchV2" && request.abi != "AutofuseLaunch") {
    return Fail(&result, Status::kInvalidArgument, "preflight", "abi_mismatch",
                "requested ABI does not match backend ABI contract");
  }
  if (request.device_id < 0) {
    return Fail(&result, Status::kInvalidArgument, "preflight", "invalid_device_id", "device_id must be non-negative");
  }
  return Status::kOk;
}

Status AscendCBackend::Run(RunRequest &request, RunResult *result) {
  const auto info = GetCapabilities();
  if (result == nullptr) return Status::kInvalidArgument;
  const auto preflight = InitializeReport(*result, request, info);
  if (preflight != Status::kOk) return preflight;
  if (ValidateRunRequest(request, info, result) != Status::kOk) return result->status;
  result->report.performance.actual.profiler_requested = request.profiler;
  result->report.performance.actual.profiler_tool_available = profile_.capabilities.profiler_available;
  result->report.performance.actual.profiler_collected = false;
  result->report.performance.actual.timing_source = "runner_wall_clock";
  if (ResolveRunSupport(request, info, profile_.capabilities, result) != Status::kOk) return result->status;
  if (ValidateHostInputCount(request, result) != Status::kOk) return result->status;

  auto runtime_api = runtime_api_ == nullptr ? CreateAclRuntimeApi() : runtime_api_;
  if (runtime_api == nullptr) {
    return Fail(result, Status::kRuntimeError, "init", "runtime_unavailable", "runtime API is unavailable");
  }
  AclRuntime runtime(runtime_api);
  KernelModule module(module_loader_ == nullptr ? CreateModuleLoader() : module_loader_);
  if (InitializeRuntimeAndModule(&request, result, info, profile_.capabilities, &runtime, &module) != Status::kOk)
    return result->status;

  ProfileScope profiler_scope(profiler_, profile_dir_);
  if (profiler_ != nullptr && profiler_scope.Begin(request.device_id) != Status::kOk) {
    return Fail(result, Status::kRuntimeError, "profiler", "profiler_start_failed",
                "in-process profiler start failed: " + profiler_->last_error());
  }

  std::vector<TensorBuffer> inputs;
  std::vector<TensorBuffer> outputs;
  if (SetupInputBuffers(request, &runtime, &inputs, result) != Status::kOk ||
      SetupOutputBuffers(request, &runtime, &outputs, result) != Status::kOk)
    return result->status;

  std::vector<uint8_t> tiling_data;
  uint64_t workspace_size = 0;
  uint32_t block_dim = 0;
  double tiling_us = 0.0;
  if (RunTilingSetup(&module, profile_, &tiling_data, &workspace_size, &block_dim, &tiling_us, result) != Status::kOk)
    return result->status;
  TensorBuffer workspace;
  if (SetupWorkspace(workspace_size, profile_, &runtime, &workspace, result) != Status::kOk ||
      ValidateLaunchConfiguration(block_dim, profile_, result) != Status::kOk)
    return result->status;

  std::vector<double> samples;
  bool kernel_timing = false;
  if (MeasureKernelExecutions(&module, &runtime, block_dim, inputs, outputs, &workspace, tiling_data, request, result,
                              &samples, &kernel_timing) != Status::kOk)
    return result->status;
  if (FetchAndDecodeOutputs(request, outputs, result) != Status::kOk) return result->status;
  return SummarizeRunPerformance(request, samples, kernel_timing, profile_.capabilities.profiler_available, tiling_us,
                                 workspace_size, block_dim, &profiler_scope, result);
}

std::unique_ptr<ExecutionBackend> CreateExecutionBackend(const std::string &name,
                                                         const std::filesystem::path &profile_path) {
  if (name != "ascendc_real_device") return nullptr;
  if (profile_path.empty()) throw std::invalid_argument("device profile is required");
  return std::make_unique<AscendCBackend>(LoadDeviceProfile(profile_path));
}
}  // namespace device_validation
