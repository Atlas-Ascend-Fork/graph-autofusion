/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_module.h"
#include "host_clock.h"

#include <dlfcn.h>

namespace device_validation {
namespace {
constexpr int32_t kMaxLaunchTensorCount = 1 << 20;
class PosixModuleLoader final : public ModuleLoader {
 public:
  void *Load(const std::filesystem::path &path) override {
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  }
  void *Symbol(void *handle, const char *name) override {
    return dlsym(handle, name);
  }
  Status Unload(void *handle) override {
    return dlclose(handle) == 0 ? Status::kOk : Status::kRuntimeError;
  }
};
}  // namespace

ModuleLoader *CreateModuleLoader() {
  static PosixModuleLoader loader;
  return &loader;
}

KernelModule::~KernelModule() {
  Unload();
}

Status KernelModule::Load(const std::filesystem::path &path, const AbiSpec &abi) {
  if (loader_ == nullptr || handle_ != nullptr) return Status::kInvalidArgument;
  handle_ = loader_->Load(path);
  if (handle_ == nullptr) return Status::kNotFound;
  missing_symbol_.clear();
  get_size_ = reinterpret_cast<GetTilingDataSizeFunc>(loader_->Symbol(handle_, abi.get_tiling_data_size.c_str()));
  if (get_size_ == nullptr) {
    missing_symbol_ = abi.get_tiling_data_size;
    Unload();
    return Status::kNotFound;
  }
  tiling_ = reinterpret_cast<AutofuseTilingFunc>(loader_->Symbol(handle_, abi.tiling.c_str()));
  if (tiling_ == nullptr) {
    missing_symbol_ = abi.tiling;
    Unload();
    return Status::kNotFound;
  }
  if (abi.launch == "AutofuseLaunchV2") {
    launch_v2_ = reinterpret_cast<AutofuseLaunchV2Func>(loader_->Symbol(handle_, abi.launch.c_str()));
    if (launch_v2_ != nullptr) {
      launch_abi_ = LaunchAbiKind::kV2;
      return Status::kOk;
    }
  } else if (abi.launch == "AutofuseLaunch") {
    void *legacy = loader_->Symbol(handle_, abi.launch.c_str());
    if (legacy != nullptr &&
        ((abi.input_count == 1 && abi.output_count == 1) || (abi.input_count == 2 && abi.output_count == 1) ||
         (abi.input_count == 3 && abi.output_count == 1))) {
      launch_legacy_1_1_ = reinterpret_cast<AutofuseLaunchLegacy1In1OutFunc>(legacy);
      launch_legacy_2_1_ = reinterpret_cast<AutofuseLaunchLegacy2In1OutFunc>(legacy);
      launch_legacy_3_1_ = reinterpret_cast<AutofuseLaunchLegacy3In1OutFunc>(legacy);
      legacy_input_count_ = abi.input_count;
      legacy_output_count_ = abi.output_count;
      launch_abi_ = LaunchAbiKind::kLegacy;
      return Status::kOk;
    }
  }
  missing_symbol_ = abi.launch;
  Unload();
  return Status::kNotFound;
}

Status KernelModule::Unload() {
  if (handle_ == nullptr) return Status::kOk;
  const auto status = loader_->Unload(handle_);
  if (status != Status::kOk) return status;
  handle_ = nullptr;
  get_size_ = nullptr;
  tiling_ = nullptr;
  launch_v2_ = nullptr;
  launch_legacy_1_1_ = nullptr;
  launch_legacy_2_1_ = nullptr;
  launch_legacy_3_1_ = nullptr;
  legacy_input_count_ = legacy_output_count_ = 0;
  launch_abi_ = LaunchAbiKind::kNone;
  return status;
}

Status KernelModule::GetTilingDataSize(size_t *size) {
  if (get_size_ == nullptr || size == nullptr) return Status::kInvalidArgument;
  *size = get_size_();
  return Status::kOk;
}

Status KernelModule::RunTiling(void *tiling_data, uint64_t *workspace_size, uint32_t *block_dim, void *extra) {
  if (tiling_ == nullptr || tiling_data == nullptr || workspace_size == nullptr || block_dim == nullptr)
    return Status::kInvalidArgument;
  return tiling_(tiling_data, workspace_size, block_dim, extra) == ge::GRAPH_SUCCESS ? Status::kOk
                                                                                     : Status::kRuntimeError;
}

Status KernelModule::Launch(AclRuntime *runtime, uint32_t block_dim, void *stream, void **inputs, int32_t input_count,
                            void **outputs, int32_t output_count, void *workspace, void *tiling_data) {
  if (!ValidLaunchRequest(runtime, stream, tiling_data, input_count, output_count, inputs, outputs))
    return Status::kInvalidArgument;
  if (launch_abi_ == LaunchAbiKind::kV2)
    return LaunchV2(block_dim, stream, inputs, input_count, outputs, output_count, workspace, tiling_data);
  return LaunchLegacy(block_dim, stream, inputs, input_count, outputs, output_count, workspace, tiling_data);
}
bool KernelModule::ValidLaunchRequest(AclRuntime *runtime, void *stream, void *tiling_data, int32_t input_count,
                                      int32_t output_count, void **inputs, void **outputs) const {
  if (launch_v2_ == nullptr && launch_legacy_1_1_ == nullptr && launch_legacy_2_1_ == nullptr &&
      launch_legacy_3_1_ == nullptr)
    return false;
  if (runtime == nullptr || !runtime->initialized() || stream == nullptr || stream != runtime->stream()) return false;
  if (tiling_data == nullptr || input_count < 0 || output_count < 0 || input_count > kMaxLaunchTensorCount ||
      output_count > kMaxLaunchTensorCount)
    return false;
  if ((input_count > 0 && inputs == nullptr) || (output_count > 0 && outputs == nullptr)) return false;
  return true;
}

Status KernelModule::LaunchV2(uint32_t block_dim, void *stream, void **inputs, int32_t input_count, void **outputs,
                              int32_t output_count, void *workspace, void *tiling_data) {
  const auto t0 = measure_kernel_duration_ ? MonotonicMilliseconds() : 0.0;
  const auto launch_status =
      launch_v2_(block_dim, stream, inputs, input_count, outputs, output_count, workspace, tiling_data);
  if (measure_kernel_duration_ && launch_status == 0)
    kernel_timing_us_.push_back(static_cast<uint64_t>((MonotonicMilliseconds() - t0) * 1000.0));
  return launch_status == 0 ? Status::kOk : Status::kRuntimeError;
}

Status KernelModule::LaunchLegacy(uint32_t block_dim, void *stream, void **inputs, int32_t input_count, void **outputs,
                                  int32_t output_count, void *workspace, void *tiling_data) {
  if (input_count != legacy_input_count_ || output_count != legacy_output_count_) return Status::kInvalidArgument;
  const auto t0 = measure_kernel_duration_ ? MonotonicMilliseconds() : 0.0;
  int64_t result = 1;
  if (input_count == 1) {
    result = launch_legacy_1_1_(block_dim, stream, inputs[0], outputs[0], workspace, tiling_data);
  } else if (input_count == 2) {
    result = launch_legacy_2_1_(block_dim, stream, inputs[0], inputs[1], outputs[0], workspace, tiling_data);
  } else if (input_count == 3) {
    result = launch_legacy_3_1_(block_dim, stream, inputs[0], inputs[1], inputs[2], outputs[0], workspace, tiling_data);
  } else {
    return Status::kInvalidArgument;
  }
  if (measure_kernel_duration_ && result == 0)
    kernel_timing_us_.push_back(static_cast<uint64_t>((MonotonicMilliseconds() - t0) * 1000.0));
  return result == 0 ? Status::kOk : Status::kRuntimeError;
}
}  // namespace device_validation
