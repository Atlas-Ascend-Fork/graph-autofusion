/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "acl_runtime.h"
#include "graph/error_codes.h"

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace device_validation {
using GetTilingDataSizeFunc = size_t (*)();
using AutofuseTilingFunc = ge::graphStatus (*)(void *, uint64_t *, uint32_t *, void *);
using AutofuseLaunchV2Func = uint32_t (*)(uint32_t, void *, void **, int32_t, void **, int32_t, void *, void *);
using AutofuseLaunchLegacy1In1OutFunc = int64_t (*)(uint32_t, void *, void *, void *, void *, void *);
using AutofuseLaunchLegacy2In1OutFunc = int64_t (*)(uint32_t, void *, void *, void *, void *, void *, void *);
using AutofuseLaunchLegacy3In1OutFunc = int64_t (*)(uint32_t, void *, void *, void *, void *, void *, void *, void *);

enum class LaunchAbiKind { kNone, kV2, kLegacy };

struct AbiSpec {
  std::string get_tiling_data_size = "GetTilingDataSize";
  std::string tiling = "AutofuseTiling";
  std::string launch = "AutofuseLaunchV2";
  int32_t input_count = 3;
  int32_t output_count = 1;
};

class ModuleLoader {
 public:
  virtual ~ModuleLoader() = default;
  virtual void *Load(const std::filesystem::path &) = 0;
  virtual void *Symbol(void *, const char *) = 0;
  virtual Status Unload(void *) = 0;
};

ModuleLoader *CreateModuleLoader();

class KernelModule {
 public:
  explicit KernelModule(ModuleLoader *loader) : loader_(loader) {}
  ~KernelModule();
  Status Load(const std::filesystem::path &, const AbiSpec &);
  Status Unload();
  Status GetTilingDataSize(size_t *);
  Status RunTiling(void *, uint64_t *, uint32_t *, void *);
  Status LaunchV2(uint32_t, void *, void **, int32_t, void **, int32_t, void *, void *);
  Status LaunchLegacy(uint32_t, void *, void **, int32_t, void **, int32_t, void *, void *);
  bool ValidLaunchRequest(AclRuntime *, void *, void *, int32_t, int32_t, void **, void **) const;
  Status Launch(AclRuntime *, uint32_t, void *, void **, int32_t, void **, int32_t, void *, void *);
  void EnableKernelTiming(bool enabled) {
    measure_kernel_duration_ = enabled;
    if (!enabled) kernel_timing_us_.clear();
  }
  std::vector<uint64_t> TakeKernelTimingSamples() {
    auto copy = kernel_timing_us_;
    kernel_timing_us_.clear();
    return copy;
  }
  bool IsLoaded() const {
    return handle_ != nullptr;
  }
  const std::string &missing_symbol() const {
    return missing_symbol_;
  }
  std::string abi() const {
    return launch_abi_ == LaunchAbiKind::kLegacy ? "AutofuseLaunch" : "AutofuseLaunchV2";
  }

 private:
  ModuleLoader *loader_;
  void *handle_ = nullptr;
  GetTilingDataSizeFunc get_size_ = nullptr;
  AutofuseTilingFunc tiling_ = nullptr;
  AutofuseLaunchV2Func launch_v2_ = nullptr;
  AutofuseLaunchLegacy1In1OutFunc launch_legacy_1_1_ = nullptr;
  AutofuseLaunchLegacy2In1OutFunc launch_legacy_2_1_ = nullptr;
  AutofuseLaunchLegacy3In1OutFunc launch_legacy_3_1_ = nullptr;
  int32_t legacy_input_count_ = 0;
  int32_t legacy_output_count_ = 0;
  LaunchAbiKind launch_abi_ = LaunchAbiKind::kNone;
  std::string missing_symbol_;
  bool measure_kernel_duration_ = false;
  std::vector<uint64_t> kernel_timing_us_;
};
}  // namespace device_validation
