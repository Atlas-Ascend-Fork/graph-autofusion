/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef STATIC_NORMAL_FEATURE_MANAGER_H
#define STATIC_NORMAL_FEATURE_MANAGER_H

#include <cstdint>
#include <string_view>
#include "kernel_compile_features.h"
#include "sk_log.h"

namespace model_spec {

enum class Feature : uint32_t { COMPILE_JOBS = 0, ENABLE_SK = 1, BASIC_COMPILE_OPTIONS = 2, SK_COMPILE_OPTIONS = 3 };

// Init owns its inputs. Init/destruction require exclusive access; Get supports concurrent readers.
class FeatureManager final {
 public:
  FeatureManager() = default;
  FeatureManager(const FeatureManager &) = delete;
  FeatureManager &operator=(const FeatureManager &) = delete;

  // Configuration errors return false; exceptions propagate. Either failure preserves the old configuration.
  [[nodiscard]] bool Init(const aclmdlRISpecOptions *options);
  template <Feature F>
  auto Get() const;
  template <Feature F>
  auto Get(std::string_view kernelEntry) const;

 private:
  void Swap(FeatureManager &other) noexcept;
  detail::CompileOptionFeature compileOptions_;
  uint64_t compileJobs_{0};
  bool enableSk_{false};
  bool initialized_{false};
};

template <Feature F>
auto FeatureManager::Get() const {
  static_assert(F == Feature::COMPILE_JOBS || F == Feature::ENABLE_SK,
                "Unsupported model feature or missing kernel entry");
  if (!initialized_) {
    SK_DLOGE("FeatureManager must be initialized before Get");
  }
  if constexpr (F == Feature::COMPILE_JOBS) {
    return compileJobs_;
  }
  if constexpr (F == Feature::ENABLE_SK) {
    return enableSk_;
  }
}

template <Feature F>
auto FeatureManager::Get(std::string_view kernelEntry) const {
  static_assert(F == Feature::BASIC_COMPILE_OPTIONS || F == Feature::SK_COMPILE_OPTIONS,
                "Unsupported kernel feature or unexpected kernel entry");
  if (!initialized_) {
    SK_DLOGE("FeatureManager must be initialized before Get");
    return std::vector<std::string>{};
  }
  if (kernelEntry.empty()) {
    SK_DLOGE("Static compile kernel entry must not be empty");
    return std::vector<std::string>{};
  }
  if constexpr (F == Feature::BASIC_COMPILE_OPTIONS) {
    return compileOptions_.Get(detail::CompileOptionType::BASIC, kernelEntry);
  }
  if constexpr (F == Feature::SK_COMPILE_OPTIONS) {
    if (!enableSk_) {
      SK_DLOGI("SuperKernel is disabled, return empty SK compile options");
      return std::vector<std::string>{};
    }
    return compileOptions_.Get(detail::CompileOptionType::SK, kernelEntry, true);
  }
}

}  // namespace model_spec
#endif
