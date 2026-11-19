/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "feature_manager.h"
#include <algorithm>
#include <limits>
#include <thread>
#include <type_traits>
#include <utility>

namespace model_spec {

uint64_t GetCompileWorkerLimit() {
  constexpr uint64_t COMPILE_CPU_PERCENT = 70;
  constexpr uint64_t FULL_PERCENT = 100;
  return std::max<uint64_t>(1, std::thread::hardware_concurrency() * COMPILE_CPU_PERCENT / FULL_PERCENT);
}

namespace {
void InitModelLevelOptions(const aclmdlRISpecOptions *options, uint64_t &compileJobs, bool &enableSk) {
  compileJobs = options == nullptr ? 0 : options->jobs;
  // jobs 是无符号字段，负数赋值后的转换值超过 int64_t 正数范围。
  if (compileJobs == 0 || compileJobs > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    compileJobs = GetCompileWorkerLimit();
  }
  enableSk = options != nullptr && options->enableSK;
}
}  // namespace

bool FeatureManager::Init(const aclmdlRISpecOptions *options, bool isScopeJit) {
  FeatureManager temporary;
  if (!temporary.compileOptions_.Init(options)) {
    return false;
  }
  InitModelLevelOptions(options, temporary.compileJobs_, temporary.enableSk_);
  temporary.isScopeJit_ = isScopeJit;
  temporary.initialized_ = true;
  Swap(temporary);
  return true;
}

void FeatureManager::Swap(FeatureManager &other) noexcept {
  // Commit must not fail after only part of the configuration has been exchanged.
  static_assert(std::is_nothrow_swappable_v<detail::CompileOptionFeature>);
  using std::swap;
  swap(compileOptions_, other.compileOptions_);
  swap(compileJobs_, other.compileJobs_);
  swap(enableSk_, other.enableSk_);
  swap(isScopeJit_, other.isScopeJit_);
  swap(initialized_, other.initialized_);
}

}  // namespace model_spec
