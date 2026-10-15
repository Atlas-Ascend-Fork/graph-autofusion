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
#include <type_traits>
#include <utility>

namespace model_spec {
namespace {
void InitModelLevelOptions(const aclmdlRISpecOptions *options, uint64_t &compileJobs, bool &enableSk) {
  compileJobs = options == nullptr || options->jobs == 0 ? 1 : options->jobs;
  enableSk = options != nullptr && options->enableSK;
}
}  // namespace

bool FeatureManager::Init(const aclmdlRISpecOptions *options) {
  FeatureManager temporary;
  if (!temporary.compileOptions_.Init(options)) {
    return false;
  }
  InitModelLevelOptions(options, temporary.compileJobs_, temporary.enableSk_);
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
  swap(initialized_, other.initialized_);
}

}  // namespace model_spec
