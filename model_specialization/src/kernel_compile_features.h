/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef KERNEL_COMPILE_FEATURES_H
#define KERNEL_COMPILE_FEATURES_H

#include <map>
#include <string>
#include <string_view>
#include <vector>
#include "aclmodel_specialization.h"

namespace model_spec {
namespace detail {

enum class CompileOptionType { BASIC, SK };

struct KernelOptionRule {
  std::string kernelPattern;  // Owned, normalized SuperKernel name pattern.
  std::vector<std::string> compileArguments;
};

// Internal objects are initialized on a fresh temporary manager, never on the active configuration.
class CompileOptionFeature final {
 public:
  [[nodiscard]] bool Init(const aclmdlRISpecOptions *options);
  std::vector<std::string> Get(CompileOptionType type, std::string_view entry, bool enableSk = false) const;

 private:
  bool InitBasic(const aclmdlRISpecOptions *options);
  bool InitSk(const aclmdlRISpecOptions *options);
  std::map<CompileOptionType, std::vector<KernelOptionRule>> rules_;
};

}  // namespace detail
}  // namespace model_spec
#endif
