/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __SK_TASK_LOWERING_H__
#define __SK_TASK_LOWERING_H__

#include <cstdint>
#include <string>
#include <vector>

#include "sk_task_ir.h"

class ScopeCoreInfo;
class SuperKernelBaseNode;
class SuperKernelGraph;
class SuperKernelOptionsManager;

class SkTaskLowering {
 public:
  SkTaskLowering(const SuperKernelOptionsManager &options, const SuperKernelGraph &graph)
      : options_(options), graph_(graph) {}

  bool Run(std::string functionName, const std::vector<SuperKernelBaseNode *> &tasks, uint16_t scopeId,
           const ScopeCoreInfo &coreInfo, SkTaskIR &output) const;

 private:
  const SuperKernelOptionsManager &options_;
  const SuperKernelGraph &graph_;
};

#endif  // __SK_TASK_LOWERING_H__
