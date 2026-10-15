/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __SK_VERIFY_SCOPE_SPLIT_H__
#define __SK_VERIFY_SCOPE_SPLIT_H__

#include <cstdint>
#include <vector>

#include "sk_scope_info.h"
#include "sk_types.h"
#include "super_kernel.h"

class SuperKernelGraph;
class SuperKernelOptionsManager;

struct ScopeVerifySplitRecord {
  uint64_t nodeId = INVALID_TASK_ID;
  aclskScopeVerifySplitType splitType = ACLSK_SCOPE_VERIFY_SPLIT_BEFORE_NODE;
  aclskScopeVerifySplitReason splitReason = ACLSK_SCOPE_VERIFY_DEADLOCK_DETECTED;
};

// Run ScheMode full-core split pass and append scope verify split records.
bool RunScopeVerifyScheModeSplitPass(SuperKernelGraph &graph, std::vector<SuperKernelScopeInfo> &scopes,
                                     std::vector<ScopeVerifySplitRecord> &splitRecords);

// Run deadlock refinement pass and append scope verify split records.
bool RunScopeVerifyDeadlockRefinePass(SuperKernelGraph &graph, SuperKernelOptionsManager &opts,
                                      std::vector<SuperKernelScopeInfo> &scopes,
                                      std::vector<ScopeVerifySplitRecord> &splitRecords);

#endif  // __SK_VERIFY_SCOPE_SPLIT_H__
