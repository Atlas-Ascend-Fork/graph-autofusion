/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __SK_SCOPE_VERIFY_H__
#define __SK_SCOPE_VERIFY_H__

#include <cstddef>

#include "super_kernel.h"

// Print scope verify input graph and nodes.
void LogScopeVerifyInput(const aclskScopeVerifyGraphInfo *verifyGraph, size_t maxSplitResultCount);

// Run scope verify workflow and fill split results for aclskScopeVerify.
aclError RunScopeVerify(const aclskScopeVerifyGraphInfo *verifyGraph, size_t maxSplitResultCount,
                        aclskScopeVerifySplitResult *splitResults, size_t *realSplitResultCount);

#endif  // __SK_SCOPE_VERIFY_H__
