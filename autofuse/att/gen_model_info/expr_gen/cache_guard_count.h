/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef EXPR_GEN_CACHE_GUARD_COUNT_H_
#define EXPR_GEN_CACHE_GUARD_COUNT_H_

#include "base/base_types.h"
#include "ascir_node_param/ascir_node_param.h"

namespace att {
using ascir_param::CacheGuardInfo;
using ascir_param::CacheGuardKind;

Expr CountGuardHits(const CacheGuardInfo &info, const Expr &block_idx);
bool IsBlockCountUniform(const CacheGuardInfo &info, const Expr &block_dim);
bool ValidateCacheGuardInfo(const CacheGuardInfo &info);
}  // namespace att

#endif  // EXPR_GEN_CACHE_GUARD_COUNT_H_
