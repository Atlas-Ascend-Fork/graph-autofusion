/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_OPTIMIZE_TASK_GENERATOR_INDIRECT_LOAD_INDIRECT_LOAD_UB_FEASIBILITY_H
#define AUTOFUSE_OPTIMIZE_TASK_GENERATOR_INDIRECT_LOAD_INDIRECT_LOAD_UB_FEASIBILITY_H

#include "task_generator/indirect_load/indirect_load_types.h"

namespace optimize {
namespace indirect_load_detail {

// [SK段SIMT化] 平台 UB 与 tensor 逻辑总量（动态 shape 返回 false）。
constexpr int64_t kIndirectLoadUbReservedBytes = 32768;  // SIMT 预留 tmp/sync/dcache 下界
// SIMT post-Reduce 的 R 域 UB 下界系数：输出 tile 双缓冲（2 份）+ Reduce 输入完整
// R 域（1 份），共 3 份 R 域驻留 UB。
constexpr int64_t kSimtRDomainUbFactor = 3;

// IndirectLoad 平台 UB 可行性：窗口 UB / SIMT UB / 组 UB 硬下界预估（自 indirect_load_schedule_case_generator.cpp
// 纯搬迁，语义不变）。
bool TryGetIndirectLoadPlatformUbSize(int64_t &ub_size);

bool TryGetTensorLogicalBytes(const af::AscTensor &tensor, int64_t &bytes);

bool IsSkWindowUbInfeasible(const af::AscNodePtr &indirect_load);

int64_t CountRDomainChainTensors(const af::AscNodePtr &indirect_load, int64_t tail);

bool IsSimtUbInfeasible(const af::AscGraph &graph, const af::AscNodePtr &indirect_load);

bool IsGroupSimdUbDoomed(const std::vector<af::AscNodePtr> &members);

bool IsGroupSimtNValueTileUbDoomed(const std::vector<af::AscNodePtr> &members);

bool IsGroupSimtL4UbDoomed(const std::vector<af::AscNodePtr> &members);

bool IsGroupUbAbsoluteDoomed(const std::vector<af::AscNodePtr> &members);

}  // namespace indirect_load_detail
}  // namespace optimize

#endif  // AUTOFUSE_OPTIMIZE_TASK_GENERATOR_INDIRECT_LOAD_{h_name.upper()}_H
