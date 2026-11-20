/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_OPTIMIZE_TASK_GENERATOR_INDIRECT_LOAD_OUTPUT_ANALYSIS_H
#define AUTOFUSE_OPTIMIZE_TASK_GENERATOR_INDIRECT_LOAD_OUTPUT_ANALYSIS_H

#include "task_generator/indirect_load/indirect_load_types.h"

#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "common_utils.h"
#include "graph_utils.h"
#include "graph/symbolizer/symbolic.h"
#include "norm_utils.h"
#include "optimize/graph_pass/pass_utils.h"
#include "optimize/graph_pass/softmax_pattern_fusion_utils.h"
#include "schedule_utils.h"

#include <algorithm>
#include <limits>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace optimize {
namespace indirect_load_detail {
// IndirectLoad 输出侧分析：post-Reduce 布局 / Reduce-Broadcast 配对 / Softmax IO / NormInfo /
// Store 目标分类 / 输出边界收集（自 indirect_load_schedule_case_generator.cpp 纯搬迁，语义不变）。
// 后置 Reduce 布局推导（axes/kinds/first_reduce）。
af::Status BuildPostReduceLayout(const af::AscNodePtr &reduce, PostReduceLayout &layout, bool &is_legal);

// SIMT post-Reduce 布局校验（boundary 为首个不可归并轴位置）。
af::Status ValidateSimtPostReduceLayout(const af::AscNodePtr &reduce, size_t &boundary, bool &is_legal);

// SIMT AR 形态（Reduce 紧邻 gather 轴）判定。
af::Status IsSimtArPostReduce(const af::AscNodePtr &reduce, bool &is_ar);

// Shape 保持尾轴 Norm（Softmax 专用）判定。
bool IsShapePreservingTailAxisNorm(const af::AscNodePtr &reduce);

// SIMD post-Reduce 布局校验。
af::Status ValidateSimdPostReduceLayout(const af::AscNodePtr &indirect_load, const af::AscNodePtr &reduce,
                                        bool &is_legal);

// R-before-gather：Reduce 轴先于 IL gather 轴（非 SK 候选跳过依据）。
bool HasReduceBeforeGatherAxis(const af::AscNodePtr &indirect_load);

// Gather 后置 Norm 区域识别与 NormInfo 保存（Softmax/单 Reduce/复合区域）。
af::Status CollectGatherNormInfo(const af::AscNodePtr &indirect_load, const RewrittenGraphAnalysis &analysis,
                                 bool &is_candidate_legal, ascir::TemplateId template_id);

// 输出侧 Store 目标合法性门控（kDense/kSliceStrided/不规则视图）。
af::Status ValidateOutputStoreTargets(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                                      bool &is_candidate_legal);

// 输出边界收集；allow_multi_consumer_reduce 仅 SK 放开归约扇出校验。
af::Status CollectRewrittenBoundaries(const af::AscNodePtr &indirect_load, RewrittenGraphAnalysis &analysis,
                                      bool &is_candidate_legal, const bool allow_multi_consumer_reduce);

}  // namespace indirect_load_detail
}  // namespace optimize

#endif  // AUTOFUSE_OPTIMIZE_TASK_GENERATOR_INDIRECT_LOAD_OUTPUT_ANALYSIS_H
