/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXPR_GEN_EXE_TIME_PASS_MANAGER_H_
#define EXPR_GEN_EXE_TIME_PASS_MANAGER_H_

#include "util/ternary_op.h"
#include "parser/tuning_space.h"
#include "expr_gen/cache_guard_count.h"

namespace att {
class ExeTimePassManager {
 public:
  explicit ExeTimePassManager(const TuningSpacePtr &tuning_space) : tuning_space_(tuning_space) {}
  ~ExeTimePassManager() = default;

  // 获取需要处理的节点
  TernaryOp UpdateNodeExeTime(const NodeInfo &node, const Expr &exe_time) const;

 private:
  Expr GetBlockDimExpr() const;
  TuningSpacePtr tuning_space_;
};
}  // namespace att

#endif
