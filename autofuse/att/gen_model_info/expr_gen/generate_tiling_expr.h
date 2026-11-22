/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXPR_GEN_GENERATE_TILING_EXPR_H_
#define EXPR_GEN_GENERATE_TILING_EXPR_H_

#include <utility>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <functional>
#include "base/base_types.h"
#include "gen_model_info.h"
#include "parser/tuning_space.h"

namespace att {
class GenerateTilingExpr {
 public:
  explicit GenerateTilingExpr(TuningSpacePtr tuning_space) : tuning_space_(std::move(tuning_space)) {}
  virtual ~GenerateTilingExpr() = default;

  af::Status Generate(ModelInfo &model_info);

 private:
  // 获取tensor内存占用表达式
  af::Status GetTensorExpr(std::map<std::string, Expr> &tensor_exprs);

  // 获取每个tensor_id的workspace表达式
  af::Status GetWorkSpaceSize(std::map<int64_t, Expr> &workspace_size_map);

  // 获取内存相关的约束
  af::Status GetBufConstraint(std::map<HardwareDef, Expr> &hardware_cons, std::map<std::string, Expr> &container_exprs);

  // 获取预留的ub空间大小
  af::Status GetReservedUbSize(Expr &reserved_ub_size);

  // 获取算子流水约束
  af::Status GetPipePerformance(std::map<PipeType, Expr> &pipe_perf_object,
                                std::map<Expr, TernaryOp, ExprCmp> &ternary_ops,
                                std::vector<PerfBreakdownGroup> &perf_breakdowns, Expr &head_cost);

  // 获取block dim约束
  af::Status GetCoreConstraint(std::map<HardwareDef, Expr> &hardware_cons);

  // 创建一个model info的轴
  af::Status MakeArg(const SubAxis *sub_axis, std::map<const SubAxis *, std::set<HardwareDef>> related_scopes,
                     AttAxisPtr &arg_info) const;

  // 初始化轴基本信息
  void InitArgInfo(const SubAxis *sub_axis, AttAxisPtr &arg_info) const;

  // 创建常量类型的轴信息
  af::Status MakeConstArg(const SubAxis *sub_axis, AttAxisPtr &arg_info) const;

  // 创建变量类型的轴信息
  af::Status MakeVarArg(const SubAxis *sub_axis, std::map<const SubAxis *, std::set<HardwareDef>> &related_scopes,
                        AttAxisPtr &arg_info) const;

  // 获取所有轴信息
  af::Status GetSubAxisArgs(std::vector<AttAxisPtr> &arg_lists);

  // 获取轴和父轴约束
  af::Status GetAxisConstraints(std::map<std::string, std::vector<std::pair<Expr, Expr>>> &eq_exprs,
                                std::map<std::string, std::vector<Expr>> &leq_exprs);

  // 获取output数量
  void GetOutputSize(uint32_t &output_size);

  // 判断是否要UB多核权衡并更新
  void UpdateNeedUBMCTradeoff(ModelInfo &model_info);

  // 辅助方法：查找A轴（Vectorized轴里从右向左数第一个非R轴、非B轴）
  std::vector<const AttAxis *> FindAAxis(const std::vector<AttAxisPtr> &arg_list) const;

  // 辅助方法：判断是否应该跳过某个轴
  bool ShouldSkipAxis(const SubAxis *sub_axis, const std::set<std::string> &reduce_split_axis_names) const;

  // 辅助方法：检查轴是否来自Reduce分核轴
  bool IsFromReduceSplit(const SubAxis *sub_axis, const std::set<std::string> &reduce_split_axis_names) const;

  // 辅助方法：收集 Reduce 分核轴名称
  std::set<std::string> CollectReduceSplitAxisNames() const;

  // 辅助方法：收集 Store 节点的 SubAxes
  std::vector<SubAxis *> CollectStoreSubAxes() const;

  // 辅助方法：通过名称匹配添加 A 轴
  void AddAxisByName(const SubAxis *sub_axis, const std::vector<AttAxisPtr> &arg_list,
                     std::vector<const AttAxis *> &result) const;

  // 辅助方法：计算惩罚的core_num_ratio
  Expr CalcPenaltyCoreNumRatio(const AttAxis *split_axis, const std::vector<const AttAxis *> &a_axes) const;

  // 辅助方法：获取Reduce分核的core_num_ratio
  Expr GetPenaltyCoreNumRatio(const ModelInfo &model_info, const AttAxis *split_axis,
                              const std::vector<const AttAxis *> &a_axes) const;

  // 辅助方法：将表达式转换为 upper_bound 形式
  Expr ApplyUpperBoundTransform(const Expr &size_expr) const;

  Expr ApplyOriginalAxisTransform(const Expr &size_expr) const;

  // 辅助方法：获取 cache_line_size
  uint32_t GetCacheLineSize() const;

  // 辅助方法：应用惩罚配置到 ModelInfo
  void ApplyPenaltyConfigToModelInfo(ModelInfo &model_info);

  // 辅助方法：应用符号转换（通用函数）
  using SymbolTransformFunc = std::function<std::string(const std::string &)>;
  Expr ApplySymbolTransform(const Expr &size_expr, const SymbolTransformFunc &transform_func) const;

  TuningSpacePtr tuning_space_;
};
}  // namespace att

#endif  // EXPR_GEN_GENERATE_TILING_EXPR_H_
