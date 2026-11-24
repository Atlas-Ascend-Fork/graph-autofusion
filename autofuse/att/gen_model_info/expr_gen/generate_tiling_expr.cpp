/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "generate_tiling_expr.h"
#include <algorithm>
#include "arg_list_manager.h"
#include "att_utils.h"
#include "buf_occupy_expr.h"
#include "pipe_perf_expr.h"
#include "common/util/mem_utils.h"

namespace att {
namespace {
// Tiling 调度配置相关常量
constexpr double kDefaultUbThreshold = 0.1;      // 默认 UB 阈值
constexpr uint32_t kDefaultCacheLineSize = 128;  // 默认CacheLine大小（字节）

const uint32_t kUBAlignValue = 32u;
const uint32_t kConcatOuterDimAlign = 16u;
template <typename T>
af::Status UpdateLastTileAxisPromptAlign(const SubAxis *sub_axis, const AttAxis &arg_info, T &size) {
  GELOGD("[DFX] UpdateLastTileAxisPromptAlign sub_axis=[%s], is_node_innerest_dim=%d, bind_multicore=%d, axis_pos=%d",
         sub_axis->ToString().c_str(), arg_info.is_node_innerest_dim, arg_info.bind_multicore,
         static_cast<int>(arg_info.axis_pos));
  if (arg_info.is_node_innerest_dim && (!arg_info.bind_multicore) && (arg_info.axis_pos == AxisPosition::INNER)) {
    auto block_len = std::gcd(sub_axis->data_type_size, kUBAlignValue);
    GE_ASSERT_TRUE(block_len != 0, "block_len is 0");
    size.prompt_align = kUBAlignValue / block_len;
    GELOGD("[DFX] Set axis[%s] prompt_align to %u (kUBAlignValue=%u / block_len=%u), data_type_size=%u",
           arg_info.name.c_str(), size.prompt_align, kUBAlignValue, block_len, sub_axis->data_type_size);
  }
  return af::SUCCESS;
}
}  // namespace
af::Status GenerateTilingExpr::GetBufConstraint(std::map<HardwareDef, Expr> &hardware_cons,
                                                std::map<std::string, Expr> &container_exprs) {
  std::unordered_map<HardwareDef, Expr> buffer_occupy;
  BufOccupEvaluatorExprPtr buf_evaluator = af::MakeShared<BufOccupyExpr>(tuning_space_);
  GE_ASSERT_NOTNULL(buf_evaluator, "Create buff evaluator expr failed.");
  GE_ASSERT_SUCCESS(buf_evaluator->GetTotalBufferOccup(buffer_occupy, container_exprs),
                    "Collect buf constraints failed.");
  for (const auto &buff : buffer_occupy) {
    hardware_cons[buff.first] = buff.second;
    GELOGD("[DFX]Add buf constraint: %d = %s", buff.first, buff.second.Str().get());
  }
  return af::SUCCESS;
}

af::Status GenerateTilingExpr::GetReservedUbSize(Expr &reserved_ub_size) {
  for (const auto &reserved_ub : tuning_space_->reserve_ub) {
    reserved_ub_size = reserved_ub_size + CreateExpr(reserved_ub.second);
  }
  return af::SUCCESS;
}

af::Status GenerateTilingExpr::GetWorkSpaceSize(std::map<int64_t, Expr> &workspace_size_map) {
  workspace_size_map = tuning_space_->workspace_size_map;
  return af::SUCCESS;
}

af::Status GenerateTilingExpr::GetPipePerformance(std::map<PipeType, Expr> &pipe_perf_object,
                                                  std::map<Expr, TernaryOp, ExprCmp> &ternary_ops,
                                                  std::vector<PerfBreakdownGroup> &perf_breakdowns, Expr &head_cost) {
  PipePerfExpr pipe_perf(tuning_space_);
  GE_ASSERT_SUCCESS(pipe_perf.GetPerfExpr(pipe_perf_object, ternary_ops, perf_breakdowns, head_cost),
                    "Get tiling performance failed.");
  return af::SUCCESS;
}

af::Status GenerateTilingExpr::GetCoreConstraint(std::map<HardwareDef, Expr> &hardware_cons) {
  Expr block_dim_max_expr = CreateExpr(0U);
  // 所有block dim取最大值
  for (auto &core_info : tuning_space_->block_dims) {
    Expr block_dim_expr = CreateExpr(1U);
    for (auto &block_axis : core_info) {
      auto axis_size = ArgListManager::GetInstance().GetArgExpr(block_axis->name);
      if (!IsValid(axis_size)) {
        axis_size = block_axis->repeat;
        ArgListManager::GetInstance().SetArgExpr(block_axis->name, axis_size);
      }
      block_dim_expr = af::sym::Mul(block_dim_expr, axis_size);
    }
    block_dim_max_expr = af::sym::Max(block_dim_expr, block_dim_max_expr);
  }
  hardware_cons[HardwareDef::CORENUM] = block_dim_max_expr;
  GELOGD("[DFX]Add core constraint: %d = %s", HardwareDef::CORENUM, block_dim_max_expr.Str().get());
  return af::SUCCESS;
}

af::Status GenerateTilingExpr::MakeArg(const SubAxis *sub_axis,
                                       std::map<const SubAxis *, std::set<HardwareDef>> related_scopes,
                                       AttAxisPtr &arg_info) const {
  InitArgInfo(sub_axis, arg_info);
  if (sub_axis->repeat.IsConstExpr()) {
    GE_ASSERT_GRAPH_SUCCESS(MakeConstArg(sub_axis, arg_info));
  } else {
    GE_ASSERT_GRAPH_SUCCESS(MakeVarArg(sub_axis, related_scopes, arg_info));
  }
  return af::SUCCESS;
}

void GenerateTilingExpr::InitArgInfo(const SubAxis *sub_axis, AttAxisPtr &arg_info) const {
  arg_info->name = sub_axis->name;
  arg_info->axis_pos = sub_axis->axis_type;
  arg_info->is_node_innerest_dim = sub_axis->is_node_innerest_dim;
  arg_info->bind_multicore = sub_axis->is_bind_multi_core;
  arg_info->is_last = sub_axis->is_last;
  arg_info->is_concat_outer_dim = sub_axis->is_concat_vec_axis && !sub_axis->is_node_innerest_dim;
  arg_info->is_concat_inner_dim = sub_axis->is_concat_vec_axis && sub_axis->is_node_innerest_dim;
  arg_info->is_reduce_split_axis = sub_axis->is_reduce_split_axis;
  arg_info->is_broadcast_split_axis = sub_axis->is_broadcast_split_axis;
}

af::Status GenerateTilingExpr::MakeConstArg(const SubAxis *sub_axis, AttAxisPtr &arg_info) const {
  auto size = af::MakeShared<SymConstInfo>(sub_axis->repeat);
  GE_ASSERT_NOTNULL(size, "Create sym const info failed.");
  std::vector<std::pair<Expr, Expr>> vars_value;
  double const_value = 0;
  auto ret = sub_axis->repeat.GetResult(vars_value, const_value);
  GE_ASSERT_GRAPH_SUCCESS(ret, "Get const expr value failed, ret [%d].", ret);
  size->const_value = static_cast<uint32_t>(const_value);
  size->value_range = sub_axis->value_range;
  size->data_type_size = sub_axis->data_type_size;
  GE_ASSERT_TRUE(sub_axis->data_type_size != 0, "sub_axis->data_type_size is 0");
  GELOGD("[DFX] MakeArg (const) axis[%s]: is_concat_inner_dim=%d, is_concat_outer_dim=%d, data_type_size=%u",
         sub_axis->name.c_str(), arg_info->is_concat_inner_dim, arg_info->is_concat_outer_dim,
         sub_axis->data_type_size);
  GE_ASSERT_GRAPH_SUCCESS(UpdateLastTileAxisPromptAlign(sub_axis, *arg_info, *size));
  if (arg_info->is_concat_inner_dim) {
    size->prompt_align = kUBAlignValue / sub_axis->data_type_size;
  }
  arg_info->size = size;
  return af::SUCCESS;
}

af::Status GenerateTilingExpr::MakeVarArg(const SubAxis *sub_axis,
                                          std::map<const SubAxis *, std::set<HardwareDef>> &related_scopes,
                                          AttAxisPtr &arg_info) const {
  Expr expr = ArgListManager::GetInstance().GetArgExpr(sub_axis->name);
  auto size = af::MakeShared<SymVarInfo>(expr);
  GE_ASSERT_NOTNULL(size, "Create sym var info failed.");
  size->align = sub_axis->align;
  if (sub_axis->axis_type == AxisPosition::INNER && sub_axis->enable_pad && sub_axis->enable_tail &&
      !sub_axis->orig_axis.empty()) {
    // Match the solver's padded tile domain without changing the logical extent.
    // Otherwise a short CV axis can have min=align=32 but max=8 or 16.
    Expr extent = af::sym::kSymbolOne;
    for (const auto *origin : sub_axis->orig_axis) {
      GE_ASSERT_NOTNULL(origin);
      GE_ASSERT_TRUE(IsValid(origin->repeat), "Original axis extent is invalid.");
      extent = af::sym::Mul(extent, origin->repeat);
    }
    size->max_value = af::sym::Mul(af::sym::Ceiling(af::sym::Div(extent, size->align)), size->align);
  }
  size->value_range = sub_axis->value_range;
  size->data_type_size = sub_axis->data_type_size;
  GE_ASSERT_TRUE(sub_axis->data_type_size != 0, "sub_axis->data_type_size is 0");
  GELOGD(
      "[DFX] MakeArg (non-const) axis[%s]: is_concat_inner_dim=%d, is_concat_outer_dim=%d, data_type_size=%u, "
      "axis_pos=%d",
      sub_axis->name.c_str(), arg_info->is_concat_inner_dim, arg_info->is_concat_outer_dim, sub_axis->data_type_size,
      static_cast<int>(arg_info->axis_pos));
  if (arg_info->is_concat_inner_dim) {
    size->prompt_align = kUBAlignValue / sub_axis->data_type_size;
  } else if (arg_info->is_concat_outer_dim) {
    size->prompt_align = kConcatOuterDimAlign;
  }
  UpdateLastTileAxisPromptAlign(sub_axis, *arg_info, *size);
  if (related_scopes.find(sub_axis) != related_scopes.end()) {
    for (auto &scope : related_scopes[sub_axis]) {
      size->related_scope.emplace_back(scope);
    }
  }
  arg_info->size = size;
  return af::SUCCESS;
}

af::Status GenerateTilingExpr::GetSubAxisArgs(std::vector<AttAxisPtr> &arg_lists) {
  std::map<SubAxis *, AttAxisPtr> relation;
  for (const auto &sub_axis : tuning_space_->sub_axes) {
    auto arg_info = af::MakeShared<AttAxis>();
    GE_ASSERT_NOTNULL(arg_info, "Create att axis failed.");
    GE_ASSERT_SUCCESS(MakeArg(sub_axis.get(), tuning_space_->related_scopes, arg_info), "Make arg info failed.");
    relation[sub_axis.get()] = arg_info;
    arg_lists.emplace_back(arg_info);
  }
  // 构造轴依赖关系
  for (const auto &iter : relation) {
    for (auto axis : iter.first->orig_axis) {
      auto att = relation.find(axis);
      if (att != relation.end()) {
        iter.second->orig_axis.emplace_back(att->second.get());
      }
    }
    for (auto axis : iter.first->parent_axis) {
      auto att = relation.find(axis);
      if (att != relation.end()) {
        iter.second->from_axis.emplace_back(att->second.get());
      }
    }
  }

  return af::SUCCESS;
}

af::Status GenerateTilingExpr::GetAxisConstraints(std::map<std::string, std::vector<std::pair<Expr, Expr>>> &eq_exprs,
                                                  std::map<std::string, std::vector<Expr>> &leq_exprs) {
  for (const auto &cur_axis : tuning_space_->sub_axes) {
    GE_ASSERT_NOTNULL(cur_axis, "Get cur_axis failed.");
    if ((cur_axis->axis_type != AxisPosition::OUTER) && (!cur_axis->parent_axis.empty())) {
      Expr father_size = CreateExpr(1U);
      for (auto &father : cur_axis->parent_axis) {
        father_size = af::sym::Mul(father_size, ArgListManager::GetInstance().GetArgExpr(father->name));
      }
      auto size = cur_axis->repeat;
      if (cur_axis->enable_tail == false) {
        eq_exprs[kFatherToChildNoTail].emplace_back(std::make_pair(father_size, size));
      } else if (cur_axis->enable_pad == true) {
        // 目前不需要
        continue;
      } else {
        leq_exprs[kFatherToChildLarger].emplace_back(af::sym::Sub(size, father_size));
      }
    }
  }
  return af::SUCCESS;
}

void GenerateTilingExpr::GetOutputSize(uint32_t &output_size) {
  uint32_t tmp_output_size = 0;
  for (const auto &node : tuning_space_->node_infos) {
    if (node.node_type == "Output") {
      tmp_output_size++;
    }
  }
  output_size = tmp_output_size;
}

af::Status GenerateTilingExpr::GetTensorExpr(std::map<std::string, Expr> &tensor_exprs) {
  for (const auto &node : tuning_space_->node_infos) {
    for (const auto &input : node.inputs) {
      GE_ASSERT_NOTNULL(input, "Get input failed.");
      Expr tensor_size_expr = ArgListManager::GetInstance().GetArgExpr(input->name);
      if (IsValid(tensor_size_expr)) {
        tensor_exprs[input->name] = tensor_size_expr;
      }
    }
    for (const auto &output : node.outputs) {
      GE_ASSERT_NOTNULL(output, "Get output failed.");
      Expr tensor_size_expr = ArgListManager::GetInstance().GetArgExpr(output->name);
      if (IsValid(tensor_size_expr)) {
        tensor_exprs[output->name] = tensor_size_expr;
      }
    }
  }
  return af::SUCCESS;
}

bool NeedUBMCTradeoff(TensorPtr tensor) {
  // 不是GM的不纳入ub mc tradeoff
  if (tensor->loc != HardwareDef::GM) {
    return false;
  }
  auto &repeats = tensor->repeat;
  auto &strides = tensor->gm_stride;
  GELOGD("tensor [%s] : repeats[%s], strides[%s]", tensor->name.c_str(), tensor->GetRepeat().c_str(),
         tensor->GetStride().c_str());
  if (repeats.size() <= 1) {
    return false;
  }
  if (repeats.size() == strides.size()) {
    Expr last_stride = af::sym::kSymbolOne;
    for (int32_t i = strides.size() - 2; i >= 0; i--) {
      if (strides[i + 1] != af::sym::kSymbolZero) {
        last_stride = strides[i + 1];
      }
      if (strides[i] == af::sym::kSymbolZero) {
        continue;
      }
      auto expect_stride = repeats[i + 1] * last_stride;
      if (strides[i] != expect_stride) {
        return true;
      }
    }
  }
  return false;
}

void GenerateTilingExpr::UpdateNeedUBMCTradeoff(ModelInfo &model_info) {
  for (auto &node_info : tuning_space_->node_infos) {
    const bool need_tradeoff_by_output =
        std::any_of(node_info.outputs.begin(), node_info.outputs.end(),
                    [](auto &output_tensor) { return NeedUBMCTradeoff(output_tensor); });
    const bool need_tradeoff_by_input = std::any_of(node_info.inputs.begin(), node_info.inputs.end(),
                                                    [](auto &input_tensor) { return NeedUBMCTradeoff(input_tensor); });
    if (need_tradeoff_by_output || need_tradeoff_by_input) {
      GELOGI(
          "model [%s] case [%d] need ub mc tradeoff, output tensor need tradeoff: %d, input tensor need tradeoff: %d",
          model_info.schedule_group_ident.GetGroupPrefixSnakeCase().c_str(), model_info.tiling_case_id,
          need_tradeoff_by_output, need_tradeoff_by_input);
      model_info.tiling_schedule_config.trade_off_config.is_enable = true;
      return;
    }
  }
}

// 判断是否应该跳过某个轴
bool GenerateTilingExpr::ShouldSkipAxis(const SubAxis *sub_axis,
                                        const std::set<std::string> &reduce_split_axis_names) const {
  // 跳过Reduce分核轴和Broadcast分核轴
  if (sub_axis->is_reduce_split_axis || sub_axis->is_broadcast_split_axis) {
    GELOGD("[DFX] ShouldSkipAxis: skip sub_axis [%s] (is_reduce_split or is_broadcast_split)", sub_axis->name.c_str());
    return true;
  }
  // 检查是否来自Reduce分核轴
  if (IsFromReduceSplit(sub_axis, reduce_split_axis_names)) {
    GELOGD("[DFX] ShouldSkipAxis: skip sub_axis [%s] as it's from reduce split", sub_axis->name.c_str());
    return true;
  }
  return false;
}

// 检查轴是否来自Reduce分核轴（通过orig_axis链检查）
bool GenerateTilingExpr::IsFromReduceSplit(const SubAxis *sub_axis,
                                           const std::set<std::string> &reduce_split_axis_names) const {
  std::set<const SubAxis *> visited;
  std::vector<const SubAxis *> to_check = {sub_axis};

  while (!to_check.empty()) {
    const SubAxis *current = to_check.back();
    to_check.pop_back();

    if (!visited.insert(current).second) {
      continue;
    }

    // 检查当前轴是否是Reduce分核轴的子轴
    for (const auto *orig_axis : current->orig_axis) {
      if (orig_axis == nullptr) {
        continue;
      }
      if (reduce_split_axis_names.find(orig_axis->name) != reduce_split_axis_names.end()) {
        GELOGD("[DFX] IsFromReduceSplit: sub_axis [%s] is from reduce split (orig_axis [%s])", sub_axis->name.c_str(),
               orig_axis->name.c_str());
        return true;
      }
      to_check.push_back(orig_axis);
    }
  }
  return false;
}

// 查找A轴：Vectorized轴里从右向左数所有非R轴
// 只从 Store 节点的 loop_axes (SubAxis) 中查找
std::vector<const AttAxis *> GenerateTilingExpr::FindAAxis(const std::vector<AttAxisPtr> &arg_list) const {
  std::vector<const AttAxis *> result;

  // 收集 Reduce 分核轴名称
  std::set<std::string> reduce_split_axis_names = CollectReduceSplitAxisNames();

  GELOGD("[DFX] FindAAxis: reduce_split_axis_names=%s",
         std::accumulate(
             reduce_split_axis_names.begin(), reduce_split_axis_names.end(), std::string(),
             [](const std::string &acc, const std::string &name) { return acc.empty() ? name : acc + "," + name; })
             .c_str());

  // 收集 Store 节点的 SubAxes
  std::vector<SubAxis *> all_sub_axes = CollectStoreSubAxes();

  // 从右向左遍历（最内层开始）
  for (auto it = all_sub_axes.rbegin(); it != all_sub_axes.rend(); ++it) {
    const SubAxis *sub_axis = *it;

    GELOGD("[DFX] FindAAxis: checking sub_axis [%s], is_reduce_split=%d, is_broadcast_split=%d", sub_axis->name.c_str(),
           sub_axis->is_reduce_split_axis, sub_axis->is_broadcast_split_axis);

    // 根据该轴是否被Reduce，判断是否应该跳过该轴
    if (ShouldSkipAxis(sub_axis, reduce_split_axis_names)) {
      continue;
    }

    // 通过 name 匹配添加到 result
    AddAxisByName(sub_axis, arg_list, result);
  }

  GELOGI("[DFX] FindAAxis: found %zu A axes[%s]", result.size(),
         std::accumulate(result.begin(), result.end(), std::string(), [](const std::string &acc, const AttAxis *axis) {
           return acc.empty() ? axis->name : acc + "," + axis->name;
         }).c_str());
  return result;
}

// 收集 Reduce 分核轴名称
std::set<std::string> GenerateTilingExpr::CollectReduceSplitAxisNames() const {
  std::set<std::string> reduce_split_axis_names;
  for (const auto &node : tuning_space_->node_infos) {
    AttUtils::CollectReduceAxisNames(node, reduce_split_axis_names);
  }
  return reduce_split_axis_names;
}

// 收集 Store 节点的 SubAxes
std::vector<SubAxis *> GenerateTilingExpr::CollectStoreSubAxes() const {
  std::vector<SubAxis *> all_sub_axes;
  for (const auto &node : tuning_space_->node_infos) {
    bool is_store = (node.node_type == kStore);
    if (node.node_ptr != nullptr) {
      is_store = AttUtils::IsStoreNode(node.node_ptr.get());
    }

    if (!is_store) {
      continue;
    }

    GELOGD("[DFX] CollectStoreSubAxes: node_name=%s, node_type=%s, inputs=%s", node.name.c_str(),
           node.node_type.c_str(),
           std::accumulate(node.inputs.begin(), node.inputs.end(), std::string(),
                           [](const std::string &acc, const auto &input) {
                             return acc.empty() ? input->ToString() : acc + "," + input->ToString();
                           })
               .c_str());

    for (const auto &input : node.inputs) {
      all_sub_axes.insert(all_sub_axes.cend(), input->dim_info.begin(), input->dim_info.end());
    }
    // 当前仅考虑一个输出节点，后续扩展时需处理多个输出节点
    break;
  }
  return all_sub_axes;
}

// 通过名称匹配添加 A 轴
void GenerateTilingExpr::AddAxisByName(const SubAxis *sub_axis, const std::vector<AttAxisPtr> &arg_list,
                                       std::vector<const AttAxis *> &result) const {
  for (const auto &axis : arg_list) {
    if (axis->name == sub_axis->name) {
      result.push_back(axis.get());
      GELOGD("[DFX] AddAxisByName: found A axis [%s]", sub_axis->name.c_str());
      break;
    }
  }
}

// 应用符号转换（通用函数）
Expr GenerateTilingExpr::ApplySymbolTransform(const Expr &size_expr, const SymbolTransformFunc &transform_func) const {
  auto symbols = size_expr.FreeSymbols();
  std::vector<std::pair<Expr, Expr>> replace_pairs;

  for (const auto &sym : symbols) {
    if (sym.IsConstExpr()) {
      continue;
    }
    std::string sym_name = Str(sym);
    std::string transformed_expr = transform_func(sym_name);
    replace_pairs.emplace_back(sym, af::Symbol(transformed_expr.c_str()));
  }

  return size_expr.Replace(replace_pairs);
}

// 将表达式转换为 upper_bound 形式
Expr GenerateTilingExpr::ApplyUpperBoundTransform(const Expr &size_expr) const {
  return ApplySymbolTransform(size_expr, [](const std::string &sym_name) {
    return sym_name + ".upper_bound(" + sym_name + ".upper_bound_vars)";
  });
}

// 将原始轴转换为 axis.value 形式
Expr GenerateTilingExpr::ApplyOriginalAxisTransform(const Expr &size_expr) const {
  return ApplySymbolTransform(size_expr, [](const std::string &sym_name) { return sym_name + ".value"; });
}

// 获取 cache_line_size
uint32_t GenerateTilingExpr::GetCacheLineSize() const {
  if (tuning_space_->tiling_schedule_config_table != nullptr) {
    return tuning_space_->tiling_schedule_config_table->GetCacheLineSize();
  }
  return kDefaultCacheLineSize;
}

// 计算惩罚的core_num_ratio
Expr GenerateTilingExpr::CalcPenaltyCoreNumRatio(const AttAxis *split_axis,
                                                 const std::vector<const AttAxis *> &a_axes) const {
  if (split_axis == nullptr || a_axes.empty()) {
    return af::Symbol(1);
  }

  GELOGI("[DFX] CalcPenaltyCoreNumRatio: split_axis=%s, a_axes_count=%zu", split_axis->name.c_str(), a_axes.size());

  // 计算所有A轴的size乘积，Tile切分轴替换为 upper_bound 形式
  Expr a_axis_size = af::Symbol(1);
  uint32_t data_type_size = split_axis->size->data_type_size;

  for (const AttAxis *a_axis : a_axes) {
    if (a_axis->size == nullptr || !a_axis->size->symbol_expr.IsValid()) {
      continue;
    }

    // 获取轴大小表达式
    Expr size_expr = a_axis->size->symbol_expr;
    Expr final_size = size_expr;

    // 只有Tile切分轴(INNER)才替换为 upper_bound 形式
    if (a_axis->axis_pos == AxisPosition::INNER) {
      final_size = ApplyUpperBoundTransform(size_expr);
      GELOGD("[DFX] CalcPenaltyCoreNumRatio: a_axis=%s (INNER, Tile), size=%s -> upper_bound=%s", a_axis->name.c_str(),
             Str(size_expr).c_str(), Str(final_size).c_str());
    } else if (a_axis->axis_pos == AxisPosition::ORIGIN) {
      final_size = ApplyOriginalAxisTransform(size_expr);
      GELOGD("[DFX] CalcPenaltyCoreNumRatio: a_axis=%s (ORIGIN, Tile), size=%s -> size_value=%s", a_axis->name.c_str(),
             Str(size_expr).c_str(), Str(final_size).c_str());
    } else {
      GELOGD("[DFX] CalcPenaltyCoreNumRatio: a_axis=%s (%s), size=%s (no upper_bound)", a_axis->name.c_str(),
             a_axis->axis_pos == AxisPosition::OUTER ? "OUTER" : "MERGED", Str(size_expr).c_str());
    }

    a_axis_size = a_axis_size * final_size;

    // 获取数据类型大小
    if (a_axis->size->data_type_size > 0) {
      data_type_size = a_axis->size->data_type_size;
    }

    GELOGD("[DFX] CalcPenaltyCoreNumRatio: accumulated=%s", Str(a_axis_size).c_str());
  }

  // 获取惩罚计算使用的CacheLine大小；特殊融合场景可使用独立粒度，不影响物理Cache Line配置。
  uint32_t cache_line_size =
      tuning_space_->penalty_cache_line_size > 0 ? tuning_space_->penalty_cache_line_size : GetCacheLineSize();

  // 计算 core_num_ratio = (a_axis_size * data_type_size) / cache_line_size
  Expr core_num_ratio = (a_axis_size * af::Symbol(data_type_size)) / af::Symbol(cache_line_size);

  GELOGI(
      "[DFX] Calculated penalty core_num_ratio: split_axis=%s, a_axes_count=%zu, "
      "total_a_size=%s, data_type_size=%u, cache_line_size=%u, ratio=%s",
      split_axis->name.c_str(), a_axes.size(), Str(a_axis_size).c_str(), data_type_size, cache_line_size,
      Str(core_num_ratio).c_str());

  return core_num_ratio;
}

Expr GenerateTilingExpr::GetPenaltyCoreNumRatio(const ModelInfo &model_info, const AttAxis *split_axis,
                                                const std::vector<const AttAxis *> &a_axes) const {
  if (model_info.enable_workspace_alignment) {
    GELOGI(
        "[DFX][WorkspaceAlignment] Workspace is aligned, skip reduce split core number penalty, "
        "model_name=%s, split_axis=%s",
        model_info.graph_name.c_str(), split_axis->name.c_str());
    return af::Symbol(1.0);
  }
  return CalcPenaltyCoreNumRatio(split_axis, a_axes);
}

// 应用惩罚配置到 ModelInfo
void GenerateTilingExpr::ApplyPenaltyConfigToModelInfo(ModelInfo &model_info) {
  // 0. 首先从 TilingScheduleConfigTable 获取基础配置（无论是否有惩罚场景）
  if (tuning_space_->tiling_schedule_config_table != nullptr) {
    model_info.tiling_schedule_config = tuning_space_->tiling_schedule_config_table->GetModelTilingScheduleConfig();
    GELOGI("[DFX] Loaded base TilingScheduleConfig from table, tiling_schedule_config=%s, model_name=%s",
           model_info.tiling_schedule_config.DebugString().c_str(), model_info.graph_name.c_str());
  } else {
    GELOGD("[DFX] tiling_schedule_config_table is null, using default TilingScheduleConfig, model_name=%s",
           model_info.graph_name.c_str());
  }

  // 1. 检查配置是否启用Reduce分核惩罚功能（通过TilingScheduleConfigTable接口）
  if (tuning_space_->tiling_schedule_config_table == nullptr) {
    GELOGD("[DFX] tiling_schedule_config_table is null, skip penalty calculation, model_name=%s",
           model_info.graph_name.c_str());
    return;
  }

  bool enable_penalty = tuning_space_->tiling_schedule_config_table->IsCoreNumThresholdPenaltyEnable();
  GELOGI("[DFX] Reduce split penalty config: enabled=%s (from TilingScheduleConfigTable), model_name=%s",
         enable_penalty ? "true" : "false", model_info.graph_name.c_str());

  if (!enable_penalty) {
    GELOGD("[DFX] Reduce split penalty is disabled by config, skip penalty calculation, model_name=%s",
           model_info.graph_name.c_str());
    return;
  }

  // 2. 检查是否存在Reduce分核Store冲突（暂时不考虑Broadcast分核场景）
  for (const auto &axis : model_info.arg_list) {
    if (axis->is_reduce_split_axis) {
      // 先查找所有A轴
      std::vector<const AttAxis *> a_axes = FindAAxis(model_info.arg_list);

      GELOGI("[DFX] Found %zu A axes for penalty calculation", a_axes.size());

      Expr penalty_core_num_ratio = GetPenaltyCoreNumRatio(model_info, axis.get(), a_axes);

      // 3. 设置惩罚配置（覆盖基础配置）
      model_info.tiling_schedule_config.trade_off_config.ub_ratio = af::Symbol(kDefaultUbThreshold);
      model_info.tiling_schedule_config.trade_off_config.core_num_ratio =
          af::sym::Min(penalty_core_num_ratio, af::Symbol(1.0));
      model_info.tiling_schedule_config.trade_off_config.is_enable = true;
      model_info.tiling_schedule_config.is_penalty_config = true;

      GELOGI(
          "[DFX] Applied Reduce split Store penalty, model_name=%s, split_axis=%s, "
          "core_num_ratio=%s, ub_ratio=%f",
          model_info.graph_name.c_str(), axis->name.c_str(), Str(penalty_core_num_ratio).c_str(), kDefaultUbThreshold);
      return;
    }
  }

  GELOGD("[DFX] No Reduce split axis found, model_name=%s", model_info.graph_name.c_str());
}

af::Status GenerateTilingExpr::Generate(ModelInfo &model_info) {
  GE_ASSERT_SUCCESS(ArgListManager::GetInstance().LoadArgList(tuning_space_), "Get tuning args failed.");
  model_info.variable_expr_map = ArgListManager::GetInstance().GetVariableExprMap();
  model_info.variable_name_map = ArgListManager::GetInstance().GetVariableNameMap();
  GELOGD("Get tuning args success.");

  GE_ASSERT_SUCCESS(GetBufConstraint(model_info.hardware_cons, model_info.container_exprs),
                    "Get buf constraints failed.");
  GELOGD("Get buf constraints success.");

  GE_ASSERT_SUCCESS(GetCoreConstraint(model_info.hardware_cons), "Get core constraints failed.");
  GELOGD("Get core constraints success.");

  GE_ASSERT_SUCCESS(GetReservedUbSize(model_info.reserved_ub_size), "Get reserved ub size failed.");
  GELOGD("Get reserved ub size success.");

  GE_ASSERT_SUCCESS(GetPipePerformance(model_info.objects, model_info.ternary_op_map, model_info.perf_breakdowns,
                                       model_info.head_cost),
                    "Get perf objects failed.");
  model_info.tiling_schedule_config_table = tuning_space_->tiling_schedule_config_table;

  GE_ASSERT_SUCCESS(GetWorkSpaceSize(model_info.workspace_size_map), "Get workspace size failed.");

  GE_ASSERT_SUCCESS(GetSubAxisArgs(model_info.arg_list), "Get args list failed.");

  GetAxisConstraints(model_info.eq_exprs, model_info.leq_exprs);
  GetOutputSize(model_info.output_size);

  // 如果已经因为 Reduce 分核使能了 trade_off，则不需要再更新
  if (!model_info.enable_group_parallel && !model_info.tiling_schedule_config.is_penalty_config) {
    UpdateNeedUBMCTradeoff(model_info);
  }

  // 检测并应用惩罚配置（根据是否存在Reduce分核场景，后续补齐Broadcast分核场景）
  ApplyPenaltyConfigToModelInfo(model_info);

  return af::SUCCESS;
}

}  // namespace att
