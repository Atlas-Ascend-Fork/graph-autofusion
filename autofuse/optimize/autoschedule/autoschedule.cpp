/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "autoschedule.h"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <string>
#include <queue>
#include "ascir.h"

#include "ascir_utils.h"
#include "schedule_utils.h"
#include "common_utils.h"
#include "indirect_load_utils.h"
#include "node_utils.h"
#include "ascendc_ir/core/ascendc_ir_impl.h"
#include "graph/symbolizer/symbolic_utils.h"

#include "autoschedule/template_generator_handler.h"

namespace {
constexpr int64_t kDefaultAxisId = -1;
constexpr int64_t kMaxBroadcastAxisSize = 16LL;
constexpr int64_t kMinNonBroadcastAxisSize = 256LL * 1024LL;

// 判断节点输出在该轴上是否仍有数据流动（stride 非零）：真正的归约塌缩轴在
// Reduce 输出上 stride 必为 0；而元素算子直接消费标量参数 Load（空间轴全退化）
// 时，输出在空间轴上 stride 非零，该轴对本节点不是归约轴。
bool IsAxisStreamingOnAnyOutput(const ascir::NodeView &node, const int64_t axis_id) {
  for (auto output : node->outputs()) {
    for (size_t i = 0UL; i < output->attr.axis.size(); ++i) {
      if (output->attr.axis[i] != axis_id) {
        continue;
      }
      if (af::SymbolicUtils::StaticCheckEq(output->attr.strides[i], af::sym::kSymbolZero) != af::TriBool::kTrue) {
        return true;
      }
    }
  }
  return false;
}

void FindNotLoopAxis(const ascir::NodeView &node, ascir::ImplGraph &impl_graph,
                     std::unordered_set<int64_t> &not_loop_axis_set, bool has_reduce, bool is_reduce_first_stage) {
  for (auto output : node->outputs()) {
    not_loop_axis_set.insert(output->attr.vectorized_axis.begin(), output->attr.vectorized_axis.end());
  }
  if (!has_reduce) {
    return;
  }
  af::AxisPtr block_axis = nullptr;
  for (const auto &axis : impl_graph.GetAllAxis()) {
    if (axis != nullptr && axis->type == ascir::Axis::Type::kAxisTypeBlockOuter) {
      block_axis = axis;
      break;
    }
  }
  // find reduce axis of input
  for (auto input : node->inputs()) {
    for (size_t i = 0; i < input->attr.repeats.size(); i++) {
      if ((af::SymbolicUtils::StaticCheckEq(input->attr.repeats[i], af::sym::kSymbolOne) != af::TriBool::kTrue) ||
          (af::SymbolicUtils::StaticCheckEq(input->attr.strides[i], af::sym::kSymbolZero) != af::TriBool::kTrue)) {
        continue;
      }
      auto r = impl_graph.FindAxis(input->attr.axis[i]);
      if (r == nullptr) {
        continue;
      }
      if (is_reduce_first_stage) {
        if (block_axis != nullptr &&
            std::find(block_axis->from.begin(), block_axis->from.end(), r->id) != block_axis->from.end()) {
          continue;
        }
      } else {
        if (r->type == ascir::Axis::Type::kAxisTypeBlockOuter || r->type == ascir::Axis::Type::kAxisTypeBlockInner) {
          continue;
        }
      }
      // Gather(IndirectLoad)+Norm：Norm 尾部元素算子直接消费标量参数 Load
      // （原 Broadcast 已被冗余消除），其退化轴不是归约轴；若据此抬升消费
      // 节点的循环层级，Norm 尾部 Cluster 会因 loop_axis 不一致被拆分，
      // 参数 Load 沦为跨 VF 外部输入并缺失父图 API call。
      // 仅 IndirectLoad 候选图启用该豁免，不改变普通图的循环轴推断语义；
      // 先做廉价的输出 stride 检查，绝大多数真归约轴在此短路，不触发全图扫描。
      if (IsAxisStreamingOnAnyOutput(node, r->id) &&
          ascgen_utils::indirect_load::FindIndirectLoadNode(impl_graph) != nullptr) {
        GELOGD("Axis[%ld] still streams on output of node[%s], keep it as loop axis candidate.", r->id,
               node->GetNamePtr());
        continue;
      }
      not_loop_axis_set.insert(input->attr.axis[i]);
    }
  }
}

bool IsNotLoopAxis(ascir::ImplGraph &impl_graph, int64_t axis, const std::unordered_set<int64_t> &not_loop_axis_set) {
  if (not_loop_axis_set.find(axis) != not_loop_axis_set.end()) {
    return true;
  }
  auto r = impl_graph.FindAxis(axis);
  if (r == nullptr) {
    return false;
  }
  if (r->from.empty()) {
    return false;
  }
  for (auto c : r->from) {
    if (!IsNotLoopAxis(impl_graph, c, not_loop_axis_set)) {
      return false;
    }
  }
  return true;
}

int64_t GetStaticAxisProduct(ascir::ImplGraph &impl_graph, const std::vector<int64_t> &axis_ids) {
  int64_t product = 1;
  for (const auto axis_id : axis_ids) {
    auto axis = impl_graph.FindAxis(axis_id);
    if (axis == nullptr || !axis->size.IsConstExpr()) {
      return -1;
    }
    int64_t static_size = 0;
    if (!axis->size.GetConstValue(static_size) || static_size <= 0) {
      return -1;
    }
    product *= static_size;
  }
  return product;
}

std::unordered_set<int64_t> CollectBroadcastAxes(const ascir::ImplGraph &impl_graph) {
  std::unordered_set<int64_t> broadcast_axes;
  for (const auto &node : impl_graph.GetAllNodes()) {
    if (!optimize::ScheduleUtils::IsLoad(node)) {
      continue;
    }
    const auto &output = node->outputs[0];
    for (size_t i = 0; i < output.attr.axis.size(); ++i) {
      if (af::SymbolicUtils::StaticCheckEq(output.attr.strides[i], af::sym::kSymbolZero) == af::TriBool::kTrue) {
        broadcast_axes.insert(output.attr.axis[i]);
      }
    }
  }
  return broadcast_axes;
}

void PartitionLoopAxes(const ascir::ImplGraph &impl_graph, const std::unordered_set<int64_t> &broadcast_axes,
                       std::vector<int64_t> &new_order, std::vector<int64_t> &brc_axes) {
  for (const auto &node : impl_graph.GetAllNodes()) {
    if (!optimize::ScheduleUtils::IsBuffer(node)) {
      new_order.reserve(node->attr.sched.axis.size());
      for (const auto axis_id : node->attr.sched.axis) {
        if (broadcast_axes.count(axis_id) > 0) {
          brc_axes.push_back(axis_id);
        } else {
          new_order.push_back(axis_id);
        }
      }
      break;
    }
  }
}

void ReorderBroadcastAxesInner(ascir::ImplGraph &impl_graph) {
  optimize::GraphPropertiesCache cache(impl_graph);
  if (cache.HasReduce() || cache.HasGather() || cache.HasCube() || cache.HasTranspose() || cache.HasConcat() ||
      cache.HasSplit()) {
    return;
  }
  auto broadcast_axes = CollectBroadcastAxes(impl_graph);
  if (broadcast_axes.empty()) {
    return;
  }
  std::vector<int64_t> new_order;
  std::vector<int64_t> brc_axes;
  PartitionLoopAxes(impl_graph, broadcast_axes, new_order, brc_axes);
  if (brc_axes.empty() || new_order.empty() || new_order.size() + brc_axes.size() > 2UL) {
    return;
  }

  int64_t brc_size = GetStaticAxisProduct(impl_graph, brc_axes);
  if (brc_size <= 0 || brc_size > kMaxBroadcastAxisSize) {
    GELOGD("Skip broadcast reorder: brc_size=%ld.", brc_size);
    return;
  }
  int64_t non_brc_size = GetStaticAxisProduct(impl_graph, new_order);
  if (non_brc_size <= 0 || non_brc_size <= kMinNonBroadcastAxisSize) {
    return;
  }
  new_order.insert(new_order.end(), brc_axes.begin(), brc_axes.end());
  for (const auto &node : impl_graph.GetAllNodes()) {
    if (!optimize::ScheduleUtils::IsBuffer(node)) {
      node->attr.sched.axis = new_order;
    }
  }
  GELOGI("Broadcast reorder: brc_size=%ld, non_brc_size=%ld, graph=[%s]", brc_size, non_brc_size,
         impl_graph.GetName().c_str());
}

void AppendIdIfNotDefault(std::stringstream &ss, const std::string &prefix, int64_t id) {
  if (id != kDefaultAxisId) {
    ss << prefix << id;
  }
}

// Returns true if an IndirectLoad prebuilt tiling case was generated, and the caller should return early.
bool TryGenIndirectLoadTilingCase(ascir::ImplGraph &graph,
                                  std::vector<optimize::autoschedule::TilingCase> &tiling_cases) {
  optimize::autoschedule::TilingCase tiling_case;
  if (!ascgen_utils::indirect_load::GetPrebuiltYTilingCase(graph, tiling_case.ub_tiling_id_y,
                                                           tiling_case.ub_tiling_y)) {
    return false;
  }
  // post-Reduce SIMT 候选不预建固定 tile 轴（pair 为空），TileTiling 阶段走通用 TileSplit
  // 按 UB 容量求解 tile 行数；其余模板 pair 非空，维持固定 tile 语义。
  tiling_case.block_tiling_id = 0;
  tiling_cases.push_back(tiling_case);
  GELOGD("[IndirectLoad] Graph[%s] generate prebuilt outer tiling case for axis[%ld]%s.", graph.GetName().c_str(),
         tiling_case.ub_tiling_id_y, tiling_case.ub_tiling_y.first == nullptr ? " with solved tile size" : "");
  return true;
}
}  // namespace

namespace optimize::autoschedule {
Status AutoSchedule::SelectLoopAxis(ascir::ImplGraph &impl_graph, bool is_reduce_fullload) const {
  bool has_reduce = false;
  for (auto node : impl_graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    node->attr.sched.loop_axis = af::kIdNone;
    if (node->attr.api.type != af::ApiType::kAPITypeCompute ||
        ascgen_utils::indirect_load::GetTemplateBehavior(node).skips_main_schedule_tiling) {
      continue;
    }
    if (ScheduleUtils::IsReduce(node) && !is_reduce_fullload) {
      has_reduce = true;
    }
    auto axis = node->attr.sched.axis;
    std::unordered_set<int64_t> not_loop_axis_set;
    FindNotLoopAxis(node, impl_graph, not_loop_axis_set, has_reduce, is_reduce_first_stage_);
    for (auto &s : axis) {
      if (IsNotLoopAxis(impl_graph, s, not_loop_axis_set)) {
        s = af::kIdNone;
        continue;
      }
    }

    auto it = std::find_if(axis.rbegin(), axis.rend(), [](const auto &val) { return val != af::kIdNone; });
    if (it != axis.rend()) {
      node->attr.sched.loop_axis = *it;
    }
    if (node->attr.sched.loop_axis == af::kIdNone && !axis.empty()) {
      // 打印原始 sched 轴列表（axis 副本中的被滤轴已被置为 kIdNone，不可用），
      // 供 not_loop 判定与 from 链排查。
      std::string sched_axis_ids;
      for (const auto &sched_axis_id : node->attr.sched.axis) {
        sched_axis_ids += std::to_string(sched_axis_id);
        sched_axis_ids += ",";
      }
      GELOGD("[IndirectLoad] SelectLoopAxis: node[%s] all axes filtered, sched_axes[%s], not_loop_set_size=%zu.",
             node->GetNamePtr(), sched_axis_ids.c_str(), not_loop_axis_set.size());
      for (const auto &entry : not_loop_axis_set) {
        GELOGD("[IndirectLoad] SelectLoopAxis: node[%s] not_loop_axis[%ld].", node->GetNamePtr(), entry);
      }
    }
    GE_ASSERT_TRUE((node->attr.sched.loop_axis != af::kIdNone), "Cannot find loop axis for node: [%s].",
                   node->GetNamePtr());
  }
  return af::SUCCESS;
}

void AutoSchedule::GenTilingCase(std::vector<TilingCase> &tiling_cases) {
  auto set_tiling_id = [](auto &field, auto tid) {
    if (tid != kDefaultAxisId) {
      field = tid;
    }
  };

  if (TryGenIndirectLoadTilingCase(graph_, tiling_cases)) {
    return;
  }

  if (cube_template_ != ascir::CubeTemplateType::kDefault) {
    for (const auto &y_id : axes_group_.y_group) {
      TilingCase tiling_case;
      set_tiling_id(tiling_case.ub_tiling_id_y, y_id);
      tiling_cases.push_back(tiling_case);
    }
    return;
  }
  auto append_reduce_case = [&tiling_cases](const TilingCase &base_case) {
    auto reduce_case = base_case;
    reduce_case.block_tiling_id = 1;
    reduce_case.reduce_is_block = true;
    tiling_cases.push_back(reduce_case);
  };

  int64_t attr_axis = -1L;
  int64_t params_size = -1L;
  bool has_gather = ScheduleUtils::GetGatherParams(graph_, attr_axis, params_size);
  if (has_gather && !(attr_axis == params_size - 1 && attr_axis == 0)) {
    int64_t cnt = 0;
    for (const auto &y_id : axes_group_.y_group) {
      if (++cnt == 1) {
        continue;
      }
      TilingCase tiling_case;
      set_tiling_id(tiling_case.ub_tiling_id_y, y_id);
      tiling_case.block_tiling_id = 0;
      tiling_cases.push_back(tiling_case);
    }
    return;
  }

  // 生成通用pattern
  // 遍历所有的group，分别从每个group中取出1个值，组成所有的tiling case
  for (const auto &x_id : axes_group_.x_group) {
    for (const auto &y_id : axes_group_.y_group) {
      for (const auto &r_id : axes_group_.r_group) {
        TilingCase tiling_case;
        set_tiling_id(tiling_case.ub_tiling_id_x, x_id);
        set_tiling_id(tiling_case.ub_tiling_id_y, y_id);
        set_tiling_id(tiling_case.ub_tiling_id_r, r_id);
        tiling_case.block_tiling_id = 0;
        if (is_reduce_first_stage_ && r_id != kDefaultAxisId) {
          append_reduce_case(tiling_case);
        } else {
          tiling_cases.push_back(tiling_case);
        }
      }
    }
  }
}

// 尽量不生成切size=1轴的tiling case
Status AutoSchedule::PruneTilingCase(std::vector<TilingCase> &tiling_cases) const {
  auto all_axis = graph_.GetAllAxis();
  for (auto it = tiling_cases.begin(); it != tiling_cases.end();) {
    // 仅处理单切分场景
    if ((it->ub_tiling_id_r != kDefaultAxisId) || (it->ub_tiling_id_x != kDefaultAxisId)) {
      ++it;
      continue;
    }

    auto axis = graph_.FindAxis(it->ub_tiling_id_y);
    GE_ASSERT_NOTNULL(axis, "Tiling case is invalid axis_id:[%ld].", it->ub_tiling_id_y);
    if (af::SymbolicUtils::StaticCheckEq(axis->size, af::sym::kSymbolOne) == af::TriBool::kTrue &&
        tiling_cases.size() > 1UL) {
      it = tiling_cases.erase(it);
      GELOGD("Axis [%s]'s size is 1, will skip cutting it.", axis->name.c_str());
    } else {
      ++it;
    }
  }
  return af::SUCCESS;
}

static std::string GetTilingCaseStr(const std::string &graph_name, const TilingCase &tiling_case) {
  std::stringstream ss;
  ss << graph_name << "_";
  AppendIdIfNotDefault(ss, "B", tiling_case.block_tiling_id);
  AppendIdIfNotDefault(ss, "X", tiling_case.ub_tiling_id_x);
  AppendIdIfNotDefault(ss, "Y", tiling_case.ub_tiling_id_y);
  AppendIdIfNotDefault(ss, "R", tiling_case.ub_tiling_id_r);
  return ss.str();
}

Status BuildIndirectLoadAxisGroup(const ascir::ImplGraph &graph, const AxisGroup &base_group, AxisGroup &axis_group) {
  const af::AscNodePtr indirect_load = ascgen_utils::indirect_load::FindIndirectLoadNode(graph);
  if (indirect_load == nullptr) {
    axis_group = base_group;
    return af::SUCCESS;
  }

  ascgen_utils::indirect_load::TemplateAxes axes;
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, axes));
  axis_group = {};
  axis_group.y_group.push_back(axes.outer_axis);
  axis_group.axes_order.resize(axis_group.y_group.size());
  std::iota(axis_group.axes_order.begin(), axis_group.axes_order.end(), 0UL);
  GELOGD("[IndirectLoad] Graph[%s] use template axis group[%s].", graph.GetName().c_str(),
         axis_group.ToString().c_str());
  return af::SUCCESS;
}

Status AutoSchedule::DoAutoSchedule() {
  graph_.SetGraphType(af::AscGraphType::kImplGraph);
  ReorderBroadcastAxesInner(graph_);

  // 生成通用模版
  std::vector<TilingCase> tiling_cases;
  GE_CHK_STATUS_RET(PrepareTilingCases(tiling_cases), "Failed to prepare tiling cases for graph: [%s]",
                    graph_.GetName().c_str());

  const bool is_last_axis_reduce = ScheduleUtils::IsLastAxisReduce(graph_);
  const bool is_reduce_full_load = (reduce_template_ == optimize::ReduceTemplateType::kAllLoad);
  for (size_t index = 0UL; index < tiling_cases.size(); ++index) {
    GE_CHK_STATUS_RET(ProcessOneTilingCase(tiling_cases[index], index, is_last_axis_reduce, is_reduce_full_load),
                      "Failed to process tiling case %zu for graph: [%s]", index, graph_.GetName().c_str());
  }

  // 生成多模板
  GE_CHK_STATUS_RET(TemplateGeneratorHandler::GenerateTemplates(graph_, schd_outputs_),
                    "Failed to generate templates for graph: [%s]", graph_.GetName().c_str());

  // 生成 UBFuse 模板 TTODO待归到多模版内
  if (cube_template_ == ascir::CubeTemplateType::kUBFuse) {
    GenUBFuseTemplates();
  }

  return af::SUCCESS;
}

Status AutoSchedule::PrepareTilingCases(std::vector<TilingCase> &tiling_cases) {
  const bool is_reduce_full_load = (reduce_template_ == optimize::ReduceTemplateType::kAllLoad);
  GE_CHK_STATUS_RET(TilingGroup::GenTilingGroup(graph_, axes_group_, is_reduce_full_load),
                    "Gen tiling group failed for graph: [%s]", graph_.GetName().c_str());
  TilingGroup::NormGroup(axes_group_);

  GenTilingCase(tiling_cases);
  GE_CHK_STATUS_RET(PruneTilingCase(tiling_cases), "Failed to prune tiling cases for graph: [%s]",
                    graph_.GetName().c_str());

  GE_ASSERT_TRUE(!tiling_cases.empty(), "No valid tiling cases for graph: [%s]. Please check graph legality.",
                 graph_.GetName().c_str());

  return af::SUCCESS;
}

Status AutoSchedule::ProcessOneTilingCase(TilingCase &tiling_case, size_t index, bool is_last_axis_reduce,
                                          bool is_reduce_full_load) const {
  const std::string graph_name = GetTilingCaseStr(ascgen_utils::GenValidName(graph_.GetName()), tiling_case);
  AutoScheduleOutput output(graph_name.c_str());
  GE_ASSERT_TRUE(output.scheduled_graph.CopyFrom(graph_), "Failed to copy graph for tiling case %zu in graph: [%s]",
                 index, graph_.GetName().c_str());

  AxisGroup axis_group;
  GE_CHK_STATUS_RET(BuildIndirectLoadAxisGroup(output.scheduled_graph, axes_group_, axis_group),
                    "Failed to build scheduler axis group for tiling case %zu in graph: [%s]", index,
                    graph_name.c_str());
  Scheduler scheduler(output.scheduled_graph, axis_group, tiling_case, is_last_axis_reduce, reduce_template_,
                      cube_template_);

  auto ret = scheduler.DoScheduler();
  if (ret == af::UNSUPPORTED) {
    GELOGW("Tiling case %zu (graph: [%s]) is unsupported, skip it.", index, graph_name.c_str());
    return af::SUCCESS;
  }
  GE_CHK_STATUS_RET(ret, "Scheduler failed for tiling case %zu in graph: [%s]", index, graph_name.c_str());

  if (tiling_case.reduce_is_block) {
    GE_ASSERT_TRUE(
        output.scheduled_graph.BindBlock(tiling_case.block_tiling_id, tiling_case.reduce_block_tiling.second->id));
    output.var_relations_["Rm_org_size"] = tiling_case.rm_org_size;
    output.var_relations_["A_org_size"] = tiling_case.a_org_size;
  }

  GE_CHK_STATUS_RET(SelectLoopAxis(output.scheduled_graph, is_reduce_full_load),
                    "Failed to select loop axis for tiling case %zu in graph: [%s]", index, graph_name.c_str());

  schd_outputs_.emplace_back(output);
  return af::SUCCESS;
}

void AutoSchedule::GenUBFuseTemplates() const {
  std::vector<AutoScheduleOutput> schd_outputs_non_db;
  schd_outputs_non_db.reserve(schd_outputs_.size());
  for (const auto &schd_output : schd_outputs_) {
    auto graph_name = schd_output.scheduled_graph.GetName() + "_non_db";
    AutoScheduleOutput output_non_db(graph_name.c_str());
    output_non_db.scheduled_graph.CopyFrom(schd_output.scheduled_graph);
    output_non_db.var_relations_ = schd_output.var_relations_;
    output_non_db.score_func = schd_output.score_func;
    schd_outputs_non_db.emplace_back(std::move(output_non_db));
  }
  schd_outputs_.insert(schd_outputs_.begin(), std::make_move_iterator(schd_outputs_non_db.begin()),
                       std::make_move_iterator(schd_outputs_non_db.end()));
}
}  // namespace optimize::autoschedule
