/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "segment_level_cache_gen.h"
#include <set>
#include <vector>
#include "base/arch_param.h"
#include "generator/preprocess/args_manager.h"

namespace att {
namespace cache {
namespace {
struct SegmentAxisExpr {
  std::string name;
  Expr size_expr;
  uint32_t data_type_size;
};

std::map<std::string, std::string> BuildVarAccessorMap(const TilingModelInfo &tiling_model_info) {
  // 与算子级缓存一致：按var名去重；多group场景accessor带 {item}_tiling_data. 前缀
  std::map<std::string, std::string> var_accessors;
  std::set<std::string> all_groups_prefix;
  for (const auto &model_info : tiling_model_info) {
    all_groups_prefix.insert(model_info.schedule_group_ident.GetGroupPrefix());
  }
  const bool is_unique_group = (all_groups_prefix.size() == 1UL);
  for (const auto &model_info : tiling_model_info) {
    ArgsManager args_manager(model_info);
    GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
    const std::string group_prefix =
        is_unique_group ? "" : (model_info.schedule_group_ident.GetItemPrefix() + "_tiling_data.");
    for (const auto &var : args_manager.GetInputVars()) {
      const std::string var_name = Str(var);
      var_accessors.emplace(var_name, "tiling_data." + group_prefix + "get_" + var_name + "()");
    }
  }
  return var_accessors;
}

std::string SerializeAxisSizeExpr(const Expr &size_expr, const std::map<std::string, std::string> &var_accessors) {
  std::vector<std::pair<Expr, Expr>> var_replacement;
  for (const auto &sym : size_expr.FreeSymbols()) {
    if (sym.GetExprType() != af::ExprType::kExprVariable) {
      continue;
    }
    const auto var_name = Str(sym);
    const auto iter = var_accessors.find(var_name);
    if (iter == var_accessors.cend()) {
      continue;
    }
    var_replacement.emplace_back(sym, CreateExpr(iter->second.c_str()));
  }
  return Str(size_expr.Replace(var_replacement));
}

// 收集三类公式相关原始轴（按轴名全局去重，轴名在算子内全局唯一）：
// R轴/最内轴取原始轴上的标记（ArgListReorder打标）；分核轴由bind_multicore切分轴回溯原始轴
void CollectSegmentAxes(const TilingModelInfo &tiling_model_info, std::vector<SegmentAxisExpr> &r_axes,
                        std::vector<SegmentAxisExpr> &innermost_axes, std::vector<SegmentAxisExpr> &multicore_axes) {
  std::set<std::string> visited_r_axes;
  std::set<std::string> visited_innermost_axes;
  std::set<std::string> visited_multicore_axes;
  for (const auto &model_info : tiling_model_info) {
    for (const auto &axis : model_info.arg_list) {
      if (axis->axis_pos == AxisPosition::ORIGIN) {
        if (axis->is_reduce_related && visited_r_axes.insert(axis->name).second) {
          r_axes.emplace_back(SegmentAxisExpr{axis->name, axis->size->symbol_expr, axis->size->data_type_size});
        }
        if (axis->is_load_store_innermost && visited_innermost_axes.insert(axis->name).second) {
          innermost_axes.emplace_back(SegmentAxisExpr{axis->name, axis->size->symbol_expr, axis->size->data_type_size});
        }
        continue;
      }
      if (axis->bind_multicore) {
        for (const auto *orig_axis : axis->orig_axis) {
          if ((orig_axis == nullptr) || (orig_axis->size == nullptr) ||
              !visited_multicore_axes.insert(orig_axis->name).second) {
            continue;
          }
          multicore_axes.emplace_back(
              SegmentAxisExpr{orig_axis->name, orig_axis->size->symbol_expr, orig_axis->size->data_type_size});
        }
      }
    }
  }
}

uint32_t GetAxisDtypeSize(const SegmentAxisExpr &axis) {
  if (axis.data_type_size == 0U) {
    GELOGW("[Segment Cache] axis %s has zero dtype size, clamp to 1.", axis.name.c_str());
    return 1U;
  }
  return axis.data_type_size;
}
}  // namespace

void SegmentLevelCacheGen::GetThresholds(const TilingModelInfo &tiling_model_info, uint32_t &r_axis_threshold,
                                         uint32_t &innermost_threshold) {
  // R轴公式阈值取vector length大小，最内轴公式阈值取cacheline大小（复用既有语义，不引入新常量）
  r_axis_threshold = arch_param::kDefaultVectorLenSize;
  innermost_threshold = arch_param::kDefaultCacheLineSize;
  if (!tiling_model_info.empty() && (tiling_model_info[0].tiling_schedule_config_table != nullptr)) {
    r_axis_threshold = tiling_model_info[0].tiling_schedule_config_table->GetVectorLenSize();
    innermost_threshold = tiling_model_info[0].tiling_schedule_config_table->GetCacheLineSize();
  }
}

std::map<SegmentGroupSlotKey, size_t> SegmentLevelCacheGen::BuildGroupSlotMap(
    const TilingModelInfo &tiling_model_info) {
  std::set<SegmentGroupSlotKey> group_slots;
  for (const auto &model_info : tiling_model_info) {
    group_slots.insert({model_info.schedule_group_ident.asc_graph_id, model_info.schedule_group_ident.impl_graph_id,
                        model_info.schedule_group_ident.group_id});
  }
  std::map<SegmentGroupSlotKey, size_t> slot_map;
  size_t slot_index = 0UL;
  for (const auto &slot_key : group_slots) {
    slot_map.emplace(slot_key, slot_index++);
  }
  return slot_map;
}

af::Status SegmentLevelCacheGen::GenSegmentCacheTypes(ge::CodePrinter &code_printer, const size_t asc_graph_num,
                                                      const size_t group_slot_num) {
  GE_ASSERT_TRUE((asc_graph_num > 0UL) && (group_slot_num > 0UL),
                 "Segment cache type size invalid, asc_graph_num %zu, group_slot_num %zu", asc_graph_num,
                 group_slot_num);
  code_printer.AddLine("// 动态shape分段缓存：3个布尔公式组合为key(0..7)，每key缓存一份模板选择结果");
  code_printer.AddLine("constexpr uint32_t kSegmentCacheKeySpace = 8U;");
  code_printer.AddLine("// 单个group选中的模板：case id + 是否Reduce切R模板(sub_case_tag为\"R\")");
  code_printer.AddLine("struct SegmentGroupCase {");
  code_printer.AddLine("  uint32_t case_id{0U};");
  code_printer.AddLine("  bool is_sub_case{false};");
  code_printer.AddLine("};");
  code_printer.AddLine("// 分段缓存条目：result_ids按asc graph索引；group_cases按(asc,result,group)扁平槽位索引");
  code_printer.AddLine("struct SegmentCacheEntry {");
  code_printer.AddLine("  bool valid{false};");
  code_printer.AddLine("  uint32_t result_ids[" + std::to_string(asc_graph_num) + "]{};");
  code_printer.AddLine("  SegmentGroupCase group_cases[" + std::to_string(group_slot_num) + "]{};");
  code_printer.AddLine("};");
  code_printer.AddLine("// 分段缓存上下文：命中条目副本 + 全量搜索成功时逐group收集的选中tag(保存用)");
  code_printer.AddLine("struct SegmentCacheContext {");
  code_printer.AddLine("  bool hit{false};");
  code_printer.AddLine("  bool allow_save{true};");
  code_printer.AddLine("  uint32_t key{0U};");
  code_printer.AddLine("  SegmentCacheEntry entry{};");
  code_printer.AddLine("  bool selected_is_sub_case[" + std::to_string(group_slot_num) + "]{};");
  code_printer.AddLine("};");
  code_printer.AddLine("");
  return af::SUCCESS;
}

af::Status SegmentLevelCacheGen::GenSegmentCacheStorage(ge::CodePrinter &code_printer, const uint32_t r_axis_threshold,
                                                        const uint32_t innermost_threshold) {
  code_printer.AddLine("namespace {");
  code_printer.AddLine(
      "// 动态shape分段缓存公式阈值（编译期常量：R轴阈值取vector_len_size，最内轴阈值取cache_line_size）");
  code_printer.AddLine("constexpr uint32_t kSegmentCacheRAxisBytesThreshold = " + std::to_string(r_axis_threshold) +
                       "U;");
  code_printer.AddLine(
      "constexpr uint32_t kSegmentCacheInnermostBytesThreshold = " + std::to_string(innermost_threshold) + "U;");
  code_printer.AddLine("// 分段缓存存储：key空间8，每key一个槽，首填后命中复用，失效后自愈重填");
  code_printer.AddLine(
      "// 与算子级缓存一致的存储机制：thread_local unique_ptr + 堆上懒分配（首填时分配），线程退出自动释放");
  code_printer.AddLine("thread_local std::unique_ptr<SegmentCacheEntry[]> g_segment_cache;");
  code_printer.AddLine("");
  code_printer.AddLine("const SegmentCacheEntry *FindSegmentCache(const uint32_t segment_key) {");
  code_printer.AddLine("  if (g_segment_cache == nullptr) {");
  code_printer.AddLine("    return nullptr;");
  code_printer.AddLine("  }");
  code_printer.AddLine("  const SegmentCacheEntry &entry = g_segment_cache[segment_key % kSegmentCacheKeySpace];");
  code_printer.AddLine("  return entry.valid ? &entry : nullptr;");
  code_printer.AddLine("}");
  code_printer.AddLine("");
  code_printer.AddLine("void InvalidateSegmentCache(const uint32_t segment_key) {");
  code_printer.AddLine("  if (g_segment_cache == nullptr) {");
  code_printer.AddLine("    return;");
  code_printer.AddLine("  }");
  code_printer.AddLine("  g_segment_cache[segment_key % kSegmentCacheKeySpace].valid = false;");
  code_printer.AddLine("}");
  code_printer.AddLine("");
  code_printer.AddLine("void SaveSegmentCache(const uint32_t segment_key, const SegmentCacheEntry &entry) {");
  code_printer.AddLine("  if (g_segment_cache == nullptr) {");
  code_printer.AddLine("    g_segment_cache = std::make_unique<SegmentCacheEntry[]>(kSegmentCacheKeySpace);");
  code_printer.AddLine("  }");
  code_printer.AddLine("  g_segment_cache[segment_key % kSegmentCacheKeySpace] = entry;");
  code_printer.AddLine("}");
  code_printer.AddLine("} // namespace");
  code_printer.AddLine("");
  return af::SUCCESS;
}

af::Status SegmentLevelCacheGen::GenKeyComputeFunc(ge::CodePrinter &code_printer,
                                                   const TilingModelInfo &tiling_model_info,
                                                   const TilingCodeGenConfig &config) {
  std::vector<SegmentAxisExpr> r_axes;
  std::vector<SegmentAxisExpr> innermost_axes;
  std::vector<SegmentAxisExpr> multicore_axes;
  CollectSegmentAxes(tiling_model_info, r_axes, innermost_axes, multicore_axes);
  const auto var_accessors = BuildVarAccessorMap(tiling_model_info);
  uint32_t r_axis_threshold = 0U;
  uint32_t innermost_threshold = 0U;
  GetThresholds(tiling_model_info, r_axis_threshold, innermost_threshold);

  code_printer.AddLine("namespace {");
  code_printer.AddLine("// 计算3个布尔公式并组合为分段key：b_r*4 + b_inner*2 + b_multicore");
  code_printer.AddLine("// 乘积链依赖&&短路截断，中间积不超过 阈值*2^32*2^4，uint64无溢出");
  code_printer.AddLine("uint32_t ComputeSegmentKey(const " + config.tiling_data_type_name + " &tiling_data) {");
  code_printer.AddLine("  // 公式①(R轴)：∏(R轴原始轴×dtype) < " + std::to_string(r_axis_threshold) + "，无R轴为true");
  code_printer.AddLine("  bool segment_r_small = true;");
  code_printer.AddLine("  uint64_t segment_r_product = 1U;");
  for (const auto &axis : r_axes) {
    const std::string axis_expr = SerializeAxisSizeExpr(axis.size_expr, var_accessors);
    code_printer.AddLine(
        "  segment_r_small = segment_r_small && ((segment_r_product = segment_r_product * "
        "static_cast<uint64_t>(" +
        axis_expr + ") * " + std::to_string(GetAxisDtypeSize(axis)) + "U) < kSegmentCacheRAxisBytesThreshold);");
  }
  code_printer.AddLine("  // 公式②(最内轴)：所有load/store最内轴 原始轴×dtype < " +
                       std::to_string(innermost_threshold) + "，全部满足为true");
  code_printer.AddLine("  bool segment_innermost_small = true;");
  for (const auto &axis : innermost_axes) {
    const std::string axis_expr = SerializeAxisSizeExpr(axis.size_expr, var_accessors);
    code_printer.AddLine("  segment_innermost_small = segment_innermost_small && ((static_cast<uint64_t>(" + axis_expr +
                         ") * " + std::to_string(GetAxisDtypeSize(axis)) +
                         "U) < kSegmentCacheInnermostBytesThreshold);");
  }
  code_printer.AddLine("  // 公式③(分核)：∏(bind_multicore原始轴) < 可用核数(tiling开始前的block_dim)，空集乘积为1");
  code_printer.AddLine("  bool segment_multicore_small = true;");
  code_printer.AddLine("  uint64_t segment_mc_product = 1U;");
  for (const auto &axis : multicore_axes) {
    const std::string axis_expr = SerializeAxisSizeExpr(axis.size_expr, var_accessors);
    code_printer.AddLine(
        "  segment_multicore_small = segment_multicore_small && ((segment_mc_product = "
        "segment_mc_product * static_cast<uint64_t>(" +
        axis_expr + ")) < static_cast<uint64_t>(tiling_data.get_block_dim()));");
  }
  code_printer.AddLine(
      "  const uint32_t segment_key = (segment_r_small ? 4U : 0U) | (segment_innermost_small ? 2U : "
      "0U) |");
  code_printer.AddLine("                                (segment_multicore_small ? 1U : 0U);");
  code_printer.AddLine(
      "  ATT_SEGMENT_CACHE_LOGD(OP_NAME, \"[Segment Cache] key=%u, r_small=%d, inner_small=%d, mc_small=%d.\",");
  code_printer.AddLine(
      "             segment_key, static_cast<int32_t>(segment_r_small), "
      "static_cast<int32_t>(segment_innermost_small),");
  code_printer.AddLine("             static_cast<int32_t>(segment_multicore_small));");
  code_printer.AddLine("  return segment_key;");
  code_printer.AddLine("}");
  code_printer.AddLine("} // namespace");
  code_printer.AddLine("");
  return af::SUCCESS;
}
}  // namespace cache
}  // namespace att
