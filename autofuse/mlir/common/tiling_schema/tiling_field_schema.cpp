/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/tiling_schema/tiling_field_schema.h"

#include "common/tiling_schema/canonical_resource_names.h"
#include "common/tiling_schema/schedule_group_names.h"

#include <cctype>
#include <set>
#include <sstream>
#include <unordered_map>

#include "common_utils.h"

namespace mlir::common {
namespace {

void SetDiagnostic(std::string *diagnostic, const std::string &message) {
  if (diagnostic != nullptr) {
    *diagnostic = message;
  }
}

std::string ExprText(const af::Expression &expression) {
  const auto text = expression.Str();
  return text == nullptr ? std::string{} : std::string(text.get());
}

// Fields are deduplicated within one (subgraph, plan, stage) scope only.  A
// scope is identified by `path`.  This mirrors legacy ProcessSingleGroup,
// which shares one dedup set across all templates of a single stage, and
// ProcessMultiGroup, which gives every (plan, stage) pair its OWN isolated
// dedup set (and therefore its own nested C++ struct) — reusing a size-var or
// queue/buffer id name across stages/plans must NOT collapse into one field,
// because those ids are small sequential integers that are commonly reused
// across independently-scheduled plans/stages for logically different
// tensors. Keying the dedup set on (path, name) instead of name alone keeps
// the flat single-scope behavior when path is shared (single-group case)
// while isolating every nested scope from every other one.
// 收集字段时的去向：字段表本身，加上已见过的 (path, name) 键集合。所有 Append* 都同时
// 需要这两项，收成一个整体传递。
struct SchemaSink {
  TilingFieldSchema &schema;
  std::set<std::string> &seen;
};

void AppendField(SchemaSink sink, std::string name, const std::string &path,
                 TilingFieldType type = TilingFieldType::kUInt32, std::string device_type_name = {}) {
  const std::string key = path + kTilingFieldScopeKeySeparator + name;
  if (sink.seen.insert(key).second) {
    sink.schema.fields.push_back({std::move(name), path, type, std::move(device_type_name)});
  }
}

void AppendCommonFusionFields(SchemaSink sink) {
  AppendField(sink, "block_dim", "fusion");
  AppendField(sink, "corenum", "fusion");
  AppendField(sink, "ub_size", "fusion");
  AppendField(sink, "hbm_size", "fusion");
}

// Mirrors codegen::TilingData::AddApiTilingData: nodes such as Transpose/Pad
// carry a device-side API tiling struct that is not decomposed into
// individual size fields.  Detect and record it as an opaque field rather
// than silently dropping it — a dropped field would make the schema diverge
// from the legacy field set without any diagnostic.
void AppendApiTilingField(const ascir::NodeView &node, const std::string &path, size_t template_id, SchemaSink sink) {
  std::string device_type_name;
  std::string field_name;
  if (ascgen_utils::GetApiTilingTypeName(node, device_type_name) != af::SUCCESS) {
    return;
  }
  if (ascgen_utils::GetApiTilingFieldName(node, field_name) != af::SUCCESS) {
    return;
  }
  field_name += "_" + std::to_string(template_id);
  AppendField(sink, std::move(field_name), path, TilingFieldType::kOpaqueStruct, std::move(device_type_name));
}

void AppendTemplateFields(const af::AscGraph &graph, const std::string &field_scope,
                          const std::string &api_tiling_scope, size_t template_id, SchemaSink sink) {
  // This traversal is deliberately legacy-only.  Do not add AscGraph to a
  // future MLIR-native schema API; replace this adapter when ATT migrates.
  for (const auto &size_var : graph.GetAllSizeVar()) {
    if (size_var == nullptr || size_var->expr.IsConstExpr()) {
      continue;
    }
    AppendField(sink, ExprText(size_var->expr), field_scope);
  }
  for (const auto &node : graph.GetAllNodes()) {
    if (node == nullptr) {
      continue;
    }
    for (const auto &output : node->outputs()) {
      if (output == nullptr) {
        continue;
      }
      if (output->attr.que.id >= 0) {
        AppendField(sink, MakeQueueSizeName(output->attr.que.id), field_scope);
      }
      if (output->attr.buf.id >= 0) {
        AppendField(sink, MakeBufferSizeName(output->attr.buf.id), field_scope);
      }
    }
    for (const auto &tmp_buffer : node->attr.tmp_buffers) {
      if (tmp_buffer.id >= 0) {
        AppendField(sink, MakeBufferSizeName(tmp_buffer.id), field_scope);
      }
    }
    AppendApiTilingField(node, api_tiling_scope, template_id, sink);
  }
}

// path for the nested (ProcessMultiGroup-equivalent) scope of one
// (subgraph, plan, stage) triple.  Empty string means "no nesting" (the
// ProcessSingleGroup-equivalent flat case), matching AppendCommonFusionFields'
// "fusion" scope and never colliding with it.
// 决定一个 plan 的字段怎么命名、怎么嵌套：plan 的坐标，加上两个全局形态开关。
struct PlanFieldNaming {
  size_t subgraph_id;
  size_t plan_id;
  bool use_simple_tiling_key;
  bool is_flat;
};

std::string NestedStagePath(const PlanFieldNaming &naming, size_t stage_id) {
  if (naming.is_flat) {
    return {};
  }
  // ATT 将各 group 的结构体直接挂在最外层 TilingData 上。
  return MakeScheduleGroupTilingFieldName(static_cast<int64_t>(naming.subgraph_id),
                                          static_cast<int64_t>(naming.plan_id), static_cast<int64_t>(stage_id));
}

bool AppendPlanFields(const ascir::ScheduledResult &scheduled_result, const PlanFieldNaming &naming, SchemaSink sink,
                      std::string *diagnostic) {
  const auto &groups = scheduled_result.schedule_groups;
  if (groups.empty()) {
    SetDiagnostic(diagnostic, "schedule result contains no stage");
    return false;
  }
  // 子图以 graph<sub>_tiling_key 选 plan，stage 以自身的 tiling_key 选 template。
  // 单子图、单 plan、单 stage 时共用裸 tiling_key，与 ATT/legacy 一致。
  if (naming.use_simple_tiling_key) {
    AppendField(sink, "tiling_key", "fusion");
  } else {
    AppendField(sink, "graph" + std::to_string(naming.subgraph_id) + "_tiling_key", "fusion");
  }
  for (size_t stage_id = 0U; stage_id < groups.size(); ++stage_id) {
    if (groups[stage_id].impl_graphs.empty()) {
      SetDiagnostic(diagnostic, "schedule stage contains no implementation graph");
      return false;
    }
    // Preserve ATT layout: scalar fields may be flat; API tiling stays group-scoped.
    const std::string field_scope = NestedStagePath(naming, stage_id);
    const std::string api_tiling_scope =
        naming.is_flat
            ? MakeScheduleGroupTilingFieldName(static_cast<int64_t>(naming.subgraph_id),
                                               static_cast<int64_t>(naming.plan_id), static_cast<int64_t>(stage_id))
            : field_scope;
    if (!naming.is_flat) {
      AppendField(sink, "tiling_key", field_scope);
    }
    for (size_t template_id = 0U; template_id < groups[stage_id].impl_graphs.size(); ++template_id) {
      AppendTemplateFields(groups[stage_id].impl_graphs[template_id], field_scope, api_tiling_scope, template_id, sink);
    }
  }
  return true;
}

std::string SanitizeIdentifier(const std::string &text) {
  std::string result;
  result.reserve(text.size());
  for (const char ch : text) {
    result.push_back((std::isalnum(static_cast<unsigned char>(ch)) != 0) ? ch : '_');
  }
  return result;
}

}  // namespace

bool BuildTilingFieldSchema(const ascir::FusedScheduledResult &fused_result, TilingFieldSchema &schema,
                            std::string *diagnostic) {
  schema.fields.clear();
  if (fused_result.node_idx_to_scheduled_results.empty()) {
    SetDiagnostic(diagnostic, "fused schedule contains no subgraph");
    return false;
  }

  // Matches legacy's single top-level `is_group` check (codegen_tiling_data.cpp,
  // TilingData::Generate): exactly one subgraph, one plan, one stage. This is
  // a single global decision, not something recomputed per subgraph/plan —
  // recomputing it locally would let two different subgraphs each qualify for
  // the unprefixed "tiling_key" name and collide.
  const bool is_flat = fused_result.node_idx_to_scheduled_results.size() == 1U &&
                       fused_result.node_idx_to_scheduled_results[0].size() == 1U &&
                       fused_result.node_idx_to_scheduled_results[0][0].schedule_groups.size() == 1U;

  std::set<std::string> seen;
  SchemaSink sink{schema, seen};
  AppendCommonFusionFields(sink);

  for (const auto &workspace : fused_result.workspace_nodes) {
    if (workspace == nullptr || workspace->outputs().empty() || workspace->outputs().front() == nullptr) {
      SetDiagnostic(diagnostic, "workspace node has no output tensor");
      schema.fields.clear();
      return false;
    }
    const auto tensor_id = workspace->outputs().front()->attr.mem.tensor_id;
    if (tensor_id < 0) {
      SetDiagnostic(diagnostic, "workspace tensor id is invalid");
      schema.fields.clear();
      return false;
    }
    AppendField(sink, "workspace" + std::to_string(tensor_id), "fusion", TilingFieldType::kUInt64);
  }

  for (size_t subgraph_id = 0U; subgraph_id < fused_result.node_idx_to_scheduled_results.size(); ++subgraph_id) {
    const auto &plans = fused_result.node_idx_to_scheduled_results[subgraph_id];
    for (size_t plan_id = 0U; plan_id < plans.size(); ++plan_id) {
      const PlanFieldNaming naming{subgraph_id, plan_id, is_flat, is_flat};
      if (!AppendPlanFields(plans[plan_id], naming, sink, diagnostic)) {
        schema.fields.clear();
        return false;
      }
    }
  }
  return true;
}

bool BuildTilingFieldSchema(const ascir::ScheduledResult &scheduled_result, size_t subgraph_id, size_t plan_id,
                            TilingFieldSchema &schema, std::string *diagnostic) {
  schema.fields.clear();
  std::set<std::string> seen;
  SchemaSink sink{schema, seen};
  AppendCommonFusionFields(sink);
  const bool is_flat = scheduled_result.schedule_groups.size() == 1U;
  const PlanFieldNaming naming{subgraph_id, plan_id, /*use_simple_tiling_key=*/true, is_flat};
  if (!AppendPlanFields(scheduled_result, naming, sink, diagnostic)) {
    schema.fields.clear();
    return false;
  }
  return true;
}

namespace {

// 按嵌套作用域分好组的字段。扁平作用域（"" 与 "fusion"）的字段直接落进最外层结构体；
// 其余每个 path 一个嵌套结构体。
struct ScopedFields {
  std::vector<const TilingField *> flat_fields;
  std::vector<std::string> scope_order;
  std::vector<std::vector<const TilingField *>> scope_fields;
};

// 作用域顺序按首次出现，不按字典序——path 形如 "graph2..." / "graph10..."，字典序会把
// 两者排反，发射出来的字段顺序就不再等于 BuildTilingFieldSchema 的遍历顺序（也就不再
// 等于 legacy 的顺序）。
ScopedFields GroupFieldsByScope(const TilingFieldSchema &schema) {
  constexpr std::string_view kFlatScopes[] = {"", "fusion"};
  auto is_flat_scope = [&](const std::string &path) {
    for (const auto &flat : kFlatScopes) {
      if (path == flat) {
        return true;
      }
    }
    return false;
  };

  ScopedFields grouped;
  std::unordered_map<std::string, size_t> scope_index;
  for (const auto &field : schema.fields) {
    if (is_flat_scope(field.path)) {
      grouped.flat_fields.push_back(&field);
      continue;
    }
    auto it = scope_index.find(field.path);
    size_t index;
    if (it == scope_index.end()) {
      index = grouped.scope_order.size();
      scope_index.emplace(field.path, index);
      grouped.scope_order.push_back(field.path);
      grouped.scope_fields.emplace_back();
    } else {
      index = it->second;
    }
    grouped.scope_fields[index].push_back(&field);
  }
  return grouped;
}

void EmitTilingField(std::ostringstream &output, const TilingField &field) {
  switch (field.type) {
    case TilingFieldType::kUInt32:
      output << "  TILING_DATA_FIELD_DEF_T(uint32_t, " << field.name << ");\n";
      break;
    case TilingFieldType::kUInt64:
      output << "  TILING_DATA_FIELD_DEF_T(uint64_t, " << field.name << ");\n";
      break;
    case TilingFieldType::kOpaqueStruct:
      if (field.device_type_name.empty()) {
        output << "  // opaque tiling field: " << field.path << "." << field.name << "\n";
      } else {
        output << "  TILING_DATA_FIELD_DEF_T_STRUCT(optiling::" << field.device_type_name << ", " << field.name
               << ");\n";
      }
      break;
  }
}

}  // namespace

std::string EmitTilingDataStructCpp(const TilingFieldSchema &schema, std::string_view struct_name) {
  const ScopedFields grouped = GroupFieldsByScope(schema);
  std::ostringstream output;

  for (size_t index = 0U; index < grouped.scope_order.size(); ++index) {
    const std::string scope_suffix = SanitizeIdentifier(grouped.scope_order[index]);
    output << "BEGIN_TILING_DATA_DEF_T(" << struct_name << "_" << scope_suffix << ")\n";
    for (const auto *field : grouped.scope_fields[index]) {
      EmitTilingField(output, *field);
    }
    output << "END_TILING_DATA_DEF_T;\n\n";
  }

  output << "BEGIN_TILING_DATA_DEF_T(" << struct_name << ")\n";
  for (const auto *field : grouped.flat_fields) {
    EmitTilingField(output, *field);
  }
  for (size_t index = 0U; index < grouped.scope_order.size(); ++index) {
    const std::string scope_suffix = SanitizeIdentifier(grouped.scope_order[index]);
    output << "  TILING_DATA_FIELD_DEF_T_STRUCT(" << struct_name << "_" << scope_suffix << ", scope_" << scope_suffix
           << ");\n";
  }
  output << "END_TILING_DATA_DEF_T;\n";
  return output.str();
}

std::optional<std::string> LookupFieldName(const TilingFieldSchema &schema, std::string_view semantic_key) {
  for (const auto &field : schema.fields) {
    if (field.name == semantic_key || field.path + "." + field.name == semantic_key) {
      return field.name;
    }
  }
  return std::nullopt;
}

}  // namespace mlir::common
