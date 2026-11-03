/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/fingerprint/schedule_fingerprint.h"

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string_view>

namespace mlir::common {
namespace {

class StableHasher {
 public:
  void Add(std::string_view value) {
    AddLengthPrefix(value.size());
    HashBytes(value);
  }

  template <typename Integer>
  void AddInteger(Integer value) {
    const std::string text = std::to_string(value);
    AddLengthPrefix(text.size());
    HashBytes(text);
  }

  std::string Finish() const {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(16) << first_ << std::setw(16) << second_;
    return stream.str();
  }

 private:
  void HashBytes(std::string_view value) {
    for (const unsigned char byte : value) {
      first_ = (first_ ^ byte) * kFnvPrime;
      second_ = (second_ ^ (byte + static_cast<unsigned char>(first_ >> 32U))) * kFnvPrime;
    }
  }

  // Every Add()/AddInteger() call must be self-delimiting so that two
  // adjacent calls can never hash to the same byte stream as a differently
  // split pair (e.g. AddInteger(1); AddInteger(23) must not collide with
  // AddInteger(12); AddInteger(3)). A raw byte is enough: it can represent
  // any length up to 255, far more than a size_t's decimal representation
  // (<=20 digits) ever needs, and unlike recursing back into Add()/
  // AddInteger() to hash the length itself, it terminates in one step.
  void AddLengthPrefix(size_t length) {
    const unsigned char byte = static_cast<unsigned char>(length < 0xFFU ? length : 0xFFU);
    first_ = (first_ ^ byte) * kFnvPrime;
    second_ = (second_ ^ (byte + static_cast<unsigned char>(first_ >> 32U))) * kFnvPrime;
  }

  static constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
  static constexpr uint64_t kFnvPrime = 1099511628211ULL;

  uint64_t first_{kFnvOffset};
  uint64_t second_{kFnvOffset ^ 0x9e3779b97f4a7c15ULL};
};

void AddExpression(StableHasher &hasher, const af::Expression &expression) {
  const auto text = expression.Str();
  hasher.Add(text == nullptr ? std::string_view{} : std::string_view(text.get()));
}

void AddAscendString(StableHasher &hasher, const char *value) {
  hasher.Add(value == nullptr ? std::string_view{} : std::string_view(value));
}

void AddTemplate(StableHasher &hasher, const af::AscGraph &graph) {
  hasher.Add(graph.GetName());
  const auto size_vars = graph.GetAllSizeVar();
  // Skip constant-expression size vars, matching
  // tiling_schema/tiling_field_schema.cpp's AppendTemplateFields. Constant size
  // vars do not become TilingData fields and do not vary with the runtime
  // shape, so including them here would make otherwise-identical schedules
  // hash differently based only on how a constant happened to be spelled
  // internally, and would make the fingerprint depend on information that
  // BuildTilingFieldSchema deliberately discards -- the two are meant to
  // agree on what counts as "part of the schedule's structure".
  std::vector<decltype(size_vars.front().get())> variable_size_vars;
  variable_size_vars.reserve(size_vars.size());
  for (const auto &size_var : size_vars) {
    if (size_var != nullptr && size_var->expr.IsConstExpr()) {
      continue;
    }
    variable_size_vars.push_back(size_var.get());
  }
  hasher.AddInteger(variable_size_vars.size());
  for (const auto *size_var : variable_size_vars) {
    if (size_var == nullptr) {
      hasher.Add("<null-size-var>");
      continue;
    }
    AddExpression(hasher, size_var->expr);
  }
}

void AddPlan(StableHasher &hasher, const ascir::ScheduledResult &scheduled_result) {
  AddAscendString(hasher, scheduled_result.score_func.GetString());
  hasher.AddInteger(static_cast<int32_t>(scheduled_result.is_reduce_mem_reuse));
  hasher.AddInteger(static_cast<int32_t>(scheduled_result.enable_group_parallel));
  hasher.AddInteger(static_cast<int32_t>(scheduled_result.cube_type));

  hasher.AddInteger(scheduled_result.var_relations.size());
  for (const auto &[dst_stage, source_stages] : scheduled_result.var_relations) {
    hasher.AddInteger(dst_stage);
    hasher.AddInteger(source_stages.size());
    for (const auto &[src_stage, relations] : source_stages) {
      hasher.AddInteger(src_stage);
      hasher.AddInteger(relations.size());
      for (const auto &[dst_var, src_expr] : relations) {
        hasher.Add(dst_var);
        AddExpression(hasher, src_expr);
      }
    }
  }

  hasher.AddInteger(scheduled_result.schedule_groups.size());
  for (const auto &group : scheduled_result.schedule_groups) {
    hasher.AddInteger(static_cast<int32_t>(group.double_buffer));
    hasher.AddInteger(group.graph_name_to_score_funcs.size());
    for (const auto &[graph_name, score_func] : group.graph_name_to_score_funcs) {
      hasher.Add(graph_name);
      hasher.Add(score_func);
    }
    hasher.AddInteger(group.impl_graphs.size());
    for (const auto &graph : group.impl_graphs) {
      AddTemplate(hasher, graph);
    }
  }
}

void AddVersions(StableHasher &hasher, const ArtifactVersionSet &versions) {
  hasher.Add(versions.target_profile_version);
  hasher.Add(versions.cann_toolchain_version);
  hasher.Add(versions.bridge_version);
  hasher.Add(versions.att_schema_version);
  hasher.Add(versions.codegen_version);
  hasher.Add(versions.pass_pipeline_version);
  hasher.Add(versions.entry_abi_version);
  // Dynamic- and static-mode artifacts, and static artifacts compiled for
  // different concrete shapes, must never collide on one cache key (see the
  // field comments on ArtifactVersionSet). std::map iterates in key order,
  // so this is deterministic regardless of insertion order.
  hasher.AddInteger(static_cast<int32_t>(versions.is_dynamic_shape));
  hasher.AddInteger(versions.static_shape_constants.size());
  for (const auto &[name, dimension] : versions.static_shape_constants) {
    hasher.Add(name);
    hasher.AddInteger(dimension);
  }
}

}  // namespace

ScheduleFingerprint ComputeScheduleFingerprint(const ascir::FusedScheduledResult &fused_result) {
  StableHasher hasher;
  hasher.Add(kScheduleFingerprintFormatVersion);
  AddAscendString(hasher, fused_result.fused_graph_name.GetString());
  hasher.AddInteger(fused_result.frontend_shape_vars_collected);
  hasher.AddInteger(fused_result.frontend_shape_vars.size());
  for (const auto &shape_var : fused_result.frontend_shape_vars) {
    AddExpression(hasher, shape_var);
  }
  hasher.AddInteger(fused_result.origin_vars.size());
  for (const auto &origin_var : fused_result.origin_vars) {
    AddExpression(hasher, origin_var);
  }
  hasher.AddInteger(fused_result.node_idx_to_scheduled_results.size());
  for (const auto &scheduled_results : fused_result.node_idx_to_scheduled_results) {
    hasher.AddInteger(scheduled_results.size());
    for (const auto &scheduled_result : scheduled_results) {
      AddPlan(hasher, scheduled_result);
    }
  }
  return {hasher.Finish()};
}

ArtifactCacheKey BuildArtifactCacheKey(const ScheduleFingerprint &structure, const ArtifactVersionSet &versions,
                                       const CompileOptions &options) {
  StableHasher hasher;
  hasher.Add(kArtifactCacheKeyFormatVersion);
  hasher.Add(structure.value);
  AddVersions(hasher, versions);
  hasher.AddInteger(options.values.size());
  for (const auto &[name, value] : options.values) {
    hasher.Add(name);
    hasher.Add(value);
  }
  return {hasher.Finish()};
}

}  // namespace mlir::common
