/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tiling_code_gen_impl.h"

#include <array>

#include "util/base_types_printer.h"
#include "common/checker.h"
#include "args_manager.h"

namespace att {

void TilingCodeGenImpl::GenSingleGroupFinalTilingObserveBegin() {
  if (!is_uniq_group_) {
    return;
  }
  tiling_func_.AddLine("  const bool emit_final_tiling = ShouldEmitFinalTiling();");
  tiling_func_.AddLine("  FinalTilingObservationSlotGuard final_tiling_observation_slot_guard(0U);");
}

void TilingCodeGenImpl::GenSingleGroupFinalTilingObserveReset() {
  // The call guard is destroyed at function exit. Cache hits reset the current slot explicitly.
}

void TilingCodeGenImpl::GenSingleGroupFinalTilingObserveEmit(const bool need_operator_cache) {
  // Inductor entry functions and CV safety fallback emit the final record themselves.  They also cover the
  // PGO-selected tiling path, where this GetTiling function is not called, so emitting here would duplicate logs.
  if (!is_uniq_group_ || config_.is_inductor_scene) {
    return;
  }
  const std::string cache_guard = need_operator_cache ? " && !cache_hit" : "";
  const auto &ident = tiling_model_info_[0].schedule_group_ident;
  tiling_func_.AddLine("  if (ret && emit_final_tiling" + cache_guard + ") {");
  tiling_func_.AddLine(
      "    EmitFinalTilingByCase(tiling_data, tiling_data.get_block_dim(), \"runtime\", "
      "tiling_case_id == -1 ? \"default\" : \"explicit\", " +
      std::to_string(ident.group_id) + "U, -1);");
  tiling_func_.AddLine("  }");
}

af::Status TilingCodeGenImpl::GenSingleGroupGetTilingCoreAlias() {
  // Inductor TopN/PGO entries probe the default tiling through the raw GetTilingCore symbol, mirroring the
  // multi-group contract.  The single-group GetTiling never emits final tiling records in the Inductor scene,
  // so forwarding keeps the probe free of observability side effects.
  if (!config_.is_inductor_scene) {
    return af::SUCCESS;
  }
  tiling_func_.AddLine("bool GetTilingCore(" + config_.tiling_data_type_name +
                       " &tiling_data, int32_t tiling_case_id, double *perf) {");
  AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kApi,
                      "bool GetTilingCore(" + config_.tiling_data_type_name +
                          " &tiling_data, int32_t tiling_case_id, double *perf = nullptr);");
  tiling_func_.AddLine("  return GetTilingWithObservation(tiling_data, tiling_case_id, perf);");
  tiling_func_.AddLine("}");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenFinalTilingReprFunction() {
  // The single-group Inductor entry uses the frontend ABI helper generated for
  // AutofuseTilingData. Multi-group entries have group-local tiling data types,
  // so generate the same JSON representation as the GE path instead of falling
  // back to a key-only record.
  if (config_.is_inductor_scene && is_uniq_group_) {
    return af::SUCCESS;
  }

  std::set<std::string> fields;
  GE_ASSERT_SUCCESS(CollectFinalTilingReprFields(fields), "Collect final tiling representation fields failed.");
  fields.erase("tiling_key");

  tiling_func_.AddLine("inline std::string GetTilingDataRepr(const " + config_.tiling_data_type_name +
                       " *tiling_data) {");
  tiling_func_.AddLine("  if (tiling_data == nullptr) { return std::string(); }");
  tiling_func_.AddLine("  std::stringstream repr;");
  tiling_func_.AddLine("  repr << \"{\\\"tiling_key\\\":\" << tiling_data->get_tiling_key()");
  for (const auto &field : fields) {
    if (!field.empty()) {
      tiling_func_.AddLine("      << \",\\\"" + field + "\\\":\" << tiling_data->get_" + field + "()");
    }
  }
  tiling_func_.AddLine("      << \"}\";");
  tiling_func_.AddLine("  return repr.str();");
  tiling_func_.AddLine("}");
  tiling_func_.AddLine("");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::CollectFinalTilingReprFields(std::set<std::string> &fields) {
  for (const auto &model_info : tiling_model_info_) {
    ArgsManager args_manager(model_info);
    GE_ASSERT_TRUE(args_manager.Process(false), "Process model info for final tiling representation failed.");
    for (const auto &arg : args_manager.GetInputVars()) {
      fields.insert(Str(arg));
    }
    for (const auto &arg : args_manager.GetSearchableVars()) {
      fields.insert(Str(arg));
    }
    for (const auto &container : model_info.container_exprs) {
      fields.insert(container.first);
    }
    for (const auto &hardware : args_manager.GetTotalHardwareCons(config_.do_variable_replace)) {
      fields.insert(BaseTypeUtils::DumpHardware(hardware.first));
    }
    if (config_.gen_extra_infos) {
      const std::array<TilingDataGenType, 3U> data_types = {TilingDataGenType::AXES_TILING_DATA_GEN,
                                                            TilingDataGenType::GENERAL_TILING_DATA_GEN,
                                                            TilingDataGenType::MEMORY_TILING_DATA_GEN};
      for (const auto type : data_types) {
        if (type == TilingDataGenType::AXES_TILING_DATA_GEN && !model_info.sub_case_tag.empty()) {
          continue;
        }
        for (const auto &data : tiling_data_manager_.GetTilingDataWithAnnotation(model_info.tiling_case_id, type)) {
          fields.insert(data.first);
        }
      }
    }
  }
  return af::SUCCESS;
}

}  // namespace att
