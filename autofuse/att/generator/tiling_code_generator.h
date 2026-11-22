/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_TILING_CODE_GENERATOR_H_
#define ATT_TILING_CODE_GENERATOR_H_

#include "base/model_info.h"
#include "tiling_code_gen_impl.h"
#include "generator_config.h"
#include "high_perf_tiling_code_gen_impl.h"
#include "axes_reorder_tiling_code_gen_impl.h"

namespace att {
struct GenTilingParams {
  std::string op_type;
  TilingModelInfo all_model_infos;
  TilingCodeGenConfig config;
  std::unordered_map<std::string, std::string> cache_reuse_info;
};

struct GenTilingTailExtParams {
  ScoreFuncs score_funcs;
  VarRelations var_relations;
  EnableGroupParallels enable_group_parallels;
  TensorIdSet workspace_tensor_id_set;
  size_t final_tiling_observation_slot_count{1UL};
};

class TilingCodeGenerator {
 public:
  af::Status GenTilingCode(const std::string &op_type, const TilingModelInfo &model_infos,
                           const TilingCodeGenConfig &config, std::map<std::string, std::string> &tiling_res);
  af::Status GenTilingCode(const std::string &op_type, const TilingModelInfo &model_infos,
                           const TilingCodeGenConfig &config);
  // for autofuse
  af::Status GenTilingCode(const std::string &op_type, const FusedParsedScheduleResult &fused_parsed_schedule_result,
                           const TilingCodeGenConfig &config, std::map<std::string, std::string> &tiling_res);

 protected:
  virtual TilingCodeGenImplPtr CreateTilingCodeGenImpl(const std::string &op_name, const TilingCodeGenConfig &config,
                                                       const TilingModelInfo &model_infos,
                                                       const ScoreFuncs &score_funcs, const bool is_uniq_group);

 private:
  af::Status GenTilingHead(const std::string &op_type, const TilingModelInfo &all_model_infos,
                           const TilingCodeGenConfig &config, std::map<std::string, std::string> &tiling_res,
                           const EnableGroupParallels &enable_group_parallels);
  af::Status GenTilingBody(const GenTilingParams &params, std::map<std::string, std::string> &tiling_res,
                           const bool is_uniq_group, uint32_t cache_capacity,
                           const EnableGroupParallels &enable_group_parallels);
  af::Status GenTilingTail(const GenTilingParams &params, std::map<std::string, std::string> &tiling_res,
                           const GenTilingTailExtParams &ext_params);

  // 辅助方法：收集所有 model_infos 和相关元数据
  af::Status CollectModelInfosAndMetadata(const FusedParsedScheduleResult &fused_parsed_schedule_result,
                                          TilingModelInfo &all_model_infos, size_t &group_num,
                                          ScoreFuncs &schedule_result_score_func, VarRelations &var_relations,
                                          EnableGroupParallels &enable_group_parallels,
                                          TensorIdSet &workspace_tensor_id_set);
  // 辅助方法：生成所有 schedule group 的 tiling 代码
  af::Status GenScheduleGroupTilingBodies(const std::string &op_type,
                                          const FusedParsedScheduleResult &fused_parsed_schedule_result,
                                          const TilingCodeGenConfig &config,
                                          const std::unordered_map<std::string, std::string> &cache_reuse_info,
                                          uint32_t cache_capacity, const EnableGroupParallels &enable_group_parallels,
                                          std::map<std::string, std::string> &tiling_res);
  af::Status GenScheduleGroupTilingGroup(const std::string &op_type, const TilingCodeGenConfig &config,
                                         const std::unordered_map<std::string, std::string> &cache_reuse_info,
                                         uint32_t cache_capacity, const EnableGroupParallels &enable_group_parallels,
                                         size_t group_num, const std::pair<const size_t, TilingModelInfo> &group_graphs,
                                         const std::map<std::string, int32_t> &final_tiling_sub_case_ids,
                                         size_t final_tiling_observation_slot,
                                         std::map<std::string, std::string> &tiling_res);
  af::Status GenScheduleResult(const std::string &op_type, const TilingCodeGenConfig &config,
                               const std::unordered_map<std::string, std::string> &cache_reuse_info,
                               uint32_t cache_capacity, const EnableGroupParallels &enable_group_parallels,
                               const ParsedScheduleResult &result,
                               const std::map<std::string, int32_t> &final_tiling_sub_case_ids,
                               size_t &final_tiling_observation_slot, std::map<std::string, std::string> &tiling_res);
  void MergeGeneratedHeaders(const TilingCodeGenImpl &impl);

  std::map<autofuse::GeneratedHeaderId, autofuse::GeneratedCode> generated_headers_;
};
}  // namespace att
#endif
