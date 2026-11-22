/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEN_MODEL_INFO_H_
#define GEN_MODEL_INFO_H_

#include <string>
#include <vector>
#include <map>
#include <set>
#include <stdint.h>

#include "base/model_info.h"
#include "parser/tuning_space.h"
#include "autofuse_config/auto_fuse_config.h"

namespace att {
struct ModelInfoSetupContext {
  const ascir::FusedScheduledResult &schedule_results;
  const ascir::ScheduledResult &schedule_result;
  const std::vector<af::AscGraph> &group_graphs;
  const std::map<std::string, std::string> &all_graph_score_funcs;
  const std::set<std::string> &dynamic_size_var_names;
  size_t asc_graph_id;
  size_t impl_graph_id;
  size_t schedule_group_id;
  bool enable_matmul_m_common_tradeoff{false};
};

af::Status GenerateModelInfo(const af::AscGraph &graph, ModelInfo &model_info, TuningSpacePtr &tuning_space,
                             const uint32_t tiling_case_id = 0U);
af::Status GenerateModelInfo(const std::vector<af::AscGraph> &graph_list, std::vector<ModelInfo> &model_info_list);
af::Status GenerateModelInfo(const std::vector<af::AscGraph> &graph_list, std::vector<ModelInfo> &model_info_list,
                             const std::map<std::string, std::string> &options, bool enable_group_parallel = false);
af::Status GenerateModelInfo(const std::vector<af::AscGraph> &graph_list, std::vector<ModelInfo> &model_info_list,
                             const std::map<std::string, std::string> &options, bool enable_group_parallel,
                             bool enable_gather_reduce_penalty);
af::Status GetModelInfoMap(const ascir::FusedScheduledResult &schedule_results,
                           const std::map<std::string, std::string> &options,
                           std::map<size_t, std::map<size_t, ParsedScheduleResult>> &out_all_model_infos);
af::Status GetAllSubImplGraphs(const ascir::FusedScheduledResult &schedule_results,
                               std::vector<std::vector<std::vector<std::vector<af::AscGraph>>>> &all_graphs,
                               std::map<std::string, std::string> &all_graph_score_funcs);
af::Status MakeJson(std::vector<ModelInfo> &model_info_list, std::string &json_info);
}  // namespace att

#endif  // GEN_MODEL_INFO_H_
