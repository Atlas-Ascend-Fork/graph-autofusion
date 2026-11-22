/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "extra_info_gen/extra_info_generator.h"
#include <set>
#include <iostream>
#include <regex>
#include "common/checker.h"
#include "code_printer.h"
#include "generator_utils/tilingdata_gen_utils.h"
#include "tiling_data_gen/tiling_data_generator.h"
#include "gen_model_info/api_tiling_gen/gen_api_tiling.h"

namespace att {
//  --------------------------------以下为tilingdata定义---------------------------
//  返回值std::string tilingdata定义
std::string ExtraInfoGenerator::WriteCoreParamData(const ModelInfo &model_info,
                                                   const TilingDataGenType tiling_data_gen_type,
                                                   std::set<std::string> &tiling_data_vars) {
  ge::CodePrinter printer;
  const auto &tiling_datas =
      tiling_data_generator_.GetTilingDataWithAnnotation(model_info.tiling_case_id, tiling_data_gen_type);
  for (const auto &tiling_data_name : tiling_datas) {
    std::vector<std::string> tiling_data_name_vec{tiling_data_name.first};
    if (TilingDataGenUtils::NeedWrittenTilingData(tiling_data_name_vec, tiling_data_vars)) {
      printer.AddLine(tiling_data_name.second);
      TilingDataGenUtils::WriteTilingDataElement(printer, tiling_data_vars, tiling_data_name_vec,
                                                 model_info.dynamic_size_var_names);
      GELOGD("Write tiling data: name[%s]", tiling_data_name.first.c_str());
    }
  }
  return printer.GetOutputStr();
}

af::Status ExtraInfoGenerator::GetExtraTilingDataDef(std::map<std::string, std::string> &type_name_to_definition) {
  std::set<std::string> tiling_data_vars;
  for (const auto &model_info : model_info_list_) {
    if (!model_info.sub_case_tag.empty()) {
      continue;
    }
    // 轴对应的TilingData相关参数
    type_name_to_definition["CoreParams"] +=
        WriteCoreParamData(model_info, TilingDataGenType::AXES_TILING_DATA_GEN, tiling_data_vars);
  }
  return af::SUCCESS;
}

af::Status ExtraInfoGenerator::GetExtraTilingVars(const uint32_t tiling_key, std::set<std::string> &tiling_vars) {
  const auto model_info = GetModelInfo(tiling_key);
  GE_ASSERT_NOTNULL(model_info);
  // 轴对应的TilingData相关参数
  WriteCoreParamData(*model_info, TilingDataGenType::AXES_TILING_DATA_GEN, tiling_vars);
  return af::SUCCESS;
}

const ModelInfo *ExtraInfoGenerator::GetModelInfo(const uint32_t tiling_key) const {
  for (const auto &model_info : model_info_list_) {
    if (model_info.tiling_case_id == tiling_key) {
      return &model_info;
    }
  }
  return nullptr;
}
}  // namespace att
