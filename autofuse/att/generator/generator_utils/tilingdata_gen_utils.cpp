/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "generator_utils/tilingdata_gen_utils.h"

namespace att {
std::string TilingDataGenUtils::GetFieldType(const std::string &var_name,
                                             const std::set<std::string> &dynamic_size_var_names) {
  if (var_name == "block_dim" || var_name == "tiling_key") {
    return "uint16_t";
  }
  // 是否动态以前端 ABI 符号集合（GetFrontendShapeVars + !IsConstExpr）为准，不再按命名推断。
  if (dynamic_size_var_names.count(var_name) != 0U) {
    return "int64_t";
  }
  const std::string tiling_key_suffix = "_tiling_key";
  if (var_name.size() > tiling_key_suffix.size() &&
      var_name.compare(var_name.size() - tiling_key_suffix.size(), tiling_key_suffix.size(), tiling_key_suffix) == 0) {
    return "uint16_t";
  }
  return "uint32_t";
}

void TilingDataGenUtils::AddElementDefinition(ge::CodePrinter &printer, const std::string &type_name,
                                              const std::string &var_name) {
  printer.AddLine(("    TILING_DATA_FIELD_DEF(" + type_name + ", ") + var_name + ")");
}

void TilingDataGenUtils::AddStructElementDefinition(ge::CodePrinter &printer, const std::string &type_name,
                                                    const std::string &var_name) {
  printer.AddLine(("    TILING_DATA_FIELD_DEF_STRUCT(" + type_name + ", ") + var_name + ")");
}

std::string TilingDataGenUtils::WriteTilingDataElement(
    std::set<std::string> &tiling_data_vars,
    const std::map<std::string, std::pair<std::string, std::string>> &var_names) {
  ge::CodePrinter printer;
  for (const auto &var_name : var_names) {
    const auto iter = tiling_data_vars.find(var_name.second.second);
    if (iter == tiling_data_vars.end()) {
      AddStructElementDefinition(printer, var_name.second.second, var_name.second.first);
      tiling_data_vars.insert(var_name.second.first);
    }
  }
  return printer.GetOutputStr();
}

std::string TilingDataGenUtils::StructElementDefine(const std::string &type_name, const std::string &details) {
  std::string struct_define("BEGIN_TILING_DATA_DEF(" + type_name + ")\n");
  struct_define += details;
  struct_define += "END_TILING_DATA_DEF";
  return struct_define;
}
}  // namespace att
