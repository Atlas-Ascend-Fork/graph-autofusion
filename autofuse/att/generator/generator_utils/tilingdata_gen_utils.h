/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_TILINGDATA_GEN_UTILS_H_
#define ATT_TILINGDATA_GEN_UTILS_H_
#include <string>
#include <set>
#include <map>
#include "code_printer.h"

namespace att {
class TilingDataGenUtils {
 public:
  template <typename Container>
  static bool NeedWrittenTilingData(const Container &var_names, std::set<std::string> &tiling_data_vars) {
    for (const auto &var_name : var_names) {
      const auto iter = tiling_data_vars.find(var_name);
      if (iter == tiling_data_vars.end()) {
        return true;
      }
    }
    return false;
  }
  template <typename Container>
  static void WriteTilingDataElement(ge::CodePrinter &printer, std::set<std::string> &tiling_data_vars,
                                     const Container &var_names) {
    for (const auto &var_name : var_names) {
      const auto iter = tiling_data_vars.find(var_name);
      if (iter == tiling_data_vars.end()) {
        AddElementDefinition(printer, GetFieldType(var_name), var_name);
        tiling_data_vars.insert(var_name);
      }
    }
  }
  template <typename Container>
  static void WriteTilingDataStruct(ge::CodePrinter &printer, std::set<std::string> &tiling_data_vars,
                                    const std::string &var_type, const Container &var_name) {
    const auto iter = tiling_data_vars.find(var_name);
    if (iter == tiling_data_vars.end()) {
      AddStructElementDefinition(printer, var_type, var_name);
      tiling_data_vars.insert(var_name);
    }
  }
  static std::string GetFieldType(const std::string &var_name);
  static void AddElementDefinition(ge::CodePrinter &printer, const std::string &type_name, const std::string &var_name);
  static void AddStructElementDefinition(ge::CodePrinter &printer, const std::string &type_name,
                                         const std::string &var_name);
  static std::string WriteTilingDataElement(
      std::set<std::string> &tiling_data_vars,
      const std::map<std::string, std::pair<std::string, std::string>> &var_names);
  static std::string StructElementDefine(const std::string &type_name, const std::string &details);
};
}  // namespace att
#endif  // ATT_TILINGDATA_GEN_UTILS_H_
