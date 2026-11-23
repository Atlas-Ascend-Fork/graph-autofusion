/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "option_register.h"
#include "common/checker.h"
#include "base/att_const_values.h"

namespace att {
void OptionRegister::RegisterOption(const std::string &name, const std::string &default_value,
                                    bool (*validate_func)(const std::string &value), const std::string &expected) {
  OptionInfo option;
  option.name = name;
  option.default_value = default_value;
  option.expected = expected;
  option.validate_func = validate_func;
  registered_options_.push_back(option);
  registered_options_name_.insert(name);
}

bool RegisterOptionsAndInitInnerOptions(std::map<std::string, std::string> &inner_options,
                                        const std::map<std::string, std::string> &options,
                                        const std::string &graphs_name) {
  using namespace att;
  OptionRegister option_register = OptionRegister();
  // RegisterOption( option名称， option默认值（空字符串视为无默认值）， 校验函数 ）
  // bool类型
  option_register.RegisterOption(kGenTilingDataDef, kIsTrue, ValidateBoolOption, "0|1");
  option_register.RegisterOption(kHighPrecision, kIsFalse, ValidateBoolOption, "0|1");
  option_register.RegisterOption(kGenExtraInfo, kIsFalse, ValidateBoolOption, "0|1");
  // 路径信息
  option_register.RegisterOption(kDumpDebugInfo, kRegisterNoDefault, ValidatePathOption,
                                 "non-empty path matching [A-Za-z\\d/_.-]+");
  option_register.RegisterOption(kOutputFilePath, kDefaultFilePath, ValidatePathOption,
                                 "non-empty path matching [A-Za-z\\d/_.-]+");
  // 命名信息
  option_register.RegisterOption(kTilingDataTypeName, graphs_name + "TilingData", ValidateIdentifierName,
                                 "identifier matching ^[a-zA-Z_][a-zA-Z0-9_]*$");
  // DurationLevel
  option_register.RegisterOption(kDurationLevelName, kDurationLevelDefault, ValidateNotNegativeNumber,
                                 "non-negative integer");
  // SolverType
  option_register.RegisterOption(kGenConfigType, kGenConfigTypeDefault, ValidateNotValidType, "HighPerf|AxesReorder");
  // 之后删除的预留信息，变量替换
  option_register.RegisterOption(kVariableReplace, kIsTrue, ValidateBoolOption, "0|1");
  return option_register.ValidateAndInitInnerOptions(inner_options, options);
}

bool OptionRegister::ValidateAndInitInnerOptions(std::map<std::string, std::string> &inner_options,
                                                 const std::map<std::string, std::string> &options) {
  // 检测是否有未注册的option
  for (const auto &pair : options) {
    if (registered_options_name_.find(pair.first) == registered_options_name_.cend()) {
      GELOGE(af::FAILED, "Option validation failed for option '%s'. It is not registered.", pair.first.c_str());
      return false;
    }
  }
  // 校验并设置默认值
  for (auto const &registered_option : registered_options_) {
    const auto iter = options.find(registered_option.name);
    if (iter != options.cend()) {
      // 开始校验
      if (!registered_option.validate_func(iter->second)) {
        GELOGE(af::FAILED, "Invalid option[%s] = %s, expected %s.", iter->first.c_str(), iter->second.c_str(),
               registered_option.expected.empty() ? "<see option description>" : registered_option.expected.c_str());
        return false;
      } else {
        inner_options[registered_option.name] = iter->second;
      }
    } else if (registered_option.default_value != "") {  // ""视为无默认值
      // 设置默认值
      inner_options[registered_option.name] = registered_option.default_value;
      GELOGI("SetDefaultValue inner_options[%s] = %s", registered_option.name.c_str(),
             inner_options[registered_option.name].c_str());
    }
  }
  return true;
}

bool ValidateBoolOption(const std::string &value) {
  if (value != "0" && value != "1") {
    return false;
  }
  return true;
}

bool ValidatePathOption(const std::string &value) {
  if (value.empty()) {
    return false;
  }
  constexpr char KValidChars[] = "[A-Za-z\\d/_.-]+";
  std::regex e(KValidChars);
  std::smatch sm;
  if (!std::regex_match(value, sm, e)) {
    return false;
  }
  return true;
}

bool ValidateNotEmptyOption(const std::string &value) {
  if (value.empty()) {
    return false;
  }
  return true;
}

bool ValidateIdentifierName(const std::string &value) {
  std::regex identifier_regex("^[a-zA-Z_][a-zA-Z0-9_]*$");
  return std::regex_match(value, identifier_regex);
}

bool ValidateNotNegativeNumber(const std::string &value) {
  if (value.empty() || value[0] == '-') {
    return false;
  }
  for (char c : value) {
    if (!std::isdigit(c)) {
      return false;
    }
  }
  return true;
}

bool ValidateNotValidType(const std::string &value) {
  if (value != "HighPerf" && value != "AxesReorder") {
    return false;
  }
  return true;
}
}  // namespace att
