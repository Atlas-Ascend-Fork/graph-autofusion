/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_OPTION_REGISTER_H_
#define ATT_OPTION_REGISTER_H_

#include <vector>
#include <map>
#include <regex>
#include <unordered_set>
#include "common/checker.h"

namespace att {
struct OptionInfo {
  std::string name;
  std::string default_value;
  std::string expected;

  bool (*validate_func)(const std::string &value);
};

class OptionRegister {
 public:
  void RegisterOption(const std::string &name, const std::string &default_value,
                      bool (*validate_func)(const std::string &value), const std::string &expected = "");

  bool ValidateAndInitInnerOptions(std::map<std::string, std::string> &inner_options,
                                   const std::map<std::string, std::string> &options);

 private:
  std::vector<OptionInfo> registered_options_;
  std::unordered_set<std::string> registered_options_name_;
};

bool ValidateBoolOption(const std::string &value);

bool ValidatePathOption(const std::string &value);

bool ValidateNotEmptyOption(const std::string &value);

bool ValidateNotNegativeNumber(const std::string &value);

bool ValidateIdentifierName(const std::string &value);

bool ValidateNotValidType(const std::string &value);

bool RegisterOptionsAndInitInnerOptions(std::map<std::string, std::string> &inner_options,
                                        const std::map<std::string, std::string> &options,
                                        const std::string &graphs_name);
}  // namespace att

#endif  // ATT_OPTION_REGISTER_H_
