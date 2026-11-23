/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/option/optimization_option.h"
#include "framework/common/debug/ge_log.h"
#include "common/ge_common/string_util.h"
#include "common/checker.h"
#include "graph/ge_local_context.h"
#include "ge_common_af/ge_api_types_af.h"

namespace af {
namespace {
const std::unordered_map<std::string, OoLevel> kOptValToLevels{
    {"O1", OoLevel::kO1},
    {"O3", OoLevel::kO3},
};

void ReportParamInvalid(const std::string &opt_name, const std::string &opt_value, const std::string &reason) {
  REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char_t *>({"parameter", "value", "reason"}),
                            std::vector<const char_t *>({opt_name.c_str(), opt_value.c_str(), reason.c_str()}));
  GELOGE(GRAPH_PARAM_INVALID, "[Oo][Check] the value [%s] of option [%s] is invalid. %s", opt_value.c_str(),
         opt_name.c_str(), reason.c_str());
}
}  // namespace

graphStatus OptimizationOption::Initialize(const std::map<std::string, std::string> &ge_options,
                                           const std::unordered_map<std::string, OoInfo> &registered_options) {
  return Initialize(ge_options, registered_options, std::unordered_set<std::string>{});
}

graphStatus OptimizationOption::Initialize(const std::map<std::string, std::string> &ge_options,
                                           const std::unordered_map<std::string, OoInfo> &registered_options,
                                           const std::unordered_set<std::string> &forbidden_option_set) {
  working_oo_level_ = OoLevel::kEnd;
  working_opt_names_to_value_.clear();
  // 1. Initialize OoLevel if possible
  if (InitWorkingOolevel(ge_options) != GRAPH_SUCCESS) {
    return GRAPH_PARAM_INVALID;
  }
  // 2. Expand optimization template
  for (const auto &opt_info : registered_options) {
    if (OoInfoUtils::IsBitSet(opt_info.second.levels, static_cast<uint32_t>(working_oo_level_))) {
      const auto value_str = OoInfoUtils::GetDefaultValue(opt_info.second, working_oo_level_);
      (void)working_opt_names_to_value_.emplace(opt_info.first, value_str);
    }
  }

  // 解析ge.optimizationSwitch的值，更新working_opt_names_to_value_
  UpdatePassSwitchByOption(ge_options, forbidden_option_set);

  // 3. Verify user-configured optimization options
  for (const auto &ge_opt : ge_options) {
    const auto iter = registered_options.find(ge_opt.first);
    if (iter == registered_options.cend()) {
      continue;
    }
    if (IsOptionValueValid(ge_opt.first, ge_opt.second, iter->second.checker) != GRAPH_SUCCESS) {
      return GRAPH_PARAM_INVALID;
    }
    working_opt_names_to_value_[ge_opt.first] = ge_opt.second;
  }

  PrintAllWorkingOo();
  GELOGI("Init optimization option success");
  return GRAPH_SUCCESS;
}

graphStatus OptimizationOption::GetValue(const std::string &opt_name, std::string &opt_value) const {
  const auto iter = working_opt_names_to_value_.find(opt_name);
  if (iter == working_opt_names_to_value_.cend()) {
    return GRAPH_FAILED;
  }
  opt_value = iter->second;
  return GRAPH_SUCCESS;
}

graphStatus OptimizationOption::IsOoLevelValid(const std::string &oo_level) {
  const auto &oo_level_iter = kOptValToLevels.find(oo_level);
  if (oo_level_iter == kOptValToLevels.end()) {
    ReportParamInvalid(GetThreadLocalContext().GetReadableName(::af::OO_LEVEL), oo_level,
                       "The value must be O1 or O3.");
    return GRAPH_PARAM_INVALID;
  }
  return GRAPH_SUCCESS;
}

graphStatus OptimizationOption::IsOptionValueValid(const std::string &opt_name, const std::string &opt_value,
                                                   OoInfo::ValueChecker checker) {
  if (checker == nullptr) {
    return GRAPH_SUCCESS;
  }
  std::string reason = "Invalid optimization option value.";
  if (!checker(opt_value, reason)) {
    if (reason.empty()) {
      reason = "Invalid optimization option value.";
    }
    ReportParamInvalid(GetThreadLocalContext().GetReadableName(opt_name), opt_value, reason);
    return PARAM_INVALID;
  }
  return GRAPH_SUCCESS;
}

graphStatus OptimizationOption::InitWorkingOolevel(const std::map<std::string, std::string> &ge_options) {
  const auto opt_iter = ge_options.find(af::OO_LEVEL);
  if (opt_iter == ge_options.end()) {
    // default oo_level is O3 if ge_option is not set
    working_oo_level_ = OoLevel::kO3;
  } else {
    if (IsOoLevelValid(opt_iter->second) != GRAPH_SUCCESS) {
      return GRAPH_PARAM_INVALID;
    }
    working_oo_level_ = kOptValToLevels.at(opt_iter->second);
  }
  GELOGI("[Oo][Print]working_oo_level is %u.", working_oo_level_);
  return GRAPH_SUCCESS;
}

bool OptimizationOption::IsPassConfiguredWithOptimizationSwitch(const std::string &pass_name) const {
  /**
   * pass开关若是通过绑定oo level写入到working_opt_names_to_value_中，键是passname，值是true/false/空
   * 若是通过optimization_switch配置写入，键是passname，值是on/off
   */
  const auto iter = working_opt_names_to_value_.find(pass_name);
  return (iter != working_opt_names_to_value_.end() && (iter->second == "on" || iter->second == "off"));
}

graphStatus OptimizationOption::SetPassSwitch(const std::string &pass_switch_str,
                                              const std::unordered_set<std::string> &forbidden_option_set,
                                              bool force_update) {
  GELOGI("Begin to set pass switch with option optimization_switch [%s]", pass_switch_str.c_str());
  std::stringstream ss(pass_switch_str);
  std::string token;

  // 拆分每一对 pass:switch
  while (std::getline(ss, token, ';')) {
    size_t pos = token.find(':');
    // 1. 格式错误处理，记录Warning日志
    if (pos == std::string::npos) {
      GELOGW("[Oo][SetPassSwitch] Invalid token format: %s", token.c_str());
      continue;
    }

    // 2. 校验冒号前后内容是否为空，冒号后面只能是on/off
    std::string pass_name = token.substr(0, pos);
    std::string pass_switch = token.substr(pos + 1);
    if (pass_name.empty() || (pass_switch != "on" && pass_switch != "off")) {
      GELOGW("[Oo][SetPassSwitch] Invalid key or value in token: %s", token.c_str());
      continue;
    }

    // 3. 黑名单检查：不能配置ge option，记录Warning日志
    if (forbidden_option_set.find(pass_name) != forbidden_option_set.end()) {
      GELOGW("[Oo][SetPassSwitch] [%s] is one of ge option names, can not be configured here", pass_name.c_str());
      continue;
    }

    // 4. 如果不是强制更新，则跳过已经通过optimization_switch配置的pass
    if (!force_update && IsPassConfiguredWithOptimizationSwitch(pass_name)) {
      GELOGW("[Oo][SetPassSwitch] [%s] is already configured, skip it", pass_name.c_str());
      continue;
    }

    working_opt_names_to_value_[pass_name] = pass_switch;
    GELOGD("[Oo][SetPassSwitch]the switch of pass [%s] is [%s]", pass_name.c_str(), pass_switch.c_str());
  }

  return GRAPH_SUCCESS;
}

graphStatus OptimizationOption::UpdatePassSwitchByOption(const std::map<std::string, std::string> &ge_options,
                                                         const std::unordered_set<std::string> &forbidden_option_set) {
  const auto iter = ge_options.find(af::OPTIMIZATION_SWITCH);
  if (iter == ge_options.end()) {
    GELOGI("No need to init optimization switch");
    return GRAPH_SUCCESS;
  }
  // ge.optimizationSwitch的配置为最高优先级，强制更新
  return SetPassSwitch(iter->second, forbidden_option_set, true);
}

void OptimizationOption::PrintAllWorkingOo() {
  for (const auto &iter : working_opt_names_to_value_) {
    GELOGD("[Oo][Print]the value [%s] of option [%s] is set successfully", iter.second.c_str(), iter.first.c_str());
  }
}

graphStatus OptimizationOption::RefreshPassSwitch(const std::string &fusion_config_str) {
  std::string optimization_switch;
  if (GetThreadLocalContext().GetOption(OPTIMIZATION_SWITCH, optimization_switch) == GRAPH_SUCCESS &&
      (optimization_switch != "forbidden_close_pass:on" && optimization_switch != "forbidden_close_pass:off")) {
    // 1.
    // 以optimization_switch的配置优先，重复配置的option，使用optimization_switch的配置;FE带过来的fusion_config_str中只有pass开关，不做黑名单校验
    // 2. tfa/atc场景默认写入optimization_switch为optimization_switch ==
    // "forbidden_close_pass:on/off"，此时以fusion_config_str中的配置为准
    // 3. forbidden_close_pass对外不可见，因此torch入口传过来的optimization_switch不会为"forbidden_close_pass:on/off"
    return SetPassSwitch(fusion_config_str, std::unordered_set<std::string>{}, false);
  } else {
    // fusion_cfg的配置优先级高于O3;FE带过来的fusion_config_str中只有pass开关，不做黑名单校验
    return SetPassSwitch(fusion_config_str, std::unordered_set<std::string>{}, true);
  }
}
}  // namespace af
