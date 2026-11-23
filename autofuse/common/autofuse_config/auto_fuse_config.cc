/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "auto_fuse_config.h"
#include <regex>
#include "auto_fuse_config_parser.h"

namespace att {
namespace {
const int32_t kBaseOfIntegerValue = 10;
// 辅助函数：解析 int64_t 配置值
inline bool ParseInt64Config(const std::string &val_str, AutoFuseConfigValue<int64_t> &config_val, int64_t &result_val,
                             bool &is_set) {
  static const std::regex int_regex(R"(^[+-]?\d+$)");
  if (!std::regex_match(val_str, int_regex)) {
    return false;
  }
  try {
    auto set_val = std::stoll(val_str);
    GE_ASSERT_SUCCESS(config_val.SetVal(set_val));
    result_val = config_val.GetVal();
    is_set = true;
    return true;
  } catch (const std::out_of_range &) {
    return false;
  }
}

template <typename T>
inline Status TrySetVal(const std::unordered_map<std::string, std::string> &config, const std::string &key,
                        AutoFuseConfigValue<T> &config_val, T &result_val, bool &is_set) {
  if (config.find(key) == config.end()) {
    return af::SUCCESS;
  }
  const std::string &val_str = config.at(key);

  if constexpr (std::is_same_v<T, int64_t>) {
    ParseInt64Config(val_str, config_val, result_val, is_set);
  } else if constexpr (std::is_same_v<T, std::string>) {
    GE_ASSERT_SUCCESS(config_val.SetVal(val_str));
    result_val = config_val.GetVal();
    is_set = true;
  }
  return af::SUCCESS;
}
}  // namespace

AutoFuseConfig &AutoFuseConfig::Instance() {
  static AutoFuseConfig config;
  return config;
}

const AutoFuseConfig &AutoFuseConfig::Config() {
  return Instance();
}

AutoFuseConfig &AutoFuseConfig::MutableConfig() {
  return Instance();
}

Status AttStrategyConfig::SetEnvVal(std::unordered_map<std::string, std::string> &merged_configs) {
  constexpr int64_t kMaxUbThreshold = 100;
  constexpr int64_t kMaxCorenumThreshold = 100;
  constexpr int64_t kMaxScheduleResultNum = 100;
  AutoFuseConfigValue<std::string> tiling_algorithm_config_val(
      std::string("AxesReorder"), std::vector<std::string>({std::string("AxesReorder"), std::string("HighPerf")}));
  AutoFuseConfigValue<std::string> force_tiling_case_val("", std::vector<std::string>());
  AutoFuseConfigValue<int64_t> force_schedule_result_val(-1L, std::vector<int64_t>({0, kMaxScheduleResultNum}));
  AutoFuseConfigValue<std::string> force_template_op_name_val("", std::vector<std::string>());
  AutoFuseConfigValue<int64_t> solution_accuracy_level_config_val(1L, std::vector<int64_t>({0, 1}));
  AutoFuseConfigValue<std::string> enable_small_shape_strategy_config_val(
      std::string("false"), std::vector<std::string>({std::string("true"), std::string("false")}));
  AutoFuseConfigValue<std::string> enable_multicore_ub_tradeoff_val(
      std::string("false"), std::vector<std::string>({std::string("true"), std::string("false")}));
  AutoFuseConfigValue<int64_t> ub_threshold_config_val(20L, std::vector<int64_t>({0, kMaxUbThreshold}));
  AutoFuseConfigValue<int64_t> corenum_threshold_config_val(40L, std::vector<int64_t>({0, kMaxCorenumThreshold}));
  AutoFuseConfigValue<std::string> att_profiling_val(
      std::string("false"), std::vector<std::string>({std::string("true"), std::string("false")}));
  AutoFuseConfigValue<std::string> enable_tiling_cache_val(
      std::string("true"), std::vector<std::string>({std::string("true"), std::string("false")}));

  // 解析具体的配置
  GE_ASSERT_SUCCESS(TrySetVal(merged_configs, kExperimentalAutofusionAttTilingAlgorithm, tiling_algorithm_config_val,
                              tiling_algorithm, set_env_tiling_algorithm));
  GE_ASSERT_SUCCESS(TrySetVal(merged_configs, kExperimentalAutofusionAttSolutionAccuracyLevel,
                              solution_accuracy_level_config_val, solution_accuracy_level,
                              set_env_solution_accuracy_level));
  GE_ASSERT_SUCCESS(TrySetVal(merged_configs, kExperimentalAutofusionAttEnableSmallShapeStrategy,
                              enable_small_shape_strategy_config_val, enable_small_shape_strategy,
                              set_env_enable_small_shape_strategy));
  GE_ASSERT_SUCCESS(TrySetVal(merged_configs, kExperimentalAutofusionAttUbThreshold, ub_threshold_config_val,
                              ub_threshold, set_env_ub_threshold));
  GE_ASSERT_SUCCESS(TrySetVal(merged_configs, kExperimentalAutofusionAttCorenumThreshold, corenum_threshold_config_val,
                              corenum_threshold, set_env_corenum_threshold));
  GE_ASSERT_SUCCESS(TrySetVal(merged_configs, kExperimentalAutofusionAttEnableMulticoreUBTradeoff,
                              enable_multicore_ub_tradeoff_val, enable_multicore_ub_tradeoff,
                              set_env_enable_multicore_ub_tradeoff));
  GE_ASSERT_SUCCESS(TrySetVal(merged_configs, kExperimentalAutofusionAttProfiling, att_profiling_val, att_profiling,
                              set_env_att_profiling));
  GE_ASSERT_SUCCESS(TrySetVal(merged_configs, kExperimentalAutofusionEnableTilingCache, enable_tiling_cache_val,
                              enable_tiling_cache, set_env_enable_tiling_cache));
  GE_ASSERT_SUCCESS(TrySetVal(merged_configs, kExperimentalAutofusionAttScheduleResult, force_schedule_result_val,
                              force_schedule_result, set_force_schedule_result));
  GE_ASSERT_SUCCESS(TrySetVal(merged_configs, kExperimentalAutofusionAttTilingCase, force_tiling_case_val,
                              force_tiling_case, set_force_tiling_case));
  GE_ASSERT_SUCCESS(TrySetVal(merged_configs, kExperimentalAutofusionAttForceOpName, force_template_op_name_val,
                              force_template_op_name, set_force_template_op_name));
  return af::SUCCESS;
}

Status AutofuseEnvParaParse(std::unordered_map<std::string, std::string> &merged_configs,
                            std::unordered_map<std::string, std::string> &env_configs) {
  // 环境变量的配置属于进程级别，ini的配置属于全局级别，所以环境变量的配置优先级较高
  for (auto &env_config : env_configs) {
    GELOGI("Update config of env, config name=%s, config value=%s", env_config.first.c_str(),
           env_config.second.c_str());
    merged_configs[env_config.first] = env_config.second;
  }

  return af::SUCCESS;
}

Status AttStrategyConfig::Init() {
  ge::AutoFuseEnvConfigParser env_config_parser(
      {}, {
              kExperimentalAutofusionAttTilingAlgorithm, kExperimentalAutofusionAttUbThreshold,
              kExperimentalAutofusionAttCorenumThreshold, kExperimentalAutofusionAttEnableSmallShapeStrategy,
              kExperimentalAutofusionAttEnableMulticoreUBTradeoff, kExperimentalAutofusionAttSolutionAccuracyLevel,
              kExperimentalAutofusionAttProfiling, kExperimentalAutofusionEnableTilingCache,
              kExperimentalAutofusionAttScheduleResult,  // 用于强制模板选择，不对外开放
              kExperimentalAutofusionAttTilingCase,      // 用于强制模板选择，不对外开放
              kExperimentalAutofusionAttForceOpName      // 用于强制模板选择，不对外开放
          });
  if (initialized_) {
    return af::SUCCESS;
  }
  auto env_configs = env_config_parser.Parse();
  if (env_configs.empty()) {
    GELOGI("No configs found, use default configs");
    return af::NOT_CHANGED;
  }
  // 环境变量的配置属于进程级别，ini的配置属于全局级别，所以环境变量的配置优先级较高
  std::unordered_map<std::string, std::string> merged_configs;
  GE_ASSERT_SUCCESS(AutofuseEnvParaParse(merged_configs, env_configs));
  GE_ASSERT_SUCCESS(SetEnvVal(merged_configs));
  GELOGI("Init config [%s=%s], [%s=%ld], [%s=%ld], [%s=%ld], [%s=%s], [%s=%s], [%s=%s], [%s=%ld], [%s=%s], [%s=%s]",
         kExperimentalAutofusionAttTilingAlgorithm, tiling_algorithm.c_str(),
         kExperimentalAutofusionAttSolutionAccuracyLevel, solution_accuracy_level,
         kExperimentalAutofusionAttUbThreshold, ub_threshold, kExperimentalAutofusionAttCorenumThreshold,
         corenum_threshold, kExperimentalAutofusionAttEnableSmallShapeStrategy, enable_small_shape_strategy.c_str(),
         kExperimentalAutofusionAttEnableMulticoreUBTradeoff, enable_multicore_ub_tradeoff.c_str(),
         kExperimentalAutofusionAttProfiling, att_profiling.c_str(), kExperimentalAutofusionAttScheduleResult,
         force_schedule_result, kExperimentalAutofusionAttTilingCase, force_tiling_case.c_str(),
         kExperimentalAutofusionAttForceOpName, force_template_op_name.c_str());
  initialized_ = true;
  return af::SUCCESS;
}

Status AttStrategyConfig::Reset() {
  *this = AttStrategyConfig();
  return af::SUCCESS;
}

Status PgoStrategyConfig::SetEnvVal(std::unordered_map<std::string, std::string> &merged_configs) {
  constexpr int64_t kMaxPgoStepVal = 1024;
  AutoFuseConfigValue<std::string> enable_autofuse_pgo_val(
      std::string("false"), std::vector<std::string>({std::string("true"), std::string("false")}));
  AutoFuseConfigValue<std::string> autofuse_pgo_algo_val(
      std::string("core_select"), std::vector<std::string>({std::string("core_select"), std::string("pruning")}));
  AutoFuseConfigValue<int64_t> autofuse_pgo_algo_step_max_val(16L, std::vector<int64_t>({1L, kMaxPgoStepVal}));

  // 解析具体的配置
  GE_ASSERT_SUCCESS(TrySetVal(merged_configs, kExperimentalAutofusionEnablePGO, enable_autofuse_pgo_val,
                              enable_autofuse_pgo, set_env_enable_autofuse_pgo));
  GE_ASSERT_SUCCESS(TrySetVal(merged_configs, kExperimentalAutofusionEnablePgoOptAlgo, autofuse_pgo_algo_val,
                              autofuse_pgo_algo_select, set_env_autofuse_pgo_algo_select));
  GE_ASSERT_SUCCESS(TrySetVal(merged_configs, kExperimentalAutofusionEnablePgoStepMax, autofuse_pgo_algo_step_max_val,
                              autofuse_pgo_algo_step_max, set_env_autofuse_pgo_algo_step_max));

  return af::SUCCESS;
}

Status PgoStrategyConfig::Init() {
  if (!is_first_init) {
    return af::SUCCESS;
  }
  ge::AutoFuseEnvConfigParser env_config_parser(
      {kExperimentalAutofusionEnablePGO},
      {kExperimentalAutofusionEnablePgoOptAlgo, kExperimentalAutofusionEnablePgoStepMax});
  std::unordered_map<std::string, std::string> merged_configs;
  auto env_configs = env_config_parser.Parse();
  if (env_configs.empty()) {
    GELOGI("No configs found, use default configs");
    return af::NOT_CHANGED;
  }
  // pgo 相关环境变量配置
  GE_ASSERT_SUCCESS(AutofuseEnvParaParse(merged_configs, env_configs));
  GE_ASSERT_SUCCESS(SetEnvVal(merged_configs));
  GELOGI("Init config [%s=%s] [%s=%s] [%s=%ld]", kExperimentalAutofusionEnablePGO, enable_autofuse_pgo.c_str(),
         kExperimentalAutofusionEnablePgoOptAlgo, autofuse_pgo_algo_select.c_str(),
         kExperimentalAutofusionEnablePgoStepMax, autofuse_pgo_algo_step_max);
  is_first_init = false;
  return af::SUCCESS;
}
}  // namespace att
