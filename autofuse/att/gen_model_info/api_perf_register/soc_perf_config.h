/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_SOC_PERF_CONFIG_H_
#define ATT_SOC_PERF_CONFIG_H_

#include <map>
#include <memory>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

#include "api_perf.h"
#include "ge_common_af/ge_api_error_codes_af.h"

namespace att {
using Json = nlohmann::json;

struct SocPerfTarget {
  std::string npu_arch;
  std::string soc_version;
};

using SocPerfKey = std::pair<std::string, std::string>;

struct SocPerfConfig {
  Json api_param_overrides = Json::object();
  std::map<std::string, std::string> ascir_model_overrides;
  std::map<std::string, std::string> ascendc_model_overrides;
};

using SocPerfConfigs = std::map<SocPerfKey, SocPerfConfig>;

const SocPerfConfigs &GetBuiltinSocPerfConfigs();

af::Status ReadSocPerfTarget(SocPerfTarget &target);
af::Status RequiresFullSocVersion(const std::string &npu_arch, bool &required);
af::Status ValidateSocPerfEntry();
af::Status ValidateSocPerfConfig(const SocPerfTarget &target, const SocPerfConfigs &configs);
af::Status ValidateSocPerfConfig(const SocPerfTarget &target);
af::Status GetSocApiParams(const SocPerfTarget &target, const std::string &base_json, const SocPerfConfigs &configs,
                           std::shared_ptr<const Json> &params);
af::Status GetSocApiParams(const SocPerfTarget &target, const std::string &base_json,
                           std::shared_ptr<const Json> &params);
af::Status ResolveSocAscIrPerf(const SocPerfTarget &target, const std::string &api_registration_key, Perf default_func,
                               Perf &selected_func, const SocPerfConfigs &configs);
af::Status ResolveSocAscIrPerf(const SocPerfTarget &target, const std::string &api_registration_key, Perf default_func,
                               Perf &selected_func);
af::Status ResolveSocAscendCPerf(const SocPerfTarget &target, const std::string &api_registration_key,
                                 AscendCPerf default_func, AscendCPerf &selected_func, const SocPerfConfigs &configs);
af::Status ResolveSocAscendCPerf(const SocPerfTarget &target, const std::string &api_registration_key,
                                 AscendCPerf default_func, AscendCPerf &selected_func);
}  // namespace att

#endif  // ATT_SOC_PERF_CONFIG_H_
