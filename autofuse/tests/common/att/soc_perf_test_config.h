/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_SOC_PERF_TEST_CONFIG_H_
#define ATT_SOC_PERF_TEST_CONFIG_H_

#include <utility>
#include "base/att_const_values.h"
#include "common/platform_context.h"
#include "gen_model_info/api_perf_register/soc_perf_config.h"
#include "graph/ascendc_ir/ascir_registry.h"

namespace att {
namespace soc_perf_test {
inline constexpr char kV35NpuArch[] = "5102";
inline constexpr char kV35AltNpuArch[] = "3510";
inline constexpr char kSocA[] = "SOC_A";
inline constexpr char kSocB[] = "SOC_B";
inline constexpr char kSocEmpty[] = "SOC_EMPTY";
inline constexpr char kRegisteredNpuArch[] = "SocPerfRegisteredV2Arch";

inline void RegisterSocPerfTestArchitecture() {
  for (const std::string &api : {std::string("Abs"), kDefaultApi}) {
    const auto &api_def = af::ascir::AscirRegistry::GetInstance().GetAll().at(api);
    api_def.AddSocImpl(
        {kRegisteredNpuArch},
        {[api] { return af::ascir::AscirRegistry::GetInstance().GetIrAttImpl(kV35AltNpuArch, api); }, nullptr, {}});
  }
}

inline af::Status SetSocPerfTestPlatform(const std::string &npu_arch, const std::string &soc_version) {
  ge::PlatformInfo platform_info;
  platform_info.soc_ver = npu_arch;
  platform_info.aiv_num = 48;
  platform_info.ub_size = 245760;
  return ge::PlatformContext::GetInstance().SetPlatformTarget(platform_info, soc_version);
}

inline Json MakeApiOverride(float k, float b) {
  Json api_override = Json::object();
  api_override["Abs"]["model_params"]["float16tofloat16"] = {{"k", k}, {"b", b}};
  return api_override;
}

inline const SocPerfConfigs &GetSocPerfTestConfigs() {
  static const SocPerfConfigs configs = [] {
    SocPerfConfigs result;
    SocPerfConfig soc_a;
    soc_a.api_param_overrides = MakeApiOverride(1.0F, 10.0F);
    soc_a.ascir_model_overrides = {{"AbsV2", "ExpV2"}, {"SelectV2", "ExpV2"}, {"Tanh", "ExpV2"}};
    soc_a.ascendc_model_overrides = {{"LoadV2", "StoreV2"}};
    result.emplace(SocPerfKey{kV35NpuArch, kSocA}, soc_a);
    result.emplace(SocPerfKey{kV35AltNpuArch, kSocA}, soc_a);

    SocPerfConfig soc_b;
    soc_b.api_param_overrides = MakeApiOverride(2.0F, 20.0F);
    soc_b.ascir_model_overrides = {{"AbsV2", "AbsV2"}};
    soc_b.ascendc_model_overrides = {{"LoadV2", "LoadV2"}};
    result.emplace(SocPerfKey{kV35NpuArch, kSocB}, soc_b);
    result.emplace(SocPerfKey{kV35NpuArch, kSocEmpty}, SocPerfConfig{});
    result.emplace(SocPerfKey{kV35AltNpuArch, kSocEmpty}, SocPerfConfig{});
    return result;
  }();
  return configs;
}
}  // namespace soc_perf_test
}  // namespace att

#endif  // ATT_SOC_PERF_TEST_CONFIG_H_
