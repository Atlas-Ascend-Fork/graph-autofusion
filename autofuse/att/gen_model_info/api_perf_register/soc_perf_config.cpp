/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "soc_perf_config.h"
#include "common/platform_context.h"
#include "common_utils.h"
#include "api_perf_factory.h"
#include "ascendc_api_perf.h"

#include <cmath>
#include <mutex>
#include <set>

namespace att {
namespace {
const char kProbeApiName[] = "Abs";
const char kV2RegisterKey[] = "V2";
const char kModelParamsKey[] = "model_params";
const char kModelTypeKey[] = "model_type";
const char kLoadStoreFuncModel[] = "LoadStoreFunc";
const char kLoadUbStrideModel[] = "LoadUbStride";
const std::set<std::string> kForbiddenRecordFields = {"data_type_size", "block_count_idx"};

struct SocPerfCacheEntry {
  SocPerfConfig config;
  std::string base_json;
  std::shared_ptr<const Json> params;
};

std::mutex g_soc_perf_cache_mutex;
std::map<SocPerfKey, std::shared_ptr<const SocPerfCacheEntry>> g_soc_perf_cache;
std::mutex g_soc_api_params_baseline_mutex;
std::map<std::string, std::weak_ptr<const Json>> g_soc_api_params_baseline_cache;
std::mutex g_soc_ascir_registration_keys_mutex;
std::map<std::string, std::set<std::string>> g_soc_ascir_registration_keys_cache;

af::Status GetProbeParamTable(const std::string &npu_arch, const PerfParamTable *&param_table) {
  param_table = nullptr;
  if (npu_arch.empty()) {
    return af::FAILED;
  }

  const auto asc_ir_att_impl = af::ascir::AscirRegistry::GetInstance().GetIrAttImpl(npu_arch, kProbeApiName);
  if (asc_ir_att_impl == nullptr) {
    return af::FAILED;
  }
  const auto api_perf_name = ge::PtrToPtr<void, ge::char_t>(asc_ir_att_impl->GetApiPerf());
  if (api_perf_name == nullptr) {
    return af::FAILED;
  }
  auto api_perf = ApiPerfFactory::Instance().Create(api_perf_name);
  if (api_perf == nullptr) {
    return af::FAILED;
  }
  param_table = api_perf->GetPerfParam();
  return param_table == nullptr ? af::FAILED : af::SUCCESS;
}

af::Status GetBaselineJson(const SocPerfTarget &target, const std::string *&base_json) {
  base_json = nullptr;
  const PerfParamTable *param_table = nullptr;
  if (GetProbeParamTable(target.npu_arch, param_table) != af::SUCCESS) {
    return af::FAILED;
  }
  base_json = param_table->GetAscendCApiPerfTable();
  return base_json == nullptr ? af::FAILED : af::SUCCESS;
}

af::Status ParseJsonObject(const std::string &json_str, Json &json) {
  json = Json::object();
  try {
    json = Json::parse(json_str);
  } catch (const nlohmann::json::exception &) {
    return af::FAILED;
  }
  return json.is_object() ? af::SUCCESS : af::FAILED;
}

bool IsFiniteFloat(const Json &value) {
  return value.is_number() && std::isfinite(static_cast<float>(value.get<double>()));
}

bool IsZeroFloat(const Json &value) {
  const float actual = static_cast<float>(value.get<double>());
  return std::fpclassify(actual) == FP_ZERO;
}

af::Status ValidateRecord(const Json &override_record, const Json &baseline_record, const std::string &model_type) {
  if (!override_record.is_object() || !baseline_record.is_object()) {
    return af::FAILED;
  }

  std::set<std::string> baseline_fields;
  for (auto item = baseline_record.begin(); item != baseline_record.end(); ++item) {
    if (kForbiddenRecordFields.find(item.key()) == kForbiddenRecordFields.end()) {
      baseline_fields.insert(item.key());
    }
  }
  if (baseline_fields.empty()) {
    return af::FAILED;
  }

  std::set<std::string> override_fields;
  for (auto item = override_record.begin(); item != override_record.end(); ++item) {
    if (kForbiddenRecordFields.find(item.key()) != kForbiddenRecordFields.end()) {
      return af::FAILED;
    }
    if (baseline_fields.find(item.key()) == baseline_fields.end() || !IsFiniteFloat(item.value())) {
      return af::FAILED;
    }
    override_fields.insert(item.key());
  }
  if (override_fields != baseline_fields) {
    return af::FAILED;
  }
  if (model_type == kLoadStoreFuncModel || model_type == kLoadUbStrideModel) {
    const auto divisor = override_record.find("a");
    if (divisor != override_record.end() && IsZeroFloat(divisor.value())) {
      return af::FAILED;
    }
  }
  return af::SUCCESS;
}

af::Status ValidateDtypeOverrides(const Json &override_params, const Json &baseline_params,
                                  const std::string &model_type) {
  if (!override_params.is_object() || !baseline_params.is_object()) {
    return af::FAILED;
  }
  for (auto item = override_params.begin(); item != override_params.end(); ++item) {
    const auto baseline_record = baseline_params.find(item.key());
    if (baseline_record == baseline_params.end() || !baseline_record.value().is_object() || !item.value().is_object()) {
      return af::FAILED;
    }
    if (ValidateRecord(item.value(), baseline_record.value(), model_type) != af::SUCCESS) {
      return af::FAILED;
    }
  }
  return af::SUCCESS;
}

af::Status ValidateApiOverride(const Json &override_api, const Json &baseline_api) {
  if (!override_api.is_object() || !baseline_api.is_object() || override_api.size() != 1U ||
      !override_api.contains(kModelParamsKey)) {
    return af::FAILED;
  }
  const auto model_type = baseline_api.find(kModelTypeKey);
  if (model_type == baseline_api.end() || !model_type.value().is_string()) {
    return af::FAILED;
  }
  if (!baseline_api.contains(kModelParamsKey)) {
    return af::FAILED;
  }
  return ValidateDtypeOverrides(override_api.at(kModelParamsKey), baseline_api.at(kModelParamsKey),
                                model_type.value().get<std::string>());
}

af::Status ValidateApiOverrides(const Json &overrides, const Json &baseline) {
  if (!overrides.is_object()) {
    return af::FAILED;
  }
  if (overrides.empty()) {
    return af::SUCCESS;
  }
  for (auto item = overrides.begin(); item != overrides.end(); ++item) {
    const auto baseline_api = baseline.find(item.key());
    if (baseline_api == baseline.end()) {
      return af::FAILED;
    }
    if (ValidateApiOverride(item.value(), baseline_api.value()) != af::SUCCESS) {
      return af::FAILED;
    }
  }
  return af::SUCCESS;
}

af::Status GetSocAscIrRegistrationKeys(const std::string &npu_arch, const std::set<std::string> *&registration_keys) {
  registration_keys = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_soc_ascir_registration_keys_mutex);
    const auto cache_iter = g_soc_ascir_registration_keys_cache.find(npu_arch);
    if (cache_iter != g_soc_ascir_registration_keys_cache.end()) {
      registration_keys = &cache_iter->second;
      return registration_keys->empty() ? af::FAILED : af::SUCCESS;
    }
  }

  std::set<std::string> new_registration_keys;
  for (const auto &ascir_item : af::ascir::AscirRegistry::GetInstance().GetAll()) {
    const auto asc_ir_att_impl = af::ascir::AscirRegistry::GetInstance().GetIrAttImpl(npu_arch, ascir_item.first);
    if (asc_ir_att_impl == nullptr) {
      continue;
    }
    const auto api_perf_name = ge::PtrToPtr<void, ge::char_t>(asc_ir_att_impl->GetApiPerf());
    auto api_perf = api_perf_name == nullptr ? ApiPerfFactory::Instance().Create(ascir_item.first)
                                             : ApiPerfFactory::Instance().Create(api_perf_name);
    if (api_perf == nullptr || api_perf->GetPerfFunc() == nullptr) {
      continue;
    }
    new_registration_keys.insert(api_perf->GetApiName());
  }
  if (new_registration_keys.empty()) {
    return af::FAILED;
  }

  std::lock_guard<std::mutex> lock(g_soc_ascir_registration_keys_mutex);
  const auto cache_iter = g_soc_ascir_registration_keys_cache.emplace(npu_arch, std::move(new_registration_keys)).first;
  registration_keys = &cache_iter->second;
  return af::SUCCESS;
}

af::Status ValidateAscIrModelOverrides(const SocPerfTarget &target,
                                       const std::map<std::string, std::string> &overrides) {
  if (overrides.empty()) {
    return af::SUCCESS;
  }

  const std::set<std::string> *registration_keys = nullptr;
  if (GetSocAscIrRegistrationKeys(target.npu_arch, registration_keys) != af::SUCCESS) {
    return af::FAILED;
  }
  for (const auto &item : overrides) {
    if (item.first.empty() || item.second.empty()) {
      return af::FAILED;
    }
    if (registration_keys->find(item.first) == registration_keys->end() ||
        EvalCosts::Instance().GetFunc(item.second) == nullptr) {
      return af::FAILED;
    }
  }
  return af::SUCCESS;
}

af::Status ValidateAscendCModelOverrides(const std::map<std::string, std::string> &overrides) {
  for (const auto &item : overrides) {
    if (item.first.empty() || item.second.empty()) {
      return af::FAILED;
    }
    if (EvalCosts::Instance().GetAscendCFunc(item.first) == nullptr ||
        EvalCosts::Instance().GetAscendCFunc(item.second) == nullptr) {
      return af::FAILED;
    }
  }
  return af::SUCCESS;
}

af::Status FindSocPerfConfig(const SocPerfTarget &target, const SocPerfConfigs &configs, const SocPerfConfig *&config) {
  config = nullptr;
  bool required = false;
  if (RequiresFullSocVersion(target.npu_arch, required) != af::SUCCESS || !required) {
    return af::FAILED;
  }
  auto iter = configs.find(SocPerfKey{target.npu_arch, target.soc_version});
  if (iter == configs.end()) {
    iter = configs.find(SocPerfKey{target.npu_arch, ""});
  }
  if (iter == configs.end()) {
    static const SocPerfConfig baseline_config;
    config = &baseline_config;
  } else {
    config = &iter->second;
  }
  return af::SUCCESS;
}

std::shared_ptr<const Json> GetBaselineParams(const std::string &base_json) {
  {
    std::lock_guard<std::mutex> lock(g_soc_api_params_baseline_mutex);
    const auto iter = g_soc_api_params_baseline_cache.find(base_json);
    if (iter != g_soc_api_params_baseline_cache.end()) {
      auto cached_params = iter->second.lock();
      if (cached_params != nullptr) {
        return cached_params;
      }
    }
  }

  Json baseline;
  if (ParseJsonObject(base_json, baseline) != af::SUCCESS) {
    return nullptr;
  }
  auto params = std::make_shared<const Json>(std::move(baseline));

  std::lock_guard<std::mutex> lock(g_soc_api_params_baseline_mutex);
  const auto cache_result = g_soc_api_params_baseline_cache.emplace(base_json, params);
  if (!cache_result.second) {
    auto cached_params = cache_result.first->second.lock();
    if (cached_params != nullptr) {
      return cached_params;
    }
    cache_result.first->second = params;
  }
  return params;
}

void ApplyApiOverrides(Json &params, const Json &api_param_overrides) {
  for (auto api_iter = api_param_overrides.begin(); api_iter != api_param_overrides.end(); ++api_iter) {
    const auto &model_params = api_iter.value().at(kModelParamsKey);
    for (auto dtype_iter = model_params.begin(); dtype_iter != model_params.end(); ++dtype_iter) {
      params[api_iter.key()][kModelParamsKey][dtype_iter.key()] = dtype_iter.value();
    }
  }
}

std::shared_ptr<const Json> CreateSocApiParams(const std::shared_ptr<const Json> &baseline,
                                               const SocPerfConfig &config) {
  if (config.api_param_overrides.empty()) {
    return baseline;
  }
  Json params = *baseline;
  ApplyApiOverrides(params, config.api_param_overrides);
  return std::make_shared<const Json>(std::move(params));
}

af::Status ValidateSocPerfConfigParts(const SocPerfTarget &target, const SocPerfConfig &config,
                                      const std::string *&base_json, std::shared_ptr<const Json> &baseline) {
  if (ValidateAscIrModelOverrides(target, config.ascir_model_overrides) != af::SUCCESS ||
      ValidateAscendCModelOverrides(config.ascendc_model_overrides) != af::SUCCESS) {
    return af::FAILED;
  }
  if (GetBaselineJson(target, base_json) != af::SUCCESS) {
    return af::FAILED;
  }
  baseline = GetBaselineParams(*base_json);
  if (baseline == nullptr) {
    return af::FAILED;
  }
  return ValidateApiOverrides(config.api_param_overrides, *baseline);
}

std::shared_ptr<const SocPerfCacheEntry> CreateBuiltinSocPerfCacheEntry(const SocPerfTarget &target) {
  const SocPerfConfigs &configs = GetBuiltinSocPerfConfigs();
  const SocPerfConfig *config = nullptr;
  if (FindSocPerfConfig(target, configs, config) != af::SUCCESS) {
    return nullptr;
  }

  const std::string *base_json = nullptr;
  std::shared_ptr<const Json> baseline;
  if (ValidateSocPerfConfigParts(target, *config, base_json, baseline) != af::SUCCESS) {
    return nullptr;
  }
  auto params = CreateSocApiParams(baseline, *config);
  if (params == nullptr) {
    return nullptr;
  }

  auto entry = std::make_shared<SocPerfCacheEntry>();
  entry->config = *config;
  entry->base_json = *base_json;
  entry->params = std::move(params);
  return entry;
}

std::shared_ptr<const SocPerfCacheEntry> GetBuiltinSocPerfCacheEntry(const SocPerfTarget &target) {
  SocPerfKey key{target.npu_arch, target.soc_version};
  if (GetBuiltinSocPerfConfigs().find(key) == GetBuiltinSocPerfConfigs().end()) {
    key.second.clear();
  }
  std::lock_guard<std::mutex> lock(g_soc_perf_cache_mutex);
  const auto cache_iter = g_soc_perf_cache.find(key);
  if (cache_iter != g_soc_perf_cache.end()) {
    return cache_iter->second;
  }

  auto entry = CreateBuiltinSocPerfCacheEntry(target);
  if (entry == nullptr) {
    return nullptr;
  }
  return g_soc_perf_cache.emplace(key, entry).first->second;
}

af::Status ResolveSocAscIrPerfFromConfig(const SocPerfTarget &target, const std::string &api_registration_key,
                                         Perf default_func, Perf &selected_func, const SocPerfConfig &config) {
  selected_func = nullptr;
  if (default_func == nullptr || api_registration_key.empty()) {
    return af::FAILED;
  }
  const auto override_iter = config.ascir_model_overrides.find(api_registration_key);
  if (override_iter == config.ascir_model_overrides.end()) {
    selected_func = default_func;
    return af::SUCCESS;
  }

  const std::set<std::string> *registration_keys = nullptr;
  if (GetSocAscIrRegistrationKeys(target.npu_arch, registration_keys) != af::SUCCESS ||
      registration_keys->find(api_registration_key) == registration_keys->end()) {
    return af::FAILED;
  }
  selected_func = EvalCosts::Instance().GetFunc(override_iter->second);
  return selected_func == nullptr ? af::FAILED : af::SUCCESS;
}

af::Status ResolveSocAscendCPerfFromConfig(const std::string &api_registration_key, AscendCPerf default_func,
                                           AscendCPerf &selected_func, const SocPerfConfig &config) {
  selected_func = nullptr;
  if (default_func == nullptr || api_registration_key.empty() ||
      EvalCosts::Instance().GetAscendCFunc(api_registration_key) == nullptr) {
    return af::FAILED;
  }
  const auto override_iter = config.ascendc_model_overrides.find(api_registration_key);
  if (override_iter == config.ascendc_model_overrides.end()) {
    selected_func = default_func;
    return af::SUCCESS;
  }
  selected_func = EvalCosts::Instance().GetAscendCFunc(override_iter->second);
  return selected_func == nullptr ? af::FAILED : af::SUCCESS;
}
}  // namespace

af::Status ReadSocPerfTarget(SocPerfTarget &target) {
  return ge::PlatformContext::GetInstance().GetPlatformTarget(target.npu_arch, target.soc_version);
}

af::Status RequiresFullSocVersion(const std::string &npu_arch, bool &required) {
  required = false;
  const PerfParamTable *param_table = nullptr;
  if (GetProbeParamTable(npu_arch, param_table) != af::SUCCESS) {
    return af::FAILED;
  }

  const auto registered_key = param_table->GetApiRegisterVerName();
  if (registered_key.empty()) {
    return af::SUCCESS;
  }
  if (registered_key == kV2RegisterKey) {
    required = true;
    return af::SUCCESS;
  }
  return af::FAILED;
}

af::Status ValidateSocPerfEntry() {
  SocPerfTarget target;
  if (ReadSocPerfTarget(target) != af::SUCCESS) {
    return af::FAILED;
  }

  bool required = false;
  if (RequiresFullSocVersion(target.npu_arch, required) != af::SUCCESS) {
    return af::FAILED;
  }
  if (!required) {
    return af::SUCCESS;
  }
  return ValidateSocPerfConfig(target);
}

af::Status ValidateSocPerfConfig(const SocPerfTarget &target, const SocPerfConfigs &configs) {
  const SocPerfConfig *config = nullptr;
  if (FindSocPerfConfig(target, configs, config) != af::SUCCESS) {
    return af::FAILED;
  }
  const std::string *base_json = nullptr;
  std::shared_ptr<const Json> baseline;
  return ValidateSocPerfConfigParts(target, *config, base_json, baseline);
}

af::Status ValidateSocPerfConfig(const SocPerfTarget &target) {
  return GetBuiltinSocPerfCacheEntry(target) == nullptr ? af::FAILED : af::SUCCESS;
}

af::Status GetSocApiParams(const SocPerfTarget &target, const std::string &base_json, const SocPerfConfigs &configs,
                           std::shared_ptr<const Json> &params) {
  params.reset();
  const SocPerfConfig *config = nullptr;
  if (FindSocPerfConfig(target, configs, config) != af::SUCCESS) {
    return af::FAILED;
  }
  const std::string *baseline_json = nullptr;
  std::shared_ptr<const Json> baseline;
  if (ValidateSocPerfConfigParts(target, *config, baseline_json, baseline) != af::SUCCESS ||
      base_json != *baseline_json) {
    return af::FAILED;
  }

  params = CreateSocApiParams(baseline, *config);
  return params == nullptr ? af::FAILED : af::SUCCESS;
}

af::Status GetSocApiParams(const SocPerfTarget &target, const std::string &base_json,
                           std::shared_ptr<const Json> &params) {
  params.reset();
  const auto cache_entry = GetBuiltinSocPerfCacheEntry(target);
  if (cache_entry == nullptr || base_json != cache_entry->base_json) {
    return af::FAILED;
  }
  params = cache_entry->params;
  return af::SUCCESS;
}

af::Status ResolveSocAscIrPerf(const SocPerfTarget &target, const std::string &api_registration_key, Perf default_func,
                               Perf &selected_func, const SocPerfConfigs &configs) {
  const SocPerfConfig *config = nullptr;
  if (FindSocPerfConfig(target, configs, config) != af::SUCCESS) {
    return af::FAILED;
  }
  return ResolveSocAscIrPerfFromConfig(target, api_registration_key, default_func, selected_func, *config);
}

af::Status ResolveSocAscIrPerf(const SocPerfTarget &target, const std::string &api_registration_key, Perf default_func,
                               Perf &selected_func) {
  const auto cache_entry = GetBuiltinSocPerfCacheEntry(target);
  if (cache_entry == nullptr) {
    return af::FAILED;
  }
  return ResolveSocAscIrPerfFromConfig(target, api_registration_key, default_func, selected_func, cache_entry->config);
}

af::Status ResolveSocAscendCPerf(const SocPerfTarget &target, const std::string &api_registration_key,
                                 AscendCPerf default_func, AscendCPerf &selected_func, const SocPerfConfigs &configs) {
  const SocPerfConfig *config = nullptr;
  if (FindSocPerfConfig(target, configs, config) != af::SUCCESS) {
    return af::FAILED;
  }
  return ResolveSocAscendCPerfFromConfig(api_registration_key, default_func, selected_func, *config);
}

af::Status ResolveSocAscendCPerf(const SocPerfTarget &target, const std::string &api_registration_key,
                                 AscendCPerf default_func, AscendCPerf &selected_func) {
  const auto cache_entry = GetBuiltinSocPerfCacheEntry(target);
  if (cache_entry == nullptr) {
    return af::FAILED;
  }
  return ResolveSocAscendCPerfFromConfig(api_registration_key, default_func, selected_func, cache_entry->config);
}
}  // namespace att
