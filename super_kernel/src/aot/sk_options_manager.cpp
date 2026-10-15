/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sk_options_manager.cpp
 * \brief Implementation of SuperKernelOptionsManager and option value classes
 */

#include <cctype>
#include <array>
#include <vector>
#include <cstdlib>
#include <limits>
#include <string>
#include <unordered_set>

#include "sk_options_manager.h"
#include "sk_common.h"
#include "common/pattern_common.h"
#include "sk_log.h"
#include <nlohmann/json.hpp>

namespace {
constexpr size_t kMaxExtendOptionLength = 1024;

using DefaultOptionFactory = std::unique_ptr<OptOptionBase> (*)();

struct DefaultOptionFactoryEntry {
  aclskOptionType optType;
  DefaultOptionFactory factory;
};

const std::array<DefaultOptionFactoryEntry, static_cast<size_t>(aclskOptionType::SK_OPTION_MAX)>
    DEFAULT_OPTION_FACTORIES = {{
        {aclskOptionType::PRELOAD_CODE,
         []() -> std::unique_ptr<OptOptionBase> {
           return std::make_unique<NumberOptOption>("preload_code", aclskOptionType::PRELOAD_CODE, 1, 0, 2);
         }},
        {aclskOptionType::SPLIT_MODE,
         []() -> std::unique_ptr<OptOptionBase> {
           return std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 4, 1, 4);
         }},
        {aclskOptionType::STREAM_FUSION,
         []() -> std::unique_ptr<OptOptionBase> {
           return std::make_unique<NumberOptOption>("stream_fusion", aclskOptionType::STREAM_FUSION, 1, 0, 1);
         }},
        {aclskOptionType::DCCI_DISABLE_ON_KERNEL,
         []() -> std::unique_ptr<OptOptionBase> {
           return std::make_unique<StringListOptOption>("dcci_disable_on_kernel",
                                                        aclskOptionType::DCCI_DISABLE_ON_KERNEL);
         }},
        {aclskOptionType::DEBUG_SYNC_ALL,
         []() -> std::unique_ptr<OptOptionBase> {
           return std::make_unique<NumberOptOption>("debug_sync_all", aclskOptionType::DEBUG_SYNC_ALL, 0, 0, 1);
         }},
        {aclskOptionType::KERNEL_MAP,
         []() -> std::unique_ptr<OptOptionBase> {
           return std::make_unique<OptOptionBase>("kernel_map", aclskOptionType::KERNEL_MAP);
         }},
        {aclskOptionType::AUTO_OP_PARALLEL,
         []() -> std::unique_ptr<OptOptionBase> {
           return std::make_unique<NumberOptOption>("auto_op_parallel", aclskOptionType::AUTO_OP_PARALLEL, 0, 0, 1);
         }},
        {aclskOptionType::DCCI_BEFORE_KERNEL_START,
         []() -> std::unique_ptr<OptOptionBase> {
           return std::make_unique<StringListOptOption>("dcci_before_kernel_start",
                                                        aclskOptionType::DCCI_BEFORE_KERNEL_START);
         }},
        {aclskOptionType::DEBUG_OP_EXEC_TRACE,
         []() -> std::unique_ptr<OptOptionBase> {
           return std::make_unique<NumberOptOption>("debug_op_exec_trace", aclskOptionType::DEBUG_OP_EXEC_TRACE, 0, 0,
                                                    1);
         }},
        {aclskOptionType::DEBUG_CROSS_CORE_SYNC_CHECK,
         []() -> std::unique_ptr<OptOptionBase> {
           return std::make_unique<NumberOptOption>("debug_cross_core_sync_check",
                                                    aclskOptionType::DEBUG_CROSS_CORE_SYNC_CHECK, 0, 0, 1);
         }},
        {aclskOptionType::OPT_EXTEND_OPTION,
         []() -> std::unique_ptr<OptOptionBase> {
           return std::make_unique<MapOptOption>("opt_extend_option", aclskOptionType::OPT_EXTEND_OPTION);
         }},
        {aclskOptionType::DEBUG_EXTEND_OPTION,
         []() -> std::unique_ptr<OptOptionBase> {
           return std::make_unique<MapOptOption>("debug_extend_option", aclskOptionType::DEBUG_EXTEND_OPTION);
         }},
        {aclskOptionType::DCCI_AFTER_KERNEL_END,
         []() -> std::unique_ptr<OptOptionBase> {
           return std::make_unique<StringListOptOption>("dcci_after_kernel_end",
                                                        aclskOptionType::DCCI_AFTER_KERNEL_END);
         }},
        {aclskOptionType::AGGRESSIVE_OPT_STRATEGIES,
         []() -> std::unique_ptr<OptOptionBase> {
           return std::make_unique<AggressiveOptStrategiesOption>("aggressive_opt_strategies",
                                                                  aclskOptionType::AGGRESSIVE_OPT_STRATEGIES);
         }},
        {aclskOptionType::UBUF_LOCK_IGNORE_KERNEL,
         []() -> std::unique_ptr<OptOptionBase> {
           return std::make_unique<StringListOptOption>("ubuf_lock_ignore_kernel",
                                                        aclskOptionType::UBUF_LOCK_IGNORE_KERNEL);
         }},
        {aclskOptionType::EARLY_START,
         []() -> std::unique_ptr<OptOptionBase> {
           return std::make_unique<NumberOptOption>(
               "early_start", aclskOptionType::EARLY_START, aclskEarlyStartValue::ACLSK_EARLY_START_DISABLED,
               aclskEarlyStartValue::ACLSK_EARLY_START_DISABLED, aclskEarlyStartValue::ACLSK_EARLY_START_ENABLED);
         }},
        {aclskOptionType::DEBUG_PER_OP_MAX_CORE_NUM,
         []() -> std::unique_ptr<OptOptionBase> {
           return std::make_unique<NumberOptOption>("debug_per_op_max_core_num",
                                                    aclskOptionType::DEBUG_PER_OP_MAX_CORE_NUM, 0, 0, 1);
         }},
    }};

const DefaultOptionFactoryEntry *FindDefaultOptionFactory(aclskOptionType optType) {
  for (const auto &entry : DEFAULT_OPTION_FACTORIES) {
    if (entry.optType == optType) {
      return &entry;
    }
  }
  return nullptr;
}

struct InnerOptionInfo {
  SkInnerOptionType optionType;
  std::unique_ptr<OptOptionBase> (*factory)();
};

const std::array<InnerOptionInfo, static_cast<size_t>(SkInnerOptionType::SK_INNER_OPTION_MAX)> DEFAULT_INNER_OPTIONS = {
    {
        {SkInnerOptionType::MIX_KERNEL_SPLIT,
         []() -> std::unique_ptr<OptOptionBase> {
           return std::make_unique<NumberOptOption>("mix_kernel_split", aclskOptionType::SK_OPTION_MAX, 0, 0, 1);
         }},
        {SkInnerOptionType::SIMT_OP_SUPPORT,
         []() -> std::unique_ptr<OptOptionBase> {
           return std::make_unique<NumberOptOption>("simt_op_support", aclskOptionType::SK_OPTION_MAX, 0, 0, 1);
         }},
    }};

uint32_t GetValidatedUintValue(const std::string &optionName, uint32_t value, uint32_t defaultValue, uint32_t minValue,
                               uint32_t maxValue) {
  if (value < minValue || value > maxValue) {
    SK_LOGW(
        "OptionName: %s, set value is invalid, value is %u, valid range is [%u, %u],"
        " the process will use default value: %u",
        optionName.c_str(), value, minValue, maxValue, defaultValue);
    return defaultValue;
  }
  SK_LOGI("OptionName: %s, set value: %u", optionName.c_str(), value);
  return value;
}

std::vector<std::string> SplitString(const std::string &input, char delimiter) {
  std::vector<std::string> tokens;
  size_t start = 0;
  while (start <= input.size()) {
    size_t end = input.find(delimiter, start);
    if (end == std::string::npos) {
      tokens.push_back(input.substr(start));
      break;
    }
    tokens.push_back(input.substr(start, end - start));
    start = end + 1;
  }
  return tokens;
}

bool IsValidExtendOptionToken(const std::string &token, bool allowSlash) {
  if (token.empty()) {
    return false;
  }
  for (char ch : token) {
    const unsigned char uchar = static_cast<unsigned char>(ch);
    if (std::isalnum(uchar) != 0 || ch == '_' || ch == '-' || ch == '.') {
      continue;
    }
    if (allowSlash && ch == '/') {
      continue;
    }
    return false;
  }
  return true;
}

bool ParseAndValidateExtendOptionValue(const char *rawValue, const std::string &optionName,
                                       std::unordered_map<std::string, std::vector<std::string>> &parsedValue) {
  if (rawValue == nullptr) {
    SK_LOGW("OptionName:%s, raw extend value is nullptr", optionName.c_str());
    return false;
  }

  const std::string input(rawValue);
  if (input.size() > kMaxExtendOptionLength) {
    SK_LOGW("OptionName:%s, raw extend value is too long: %zu", optionName.c_str(), input.size());
    return false;
  }

  const std::string trimmedInput = sk::TrimString(input);
  if (trimmedInput.empty()) {
    SK_LOGW("OptionName:%s, raw extend value is empty after trim", optionName.c_str());
    return false;
  }

  std::unordered_map<std::string, std::vector<std::string>> tmpResult;
  const std::vector<std::string> pairs = SplitString(trimmedInput, ':');
  for (const std::string &rawPair : pairs) {
    const std::string pair = sk::TrimString(rawPair);
    if (pair.empty()) {
      SK_LOGW("OptionName:%s, extend pair is empty", optionName.c_str());
      return false;
    }

    const size_t eqPos = pair.find('=');
    if (eqPos == std::string::npos || eqPos == 0 || eqPos == pair.size() - 1 ||
        pair.find('=', eqPos + 1) != std::string::npos) {
      SK_LOGW("OptionName:%s, extend pair format is invalid: %s", optionName.c_str(), pair.c_str());
      return false;
    }

    const std::string key = sk::TrimString(pair.substr(0, eqPos));
    if (!IsValidExtendOptionToken(key, false)) {
      SK_LOGW("OptionName:%s, extend key is invalid: %s", optionName.c_str(), key.c_str());
      return false;
    }
    if (tmpResult.find(key) != tmpResult.end()) {
      SK_LOGW("OptionName:%s, extend key is duplicated: %s", optionName.c_str(), key.c_str());
      return false;
    }

    const std::vector<std::string> rawValues = SplitString(pair.substr(eqPos + 1), ',');
    std::vector<std::string> valueList;
    valueList.reserve(rawValues.size());
    for (const std::string &rawSubValue : rawValues) {
      const std::string value = sk::TrimString(rawSubValue);
      if (!IsValidExtendOptionToken(value, true)) {
        SK_LOGW("OptionName:%s, extend value is invalid: %s", optionName.c_str(), value.c_str());
        return false;
      }
      valueList.push_back(value);
    }
    tmpResult.emplace(key, std::move(valueList));
  }

  parsedValue = std::move(tmpResult);
  return true;
}
}  // namespace

void NumberOptOption::SetValue(const uint32_t value) {
  if (value < optValueMin || value > optValueMax) {
    SK_LOGW(
        "OptionName: %s, set value is invalid, value is %u, valid range is [%u, %u],"
        " the process will use default value: %u",
        optionName.c_str(), value, optValueMin, optValueMax, optValue);
    return;
  }
  SK_LOGI("OptionName: %s, set value: %u", optionName.c_str(), value);
  optValue = value;
}

uint32_t NumberOptOption::GetIntValue() const {
  return optValue;
}

void StringOptOption::SetValue(const std::string &value) {
  if (value.empty()) {
    SK_LOGW("OptionName: %s, set value is invalid, value is empty", optionName.c_str());
    return;
  }
  SK_LOGI("OptionName: %s, set value: %s", optionName.c_str(), value.c_str());
  optValue = value;
}

std::string StringOptOption::GetStringValue() const {
  return optValue;
}

void StringListOptOption::SetValue(const std::vector<std::string> &value) {
  if (value.empty()) {
    SK_LOGW("OptionName: %s, set value is invalid, value is empty", optionName.c_str());
    return;
  }
  for (size_t i = 0; i < value.size(); i++) {
    SK_LOGI("OptionName:%s, value[%zu]: %s", optionName.c_str(), i, value[i].c_str());
  }
  optValue = value;
}

std::vector<std::string> StringListOptOption::GetStringListValue() const {
  return optValue;
}

void MapOptOption::SetValue(const std::unordered_map<std::string, std::vector<std::string>> &value) {
  if (value.empty()) {
    SK_LOGW("OptionName:%s, set value is invalid, value is empty", optionName.c_str());
    return;
  }
  for (const auto &pair : value) {
    const std::string &key = pair.first;
    const std::vector<std::string> &valList = pair.second;
    SK_LOGI("OptionName:%s, key: %s, values:", optionName.c_str(), key.c_str());
    for (const auto &val : valList) {
      SK_LOGI("  %s", val.c_str());
    }
  }
  optValue = value;
}

std::unordered_map<std::string, std::vector<std::string>> MapOptOption::GetMapValue() const {
  return optValue;
}

void SuperKernelOptionsManager::AddOption(std::unique_ptr<OptOptionBase> option) {
  if (option == nullptr) {
    SK_LOGI("option is nullptr");
    return;
  }
  auto iter = optionMap.find(option->GetType());
  if (iter != optionMap.end()) {
    SK_LOGW("OptionName:%s, already exists", option->GetName().c_str());
    return;
  }
  optionMap[option->GetType()] = std::move(option);
}

OptOptionBase *SuperKernelOptionsManager::GetOption(aclskOptionType optType) {
  auto iter = optionMap.find(optType);
  if (iter == optionMap.end()) {
    return nullptr;
  }
  return iter->second.get();
}

const OptOptionBase *SuperKernelOptionsManager::GetOption(aclskOptionType optType) const {
  auto iter = optionMap.find(optType);
  if (iter == optionMap.end()) {
    return nullptr;
  }
  return iter->second.get();
}

bool SuperKernelOptionsManager::IsInnerOptionEnabled(SkInnerOptionType optionType) const {
  const auto iter = innerOptionMap.find(optionType);
  return iter != innerOptionMap.end() && iter->second != nullptr && iter->second->GetIntValue() == 1;
}

bool SuperKernelOptionsManager::MatchKernelNameInList(const std::vector<std::string> &kernelList,
                                                      const std::string &kernelName) const {
  for (size_t i = 0; i < kernelList.size(); i++) {
    if (MatchRegex(kernelList[i], kernelName)) {
      SK_LOGI("kernel: %s matches pattern: %s", kernelName.c_str(), kernelList[i].c_str());
      return true;
    }
  }
  return false;
}

bool SuperKernelOptionsManager::JudgeUbufLockIgnoreKernel(const std::vector<std::string> &ignoredKernels,
                                                          const std::string &opName) const {
  for (const auto &ignoredKernel : ignoredKernels) {
    if (MatchRegex(ignoredKernel, opName)) {
      SK_LOGI("op: %s match ubuf lock ignore kernel option: %s, op will ignore mix kernel split", opName.c_str(),
              ignoredKernel.c_str());
      return true;
    }
  }
  return false;
}

bool SuperKernelOptionsManager::MatchRegex(const std::string &pattern, const std::string &opName) {
  return sk::MatchKernelNamePattern(pattern, opName);
}

bool SuperKernelOptionsManager::EnableDebug() const {
  auto iterSyncAll = optionMap.find(aclskOptionType::DEBUG_SYNC_ALL);
  const bool enableSyncAll =
      (iterSyncAll != optionMap.end() && iterSyncAll->second != nullptr && iterSyncAll->second->GetIntValue() == 1);
  if (enableSyncAll) {
    SK_LOGI("debug mode enabled");
    return true;
  }
  return false;
}

void SuperKernelOptionsManager::RegisterDefaultSkOptions() {
  for (int32_t i = 0; i < static_cast<int32_t>(aclskOptionType::SK_OPTION_MAX); ++i) {
    auto type = static_cast<aclskOptionType>(i);
    if (optionMap.find(type) != optionMap.end()) {
      continue;
    }
    const auto *entry = FindDefaultOptionFactory(type);
    if (entry == nullptr) {
      continue;
    }
    optionMap[type] = entry->factory();
  }
}

void SuperKernelOptionsManager::RegisterDefaultInnerOptions() {
  for (const auto &option : DEFAULT_INNER_OPTIONS) {
    if (innerOptionMap.find(option.optionType) == innerOptionMap.end()) {
      innerOptionMap[option.optionType] = option.factory();
    }
  }
}

void SuperKernelOptionsManager::ApplyArchSpecificOptions() {
  const SkKernelArch kernelArch = GetCurrentSkKernelArch();
  const auto enableOption = [this](SkInnerOptionType optionType) {
    const auto iter = innerOptionMap.find(optionType);
    if (iter != innerOptionMap.end() && iter->second != nullptr) {
      iter->second->SetValue(1);
    }
  };

  switch (kernelArch) {
    case SkKernelArch::DAV_3510:
      enableOption(SkInnerOptionType::MIX_KERNEL_SPLIT);
      enableOption(SkInnerOptionType::SIMT_OP_SUPPORT);
      break;
    case SkKernelArch::DAV_2201:
    default:
      break;
  }

  SK_LOGI("ApplyArchSpecificOptions: kernelArch=%s, mixKernelSplit=%d, simtOpSupport=%d", to_string(kernelArch),
          static_cast<int>(IsInnerOptionEnabled(SkInnerOptionType::MIX_KERNEL_SPLIT)),
          static_cast<int>(IsInnerOptionEnabled(SkInnerOptionType::SIMT_OP_SUPPORT)));
}

void SuperKernelOptionsManager::RegisterDefaultOptions() {
  RegisterDefaultSkOptions();
  RegisterDefaultInnerOptions();
  ApplyArchSpecificOptions();
}

void SuperKernelOptionsManager::SetOptOptionValue(const aclskOption *option) {
  if (option == nullptr) {
    SK_LOGW("sub aclskOption is nullptr");
    return;
  }
  auto type = option->optionType;
  auto iter = optionMap.find(type);
  if (iter == optionMap.end()) {
    const auto *entry = FindDefaultOptionFactory(type);
    if (entry != nullptr) {
      optionMap[type] = entry->factory();
      iter = optionMap.find(type);
    }
  }
  if (iter == optionMap.end()) {
    SK_LOGI("OptionType: %d is not supported now", static_cast<int>(type));
    return;
  }
  auto *subOption = iter->second.get();
  switch (type) {
    case aclskOptionType::PRELOAD_CODE:
      subOption->SetValue(option->preload.preloadMode);
      break;
    case aclskOptionType::SPLIT_MODE:
      subOption->SetValue(option->splitMode.splitCnt);
      break;
    case aclskOptionType::DCCI_DISABLE_ON_KERNEL: {
      std::vector<std::string> vecValue;
      const size_t kernelCnt = static_cast<size_t>(option->disableKernelDcci.kernelCnt);
      if (kernelCnt > 0 && option->disableKernelDcci.kernelNames == nullptr) {
        SK_LOGW("OptionName:%s, kernelNames is nullptr while kernelCnt is %zu", subOption->GetName().c_str(),
                kernelCnt);
        break;
      }
      vecValue.reserve(kernelCnt);
      for (size_t i = 0; i < kernelCnt; i++) {
        if (option->disableKernelDcci.kernelNames[i] == nullptr) {
          SK_LOGW("OptionName:%s, kernelNames[%zu] is nullptr, skip", subOption->GetName().c_str(), i);
          continue;
        }
        vecValue.push_back(std::string(option->disableKernelDcci.kernelNames[i]));
      }
      subOption->SetValue(vecValue);
      break;
    }
    case aclskOptionType::DCCI_BEFORE_KERNEL_START: {
      std::vector<std::string> vecValue;
      const size_t kernelCnt = static_cast<size_t>(option->dcciBeforeKernelStart.kernelCnt);
      if (kernelCnt > 0 && option->dcciBeforeKernelStart.kernelNames == nullptr) {
        SK_LOGW("OptionName:%s, kernelNames is nullptr while kernelCnt is %zu", subOption->GetName().c_str(),
                kernelCnt);
        break;
      }
      vecValue.reserve(kernelCnt);
      for (size_t i = 0; i < kernelCnt; i++) {
        if (option->dcciBeforeKernelStart.kernelNames[i] == nullptr) {
          SK_LOGW("OptionName:%s, kernelNames[%zu] is nullptr, skip", subOption->GetName().c_str(), i);
          continue;
        }
        vecValue.push_back(std::string(option->dcciBeforeKernelStart.kernelNames[i]));
      }
      subOption->SetValue(vecValue);
      break;
    }
    case aclskOptionType::DCCI_AFTER_KERNEL_END: {
      std::vector<std::string> vecValue;
      const size_t kernelCnt = static_cast<size_t>(option->dcciAfterKernelEnd.kernelCnt);
      if (kernelCnt > 0 && option->dcciAfterKernelEnd.kernelNames == nullptr) {
        SK_LOGW("OptionName:%s, kernelNames is nullptr while kernelCnt is %zu", subOption->GetName().c_str(),
                kernelCnt);
        break;
      }
      vecValue.reserve(kernelCnt);
      for (size_t i = 0; i < kernelCnt; i++) {
        if (option->dcciAfterKernelEnd.kernelNames[i] == nullptr) {
          SK_LOGW("OptionName:%s, kernelNames[%zu] is nullptr, skip", subOption->GetName().c_str(), i);
          continue;
        }
        vecValue.push_back(std::string(option->dcciAfterKernelEnd.kernelNames[i]));
      }
      subOption->SetValue(vecValue);
      break;
    }
    case aclskOptionType::AGGRESSIVE_OPT_STRATEGIES: {
      auto *aggressiveOpt = static_cast<AggressiveOptStrategiesOption *>(subOption);
      auto aggressiveOpts = option->aggressiveOpts;
      aggressiveOpts.eventBreakerBypass =
          GetValidatedUintValue("event_breaker_bypass", option->aggressiveOpts.eventBreakerBypass, 0, 0,
                                std::numeric_limits<decltype(option->aggressiveOpts.eventBreakerBypass)>::max());
      aggressiveOpts.valueBreakerBypass =
          GetValidatedUintValue("value_breaker_bypass", option->aggressiveOpts.valueBreakerBypass,
                                ACLSK_VALUE_BREAKER_BYPASS_NONE, ACLSK_VALUE_BREAKER_BYPASS_NONE,
                                std::numeric_limits<decltype(option->aggressiveOpts.valueBreakerBypass)>::max());
      aggressiveOpts.taskBreakerBypass =
          GetValidatedUintValue("task_breaker_bypass", option->aggressiveOpts.taskBreakerBypass, 0, 0, 1);
      SK_LOGI(
          "Aggressive opt strategies set: eventBreakerBypass=%u, valueBreakerBypass=%u, "
          "taskBreakerBypass=%u",
          aggressiveOpts.eventBreakerBypass, aggressiveOpts.valueBreakerBypass, aggressiveOpts.taskBreakerBypass);
      aggressiveOpt->SetValue(aggressiveOpts);
      break;
    }
    case aclskOptionType::UBUF_LOCK_IGNORE_KERNEL: {
      std::vector<std::string> vecValue;
      const size_t kernelCnt = static_cast<size_t>(option->ubufLockIgnoreKernel.ubufLockIgnoreKernelCnt);
      if (kernelCnt > 0 && option->ubufLockIgnoreKernel.ubufLockIgnoreKernel == nullptr) {
        SK_LOGW("OptionName:%s, ubufLockIgnoreKernel is nullptr while kernelCnt is %zu", subOption->GetName().c_str(),
                kernelCnt);
        break;
      }
      vecValue.reserve(kernelCnt);
      for (size_t i = 0; i < kernelCnt; i++) {
        if (option->ubufLockIgnoreKernel.ubufLockIgnoreKernel[i] == nullptr) {
          SK_LOGW("OptionName:%s, ubufLockIgnoreKernel[%zu] is nullptr, skip", subOption->GetName().c_str(), i);
          continue;
        }
        vecValue.push_back(std::string(option->ubufLockIgnoreKernel.ubufLockIgnoreKernel[i]));
      }
      subOption->SetValue(vecValue);
      break;
    }
    case aclskOptionType::DEBUG_SYNC_ALL:
      subOption->SetValue(option->debugSync.debugSyncAll);
      break;
    case aclskOptionType::OPT_EXTEND_OPTION: {
      std::unordered_map<std::string, std::vector<std::string>> parsedValue;
      if (ParseAndValidateExtendOptionValue(option->optExtend.value, subOption->GetName(), parsedValue)) {
        subOption->SetValue(parsedValue);
      }
      break;
    }
    case aclskOptionType::DEBUG_EXTEND_OPTION: {
      std::unordered_map<std::string, std::vector<std::string>> parsedValue;
      if (ParseAndValidateExtendOptionValue(option->debugExtend.value, subOption->GetName(), parsedValue)) {
        subOption->SetValue(parsedValue);
      }
      break;
    }
    case aclskOptionType::STREAM_FUSION:
      subOption->SetValue(option->streamFusion.streamFusion);
      break;
    case aclskOptionType::AUTO_OP_PARALLEL:
      subOption->SetValue(option->autoOpParallel.enableAutoOpParallel);
      SK_LOGI("Auto op parallel option set: enable=%u", option->autoOpParallel.enableAutoOpParallel);
      break;
    case aclskOptionType::DEBUG_CROSS_CORE_SYNC_CHECK:
      subOption->SetValue(option->debugCrossCoreSyncCheck.enableCrossCoreSyncCheck);
      SK_LOGI("Debug cross-core sync check option set: enable=%u",
              option->debugCrossCoreSyncCheck.enableCrossCoreSyncCheck);
      break;
    case aclskOptionType::DEBUG_OP_EXEC_TRACE:
      subOption->SetValue(option->debugOpExecTrace.enableOpExecTrace);
      SK_LOGI("Debug op exec trace option set: enable=%u", option->debugOpExecTrace.enableOpExecTrace);
      break;
    case aclskOptionType::EARLY_START:
      subOption->SetValue(option->earlyStart.enableEarlyStart);
      SK_LOGI("Early start option set: enable=%u", option->earlyStart.enableEarlyStart);
      break;
    case aclskOptionType::DEBUG_PER_OP_MAX_CORE_NUM:
      subOption->SetValue(option->debugPerOpMaxCoreNum.enableDebugPerOpMaxCoreNum);
      SK_LOGI("Debug per op max core num option set: enable=%u",
              option->debugPerOpMaxCoreNum.enableDebugPerOpMaxCoreNum);
      if (option->debugPerOpMaxCoreNum.enableDebugPerOpMaxCoreNum == 1) {
        auto crossCoreSyncOpt = GetOption(aclskOptionType::DEBUG_CROSS_CORE_SYNC_CHECK);
        if (crossCoreSyncOpt == nullptr || crossCoreSyncOpt->GetIntValue() != 1) {
          aclskOption crossCoreSyncOption;
          crossCoreSyncOption.optionType = aclskOptionType::DEBUG_CROSS_CORE_SYNC_CHECK;
          crossCoreSyncOption.debugCrossCoreSyncCheck.enableCrossCoreSyncCheck = 1;
          SetOptOptionValue(&crossCoreSyncOption);
          SK_LOGI("[DEBUG_PER_OP_MAX_CORE_NUM] auto-enabled DEBUG_CROSS_CORE_SYNC_CHECK");
        }
      }
      break;
    default:
      SK_LOGI("OptionType: %d is not supported now", static_cast<int>(type));
      break;
  }
}

void SuperKernelOptionsManager::ParseOptions(const aclskOptions *options) {
  RegisterDefaultOptions();
  if (options == nullptr) {
    SK_LOGI("aclskOption is nullptr");
    return;
  }
  SK_LOGI("Options nums: %d", static_cast<int>(options->numOptions));
  if (options->numOptions > 0 && options->options == nullptr) {
    SK_LOGW("aclskOptions options is nullptr while numOptions is %zu", options->numOptions);
    return;
  }
  std::unordered_set<aclskOptionType> parsedTypes;
  for (size_t i = 0; i < static_cast<size_t>(options->numOptions); i++) {
    const aclskOptionType optionType = options->options[i].optionType;
    if (parsedTypes.find(optionType) != parsedTypes.end()) {
      const auto iter = optionMap.find(optionType);
      const char *optionName =
          (iter != optionMap.end() && iter->second != nullptr) ? iter->second->GetName().c_str() : "unsupported";
      SK_LOGW("OptionName %s already parsed", optionName);
      continue;
    }
    parsedTypes.insert(optionType);
    SetOptOptionValue(&options->options[i]);
  }
}

nlohmann::ordered_json SuperKernelOptionsManager::ToJson() const {
  nlohmann::ordered_json optionsJson = nlohmann::ordered_json::object();

  for (int32_t i = 0; i < static_cast<int32_t>(aclskOptionType::SK_OPTION_MAX); ++i) {
    auto type = static_cast<aclskOptionType>(i);
    const auto iter = optionMap.find(type);
    if (iter == optionMap.end()) {
      continue;
    }

    const OptOptionBase *opt = iter->second.get();
    if (opt == nullptr) {
      continue;
    }

    nlohmann::ordered_json optJson;
    optJson["name"] = opt->GetName();
    optJson["type"] = static_cast<int>(type);

    switch (type) {
      case aclskOptionType::PRELOAD_CODE:
      case aclskOptionType::SPLIT_MODE:
      case aclskOptionType::DEBUG_SYNC_ALL:
      case aclskOptionType::STREAM_FUSION:
      case aclskOptionType::AUTO_OP_PARALLEL:
      case aclskOptionType::DEBUG_CROSS_CORE_SYNC_CHECK:
      case aclskOptionType::DEBUG_OP_EXEC_TRACE:
      case aclskOptionType::EARLY_START:
      case aclskOptionType::DEBUG_PER_OP_MAX_CORE_NUM:
        optJson["value"] = opt->GetIntValue();
        break;

      case aclskOptionType::DCCI_DISABLE_ON_KERNEL:
      case aclskOptionType::DCCI_BEFORE_KERNEL_START:
      case aclskOptionType::DCCI_AFTER_KERNEL_END:
      case aclskOptionType::UBUF_LOCK_IGNORE_KERNEL:
        optJson["value"] = opt->GetStringListValue();
        break;

      case aclskOptionType::OPT_EXTEND_OPTION:
      case aclskOptionType::DEBUG_EXTEND_OPTION:
        optJson["value"] = opt->GetMapValue();
        break;

      case aclskOptionType::AGGRESSIVE_OPT_STRATEGIES: {
        const auto *aggressiveOpt = static_cast<const AggressiveOptStrategiesOption *>(opt);
        const auto &value = aggressiveOpt->GetValue();
        optJson["value"] = {{"eventBreakerBypass", value.eventBreakerBypass},
                            {"valueBreakerBypass", value.valueBreakerBypass},
                            {"taskBreakerBypass", value.taskBreakerBypass}};
        break;
      }

      default:
        optJson["value"] = nullptr;
        break;
    }

    optionsJson[opt->GetName()] = optJson;
  }

  nlohmann::ordered_json innerOptionsJson = nlohmann::ordered_json::object();
  for (const auto &option : DEFAULT_INNER_OPTIONS) {
    const auto iter = innerOptionMap.find(option.optionType);
    if (iter == innerOptionMap.end() || iter->second == nullptr) {
      continue;
    }

    const OptOptionBase *innerOption = iter->second.get();
    nlohmann::ordered_json optionJson;
    optionJson["name"] = innerOption->GetName();
    optionJson["type"] = static_cast<int>(option.optionType);
    optionJson["value"] = IsInnerOptionEnabled(option.optionType);
    innerOptionsJson[innerOption->GetName()] = optionJson;
  }
  optionsJson["inner_options"] = innerOptionsJson;

  return optionsJson;
}
