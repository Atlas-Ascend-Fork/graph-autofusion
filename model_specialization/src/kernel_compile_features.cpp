/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_compile_features.h"
#include "common/pattern_common.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <utility>
#include "sk_log.h"

namespace model_spec {
namespace detail {
namespace {
bool PrepareKernelNamePattern(const std::string &pattern, std::string &normalized) {
  // 对 kernelName 做 trim，然后校验匹配模式是否合法，并拒绝以 * 开头的非法模式
  normalized = sk::TrimString(pattern);
  return sk::IsValidRegexPattern(normalized) && normalized[0] != '*';
}

bool Invalid(const std::string &field, const char *reason) {
  SK_DLOGE("Static compile options: %s: %s", field.c_str(), reason);
  return false;
}

template <typename T>
bool CheckArray(const T *data, uint64_t count, const std::string &field) {
  // 零数量允许空指针；这里只校验输入约定和容量上限，实际可读长度由调用方保证。
  if (count != 0 && data == nullptr) {
    return Invalid(field, "nonzero count requires a non-null array");
  }
  if (count > static_cast<uint64_t>(std::numeric_limits<ptrdiff_t>::max()) / sizeof(T) ||
      count > std::vector<T>().max_size()) {
    return Invalid(field, "count exceeds addressable array capacity");
  }
  return true;
}

bool CopyStrings(const char *const *data, uint64_t count, const std::string &field, std::vector<std::string> &out) {
  // 追加深拷贝而非覆盖，供同类型名单合并使用；编译参数按完整字符串保存，不拆分。
  if (!CheckArray(data, count, field)) {
    return false;
  }
  if (count > out.max_size() - out.size()) {
    return Invalid(field, "combined count exceeds string vector capacity");
  }
  for (uint64_t i = 0; i < count; ++i) {
    if (data[i] == nullptr || data[i][0] == '\0') {
      return Invalid(field + "[" + std::to_string(i) + "]", "string must not be null or empty");
    }
    out.emplace_back(data[i]);
  }
  return true;
}

// 把一个 SK feature 的 kernelNames 转成多条 KernelOptionRule
bool AppendKernelFeatureRules(const aclmdlRISpecSKFeature &feature, const std::string &field,
                              const std::vector<std::string> &arguments, std::vector<KernelOptionRule> &output) {
  std::vector<std::string> patterns;
  if (!CopyStrings(feature.kernelNames, feature.kernelNameCount, field + ".kernelNames", patterns)) {
    return false;
  }

  for (size_t i = 0; i < patterns.size(); ++i) {
    KernelOptionRule rule;
    if (!PrepareKernelNamePattern(patterns[i], rule.kernelPattern)) {
      return Invalid(field + ".kernelNames[" + std::to_string(i) + "]", "invalid SuperKernel name pattern");
    }
    rule.compileArguments = arguments;
    output.push_back(std::move(rule));
  }
  return true;
}

bool ConvertDebugPerOpMaxCoreNum(const aclmdlRISpecSKFeature &feature, const std::string &field,
                                 std::vector<KernelOptionRule> &output) {
  return AppendKernelFeatureRules(feature, field, {"-D__ASCENDC_SUPER_KERNEL_DEBUG__"}, output);
}

bool ConvertDcciDisableOnKernel(const aclmdlRISpecSKFeature &feature, const std::string &field,
                                std::vector<KernelOptionRule> &output) {
  // 对齐静态 sub-combine：非空名单启用所有 kernel 的 Get/SetValue DCCI，
  // 仅命中名单的 kernel 设置 SK_BIND disableDcci；两者不可共用同一个宏。
  if (feature.kernelNameCount != 0) {
    output.push_back({".*", {"-D__ASCENDC_SUPER_KERNEL_ENABLE_GM_GET_SET_VALUE_DCCI__"}});
  }
  return AppendKernelFeatureRules(feature, field, {"-D__ASCENDC_SUPER_KERNEL_DISABLE_DCCI__"}, output);
}

struct SkConverter {
  const char *feature;
  bool (*convert)(const aclmdlRISpecSKFeature &, const std::string &, std::vector<KernelOptionRule> &);
};
// 集中登记当前需要处理的SK feature转换入口；convert为nullptr或表外feature暂不处理。
constexpr SkConverter SK_OPTIONS[] = {
    {"PRELOAD_CODE", nullptr},
    {"SPLIT_MODE", nullptr},
    {"STREAM_FUSION", nullptr},
    {"DCCI_DISABLE_ON_KERNEL", ConvertDcciDisableOnKernel},
    {"DEBUG_SYNC_ALL", nullptr},
    {"KERNEL_MAP", nullptr},
    {"AUTO_OP_PARALLEL", nullptr},
    {"DCCI_BEFORE_KERNEL_START", nullptr},
    {"DEBUG_OP_EXEC_TRACE", nullptr},
    {"DEBUG_CROSS_CORE_SYNC_CHECK", nullptr},
    {"OPT_EXTEND_OPTION", nullptr},
    {"DEBUG_EXTEND_OPTION", nullptr},
    {"DCCI_AFTER_KERNEL_END", nullptr},
    {"AGGRESSIVE_OPT_STRATEGIES", nullptr},
    {"UBUF_LOCK_IGNORE_KERNEL", nullptr},
    {"EARLY_START", nullptr},
    {"DEBUG_PER_OP_MAX_CORE_NUM", ConvertDebugPerOpMaxCoreNum},
};

constexpr bool StringEqual(const char *lhs, const char *rhs) {
  while (*lhs != '\0' && *rhs != '\0') {
    if (*lhs != *rhs) {
      return false;
    }
    ++lhs;
    ++rhs;
  }
  return *lhs == *rhs;
}

constexpr bool UniqueSkOptions() {
  for (size_t i = 0; i < std::size(SK_OPTIONS); ++i) {
    for (size_t j = i + 1; j < std::size(SK_OPTIONS); ++j) {
      if (StringEqual(SK_OPTIONS[i].feature, SK_OPTIONS[j].feature)) {
        return false;
      }
    }
  }
  return true;
}
// 编译期间拒绝重复feature名，避免查询时只找到第一项而掩盖配置表错误。
static_assert(UniqueSkOptions(), "Each SK feature name must have exactly one descriptor");
}  // namespace

bool CompileOptionFeature::Init(const aclmdlRISpecOptions *options) {
  // 本对象由临时Manager初始化；任一步失败时由Manager丢弃，不覆盖旧的有效配置。
  rules_.clear();
  return options == nullptr || (InitBasic(options) && InitSk(options));
}

bool CompileOptionFeature::InitBasic(const aclmdlRISpecOptions *options) {
  const auto count = options->specCompileOptionCount;
  const auto *basicOptions = options->specCompileOptions;
  auto &rules = rules_[CompileOptionType::BASIC];
  if (!CheckArray(basicOptions, count, "specCompileOptions") || count > rules.max_size()) {
    return Invalid("specCompileOptions", "invalid rule array");
  }
  for (uint64_t i = 0; i < count; ++i) {
    const auto field = "specCompileOptions[" + std::to_string(i) + "]";
    const auto &option = basicOptions[i];
    KernelOptionRule rule;
    // nullptr表示全局规则，与局部规则共用列表并保留输入顺序；空串仍由模式校验拒绝。
    if (!PrepareKernelNamePattern(option.kernelName == nullptr ? ".*" : option.kernelName, rule.kernelPattern)) {
      return Invalid(field + ".kernelName", "invalid SuperKernel name pattern");
    }
    if (!CopyStrings(option.compileOptions, option.compileOptionCount, field + ".compileOptions",
                     rule.compileArguments)) {
      return false;
    }
    rules.push_back(std::move(rule));
  }
  return true;
}

bool CompileOptionFeature::InitSk(const aclmdlRISpecOptions *options) {
  if (!CheckArray(options->specSKFeatures, options->specSKFeatureCount, "specSKFeatures")) {
    return false;
  }
  for (size_t i = 0; i < options->specSKFeatureCount; ++i) {
    const auto &feature = options->specSKFeatures[i];
    const auto field = "specSKFeatures[" + std::to_string(i) + "]";
    if (feature.feature == nullptr || feature.feature[0] == '\0') {
      return Invalid(field + ".feature", "feature must not be null or empty");
    }
    const auto converter = std::find_if(std::begin(SK_OPTIONS), std::end(SK_OPTIONS), [&](const SkConverter &item) {
      return std::string_view(item.feature) == feature.feature;
    });
    if (converter == std::end(SK_OPTIONS) || converter->convert == nullptr) {
      continue;
    }
    if (!converter->convert(feature, field, rules_[CompileOptionType::SK])) {
      return Invalid(field + "." + converter->feature, "invalid SK feature configuration");
    }
  }
  return true;
}

std::vector<std::string> CompileOptionFeature::Get(CompileOptionType type, std::string_view entry,
                                                   bool enableSk) const {
  if (type != CompileOptionType::BASIC && type != CompileOptionType::SK) {
    SK_DLOGE("Unknown static compile option channel: %u", static_cast<unsigned>(type));
    return {};
  }
  std::vector<std::string> result;
  if (type == CompileOptionType::SK && enableSk) {
    result.emplace_back("--enable-super-kernel");
  }
  const auto rules = rules_.find(type);
  if (rules == rules_.end()) {
    return result;
  }
  const std::string kernelName(entry);
  for (const auto &rule : rules->second) {
    if (sk::MatchKernelNamePattern(rule.kernelPattern, kernelName)) {
      for (const auto &argument : rule.compileArguments) {
        if (type == CompileOptionType::BASIC) {
          result.push_back(argument);
          continue;
        }
        // SK规则可能因多个模式同时命中而重复；只保留首次出现的完整参数。
        if (std::find(result.begin(), result.end(), argument) == result.end()) {
          result.push_back(argument);
        }
      }
    }
  }
  return result;
}

}  // namespace detail
}  // namespace model_spec
