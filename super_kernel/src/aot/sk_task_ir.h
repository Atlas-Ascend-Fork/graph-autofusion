/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __SK_TASK_IR_H__
#define __SK_TASK_IR_H__

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "sk_common.h"

enum class SkQueueType : uint8_t {
  AIC,
  AIV,
  MIX_1_1,
  MIX_1_2,
  UNKNOWN,
};

inline const char *to_string(SkQueueType type) {
  switch (type) {
    case SkQueueType::AIC:
      return "AIC";
    case SkQueueType::AIV:
      return "AIV";
    case SkQueueType::MIX_1_1:
      return "MIX_1_1";
    case SkQueueType::MIX_1_2:
      return "MIX_1_2";
    default:
      return "UNKNOWN";
  }
}

enum class SkTaskKind : uint8_t {
  KERNEL,
  NOTIFY,
  WAIT,
  RESET,
};

constexpr size_t kSkCoreTypeCount = 2;  // Array indices: 0 = AIC, 1 = AIV.

struct SkResolvedFunctionDesc {
  std::array<uint64_t, kSkCoreTypeCount> functionAddress{};
  std::array<uint64_t, kSkCoreTypeCount> functionOffset{};
  std::array<uint64_t, kSkCoreTypeCount> prefetchCount{};
};

enum class SkTaskAttributeId : uint32_t {
  KERNEL_SCHE_MODE = 0x1001,
  KERNEL_BLOCK_DIM_SCALE_UP = 0x1002,
  KERNEL_MIX_SPLIT = 0x1003,
  EARLY_START_SET = 0x1101,
  EARLY_START_WAIT = 0x1102,
  SIMT_ENABLED = 0x1201,
  SIMT_DYN_UBUF_SIZE = 0x1202,
  SIMT_ALLOC_UBUF_SIZE = 0x1203,
  KERNEL_EFFECTIVE_DEBUG_OPTIONS = 0x1301,
  CONFIG_DEBUG_ENABLED = 0x2001,
  CONFIG_DEBUG_SYNC_ALL = 0x2002,
  CONFIG_STREAM_FUSION_ENABLED = 0x2003,
  CONFIG_MIX_KERNEL_SPLIT_ENABLED = 0x2004,
  CONFIG_EARLY_START_ENABLED = 0x2005,
  CONFIG_SIMT_ENABLED = 0x2006,
  CONFIG_PROFILING_ENABLED = 0x2007,
  CONFIG_OP_TRACE_ENABLED = 0x2008,
  CONFIG_CROSS_CORE_SYNC_CHECK_ENABLED = 0x2009,
};

enum class SkTaskAttributeType : uint16_t {
  BOOL = 1,
  UINT64 = 2,
};

enum class SkTaskAttributeFlag : uint16_t {
  OPTIONAL = 0,
  REQUIRED = 1,
};

struct SkTaskAttribute {
  SkTaskAttributeId id;
  SkTaskAttributeType type;
  SkTaskAttributeFlag flags = SkTaskAttributeFlag::OPTIONAL;
  uint64_t value = 0;
};

enum class SkTaskAttributeScope : uint8_t {
  KERNEL,
  CONFIG,
};

struct SkTaskAttributeSchema {
  SkTaskAttributeScope scope;
  SkTaskAttributeType type;
};

inline std::optional<SkTaskAttributeSchema> GetTaskAttributeSchema(SkTaskAttributeId id) {
  switch (id) {
    case SkTaskAttributeId::KERNEL_SCHE_MODE:
    case SkTaskAttributeId::KERNEL_BLOCK_DIM_SCALE_UP:
    case SkTaskAttributeId::KERNEL_MIX_SPLIT:
    case SkTaskAttributeId::EARLY_START_SET:
    case SkTaskAttributeId::EARLY_START_WAIT:
    case SkTaskAttributeId::SIMT_ENABLED:
      return SkTaskAttributeSchema{SkTaskAttributeScope::KERNEL, SkTaskAttributeType::BOOL};
    case SkTaskAttributeId::SIMT_DYN_UBUF_SIZE:
    case SkTaskAttributeId::SIMT_ALLOC_UBUF_SIZE:
    case SkTaskAttributeId::KERNEL_EFFECTIVE_DEBUG_OPTIONS:
      return SkTaskAttributeSchema{SkTaskAttributeScope::KERNEL, SkTaskAttributeType::UINT64};
    case SkTaskAttributeId::CONFIG_DEBUG_ENABLED:
    case SkTaskAttributeId::CONFIG_DEBUG_SYNC_ALL:
    case SkTaskAttributeId::CONFIG_STREAM_FUSION_ENABLED:
    case SkTaskAttributeId::CONFIG_MIX_KERNEL_SPLIT_ENABLED:
    case SkTaskAttributeId::CONFIG_EARLY_START_ENABLED:
    case SkTaskAttributeId::CONFIG_SIMT_ENABLED:
    case SkTaskAttributeId::CONFIG_PROFILING_ENABLED:
    case SkTaskAttributeId::CONFIG_OP_TRACE_ENABLED:
    case SkTaskAttributeId::CONFIG_CROSS_CORE_SYNC_CHECK_ENABLED:
      return SkTaskAttributeSchema{SkTaskAttributeScope::CONFIG, SkTaskAttributeType::BOOL};
    default:
      return std::nullopt;
  }
}

inline void SetTaskAttribute(std::vector<SkTaskAttribute> &attributes, SkTaskAttributeId id, SkTaskAttributeType type,
                             uint64_t value, SkTaskAttributeFlag flags = SkTaskAttributeFlag::OPTIONAL) {
  for (auto &attribute : attributes) {
    if (attribute.id == id) {
      attribute = {id, type, flags, value};
      return;
    }
  }
  attributes.push_back({id, type, flags, value});
}

inline void SetBoolAttribute(std::vector<SkTaskAttribute> &attributes, SkTaskAttributeId id, bool value,
                             SkTaskAttributeFlag flags = SkTaskAttributeFlag::OPTIONAL) {
  SetTaskAttribute(attributes, id, SkTaskAttributeType::BOOL, value ? 1U : 0U, flags);
}

inline void SetUint64Attribute(std::vector<SkTaskAttribute> &attributes, SkTaskAttributeId id, uint64_t value,
                               SkTaskAttributeFlag flags = SkTaskAttributeFlag::OPTIONAL) {
  SetTaskAttribute(attributes, id, SkTaskAttributeType::UINT64, value, flags);
}

inline const SkTaskAttribute *FindTaskAttribute(const std::vector<SkTaskAttribute> &attributes, SkTaskAttributeId id) {
  for (const auto &attribute : attributes) {
    if (attribute.id == id) {
      return &attribute;
    }
  }
  return nullptr;
}

inline bool GetBoolAttribute(const std::vector<SkTaskAttribute> &attributes, SkTaskAttributeId id, bool defaultValue) {
  const SkTaskAttribute *attribute = FindTaskAttribute(attributes, id);
  return attribute == nullptr || attribute->type != SkTaskAttributeType::BOOL || attribute->value > 1U
             ? defaultValue
             : attribute->value != 0;
}

inline std::optional<uint64_t> GetUint64Attribute(const std::vector<SkTaskAttribute> &attributes,
                                                  SkTaskAttributeId id) {
  const SkTaskAttribute *attribute = FindTaskAttribute(attributes, id);
  if (attribute == nullptr || attribute->type != SkTaskAttributeType::UINT64) {
    return std::nullopt;
  }
  return attribute->value;
}

inline bool ValidateTaskAttributes(const std::vector<SkTaskAttribute> &attributes, SkTaskAttributeScope expectedScope) {
  for (size_t i = 0; i < attributes.size(); ++i) {
    const auto &attribute = attributes[i];
    if (attribute.flags != SkTaskAttributeFlag::OPTIONAL && attribute.flags != SkTaskAttributeFlag::REQUIRED) {
      return false;
    }
    if (attribute.type != SkTaskAttributeType::BOOL && attribute.type != SkTaskAttributeType::UINT64) {
      return false;
    }
    for (size_t j = 0; j < i; ++j) {
      if (attributes[j].id == attribute.id) {
        return false;
      }
    }
    const auto schema = GetTaskAttributeSchema(attribute.id);
    if (!schema.has_value()) {
      if (attribute.flags == SkTaskAttributeFlag::REQUIRED) {
        return false;
      }
      continue;
    }
    if (schema->scope != expectedScope || schema->type != attribute.type ||
        (attribute.type == SkTaskAttributeType::BOOL && attribute.value > 1U)) {
      return false;
    }
  }
  return true;
}

inline bool ValidateKernelAttributes(const std::vector<SkTaskAttribute> &attributes) {
  return ValidateTaskAttributes(attributes, SkTaskAttributeScope::KERNEL);
}

inline bool ValidateTaskConfigAttributes(const std::vector<SkTaskAttribute> &attributes) {
  return ValidateTaskAttributes(attributes, SkTaskAttributeScope::CONFIG);
}

struct SkKernelDesc {
  SkKernelType kernelType = SkKernelType::DEFAULT;
  uint32_t originalNumBlocks = 0;
  uint32_t cubeNum = 0;
  uint32_t vectorNum = 0;
  std::vector<SkTaskAttribute> attributes;
  std::string functionName;
  const void *deviceArgs = nullptr;
  uint32_t argsSize = 0;
  aclrtBinHandle binaryHandle = nullptr;
  aclrtFuncHandle originalFunction = nullptr;
  std::vector<SkResolvedFunctionDesc> functions;
};

struct SkEventDesc {
  void *address = nullptr;
  uint64_t value = std::numeric_limits<uint64_t>::max();
  uint32_t waitFlag = std::numeric_limits<uint32_t>::max();
};

struct SkTaskDesc {
  uint64_t nodeId = std::numeric_limits<uint64_t>::max();
  SkTaskKind kind = SkTaskKind::KERNEL;
  SkQueueType queueType = SkQueueType::UNKNOWN;
  std::string debugLabel;
  std::vector<uint64_t> successorIds;
  std::variant<SkKernelDesc, SkEventDesc> payload = SkKernelDesc{};
};

struct SkCoreSpec {
  SkKernelType type = SkKernelType::DEFAULT;
  uint32_t numBlocks = 0;
  uint32_t cubeNum = 0;
  uint32_t vectorNum = 0;

  bool IsValid() const {
    return type != SkKernelType::DEFAULT && numBlocks != 0;
  }
};

constexpr uint32_t kSkDefaultSplitBinCount = 4;

struct SkTaskConfig {
  uint32_t splitBinCount = kSkDefaultSplitBinCount;
  uint32_t counterSlotCount = 0;
  SkKernelArch kernelArch = SkKernelArch::UNKNOWN;
  std::vector<SkTaskAttribute> attributes;
};

struct SkTaskIR {
  std::string functionName;
  uint16_t scopeId = 0;
  SkCoreSpec coreSpec;
  SkTaskConfig config;
  std::vector<SkTaskDesc> tasks;
};

#endif  // __SK_TASK_IR_H__
