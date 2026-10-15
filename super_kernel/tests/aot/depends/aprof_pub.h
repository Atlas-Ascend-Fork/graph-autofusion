/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file aprof_pub.h
 * @brief Stub header for profiling types and functions
 */

#ifndef APROF_PUB_H
#define APROF_PUB_H

#include <cstdint>
#include <cstddef>
#include <cstring>

// Profiling task types
constexpr uint32_t MSPROF_GE_TASK_TYPE_AICORE = 1;
constexpr uint32_t MSPROF_GE_TASK_TYPE_AIC = 2;
constexpr uint32_t MSPROF_GE_TASK_TYPE_AIV = 3;
constexpr uint32_t MSPROF_GE_TASK_TYPE_MIX_AIC = 4;
constexpr uint32_t MSPROF_GE_TASK_TYPE_MIX_AIV = 5;

// Tensor data shape length
constexpr int MSPROF_GE_TENSOR_DATA_SHAPE_LEN = 8;

/**
 * @brief Tensor data structure for profiling
 */
struct MsrofTensorData {
  uint32_t tensorType;  // 0: input, 1: output
  uint32_t format;      // data format
  uint32_t dataType;    // data type
  uint32_t reserve;
  int64_t shape[MSPROF_GE_TENSOR_DATA_SHAPE_LEN];  // tensor shape
};

/**
 * @brief Convert string to profiling ID (stub implementation)
 * @param str Input string
 * @param len String length
 * @return Hash ID of the string
 */
inline uint64_t MsprofStr2Id(const char *str, size_t len) {
  if (str == nullptr || len == 0) {
    return 0;
  }
  // Simple hash function for stub
  uint64_t hash = 5381;
  for (size_t i = 0; i < len; ++i) {
    hash = ((hash << 5) + hash) + static_cast<uint64_t>(static_cast<unsigned char>(str[i]));
  }
  return hash;
}

/**
 * @brief Convert profiling ID to string (stub implementation)
 * @param id Profiling ID
 * @return Static string representation (stub returns "unknown")
 */
inline char *MsprofId2Str(size_t id) {
  static char unknownStr[] = "unknown";
  return unknownStr;
}

/**
 * @brief Get profiling path (stub implementation)
 * @return Static string of profiling path
 */
char *MsprofGetPath();

/**
 * @brief Register profiling callback (stub implementation)
 * @param type Profiling callback type
 * @param callback Callback function pointer
 * @return 0 on success
 */
int32_t MsprofRegisterCallback(uint32_t type, int32_t (*callback)(uint32_t, void *, uint32_t));

#endif  // APROF_PUB_H
