/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MICRO_DTYPE_UTILS_H
#define MICRO_DTYPE_UTILS_H

#include <string>
#include <cstdint>

namespace codegen {

constexpr int DTYPE_SIZE_1BYTE = 1;
constexpr int DTYPE_SIZE_2BYTE = 2;
constexpr int DTYPE_SIZE_4BYTE = 4;
constexpr int DTYPE_SIZE_8BYTE = 8;

inline uint32_t GetDtypeSizeByName(const std::string &dtype_name) {
  if (dtype_name == "bool") {
    return DTYPE_SIZE_1BYTE;
  }
  if (dtype_name == "half") {
    return DTYPE_SIZE_2BYTE;
  }
  if (dtype_name == "float") {
    return DTYPE_SIZE_4BYTE;
  }
  if (dtype_name == "double") {
    return DTYPE_SIZE_8BYTE;
  }
  for (size_t i = 0; i < dtype_name.size(); ++i) {
    if (std::isdigit(dtype_name[i])) {
      uint32_t bits = 0;
      while (i < dtype_name.size() && std::isdigit(dtype_name[i])) {
        bits = bits * 10 + (dtype_name[i] - '0');
        ++i;
      }
      return bits / 8;
    }
  }
  return DTYPE_SIZE_4BYTE;
}

inline std::string GetUintDtypeNameBySize(uint32_t dtype_size) {
  if (dtype_size == DTYPE_SIZE_1BYTE) {
    return "uint8_t";
  } else if (dtype_size == DTYPE_SIZE_2BYTE) {
    return "uint16_t";
  } else if (dtype_size == DTYPE_SIZE_4BYTE) {
    return "uint32_t";
  } else if (dtype_size == DTYPE_SIZE_8BYTE) {
    return "uint64_t";
  }
  return "uint32_t";
}

}  // namespace codegen

#endif  // MICRO_DTYPE_UTILS_H
