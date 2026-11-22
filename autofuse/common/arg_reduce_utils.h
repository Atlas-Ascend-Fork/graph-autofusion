/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_COMMON_ARG_REDUCE_UTILS_H_
#define AUTOFUSE_COMMON_ARG_REDUCE_UTILS_H_

#include <string>

namespace ascgen_utils {
inline bool IsArgMinType(const std::string &type) {
  return type == "Argmin" || type == "ArgminWithValue" || type == "ArgMinMultiRPhase1" ||
         type == "ArgMinMultiRPhase2" || type == "ArgMinMultiRPhase2WithValue";
}

inline bool IsArgMaxType(const std::string &type) {
  return type == "Argmax" || type == "ArgmaxWithValue" || type == "ArgMaxMultiRPhase1" ||
         type == "ArgMaxMultiRPhase2" || type == "ArgMaxMultiRPhase2WithValue";
}

inline bool IsArgReduceType(const std::string &type) {
  return IsArgMaxType(type) || IsArgMinType(type);
}

inline bool IsArgReduceIndexFrontend(const std::string &type) {
  return type == "Argmax" || type == "Argmin";
}

inline bool IsArgReduceWithValueFrontend(const std::string &type) {
  return type == "ArgmaxWithValue" || type == "ArgminWithValue";
}

inline bool IsArgReduceFrontend(const std::string &type) {
  return IsArgReduceIndexFrontend(type) || IsArgReduceWithValueFrontend(type);
}

inline bool IsArgReducePhase1(const std::string &type) {
  return type == "ArgMaxMultiRPhase1" || type == "ArgMinMultiRPhase1";
}

inline bool IsArgReducePhase2(const std::string &type) {
  return type == "ArgMaxMultiRPhase2" || type == "ArgMaxMultiRPhase2WithValue" || type == "ArgMinMultiRPhase2" ||
         type == "ArgMinMultiRPhase2WithValue";
}

inline bool HasArgReduceValueOutput(const std::string &type) {
  return IsArgReduceWithValueFrontend(type) || IsArgReducePhase1(type) || type == "ArgMaxMultiRPhase2WithValue" ||
         type == "ArgMinMultiRPhase2WithValue";
}

inline const char *ArgReducePhase1Type(const std::string &type) {
  return IsArgMinType(type) ? "ArgMinMultiRPhase1" : (IsArgMaxType(type) ? "ArgMaxMultiRPhase1" : "");
}
}  // namespace ascgen_utils

#endif  // AUTOFUSE_COMMON_ARG_REDUCE_UTILS_H_
