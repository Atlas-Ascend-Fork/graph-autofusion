/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pattern_common.h"
#include <cctype>
#include <vector>
#include "sk_log.h"

namespace sk {
std::string TrimString(const std::string &pattern) {
  size_t start = 0;
  while (start < pattern.size() && std::isspace(static_cast<unsigned char>(pattern[start])) != 0) {
    ++start;
  }
  size_t end = pattern.size();
  while (end > start && std::isspace(static_cast<unsigned char>(pattern[end - 1])) != 0) {
    --end;
  }
  return pattern.substr(start, end - start);
}

bool IsValidRegexPattern(const std::string &pattern) {
  if (pattern.empty()) {
    return false;
  }
  for (char ch : pattern) {
    const unsigned char uchar = static_cast<unsigned char>(ch);
    if (std::isalnum(uchar) != 0 || ch == '_' || ch == '-' || ch == '.' || ch == '*') {
      continue;
    }
    return false;
  }
  return true;
}

bool MatchKernelNamePattern(const std::string &pattern, const std::string &opName) {
  const std::string trimmedPattern = TrimString(pattern);
  if (trimmedPattern.empty()) {
    SK_LOGE("pattern is empty after trim");
    return false;
  }
  if (!IsValidRegexPattern(trimmedPattern)) {
    SK_LOGE("pattern contains invalid characters, only alphanumeric, '_', '-', '.', '*' are allowed: %s",
            trimmedPattern.c_str());
    return false;
  }
  if (trimmedPattern[0] == '*') {
    SK_LOGE("invalid pattern starts with '*': %s", trimmedPattern.c_str());
    return false;
  }
  size_t m = opName.size();
  size_t n = trimmedPattern.size();

  auto matches = [&](size_t i, size_t j) {
    if (i == 0 || j == 0) {
      return false;
    }
    if (trimmedPattern[j - 1] == '.') {
      return true;
    }
    return opName[i - 1] == trimmedPattern[j - 1];
  };

  std::vector<std::vector<size_t>> matchFlag(m + 1, std::vector<size_t>(n + 1));
  matchFlag[0][0] = true;
  for (size_t i = 0; i <= m; ++i) {
    for (size_t j = 1; j <= n; ++j) {
      if (trimmedPattern[j - 1] == '*') {
        if (j >= 2) {
          matchFlag[i][j] |= matchFlag[i][j - 2];
          if (matches(i, j - 1)) {
            matchFlag[i][j] |= matchFlag[i - 1][j];
          }
        }
      } else {
        if (matches(i, j)) {
          matchFlag[i][j] |= matchFlag[i - 1][j - 1];
        }
      }
    }
  }
  return matchFlag[m][n];
}
}  // namespace sk
