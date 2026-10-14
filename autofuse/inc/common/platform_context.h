/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCGEN_DEV_BASE_COMMON_PLATFORM_CONTEXT_H
#define ASCGEN_DEV_BASE_COMMON_PLATFORM_CONTEXT_H

#include <string>
#include <mutex>
#include "ge_common_af/ge_api_error_codes_af.h"

namespace ge {
struct PlatformInfo {
  std::string soc_ver;
  int64_t aiv_num = 0;
  int64_t ub_size = 0;
  int64_t l2_size = 0;
};

class PlatformContext {
 public:
  static PlatformContext &GetInstance();

  PlatformContext(const PlatformContext &) = delete;
  PlatformContext &operator=(const PlatformContext &) = delete;
  PlatformContext(PlatformContext &&) = delete;
  PlatformContext &operator=(PlatformContext &&) = delete;

  // 外部设置 platform
  void SetPlatform(const std::string &platform_name);

  void SetPlatformInfo(const PlatformInfo &platform_info);

  af::Status SetPlatformTarget(const PlatformInfo &platform_info, const std::string &soc_version);

  af::Status GetPlatformTarget(std::string &npu_arch, std::string &soc_version);

  void SetUbSizeOverride(int64_t ub_size);

  // 获取当前 platform 字符串
  af::Status GetCurrentPlatformString(std::string &platform_name);

  void Reset();

  af::Status GetPlatformInfo(PlatformInfo &platform_info);

  bool TryGetInitializedPlatformInfo(PlatformInfo &platform_info);

  bool TryGetUbSizeOverride(int64_t &ub_size) const;

 private:
  af::Status Initialize();
  af::Status InitPlatformInfo();
  PlatformContext() = default;
  PlatformInfo platform_info_;
  int64_t ub_size_override_ = 0;
  bool has_ub_size_override_ = false;
  bool initialized_ = false;
  static std::mutex mutex_;
};
}  // namespace ge
#endif  // ASCGEN_DEV_BASE_COMMON_PLATFORM_CONTEXT_H
