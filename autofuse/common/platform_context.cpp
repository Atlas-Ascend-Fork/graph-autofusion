/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/platform_context.h"
#include "rt_external_base.h"
#include "common/checker.h"

namespace {
const uint32_t kSocStrMaxLen = 32U;
const uint32_t kMaxValueLen = 16U;
const char *kSocInfo = "SoCInfo";
const char *kAICoreSpec = "AICoreSpec";
const char *kVectorCoreCnt = "vector_core_cnt";
const char *kUbSize = "ub_size";
const char *kL2Size = "l2_size";
std::string g_full_soc_version;

bool ParseInt64(const char *value, const char *key_name, int64_t &result) {
  try {
    std::string str_value(value);
    result = static_cast<int64_t>(std::stoll(str_value));
  } catch (...) {
    GE_LOGE("ParseInt64 failed: invalid value '%s' for %s", value, key_name);
    return false;
  }
  return true;
}
}  // namespace
namespace ge {
PlatformContext &PlatformContext::GetInstance() {
  static PlatformContext instance;
  return instance;
}
std::mutex PlatformContext::mutex_;

void PlatformContext::SetPlatform(const std::string &platform_name) {
  std::lock_guard<std::mutex> lg(mutex_);
  g_full_soc_version.clear();
  if (!platform_name.empty()) {
    platform_info_.soc_ver = platform_name;
    initialized_ = true;
    GELOGI("Set platform: soc_ver=%s", platform_info_.soc_ver.c_str());
  }
}

void PlatformContext::SetPlatformInfo(const PlatformInfo &platform_info) {
  std::lock_guard<std::mutex> lg(mutex_);
  g_full_soc_version.clear();
  if (platform_info.ub_size > 0) {
    ub_size_override_ = platform_info.ub_size;
    has_ub_size_override_ = true;
  }
  if (!platform_info.soc_ver.empty()) {
    platform_info_ = platform_info;
    initialized_ = true;
    GELOGI("Set platform info: soc_ver=%s, aiv_num=%lld, ub_size=%lld, l2_size=%lld", platform_info_.soc_ver.c_str(),
           platform_info_.aiv_num, platform_info_.ub_size, platform_info_.l2_size);
  }
}

af::Status PlatformContext::SetPlatformTarget(const PlatformInfo &platform_info, const std::string &soc_version) {
  if (platform_info.soc_ver.empty() && !soc_version.empty()) {
    return af::FAILED;
  }

  std::lock_guard<std::mutex> lg(mutex_);
  g_full_soc_version.clear();
  if (platform_info.ub_size > 0) {
    ub_size_override_ = platform_info.ub_size;
    has_ub_size_override_ = true;
  }
  if (!platform_info.soc_ver.empty()) {
    platform_info_ = platform_info;
    g_full_soc_version = soc_version;
    initialized_ = true;
    return af::SUCCESS;
  }

  initialized_ = false;
  return af::SUCCESS;
}

af::Status PlatformContext::GetPlatformTarget(std::string &npu_arch, std::string &soc_version) {
  npu_arch.clear();
  soc_version.clear();
  if (!initialized_) {
    g_full_soc_version.clear();
    GE_ASSERT_SUCCESS(Initialize(), "Failed to init platform info.");
  }
  std::lock_guard<std::mutex> lg(mutex_);
  npu_arch = platform_info_.soc_ver;
  soc_version = g_full_soc_version;
  return af::SUCCESS;
}

void PlatformContext::Reset() {
  std::lock_guard<std::mutex> lg(mutex_);
  initialized_ = false;
  platform_info_ = PlatformInfo{};
  g_full_soc_version.clear();
  ub_size_override_ = 0;
  has_ub_size_override_ = false;
}

void PlatformContext::SetUbSizeOverride(int64_t ub_size) {
  std::lock_guard<std::mutex> lg(mutex_);
  ub_size_override_ = ub_size;
  has_ub_size_override_ = true;
  GELOGI("Set UB size override: ub_size=%lld", ub_size_override_);
}

af::Status PlatformContext::GetCurrentPlatformString(std::string &platform_name) {
  if (!initialized_) {
    GE_ASSERT_SUCCESS(Initialize(), "Failed to init platform info with name %s.", platform_name.c_str());
  }
  std::lock_guard<std::mutex> lg(mutex_);
  platform_name = platform_info_.soc_ver;
  return af::SUCCESS;
}

af::Status PlatformContext::Initialize() {
  std::lock_guard<std::mutex> lg(mutex_);
  if (initialized_) {
    return af::SUCCESS;
  }
  g_full_soc_version.clear();
  char soc_version[kSocStrMaxLen] = {};
  auto res = rtGetSocSpec("version", "NpuArch", soc_version, kSocStrMaxLen);
  GE_ASSERT_TRUE(res == RT_ERROR_NONE, "Failed to get npu arch str.");
  platform_info_.soc_ver = std::string(soc_version);

  GE_ASSERT_SUCCESS(InitPlatformInfo(), "Failed to init platform info.");
  return af::SUCCESS;
}

af::Status PlatformContext::InitPlatformInfo() {
  char aiv_num_str[kMaxValueLen] = {};
  auto res = rtGetSocSpec(kSocInfo, kVectorCoreCnt, aiv_num_str, kMaxValueLen);
  GE_ASSERT_TRUE(res == RT_ERROR_NONE, "Failed to get vector_core_cnt.");
  GE_ASSERT_TRUE(ParseInt64(aiv_num_str, "vector_core_cnt", platform_info_.aiv_num),
                 "Failed to parse vector_core_cnt.");

  char ub_size_str[kMaxValueLen] = {};
  res = rtGetSocSpec(kAICoreSpec, kUbSize, ub_size_str, kMaxValueLen);
  GE_ASSERT_TRUE(res == RT_ERROR_NONE, "Failed to get ub_size.");
  GE_ASSERT_TRUE(ParseInt64(ub_size_str, "ub_size", platform_info_.ub_size), "Failed to parse ub_size.");

  char l2_size_str[kMaxValueLen] = {};
  res = rtGetSocSpec(kSocInfo, kL2Size, l2_size_str, kMaxValueLen);
  GE_ASSERT_TRUE(res == RT_ERROR_NONE, "Failed to get l2_size.");
  GE_ASSERT_TRUE(ParseInt64(l2_size_str, "l2_size", platform_info_.l2_size), "Failed to parse l2_size.");
  initialized_ = true;
  GELOGI("Platform info: soc_ver=%s, aiv_num=%lld, ub_size=%lld, l2_size=%lld", platform_info_.soc_ver.c_str(),
         platform_info_.aiv_num, platform_info_.ub_size, platform_info_.l2_size);

  return af::SUCCESS;
}

af::Status PlatformContext::GetPlatformInfo(PlatformInfo &platform_info) {
  if (!initialized_) {
    GE_ASSERT_SUCCESS(Initialize(), "Failed to init platform info.");
  }
  std::lock_guard<std::mutex> lg(mutex_);
  platform_info = platform_info_;
  GELOGD("GetPlatformInfo: soc_ver=%s, aiv_num=%lld, ub_size=%lld", platform_info_.soc_ver.c_str(),
         platform_info_.aiv_num, platform_info_.ub_size);
  return af::SUCCESS;
}

bool PlatformContext::TryGetInitializedPlatformInfo(PlatformInfo &platform_info) {
  std::lock_guard<std::mutex> lg(mutex_);
  if (!initialized_) {
    platform_info = PlatformInfo{};
    return false;
  }
  platform_info = platform_info_;
  return true;
}

bool PlatformContext::TryGetUbSizeOverride(int64_t &ub_size) const {
  std::lock_guard<std::mutex> lg(mutex_);
  if (!has_ub_size_override_) {
    ub_size = 0;
    return false;
  }
  ub_size = ub_size_override_;
  return true;
}
}  // namespace ge
