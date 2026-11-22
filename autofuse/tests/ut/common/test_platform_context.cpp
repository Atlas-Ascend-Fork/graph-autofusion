/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#define private public
#include "common/platform_context.h"
#undef private
#include "tests/depends/runtime/src/runtime_stub.h"

namespace {
class RuntimeSocSpecFailStub : public ge::RuntimeStub {
 public:
  rtError_t rtGetSocSpec(const char *label, const char *key, char *val, const uint32_t maxLen) override {
    (void)label;
    (void)key;
    (void)val;
    (void)maxLen;
    return -1;
  }
};
}  // namespace

class PlatformContextTest : public testing::Test {
 protected:
  void TearDown() override {
    ge::PlatformContext::GetInstance().Reset();
    ge::RuntimeStub::Reset();
  }
};

TEST_F(PlatformContextTest, TryGetInitializedPlatformInfoReturnsFalseAfterReset) {
  ge::PlatformContext::GetInstance().Reset();

  ge::PlatformInfo info;
  EXPECT_FALSE(ge::PlatformContext::GetInstance().TryGetInitializedPlatformInfo(info));
  EXPECT_EQ(info.ub_size, 0);
}

TEST_F(PlatformContextTest, TryGetInitializedPlatformInfoReturnsInjectedInfo) {
  ge::PlatformContext::GetInstance().Reset();
  ge::PlatformInfo injected;
  injected.soc_ver = "mock_soc";
  injected.aiv_num = 20;
  injected.ub_size = 262144;
  ge::PlatformContext::GetInstance().SetPlatformInfo(injected);

  ge::PlatformInfo actual;
  EXPECT_TRUE(ge::PlatformContext::GetInstance().TryGetInitializedPlatformInfo(actual));
  EXPECT_EQ(actual.soc_ver, "mock_soc");
  EXPECT_EQ(actual.aiv_num, 20);
  EXPECT_EQ(actual.ub_size, 262144);
}

TEST_F(PlatformContextTest, SetPlatformInfoWithEmptySocVerStoresUbSizeOverride) {
  ge::PlatformContext::GetInstance().Reset();
  ge::PlatformInfo injected;
  injected.ub_size = 262144;
  ge::PlatformContext::GetInstance().SetPlatformInfo(injected);

  ge::PlatformInfo actual;
  EXPECT_FALSE(ge::PlatformContext::GetInstance().TryGetInitializedPlatformInfo(actual));

  int64_t ub_size = 0;
  EXPECT_TRUE(ge::PlatformContext::GetInstance().TryGetUbSizeOverride(ub_size));
  EXPECT_EQ(ub_size, 262144);
}

TEST_F(PlatformContextTest, SetUbSizeOverrideDoesNotInitializePlatformInfo) {
  ge::PlatformContext::GetInstance().Reset();
  ge::PlatformContext::GetInstance().SetUbSizeOverride(2048);

  ge::PlatformInfo actual;
  int64_t ub_size = 0;
  EXPECT_FALSE(ge::PlatformContext::GetInstance().TryGetInitializedPlatformInfo(actual));
  EXPECT_TRUE(ge::PlatformContext::GetInstance().TryGetUbSizeOverride(ub_size));
  EXPECT_EQ(ub_size, 2048);
}

TEST_F(PlatformContextTest, ResetClearsUbSizeOverride) {
  ge::PlatformContext::GetInstance().Reset();
  ge::PlatformContext::GetInstance().SetUbSizeOverride(262144);
  ge::PlatformContext::GetInstance().Reset();

  int64_t ub_size = 0;
  EXPECT_FALSE(ge::PlatformContext::GetInstance().TryGetUbSizeOverride(ub_size));
  EXPECT_EQ(ub_size, 0);
}

TEST_F(PlatformContextTest, GetPlatformInfoInitializesUbSizeFromRuntime) {
  ge::RuntimeStub::Reset();

  ge::PlatformInfo actual;
  EXPECT_EQ(ge::PlatformContext::GetInstance().GetPlatformInfo(actual), af::SUCCESS);
  EXPECT_EQ(actual.soc_ver, "2201");
  EXPECT_EQ(actual.aiv_num, 48);
  EXPECT_EQ(actual.ub_size, 245760);
}

TEST_F(PlatformContextTest, SetPlatformTargetStoresAndGetDualFields) {
  ge::PlatformContext::GetInstance().Reset();
  ge::PlatformInfo platform_info;
  platform_info.soc_ver = "5102";
  platform_info.aiv_num = 48;
  platform_info.ub_size = 245760;
  EXPECT_EQ(ge::PlatformContext::GetInstance().SetPlatformTarget(platform_info, "MC62CM12AA"), af::SUCCESS);

  std::string npu_arch;
  std::string soc_version;
  EXPECT_EQ(ge::PlatformContext::GetInstance().GetPlatformTarget(npu_arch, soc_version), af::SUCCESS);
  EXPECT_EQ(npu_arch, "5102");
  EXPECT_EQ(soc_version, "MC62CM12AA");
}

TEST_F(PlatformContextTest, SetPlatformTargetUpdatesUbSizeOverride) {
  ge::PlatformContext::GetInstance().Reset();
  ge::PlatformInfo old_info;
  old_info.ub_size = 4096;
  ge::PlatformContext::GetInstance().SetPlatformInfo(old_info);

  int64_t ub_size = 0;
  ASSERT_TRUE(ge::PlatformContext::GetInstance().TryGetUbSizeOverride(ub_size));
  ASSERT_EQ(ub_size, 4096);

  ge::PlatformInfo new_info;
  new_info.soc_ver = "5102";
  new_info.ub_size = 262144;
  ASSERT_EQ(ge::PlatformContext::GetInstance().SetPlatformTarget(new_info, "MC62CM12AA"), af::SUCCESS);
  EXPECT_TRUE(ge::PlatformContext::GetInstance().TryGetUbSizeOverride(ub_size));
  EXPECT_EQ(ub_size, 262144);
}

TEST_F(PlatformContextTest, InitializeKeepsExplicitFullSocVersion) {
  ge::PlatformContext::GetInstance().Reset();
  ge::PlatformInfo platform_info;
  platform_info.soc_ver = "5102";
  platform_info.aiv_num = 48;
  platform_info.ub_size = 262144;
  ASSERT_EQ(ge::PlatformContext::GetInstance().SetPlatformTarget(platform_info, "MC62CM12AA"), af::SUCCESS);
  ASSERT_EQ(ge::PlatformContext::GetInstance().Initialize(), af::SUCCESS);

  std::string npu_arch;
  std::string soc_version;
  EXPECT_EQ(ge::PlatformContext::GetInstance().GetPlatformTarget(npu_arch, soc_version), af::SUCCESS);
  EXPECT_EQ(npu_arch, "5102");
  EXPECT_EQ(soc_version, "MC62CM12AA");
}

TEST_F(PlatformContextTest, OldSetPlatformClearsFullSocVersion) {
  ge::PlatformContext::GetInstance().Reset();
  ge::PlatformInfo platform_info;
  platform_info.soc_ver = "5102";
  EXPECT_EQ(ge::PlatformContext::GetInstance().SetPlatformTarget(platform_info, "MC62CM12AA"), af::SUCCESS);

  ge::PlatformContext::GetInstance().SetPlatform("3510");
  std::string npu_arch;
  std::string soc_version;
  EXPECT_EQ(ge::PlatformContext::GetInstance().GetPlatformTarget(npu_arch, soc_version), af::SUCCESS);
  EXPECT_EQ(npu_arch, "3510");
  EXPECT_EQ(soc_version, "");
}

TEST_F(PlatformContextTest, OldSetPlatformInfoWithEmptySocVerClearsFullSocVersion) {
  ge::PlatformContext::GetInstance().Reset();
  ge::PlatformInfo platform_info;
  platform_info.soc_ver = "5102";
  platform_info.ub_size = 245760;
  EXPECT_EQ(ge::PlatformContext::GetInstance().SetPlatformTarget(platform_info, "MC62CM12AA"), af::SUCCESS);

  ge::PlatformInfo empty_arch;
  empty_arch.ub_size = 245760;
  ge::PlatformContext::GetInstance().SetPlatformInfo(empty_arch);
  std::string npu_arch;
  std::string soc_version;
  EXPECT_EQ(ge::PlatformContext::GetInstance().GetPlatformTarget(npu_arch, soc_version), af::SUCCESS);
  EXPECT_EQ(npu_arch, "5102");
  EXPECT_EQ(soc_version, "");
}

TEST_F(PlatformContextTest, ResetClearsFullSocVersion) {
  ge::PlatformContext::GetInstance().Reset();
  ge::PlatformInfo platform_info;
  platform_info.soc_ver = "5102";
  EXPECT_EQ(ge::PlatformContext::GetInstance().SetPlatformTarget(platform_info, "MC62CM12AA"), af::SUCCESS);
  ge::PlatformContext::GetInstance().Reset();

  std::string npu_arch;
  std::string soc_version;
  EXPECT_EQ(ge::PlatformContext::GetInstance().GetPlatformTarget(npu_arch, soc_version), af::SUCCESS);
  EXPECT_EQ(npu_arch, "2201");
  EXPECT_EQ(soc_version, "");
}

TEST_F(PlatformContextTest, SetPlatformTargetRejectsFullSocVersionWithoutNpuArch) {
  ge::PlatformContext::GetInstance().Reset();
  ge::PlatformInfo platform_info;
  platform_info.soc_ver = "5102";
  EXPECT_EQ(ge::PlatformContext::GetInstance().SetPlatformTarget(platform_info, "MC62CM12AA"), af::SUCCESS);

  ge::PlatformInfo empty_arch;
  EXPECT_NE(ge::PlatformContext::GetInstance().SetPlatformTarget(empty_arch, "MC62CM12AA"), af::SUCCESS);
  std::string npu_arch;
  std::string soc_version;
  EXPECT_EQ(ge::PlatformContext::GetInstance().GetPlatformTarget(npu_arch, soc_version), af::SUCCESS);
  EXPECT_EQ(npu_arch, "5102");
  EXPECT_EQ(soc_version, "MC62CM12AA");
}

TEST_F(PlatformContextTest, InitializeFailureDoesNotExposeFullSocVersion) {
  ge::PlatformContext::GetInstance().Reset();
  ge::RuntimeStub::SetInstance(std::make_shared<RuntimeSocSpecFailStub>());

  std::string npu_arch = "stale_arch";
  std::string soc_version = "stale_soc";
  EXPECT_NE(ge::PlatformContext::GetInstance().GetPlatformTarget(npu_arch, soc_version), af::SUCCESS);
  EXPECT_EQ(npu_arch, "");
  EXPECT_EQ(soc_version, "");
}
