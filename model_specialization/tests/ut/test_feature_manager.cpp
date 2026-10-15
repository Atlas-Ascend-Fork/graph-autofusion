/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "feature_manager.h"

#include <future>
#include <limits>
#include <vector>
#include "gtest/gtest.h"

namespace model_spec {
namespace {
using Arguments = std::vector<std::string>;

class FeatureManagerTest : public testing::Test {
 protected:
  FeatureManager manager;
  aclmdlRISpecOptions options{};
};

TEST_F(FeatureManagerTest, DefaultsAndInvalidQueries) {
  EXPECT_EQ(manager.Get<Feature::COMPILE_JOBS>(), 0U);
  EXPECT_FALSE(manager.Get<Feature::ENABLE_SK>());
  EXPECT_TRUE(manager.Get<Feature::BASIC_COMPILE_OPTIONS>("add").empty());
  EXPECT_TRUE(manager.Get<Feature::SK_COMPILE_OPTIONS>("add").empty());

  ASSERT_TRUE(manager.Init(nullptr));
  EXPECT_EQ(manager.Get<Feature::COMPILE_JOBS>(), 1U);
  EXPECT_FALSE(manager.Get<Feature::ENABLE_SK>());
  EXPECT_TRUE(manager.Get<Feature::BASIC_COMPILE_OPTIONS>("add").empty());
  EXPECT_TRUE(manager.Get<Feature::SK_COMPILE_OPTIONS>("add").empty());
  EXPECT_TRUE(manager.Get<Feature::BASIC_COMPILE_OPTIONS>("").empty());
  EXPECT_TRUE(manager.Get<Feature::SK_COMPILE_OPTIONS>("").empty());
}

TEST_F(FeatureManagerTest, JobsAndEnableSkAreModelLevelOptions) {
  options.jobs = 0;
  options.enableSK = false;
  ASSERT_TRUE(manager.Init(&options));
  EXPECT_EQ(manager.Get<Feature::COMPILE_JOBS>(), 1U);
  EXPECT_FALSE(manager.Get<Feature::ENABLE_SK>());

  options.jobs = 8;
  options.enableSK = true;
  ASSERT_TRUE(manager.Init(&options));
  EXPECT_EQ(manager.Get<Feature::COMPILE_JOBS>(), 8U);
  EXPECT_TRUE(manager.Get<Feature::ENABLE_SK>());
  EXPECT_EQ(manager.Get<Feature::SK_COMPILE_OPTIONS>("add"), (Arguments{"--enable-super-kernel"}));
}

TEST_F(FeatureManagerTest, BasicCompileOptionsUseGlobalAndPatternRules) {
  const char *globalArgs[] = {"-DGLOBAL=1"};
  const char *addArgs[] = {"-O2", "-DADD=1"};
  const char *matmulArgs[] = {"-O3", "-DMATMUL=1"};
  aclmdlRISpecCompileOption basic[] = {
      {nullptr, 1, globalArgs},
      {"addcustom.*", 2, addArgs},
      {"matmul_.*", 2, matmulArgs},
  };
  options.specCompileOptionCount = 3;
  options.specCompileOptions = basic;
  ASSERT_TRUE(manager.Init(&options));

  EXPECT_EQ(manager.Get<Feature::BASIC_COMPILE_OPTIONS>("addcustom_tiling"),
            (Arguments{"-DGLOBAL=1", "-O2", "-DADD=1"}));
  EXPECT_EQ(manager.Get<Feature::BASIC_COMPILE_OPTIONS>("matmul_aic_only"),
            (Arguments{"-DGLOBAL=1", "-O3", "-DMATMUL=1"}));
  EXPECT_EQ(manager.Get<Feature::BASIC_COMPILE_OPTIONS>("AddCustom_tiling"), (Arguments{"-DGLOBAL=1"}));
}

TEST_F(FeatureManagerTest, BasicCompileOptionsAreDeepCopied) {
  {
    std::string pattern = "add.*";
    std::string argument = "-O2";
    const char *args[] = {argument.c_str()};
    aclmdlRISpecCompileOption basic{pattern.c_str(), 1, args};
    options.specCompileOptionCount = 1;
    options.specCompileOptions = &basic;
    ASSERT_TRUE(manager.Init(&options));
  }
  auto result = manager.Get<Feature::BASIC_COMPILE_OPTIONS>("addcustom");
  ASSERT_EQ(result, (Arguments{"-O2"}));
  result[0] = "-O0";
  EXPECT_EQ(manager.Get<Feature::BASIC_COMPILE_OPTIONS>("addcustom"), (Arguments{"-O2"}));
}

TEST_F(FeatureManagerTest, InvalidBasicInputsFail) {
  aclmdlRISpecCompileOption basic{};
  options.specCompileOptionCount = 1;
  options.specCompileOptions = &basic;

  for (const char *pattern : {"", " \t ", "[a-z]", "^add", "add$", "add+", "add?", "(add)", "add|mul", "*add"}) {
    basic.kernelName = pattern;
    EXPECT_FALSE(manager.Init(&options)) << pattern;
  }

  basic.kernelName = "add.*";
  basic.compileOptionCount = 1;
  basic.compileOptions = nullptr;
  EXPECT_FALSE(manager.Init(&options));

  const char *badArgs[] = {nullptr};
  basic.compileOptions = badArgs;
  EXPECT_FALSE(manager.Init(&options));
  badArgs[0] = "";
  EXPECT_FALSE(manager.Init(&options));

  basic.compileOptionCount = 0;
  basic.compileOptions = nullptr;
  EXPECT_TRUE(manager.Init(&options));
}

TEST_F(FeatureManagerTest, EnableSkControlsSkOptionEmission) {
  const char *debugNames[] = {".*"};
  aclmdlRISpecSKFeature debugFeature{"DEBUG_PER_OP_MAX_CORE_NUM", 1, debugNames};
  options.specSKFeatureCount = 1;
  options.specSKFeatures = &debugFeature;

  options.enableSK = false;
  ASSERT_TRUE(manager.Init(&options));
  EXPECT_TRUE(manager.Get<Feature::SK_COMPILE_OPTIONS>("add").empty());

  options.enableSK = true;
  ASSERT_TRUE(manager.Init(&options));
  EXPECT_EQ(manager.Get<Feature::SK_COMPILE_OPTIONS>("add"),
            (Arguments{"--enable-super-kernel", "-D__ASCENDC_SUPER_KERNEL_DEBUG__"}));
}

TEST_F(FeatureManagerTest, SkFeaturesGenerateKernelSpecificOptions) {
  const char *debugNames[] = {"add.*", "matmul_.*"};
  const char *dcciNames[] = {"add.*"};
  aclmdlRISpecSKFeature skFeatures[] = {
      {"DEBUG_PER_OP_MAX_CORE_NUM", 2, debugNames},
      {"DCCI_DISABLE_ON_KERNEL", 1, dcciNames},
  };
  options.enableSK = true;
  options.specSKFeatureCount = 2;
  options.specSKFeatures = skFeatures;
  ASSERT_TRUE(manager.Init(&options));

  EXPECT_EQ(manager.Get<Feature::SK_COMPILE_OPTIONS>("addcustom"),
            (Arguments{"--enable-super-kernel", "-D__ASCENDC_SUPER_KERNEL_DEBUG__",
                       "-D__ASCENDC_SUPER_KERNEL_ENABLE_GM_GET_SET_VALUE_DCCI__",
                       "-D__ASCENDC_SUPER_KERNEL_DISABLE_DCCI__"}));
  EXPECT_EQ(manager.Get<Feature::SK_COMPILE_OPTIONS>("matmul_123"),
            (Arguments{"--enable-super-kernel", "-D__ASCENDC_SUPER_KERNEL_DEBUG__",
                       "-D__ASCENDC_SUPER_KERNEL_ENABLE_GM_GET_SET_VALUE_DCCI__"}));
  EXPECT_EQ(manager.Get<Feature::SK_COMPILE_OPTIONS>("relu"),
            (Arguments{"--enable-super-kernel", "-D__ASCENDC_SUPER_KERNEL_ENABLE_GM_GET_SET_VALUE_DCCI__"}));
}

TEST_F(FeatureManagerTest, EmptySkFeatureKernelListGeneratesNoFeatureRules) {
  aclmdlRISpecSKFeature debugFeature{"DEBUG_PER_OP_MAX_CORE_NUM", 0, nullptr};
  options.enableSK = true;
  options.specSKFeatureCount = 1;
  options.specSKFeatures = &debugFeature;
  ASSERT_TRUE(manager.Init(&options));
  EXPECT_EQ(manager.Get<Feature::SK_COMPILE_OPTIONS>("add"), (Arguments{"--enable-super-kernel"}));
}

TEST_F(FeatureManagerTest, IgnoredKnownSkFeaturesDoNotReadKernelNames) {
  const char *badNames[] = {nullptr};
  aclmdlRISpecSKFeature features[] = {
      {"PRELOAD_CODE", 1, badNames},
      {"SPLIT_MODE", 1, badNames},
      {"STREAM_FUSION", 1, badNames},
      {"DEBUG_SYNC_ALL", 1, badNames},
      {"KERNEL_MAP", 1, badNames},
      {"AUTO_OP_PARALLEL", 1, badNames},
      {"DCCI_BEFORE_KERNEL_START", 1, badNames},
      {"DEBUG_OP_EXEC_TRACE", 1, badNames},
      {"DEBUG_CROSS_CORE_SYNC_CHECK", 1, badNames},
      {"OPT_EXTEND_OPTION", 1, badNames},
      {"DEBUG_EXTEND_OPTION", 1, badNames},
      {"DCCI_AFTER_KERNEL_END", 1, badNames},
      {"AGGRESSIVE_OPT_STRATEGIES", 1, badNames},
      {"UBUF_LOCK_IGNORE_KERNEL", 1, badNames},
      {"EARLY_START", 1, badNames},
  };
  options.enableSK = true;
  options.specSKFeatureCount = sizeof(features) / sizeof(features[0]);
  options.specSKFeatures = features;

  ASSERT_TRUE(manager.Init(&options));
  EXPECT_EQ(manager.Get<Feature::SK_COMPILE_OPTIONS>("add"), (Arguments{"--enable-super-kernel"}));
}

TEST_F(FeatureManagerTest, InvalidSkFeatureInputsFail) {
  aclmdlRISpecSKFeature feature{};
  options.enableSK = true;
  options.specSKFeatureCount = 1;
  options.specSKFeatures = &feature;

  EXPECT_FALSE(manager.Init(&options));
  feature.feature = "";
  EXPECT_FALSE(manager.Init(&options));
  feature.feature = "unknown-feature";
  EXPECT_TRUE(manager.Init(&options));
  EXPECT_EQ(manager.Get<Feature::SK_COMPILE_OPTIONS>("add"), (Arguments{"--enable-super-kernel"}));

  feature.feature = "DEBUG_PER_OP_MAX_CORE_NUM";
  feature.kernelNameCount = 1;
  feature.kernelNames = nullptr;
  EXPECT_FALSE(manager.Init(&options));

  const char *badNames[] = {nullptr};
  feature.kernelNames = badNames;
  EXPECT_FALSE(manager.Init(&options));
  badNames[0] = "";
  EXPECT_FALSE(manager.Init(&options));
  badNames[0] = "[";
  EXPECT_FALSE(manager.Init(&options));
}

TEST_F(FeatureManagerTest, RepeatedSkFeatureUsesNormalRuleProcessing) {
  const char *names[] = {".*"};
  aclmdlRISpecSKFeature features[] = {
      {"DEBUG_PER_OP_MAX_CORE_NUM", 1, names},
      {"DEBUG_PER_OP_MAX_CORE_NUM", 1, names},
  };
  options.enableSK = true;
  options.specSKFeatureCount = 2;
  options.specSKFeatures = features;
  ASSERT_TRUE(manager.Init(&options));
  EXPECT_EQ(manager.Get<Feature::SK_COMPILE_OPTIONS>("add"),
            (Arguments{"--enable-super-kernel", "-D__ASCENDC_SUPER_KERNEL_DEBUG__"}));
}

TEST_F(FeatureManagerTest, CountPointerMismatchAndOverflowRejectedBeforeReading) {
  options.specCompileOptionCount = 1;
  EXPECT_FALSE(manager.Init(&options));
  options.specCompileOptionCount = 0;
  options.specSKFeatureCount = 1;
  EXPECT_FALSE(manager.Init(&options));
  options.specSKFeatureCount = 0;

  aclmdlRISpecCompileOption basic{".*", 0, nullptr};
  options.specCompileOptions = &basic;
  options.specCompileOptionCount = std::numeric_limits<uint64_t>::max();
  EXPECT_FALSE(manager.Init(&options));

  options.specCompileOptionCount = 1;
  const char *args[] = {"-O2"};
  basic.compileOptions = args;
  basic.compileOptionCount = std::numeric_limits<uint64_t>::max();
  EXPECT_FALSE(manager.Init(&options));
}

TEST_F(FeatureManagerTest, FailedReinitializationPreservesPreviousConfiguration) {
  const char *args[] = {"-O2"};
  aclmdlRISpecCompileOption basic{"add.*", 1, args};
  const char *names[] = {"add.*"};
  aclmdlRISpecSKFeature feature{"DEBUG_PER_OP_MAX_CORE_NUM", 1, names};
  options.jobs = 4;
  options.enableSK = true;
  options.specCompileOptionCount = 1;
  options.specCompileOptions = &basic;
  options.specSKFeatureCount = 1;
  options.specSKFeatures = &feature;
  ASSERT_TRUE(manager.Init(&options));

  options.jobs = 16;
  feature.kernelNames = nullptr;
  EXPECT_FALSE(manager.Init(&options));
  EXPECT_EQ(manager.Get<Feature::COMPILE_JOBS>(), 4U);
  EXPECT_TRUE(manager.Get<Feature::ENABLE_SK>());
  EXPECT_EQ(manager.Get<Feature::BASIC_COMPILE_OPTIONS>("add"), (Arguments{"-O2"}));
  EXPECT_EQ(manager.Get<Feature::SK_COMPILE_OPTIONS>("add"),
            (Arguments{"--enable-super-kernel", "-D__ASCENDC_SUPER_KERNEL_DEBUG__"}));

  ASSERT_TRUE(manager.Init(nullptr));
  EXPECT_EQ(manager.Get<Feature::COMPILE_JOBS>(), 1U);
  EXPECT_FALSE(manager.Get<Feature::ENABLE_SK>());
  EXPECT_TRUE(manager.Get<Feature::BASIC_COMPILE_OPTIONS>("add").empty());
  EXPECT_TRUE(manager.Get<Feature::SK_COMPILE_OPTIONS>("add").empty());
}

TEST_F(FeatureManagerTest, ConcurrentGetReadsStableConfiguration) {
  const char *args[] = {"-O2"};
  aclmdlRISpecCompileOption basic{"add_.*", 1, args};
  options.specCompileOptions = &basic;
  options.specCompileOptionCount = 1;
  ASSERT_TRUE(manager.Init(&options));

  std::vector<std::future<bool>> futures;
  for (int i = 0; i < 8; ++i) {
    futures.emplace_back(std::async(std::launch::async, [&]() {
      for (int j = 0; j < 100; ++j) {
        if (manager.Get<Feature::BASIC_COMPILE_OPTIONS>("add_123") != Arguments{"-O2"}) {
          return false;
        }
      }
      return true;
    }));
  }
  for (auto &future : futures) {
    EXPECT_TRUE(future.get());
  }
}

}  // namespace
}  // namespace model_spec
