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

#include <limits>
#include <vector>
#include "gtest/gtest.h"

namespace model_spec {
namespace {
using Arguments = std::vector<std::string>;

class FeatureManagerSt : public testing::Test {
 protected:
  FeatureManager manager;
  aclmdlRISpecOptions options{};
};

TEST_F(FeatureManagerSt, DefaultsAndModelLevelSwitches) {
  ASSERT_TRUE(manager.Init(nullptr));
  EXPECT_EQ(manager.Get<Feature::COMPILE_JOBS>(), 1U);
  EXPECT_FALSE(manager.Get<Feature::ENABLE_SK>());
  EXPECT_TRUE(manager.Get<Feature::BASIC_COMPILE_OPTIONS>("addcustom").empty());
  EXPECT_TRUE(manager.Get<Feature::SK_COMPILE_OPTIONS>("addcustom").empty());

  options.jobs = std::numeric_limits<uint64_t>::max();
  options.enableSK = true;
  ASSERT_TRUE(manager.Init(&options));
  EXPECT_EQ(manager.Get<Feature::COMPILE_JOBS>(), options.jobs);
  EXPECT_TRUE(manager.Get<Feature::ENABLE_SK>());
  EXPECT_EQ(manager.Get<Feature::SK_COMPILE_OPTIONS>("addcustom"), (Arguments{"--enable-super-kernel"}));
}

TEST_F(FeatureManagerSt, PublicApiInitializesAndQueriesMultipleKernels) {
  const char *addArgs[] = {"-O2", "-DADD=1"};
  const char *reluArgs[] = {"-O2", "-DSLOPE=0.125"};
  const char *matmulArgs[] = {"-O3", "-DMATMUL=1"};
  const char *mixArgs[] = {"-DFUSED_RELU=1"};
  const char *globalArgs[] = {"-DGLOBAL=1"};
  aclmdlRISpecCompileOption basic[] = {
      {nullptr, 1, globalArgs},     {"addcustom.*", 2, addArgs},         {"leakyrelu_.*", 2, reluArgs},
      {"matmul_.*", 2, matmulArgs}, {"matmul_leakyrelu_.*", 1, mixArgs},
  };
  const char *debugVectorNames[] = {"addcustom.*", "leakyrelu_.*"};
  const char *debugMatmulNames[] = {"matmul_.*", "matmul_leakyrelu_.*"};
  const char *dcciNames[] = {"addcustom.*", "matmul_leakyrelu_.*"};
  const char *ignoredAllNames[] = {".*"};
  const char *ignoredAddNames[] = {"addcustom.*"};
  const char *ignoredReluNames[] = {"leakyrelu_.*"};
  const char *ignoredMatmulNames[] = {"matmul_.*"};
  aclmdlRISpecSKFeature skFeatures[] = {
      {"DEBUG_PER_OP_MAX_CORE_NUM", 2, debugVectorNames},
      {"DEBUG_PER_OP_MAX_CORE_NUM", 2, debugMatmulNames},
      {"PRELOAD_CODE", 1, ignoredAllNames},
      {"SPLIT_MODE", 1, ignoredAddNames},
      {"STREAM_FUSION", 1, ignoredMatmulNames},
      {"DCCI_DISABLE_ON_KERNEL", 2, dcciNames},
      {"DEBUG_SYNC_ALL", 1, ignoredReluNames},
      {"DCCI_BEFORE_KERNEL_START", 1, ignoredAddNames},
      {"DCCI_AFTER_KERNEL_END", 1, ignoredMatmulNames},
      {"UBUF_LOCK_IGNORE_KERNEL", 1, ignoredAllNames},
      {"EARLY_START", 1, ignoredReluNames},
  };
  options.jobs = 4;
  options.enableSK = true;
  options.specCompileOptionCount = 5;
  options.specCompileOptions = basic;
  options.specSKFeatureCount = sizeof(skFeatures) / sizeof(skFeatures[0]);
  options.specSKFeatures = skFeatures;
  ASSERT_TRUE(manager.Init(&options));

  EXPECT_EQ(manager.Get<Feature::COMPILE_JOBS>(), 4U);
  EXPECT_EQ(manager.Get<Feature::BASIC_COMPILE_OPTIONS>("addcustom_tiling"),
            (Arguments{"-DGLOBAL=1", "-O2", "-DADD=1"}));
  EXPECT_EQ(manager.Get<Feature::BASIC_COMPILE_OPTIONS>("matmul_leakyrelu_123"),
            (Arguments{"-DGLOBAL=1", "-O3", "-DMATMUL=1", "-DFUSED_RELU=1"}));
  EXPECT_EQ(manager.Get<Feature::SK_COMPILE_OPTIONS>("addcustom_tiling"),
            (Arguments{"--enable-super-kernel", "-D__ASCENDC_SUPER_KERNEL_DEBUG__",
                       "-D__ASCENDC_SUPER_KERNEL_ENABLE_GM_GET_SET_VALUE_DCCI__",
                       "-D__ASCENDC_SUPER_KERNEL_DISABLE_DCCI__"}));
  EXPECT_EQ(manager.Get<Feature::SK_COMPILE_OPTIONS>("matmul_aic_only"),
            (Arguments{"--enable-super-kernel", "-D__ASCENDC_SUPER_KERNEL_DEBUG__",
                       "-D__ASCENDC_SUPER_KERNEL_ENABLE_GM_GET_SET_VALUE_DCCI__"}));
  EXPECT_EQ(manager.Get<Feature::SK_COMPILE_OPTIONS>("matmul_leakyrelu_123"),
            (Arguments{"--enable-super-kernel", "-D__ASCENDC_SUPER_KERNEL_DEBUG__",
                       "-D__ASCENDC_SUPER_KERNEL_ENABLE_GM_GET_SET_VALUE_DCCI__",
                       "-D__ASCENDC_SUPER_KERNEL_DISABLE_DCCI__"}));
}

TEST_F(FeatureManagerSt, EnableSkZeroSuppressesSkArguments) {
  const char *debugNames[] = {".*"};
  aclmdlRISpecSKFeature skFeature{"DEBUG_PER_OP_MAX_CORE_NUM", 1, debugNames};
  options.enableSK = false;
  options.specSKFeatureCount = 1;
  options.specSKFeatures = &skFeature;
  ASSERT_TRUE(manager.Init(&options));
  EXPECT_TRUE(manager.Get<Feature::SK_COMPILE_OPTIONS>("addcustom").empty());
}

TEST_F(FeatureManagerSt, FailedReinitializationPreservesAllPreviousFeatures) {
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
  EXPECT_EQ(manager.Get<Feature::BASIC_COMPILE_OPTIONS>("add"), (Arguments{"-O2"}));
  EXPECT_EQ(manager.Get<Feature::SK_COMPILE_OPTIONS>("add"),
            (Arguments{"--enable-super-kernel", "-D__ASCENDC_SUPER_KERNEL_DEBUG__"}));

  ASSERT_TRUE(manager.Init(nullptr));
  EXPECT_EQ(manager.Get<Feature::COMPILE_JOBS>(), 1U);
  EXPECT_FALSE(manager.Get<Feature::ENABLE_SK>());
  EXPECT_TRUE(manager.Get<Feature::BASIC_COMPILE_OPTIONS>("add").empty());
  EXPECT_TRUE(manager.Get<Feature::SK_COMPILE_OPTIONS>("add").empty());
}

}  // namespace
}  // namespace model_spec
