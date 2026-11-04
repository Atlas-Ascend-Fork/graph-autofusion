/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ascendc_api_registry.h"
#include "gtest/gtest.h"

namespace codegen {
TEST(AscendCCubeApiRegisterMatmul, TensorCVBlockIsAvailableToGeneratedKernels) {
  const auto &source = AscendCApiRegistry::GetInstance().GetFileContent("mat_mul_tensor_cv_block.h");
  EXPECT_NE(source.find("class TensorCVBlockMmad<"), std::string::npos);
  EXPECT_NE(source.find("CopyTensorCVL0CToUB(ubTensor"), std::string::npos);
}

TEST(AscendCCubeApiRegisterMatmul, TensorCommonImplementationIsAvailableToAllEntrypoints) {
  auto &registry = AscendCApiRegistry::GetInstance();
  const auto &dependencies = registry.GetFileContent("mat_mul_tensor_include_headers.h");
  EXPECT_NE(dependencies.find("arch35/mat_mul_streamk.h"), std::string::npos);
  EXPECT_NE(dependencies.find("arch35/mat_mul_basic_split_k.h"), std::string::npos);
  const auto &source = registry.GetFileContent("mat_mul_tensor_common.h");
  const std::string entry = "void MatMulTensorCommonKernel(";
  ASSERT_NE(source.find(entry), std::string::npos);
  for (const auto *header : {"matmul.h", "matmul_dynamic.h", "batch_matmul.h", "batch_matmul_dynamic.h"}) {
    EXPECT_NE(registry.GetFileContent(header).find("MatmulV3Advanced::MatMulTensorCommonKernel<"), std::string::npos);
  }
}

TEST(AscendCCubeApiRegisterMatmul, TensorCommonSchedulerIsAvailableToGeneratedKernels) {
  const auto &source = AscendCApiRegistry::GetInstance().GetFileContent("mat_mul_tensor_common_scheduler.h");
  EXPECT_NE(source.find("class TensorCommonBlockScheduler"), std::string::npos);
  EXPECT_NE(source.find("blockIdx = MapBlockIndex(blockIdx)"), std::string::npos);
}
}  // namespace codegen
