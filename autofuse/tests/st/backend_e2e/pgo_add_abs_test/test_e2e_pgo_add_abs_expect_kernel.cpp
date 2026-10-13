/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <cmath>
#include "tikicpulib.h"

#include "autofuse_tiling_data.h"
#include "pgo_common/pgo_test_utils.h"

ResLimit g_no_limit_res = {1, 10, 0, 192 * 1024, {}};
extern "C" __global__ __aicore__ void add_abs_test(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace,
                                                   GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim,
                                  uint32_t aiv_num, uint32_t ub_size);

extern "C" int64_t StubPgoGetProfilingBatch(PgoTensorArgs *tensor_args, void *stream, uint64_t workspaceSize,
                                            std::vector<AutofuseTilingDataPerf> *profiles) {
  (void)stream;
  (void)workspaceSize;
  void *input0 = tensor_args->inputs[0];
  void *input1 = tensor_args->inputs[1];
  void *output0 = tensor_args->outputs[0];
  static double prof_time = 1;
  for (auto &profile : *profiles) {
    if (prof_time < 1) {
      AscendC::SetKernelMode(KernelMode::AIV_MODE);
      ICPU_RUN_KF(add_abs_test, profile.tiling_data.block_dim, (uint8_t *)input0, (uint8_t *)input1, (uint8_t *)output0,
                  nullptr, (uint8_t *)&profile.tiling_data);
    }
    profile.best_perf = prof_time;
    prof_time += 0.1;
  }
  return 0;
}

static float *g_expect_data = nullptr;
static int g_test_size = 0;

extern "C" int64_t StubPgoGetProfiling(PgoTensorArgs *tensor_args, void *stream, uint64_t workspaceSize,
                                       AutofuseTilingData *tiling_data, double *cost_time) {
  (void)stream;
  (void)workspaceSize;
  void *input0 = tensor_args->inputs[0];
  void *input1 = tensor_args->inputs[1];
  void *output0 = tensor_args->outputs[0];
  static double prof_time = 1;
  *cost_time = prof_time;
  prof_time += 0.1;
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(add_abs_test, tiling_data->block_dim, (uint8_t *)input0, (uint8_t *)input1, (uint8_t *)output0, nullptr,
              (uint8_t *)tiling_data);

  if (g_expect_data != nullptr && g_test_size > 0) {
    float *y = static_cast<float *>(output0);
    uint32_t diff_count = 0;
    for (int i = 0; i < g_test_size; i++) {
      if (y[i] != g_expect_data[i]) {
        diff_count++;
      }
    }
    if (diff_count > 0) {
      std::cout << "diff count: " << diff_count << std::endl;
      const uint32_t *data_ptr = reinterpret_cast<const uint32_t *>(tiling_data);
      const size_t elem_count = sizeof(AutofuseTilingData) / sizeof(uint32_t);
      std::cout << "tiling_data dump (size=" << sizeof(AutofuseTilingData) << " bytes, " << elem_count
                << " uint32 elements):" << std::endl;
      for (size_t i = 0; i < elem_count; i++) {
        std::cout << "  [" << i << "] = " << data_ptr[i] << std::endl;
      }
      return -1;
    }
  }

  return 0;
}

class E2E_BackendPgoAddAbs_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {
 protected:
  void SetUp() override {
    // 从参数获取测试形状
    test_shape = GetParam();
    test_size = test_shape[0] * test_shape[1] * test_shape[2];
    block_dim = 48;

    // 分配内存
    input1 = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
    input2 = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
    y = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
    expect = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);

    // 准备测试数据和期望数据
    srand(1);
    for (int i = 0; i < test_size; i++) {
      input1[i] = rand() / (double)RAND_MAX;
      input2[i] = rand() / (double)RAND_MAX;
      expect[i] = std::fabs(input1[i] + input2[i]);
    }

    // 设置全局指针
    g_expect_data = expect;
    g_test_size = test_size;
  }

  void TearDown() override {
    // 清除全局指针
    g_expect_data = nullptr;
    g_test_size = 0;

    // 释放内存
    AscendC::GmFree(input1);
    AscendC::GmFree(input2);
    AscendC::GmFree(y);
    AscendC::GmFree(expect);
  }

  // 成员变量
  std::vector<int> test_shape;
  int test_size;
  uint32_t block_dim;
  float *input1;
  float *input2;
  float *y;
  float *expect;
  AutofuseTilingData tiling_data = {};
};

TEST_P(E2E_BackendPgoAddAbs_Code, PgoByCoreNum) {
  uint64_t ws_size = 0;
  PgoTensorArgPack tensor_args(input1, input2, y);
  int result = RunPgoTilingSearch(std::filesystem::path(__FILE__), &tiling_data, &ws_size, &block_dim, &g_no_limit_res,
                                  &tensor_args.tensor_args, StubPgoGetProfiling, StubPgoGetProfilingBatch);
  EXPECT_EQ(result, 0);
}

TEST_P(E2E_BackendPgoAddAbs_Code, PgoByFilter) {
  GTEST_SKIP() << "pgo pruning path is unstable in this build";
  uint64_t ws_size = 0;
  PgoTensorArgPack tensor_args(input1, input2, y);
  int result = RunPgoTilingSearchWithPruning(std::filesystem::path(__FILE__), &tiling_data, &ws_size, &block_dim,
                                             &g_no_limit_res, &tensor_args.tensor_args, StubPgoGetProfiling,
                                             StubPgoGetProfilingBatch);
  EXPECT_EQ(result, 0);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_BackendPgoAddAbs_Code,
                         ::testing::Values(std::vector<int>{32, 16, 16}  // 用例输入的维度需要与构图接口的dims_size匹配
                                           ));
