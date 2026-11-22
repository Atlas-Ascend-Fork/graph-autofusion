/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* 尾轴广播 32B 槽位源 RemovePad 归一化的 CPU 仿真数值验证。
 * 语义：y[i][j][k] = (x0[i][j] - max(x0[i][:])) + x1[i][j][k]，x0 形状 [100,8,1]（尾轴退化，行宽 32B
 * 触发 32B 槽位布局，z1 轴归约保持实轴），x1/y 形状 [100,8,50]。若广播源未插入 RemovePad 归一化，
 * BroadcastExtend 会把槽位间隙当地址错读源数据。 */

#include <gtest/gtest.h>
#include <cmath>
#include "tikicpulib.h"
#include "autofuse_tiling_data.h"

extern "C" __global__ __aicore__ void tail_brc_remove_pad_test(GM_ADDR input, GM_ADDR input1, GM_ADDR y,
                                                               GM_ADDR workspace, GM_ADDR gm_tiling_data);
extern "C" int64_t AutofuseTiling(AutofuseTilingData *tiling, uint32_t *workspaceSize, uint64_t *blockDim,
                                  uint32_t aiv_num, uint32_t ub_size);

namespace {
class E2E_TailBrcRemovePad_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

// GM 缓冲、tiling 数据与场景几何维度的打包载体，在 Prepare/Verify 辅助函数与 TEST_P 之间传递。
struct TailBrcRunCtx {
  float *x0 = nullptr;
  float *x1 = nullptr;
  float *y = nullptr;
  int n = 0;
  int k = 0;
  AutofuseTilingData tiling_data;
};

// 分配 GM 缓冲（保留 +32 松弛）、按原序列填充输入、执行 tiling 与 tiling_key 诊断覆写并 launch kernel。
void PrepareTailBrcRemovePadRun(TailBrcRunCtx &ctx, int m, int n, int k) {
  const int brc_input_size = m * n;
  const int add_input_size = m * n * k;
  const int output_size = m * n * k;
  uint64_t block_dim = 48;

  ctx.n = n;
  ctx.k = k;
  ctx.x0 = (float *)AscendC::GmAlloc(brc_input_size * sizeof(float) + 32);
  ctx.x1 = (float *)AscendC::GmAlloc(add_input_size * sizeof(float) + 32);
  ctx.y = (float *)AscendC::GmAlloc(output_size * sizeof(float) + 32);
  ASSERT_NE(ctx.x0, nullptr);
  ASSERT_NE(ctx.x1, nullptr);
  ASSERT_NE(ctx.y, nullptr);

  srand(2);
  for (int i = 0; i < brc_input_size; ++i) {
    ctx.x0[i] = rand() / (double)RAND_MAX;
  }
  for (int i = 0; i < add_input_size; ++i) {
    ctx.x1[i] = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
  }

  uint32_t ws_size = 0;
  AutofuseTiling(&ctx.tiling_data, &ws_size, &block_dim, 48, 192 * 1024);
  std::cout << "graph0_tiling_key = " << ctx.tiling_data.graph0_tiling_key
            << ", result2_g0_tiling_key = " << ctx.tiling_data.graph0_result2_g0_tiling_data.tiling_key
            << ", result0_g0_tiling_key = " << ctx.tiling_data.graph0_result0_g0_tiling_data.tiling_key
            << ", block_dim = " << ctx.tiling_data.block_dim << std::endl;
  // 诊断覆写：强制走全载模板的非 nddma 变体（CPU 仿真器不支持 DataCopyNddma 指令，pv_ccu unknown instr）。
  ctx.tiling_data.graph0_result2_g0_tiling_data.tiling_key = 0;
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(tail_brc_remove_pad_test, ctx.tiling_data.block_dim, (uint8_t *)ctx.x0, (uint8_t *)ctx.x1,
              (uint8_t *)ctx.y, nullptr, (uint8_t *)&ctx.tiling_data);
}

// 校验固定 (i, j) 行内全部尾轴元素：y 应等于 x0 减行内最大值后加 x1；超阈值元素最多打印前 10 个。
void CheckTailBrcRow(const TailBrcRunCtx &ctx, float brc_val, int i, int j, uint32_t *diff_count) {
  for (int kk = 0; kk < ctx.k; ++kk) {
    const int offset = (i * ctx.n + j) * ctx.k + kk;
    const float expect = brc_val + ctx.x1[offset];
    const double diff = std::fabs(static_cast<double>(ctx.y[offset]) - static_cast<double>(expect));
    if (diff > 1e-5) {
      if (*diff_count < 10) {
        std::cout << "y[" << i << "][" << j << "][" << kk << "] = " << ctx.y[offset] << ", expect = " << expect
                  << std::endl;
      }
      (*diff_count)++;
    }
  }
}

// CPU 参考实现：x0 减去行内最大值后按尾轴扩维到 k，再与 x1 相加；返回超阈值元素总数。
uint32_t VerifyTailBrcRemovePadResult(const TailBrcRunCtx &ctx, int m) {
  const int n = ctx.n;
  std::vector<float> max_val(m, 0.0f);
  for (int i = 0; i < m; ++i) {
    float cur_max = ctx.x0[i * n];
    for (int j = 1; j < n; ++j) {
      cur_max = (ctx.x0[i * n + j] > cur_max) ? ctx.x0[i * n + j] : cur_max;
    }
    max_val[i] = cur_max;
  }
  uint32_t diff_count = 0;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      const float brc_val = ctx.x0[i * n + j] - max_val[i];
      CheckTailBrcRow(ctx, brc_val, i, j, &diff_count);
    }
  }
  return diff_count;
}

TEST_P(E2E_TailBrcRemovePad_Code, CalculateCorrect) {
  // CPU 仿真器不支持 {R,1} 部分向量 RA VF reduce 的寄存器写（pv_reg_write 断言，含 R=8 满向量），
  // 本场景 kernel 无法在仿真器上执行；数值校验代码保留，待仿真器支持后移除 GTEST_SKIP 启用。
  GTEST_SKIP() << "CPU simulator does not support RA VF reduce (pv_reg_write assert); run on hardware instead.";
  const int m = GetParam()[0];
  const int n = GetParam()[1];
  const int k = GetParam()[2];
  const int output_size = m * n * k;

  TailBrcRunCtx ctx;
  PrepareTailBrcRemovePadRun(ctx, m, n, k);
  const uint32_t diff_count = VerifyTailBrcRemovePadResult(ctx, m);
  EXPECT_EQ(diff_count, 0) << " of " << output_size;

  AscendC::GmFree(ctx.x0);
  AscendC::GmFree(ctx.x1);
  AscendC::GmFree(ctx.y);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_TailBrcRemovePad_Code,
                         ::testing::Values(std::vector<int>{100, 8, 50}));

}  // namespace
