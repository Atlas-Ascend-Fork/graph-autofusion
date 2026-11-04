/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include <limits>

#include "tikicpulib.h"
#include "kernel_operator.h"
using namespace AscendC;

#include "utils.h"
#include "test_api_utils.h"
#include "reduce_padding_generated.h"
constexpr int kReduceOpMin = 0;
constexpr int kReduceOpMax = 1;
constexpr int kReduceOpSum = 2;
constexpr int kReduceOpProd = 3;

#ifndef INFINITY
#define INFINITY (1.0f / 0.0f)
#endif

class TestApiReduceInit : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

TEST_P(TestApiReduceInit, Test_reduce_init_max) {
  // 构造测试输入和预期结果
  int32_t dim_a = this->GetParam()[0];
  int32_t dim_r = this->GetParam()[1];
  int32_t dim_r_current = this->GetParam()[2];
  int32_t inner_r = this->GetParam()[3];
  int32_t inner_r_align = (inner_r + 7) / 8 * 8;
  auto *x = (float *)AscendC::GmAlloc(sizeof(float) * dim_a * dim_r);
  float expect[dim_a * dim_r];

  for (int i = 0; i < dim_a * (dim_r / inner_r_align); i++) {
    for (int j = 0; j < inner_r_align; j++) {
      int32_t index = i * inner_r_align + j;
      if (j >= inner_r) {
        expect[index] = -INFINITY;
        x[index] = 1.0f;
      } else {
        expect[index] = 1.0f;
        x[index] = 1.0f;
      }
    }
  }

  // 构造Api调用函数
  auto kernel = [](uint64_t dim_a, uint64_t dim_r, uint64_t dim_r_current, uint64_t inner_r, float *x) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf;
    tpipe.InitBuffer(xbuf, sizeof(float) * dim_a * dim_r);

    auto x_tensor = xbuf.Get<float>();

    GmToUb(x_tensor, x, dim_a * dim_r);

    RunPadding<float, 1, false>(x_tensor, dim_a, dim_r, dim_r_current, inner_r);

    UbToGm(x, x_tensor, dim_a * dim_r);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, dim_a, dim_r, dim_r_current, inner_r, x);

  // 验证结果
  for (int i = 0; i < dim_a * dim_r; ++i) {
    EXPECT_EQ(x[i], expect[i]) << "offset=" << i;
  }
  AscendC::GmFree(x);
}

TEST_P(TestApiReduceInit, Test_reduce_init_min) {
  // 构造测试输入和预期结果
  int32_t dim_a = this->GetParam()[0];
  int32_t dim_r = this->GetParam()[1];
  int32_t dim_r_current = this->GetParam()[2];
  int32_t inner_r = this->GetParam()[3];
  int32_t inner_r_align = (inner_r + 7) / 8 * 8;
  auto *x = (float *)AscendC::GmAlloc(sizeof(float) * dim_a * dim_r);
  float expect[dim_a * dim_r];

  for (int i = 0; i < dim_a * (dim_r / inner_r_align); i++) {
    for (int j = 0; j < inner_r_align; j++) {
      int32_t index = i * inner_r_align + j;
      if (j >= inner_r) {
        expect[index] = INFINITY;
        x[index] = 1.0f;
      } else {
        expect[index] = 1.0f;
        x[index] = 1.0f;
      }
    }
  }

  // 构造Api调用函数
  auto kernel = [](uint64_t dim_a, uint64_t dim_r, uint64_t dim_r_current, uint64_t inner_r, float *x) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf;
    tpipe.InitBuffer(xbuf, sizeof(float) * dim_a * dim_r);

    auto x_tensor = xbuf.Get<float>();

    GmToUb(x_tensor, x, dim_a * dim_r);

    RunPadding<float, 0, false>(x_tensor, dim_a, dim_r, dim_r_current, inner_r);

    UbToGm(x, x_tensor, dim_a * dim_r);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, dim_a, dim_r, dim_r_current, inner_r, x);

  // 验证结果
  for (int i = 0; i < dim_a * dim_r; ++i) {
    EXPECT_EQ(x[i], expect[i]) << "offset=" << i;
  }
  AscendC::GmFree(x);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, TestApiReduceInit,
                         ::testing::Values(std::vector<int>{32, 32, 32, 7}, std::vector<int>{300, 32, 32, 7}));

class TestApiReduceInit_RepeatStrideOverLimit : public testing::Test,
                                                public testing::WithParamInterface<std::vector<int>> {};

TEST_P(TestApiReduceInit_RepeatStrideOverLimit, Test_reduce_init_sum) {
  // 构造测试输入和预期结果
  int32_t dim_a = this->GetParam()[0];
  int32_t dim_r = this->GetParam()[1];
  int32_t dim_r_current = this->GetParam()[2];
  int32_t inner_r = this->GetParam()[3];
  int32_t inner_r_align = (inner_r + 7) / 8 * 8;
  auto *x = (float *)AscendC::GmAlloc(sizeof(float) * dim_a * dim_r);
  float expect[dim_a * dim_r];

  for (int i = 0; i < dim_a * (dim_r / inner_r_align); i++) {
    for (int j = 0; j < inner_r_align; j++) {
      int32_t index = i * inner_r_align + j;
      if (j >= inner_r) {
        expect[index] = 0;
        x[index] = 1.0f;
      } else {
        expect[index] = 1.0f;
        x[index] = 1.0f;
      }
    }
  }

  // 构造Api调用函数
  auto kernel = [](uint64_t dim_a, uint64_t dim_r, uint64_t dim_r_current, uint64_t inner_r, float *x) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf;
    tpipe.InitBuffer(xbuf, sizeof(float) * dim_a * dim_r);

    auto x_tensor = xbuf.Get<float>();

    GmToUb(x_tensor, x, dim_a * dim_r);

    RunPadding<float, 2, true>(x_tensor, dim_a, dim_r, dim_r_current, inner_r);

    UbToGm(x, x_tensor, dim_a * dim_r);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, dim_a, dim_r, dim_r_current, inner_r, x);

  // 验证结果
  for (int i = 0; i < dim_a * dim_r; ++i) {
    EXPECT_EQ(x[i], expect[i]) << "offset=" << i;
  }
  AscendC::GmFree(x);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, TestApiReduceInit_RepeatStrideOverLimit,
                         ::testing::Values(std::vector<int>{4, 2136, 2136, 2132}));

class TestApiReduceInitPhysicalStride : public testing::Test, public testing::WithParamInterface<std::vector<int>> {
 protected:
  template <int ReduceType>
  void CheckPadding(float identity) {
    const auto &shape = GetParam();
    const int dim_a = shape[0];
    const int outer_nominal = shape[1];
    const int outer_actual = shape[2];
    const int inner_stride = shape[3];
    const int inner_actual = shape[4];
    const int count = dim_a * outer_nominal * inner_stride;
    auto *x = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * count));
    std::vector<float> expected(count, identity);
    for (int a = 0; a < dim_a; ++a) {
      for (int r = 0; r < outer_nominal; ++r) {
        for (int i = 0; i < inner_stride; ++i) {
          const int offset = (a * outer_nominal + r) * inner_stride + i;
          x[offset] = 99.0f;
          if (r < outer_actual && i < inner_actual) {
            expected[offset] = x[offset] = static_cast<float>(offset + 1);
          }
        }
      }
    }
    auto kernel = [](int a, int r_nominal, int r_actual, int pitch, int actual, float *data) {
      TPipe pipe;
      TBuf<TPosition::VECCALC> buffer;
      pipe.InitBuffer(buffer, sizeof(float) * a * r_nominal * pitch);
      auto local = buffer.Get<float>();
      GmToUb(local, data, a * r_nominal * pitch);
      RunPadding<float, ReduceType, false>(local, a, r_nominal * pitch, r_actual * pitch, actual, pitch);
      UbToGm(data, local, a * r_nominal * pitch);
    };
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1, dim_a, outer_nominal, outer_actual, inner_stride, inner_actual, x);
    for (int i = 0; i < count; ++i) {
      EXPECT_EQ(x[i], expected[i]) << "physical offset " << i;
    }
    AscendC::GmFree(x);
  }
};

TEST_P(TestApiReduceInitPhysicalStride, Sum) {
  CheckPadding<kReduceOpSum>(0.0f);
}

TEST_P(TestApiReduceInitPhysicalStride, Max) {
  CheckPadding<kReduceOpMax>(-std::numeric_limits<float>::infinity());
}

TEST_P(TestApiReduceInitPhysicalStride, Min) {
  CheckPadding<kReduceOpMin>(std::numeric_limits<float>::infinity());
}

TEST_P(TestApiReduceInitPhysicalStride, Prod) {
  CheckPadding<kReduceOpProd>(1.0f);
}

// `{dim_a, outer_nominal, outer_actual, inner_stride, inner_actual}`。
//
// 外层 R 填充那一支（`dim_r_current != dim_r`）按 R 段遍历、用 `dim_a` 当 repeatTime、
// 用行距当 dstRepeatStride，而 `Duplicate` 这两个形参都是 `uint8_t`，超了是**静默截断**。
// 下面四组专打这条编码边界，缺一不可：
//   {255,…} 分批前的最后一格；{256,…} repeatTime 截成 0、整段一行都不填；
//   {257,…} 跨一次分批，验证第二批的基址；
//   {17, 2, 1, 1024, 1024} 行距 `2048/8 = 256`，刚过 dstRepeatStride 上限。
// 这一组还必须满足 `repeat_count < dim_a`（否则走的是另一支）：`dim_r_diff = 1024` →
// `repeat_count = 16 < 17`。
//
// 此前这个套件里 `dim_a` 最大到 1025，但那几组 `outer_actual == outer_nominal`，整个外层 R
// 填充段压根不进；而进得去的那几组 `dim_a` 都是个位数。大 `dim_a` 与外层 R 尾块**从未组合
// 出现过**，截断因此一直没被测到。
INSTANTIATE_TEST_SUITE_P(InnerAndOuterTails, TestApiReduceInitPhysicalStride,
                         ::testing::Values(std::vector<int>{1, 2, 2, 32, 17}, std::vector<int>{3, 3, 2, 32, 17},
                                           std::vector<int>{9, 2, 2, 32, 17}, std::vector<int>{1, 2, 2, 32, 25},
                                           std::vector<int>{1, 2, 2, 32, 32}, std::vector<int>{1025, 2, 2, 8, 3},
                                           std::vector<int>{2040, 1, 1, 8, 3}, std::vector<int>{2048, 1, 1, 8, 3},
                                           std::vector<int>{255, 3, 2, 32, 32}, std::vector<int>{256, 3, 2, 32, 32},
                                           std::vector<int>{257, 3, 2, 32, 32}, std::vector<int>{17, 2, 1, 1024, 1024},
                                           std::vector<int>{4, 2, 2, 2136, 17}, std::vector<int>{3, 2, 2, 2136, 17},
                                           std::vector<int>{3, 3, 2, 32, 16}));

// half has sixteen lanes per block and 128 lanes per repeat. Nine rows exercise
// both mask words in a full repeat and the separate remainder-row path.
template <int ReduceType>
void CheckHalfReducePadding() {
  constexpr int rows = 9;
  constexpr int pitch = 32;
  constexpr int actual = 17;
  auto *data = static_cast<half *>(AscendC::GmAlloc(rows * pitch * sizeof(half)));
  ASSERT_NE(data, nullptr);
  for (int i = 0; i < rows * pitch; ++i) {
    data[i] = half(7.0f);
  }
  auto kernel = [](half *data) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, rows * pitch * sizeof(half));
    auto local = buffer.Get<half>();
    GmToUb(local, data, rows * pitch);
    RunPadding<half, ReduceType, true>(local, rows, pitch, pitch, actual, pitch);
    UbToGm(data, local, rows * pitch);
  };
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, data);
  const float identity =
      ReduceType == kReduceOpMax ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < pitch; ++col) {
      EXPECT_EQ(static_cast<float>(data[row * pitch + col]), col < actual ? 7.0f : identity)
          << "row=" << row << " col=" << col;
    }
  }
  AscendC::GmFree(data);
}

TEST(TestApiReduceInitHalf, MaxPreservesDataAndPadsBothMaskWords) {
  CheckHalfReducePadding<kReduceOpMax>();
}

TEST(TestApiReduceInitHalf, MinPreservesDataAndPadsBothMaskWords) {
  CheckHalfReducePadding<kReduceOpMin>();
}

// Check exact integer identities and both guards around the destination view.
// These execute generated native Duplicate calls, including a tail smaller than
// one block and an entirely inactive outer-R row.
template <int Kind>
void CheckIntegerPadding(int32_t identity) {
  constexpr int guard = 8;
  constexpr int count = 3 * 2 * 16;
  constexpr int total = count + 2 * guard;
  auto *data = static_cast<int32_t *>(AscendC::GmAlloc(total * sizeof(int32_t)));
  ASSERT_NE(data, nullptr);
  for (int i = 0; i < total; ++i) {
    data[i] = 123456789;
  }
  auto kernel = [](int32_t *data) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, total * sizeof(int32_t));
    auto local = buffer.Get<int32_t>();
    GmToUb(local, data, total);
    RunPadding<int32_t, Kind, false>(local[guard], 3, 32, 16, 5, 16);
    UbToGm(data, local, total);
  };
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, data);
  for (int i = 0; i < total; ++i) {
    bool valid = i < guard || i >= guard + count || (i - guard) % 32 < 5;
    EXPECT_EQ(data[i], valid ? 123456789 : identity) << "offset=" << i;
  }
  AscendC::GmFree(data);
}
TEST(ReducePaddingLoweringInteger, Minimum) {
  CheckIntegerPadding<0>(INT32_MAX);
}
TEST(ReducePaddingLoweringInteger, Maximum) {
  CheckIntegerPadding<1>(INT32_MIN);
}
TEST(ReducePaddingLoweringInteger, Sum) {
  CheckIntegerPadding<2>(0);
}
TEST(ReducePaddingLoweringInteger, Product) {
  CheckIntegerPadding<3>(1);
}
TEST(ReducePaddingLoweringInteger, Any) {
  CheckIntegerPadding<4>(0);
}
TEST(ReducePaddingLoweringInteger, All) {
  CheckIntegerPadding<5>(1);
}
TEST(ReducePaddingLoweringInteger, Mean) {
  CheckIntegerPadding<6>(0);
}
