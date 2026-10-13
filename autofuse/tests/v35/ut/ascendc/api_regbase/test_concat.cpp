/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "api_regbase/concat.h"

using namespace AscendC;

template <typename T, size_t N>
struct TestConcatParam {
  T *x[N]{};
  T *y{};
  T *y_expected{};
  concat::ConcatTiling<N> tiling{};
};

class RegbaseApiConcatTest : public testing::Test {
 protected:
  template <typename T, size_t N>
  static void InvokeKernel(TestConcatParam<T, N> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x_bufs[N];
    LocalTensor<T> input_tensors[N];
    __ubuf__ T *src_addrs[N];
    const auto &tiling = param.tiling;
    std::set<uint32_t> distinct_cols;
    for (size_t i = 0; i < N; ++i) {
      distinct_cols.emplace(tiling.num_srcs_cols[i]);
      auto x_size = tiling.num_rows * tiling.num_srcs_cols[i];
      tpipe.InitBuffer(x_bufs[i], sizeof(T) * x_size);
      input_tensors[i] = x_bufs[i].template Get<T>();
      src_addrs[i] = (__ubuf__ T *)input_tensors[i].GetPhyAddr();
      GmToUb(input_tensors[i], param.x[i], x_size);
    }
    TBuf<TPosition::VECCALC> ybuf, tmp;
    auto y_size = tiling.num_rows * tiling.num_dst_cols;
    tpipe.InitBuffer(ybuf, sizeof(T) * AlignUp(y_size, ONE_BLK_SIZE / sizeof(T)));
    tpipe.InitBuffer(tmp, 1024);
    LocalTensor<T> l_y = ybuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = tmp.Get<uint8_t>();

    auto tmp_buf = tmp.Get<uint8_t>();
    // concat::ConcatExtend<T, 2>((__ubuf__ T *)l_y.GetPhyAddr(), src_addrs, tmp_buf, param.tiling);
    if (distinct_cols.size() == 1) {
      concat::ConcatExtendDyn<T, N>((__ubuf__ T *)l_y.GetPhyAddr(), src_addrs, tmp_buf, param.tiling);
    } else {
      concat::ConcatExtend<T, N>((__ubuf__ T *)l_y.GetPhyAddr(), src_addrs, tmp_buf, param.tiling);
    }
    UbToGm(param.y, l_y, y_size);
  }

  template <typename T, size_t N>
  static void CreateTensorInput(TestConcatParam<T, N> &param) {
    // 构造测试输入和预期结果
    const auto &tiling = param.tiling;
    auto y_size = tiling.num_rows * tiling.num_dst_cols;
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * y_size));
    param.y_expected = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * y_size));
    uint32_t dst_col_offset = 0;
    for (size_t i = 0; i < N; ++i) {
      auto x_size = tiling.num_rows * param.tiling.num_srcs_cols[i];
      auto &x = param.x[i];
      x = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * x_size));
      size_t index = 10000 * i;
      const auto src_cols = param.tiling.num_srcs_cols[i];
      for (uint32_t row = 0; row < param.tiling.num_rows; ++row) {
        for (uint32_t col = 0; col < src_cols; ++col) {
          ++index;
          x[row * param.tiling.num_srcs_cols[i] + col] = index;
          param.y_expected[row * param.tiling.num_dst_cols + dst_col_offset + col] = index;
        }
      }
      dst_col_offset += src_cols;
    }
  }

  template <typename T, size_t N>
  static void FreeTensorInput(TestConcatParam<T, N> &param) {
    AscendC::GmFree(param.y);
    AscendC::GmFree(param.y_expected);
    for (size_t i = 0; i < N; ++i) {
      AscendC::GmFree(param.x[i]);
    }
  }
  template <typename T>
  static uint32_t Valid(T *y, T *exp, size_t comp_size) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < comp_size; i++) {
      if (y[i] != exp[i]) {
        diff_count++;
      }
    }
    return diff_count;
  }

  template <typename T, size_t N>
  static void ConcatTest(uint32_t rows, const std::vector<uint32_t> &src_cols) {
    TestConcatParam<T, N> param{};
    param.tiling.num_rows = rows;
    for (size_t i = 0; i < N; ++i) {
      param.tiling.num_srcs_cols[i] = src_cols[i];
      param.tiling.num_dst_cols += src_cols[i];
    }
    CreateTensorInput(param);

    // 构造Api调用函数
    auto kernel = [&param] { InvokeKernel(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    // 验证结果
    const uint32_t diff_count = Valid(param.y, param.y_expected, param.tiling.num_rows * param.tiling.num_dst_cols);
    EXPECT_EQ(diff_count, 0);
    FreeTensorInput(param);
  }
};

TEST_F(RegbaseApiConcatTest, ConcatSuccess) {
  ConcatTest<uint32_t, 2>(16, {7, 7});
  ConcatTest<uint16_t, 2>(16, {31, 31});
  ConcatTest<uint8_t, 2>(16, {63, 63});
  ConcatTest<uint16_t, 3>(4, {63, 63, 63});
  ConcatTest<uint8_t, 2>(16, {31, 33});
  ConcatTest<uint8_t, 2>(16, {255, 257});
}

TEST_F(RegbaseApiConcatTest, GenMaskRegPreservesHighBits) {
  constexpr uint32_t kMaskWordCount = 8;
  constexpr uint32_t kExpectedMask = 0x000fffffU;
  auto output = static_cast<uint32_t *>(AscendC::GmAlloc(kMaskWordCount * sizeof(uint32_t)));
  auto kernel = [output] {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> index_buf;
    tpipe.InitBuffer(index_buf, kMaskWordCount * sizeof(uint32_t));
    auto index = index_buf.Get<uint32_t>();

    concat::GenMaskReg<uint16_t>(16, 16, 10, index.GetPhyAddr());
    UbToGm(output, index, kMaskWordCount);
  };

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1);

  for (uint32_t i = 0; i < kMaskWordCount; ++i) {
    EXPECT_EQ(output[i], kExpectedMask);
  }
  AscendC::GmFree(output);
}
