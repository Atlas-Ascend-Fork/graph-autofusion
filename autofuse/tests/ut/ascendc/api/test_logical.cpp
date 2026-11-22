/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_logical.cpp
 * \brief
 */

#include "tikicpulib.h"
#include "kernel_operator.h"

#include <kernel_tpipe.h>
#include <algorithm>
#include <cstdlib>

#include "gtest/gtest.h"

using namespace AscendC;
#include "utils.h"
#include "logical.h"

template <class T>
void GmToUb(LocalTensor<T> &local, T *gm, int size) {
  for (int i = 0; i < size; i++) {
    local.SetValue(i, gm[i]);
  }
}

template <class T>
void UbToGm(T *gm, LocalTensor<T> &local, int size) {
  for (int i = 0; i < size; i++) {
    gm[i] = local.GetValue(i);
  }
}

template <typename T>
void InitParams(LocalTensor<T> &l_x1, LocalTensor<T> &l_x2, LocalTensor<uint8_t> &l_y, LocalTensor<uint8_t> &l_tmp,
                uint32_t size) {
  TPipe pipe;
  TBuf<TPosition::VECCALC> x1_buf, x2_buf, y_buf, tmp_buf;
  pipe.InitBuffer(x1_buf, sizeof(T) * size);
  pipe.InitBuffer(x2_buf, sizeof(T) * size);
  pipe.InitBuffer(y_buf, sizeof(uint8_t) * size);
  constexpr int tmp_size = 8 * 1024;
  pipe.InitBuffer(tmp_buf, tmp_size);

  l_x1 = x1_buf.Get<T>();
  l_x2 = x2_buf.Get<T>();
  l_y = y_buf.Get<uint8_t>();
  l_tmp = tmp_buf.Get<uint8_t>();
}

template <typename T>
void TestLogicalCommon(const std::string &logical, uint32_t size) {
  ASSERT_TRUE(logical == "or" || logical == "and");

  auto *x1 = static_cast<T *>(GmAlloc(sizeof(T) * size));
  auto *x2 = static_cast<T *>(GmAlloc(sizeof(T) * size));
  auto *y = static_cast<uint8_t *>(GmAlloc(sizeof(T) * size));

  uint8_t expect[size];

  for (int i = 0; i < size; i++) {
    T tmp1 = std::is_same_v<T, int64_t> ? static_cast<T>(std::rand() % 21 - 10) : static_cast<T>(std::rand() % 2);
    x1[i] = tmp1;

    T tmp2 = std::is_same_v<T, int64_t> ? static_cast<T>(std::rand() % 21 - 10) : static_cast<T>(std::rand() % 2);
    x2[i] = tmp2;
    if (logical == "or") {
      expect[i] = static_cast<uint8_t>((static_cast<float>(tmp1) != 0.0f) || (static_cast<float>(tmp2) != 0.0f));
    } else {
      expect[i] = static_cast<uint8_t>((static_cast<float>(tmp1) != 0.0f) && (static_cast<float>(tmp2) != 0.0f));
    }
  }

  auto kernel = [](uint32_t size, T *x1, T *x2, uint8_t *y, const std::string &logical) {
    LocalTensor<T> l_x1;
    LocalTensor<T> l_x2;
    LocalTensor<uint8_t> l_y;
    LocalTensor<uint8_t> l_tmp;

    InitParams(l_x1, l_x2, l_y, l_tmp, size);

    GmToUb(l_x1, x1, size);
    GmToUb(l_x2, x2, size);
    GmToUb(l_y, y, size);

    if (logical == "or") {
      LogicalOr(l_y, l_x1, l_x2, l_tmp, size);
    } else {
      LogicalAnd(l_y, l_x1, l_x2, l_tmp, size);
    }

    UbToGm(y, l_y, size);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, size, x1, x2, y, logical);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < size; i++) {
    if (y[i] != expect[i]) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0);
  GmFree(x1);
  GmFree(x2);
  GmFree(y);
}

template <typename T>
void InitParamsScalarExtend(LocalTensor<T> &l_x1, LocalTensor<uint8_t> &l_y, LocalTensor<uint8_t> &l_tmp,
                            uint32_t size) {
  TPipe pipe;
  TBuf<TPosition::VECCALC> x1_buf, y_buf, tmp_buf;
  pipe.InitBuffer(x1_buf, sizeof(T) * size);
  pipe.InitBuffer(y_buf, sizeof(uint8_t) * size);
  constexpr int tmp_size = 8 * 1024;
  pipe.InitBuffer(tmp_buf, tmp_size);

  l_x1 = x1_buf.Get<T>();
  l_y = y_buf.Get<uint8_t>();
  l_tmp = tmp_buf.Get<uint8_t>();
}

template <typename T>
void TestLogicalCommonScalarExtend(const std::string &logical, uint32_t size) {
  ASSERT_TRUE(logical == "orScalarExtend" || logical == "andScalarExtend");

  auto *x1 = static_cast<T *>(GmAlloc(sizeof(T) * size));
  auto *y = static_cast<uint8_t *>(GmAlloc(sizeof(T) * size));

  uint8_t expect[size];

  T tmp2 = std::is_same_v<T, int64_t> ? static_cast<T>(std::rand() % 21 - 10) : static_cast<T>(std::rand() % 2);
  auto x2 = tmp2;
  for (int i = 0; i < size; i++) {
    T tmp1 = std::is_same_v<T, int64_t> ? static_cast<T>(std::rand() % 21 - 10) : static_cast<T>(std::rand() % 2);
    x1[i] = tmp1;

    if (logical == "orScalarExtend") {
      expect[i] = static_cast<uint8_t>((static_cast<float>(tmp1) != 0.0f) || (static_cast<float>(tmp2) != 0.0f));
    } else {
      expect[i] = static_cast<uint8_t>((static_cast<float>(tmp1) != 0.0f) && (static_cast<float>(tmp2) != 0.0f));
    }
  }

  auto kernel = [](uint32_t size, T *x1, T x2, uint8_t *y, const std::string &logical) {
    LocalTensor<T> l_x1;
    LocalTensor<uint8_t> l_y;
    LocalTensor<uint8_t> l_tmp;

    InitParamsScalarExtend(l_x1, l_y, l_tmp, size);

    GmToUb(l_x1, x1, size);
    GmToUb(l_y, y, size);

    T l_x2 = x2;
    if (logical == "orScalarExtend") {
      LogicalOrScalarExtend(l_y, l_x1, l_x2, l_tmp, size);
    } else {
      LogicalAndScalarExtend(l_y, l_x1, l_x2, l_tmp, size);
    }

    UbToGm(y, l_y, size);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, size, x1, x2, y, logical);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < size; i++) {
    if (y[i] != expect[i]) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0);
  GmFree(x1);
  GmFree(y);
}

template <typename T>
void InitParamsUnalign(LocalTensor<T> &l_x1, LocalTensor<T> &l_x2, LocalTensor<uint8_t> &l_y,
                       LocalTensor<uint8_t> &l_tmp, uint32_t size) {
  TPipe pipe;
  TBuf<TPosition::VECCALC> x1_buf, x2_buf, y_buf, tmp_buf;
  pipe.InitBuffer(x1_buf, sizeof(T) * size);
  pipe.InitBuffer(x2_buf, sizeof(T) * size);
  pipe.InitBuffer(y_buf, sizeof(uint8_t) * size);
  constexpr int tmp_size = 16928;  // 任意非对齐长度
  pipe.InitBuffer(tmp_buf, tmp_size);

  l_x1 = x1_buf.Get<T>();
  l_x2 = x2_buf.Get<T>();
  l_y = y_buf.Get<uint8_t>();
  l_tmp = tmp_buf.Get<uint8_t>();
}

template <typename T>
void TestLogicalCommonUnalign(const std::string &logical, uint32_t size) {
  ASSERT_TRUE(logical == "or" || logical == "and");

  auto *x1 = static_cast<T *>(GmAlloc(sizeof(T) * size));
  auto *x2 = static_cast<T *>(GmAlloc(sizeof(T) * size));
  auto *y = static_cast<uint8_t *>(GmAlloc(sizeof(T) * size));

  uint8_t expect[size];

  for (int i = 0; i < size; i++) {
    T tmp1 = std::rand() % 2;
    x1[i] = tmp1;

    T tmp2 = std::rand() % 2;
    x2[i] = tmp2;
    if (logical == "or") {
      expect[i] = std::max(tmp1, tmp2);
    } else {
      expect[i] = tmp1 * tmp2;
    }
  }

  auto kernel = [](uint32_t size, T *x1, T *x2, uint8_t *y, const std::string &logical) {
    LocalTensor<T> l_x1;
    LocalTensor<T> l_x2;
    LocalTensor<uint8_t> l_y;
    LocalTensor<uint8_t> l_tmp;

    InitParamsUnalign(l_x1, l_x2, l_y, l_tmp, size);

    GmToUb(l_x1, x1, size);
    GmToUb(l_x2, x2, size);
    GmToUb(l_y, y, size);

    if (logical == "or") {
      LogicalOr(l_y, l_x1, l_x2, l_tmp, size);
    } else {
      LogicalAnd(l_y, l_x1, l_x2, l_tmp, size);
    }

    UbToGm(y, l_y, size);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, size, x1, x2, y, logical);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < size; i++) {
    if (y[i] != expect[i]) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0);
  GmFree(x1);
  GmFree(x2);
  GmFree(y);
}

// todo int4b_t类型有问题后续补充
TEST(TestApiLogical, Test_1_blk) {
  // 1 block
  uint32_t size = ONE_BLK_HALF_NUM;
  TestLogicalCommon<float>("or", size);
  TestLogicalCommon<half>("or", size);
  TestLogicalCommon<uint8_t>("or", size);
  TestLogicalCommon<int8_t>("or", size);
  TestLogicalCommon<int16_t>("or", size);
  TestLogicalCommon<int32_t>("or", size);
  TestLogicalCommon<int64_t>("or", size);

  TestLogicalCommon<float>("and", size);
  TestLogicalCommon<half>("and", size);
  TestLogicalCommon<uint8_t>("and", size);
  TestLogicalCommon<int8_t>("and", size);
  TestLogicalCommon<int16_t>("and", size);
  TestLogicalCommon<int32_t>("and", size);
  TestLogicalCommon<int64_t>("and", size);
}

TEST(TestApiLogical, Test_1_repeat) {
  // 1 repeat
  uint32_t size = ONE_REPEAT_HALF_SIZE;
  TestLogicalCommon<float>("or", size);
  TestLogicalCommon<half>("or", size);
  TestLogicalCommon<uint8_t>("or", size);
  TestLogicalCommon<int8_t>("or", size);
  TestLogicalCommon<int16_t>("or", size);
  TestLogicalCommon<int32_t>("or", size);
  TestLogicalCommon<int64_t>("or", size);

  TestLogicalCommon<float>("and", size);
  TestLogicalCommon<half>("and", size);
  TestLogicalCommon<uint8_t>("and", size);
  TestLogicalCommon<int8_t>("and", size);
  TestLogicalCommon<int16_t>("and", size);
  TestLogicalCommon<int32_t>("and", size);
  TestLogicalCommon<int64_t>("and", size);
}

// todo float int32 int64会超出测试缓存区空间后续补充用例
TEST(TestApiLogical, Test_max_repeat) {
  // max repeat
  uint32_t size = MAX_REPEAT_TIMES * ONE_REPEAT_HALF_SIZE;
  TestLogicalCommon<half>("or", size);
  TestLogicalCommon<uint8_t>("or", size);
  TestLogicalCommon<int8_t>("or", size);
  TestLogicalCommon<int16_t>("or", size);

  TestLogicalCommon<half>("and", size);
  TestLogicalCommon<uint8_t>("and", size);
  TestLogicalCommon<int8_t>("and", size);
  TestLogicalCommon<int16_t>("and", size);
}

TEST(TestApiLogical, Test_less_1_blk) {
  // less than 1 block
  uint32_t size = ONE_BLK_HALF_NUM - 1;
  TestLogicalCommon<float>("or", size);
  TestLogicalCommon<half>("or", size);
  TestLogicalCommon<uint8_t>("or", size);
  TestLogicalCommon<int8_t>("or", size);
  TestLogicalCommon<int16_t>("or", size);
  TestLogicalCommon<int32_t>("or", size);
  TestLogicalCommon<int64_t>("or", size);

  TestLogicalCommon<float>("and", size);
  TestLogicalCommon<half>("and", size);
  TestLogicalCommon<uint8_t>("and", size);
  TestLogicalCommon<int8_t>("and", size);
  TestLogicalCommon<int16_t>("and", size);
  TestLogicalCommon<int32_t>("and", size);
  TestLogicalCommon<int64_t>("and", size);
}

TEST(TestApiLogical, Test_less_1_repeat) {
  // less than 1 repeat
  uint32_t size = ONE_REPEAT_HALF_SIZE - 1;
  TestLogicalCommon<float>("or", size);
  TestLogicalCommon<half>("or", size);
  TestLogicalCommon<uint8_t>("or", size);
  TestLogicalCommon<int8_t>("or", size);
  TestLogicalCommon<int16_t>("or", size);
  TestLogicalCommon<int32_t>("or", size);
  TestLogicalCommon<int64_t>("or", size);

  TestLogicalCommon<float>("and", size);
  TestLogicalCommon<half>("and", size);
  TestLogicalCommon<uint8_t>("and", size);
  TestLogicalCommon<int8_t>("and", size);
  TestLogicalCommon<int16_t>("and", size);
  TestLogicalCommon<int32_t>("and", size);
  TestLogicalCommon<int64_t>("and", size);
}

// todo 同max repeat
TEST(TestApiLogical, Test_less_max_repeat) {
  // less than max repeat
  uint32_t size = (MAX_REPEAT_TIMES - 1) * ONE_REPEAT_HALF_SIZE;
  TestLogicalCommon<half>("or", size);
  TestLogicalCommon<uint8_t>("or", size);
  TestLogicalCommon<int8_t>("or", size);
  TestLogicalCommon<int16_t>("or", size);

  TestLogicalCommon<half>("and", size);
  TestLogicalCommon<uint8_t>("and", size);
  TestLogicalCommon<int8_t>("and", size);
  TestLogicalCommon<int16_t>("and", size);
}

// todo 同max repeat
TEST(TestApiLogical, Test_mix) {
  // mix block, repeat, max repeat
  uint32_t size = MAX_REPEAT_TIMES * ONE_REPEAT_HALF_SIZE + ONE_REPEAT_HALF_SIZE + ONE_BLK_HALF_NUM;
  TestLogicalCommon<half>("or", size);
  TestLogicalCommon<uint8_t>("or", size);
  TestLogicalCommon<int8_t>("or", size);
  TestLogicalCommon<int16_t>("or", size);

  TestLogicalCommon<half>("and", size);
  TestLogicalCommon<uint8_t>("and", size);
  TestLogicalCommon<int8_t>("and", size);
  TestLogicalCommon<int16_t>("and", size);
}

// --------------------- 输入是一个tensor和一个scalar相关的基础测试 -------------------------------//
// todo int4b_t类型有问题后续补充
TEST(TestApiLogical, Test_1_blk_scalar_extend) {
  // 1 block
  uint32_t size = ONE_BLK_HALF_NUM;
  TestLogicalCommonScalarExtend<float>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<half>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<uint8_t>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<int8_t>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<int16_t>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<int32_t>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<int64_t>("orScalarExtend", size);

  TestLogicalCommonScalarExtend<float>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<half>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<uint8_t>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<int8_t>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<int16_t>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<int32_t>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<int64_t>("andScalarExtend", size);
}

TEST(TestApiLogical, Test_1_repeat_scalar_extend) {
  // 1 repeat
  uint32_t size = ONE_REPEAT_HALF_SIZE;
  TestLogicalCommonScalarExtend<float>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<half>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<uint8_t>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<int8_t>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<int16_t>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<int32_t>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<int64_t>("orScalarExtend", size);

  TestLogicalCommonScalarExtend<float>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<half>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<uint8_t>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<int8_t>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<int16_t>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<int32_t>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<int64_t>("andScalarExtend", size);
}

// todo float int32 int64会超出测试缓存区空间后续补充用例
TEST(TestApiLogical, Test_max_repeat_scalar_extend) {
  // max repeat
  uint32_t size = MAX_REPEAT_TIMES * ONE_REPEAT_HALF_SIZE;
  TestLogicalCommonScalarExtend<half>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<uint8_t>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<int8_t>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<int16_t>("orScalarExtend", size);

  TestLogicalCommonScalarExtend<half>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<uint8_t>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<int8_t>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<int16_t>("andScalarExtend", size);
}

TEST(TestApiLogical, Test_less_1_blk_scalar_extend) {
  // less than 1 block
  uint32_t size = ONE_BLK_HALF_NUM - 1;
  TestLogicalCommonScalarExtend<float>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<half>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<uint8_t>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<int8_t>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<int16_t>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<int32_t>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<int64_t>("orScalarExtend", size);

  TestLogicalCommonScalarExtend<float>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<half>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<uint8_t>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<int8_t>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<int16_t>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<int32_t>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<int64_t>("andScalarExtend", size);
}

TEST(TestApiLogical, Test_less_1_repeat_scalar_extend) {
  // less than 1 repeat
  uint32_t size = ONE_REPEAT_HALF_SIZE - 1;
  TestLogicalCommonScalarExtend<float>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<half>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<uint8_t>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<int8_t>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<int16_t>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<int32_t>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<int64_t>("orScalarExtend", size);

  TestLogicalCommonScalarExtend<float>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<half>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<uint8_t>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<int8_t>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<int16_t>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<int32_t>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<int64_t>("andScalarExtend", size);
}

// todo 同max repeat
TEST(TestApiLogical, Test_less_max_repeat_scalar_extend) {
  // less than max repeat
  uint32_t size = (MAX_REPEAT_TIMES - 1) * ONE_REPEAT_HALF_SIZE;
  TestLogicalCommonScalarExtend<half>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<uint8_t>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<int8_t>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<int16_t>("orScalarExtend", size);

  TestLogicalCommonScalarExtend<half>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<uint8_t>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<int8_t>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<int16_t>("andScalarExtend", size);
}

// todo 同max repeat
TEST(TestApiLogical, Test_mix_scalar_extend) {
  // mix block, repeat, max repeat
  uint32_t size = MAX_REPEAT_TIMES * ONE_REPEAT_HALF_SIZE + ONE_REPEAT_HALF_SIZE + ONE_BLK_HALF_NUM;
  TestLogicalCommonScalarExtend<half>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<uint8_t>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<int8_t>("orScalarExtend", size);
  TestLogicalCommonScalarExtend<int16_t>("orScalarExtend", size);

  TestLogicalCommonScalarExtend<half>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<uint8_t>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<int8_t>("andScalarExtend", size);
  TestLogicalCommonScalarExtend<int16_t>("andScalarExtend", size);
}

// --------------------- tmp_buf使用ub不对齐相关的基础测试 -------------------------------//
// todo int4b_t类型有问题后续补充
TEST(TestApiLogical, Test_1_blk_unalign) {
  uint32_t size = 6160;  // 任意非对齐长度
  TestLogicalCommonUnalign<float>("or", size);
  TestLogicalCommonUnalign<half>("or", size);
  TestLogicalCommonUnalign<uint8_t>("or", size);
  TestLogicalCommonUnalign<int8_t>("or", size);
  TestLogicalCommonUnalign<int16_t>("or", size);
  TestLogicalCommonUnalign<int32_t>("or", size);
  TestLogicalCommonUnalign<int64_t>("or", size);

  TestLogicalCommonUnalign<float>("and", size);
  TestLogicalCommonUnalign<half>("and", size);
  TestLogicalCommonUnalign<uint8_t>("and", size);
  TestLogicalCommonUnalign<int8_t>("and", size);
  TestLogicalCommonUnalign<int16_t>("and", size);
  TestLogicalCommonUnalign<int32_t>("and", size);
  TestLogicalCommonUnalign<int64_t>("and", size);
}
