/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * test_bitwise_and.cpp
 */

#include <cmath>
#include <random>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "api_regbase/bitwise_and.h"

using namespace AscendC;

namespace af {

class TestRegbaseApiBitwiseandUT : public testing::Test {
 protected:
  // Tensor - Tensor 场景
  template <typename T>
  static void InvokeTensorTensorKernel(BinaryInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, x2buf, ybuf;
    tpipe.InitBuffer(x1buf, sizeof(T) * param.size);
    tpipe.InitBuffer(x2buf, sizeof(T) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(T)));

    LocalTensor<T> l_y = ybuf.Get<T>();
    LocalTensor<T> l_x1 = x1buf.Get<T>();
    LocalTensor<T> l_x2 = x2buf.Get<T>();

    GmToUb(l_x1, param.x1, param.size);
    GmToUb(l_x2, param.x2, param.size);
    BitwiseAnd(l_y, l_x1, l_x2, param.size);
    UbToGm(param.y, l_y, param.size);
  }

  template <typename T>
  static void CreateTensorInput(BinaryInputParam<T> &param) {
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x2 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    std::mt19937 eng(1);
    int input_range = 100;

    for (uint32_t i = 0; i < param.size; i++) {
      if constexpr (std::is_same_v<T, float>) {
        std::uniform_real_distribution<float> distr(-50.0f, 50.0f);
        param.x1[i] = distr(eng);
        param.x2[i] = distr(eng) + 1.0f;  // 避免除零
        param.exp[i] = param.x1[i] & param.x2[i];
      } else if constexpr (std::is_same_v<T, int64_t>) {
        std::uniform_int_distribution<int64_t> distr(-10000, 10000);
        param.x1[i] = distr(eng);
        param.x2[i] = distr(eng) + 1;
        param.exp[i] = param.x1[i] & param.x2[i];
      } else {
        std::uniform_int_distribution<int> distr(-input_range, input_range);
        param.x1[i] = static_cast<T>(distr(eng));
        param.x2[i] = static_cast<T>(distr(eng) + 1);
        param.exp[i] = param.x1[i] & param.x2[i];
      }
    }
  }

  template <typename T>
  static void FreeTensorInput(BinaryInputParam<T> &param) {
    AscendC::GmFree(param.y);
    AscendC::GmFree(param.exp);
    AscendC::GmFree(param.x2);
    AscendC::GmFree(param.x1);
  }
  // Tensor - Tensor 测试
  template <typename T>
  static void BitwiseandTensorTensorTest(uint32_t size) {
    BinaryInputParam<T> param{};
    param.size = size;
    CreateTensorInput(param);

    // 构造Api调用函数
    auto kernel = [&param] { InvokeTensorTensorKernel(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param.y, param.exp, param.size);
    EXPECT_EQ(diff_count, 0);
    FreeTensorInput(param);
  }

  // bool 与 uint8 位模式一致：数据经 uint8 tensor 装载后以 bool 视图传入 BitwiseAndExtend，
  // 规避部分模拟器对 LocalTensor<bool> 装载路径的差异；API 仍按 bool 实例化
  static void InvokeBoolTensorTensorKernel(BinaryInputParam<uint8_t> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, x2buf, ybuf;
    tpipe.InitBuffer(x1buf, sizeof(uint8_t) * param.size);
    tpipe.InitBuffer(x2buf, sizeof(uint8_t) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(uint8_t) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(uint8_t)));

    auto x1U8 = x1buf.Get<uint8_t>();
    auto x2U8 = x2buf.Get<uint8_t>();
    auto yU8 = ybuf.Get<uint8_t>();
    LocalTensor<bool> l_y = reinterpret_cast<const LocalTensor<bool> &>(yU8);
    LocalTensor<bool> l_x1 = reinterpret_cast<const LocalTensor<bool> &>(x1U8);
    LocalTensor<bool> l_x2 = reinterpret_cast<const LocalTensor<bool> &>(x2U8);

    GmToUb(x1U8, param.x1, param.size);
    GmToUb(x2U8, param.x2, param.size);
    BitwiseAndExtend(l_y, l_x1, l_x2, param.size);
    UbToGm(param.y, yU8, param.size);
  }

  // bool 输入：BitwiseAndExtend 内部转 uint8 视图调用 BitwiseAnd，0/1 域按位与等价逐元素与
  static void BitwiseandBoolTest(uint32_t size) {
    BinaryInputParam<uint8_t> param{};
    param.size = size;
    param.y = static_cast<uint8_t *>(AscendC::GmAlloc(sizeof(uint8_t) * param.size));
    param.exp = static_cast<uint8_t *>(AscendC::GmAlloc(sizeof(uint8_t) * param.size));
    param.x1 = static_cast<uint8_t *>(AscendC::GmAlloc(sizeof(uint8_t) * param.size));
    param.x2 = static_cast<uint8_t *>(AscendC::GmAlloc(sizeof(uint8_t) * param.size));

    srand(1);
    for (uint32_t i = 0; i < param.size; i++) {
      param.x1[i] = static_cast<uint8_t>(rand() % 2);
      param.x2[i] = static_cast<uint8_t>(rand() % 2);
      param.exp[i] = param.x1[i] & param.x2[i];
    }

    auto kernel = [&param] { InvokeBoolTensorTensorKernel(param); };
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param.y, param.exp, param.size);
    EXPECT_EQ(diff_count, 0);
    FreeTensorInput(param);
  }
};

// ============ Tensor - Tensor 测试 (新增数据类型: DT_INT8, DT_INT64, DT_BF16) ============
TEST_F(TestRegbaseApiBitwiseandUT, Bitwiseand_TensorTensor_Test) {
  // int8
  BitwiseandTensorTensorTest<int8_t>(ONE_BLK_SIZE / sizeof(int8_t));
  BitwiseandTensorTensorTest<int8_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  BitwiseandTensorTensorTest<int8_t>((ONE_BLK_SIZE - sizeof(int8_t)) / sizeof(int8_t));
  BitwiseandTensorTensorTest<int8_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int8_t));
  BitwiseandTensorTensorTest<int8_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int8_t));
  BitwiseandTensorTensorTest<int8_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int8_t));

  // int64
  BitwiseandTensorTensorTest<int64_t>(ONE_BLK_SIZE / sizeof(int64_t));
  BitwiseandTensorTensorTest<int64_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int64_t));
  BitwiseandTensorTensorTest<int64_t>((ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t));
  BitwiseandTensorTensorTest<int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t));
  BitwiseandTensorTensorTest<int64_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int64_t));
  BitwiseandTensorTensorTest<int64_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int64_t));

  // uint32
  BitwiseandTensorTensorTest<uint32_t>(ONE_BLK_SIZE / sizeof(uint32_t));
  BitwiseandTensorTensorTest<uint32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(uint32_t));
  BitwiseandTensorTensorTest<uint32_t>((ONE_BLK_SIZE - sizeof(uint32_t)) / sizeof(uint32_t));
  BitwiseandTensorTensorTest<uint32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(uint32_t));
  BitwiseandTensorTensorTest<uint32_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint32_t));
  BitwiseandTensorTensorTest<uint32_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint32_t));

  // uint64
  BitwiseandTensorTensorTest<uint64_t>(ONE_BLK_SIZE / sizeof(uint64_t));
  BitwiseandTensorTensorTest<uint64_t>(ONE_REPEAT_BYTE_SIZE / sizeof(uint64_t));
  BitwiseandTensorTensorTest<uint64_t>((ONE_BLK_SIZE - sizeof(uint64_t)) / sizeof(uint64_t));
  BitwiseandTensorTensorTest<uint64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(uint64_t));
  BitwiseandTensorTensorTest<uint64_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint64_t));
  BitwiseandTensorTensorTest<uint64_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint64_t));
}

// bool 输入：regbase BitwiseAndExtend 按 uint8 视图执行（多 repeat 等场景由既有 uint8 用例覆盖）
TEST_F(TestRegbaseApiBitwiseandUT, Bitwiseand_Bool_Test) {
  BitwiseandBoolTest(ONE_BLK_SIZE / sizeof(uint8_t));
}

}  // namespace af
