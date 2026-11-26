/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "securec.h"
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
#include "removepad.h"

namespace {
template <typename T>
void CheckRemovePad(uint32_t width, uint32_t pitch, uint32_t rows) {
  if (width == 0U) {
    ADD_FAILURE() << "RemovePad test width must be positive.";
    return;
  }
  const uint32_t count = width * rows;
  const uint32_t capacity = (pitch * rows + 128) * sizeof(T);
  auto *result = static_cast<T *>(AscendC::GmAlloc(count * sizeof(T)));
  ASSERT_NE(result, nullptr);
  auto kernel = [=](T *output) {
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> src_buf, dst_buf;
    pipe.InitBuffer(src_buf, capacity);
    pipe.InitBuffer(dst_buf, capacity);
    auto src = src_buf.Get<T>();
    auto dst = dst_buf.Get<T>();
    // Raw patterns include NaNs and signed zero; RemovePad must preserve every bit.
    auto src_bits = src.template ReinterpretCast<uint16_t>();
    for (uint32_t i = 0; i < capacity / sizeof(uint16_t); ++i) {
      src_bits.SetValue(i, static_cast<uint16_t>(i * 7919U + 0x8000U));
    }
    RemovePadExtend(dst, src, width, pitch, rows);
    UbToGm(output, dst, count);
  };
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, result);
  for (uint32_t i = 0; i < count; ++i) {
    uint16_t actual[sizeof(T) / sizeof(uint16_t)]{};
    const auto copy_status = memcpy_s(actual, sizeof(actual), result + i, sizeof(T));
    if (copy_status != EOK) {
      ADD_FAILURE() << "Failed to copy RemovePad output bits, status=" << copy_status;
      AscendC::GmFree(result);
      return;
    }
    const uint32_t src_index = (i / width * pitch + i % width) * sizeof(T) / sizeof(uint16_t);
    for (uint32_t j = 0; j < sizeof(T) / sizeof(uint16_t); ++j) {
      EXPECT_EQ(actual[j], static_cast<uint16_t>((src_index + j) * 7919U + 0x8000U)) << "element=" << i;
    }
  }
  AscendC::GmFree(result);
}

template <typename T>
class RemovePadExtendTest : public testing::Test {};
using RemovePadTypes = testing::Types<int16_t, uint16_t, int32_t, uint32_t, half, float, bfloat16_t>;
TYPED_TEST_SUITE(RemovePadExtendTest, RemovePadTypes);

TYPED_TEST(RemovePadExtendTest, CompactRowsPreserveBits) {
  for (const uint32_t width : {1U, 5U, 15U, 16U, 17U, 127U, 129U, 448U}) {
    for (const uint32_t rows : {1U, 3U}) {
      CheckRemovePad<TypeParam>(width, width, rows);
    }
  }
}

TYPED_TEST(RemovePadExtendTest, PaddedRowsPreserveBits) {
  const uint32_t align = 32 / sizeof(TypeParam);
  for (const uint32_t width : {1U, 5U, 17U, 129U}) {
    CheckRemovePad<TypeParam>(width, (width + align - 1) / align * align, 3);
  }
}
}  // namespace
