/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstddef>
#include <string>

#include "gtest/gtest.h"
#include "codegen/ascendc_api_registry.h"

namespace codegen {
namespace {
size_t CountOccurrences(const std::string &text, const std::string &needle) {
  size_t count = 0;
  for (size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + needle.size())) {
    ++count;
  }
  return count;
}

// 登记表里取到的就是包装头的原文，它会被整段内联进生成 kernel，所以循环的形状在这里钉住：
// 只有一层循环、归纳变量是 uint16_t（A5 设备编译器对 `__VEC_SCOPE__` 内的循环有这条要求，CPU
// 模拟器查不出来）、两个寄存器声明在循环之前、循环里一次掩码更新、调一次寄存器标量接口、一进一出。
void ExpectScalarWrapperLoopShape(const std::string &source, const std::string &api) {
  const std::string loopHead = "for (uint16_t i = 0; i < repeat_time; i++) {";
  const std::string regTensor = "AscendC::MicroAPI::RegTensor<T>";
  const std::string updateMask = "mask = AscendC::MicroAPI::UpdateMask<T>(count);";
  const size_t loopAt = source.find(loopHead);
  ASSERT_NE(loopAt, std::string::npos);

  EXPECT_EQ(CountOccurrences(source, loopHead), 1U);
  EXPECT_EQ(CountOccurrences(source, regTensor), 2U);
  EXPECT_LT(source.find(regTensor), loopAt);
  EXPECT_EQ(CountOccurrences(source, updateMask), 1U);
  EXPECT_GT(source.find(updateMask), loopAt);
  EXPECT_EQ(CountOccurrences(source, "AscendC::MicroAPI::DataCopy("), 2U);
  EXPECT_EQ(CountOccurrences(source, "AscendC::MicroAPI::" + api + "("), 1U);
}
}  // namespace

// ui16 / ui32 的 tensor-scalar 包装（`api_regbase/scalar_add.h` 等三个头）：三个键都要从登记表
// 取得到内容，取到的是完整的包装（循环体 `*ImplVF` 加入口 `*Extend`），且内联的是寄存器标量接口。
// 键取不到时生成 kernel 里没有包装的定义，只在设备编译那一步才报未定义，所以要在这里钉住。
TEST(AscendcRegBaseApiRegisterScalarTest, RegistersCompleteScalarAddSource) {
  const auto &source = AscendCApiRegistry::GetInstance().GetFileContent("scalar_add_reg_base.h");

  ASSERT_FALSE(source.empty());
  EXPECT_NE(source.find("__aicore__ inline void AddsImplVF("), std::string::npos);
  EXPECT_NE(source.find("__aicore__ inline void AddsExtend("), std::string::npos);
  EXPECT_NE(source.find("VF_CALL<AddsImplVF<T>>("), std::string::npos);
  EXPECT_NE(source.find("AscendC::MicroAPI::Adds("), std::string::npos);
  EXPECT_EQ(source.find("#include \""), std::string::npos);
  ExpectScalarWrapperLoopShape(source, "Adds");
}

TEST(AscendcRegBaseApiRegisterScalarTest, RegistersCompleteScalarMaximumSource) {
  const auto &source = AscendCApiRegistry::GetInstance().GetFileContent("scalar_maximum_reg_base.h");

  ASSERT_FALSE(source.empty());
  EXPECT_NE(source.find("__aicore__ inline void MaxsImplVF("), std::string::npos);
  EXPECT_NE(source.find("__aicore__ inline void MaxsExtend("), std::string::npos);
  EXPECT_NE(source.find("VF_CALL<MaxsImplVF<T>>("), std::string::npos);
  EXPECT_NE(source.find("AscendC::MicroAPI::Maxs("), std::string::npos);
  EXPECT_EQ(source.find("#include \""), std::string::npos);
  ExpectScalarWrapperLoopShape(source, "Maxs");
}

TEST(AscendcRegBaseApiRegisterScalarTest, RegistersCompleteScalarMinimumSource) {
  const auto &source = AscendCApiRegistry::GetInstance().GetFileContent("scalar_minimum_reg_base.h");

  ASSERT_FALSE(source.empty());
  EXPECT_NE(source.find("__aicore__ inline void MinsImplVF("), std::string::npos);
  EXPECT_NE(source.find("__aicore__ inline void MinsExtend("), std::string::npos);
  EXPECT_NE(source.find("VF_CALL<MinsImplVF<T>>("), std::string::npos);
  EXPECT_NE(source.find("AscendC::MicroAPI::Mins("), std::string::npos);
  EXPECT_EQ(source.find("#include \""), std::string::npos);
  ExpectScalarWrapperLoopShape(source, "Mins");
}
}  // namespace codegen
