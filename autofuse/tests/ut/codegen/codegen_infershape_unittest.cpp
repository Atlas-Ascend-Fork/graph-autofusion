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
#include "node_utils_ex.h"
#include "graph_utils.h"
#include "codegen_infershape.h"
#include "codegen.h"
#include <string>

using namespace ge;

class UTestCodegenInfershape : public testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST(UTestCodegenInfershape, TestInfershapeFuncWithLambda) {
  vector<vector<std::string>> symbol_shape_str{{"s0 + s1", "s1 * s2", "s2 - s1"}};
  codegen::InfershapeGen testGen;
  std::string s0_source = R"([&]() -> int64_t {
    auto *tensor = context->GetGraphInputTensor(0);
    if (tensor == nullptr) {
      return gert::Shape::kInvalidDimValue;
    }
    return tensor->GetOriginShape().GetDim(1);
  }())";
  std::string s1_source = R"([&]() -> int64_t {
    auto *tensor = context->GetGraphInputTensor(1);
    if (tensor == nullptr) {
      return gert::Shape::kInvalidDimValue;
    }
    return tensor->GetOriginShape().GetDim(1);
  }())";
  std::string s2_source = R"([&]() -> int64_t {
    auto *tensor = context->GetGraphInputTensor(2);
    if (tensor == nullptr) {
      return gert::Shape::kInvalidDimValue;
    }
    return tensor->GetOriginShape().GetDim(1);
  }())";

  std::map<std::string, std::string> shape_info = {{"s0", s0_source}, {"s1", s1_source}, {"s2", s2_source}};
  std::string code = testGen.GenInferShapeFunc(symbol_shape_str, shape_info);
  std::string expect = R"(
#include <cmath>
#include <type_traits>
#include <unordered_map>
#include "exe_graph/runtime/infer_shape_context.h"

namespace {
#define Max(a, b) ((a) > (b) ? (a) : (b))
#define Min(a, b) ((a) < (b) ? (a) : (b))
#define Pow(a, b) (std::pow(a, b))
#define Exp(a) (std::exp(a))
#define Log(a) (std::log(a))
#define Sqrt(a) (std::sqrt(a))
#define Ceiling(a) (std::ceil(a))
#define Floor(a) (std::floor(a))
#define Abs(a) (std::abs(a))
#define Rational(a, b) (static_cast<double>(a) / static_cast<double>(b))

const double kThreshold = 0.00001;

template <typename Ta, typename Tb>
auto Mod(Ta left, Tb right) -> decltype(left % right) {
  return left % right;
}

// 针对浮点数的特化版本（使用 std::fmod）
template <typename Ta, typename Tb>
auto Mod(Ta left, Tb right) ->
    typename std::enable_if<std::is_floating_point<Ta>::value || std::is_floating_point<Tb>::value,
                            decltype(std::fmod(left, right))>::type {
  return std::fmod(left, right);
}

class InferShapeSymbolEvalContext : public gert::InferShapeContext {
 public:
  const gert::Tensor *GetGraphInputTensor(size_t data_index) const {
    auto *tensor = GetInputPointer<gert::Tensor>(data_index + 1);
    if (tensor == nullptr) {
      return nullptr;
    }
    return tensor;
  }
};
static_assert(std::is_standard_layout<InferShapeSymbolEvalContext>::value,
              "The class InferShapeSymbolEvalContext must be a POD");
} // namespace

extern "C" ge::graphStatus InferShape(InferShapeSymbolEvalContext *context)
{
  auto s0 = [&]() -> int64_t {
    auto *tensor = context->GetGraphInputTensor(0);
    if (tensor == nullptr) {
      return gert::Shape::kInvalidDimValue;
    }
    return tensor->GetOriginShape().GetDim(1);
  }();
  auto s1 = [&]() -> int64_t {
    auto *tensor = context->GetGraphInputTensor(1);
    if (tensor == nullptr) {
      return gert::Shape::kInvalidDimValue;
    }
    return tensor->GetOriginShape().GetDim(1);
  }();
  auto s2 = [&]() -> int64_t {
    auto *tensor = context->GetGraphInputTensor(2);
    if (tensor == nullptr) {
      return gert::Shape::kInvalidDimValue;
    }
    return tensor->GetOriginShape().GetDim(1);
  }();

  context->GetOutputShape(0)->SetDimNum(0);
  context->GetOutputShape(0)->AppendDim(s0 + s1);
  context->GetOutputShape(0)->AppendDim(s1 * s2);
  context->GetOutputShape(0)->AppendDim(s2 - s1);
  return ge::GRAPH_SUCCESS;
}
)";
  EXPECT_EQ(code, expect);
}

TEST(UTestCodegenInfershape, Test_WithRational) {
  codegen::CodegenOptions opt;
  codegen::Codegen codegen(opt);
  vector<vector<std::string>> symbol_shape_str{
      {"(Rational(1 , 1800000) * s5 * s6 * s7)", "(Rational(1 , 1800000) * s5)"}};
  std::map<std::string, std::string> shape_info;
  shape_info["s5"] = "3000";
  shape_info["s6"] = "300";
  shape_info["s7"] = "4";

  std::string code = codegen.GenerateInferShape(symbol_shape_str, shape_info);

  auto expect_code = R"(extern "C" ge::graphStatus InferShape(InferShapeSymbolEvalContext *context)
{
  auto s5 = 3000;
  auto s6 = 300;
  auto s7 = 4;

  context->GetOutputShape(0)->SetDimNum(0);
  // 表达式中包含Rational, 结果可能是浮点数, 强转成整形会舍去小数部分导致结果错误, 因此要进行四舍五入处理
  double expr_value_0 = (Rational(1 , 1800000) * s5 * s6 * s7);
  int64_t round_value_0 = std::round(expr_value_0);
  // 对损失的小数部分做校验, 小于设定的阈值才认为计算成功
  if ((fabs(expr_value_0 - static_cast<double>(round_value_0)) > kThreshold)) {
    return ge::GRAPH_FAILED;
  }
  context->GetOutputShape(0)->AppendDim(round_value_0);
  // 表达式中包含Rational, 结果可能是浮点数, 强转成整形会舍去小数部分导致结果错误, 因此要进行四舍五入处理
  double expr_value_1 = (Rational(1 , 1800000) * s5);
  int64_t round_value_1 = std::round(expr_value_1);
  // 对损失的小数部分做校验, 小于设定的阈值才认为计算成功
  if ((fabs(expr_value_1 - static_cast<double>(round_value_1)) > kThreshold)) {
    return ge::GRAPH_FAILED;
  }
  context->GetOutputShape(0)->AppendDim(round_value_1);
  return ge::GRAPH_SUCCESS;
}
)";

  std::string func_start = "extern \"C\" ge::graphStatus InferShape(InferShapeSymbolEvalContext *context)";
  auto func_pos = code.find(func_start);
  auto gen_code = code.substr(func_pos, std::string(expect_code).size());
  ASSERT_EQ(gen_code, expect_code);
}
