/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <cstring>
#include <limits>
#include <tuple>
#include <type_traits>
#include "program/tiling_expression_lowering.h"
namespace att {
namespace {
using Type = TilingProgramIRValueType;
using Value = TilingProgramIRValue;
using Status = TilingProgramValueStatus;
using Op = TilingProgramIROpcode;
template <typename T>
constexpr Type TypeOf() {
  if constexpr (std::is_same_v<T, bool>) {
    return Type::kBoolean;
  } else if constexpr (std::is_same_v<T, float>) {
    return Type::kFloat32;
  } else if constexpr (std::is_same_v<T, double>) {
    return Type::kFloat64;
  } else if constexpr (sizeof(T) == 1) {
    return std::is_signed_v<T> ? Type::kInt8 : Type::kUint8;
  } else if constexpr (sizeof(T) == 2) {
    return std::is_signed_v<T> ? Type::kInt16 : Type::kUint16;
  } else if constexpr (sizeof(T) == 4) {
    return std::is_signed_v<T> ? Type::kInt32 : Type::kUint32;
  } else {
    static_assert(sizeof(T) == 8);
    return std::is_signed_v<T> ? Type::kInt64 : Type::kUint64;
  }
}
template <typename T>
Value Pack(T value) {
  using Bits = std::conditional_t<
      sizeof(T) == 1, uint8_t,
      std::conditional_t<sizeof(T) == 2, uint16_t, std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>>>;
  Bits bits = 0;
  std::memcpy(&bits, &value, sizeof(T));
  return {TypeOf<T>(), bits};
}
// Frozen from original A host_impl.cpp, SHA e451f8f3f63b3f4e493155905143ec7df9ce510560e870f632916b7bc5e7eb22.
// Intentionally independent of the candidate shared helper so a helper regression changes the result.
template <typename T, typename U>
auto OriginalMin(T a, U b) {
  return static_cast<double>(a) < static_cast<double>(b) ? a : b;
}
template <typename T, typename U>
auto OriginalMax(T a, U b) {
  return static_cast<double>(a) > static_cast<double>(b) ? a : b;
}
template <typename T>
std::vector<T> Values() {
  if constexpr (std::is_same_v<T, bool>) {
    return {false, true};
  } else if constexpr (std::is_floating_point_v<T>) {
    return {T(0.0),
            T(-0.0),
            T(-1.5),
            T(1.5),
            std::numeric_limits<T>::lowest(),
            std::numeric_limits<T>::max(),
            std::numeric_limits<T>::infinity(),
            -std::numeric_limits<T>::infinity(),
            std::numeric_limits<T>::quiet_NaN(),
            T(9007199254740992.0)};
  } else {
    std::vector<T> values{T(0), T(1), std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max()};
    if constexpr (std::is_signed_v<T>) {
      values.push_back(T(-1));
    }
    if constexpr (sizeof(T) == 8) {
      values.push_back(T(9007199254740993ULL));
    }
    return values;
  }
}
template <typename L, typename R>
void CheckPair() {
  SCOPED_TRACE(static_cast<int>(TypeOf<L>()));
  SCOPED_TRACE(static_cast<int>(TypeOf<R>()));
  Type result;
  ASSERT_EQ(GetTilingProgramIRArithmeticType(TypeOf<L>(), TypeOf<R>(), &result), Status::kOk);
  EXPECT_EQ(result, TypeOf<decltype(L{} + R{})>());
  for (const auto op : {Op::kMin, Op::kMax}) {
    ASSERT_EQ(GetTilingProgramIRBinaryType(op, TypeOf<L>(), TypeOf<R>(), &result), Status::kOk);
    EXPECT_EQ(result, (TypeOf<decltype(OriginalMin(L{}, R{}))>()));
    for (const L left : Values<L>()) {
      for (const R right : Values<R>()) {
        const auto expected = op == Op::kMin ? Pack(OriginalMin(left, right)) : Pack(OriginalMax(left, right));
        auto a = Pack(left), b = Pack(right);
        Value actual{Type::kUint8, 42};
        ASSERT_EQ(EvaluateTilingProgramIRBinary(op, a, b, &actual), Status::kOk);
        EXPECT_EQ(actual.type, expected.type);
        EXPECT_EQ(actual.raw_bits, expected.raw_bits);
        ASSERT_EQ(EvaluateTilingProgramIRBinary(op, a, b, &a), Status::kOk);
        EXPECT_EQ(a.type, expected.type);
        EXPECT_EQ(a.raw_bits, expected.raw_bits);
      }
    }
  }
}
using Types = std::tuple<bool, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float, double>;
template <typename L, size_t... I>
void CheckRow(std::index_sequence<I...>) {
  (CheckPair<L, std::tuple_element_t<I, Types>>(), ...);
}
template <size_t... I>
void CheckAll(std::index_sequence<I...> sequence) {
  (CheckRow<std::tuple_element_t<I, Types>>(sequence), ...);
}
}  // namespace
TEST(TilingProgramMixedHelperTest, AllTypePairsMatchOriginalNativeHelpersAndArithmeticTypes) {
  CheckAll(std::make_index_sequence<std::tuple_size_v<Types>>{});
}
TEST(TilingProgramMixedHelperTest, InvalidInputsPreserveOutputAndOrdinaryArithmeticRequiresCasts) {
  const auto invalid = static_cast<Type>(255);
  Type type = Type::kFloat32;
  EXPECT_EQ(GetTilingProgramIRArithmeticType(invalid, Type::kInt64, &type), Status::kUnsupportedType);
  EXPECT_EQ(type, Type::kFloat32);
  EXPECT_EQ(GetTilingProgramIRArithmeticType(Type::kInt64, Type::kInt64, nullptr), Status::kInvalidArgument);
  EXPECT_EQ(GetTilingProgramIRBinaryType(Op::kMin, invalid, Type::kInt64, &type), Status::kUnsupportedType);
  EXPECT_EQ(type, Type::kFloat32);
  Value output{Type::kInt8, 23};
  EXPECT_EQ(EvaluateTilingProgramIRBinary(Op::kMax, {Type::kBoolean, 2}, Pack(1), &output), Status::kInvalidValue);
  EXPECT_EQ(output.type, Type::kInt8);
  EXPECT_EQ(output.raw_bits, 23U);
  EXPECT_EQ(EvaluateTilingProgramIRBinary(Op::kAdd, Pack(int64_t(1)), Pack(uint64_t(2)), &output),
            Status::kInvalidArgument);
  EXPECT_EQ(output.raw_bits, 23U);
}
TEST(TilingProgramMixedHelperTest, VerifierAndLoweringKeepOriginalTypesThroughSelection) {
  TilingProgramIRExpressionDefinition definition;
  definition.constants = {Pack(int64_t(-1)), Pack(uint64_t(1))};
  TilingProgramIRExpression left, right, root;
  left.value_type = Type::kInt64;
  right.value_type = Type::kUint64;
  right.constant_id = 1;
  root.kind = TilingProgramIRExpressionKind::kOperation;
  root.opcode = Op::kMin;
  root.value_type = Type::kUint64;
  root.arg_count = 2;
  root.args = {0, 1, 0};
  definition.nodes = {left, right, root};
  TilingProgramIRExpressionCode code;
  ASSERT_EQ(LowerTilingProgramIRExpression(definition, &code).status, TilingExpressionStatus::kOk);
  std::shared_ptr<const TilingProgramIRCompiledExpression> compiled;
  ASSERT_EQ(BuildTilingProgramIRExpression(code, &compiled).status, TilingExpressionStatus::kOk);
  std::vector<Value> scratch(GetTilingExpressionScratchElements(*compiled));
  Value result;
  ASSERT_EQ(
      EvaluateTilingProgramIRExpression(*compiled, nullptr, nullptr, scratch.data(), scratch.size(), &result).status,
      TilingExpressionStatus::kOk);
  EXPECT_EQ(result.type, Type::kUint64);
  EXPECT_EQ(result.raw_bits, UINT64_MAX);
  // Promoting -1 to uint64 before comparing would incorrectly choose 1.
  definition.nodes.back().opcode = Op::kAdd;
  EXPECT_EQ(LowerTilingProgramIRExpression(definition, &code).status, TilingExpressionStatus::kInvalidProgram);
}
}  // namespace att
