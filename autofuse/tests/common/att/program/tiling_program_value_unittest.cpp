/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <type_traits>

#include <gtest/gtest.h>
#include "program/tiling_program_value.h"

namespace att {
namespace {
using Type = TilingProgramIRValueType;
using Status = TilingProgramValueStatus;

template <typename T>
constexpr Type Tag() {
  if constexpr (std::is_same_v<T, bool>) {
    return Type::kBoolean;
  } else if constexpr (std::is_same_v<T, int8_t>) {
    return Type::kInt8;
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    return Type::kUint8;
  } else if constexpr (std::is_same_v<T, int16_t>) {
    return Type::kInt16;
  } else if constexpr (std::is_same_v<T, uint16_t>) {
    return Type::kUint16;
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return Type::kInt32;
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return Type::kUint32;
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return Type::kInt64;
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    return Type::kUint64;
  } else if constexpr (std::is_same_v<T, float>) {
    return Type::kFloat32;
  } else {
    static_assert(std::is_same_v<T, double>);
    return Type::kFloat64;
  }
}

template <typename T>
TilingProgramIRValue Value(T value) {
  if constexpr (std::is_same_v<T, bool>) {
    return {Tag<T>(), value ? 1U : 0U};
  } else if constexpr (std::is_integral_v<T>) {
    using U = std::make_unsigned_t<T>;
    return {Tag<T>(), static_cast<U>(value)};
  } else if constexpr (std::is_same_v<T, float>) {
    uint32_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    return {Tag<T>(), bits};
  } else {
    uint64_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    return {Tag<T>(), bits};
  }
}

template <typename To, typename From>
void CheckNative(From value) {
  // Volatile forces the same actual host conversion that the generated native code uses.
  volatile From source = value;
  To native_value;
  if constexpr (std::is_same_v<To, bool> && std::is_floating_point_v<From>) {
    // Test the native contextual bool conversion without spelling a floating equality.
    if (source) {
      native_value = true;
    } else {
      native_value = false;
    }
  } else {
    native_value = static_cast<To>(source);
  }
  const auto expected = Value(native_value);
  TilingProgramIRValue output{Type::kUint8, 43U};
  ASSERT_EQ(CastTilingProgramIRValue(Value(value), Tag<To>(), &output), Status::kOk);
  EXPECT_EQ(output.type, expected.type);
  EXPECT_EQ(output.raw_bits, expected.raw_bits)
      << "from=" << static_cast<int>(Tag<From>()) << " to=" << static_cast<int>(Tag<To>());
}

template <typename From>
void CheckIntegerSource(From value) {
  CheckNative<bool>(value);
  CheckNative<int8_t>(value);
  CheckNative<uint8_t>(value);
  CheckNative<int16_t>(value);
  CheckNative<uint16_t>(value);
  CheckNative<int32_t>(value);
  CheckNative<uint32_t>(value);
  CheckNative<int64_t>(value);
  CheckNative<uint64_t>(value);
  CheckNative<float>(value);
  CheckNative<double>(value);
}

template <typename From>
void CheckIntegerType() {
  for (From value :
       {std::numeric_limits<From>::lowest(), std::numeric_limits<From>::max(), From(0), From(1), From(-1)}) {
    CheckIntegerSource(value);
  }
  std::mt19937_64 random(20261102U);
  for (size_t i = 0U; i < 100U; ++i) {
    CheckIntegerSource(static_cast<From>(random()));
  }
}

void Reject(TilingProgramIRValue input, Type target, Status status) {
  TilingProgramIRValue output{Type::kUint8, 43U};
  EXPECT_EQ(CastTilingProgramIRValue(input, target, &output), status);
  EXPECT_EQ(output.type, Type::kUint8);
  EXPECT_EQ(output.raw_bits, 43U);
}

template <typename To>
void CheckFloatingIntegerBounds() {
  const double upper = std::ldexp(1.0, std::numeric_limits<To>::digits);
  const double lower = std::is_signed_v<To> ? -upper : 0.0;
  CheckNative<To>(lower);
  CheckNative<To>(std::nextafter(upper, 0.0));
  CheckNative<To>(0.0);
  CheckNative<To>(-0.0);
  CheckNative<To>(-0.99);
  CheckNative<To>(1.99);
  if constexpr (std::is_signed_v<To>) {
    CheckNative<To>(-1.99);
  }
  Reject(Value(upper), Tag<To>(), Status::kArithmeticError);
  if constexpr (std::is_signed_v<To>) {
    Reject(Value(lower - std::max(1.0, std::abs(lower) * 0x1p-52)), Tag<To>(), Status::kArithmeticError);
  } else {
    Reject(Value(-1.0), Tag<To>(), Status::kArithmeticError);
  }
  for (double value : {std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
    Reject(Value(value), Tag<To>(), Status::kArithmeticError);
  }
}
}  // namespace

TEST(TilingProgramValueTest, ChecksRequiredHostRoundingProfile) {
  EXPECT_EQ(std::fegetround(), FE_TONEAREST);
}

TEST(TilingProgramValueTest, KeepsReferenceTypeIds) {
  EXPECT_EQ(static_cast<uint8_t>(Type::kBoolean), 1U);
  EXPECT_EQ(static_cast<uint8_t>(Type::kInt64), 2U);
  EXPECT_EQ(static_cast<uint8_t>(Type::kFloat64), 3U);
  EXPECT_EQ(static_cast<uint8_t>(Type::kUint64), 4U);
}

TEST(TilingProgramValueTest, NativeIntegerConversionMatrix) {
  CheckIntegerType<bool>();
  CheckIntegerType<int8_t>();
  CheckIntegerType<uint8_t>();
  CheckIntegerType<int16_t>();
  CheckIntegerType<uint16_t>();
  CheckIntegerType<int32_t>();
  CheckIntegerType<uint32_t>();
  CheckIntegerType<int64_t>();
  CheckIntegerType<uint64_t>();
}

TEST(TilingProgramValueTest, FloatToIntegerBoundsMatchNativeWithinDefinedDomain) {
  CheckFloatingIntegerBounds<int8_t>();
  CheckFloatingIntegerBounds<uint8_t>();
  CheckFloatingIntegerBounds<int16_t>();
  CheckFloatingIntegerBounds<uint16_t>();
  CheckFloatingIntegerBounds<int32_t>();
  CheckFloatingIntegerBounds<uint32_t>();
  CheckFloatingIntegerBounds<int64_t>();
  CheckFloatingIntegerBounds<uint64_t>();
  CheckNative<int8_t>(127.99F);
  CheckNative<int8_t>(-128.99F);
  Reject(Value(128.0F), Type::kInt8, Status::kArithmeticError);
  Reject(Value(0x1p64F), Type::kUint64, Status::kArithmeticError);
}

TEST(TilingProgramValueTest, FloatingConversionsPreserveNativeBitsAndBooleanMeaning) {
  for (double value : {0.0, -0.0, 1.0, -1.0, 0x1.000001p0, 0x1.000003p0, std::numeric_limits<double>::denorm_min(),
                       std::numeric_limits<double>::max(), std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
    CheckNative<float>(value);
    CheckNative<double>(value);
    CheckNative<bool>(value);
  }
  for (float value :
       {0.0F, -0.0F, 1.0F, -1.0F, std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::max(),
        std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
    CheckNative<float>(value);
    CheckNative<double>(value);
    CheckNative<bool>(value);
  }
}

TEST(TilingProgramValueTest, IntegerToFloatHasNoIntermediateDoubleRounding) {
  volatile uint64_t native = (UINT64_C(1) << 63U) + (UINT64_C(1) << 39U) + 1U;
  volatile double intermediate = static_cast<double>(native);
  ASSERT_NE(Value(static_cast<float>(native)).raw_bits, Value(static_cast<float>(intermediate)).raw_bits);
  CheckNative<float>(static_cast<uint64_t>(native));
  CheckNative<double>(UINT64_C(9007199254740993));
  CheckNative<uint64_t>(UINT64_C(9007199254740993));
}

TEST(TilingProgramValueTest, IdentityPreservesNanPayloadAndAllowsAlias) {
  for (auto value :
       {TilingProgramIRValue{Type::kFloat32, UINT64_C(0x7fc01234)},
        TilingProgramIRValue{Type::kFloat64, UINT64_C(0x7ff8000000004321)},
        TilingProgramIRValue{Type::kFloat32, UINT64_C(0x80000000)}, TilingProgramIRValue{Type::kInt64, UINT64_MAX}}) {
    const auto saved = value;
    ASSERT_EQ(CastTilingProgramIRValue(value, value.type, &value), Status::kOk);
    EXPECT_EQ(value.raw_bits, saved.raw_bits);
  }
  auto value = Value(int8_t(-1));
  ASSERT_EQ(CastTilingProgramIRValue(value, Type::kInt64, &value), Status::kOk);
  EXPECT_EQ(value.raw_bits, UINT64_MAX);
  value = Value(std::numeric_limits<double>::quiet_NaN());
  const auto saved = value;
  EXPECT_EQ(CastTilingProgramIRValue(value, Type::kInt64, &value), Status::kArithmeticError);
  EXPECT_EQ(value.type, saved.type);
  EXPECT_EQ(value.raw_bits, saved.raw_bits);
}

TEST(TilingProgramValueTest, InvalidValuesAndTypesCannotPublishOutput) {
  for (auto value :
       {TilingProgramIRValue{Type::kBoolean, 2U}, TilingProgramIRValue{Type::kInt8, 256U},
        TilingProgramIRValue{Type::kUint16, 65536U}, TilingProgramIRValue{Type::kFloat32, UINT64_C(1) << 32U}}) {
    EXPECT_EQ(ValidateTilingProgramIRValue(value), Status::kInvalidValue);
    Reject(value, Type::kInt64, Status::kInvalidValue);
  }
  Reject({static_cast<Type>(255), 0U}, Type::kInt64, Status::kUnsupportedType);
  Reject(Value(int64_t(0)), static_cast<Type>(255), Status::kUnsupportedType);
  EXPECT_EQ(CastTilingProgramIRValue(Value(0.0), Type::kInt64, nullptr), Status::kInvalidArgument);
}

namespace {
using Op = TilingProgramIROpcode;
template <typename T>
void CheckBinaryNative(T a, T b) {
  for (Op op : {Op::kAdd, Op::kSub, Op::kMul, Op::kDiv, Op::kMod}) {
    volatile T left = a, right = b;
    T expected = 0;
    switch (op) {
      case Op::kAdd:
        expected = left + right;
        break;
      case Op::kSub:
        expected = left - right;
        break;
      case Op::kMul:
        expected = left * right;
        break;
      case Op::kDiv:
        expected = left / right;
        break;
      case Op::kMod:
        if constexpr (std::is_integral_v<T>) {
          expected = left % right;
        } else {
          continue;
        }
        break;
      default:
        FAIL();
    }
    auto output = Value(a);
    ASSERT_EQ(EvaluateTilingProgramIRBinary(op, output, Value(b), &output), Status::kOk);
    EXPECT_EQ(output.type, Tag<T>());
    EXPECT_EQ(output.raw_bits, Value(expected).raw_bits);
  }
}
template <typename T>
void CheckBinaryMatrix() {
  for (int a : {-999, -7, 0, 9, 1000}) {
    for (int b : {-13, -1, 1, 7}) {
      if constexpr (std::is_unsigned_v<T>) {
        CheckBinaryNative<T>(static_cast<T>(a), static_cast<T>(b));
      } else {
        CheckBinaryNative<T>(T(a), T(b));
      }
    }
  }
}
void RejectBinary(Op op, TilingProgramIRValue a, TilingProgramIRValue b, Status status) {
  const auto saved = a;
  EXPECT_EQ(EvaluateTilingProgramIRBinary(op, a, b, &a), status);
  EXPECT_EQ(a.type, saved.type);
  EXPECT_EQ(a.raw_bits, saved.raw_bits);
}
template <typename T>
void CheckSignedErrors() {
  const T low = std::numeric_limits<T>::min(), high = std::numeric_limits<T>::max();
  RejectBinary(Op::kAdd, Value(high), Value(T(1)), Status::kArithmeticError);
  RejectBinary(Op::kSub, Value(low), Value(T(1)), Status::kArithmeticError);
  RejectBinary(Op::kMul, Value(low), Value(T(-1)), Status::kArithmeticError);
  for (Op op : {Op::kDiv, Op::kMod}) {
    RejectBinary(op, Value(low), Value(T(-1)), Status::kArithmeticError);
    RejectBinary(op, Value(T(1)), Value(T(0)), Status::kArithmeticError);
  }
}
}  // namespace

TEST(TilingProgramValueTest, BinaryPromotedTypesMatchNative) {
  CheckBinaryMatrix<int32_t>();
  CheckBinaryMatrix<uint32_t>();
  CheckBinaryMatrix<int64_t>();
  CheckBinaryMatrix<uint64_t>();
  CheckBinaryMatrix<float>();
  CheckBinaryMatrix<double>();
  CheckBinaryNative<int64_t>(INT64_C(9007199254740993), INT64_C(7));
  CheckBinaryNative<uint64_t>(UINT64_C(9007199254740993), UINT64_C(7));
}

TEST(TilingProgramValueTest, SignedOverflowAndUndefinedDivisionAreRejectedBeforePublication) {
  CheckSignedErrors<int32_t>();
  CheckSignedErrors<int64_t>();
  RejectBinary(Op::kDiv, Value(uint64_t(1)), Value(uint64_t(0)), Status::kArithmeticError);
  RejectBinary(Op::kMod, Value(uint32_t(1)), Value(uint32_t(0)), Status::kArithmeticError);
}

TEST(TilingProgramValueTest, UnsignedArithmeticWrapsAtTheDeclaredWidth) {
  CheckBinaryNative<uint32_t>(UINT32_MAX, uint32_t(2));
  CheckBinaryNative<uint64_t>(UINT64_MAX, uint64_t(2));
  CheckBinaryNative<uint32_t>(0U, UINT32_MAX);
  CheckBinaryNative<uint64_t>(0U, UINT64_MAX);
}

TEST(TilingProgramValueTest, Float32RoundsEachStepAndDivisionKeepsIeeeBehavior) {
  auto result = Value(16777216.0F);
  ASSERT_EQ(EvaluateTilingProgramIRBinary(Op::kAdd, result, Value(1.0F), &result), Status::kOk);
  ASSERT_EQ(EvaluateTilingProgramIRBinary(Op::kSub, result, Value(16777216.0F), &result), Status::kOk);
  EXPECT_EQ(result.raw_bits, Value(0.0F).raw_bits);
  ASSERT_EQ(EvaluateTilingProgramIRBinary(Op::kDiv, Value(1.0F), Value(-0.0F), &result), Status::kOk);
  EXPECT_EQ(result.raw_bits, Value(-std::numeric_limits<float>::infinity()).raw_bits);
  ASSERT_EQ(EvaluateTilingProgramIRBinary(Op::kDiv, Value(0.0), Value(0.0), &result), Status::kOk);
  double value;
  std::memcpy(&value, &result.raw_bits, sizeof(value));
  EXPECT_TRUE(std::isnan(value));
}

TEST(TilingProgramValueTest, BinaryRejectsImplicitPromotionsAndUnsupportedOperations) {
  RejectBinary(Op::kAdd, Value(int32_t(1)), Value(uint32_t(1)), Status::kInvalidArgument);
  RejectBinary(Op::kAdd, Value(int8_t(1)), Value(int8_t(2)), Status::kUnsupportedType);
  RejectBinary(Op::kLogicalAnd, Value(int64_t(1)), Value(int64_t(0)), Status::kUnsupportedOperation);
  RejectBinary(static_cast<Op>(65535), Value(1.0F), Value(2.0F), Status::kUnsupportedOperation);
  RejectBinary(Op::kAdd, {Type::kUint32, UINT64_MAX}, Value(uint32_t(1)), Status::kInvalidValue);
  EXPECT_EQ(EvaluateTilingProgramIRBinary(Op::kAdd, Value(1.0), Value(2.0), nullptr), Status::kInvalidArgument);
}

namespace {
template <typename T>
void CheckComparisons(T a, T b) {
  const bool expected[] = {std::equal_to<T>{}(a, b), std::not_equal_to<T>{}(a, b), a >= b, a > b, a <= b, a < b};
  const Op operations[] = {Op::kEq, Op::kNe, Op::kGe, Op::kGt, Op::kLe, Op::kLt};
  for (size_t i = 0; i < 6; ++i) {
    auto output = Value(a);
    ASSERT_EQ(EvaluateTilingProgramIRBinary(operations[i], output, Value(b), &output), Status::kOk);
    EXPECT_EQ(output.type, Type::kBoolean);
    EXPECT_EQ(output.raw_bits, expected[i] ? 1U : 0U);
  }
}
void CheckSelection(Op op, TilingProgramIRValue a, TilingProgramIRValue b, TilingProgramIRValue expected) {
  ASSERT_EQ(EvaluateTilingProgramIRBinary(op, a, b, &a), Status::kOk);
  EXPECT_EQ(a.type, expected.type);
  EXPECT_EQ(a.raw_bits, expected.raw_bits);
}
}  // namespace
TEST(TilingProgramValueTest, ComparisonsUseComputeTypeInsteadOfDoubleConversion) {
  CheckComparisons<int32_t>(-7, 9);
  CheckComparisons<uint32_t>(UINT32_MAX, 0);
  CheckComparisons<int64_t>(INT64_C(9007199254740993), INT64_C(9007199254740992));
  CheckComparisons<uint64_t>(UINT64_MAX, UINT64_MAX - 1);
  CheckComparisons<float>(-0.0F, 0.0F);
  CheckComparisons<double>(-0.0, 0.0);
  for (double a : {1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
    for (double b : {1.0, -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
      CheckComparisons<double>(a, b);
      CheckComparisons<float>(static_cast<float>(a), static_cast<float>(b));
    }
  }
}
TEST(TilingProgramValueTest, NativeMinMaxRetainsDoubleComparisonForLargeIntegers) {
  const auto low = Value(INT64_C(9007199254740992)), high = Value(INT64_C(9007199254740993));
  // Both integers compare equal after the ORIGINAL helper converts to double.
  CheckSelection(Op::kMin, low, high, high);
  CheckSelection(Op::kMax, high, low, low);
  const auto uhigh = Value(UINT64_MAX), ulow = Value(UINT64_MAX - 1);
  CheckSelection(Op::kMin, ulow, uhigh, uhigh);
  CheckSelection(Op::kMax, uhigh, ulow, ulow);
  CheckSelection(Op::kMin, Value(int32_t(-7)), Value(int32_t(9)), Value(int32_t(-7)));
}
TEST(TilingProgramValueTest, NativeMinMaxSelectsRightSignedZeroOnTies) {
  for (Op op : {Op::kMin, Op::kMax}) {
    CheckSelection(op, Value(0.0), Value(-0.0), Value(-0.0));
    CheckSelection(op, Value(-0.0), Value(0.0), Value(0.0));
    CheckSelection(op, Value(0.0F), Value(-0.0F), Value(-0.0F));
  }
}
TEST(TilingProgramValueTest, NativeMinMaxSelectsRightOperandForUnorderedComparisons) {
  const TilingProgramIRValue nan{Type::kFloat64, UINT64_C(0x7ff8000000000123)};
  const TilingProgramIRValue nanf{Type::kFloat32, UINT64_C(0x7fc01234)};
  for (Op op : {Op::kMin, Op::kMax}) {
    CheckSelection(op, nan, Value(7.0), Value(7.0));
    CheckSelection(op, Value(7.0), nan, nan);
    CheckSelection(op, nanf, Value(7.0F), Value(7.0F));
    CheckSelection(op, Value(7.0F), nanf, nanf);
  }
}

namespace {
void CheckUnary(Op op, TilingProgramIRValue input, TilingProgramIRValue expected) {
  ASSERT_EQ(EvaluateTilingProgramIRUnary(op, input, &input), Status::kOk);
  EXPECT_EQ(input.type, expected.type);
  EXPECT_EQ(input.raw_bits, expected.raw_bits);
}
void RejectUnary(Op op, TilingProgramIRValue input, Status expected) {
  const auto saved = input;
  EXPECT_EQ(EvaluateTilingProgramIRUnary(op, input, &input), expected);
  EXPECT_EQ(input.type, saved.type);
  EXPECT_EQ(input.raw_bits, saved.raw_bits);
}
}  // namespace
TEST(TilingProgramValueTest, UnaryPreservesNativeHelperSemantics) {
  CheckUnary(Op::kAbs, Value(-7.0F), Value(7.0F));
  CheckUnary(Op::kAbs, Value(-0.0), Value(-0.0));
  CheckUnary(Op::kAbs, Value(-0.0F), Value(-0.0F));
  CheckUnary(Op::kAbs, {Type::kFloat64, UINT64_C(0x7ff8000000000123)}, {Type::kFloat64, UINT64_C(0xfff8000000000123)});
  CheckUnary(Op::kNeg, Value(-0.0), Value(0.0));
  CheckUnary(Op::kNeg, Value(uint32_t(1)), Value(UINT32_MAX));
  CheckUnary(Op::kNeg, Value(uint64_t(1)), Value(UINT64_MAX));
  CheckUnary(Op::kAbs, Value(int64_t(-7)), Value(int64_t(7)));
  // The original helper truncates first; these are deliberately not std::ceil/floor.
  CheckUnary(Op::kCeiling, Value(-1.5), Value(0.0));
  CheckUnary(Op::kFloor, Value(-1.5), Value(-1.0));
  CheckUnary(Op::kCeiling, Value(1.0 + 1e-9), Value(1.0));
  CheckUnary(Op::kCeiling, Value(1.5F), Value(2.0F));
  CheckUnary(Op::kFloor, Value(-0.0F), Value(0.0F));
  CheckUnary(Op::kCeiling, Value(UINT64_MAX), Value(UINT64_MAX));
  CheckUnary(Op::kFloor, Value(INT64_MIN), Value(INT64_MIN));
  CheckUnary(Op::kNot, Value(true), Value(false));
  CheckUnary(Op::kNot, Value(false), Value(true));
}
TEST(TilingProgramValueTest, LogUsesDoubleResultAndNativeDomainBehavior) {
  for (auto input :
       {Value(int32_t(2)), Value(uint32_t(2)), Value(int64_t(2)), Value(uint64_t(2)), Value(2.0F), Value(2.0)}) {
    CheckUnary(Op::kLog, input, Value(std::log(2.0)));
  }
  CheckUnary(Op::kLog, Value(0.0F), Value(-std::numeric_limits<double>::infinity()));
  TilingProgramIRValue output;
  ASSERT_EQ(EvaluateTilingProgramIRUnary(Op::kLog, Value(-1.0), &output), Status::kOk);
  double result;
  std::memcpy(&result, &output.raw_bits, sizeof(result));
  EXPECT_EQ(output.type, Type::kFloat64);
  EXPECT_TRUE(std::isnan(result));
}
TEST(TilingProgramValueTest, UnaryRejectsUndefinedNativeConversionsAndPreservesOutput) {
  for (Op op : {Op::kAbs, Op::kNeg}) {
    RejectUnary(op, Value(INT32_MIN), Status::kArithmeticError);
    RejectUnary(op, Value(INT64_MIN), Status::kArithmeticError);
  }
  for (Op op : {Op::kCeiling, Op::kFloor}) {
    for (double value : {0x1p63, -0x1.0000000000001p63, std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
      RejectUnary(op, Value(value), Status::kArithmeticError);
    }
    RejectUnary(op, Value(0x1p63F), Status::kArithmeticError);
  }
  RejectUnary(Op::kNeg, Value(int8_t(1)), Status::kUnsupportedType);
  RejectUnary(Op::kNot, Value(1.0), Status::kUnsupportedType);
  RejectUnary(Op::kNot, {Type::kBoolean, 2}, Status::kInvalidValue);
  RejectUnary(Op::kAdd, Value(int32_t(1)), Status::kInvalidArgument);
  RejectUnary(Op::kAlign, Value(1.0), Status::kUnsupportedOperation);
  EXPECT_EQ(EvaluateTilingProgramIRUnary(Op::kNeg, Value(1.0), nullptr), Status::kInvalidArgument);
}
TEST(TilingProgramValueTest, SignatureChecksDoNotExecuteValuesOrPublishOnFailure) {
  Type out = Type::kUint8;
  EXPECT_EQ(GetTilingProgramIRUnaryType(Op::kLog, Type::kFloat32, &out), Status::kOk);
  EXPECT_EQ(out, Type::kFloat64);
  out = Type::kUint8;
  EXPECT_EQ(GetTilingProgramIRBinaryType(Op::kLog, Type::kFloat32, &out), Status::kInvalidArgument);
  EXPECT_EQ(out, Type::kUint8);
  EXPECT_EQ(GetTilingProgramIRUnaryType(Op::kNeg, static_cast<Type>(255), &out), Status::kUnsupportedType);
  EXPECT_EQ(out, Type::kUint8);
}

TEST(TilingProgramValueTest, PowReturnsDoubleEvenForFloatAndIntegerInputs) {
  for (auto input :
       {Value(int32_t(3)), Value(uint32_t(3)), Value(int64_t(3)), Value(uint64_t(3)), Value(3.0F), Value(3.0)}) {
    Type result_type;
    ASSERT_EQ(GetTilingProgramIRBinaryType(Op::kPow, input.type, &result_type), Status::kOk);
    EXPECT_EQ(result_type, Type::kFloat64);
    auto output = input;
    ASSERT_EQ(EvaluateTilingProgramIRBinary(Op::kPow, output, output, &output), Status::kOk);
    EXPECT_EQ(output.type, Type::kFloat64);
    EXPECT_EQ(output.raw_bits, Value(27.0).raw_bits);
  }
  TilingProgramIRValue output;
  ASSERT_EQ(EvaluateTilingProgramIRBinary(Op::kPow, Value(3.0F), Value(-1.0F), &output), Status::kOk);
  EXPECT_EQ(output.raw_bits, Value(::pow(3.0, -1.0)).raw_bits);
  EXPECT_NE(output.raw_bits, Value(static_cast<double>(1.0F / 3.0F)).raw_bits);
}
TEST(TilingProgramValueTest, FloatingModPreservesWidthSignsAndIeeeDomain) {
  for (auto values : {std::pair{Value(-5.5F), Value(2.0F)}, std::pair{Value(-5.5), Value(2.0)}}) {
    auto output = values.first;
    ASSERT_EQ(EvaluateTilingProgramIRBinary(Op::kMod, values.first, values.second, &output), Status::kOk);
    EXPECT_EQ(output.type, values.first.type);
    EXPECT_EQ(output.raw_bits, output.type == Type::kFloat32 ? Value(-1.5F).raw_bits : Value(-1.5).raw_bits);
  }
  TilingProgramIRValue output;
  ASSERT_EQ(EvaluateTilingProgramIRBinary(Op::kMod, Value(-0.0F), Value(2.0F), &output), Status::kOk);
  EXPECT_EQ(output.raw_bits, Value(-0.0F).raw_bits);
  ASSERT_EQ(EvaluateTilingProgramIRBinary(Op::kMod, Value(1.0), Value(0.0), &output), Status::kOk);
  double result;
  std::memcpy(&result, &output.raw_bits, sizeof(result));
  EXPECT_TRUE(std::isnan(result));
}
TEST(TilingProgramValueTest, NativeTernarySelectsThenConvertsToDoubleAndAllowsAlias) {
  for (bool condition : {false, true}) {
    auto c = Value(condition);
    ASSERT_EQ(EvaluateTilingProgramIRTernary(Op::kTernary, c, Value(-0.0F), Value(7.0F), &c), Status::kOk);
    EXPECT_EQ(c.type, Type::kFloat64);
    EXPECT_EQ(c.raw_bits, Value(condition ? -0.0 : 7.0).raw_bits);
  }
  auto output = Value(UINT64_MAX);
  ASSERT_EQ(EvaluateTilingProgramIRTernary(Op::kTernary, Value(true), output, Value(uint64_t(0)), &output),
            Status::kOk);
  EXPECT_EQ(output.raw_bits, Value(static_cast<double>(UINT64_MAX)).raw_bits);
  TilingProgramIRValue selected{Type::kFloat64, UINT64_C(0x7ff8000000000123)};
  ASSERT_EQ(EvaluateTilingProgramIRTernary(Op::kTernary, Value(false), Value(1.0), selected, &output), Status::kOk);
  EXPECT_EQ(output.raw_bits, selected.raw_bits);
}
TEST(TilingProgramValueTest, TernaryRejectsInvalidArityTypesAndValuesWithoutPublication) {
  auto output = Value(int64_t(99));
  EXPECT_EQ(EvaluateTilingProgramIRTernary(Op::kTernary, Value(int32_t(1)), Value(1.0), Value(2.0), &output),
            Status::kInvalidArgument);
  EXPECT_EQ(EvaluateTilingProgramIRTernary(Op::kTernary, Value(true), Value(1.0F), Value(2.0), &output),
            Status::kInvalidArgument);
  EXPECT_EQ(EvaluateTilingProgramIRTernary(Op::kTernary, {Type::kBoolean, 2}, Value(1.0), Value(2.0), &output),
            Status::kInvalidValue);
  EXPECT_EQ(EvaluateTilingProgramIRTernary(Op::kTernary, Value(true), {static_cast<Type>(99), 1},
                                           {static_cast<Type>(99), 2}, &output),
            Status::kUnsupportedType);
  EXPECT_EQ(EvaluateTilingProgramIRTernary(Op::kAdd, Value(true), Value(1.0), Value(2.0), &output),
            Status::kInvalidArgument);
  EXPECT_EQ(output.type, Type::kInt64);
  EXPECT_EQ(output.raw_bits, 99U);
  EXPECT_EQ(EvaluateTilingProgramIRTernary(Op::kTernary, Value(true), Value(1.0), Value(2.0), nullptr),
            Status::kInvalidArgument);
}
}  // namespace att
