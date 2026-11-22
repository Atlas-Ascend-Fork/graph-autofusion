/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "tiling_program_value.h"

#include <cmath>
#include <functional>
#include <limits>
#include <initializer_list>
#include <type_traits>

namespace att {
namespace {
namespace native_scalar {
#include "core/scalar/expression_helpers.inc"
#include "core/scalar/scalar_helpers.inc"
template <typename T>
auto EvaluateLog(T value) {
  return Log(value);
}
template <typename T>
auto EvaluatePow(T left, T right) {
  static_assert(std::is_same_v<decltype(Pow(left, right)), double>, "native Pow profile requires double result");
  return Pow(left, right);
}
}  // namespace native_scalar
#undef Log
#undef Pow
#undef Rational
#undef ExpectEq
#undef ExpectNe
#undef ExpectLe
#undef ExpectLt
#undef LogicAnd
#undef LogicOr
#undef True
#undef False
static_assert(sizeof(float) == 4U && sizeof(double) == 8U, "IR requires IEEE binary32/binary64");
static_assert(std::numeric_limits<float>::is_iec559 && std::numeric_limits<double>::is_iec559,
              "IR requires IEEE floating point");
static_assert(std::numeric_limits<int64_t>::min() == -std::numeric_limits<int64_t>::max() - 1,
              "IR requires two's-complement integers");

template <typename T>
using Bits = std::conditional_t<
    sizeof(T) == 1U, uint8_t,
    std::conditional_t<sizeof(T) == 2U, uint16_t, std::conditional_t<sizeof(T) == 4U, uint32_t, uint64_t>>>;

template <typename T>
T ReadValue(const TilingProgramIRValue &value) {
  const Bits<T> bits = static_cast<Bits<T>>(value.raw_bits);
  return CopyTilingScalarBits<T>(bits);
}

template <typename T>
uint64_t RawBits(T value) {
  return CopyTilingScalarBits<Bits<T>>(value);
}

uint32_t Width(TilingProgramIRValueType type) {
  switch (type) {
    case TilingProgramIRValueType::kBoolean:
      return 1U;
    case TilingProgramIRValueType::kInt8:
    case TilingProgramIRValueType::kUint8:
      return 8U;
    case TilingProgramIRValueType::kInt16:
    case TilingProgramIRValueType::kUint16:
      return 16U;
    case TilingProgramIRValueType::kInt32:
    case TilingProgramIRValueType::kUint32:
    case TilingProgramIRValueType::kFloat32:
      return 32U;
    case TilingProgramIRValueType::kInt64:
    case TilingProgramIRValueType::kUint64:
    case TilingProgramIRValueType::kFloat64:
      return 64U;
    default:
      return 0U;
  }
}

template <typename To, typename From>
TilingProgramValueStatus Convert(From value, uint64_t &result) {
  if constexpr (std::is_same_v<To, bool>) {
    if constexpr (std::is_floating_point_v<From>) {
      result = std::fpclassify(value) == FP_ZERO ? 0U : 1U;
    } else {
      result = value != 0 ? 1U : 0U;
    }
  } else if constexpr (std::is_integral_v<To> && std::is_floating_point_v<From>) {
    // binary32 -> binary64 is exact. The powers-of-two bounds remain exact for 64-bit targets.
    const double truncated = std::trunc(static_cast<double>(value));
    constexpr int value_bits = std::numeric_limits<To>::digits;
    const double upper = std::ldexp(1.0, value_bits);
    const double lower = std::is_signed_v<To> ? -upper : 0.0;
    if (!std::isfinite(truncated) || truncated < lower || truncated >= upper) {
      return TilingProgramValueStatus::kArithmeticError;
    }
    result = RawBits(static_cast<To>(value));
  } else if constexpr (std::is_integral_v<To> && std::is_integral_v<From>) {
    // Conversion to the unsigned representation is defined modulo 2^width. Reconstructing the
    // signed value is deferred to ReadValue; no implementation-defined signed cast is required.
    result = static_cast<Bits<To>>(value);
  } else {
    result = RawBits(static_cast<To>(value));
  }
  return TilingProgramValueStatus::kOk;
}

template <typename From>
TilingProgramValueStatus ConvertTo(From value, TilingProgramIRValueType target, uint64_t &result) {
  switch (target) {
    case TilingProgramIRValueType::kBoolean:
      return Convert<bool>(value, result);
    case TilingProgramIRValueType::kInt8:
      return Convert<int8_t>(value, result);
    case TilingProgramIRValueType::kUint8:
      return Convert<uint8_t>(value, result);
    case TilingProgramIRValueType::kInt16:
      return Convert<int16_t>(value, result);
    case TilingProgramIRValueType::kUint16:
      return Convert<uint16_t>(value, result);
    case TilingProgramIRValueType::kInt32:
      return Convert<int32_t>(value, result);
    case TilingProgramIRValueType::kUint32:
      return Convert<uint32_t>(value, result);
    case TilingProgramIRValueType::kInt64:
      return Convert<int64_t>(value, result);
    case TilingProgramIRValueType::kUint64:
      return Convert<uint64_t>(value, result);
    case TilingProgramIRValueType::kFloat32:
      return Convert<float>(value, result);
    case TilingProgramIRValueType::kFloat64:
      return Convert<double>(value, result);
    default:
      return TilingProgramValueStatus::kUnsupportedType;
  }
}

TilingProgramValueStatus ConvertValue(const TilingProgramIRValue &input, TilingProgramIRValueType target,
                                      uint64_t &result) {
  switch (input.type) {
    case TilingProgramIRValueType::kBoolean:
      return ConvertTo(input.raw_bits != 0U, target, result);
    case TilingProgramIRValueType::kInt8:
      return ConvertTo(ReadValue<int8_t>(input), target, result);
    case TilingProgramIRValueType::kUint8:
      return ConvertTo(ReadValue<uint8_t>(input), target, result);
    case TilingProgramIRValueType::kInt16:
      return ConvertTo(ReadValue<int16_t>(input), target, result);
    case TilingProgramIRValueType::kUint16:
      return ConvertTo(ReadValue<uint16_t>(input), target, result);
    case TilingProgramIRValueType::kInt32:
      return ConvertTo(ReadValue<int32_t>(input), target, result);
    case TilingProgramIRValueType::kUint32:
      return ConvertTo(ReadValue<uint32_t>(input), target, result);
    case TilingProgramIRValueType::kInt64:
      return ConvertTo(ReadValue<int64_t>(input), target, result);
    case TilingProgramIRValueType::kUint64:
      return ConvertTo(ReadValue<uint64_t>(input), target, result);
    case TilingProgramIRValueType::kFloat32:
      return ConvertTo(ReadValue<float>(input), target, result);
    case TilingProgramIRValueType::kFloat64:
      return ConvertTo(ReadValue<double>(input), target, result);
    default:
      return TilingProgramValueStatus::kUnsupportedType;
  }
}

template <typename T>
TilingProgramValueStatus Arithmetic(TilingProgramIROpcode opcode, const TilingProgramIRValue &left,
                                    const TilingProgramIRValue &right, TilingProgramIRValue *output) {
  const T a = ReadValue<T>(left);
  const T b = ReadValue<T>(right);
  T value = 0;
  bool overflow = false;
  if constexpr (std::is_integral_v<T>) {
    if (opcode == TilingProgramIROpcode::kDiv || opcode == TilingProgramIROpcode::kMod) {
      if (b == 0) {
        return TilingProgramValueStatus::kArithmeticError;
      }
      if constexpr (std::is_signed_v<T>) {
        if (a == std::numeric_limits<T>::min() && b == -1) {
          return TilingProgramValueStatus::kArithmeticError;
        }
      }
      value = opcode == TilingProgramIROpcode::kDiv ? a / b : a % b;
    } else if constexpr (std::is_signed_v<T>) {
      switch (opcode) {
        case TilingProgramIROpcode::kAdd:
          overflow = __builtin_add_overflow(a, b, &value);
          break;
        case TilingProgramIROpcode::kSub:
          overflow = __builtin_sub_overflow(a, b, &value);
          break;
        case TilingProgramIROpcode::kMul:
          overflow = __builtin_mul_overflow(a, b, &value);
          break;
        default:
          return TilingProgramValueStatus::kUnsupportedOperation;
      }
    } else {
      switch (opcode) {
        case TilingProgramIROpcode::kAdd:
          value = a + b;
          break;
        case TilingProgramIROpcode::kSub:
          value = a - b;
          break;
        case TilingProgramIROpcode::kMul:
          value = a * b;
          break;
        default:
          return TilingProgramValueStatus::kUnsupportedOperation;
      }
    }
  } else {
    switch (opcode) {
      case TilingProgramIROpcode::kAdd:
        value = a + b;
        break;
      case TilingProgramIROpcode::kSub:
        value = a - b;
        break;
      case TilingProgramIROpcode::kMul:
        value = a * b;
        break;
      case TilingProgramIROpcode::kDiv:
        value = a / b;
        break;
      case TilingProgramIROpcode::kMod:
        value = native_scalar::Mod(a, b);
        break;
      default:
        return TilingProgramValueStatus::kUnsupportedOperation;
    }
  }
  if (overflow) {
    return TilingProgramValueStatus::kArithmeticError;
  }
  *output = TilingProgramIRValue{left.type, RawBits(value)};
  return TilingProgramValueStatus::kOk;
}

template <typename T>
TilingProgramValueStatus Binary(TilingProgramIROpcode opcode, const TilingProgramIRValue &left,
                                const TilingProgramIRValue &right, TilingProgramIRValue *output) {
  const T a = ReadValue<T>(left), b = ReadValue<T>(right);
  if (opcode == TilingProgramIROpcode::kPow) {
    *output = {TilingProgramIRValueType::kFloat64, RawBits(native_scalar::EvaluatePow(a, b))};
    return TilingProgramValueStatus::kOk;
  }
  bool comparison = false;
  switch (opcode) {
    case TilingProgramIROpcode::kIsEqual:
      comparison = native_scalar::IsEqual(a, b);
      break;
    case TilingProgramIROpcode::kEq:
      comparison = std::equal_to<T>{}(a, b);
      break;
    case TilingProgramIROpcode::kNe:
      comparison = std::not_equal_to<T>{}(a, b);
      break;
    case TilingProgramIROpcode::kGe:
      comparison = a >= b;
      break;
    case TilingProgramIROpcode::kGt:
      comparison = a > b;
      break;
    case TilingProgramIROpcode::kLe:
      comparison = a <= b;
      break;
    case TilingProgramIROpcode::kLt:
      comparison = a < b;
      break;
    default:
      return Arithmetic<T>(opcode, left, right, output);
  }
  *output = TilingProgramIRValue{TilingProgramIRValueType::kBoolean, comparison ? 1U : 0U};
  return TilingProgramValueStatus::kOk;
}

template <typename L, typename R>
TilingProgramValueStatus NativeMinMax(TilingProgramIROpcode opcode, const TilingProgramIRValue &left,
                                      const TilingProgramIRValue &right, TilingProgramIRValueType result_type,
                                      TilingProgramIRValue *output) {
  const L a = ReadValue<L>(left);
  const R b = ReadValue<R>(right);
  // Do not promote a/b first: the original helper compares their separate double conversions.
  const auto value = opcode == TilingProgramIROpcode::kMin ? native_scalar::Min(a, b) : native_scalar::Max(a, b);
  *output = {result_type, RawBits(value)};
  return TilingProgramValueStatus::kOk;
}
template <typename L>
TilingProgramValueStatus NativeMinMaxRight(TilingProgramIROpcode opcode, const TilingProgramIRValue &left,
                                           const TilingProgramIRValue &right, TilingProgramIRValueType result_type,
                                           TilingProgramIRValue *output) {
  switch (right.type) {
    case TilingProgramIRValueType::kBoolean:
      return NativeMinMax<L, bool>(opcode, left, right, result_type, output);
    case TilingProgramIRValueType::kInt8:
      return NativeMinMax<L, int8_t>(opcode, left, right, result_type, output);
    case TilingProgramIRValueType::kUint8:
      return NativeMinMax<L, uint8_t>(opcode, left, right, result_type, output);
    case TilingProgramIRValueType::kInt16:
      return NativeMinMax<L, int16_t>(opcode, left, right, result_type, output);
    case TilingProgramIRValueType::kUint16:
      return NativeMinMax<L, uint16_t>(opcode, left, right, result_type, output);
    case TilingProgramIRValueType::kInt32:
      return NativeMinMax<L, int32_t>(opcode, left, right, result_type, output);
    case TilingProgramIRValueType::kUint32:
      return NativeMinMax<L, uint32_t>(opcode, left, right, result_type, output);
    case TilingProgramIRValueType::kInt64:
      return NativeMinMax<L, int64_t>(opcode, left, right, result_type, output);
    case TilingProgramIRValueType::kUint64:
      return NativeMinMax<L, uint64_t>(opcode, left, right, result_type, output);
    case TilingProgramIRValueType::kFloat32:
      return NativeMinMax<L, float>(opcode, left, right, result_type, output);
    case TilingProgramIRValueType::kFloat64:
      return NativeMinMax<L, double>(opcode, left, right, result_type, output);
    default:
      return TilingProgramValueStatus::kUnsupportedType;
  }
}
TilingProgramValueStatus NativeMinMaxDispatch(TilingProgramIROpcode opcode, const TilingProgramIRValue &left,
                                              const TilingProgramIRValue &right, TilingProgramIRValueType result_type,
                                              TilingProgramIRValue *output) {
  switch (left.type) {
    case TilingProgramIRValueType::kBoolean:
      return NativeMinMaxRight<bool>(opcode, left, right, result_type, output);
    case TilingProgramIRValueType::kInt8:
      return NativeMinMaxRight<int8_t>(opcode, left, right, result_type, output);
    case TilingProgramIRValueType::kUint8:
      return NativeMinMaxRight<uint8_t>(opcode, left, right, result_type, output);
    case TilingProgramIRValueType::kInt16:
      return NativeMinMaxRight<int16_t>(opcode, left, right, result_type, output);
    case TilingProgramIRValueType::kUint16:
      return NativeMinMaxRight<uint16_t>(opcode, left, right, result_type, output);
    case TilingProgramIRValueType::kInt32:
      return NativeMinMaxRight<int32_t>(opcode, left, right, result_type, output);
    case TilingProgramIRValueType::kUint32:
      return NativeMinMaxRight<uint32_t>(opcode, left, right, result_type, output);
    case TilingProgramIRValueType::kInt64:
      return NativeMinMaxRight<int64_t>(opcode, left, right, result_type, output);
    case TilingProgramIRValueType::kUint64:
      return NativeMinMaxRight<uint64_t>(opcode, left, right, result_type, output);
    case TilingProgramIRValueType::kFloat32:
      return NativeMinMaxRight<float>(opcode, left, right, result_type, output);
    case TilingProgramIRValueType::kFloat64:
      return NativeMinMaxRight<double>(opcode, left, right, result_type, output);
    default:
      return TilingProgramValueStatus::kUnsupportedType;
  }
}

struct OperationSignature {
  uint8_t arity;
  uint32_t types;
  TilingProgramIRValueType result;
};
constexpr uint32_t TypeBit(TilingProgramIRValueType type) {
  return 1U << static_cast<uint8_t>(type);
}
constexpr uint32_t kIntegerTypes =
    TypeBit(TilingProgramIRValueType::kInt32) | TypeBit(TilingProgramIRValueType::kUint32) |
    TypeBit(TilingProgramIRValueType::kInt64) | TypeBit(TilingProgramIRValueType::kUint64);
constexpr uint32_t kNumericTypes =
    kIntegerTypes | TypeBit(TilingProgramIRValueType::kFloat32) | TypeBit(TilingProgramIRValueType::kFloat64);
constexpr uint32_t kScalarTypes = kNumericTypes | TypeBit(TilingProgramIRValueType::kBoolean) |
                                  TypeBit(TilingProgramIRValueType::kInt8) | TypeBit(TilingProgramIRValueType::kUint8) |
                                  TypeBit(TilingProgramIRValueType::kInt16) |
                                  TypeBit(TilingProgramIRValueType::kUint16);
constexpr auto kSameType = static_cast<TilingProgramIRValueType>(0);
constexpr auto kBoolType = TilingProgramIRValueType::kBoolean;
constexpr OperationSignature kOperations[] = {
    {0, 0, kSameType},  // none
    {2, kNumericTypes, kSameType},
    {2, kNumericTypes, kSameType},
    {2, kNumericTypes, kSameType},  // add/sub/mul
    {2, kNumericTypes, kSameType},
    {2, kNumericTypes, kSameType},
    {2, kNumericTypes, kSameType},  // div/max/min
    {2, kNumericTypes, kSameType},
    {2, kNumericTypes, TilingProgramIRValueType::kFloat64},  // mod/pow
    {1, kNumericTypes, kSameType},
    {1, kNumericTypes, TilingProgramIRValueType::kFloat64},  // abs/log
    {1, kNumericTypes, kSameType},
    {1, kNumericTypes, kSameType},
    {0, 0, kSameType},  // ceiling/floor/align
    {2, kNumericTypes, kBoolType},
    {2, kNumericTypes, kBoolType},
    {2, kNumericTypes, kBoolType},  // eq/ne/ge
    {2, kNumericTypes, kBoolType},
    {2, kNumericTypes, kBoolType},
    {2, kNumericTypes, kBoolType},  // gt/le/lt
    {1, TypeBit(kBoolType), kBoolType},
    {0, 0, kSameType},
    {0, 0, kSameType},  // not/and/or
    {1, kNumericTypes, kSameType},
    {3, kScalarTypes, TilingProgramIRValueType::kFloat64},       // neg/ternary
    {2, TypeBit(TilingProgramIRValueType::kFloat64), kBoolType}  // native IsEqual(double, double)
};
TilingProgramValueStatus OperationType(TilingProgramIROpcode opcode, TilingProgramIRValueType input, uint8_t arity,
                                       TilingProgramIRValueType *output) {
  if (output == nullptr) {
    return TilingProgramValueStatus::kInvalidArgument;
  }
  const size_t op = static_cast<uint16_t>(opcode);
  if (op >= sizeof(kOperations) / sizeof(kOperations[0]) || kOperations[op].arity == 0) {
    return TilingProgramValueStatus::kUnsupportedOperation;
  }
  const auto &entry = kOperations[op];
  if (entry.arity != arity) {
    return TilingProgramValueStatus::kInvalidArgument;
  }
  const auto type = static_cast<uint8_t>(input);
  if (type >= std::numeric_limits<uint32_t>::digits || (entry.types & (1U << type)) == 0) {
    return TilingProgramValueStatus::kUnsupportedType;
  }
  *output = entry.result == kSameType ? input : entry.result;
  return TilingProgramValueStatus::kOk;
}
template <typename T>
TilingProgramValueStatus Unary(TilingProgramIROpcode opcode, const TilingProgramIRValue &input,
                               TilingProgramIRValue *output) {
  const T value = ReadValue<T>(input);
  if (opcode == TilingProgramIROpcode::kLog) {
    const double result = native_scalar::EvaluateLog(value);
    *output = {TilingProgramIRValueType::kFloat64, RawBits(result)};
    return TilingProgramValueStatus::kOk;
  }
  if (opcode == TilingProgramIROpcode::kCeiling || opcode == TilingProgramIROpcode::kFloor) {
    if constexpr (std::is_floating_point_v<T>) {
      TilingProgramIRValue integer;
      const auto status = CastTilingProgramIRValue(input, TilingProgramIRValueType::kInt64, &integer);
      if (status != TilingProgramValueStatus::kOk) {
        return status;
      }
    }
  } else if constexpr (std::is_signed_v<T> && std::is_integral_v<T>) {
    if (value == std::numeric_limits<T>::min()) {
      return TilingProgramValueStatus::kArithmeticError;
    }
  }
  T result;
  switch (opcode) {
    case TilingProgramIROpcode::kNeg:
      result = -value;
      break;
    case TilingProgramIROpcode::kAbs:
      result = native_scalar::Abs(value);
      break;
    case TilingProgramIROpcode::kCeiling:
      result = native_scalar::Ceiling(value);
      break;
    case TilingProgramIROpcode::kFloor:
      result = native_scalar::Floor(value);
      break;
    default:
      return TilingProgramValueStatus::kUnsupportedOperation;
  }
  *output = {input.type, RawBits(result)};
  return TilingProgramValueStatus::kOk;
}

template <typename T>
TilingProgramValueStatus Ternary(const TilingProgramIRValue &condition, const TilingProgramIRValue &left,
                                 const TilingProgramIRValue &right, TilingProgramIRValue *output) {
  const double value = native_scalar::TernaryOp(condition.raw_bits != 0, ReadValue<T>(left), ReadValue<T>(right));
  *output = {TilingProgramIRValueType::kFloat64, RawBits(value)};
  return TilingProgramValueStatus::kOk;
}
}  // namespace

TilingProgramValueStatus ValidateTilingProgramIRValue(const TilingProgramIRValue &value) {
  const uint32_t width = Width(value.type);
  if (width == 0U) {
    return TilingProgramValueStatus::kUnsupportedType;
  }
  if (width < 64U && (value.raw_bits >> width) != 0U) {
    return TilingProgramValueStatus::kInvalidValue;
  }
  return TilingProgramValueStatus::kOk;
}

TilingProgramValueStatus CastTilingProgramIRValue(const TilingProgramIRValue &input, TilingProgramIRValueType target,
                                                  TilingProgramIRValue *output) {
  if (output == nullptr) {
    return TilingProgramValueStatus::kInvalidArgument;
  }
  const auto status = ValidateTilingProgramIRValue(input);
  if (status != TilingProgramValueStatus::kOk) {
    return status;
  }
  if (Width(target) == 0U) {
    return TilingProgramValueStatus::kUnsupportedType;
  }
  if (input.type == target) {
    *output = input;
    return TilingProgramValueStatus::kOk;
  }
  uint64_t bits = 0U;
  const auto converted = ConvertValue(input, target, bits);
  if (converted == TilingProgramValueStatus::kOk) {
    *output = TilingProgramIRValue{target, bits};
  }
  return converted;
}

TilingProgramValueStatus EvaluateTilingProgramIRBinary(TilingProgramIROpcode opcode, const TilingProgramIRValue &left,
                                                       const TilingProgramIRValue &right,
                                                       TilingProgramIRValue *output) {
  if (output == nullptr) {
    return TilingProgramValueStatus::kInvalidArgument;
  }
  for (const auto *value : {&left, &right}) {
    const auto status = ValidateTilingProgramIRValue(*value);
    if (status != TilingProgramValueStatus::kOk) {
      return status;
    }
  }
  TilingProgramIRValueType result_type;
  const auto supported = GetTilingProgramIRBinaryType(opcode, left.type, right.type, &result_type);
  if (supported != TilingProgramValueStatus::kOk) {
    return supported;
  }
  if (opcode == TilingProgramIROpcode::kMin || opcode == TilingProgramIROpcode::kMax) {
    return NativeMinMaxDispatch(opcode, left, right, result_type, output);
  }
  switch (left.type) {
    case TilingProgramIRValueType::kInt32:
      return Binary<int32_t>(opcode, left, right, output);
    case TilingProgramIRValueType::kUint32:
      return Binary<uint32_t>(opcode, left, right, output);
    case TilingProgramIRValueType::kInt64:
      return Binary<int64_t>(opcode, left, right, output);
    case TilingProgramIRValueType::kUint64:
      return Binary<uint64_t>(opcode, left, right, output);
    case TilingProgramIRValueType::kFloat32:
      return Binary<float>(opcode, left, right, output);
    case TilingProgramIRValueType::kFloat64:
      return Binary<double>(opcode, left, right, output);
    default:
      return TilingProgramValueStatus::kUnsupportedType;
  }
}

TilingProgramValueStatus GetTilingProgramIRBinaryType(TilingProgramIROpcode opcode, TilingProgramIRValueType input,
                                                      TilingProgramIRValueType *output) {
  return GetTilingProgramIRBinaryType(opcode, input, input, output);
}
TilingProgramValueStatus GetTilingProgramIRArithmeticType(TilingProgramIRValueType left, TilingProgramIRValueType right,
                                                          TilingProgramIRValueType *output) {
  using Type = TilingProgramIRValueType;
  using Status = TilingProgramValueStatus;
  if (output == nullptr) {
    return Status::kInvalidArgument;
  }
  if (Width(left) == 0 || Width(right) == 0) {
    return Status::kUnsupportedType;
  }
  if (Width(left) < Width(Type::kInt32)) {
    left = Type::kInt32;
  }
  if (Width(right) < Width(Type::kInt32)) {
    right = Type::kInt32;
  }
  Type result;
  if (left == Type::kFloat64 || right == Type::kFloat64) {
    result = Type::kFloat64;
  } else if (left == Type::kFloat32 || right == Type::kFloat32) {
    result = Type::kFloat32;
  } else if (left == Type::kUint64 || right == Type::kUint64) {
    result = Type::kUint64;
  } else if (left == Type::kInt64 || right == Type::kInt64) {
    result = Type::kInt64;
  } else if (left == Type::kUint32 || right == Type::kUint32) {
    result = Type::kUint32;
  } else {
    result = Type::kInt32;
  }
  *output = result;
  return Status::kOk;
}
TilingProgramValueStatus GetTilingProgramIRBinaryType(TilingProgramIROpcode opcode, TilingProgramIRValueType left,
                                                      TilingProgramIRValueType right,
                                                      TilingProgramIRValueType *output) {
  using Status = TilingProgramValueStatus;
  if (output == nullptr) {
    return Status::kInvalidArgument;
  }
  if (opcode == TilingProgramIROpcode::kMin || opcode == TilingProgramIROpcode::kMax) {
    if (Width(left) == 0 || Width(right) == 0) {
      return Status::kUnsupportedType;
    }
    if (left == right) {
      *output = left;
      return Status::kOk;
    }
    return GetTilingProgramIRArithmeticType(left, right, output);
  }
  if (left != right) {
    return Status::kInvalidArgument;
  }
  return OperationType(opcode, left, kTilingBinaryArity, output);
}
TilingProgramValueStatus GetTilingProgramIRUnaryType(TilingProgramIROpcode opcode, TilingProgramIRValueType input,
                                                     TilingProgramIRValueType *output) {
  return OperationType(opcode, input, kTilingUnaryArity, output);
}
TilingProgramValueStatus EvaluateTilingProgramIRUnary(TilingProgramIROpcode opcode, const TilingProgramIRValue &input,
                                                      TilingProgramIRValue *output) {
  if (output == nullptr) {
    return TilingProgramValueStatus::kInvalidArgument;
  }
  auto status = ValidateTilingProgramIRValue(input);
  if (status != TilingProgramValueStatus::kOk) {
    return status;
  }
  TilingProgramIRValueType type;
  status = GetTilingProgramIRUnaryType(opcode, input.type, &type);
  if (status != TilingProgramValueStatus::kOk) {
    return status;
  }
  if (opcode == TilingProgramIROpcode::kNot) {
    *output = {type, input.raw_bits == 0 ? 1U : 0U};
    return TilingProgramValueStatus::kOk;
  }
  switch (input.type) {
    case TilingProgramIRValueType::kInt32:
      return Unary<int32_t>(opcode, input, output);
    case TilingProgramIRValueType::kUint32:
      return Unary<uint32_t>(opcode, input, output);
    case TilingProgramIRValueType::kInt64:
      return Unary<int64_t>(opcode, input, output);
    case TilingProgramIRValueType::kUint64:
      return Unary<uint64_t>(opcode, input, output);
    case TilingProgramIRValueType::kFloat32:
      return Unary<float>(opcode, input, output);
    case TilingProgramIRValueType::kFloat64:
      return Unary<double>(opcode, input, output);
    default:
      return TilingProgramValueStatus::kUnsupportedType;
  }
}
TilingProgramValueStatus GetTilingProgramIRTernaryType(TilingProgramIROpcode opcode, TilingProgramIRValueType condition,
                                                       TilingProgramIRValueType branch,
                                                       TilingProgramIRValueType *output) {
  if (condition != TilingProgramIRValueType::kBoolean) {
    return TilingProgramValueStatus::kInvalidArgument;
  }
  return OperationType(opcode, branch, kTilingTernaryArity, output);
}
TilingProgramValueStatus EvaluateTilingProgramIRTernary(TilingProgramIROpcode opcode,
                                                        const TilingProgramIRValue &condition,
                                                        const TilingProgramIRValue &left,
                                                        const TilingProgramIRValue &right,
                                                        TilingProgramIRValue *output) {
  if (output == nullptr || left.type != right.type) {
    return TilingProgramValueStatus::kInvalidArgument;
  }
  for (const auto *value : {&condition, &left, &right}) {
    const auto status = ValidateTilingProgramIRValue(*value);
    if (status != TilingProgramValueStatus::kOk) {
      return status;
    }
  }
  TilingProgramIRValueType type;
  const auto status = GetTilingProgramIRTernaryType(opcode, condition.type, left.type, &type);
  if (status != TilingProgramValueStatus::kOk) {
    return status;
  }
  switch (left.type) {
    case TilingProgramIRValueType::kBoolean:
      return Ternary<bool>(condition, left, right, output);
    case TilingProgramIRValueType::kInt8:
      return Ternary<int8_t>(condition, left, right, output);
    case TilingProgramIRValueType::kUint8:
      return Ternary<uint8_t>(condition, left, right, output);
    case TilingProgramIRValueType::kInt16:
      return Ternary<int16_t>(condition, left, right, output);
    case TilingProgramIRValueType::kUint16:
      return Ternary<uint16_t>(condition, left, right, output);
    case TilingProgramIRValueType::kInt32:
      return Ternary<int32_t>(condition, left, right, output);
    case TilingProgramIRValueType::kUint32:
      return Ternary<uint32_t>(condition, left, right, output);
    case TilingProgramIRValueType::kInt64:
      return Ternary<int64_t>(condition, left, right, output);
    case TilingProgramIRValueType::kUint64:
      return Ternary<uint64_t>(condition, left, right, output);
    case TilingProgramIRValueType::kFloat32:
      return Ternary<float>(condition, left, right, output);
    case TilingProgramIRValueType::kFloat64:
      return Ternary<double>(condition, left, right, output);
    default:
      return TilingProgramValueStatus::kUnsupportedType;
  }
}
}  // namespace att
