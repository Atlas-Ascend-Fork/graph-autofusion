/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATT_PROGRAM_TILING_PROGRAM_VALUE_H_
#define ATT_PROGRAM_TILING_PROGRAM_VALUE_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace att {
constexpr uint32_t kTilingUnaryArity = 1U;
constexpr uint32_t kTilingBinaryArity = 2U;
constexpr uint32_t kTilingTernaryArity = 3U;

// Copy the native representation between equal-width scalar types.
template <class To, class From>
To CopyTilingScalarBits(const From &source) {
  static_assert(sizeof(To) == sizeof(From), "Bit copy requires equal widths");
  static_assert(std::is_arithmetic_v<To> && std::is_arithmetic_v<From>, "Bit copy requires scalar types");
  To result{};
  const auto *bytes = reinterpret_cast<const std::byte *>(&source);
  std::copy_n(bytes, sizeof(result), reinterpret_cast<std::byte *>(&result));
  return result;
}

// The first four IDs retain the reference IR identities. These are not AfValueTypeTag IDs.
enum class TilingProgramIRValueType : uint8_t {
  kBoolean = 1U,
  kInt64 = 2U,
  kFloat64 = 3U,
  kUint64 = 4U,
  kFloat32 = 5U,
  kInt8 = 6U,
  kUint8 = 7U,
  kInt16 = 8U,
  kUint16 = 9U,
  kInt32 = 10U,
  kUint32 = 11U,
};

enum class TilingProgramIROpcode : uint16_t {
  kNone = 0U,
  kAdd = 1U,
  kSub = 2U,
  kMul = 3U,
  kDiv = 4U,
  kMax = 5U,
  kMin = 6U,
  kMod = 7U,
  kPow = 8U,
  kAbs = 9U,
  kLog = 10U,
  kCeiling = 11U,
  kFloor = 12U,
  kAlign = 13U,
  kEq = 14U,
  kNe = 15U,
  kGe = 16U,
  kGt = 17U,
  kLe = 18U,
  kLt = 19U,
  kNot = 20U,
  kLogicalAnd = 21U,
  kLogicalOr = 22U,
  kNeg = 23U,
  kTernary = 24U,
  kIsEqual = 25U,
};

struct TilingProgramIRValue {
  TilingProgramIRValueType type = TilingProgramIRValueType::kInt64;
  uint64_t raw_bits = 0U;
};

enum class TilingProgramValueStatus {
  kOk,
  kInvalidArgument,
  kUnsupportedType,
  kInvalidValue,
  kArithmeticError,
  kUnsupportedOperation,
};

// Narrow values occupy the low width bits, with all unused bits zero; bool is exactly 0 or 1.
// Float payloads preserve every bit, including NaN and signed zero.
TilingProgramValueStatus ValidateTilingProgramIRValue(const TilingProgramIRValue &value);

// One explicit C++ scalar conversion; no implicit promotion or intermediate double for integer input.
// Integer narrowing follows the verified two's-complement numeric profile. Float-to-integer
// conversion rejects nonfinite/unrepresentable truncated values before the native cast.
// Floating conversions require the caller's verified IEEE numeric profile and rounding environment.
// Failure leaves output unchanged. This internal operation does not advertise runtime capabilities.
TilingProgramValueStatus CastTilingProgramIRValue(const TilingProgramIRValue &input, TilingProgramIRValueType target,
                                                  TilingProgramIRValue *output);
// Ordinary arithmetic/comparisons require the same promoted compute type; no implicit casts.
// Native Min/Max preserve both original operand types, including mixed and narrow values.
// Min/Max retain native double comparison and right-side ties; no control-flow operation is eager.
TilingProgramValueStatus EvaluateTilingProgramIRBinary(TilingProgramIROpcode opcode, const TilingProgramIRValue &left,
                                                       const TilingProgramIRValue &right, TilingProgramIRValue *output);
// Static operation/type lookup shared by verifier and evaluator; does not execute values.
TilingProgramValueStatus GetTilingProgramIRBinaryType(TilingProgramIROpcode opcode, TilingProgramIRValueType input,
                                                      TilingProgramIRValueType *output);
// C++ usual arithmetic conversions, including integer promotions; static type query only.
TilingProgramValueStatus GetTilingProgramIRArithmeticType(TilingProgramIRValueType left, TilingProgramIRValueType right,
                                                          TilingProgramIRValueType *output);
// Two-input signature. Only native Min/Max accept different operand types.
TilingProgramValueStatus GetTilingProgramIRBinaryType(TilingProgramIROpcode opcode, TilingProgramIRValueType left,
                                                      TilingProgramIRValueType right, TilingProgramIRValueType *output);
TilingProgramValueStatus GetTilingProgramIRUnaryType(TilingProgramIROpcode opcode, TilingProgramIRValueType input,
                                                     TilingProgramIRValueType *output);
TilingProgramValueStatus EvaluateTilingProgramIRUnary(TilingProgramIROpcode opcode, const TilingProgramIRValue &input,
                                                      TilingProgramIRValue *output);
// Eager native TernaryOp helper, distinct from short-circuit control flow. Both branch values must
// already have the same C++ conditional compute type; the native helper result is always double.
TilingProgramValueStatus GetTilingProgramIRTernaryType(TilingProgramIROpcode opcode, TilingProgramIRValueType condition,
                                                       TilingProgramIRValueType branch,
                                                       TilingProgramIRValueType *output);
TilingProgramValueStatus EvaluateTilingProgramIRTernary(TilingProgramIROpcode opcode,
                                                        const TilingProgramIRValue &condition,
                                                        const TilingProgramIRValue &left,
                                                        const TilingProgramIRValue &right,
                                                        TilingProgramIRValue *output);
}  // namespace att

#endif  // ATT_PROGRAM_TILING_PROGRAM_VALUE_H_
