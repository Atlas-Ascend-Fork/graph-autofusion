/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Cast.cpp - AFIR cast op implementation -------------------*- C++ -*-===//
//
// (输入, 输出) 对表。**两张，不是一张**：mem 侧 legacy 的 `kCastTypePairs`
// （`ascir_builtin_ops_v1.cpp`）36 对，reg 侧那一版（`ascir_builtin_ops_v2.cpp`）43 对
// ——两边互有对方没有的行，mem 独有 3 对（i4 那一族），reg 独有 10 对（i8 / i64 那两族
// 更全）。
//
// **三个读点共用这里的查表**：ODS 的 `verify()` 查并集（46 对），两个
// implementation 的 capability 判据各查自己那一半（`UnaryElementwise.cpp` 的
// `ResultElementType::CastPair`）。分开写就会分开错，而错法是静默的——一边建得出 op、
// 另一边永远选不出 implementation。与 `trueDivResultElementType` 同一条。
//
//===----------------------------------------------------------------------===//

#include "dialect/AFIR/IR/AFIR.h"

#include "dialect/AFIR/IR/AFIRElementwiseSupport.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::afir;

namespace {

enum class CastTypeId : unsigned {
  F32 = 0,
  F16 = 1,
  I64 = 2,
  I32 = 3,
  I16 = 4,
  BF16 = 5,
  I8 = 6,
  UI8 = 7,
  I4 = 8,
  UI32 = 9,
  UI16 = 10,
  UI64 = 11,
  Unknown = 999
};

CastTypeId getSignedIntegerTypeId(unsigned width) {
  switch (width) {
    case 64:
      return CastTypeId::I64;
    case 32:
      return CastTypeId::I32;
    case 16:
      return CastTypeId::I16;
    case 8:
      return CastTypeId::I8;
    case 4:
      return CastTypeId::I4;
    default:
      return CastTypeId::Unknown;
  }
}

CastTypeId getUnsignedIntegerTypeId(unsigned width) {
  switch (width) {
    case 64:
      return CastTypeId::UI64;
    case 32:
      return CastTypeId::UI32;
    case 16:
      return CastTypeId::UI16;
    case 8:
      return CastTypeId::UI8;
    default:
      return CastTypeId::Unknown;
  }
}

CastTypeId getTypeId(Type type) {
  if (auto floatTy = llvm::dyn_cast<FloatType>(type)) {
    if (floatTy.isF32()) return CastTypeId::F32;
    if (floatTy.isF16()) return CastTypeId::F16;
    if (floatTy.isBF16()) return CastTypeId::BF16;
    return CastTypeId::Unknown;
  }
  if (auto intTy = llvm::dyn_cast<IntegerType>(type)) {
    if (intTy.isUnsigned()) return getUnsignedIntegerTypeId(intTy.getWidth());
    if (intTy.isSignless() || intTy.isSigned()) return getSignedIntegerTypeId(intTy.getWidth());
  }
  return CastTypeId::Unknown;
}

using CastPairSet = llvm::DenseSet<std::pair<CastTypeId, CastTypeId>>;

// mem 侧：`ascir_builtin_ops_v1.cpp` 的 `kCastTypePairs`，36 对，逐行抄来的。
const CastPairSet &getMemBasedCastPairs() {
  using CT = CastTypeId;
  static const CastPairSet pairs = {{CT::F32, CT::F32},  {CT::F32, CT::F16},  {CT::F32, CT::I64},  {CT::F32, CT::I32},
                                    {CT::F32, CT::I16},  {CT::F32, CT::BF16}, {CT::F16, CT::F32},  {CT::F16, CT::I32},
                                    {CT::F16, CT::I16},  {CT::F16, CT::I8},   {CT::F16, CT::UI8},  {CT::F16, CT::I4},
                                    {CT::F16, CT::I64},  {CT::I4, CT::F16},   {CT::UI8, CT::F16},  {CT::UI8, CT::F32},
                                    {CT::UI8, CT::I32},  {CT::UI8, CT::I16},  {CT::UI8, CT::I8},   {CT::UI8, CT::I4},
                                    {CT::I8, CT::F16},   {CT::I8, CT::UI8},   {CT::I16, CT::F16},  {CT::I16, CT::F32},
                                    {CT::I16, CT::UI16}, {CT::I32, CT::F32},  {CT::I32, CT::I64},  {CT::I32, CT::I16},
                                    {CT::I32, CT::F16},  {CT::I32, CT::UI32}, {CT::I64, CT::I32},  {CT::BF16, CT::F32},
                                    {CT::BF16, CT::I32}, {CT::UI32, CT::I32}, {CT::UI16, CT::I16}, {CT::UI64, CT::I64}};
  return pairs;
}

// reg 侧：`ascir_builtin_ops_v2.cpp` 的同名表，44 对里的 43 对。
// **少的那一对是 `{DT_FLOAT, DT_BOOL}`**——AFIR 不建模 bool（`ui8` 在这一层是一个字节的
// 无符号整数，不是真值类型），把它写成 `f32 -> ui8` 会多说一件 reg 侧没有的供给。
const CastPairSet &getRegBasedCastPairs() {
  using CT = CastTypeId;
  static const CastPairSet pairs = {
      {CT::F32, CT::F32},  {CT::F32, CT::F16},  {CT::F32, CT::I64},  {CT::F32, CT::I32},  {CT::F32, CT::I16},
      {CT::F32, CT::BF16}, {CT::F32, CT::I8},   {CT::F16, CT::F32},  {CT::F16, CT::I32},  {CT::F16, CT::I16},
      {CT::F16, CT::I8},   {CT::F16, CT::UI8},  {CT::F16, CT::I64},  {CT::UI8, CT::F16},  {CT::UI8, CT::F32},
      {CT::UI8, CT::I32},  {CT::UI8, CT::I16},  {CT::UI8, CT::I8},   {CT::UI8, CT::I64},  {CT::I8, CT::F16},
      {CT::I8, CT::UI8},   {CT::I8, CT::F32},   {CT::I8, CT::I16},   {CT::I16, CT::F16},  {CT::I16, CT::F32},
      {CT::I16, CT::UI16}, {CT::I16, CT::I8},   {CT::I16, CT::UI8},  {CT::I32, CT::F32},  {CT::I32, CT::I64},
      {CT::I32, CT::I16},  {CT::I32, CT::F16},  {CT::I32, CT::UI32}, {CT::I64, CT::I32},  {CT::I64, CT::F32},
      {CT::I64, CT::UI8},  {CT::I64, CT::UI64}, {CT::I64, CT::F16},  {CT::BF16, CT::F32}, {CT::BF16, CT::I32},
      {CT::UI32, CT::I32}, {CT::UI16, CT::I16}, {CT::UI64, CT::I64}};
  return pairs;
}

// 并集，46 对。ODS 的 `verify()` 查它：这一层只说「有没有哪一侧供得起」，
// 选哪一侧是 lowering 的事。
const CastPairSet &getSupportedCastPairs() {
  static const CastPairSet pairs = [] {
    CastPairSet all = getMemBasedCastPairs();
    all.insert(getRegBasedCastPairs().begin(), getRegBasedCastPairs().end());
    return all;
  }();
  return pairs;
}

bool isSupportedCast(const CastPairSet &pairs, Type inputType, Type outputType) {
  const CastTypeId inputId = getTypeId(inputType);
  const CastTypeId outputId = getTypeId(outputType);
  if (inputId == CastTypeId::Unknown || outputId == CastTypeId::Unknown) {
    return false;
  }
  return pairs.count({inputId, outputId}) > 0;
}

}  // namespace

bool mlir::afir::elementwise::isSupportedCastPair(CastSupply supply, Type input, Type result) {
  const CastPairSet &pairs = supply == CastSupply::MemBased ? getMemBasedCastPairs() : getRegBasedCastPairs();
  return isSupportedCast(pairs, getElementTypeOrSelf(input), getElementTypeOrSelf(result));
}

LogicalResult mlir::afir::elementwise::verifyCastPair(Operation *op, Type input, Type result) {
  const Type inputElement = getElementTypeOrSelf(input);
  const Type resultElement = getElementTypeOrSelf(result);
  if (!isSupportedCast(getSupportedCastPairs(), inputElement, resultElement)) {
    return op->emitOpError("unsupported cast from ") << inputElement << " to " << resultElement;
  }
  return success();
}
