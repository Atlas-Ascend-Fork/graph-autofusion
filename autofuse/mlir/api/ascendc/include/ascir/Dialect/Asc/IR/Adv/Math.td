/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_ADV_MATH_TD
#define ASC_ADV_MATH_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Unary math library operations
//===----------------------------------------------------------------------===//

def AcoshOp : UnaryMathOp<"acosh", "Acosh">;
def AcosOp : UnaryMathOp<"acos", "Acos">;
def AsinhOp : UnaryMathOp<"asinh", "Asinh">;
def AsinOp : UnaryMathOp<"asin", "Asin">;
def AtanhOp : UnaryMathOp<"atanh", "Atanh">;
def AtanOp : UnaryMathOp<"atan", "Atan">;
def CeilOp : UnaryMathOp<"ceil", "Ceil">;
def CoshOp : UnaryMathOp<"cosh", "Cosh">;
def CosOp : UnaryMathOp<"cos", "Cos">;
def DigammaOp : UnaryMathOp<"digamma", "Digamma">;
def ErfcOp : UnaryMathOp<"erfc", "Erfc">;
def ErfOp : UnaryMathOp<"erf", "Erf">;
def FloorOp : UnaryMathOp<"floor", "Floor">;
def FracOp : UnaryMathOp<"frac", "Frac">;
def LgammaOp : UnaryMathOp<"lgamma", "Lgamma">;
def LogOp : UnaryMathOp<"log", "Log">;
def Log2Op : UnaryMathOp<"log2", "Log2">;
def RoundOp : UnaryMathOp<"round", "Round">;
def SignOp : UnaryMathOp<"sign", "Sign">;
def SinhOp : UnaryMathOp<"sinh", "Sinh">;
def SinOp : UnaryMathOp<"sin", "Sin">;
def TanhOp : UnaryMathOp<"tanh", "Tanh">;
def TanOp : UnaryMathOp<"tan", "Tan">;
def TruncOp : UnaryMathOp<"trunc", "Trunc">;

//===----------------------------------------------------------------------===//
// Binary math library operations
//===----------------------------------------------------------------------===//

def PowerOp : BinaryMathOp<"power", "Power">;
def XorOp : BinaryMathOp<"xor", "Xor">;

//===----------------------------------------------------------------------===//
// Other math library operations
//===----------------------------------------------------------------------===//

def AxpyOp : MathLibraryOp<"axpy", "Axpy"> {
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src, AnyType:$scalar,
                   Optional<AnyType>:$sharedTmpBuffer, AnyType:$calCount, AnyType:$isReuseSource);
  let extraClassDeclaration = extraClassDeclarationBase # [{
    SmallVector<Value> getDstTensors()
    {
        SmallVector<Value> dstTensors{getDst()};
        if (auto tensor = getSharedTmpBuffer())
            dstTensors.emplace_back(tensor);
        return dstTensors;
    }
    SmallVector<Value> getSrcTensors() { return {getSrc()}; }
  }];
}

def ClampMaxOp : MathLibraryOp<"clamp_max", "ClampMax"> {
  let arguments = (ins AscendC_LocalTensor:$dst, AnyType:$src, Optional<AnyType>:$sharedTmpBuffer,
                   AnyType:$scalar, AnyType:$calCount, AnyType:$isReuseSource);
  let extraClassDeclaration = extraClassDeclarationBase # [{
    SmallVector<Value> getDstTensors()
    {
        SmallVector<Value> dstTensors{getDst()};
        if (auto tensor = getSharedTmpBuffer())
            dstTensors.emplace_back(tensor);
        return dstTensors;
    }
    SmallVector<Value> getSrcTensors() { return {getSrc()}; }
  }];
}

def ClampMinOp : MathLibraryOp<"clamp_min", "ClampMin"> {
  let arguments = (ins AscendC_LocalTensor:$dst, AnyType:$src, Optional<AnyType>:$sharedTmpBuffer,
                   AnyType:$scalar, AnyType:$calCount, AnyType:$isReuseSource);
  let extraClassDeclaration = extraClassDeclarationBase # [{
    SmallVector<Value> getDstTensors()
    {
        SmallVector<Value> dstTensors{getDst()};
        if (auto tensor = getSharedTmpBuffer())
            dstTensors.emplace_back(tensor);
        return dstTensors;
    }
    SmallVector<Value> getSrcTensors() { return {getSrc()}; }
  }];
}

def CumSumOp : MathLibraryOp<"cumsum", "CumSum"> {
  let arguments = (ins AnyType:$dst, AnyType:$lastRow, AnyType:$src,
                    Optional<AnyType>:$sharedTmpBuffer, AnyType:$lastAxis,
                    AnyType:$reuseSource, AnyType:$outputLastRow);
  let extraClassDeclaration = extraClassDeclarationBase # [{
    SmallVector<Value> getDstTensors()
    {
      SmallVector<Value> dstTensors{getDst()};
      if (auto tensor = getSharedTmpBuffer())
          dstTensors.emplace_back(tensor);
      return dstTensors;
    }
    SmallVector<Value> getSrcTensors() { return {getSrc()}; }
  }];
}

def ExpOp : MathLibraryOp<"exp", "Exp"> {
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src, AnyType:$calCount,
                   AnyType:$taylorExpandLevel, Optional<AnyType>:$sharedTmpBuffer, AnyType:$isReuseSource);
  let extraClassDeclaration = extraClassDeclarationBase # [{
    SmallVector<Value> getDstTensors()
    {
        SmallVector<Value> dstTensors{getDst()};
        if (auto tensor = getSharedTmpBuffer())
            dstTensors.emplace_back(tensor);
        return dstTensors;
    }
    SmallVector<Value> getSrcTensors() { return {getSrc()}; }
  }];
}

#endif // ASC_ADV_MATH_TD
