/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __PYASCIR_H__
#define __PYASCIR_H__

#include <Python.h>

#ifdef __cplusplus
extern "C" {
#endif

void pyautofuse_type_init();
void pyascir_type_init();
void pyascir_types_type_init();

PyMODINIT_FUNC PyInit_ascir(void);
PyMODINIT_FUNC PyInit_pyautofuse(void);

#ifdef __cplusplus
}
#endif
// 定义统一的算子注册列表
#define REGISTERED_OPS            \
  OP(Data)                        \
  OP(Scalar)                      \
  OP(ScalarData)                  \
  OP(Workspace)                   \
  OP(Output)                      \
  OP(IndexExpr)                   \
  OP(Arange)                      \
  OP(Load)                        \
  OP(Broadcast)                   \
  OP(Store)                       \
  OP(Cast)                        \
  OP(Abs)                         \
  OP(Exp)                         \
  OP(Exp2)                        \
  OP(Floor)                       \
  OP(Fma)                         \
  OP(BitwiseNot)                  \
  OP(BitwiseOr)                   \
  OP(BitwiseXor)                  \
  OP(Ceil)                        \
  OP(Cos)                         \
  OP(Acos)                        \
  OP(Cosh)                        \
  OP(Atan2)                       \
  OP(Sqrt)                        \
  OP(Rsqrt)                       \
  OP(RemovePad)                   \
  OP(Pad)                         \
  OP(Round)                       \
  OP(Neg)                         \
  OP(Relu)                        \
  OP(Reciprocal)                  \
  OP(Erf)                         \
  OP(Sign)                        \
  OP(Tanh)                        \
  OP(Sin)                         \
  OP(Acosh)                       \
  OP(Asinh)                       \
  OP(Asin)                        \
  OP(Atan)                        \
  OP(Atanh)                       \
  OP(Digamma)                     \
  OP(Erfc)                        \
  OP(Erfcx)                       \
  OP(RShift)                      \
  OP(Isnan)                       \
  OP(Max)                         \
  OP(ArgMax)                      \
  OP(Any)                         \
  OP(All)                         \
  OP(Sum)                         \
  OP(Min)                         \
  OP(Mean)                        \
  OP(Prod)                        \
  OP(Softmax)                     \
  OP(Ge)                          \
  OP(Ne)                          \
  OP(Eq)                          \
  OP(Gt)                          \
  OP(Le)                          \
  OP(Add)                         \
  OP(Sub)                         \
  OP(Div)                         \
  OP(Mul)                         \
  OP(TrueDiv)                     \
  OP(Remainder)                   \
  OP(ModifiedBesselI0)            \
  OP(ModifiedBesselI1)            \
  OP(ModifiedBesselK0)            \
  OP(ModifiedBesselK1)            \
  OP(LaguerrePolynomialL)         \
  OP(LegendrePolynomialP)         \
  OP(AiryAi)                      \
  OP(Erfinv)                      \
  OP(Minimum)                     \
  OP(Maximum)                     \
  OP(LogicalOr)                   \
  OP(LogicalNot)                  \
  OP(LogicalAnd)                  \
  OP(Select)                      \
  OP(Sigmoid)                     \
  OP(Concat)                      \
  OP(MatMul)                      \
  OP(MatMulBias)                  \
  OP(BatchMatMul)                 \
  OP(BatchMatMulBias)             \
  OP(Where)                       \
  OP(Gather)                      \
  OP(IndirectLoad)                \
  OP(Transpose)                   \
  OP(BitwiseAnd)                  \
  OP(Ln)                          \
  OP(FloorToInt)                  \
  OP(Fmod)                        \
  OP(Hypot)                       \
  OP(Lgamma)                      \
  OP(Log10)                       \
  OP(LogicalXor)                  \
  OP(Log1p)                       \
  OP(Expm1)                       \
  OP(Log2)                        \
  OP(LShift)                      \
  OP(Mod)                         \
  OP(Lt)                          \
  OP(Pow)                         \
  OP(ClipByValue)                 \
  OP(LeakyRelu)                   \
  OP(Nop)                         \
  OP(Transpose)                   \
  OP(IsFinite)                    \
  OP(IsInf)                       \
  OP(MaskedFill)                  \
  OP(FloorDiv)                    \
  OP(Gelu)                        \
  OP(Split)                       \
  OP(Axpy)                        \
  OP(CopySign)                    \
  OP(Ceil2Int)                    \
  OP(Rand)                        \
  OP(Randn)                       \
  OP(RoundToInt)                  \
  OP(Square)                      \
  OP(Sinh)                        \
  OP(Tan)                         \
  OP(Trunc)                       \
  OP(TruncToInt)                  \
  OP(TruncDiv)                    \
  OP(Xor)                         \
  OP(I0)                          \
  OP(I0e)                         \
  OP(I1e)                         \
  OP(BesselJ0)                    \
  OP(BesselJ1)                    \
  OP(BesselY0)                    \
  OP(BesselY1)                    \
  OP(ScaledModifiedBesselK0)      \
  OP(ScaledModifiedBesselK1)      \
  OP(SphericalBesselJ0)           \
  OP(Ndtr)                        \
  OP(Ndtri)                       \
  OP(LogNdtr)                     \
  OP(NextAfter)                   \
  OP(PolyGamma)                   \
  OP(SignBit)                     \
  OP(Frexp)                       \
  OP(Igamma)                      \
  OP(Igammac)                     \
  OP(Zeta)                        \
  OP(ShiftedChebyshevPolynomialT) \
  OP(ShiftedChebyshevPolynomialU) \
  OP(ShiftedChebyshevPolynomialV) \
  OP(ShiftedChebyshevPolynomialW) \
  OP(ChebyshevPolynomialT)        \
  OP(ChebyshevPolynomialU)        \
  OP(ChebyshevPolynomialV)        \
  OP(ChebyshevPolynomialW)        \
  OP(HermitePolynomialH)          \
  OP(HermitePolynomialHe)         \
  OP(Unsupported)
#endif
