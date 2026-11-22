/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __V1_ASCIR_ATT_IMPL__
#define __V1_ASCIR_ATT_IMPL__

#include "ascendc_ir.h"
#include "graph/def_types.h"

namespace af {
namespace ascir {
#define JOIN(a, b) a##b
#define IR_NAME(ir_name) #ir_name
#define REG_ASC_IR_ATT_V1_CLASS_DEFINE(ir_name)             \
  class JOIN(ir_name, AscIrAttImpl) : public AscIrAtt {     \
   public:                                                  \
    JOIN(ir_name, AscIrAttImpl)() = default;                \
    ~JOIN(ir_name, AscIrAttImpl)() = default;               \
    [[nodiscard]] void *GetApiPerf() const override {       \
      static char_t api_perf_name[] = IR_NAME(ir_name);     \
      return PtrToPtr<void, char_t>(api_perf_name);         \
    }                                                       \
    virtual void *GetAscendCApiPerfTable() const override { \
      static char_t api_perf_name[] = IR_NAME(ir_name);     \
      return PtrToPtr<void, char_t>(api_perf_name);         \
    }                                                       \
  }
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Add);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Gather);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Abs);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Broadcast);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Cast);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Div);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Erf);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Exp);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(LogicalAnd);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(LogicalOr);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(LogicalNot);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Maximum);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Minimum);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Min);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Mul);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Neg);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Reciprocal);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Relu);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(ReduceAll);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(ReduceAny);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(ReduceMax);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(ReduceArgmax);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(ReduceArgMaxMultiRPhase1);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(ReduceArgMaxMultiRPhase2);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(ReduceMean);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(ReduceMin);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(ReduceSum);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(ReduceProd);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(RemovePad);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Rsqrt);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Select);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Sign);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Sqrt);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Sub);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Sum);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Tanh);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Where);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Ge);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Eq);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Ne);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Gt);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Le);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Lt);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Ub2ub);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Load);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Store);
// 不需要建模的ASCIR
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Data);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Scalar);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(IndexExpr);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Output);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Workspace);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(MatMul);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Conv2D);
// 目前无建模的ASCIR
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Pad);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Nop);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Ln);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Isnan);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(IsFinite);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(IsInf);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(MaskedFill);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Max);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Mean);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Prod);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Any);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(All);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Sigmoid);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(TrueDiv);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Remainder);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Pow);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(ClipByValue);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Concat);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(LeakyRelu);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(BitwiseAnd);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Transpose);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(FloorDiv);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Gelu);
REG_ASC_IR_ATT_V1_CLASS_DEFINE(Axpy);
}  // namespace ascir
}  // namespace af

#endif  //__ASCIR_ATT_IMPL__
