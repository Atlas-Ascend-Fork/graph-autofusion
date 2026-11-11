/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascendc_ir.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace af {
namespace ascir {
constexpr uint32_t INT32_SIZE = 4U;
constexpr uint32_t INT64_SIZE = 8U;
constexpr int32_t ONE = 1;
constexpr int32_t TWO = 2;
constexpr int32_t FOUR = 4;
constexpr int32_t FIVE = 5;
constexpr int32_t EIGHT = 8;
constexpr int32_t NINE = 9;
constexpr int32_t TEN = 10;
constexpr int32_t INDICES_DIV_INT32 = 27;
constexpr int32_t INDICES_MUL_INT32 = 18;
constexpr int32_t INDICES_ADD_INT32 = 24;
constexpr int32_t INDICES_DIV_INT64 = 6;
constexpr int32_t INDICES_MUL_INT64 = 16;
constexpr int32_t INDICES_ADD_INT64 = 44;
constexpr int32_t CRITICAL_POINT_INT32 = 24576;
constexpr int32_t CRITICAL_POINT_INT64 = 44237;
constexpr int32_t PARAM_UPPER_LIMIT_ONE_AXIS = 30000;
constexpr int32_t PARAM_UPPER_LIMIT_MIDDLE_AXIS = 20000;
constexpr int32_t INDEX_UPPER_LIMIT_MIDDLE_AXIS = 750;
constexpr int32_t MIN_TEMP_SIZE = 32;
constexpr int32_t ONE_UNIT = 1024;
constexpr int32_t EXTRA_REDUNDANCY = 1024;
Expression CalOneAxisTempSize(Expression param_size, Expression index_size, Expression param_type_size,
                              Expression index_type_size) {
  Expression TempSize;
  Expression indices_div;
  Expression indices_add;
  Expression param_div;
  Expression critical_point;
  if (SymbolicUtils::StaticCheckEq(index_type_size, Symbol(INT32_SIZE)) == TriBool::kTrue) {
    indices_div = Symbol(INDICES_DIV_INT32);
    param_div = Symbol(NINE);
    indices_add = Symbol(INDICES_ADD_INT32);
    critical_point = Symbol(CRITICAL_POINT_INT32);
  } else if (SymbolicUtils::StaticCheckEq(index_type_size, Symbol(INT64_SIZE)) == TriBool::kTrue) {
    indices_div = Symbol(INDICES_DIV_INT64);
    param_div = Symbol(EIGHT);
    indices_add = Symbol(INDICES_ADD_INT64);
    critical_point = Symbol(CRITICAL_POINT_INT64);
  }
  Expression indices_tmp = sym::Div(index_size, indices_div);
  Expression param_tmp = sym::Mul(param_size, param_type_size);
  Expression judge_tmp = sym::Add(indices_tmp, param_tmp);
  TempSize = judge_tmp + Symbol(EXTRA_REDUNDANCY);
  if (SymbolicUtils::StaticCheckGt(judge_tmp, critical_point) == TriBool::kTrue) {
    TempSize = sym::Add(sym::Mul(sym::Mul(sym::Div(param_div, Symbol(TEN)), param_type_size), param_size),
                        sym::Mul(indices_add, Symbol(ONE_UNIT)));
    // The heuristic can reserve less than the params payload for 8-byte values.
    // Use the existing SIMT path instead of reserving an unusable full-load buffer.
    if (SymbolicUtils::StaticCheckLt(TempSize, param_tmp + Symbol(EXTRA_REDUNDANCY)) == TriBool::kTrue) {
      return Symbol(MIN_TEMP_SIZE);
    }
  }
  return TempSize;
}
std::vector<std::unique_ptr<TmpBufDesc>> CalcGatherTmpSizeV2(const AscNode &node) {
  std::vector<std::unique_ptr<TmpBufDesc>> tmpBufDescs;
  Expression TempSize;
  AscNodeInputs node_inputs = node.inputs;
  AscNodeOutputs node_outputs = node.outputs;
  auto typeSizeT1 = Expression(Symbol(GetSizeByDataType(node_inputs[0].attr.dtype)));
  auto typeSizeT2 = Expression(Symbol(GetSizeByDataType(node_inputs[1].attr.dtype)));
  auto node_attr = node.attr.ir_attr.get();
  int64_t gather_axis_value = -1;
  node_attr->GetAttrValue("axis", gather_axis_value);
  Expression param_size = Symbol(ONE);
  Expression indices_size = Symbol(ONE);
  bool is_dynamic = false;
  bool is_middle_axis = false;
  for (int32_t i = 0; i < static_cast<int32_t>(node_inputs[0].attr.repeats.size()); i++) {
    if (node_inputs[0].attr.repeats[i].IsConstExpr() == false) {
      is_dynamic = true;
    }
    param_size = sym::Mul(param_size, node_inputs[0].attr.repeats[i]);
  }
  for (int32_t i = 0; i < static_cast<int32_t>(node_inputs[1].attr.repeats.size()); i++) {
    if (node_inputs[1].attr.repeats[i].IsConstExpr() == false) {
      is_dynamic = true;
    }
    indices_size = sym::Mul(indices_size, node_inputs[1].attr.repeats[i]);
  }
  is_middle_axis =
      (gather_axis_value != static_cast<int32_t>(node_inputs[0].attr.repeats.size() - 1) &&
       SymbolicUtils::StaticCheckLe(param_size, Symbol(PARAM_UPPER_LIMIT_MIDDLE_AXIS)) == TriBool::kTrue &&
       SymbolicUtils::StaticCheckGe(indices_size, Symbol(INDEX_UPPER_LIMIT_MIDDLE_AXIS)) == TriBool::kTrue);
  GE_CHK_BOOL_RET_SPECIAL_STATUS(node_inputs.Size() < TWO, tmpBufDescs, "node.inputs.Size less than TWO");
  if (is_dynamic == false) {
    if (node_inputs[0].attr.repeats.size() == 1 &&
        SymbolicUtils::StaticCheckLe(param_size, Symbol(PARAM_UPPER_LIMIT_ONE_AXIS)) == TriBool::kTrue) {
      TempSize = CalOneAxisTempSize(param_size, indices_size, typeSizeT1, typeSizeT2);
    } else if (is_middle_axis) {
      TempSize = sym::Mul(typeSizeT1, param_size);
    } else {
      TempSize = Symbol(MIN_TEMP_SIZE);
    }
  } else {
    TempSize = Symbol(MIN_TEMP_SIZE);
  }
  TmpBufDesc desc = {TempSize, -1};
  tmpBufDescs.emplace_back(std::make_unique<TmpBufDesc>(desc));
  return tmpBufDescs;
}
}  // namespace ascir
}  // namespace af
