/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "default_reg_func.h"

namespace af {
namespace ascir {

Expression GetCompareNormalNoLoopTmpSize(const AscNode &node, const std::vector<uint32_t> &axis_ids) {
  auto node_inputs = node.inputs;
  uint8_t typeSize = node_inputs[0].attr.dtype == ge::DataType::DT_FLOAT16 ? 2 : 4;
  EXPECT_SYMBOL_NE(Symbol(typeSize), Symbol(0));
  // 对齐 compare_v2.h CompareNormalNoLoop 的 tmp 布局：select_out 是 half 张量，从 compare_out 末尾 32B
  // 对齐处起始；每个 select repeat 写 8 个 block(256B)，字节数是 SetSize 元素数(128 * repeat)的 2 倍。
  Expression compare_out_size = Symbol(256 / 8 / typeSize) * node_inputs[0].attr.repeats[axis_ids[0]];
  Expression select_out_size =
      Symbol(256) * ((node_inputs[0].attr.repeats[axis_ids[0]] * Symbol(2) + Symbol(typeSize - 1)) / Symbol(typeSize));
  return sym::Align(compare_out_size, 32) + select_out_size;  // 32字节对齐
}

std::vector<std::unique_ptr<TmpBufDesc>> GetCompareNormalTmpSize(const AscNode &node, const std::string &mode) {
  auto node_inputs = node.inputs;
  std::vector<uint32_t> axis_ids = {UINT32_MAX, UINT32_MAX};
  // 在axis中找到vectorized_axis最后一个元素在axis的位置
  uint32_t vec_last_axis_pos_in_axis = std::find(node_inputs[0].attr.axis.begin(), node_inputs[0].attr.axis.end(),
                                                 node_inputs[0].attr.vectorized_axis.back()) -
                                       node_inputs[0].attr.axis.begin();
  for (auto vec_axis : node_inputs[0].attr.vectorized_axis) {
    auto pos = std::find(node_inputs[0].attr.axis.begin(), node_inputs[0].attr.axis.end(), vec_axis);
    GE_ASSERT_TRUE(pos != node_inputs[0].attr.axis.end(), "Incorrect axis ID in vectorized_axis");
    uint32_t axis_id = static_cast<uint32_t>(pos - node_inputs[0].attr.axis.begin());
    if (axis_id < axis_ids[1]) {
      if (axis_id < axis_ids[0]) {
        axis_ids[1] = axis_ids[0];
        axis_ids[0] = axis_id;
      } else {
        axis_ids[1] = axis_id;
      }
    }
  }
  if (axis_ids[1] != UINT32_MAX) {
    // int64 比较的 tmp 需求 = 单行字节数 × 一次调用处理的行数。行数取“除最内向量化轴外其余向量化轴的
    // repeats 乘积”。这里不能用 vectorized_strides 相乘：退化轴(repeat==1/stride==0)会把整项打成 0，
    // tiling 就会给该算子分配 0 字节 tmp，device 侧按 repeat_times 切分时直接越过 UB。
    Expression vec_rows = sym::kSymbolOne;
    for (size_t i = 0UL; i + 1UL < node_inputs[0].attr.vectorized_axis.size(); ++i) {
      auto axis_pos = std::find(node_inputs[0].attr.axis.begin(), node_inputs[0].attr.axis.end(),
                                node_inputs[0].attr.vectorized_axis[i]);
      GE_ASSERT_TRUE(axis_pos != node_inputs[0].attr.axis.end(), "Incorrect vectorized axis ID [%ld]",
                     static_cast<int64_t>(node_inputs[0].attr.vectorized_axis[i]));
      const size_t axis_index = static_cast<size_t>(std::distance(node_inputs[0].attr.axis.begin(), axis_pos));
      vec_rows = sym::Mul(vec_rows, node_inputs[0].attr.repeats[axis_index]);
    }
    GELOGD("[GetInputSize] repeats per call (rows) is: %s", vec_rows.Str().get());
    if (node_inputs[0].attr.dtype == ge::DataType::DT_FLOAT16 || node_inputs[0].attr.dtype == ge::DataType::DT_FLOAT ||
        (node_inputs[0].attr.dtype == ge::DataType::DT_INT32 && (mode == "EQ" || mode == "NE"))) {
      GELOGD("[GetInputSize] axis id is: %u, %u", axis_ids[0], axis_ids[1]);
      GELOGD("[GetInputSize] inputs[0].repeat is: %s, %s", node_inputs[0].attr.repeats[axis_ids[0]].Str().get(),
             node_inputs[0].attr.repeats[axis_ids[1]].Str().get());
      GELOGD("[GetInputSize] inputs[0].vectorized_strides is: %s, %s",
             node_inputs[0].attr.vectorized_strides[0].Str().get(),
             node_inputs[0].attr.vectorized_strides[1].Str().get());
      Expression input_size =
          Symbol(256) * node_inputs[0].attr.repeats[axis_ids[1]] * node_inputs[0].attr.vectorized_strides[1];
      Expression total_size = input_size + Symbol(32) * node_inputs[0].attr.repeats[axis_ids[1]] *
                                               node_inputs[0].attr.vectorized_strides[1];
      return GetTmpBuffer(sym::Max(total_size, GetCompareNormalNoLoopTmpSize(node, axis_ids)));
    } else if (node_inputs[0].attr.dtype == ge::DataType::DT_INT32) {
      GELOGD("[GetInputSize] axis id is: %u", axis_ids[1]);
      GELOGD("[GetInputSize] inputs[1].repeat is: %s", node_inputs[0].attr.repeats[axis_ids[0]].Str().get());
      GELOGD("[GetInputSize] inputs[1].vectorized_strides is: %s",
             node_inputs[0].attr.vectorized_strides[1].Str().get());
      Expression input_size = node_inputs[0].attr.repeats[axis_ids[0]] * node_inputs[0].attr.vectorized_strides[1];
      Expression total_size = input_size * Symbol(256);
      return GetTmpBuffer(total_size);
    } else if (node_inputs[0].attr.dtype == ge::DataType::DT_INT64) {
      if (mode == "EQ" || mode == "NE") {
        // CompareExtendInt64EqNe: ONE_REPEAT_BYTE_SIZE + 2 * ONE_REPEAT_BYTE_SIZE * rows
        Expression total_size = vec_rows * Symbol(256) * Symbol(2) + Symbol(256);
        return GetTmpBuffer(total_size);
      } else {
        // 找到last axis, 并32字节对齐
        Expression last_axis_size = sym::Align(node_inputs[0].attr.repeats[vec_last_axis_pos_in_axis], 32);
        GELOGD("[GetInputSize] last axis size is: %s", last_axis_size.Str().get());
        // CompareExtendInt64GtGeLe: 40 * rows * Align(last_axis, 32)
        Expression total_size = vec_rows * last_axis_size * Symbol(8) * Symbol(5);
        GELOGD("[GetInputSize] int64 compare tmp size is: %s", total_size.Str().get());
        return GetTmpBuffer(total_size);
      }
    } else {
      return CalcDefaultTmpSize(node);
    }
  } else {
    GELOGD("[GetInputSize] axis id is: %u", axis_ids[0]);
    GELOGD("[GetInputSize] inputs[0].repeat is: %s", node_inputs[0].attr.repeats[axis_ids[0]].Str().get());
    GELOGD("[GetInputSize] inputs[0].vectorized_strides is: %s", node_inputs[0].attr.vectorized_strides[0].Str().get());
    Expression input_size = node_inputs[0].attr.repeats[axis_ids[0]] * node_inputs[0].attr.vectorized_strides[0];
    Expression total_size = Symbol(ge::GetSizeByDataType(node_inputs[0].attr.dtype)) * input_size;
    if (node_inputs[0].attr.dtype == ge::DataType::DT_INT64) {
      total_size = total_size * Symbol(5);  // Five times the temp buf is required.
    }
    return GetTmpBuffer(total_size);
  }
}

std::vector<std::unique_ptr<TmpBufDesc>> CalcGeTmpSize(const AscNode &node) {
  return GetCompareNormalTmpSize(node, "GE");
}

std::vector<std::unique_ptr<TmpBufDesc>> CalcEqTmpSize(const AscNode &node) {
  return GetCompareNormalTmpSize(node, "EQ");
}

std::vector<std::unique_ptr<TmpBufDesc>> CalcNeTmpSize(const AscNode &node) {
  return GetCompareNormalTmpSize(node, "NE");
}

std::vector<std::unique_ptr<TmpBufDesc>> CalcGtTmpSize(const AscNode &node) {
  return GetCompareNormalTmpSize(node, "GT");
}

std::vector<std::unique_ptr<TmpBufDesc>> CalcLeTmpSize(const AscNode &node) {
  return GetCompareNormalTmpSize(node, "LE");
}

std::vector<std::unique_ptr<TmpBufDesc>> CalcLtTmpSize(const AscNode &node) {
  return GetCompareNormalTmpSize(node, "LT");
}

}  // namespace ascir
}  // namespace af
