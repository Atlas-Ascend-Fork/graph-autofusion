/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "ascir.h"
#include "ascir_ops.h"
#include "v35/ascir/generator/v2_ascir_codegen_impl.h"

namespace af {
namespace ascir {
namespace {

void InitNodeAttrs(af::AscGraph &graph, const char *node_type, ge::DataType dtype, const af::Axis &z0) {
  auto node = graph.FindNode(node_type);
  node->attr.api.compute_type = af::ComputeType::kComputeElewise;
  node->attr.api.type = af::ApiType::kAPITypeCompute;
  node->attr.api.unit = af::ComputeUnit::kUnitVector;
  node->attr.sched.loop_axis = z0.id;

  auto &output_attr = node->outputs[0].attr;
  output_attr.vectorized_axis = {z0.id};
  output_attr.vectorized_strides = {af::ops::One};
  output_attr.dtype = dtype;
  output_attr.mem.position = af::Position::kPositionVecOut;
  output_attr.mem.tensor_id = 0;
  output_attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  output_attr.que.id = 1;
  output_attr.opt.merge_scope = af::kIdNone;
}

template <typename OperatorType>
void BuildLegacyGraph(af::AscGraph &graph, const char *node_name, const af::Expression &size, const af::Axis &axis,
                      ge::DataType dtype) {
  OperatorType node(node_name, graph);
  *node.y.axis = {axis.id};
  *node.y.repeats = {size};
  *node.y.strides = {af::ops::One};
  InitNodeAttrs(graph, node_name, dtype, axis);
}

}  // namespace

void CheckLegacyDtypes(const char *node_type) {
  const std::vector<ge::DataType> dtypes = {ge::DT_UINT32, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};

  for (size_t i = 0U; i < dtypes.size(); ++i) {
    af::AscGraph graph((std::string(node_type) + "_legacy_dtype_contract").c_str());
    auto size = graph.CreateSizeVar("size");
    auto axis = graph.CreateAxis("axis", size);
    if (node_type == "Rand") {
      BuildLegacyGraph<af::ascir_op::Rand>(graph, "Rand", size, axis, dtypes[i]);
    } else {
      BuildLegacyGraph<af::ascir_op::Randn>(graph, "Randn", size, axis, dtypes[i]);
    }

    auto node = graph.FindNode(node_type);
    ASSERT_NE(node, nullptr);
    EXPECT_TRUE(RandSeedOffsetAscIrCodegenImplV2().IsNodeValid(*node));
  }
}

TEST(RandDtypeContractSt, LegacyRandDtypesMatchTorchairExpectation) {
  CheckLegacyDtypes("Rand");
}

TEST(RandDtypeContractSt, LegacyRandnDtypesMatchTorchairExpectation) {
  CheckLegacyDtypes("Randn");
}

}  // namespace ascir
}  // namespace af
