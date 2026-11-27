/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"
#include "base/att_const_values.h"
#include "api_perf_register/ascendc_api_perf.h"

using namespace att;
using namespace af::sym;

TEST(DataMovePerfST, CoversDensForDataMoveAndVectorCompute) {
  const std::vector<std::string> op_types = {"T_LoadTscm",  "CopyL2ToL1",     "T_LoadA",      "T_LoadB",
                                             "CopyL0CToL2", "T_FixPipeTrans", "VectorCompute"};
  for (const auto &op_type : op_types) {
    std::vector<TensorShapeInfo> input_shapes(1);
    std::vector<TensorShapeInfo> output_shapes(1);
    input_shapes[0].data_type = "float16";
    input_shapes[0].data_type_size = 2;
    input_shapes[0].dims = {CreateExpr(64), CreateExpr(128)};
    output_shapes[0].data_type = "float16";
    output_shapes[0].data_type_size = 2;
    output_shapes[0].dims = {CreateExpr(64), CreateExpr(128)};
    PerfOutputInfo perf_res;
    NodeInfo node;
    auto perf = GetPerfFunc(op_type);
    ASSERT_NE(perf, nullptr);
    EXPECT_EQ(perf(input_shapes, output_shapes, node, perf_res), af::SUCCESS);
  }
}
