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

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir/generator/v2_ascir_codegen_impl.h"

using namespace ge;
using namespace af::ascir_op;

namespace {
TEST(AtanCodegen, AtanFloatDtypesConvertToFloat) {
  for (const auto dtype : {DT_FLOAT, DT_FLOAT16, DT_BF16}) {
    SCOPED_TRACE(::testing::Message() << "dtype=" << static_cast<int32_t>(dtype));
    af::AscGraph graph("atan_dtype_conversion");

    af::ascir_op::Data data("data", graph);
    data.y.dtype = dtype;

    Atan atan("atan");
    atan.x = data.y;
    atan.y.dtype = dtype;
    graph.AddNode(atan);

    auto atan_node = graph.FindNode("atan");
    ASSERT_NE(atan_node, nullptr);

    af::ascir::AtanAscIrCodegenImplV2 codegen;
    const auto [input_dtypes, output_dtypes] = codegen.GetConversionDtype(*atan_node);
    ASSERT_EQ(input_dtypes.size(), 1U);
    ASSERT_EQ(output_dtypes.size(), 1U);
    EXPECT_EQ(input_dtypes[0], DT_FLOAT);
    EXPECT_EQ(output_dtypes[0], DT_FLOAT);
  }
}
}  // namespace
