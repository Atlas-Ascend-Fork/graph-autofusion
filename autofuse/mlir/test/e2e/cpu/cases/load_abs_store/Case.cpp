/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> abs -> store CPU end-to-end case ------*- C++ -*-===//
//
// The end-to-end half of the elementwise slice's vertical probe. Unlike
// load_add_store, this graph exercises all three of the slice's foundations at
// once on a real schedule: the scheduler allocates a tmp buffer for the
// node, the lowering picks the Autofuse extension API rather than a stock
// AscendC call, and the kernel-source layer has to bring the extension header
// in for the kernel to compile at all.
//
// int32 rather than float on purpose: the stock `AscendC::Abs` has no integer
// overload at all, so int32 is the case the extension wrapper exists for. On
// float the wrapper just forwards to the stock call, which would leave the
// interesting half of the D2 difference untested.
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_abs_store/LoadAbsStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_abs_store::GetShapeConfig(load_abs_store::kOutputShapes, 0).shape;
  const auto input_shape = ToExpressions(load_abs_store::GetShapeConfig(load_abs_store::kInputShapes, 0).shape);
  AscGraphBuilder builder(load_abs_store::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_INT32).Load("load0", "input0", input_shape, {});

  auto graph = builder.Abs("abs", "load0").Store("store", "abs").Output("output", "store", 0, af::DT_INT32).Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_abs_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // The extension wrapper, not the stock `AscendC::Abs`: this target does not
  // ship the stock vector math API. Only the call site can say so -- the
  // wrapper's own body calls `AscendC::Abs` for the float case, and that body
  // is inlined into the kernel, so searching for the stock name proves nothing.
  EXPECT_NE(kernel.find("AbsExtend<int32_t>("), std::string::npos);
  // The header text is inlined rather than left as an `#include`, because the
  // kernel is compiled as a single translation unit with no include path.
  EXPECT_EQ(kernel.find("#include \"abs.h\""), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_ABS_H__"), std::string::npos);
  // The tmp buffer is a TBuf sized from the canonical resource name, not
  // from anything AFIR computed.
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
}
}  // namespace mlir::afir
