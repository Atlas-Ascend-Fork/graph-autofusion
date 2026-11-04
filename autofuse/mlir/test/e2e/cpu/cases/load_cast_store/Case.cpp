/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> cast -> store CPU case ----------------*- C++ -*-===//
//
// harness 把 soc 钉在 Ascend910B1（cpu/CMakeLists.txt），走 V1 也就是 mem-based 一侧：
// `CastExtend(dst, src, tmp_buf, size)`，扩展头 `cast.h`，要 tmp buffer。reg 侧那份
// 同名而签名完全不同（三个数组、不要 tmp），在这里到不了，由
// afir-cast-reg-base-extension-api.mlir 钉。
//
// 取 f32 -> i32：两侧的对表里都有这一行，而且它落在包装的 `AscendC::Cast` 直通支上
// （`GetRoundMode<float, int32_t>()` 给 `CAST_TRUNC`），不走那两条要中间缓冲的分支。
// 挑整数目的地而不是 f16，是因为 GM 上比整数不用管 half 在主机侧怎么表示；
// `CAST_TRUNC` 又正好让「截断」与「四舍五入」在 ±0.5 上分得开。
//
// **这是本切片里唯一输入与结果必须不同 dtype 的端到端用例。** 谓词族（isnan 等）也
// 输出另一个 dtype，但那是恒定的 ui8；cast 的输出 dtype 是调用方给的，一路上从
// AscGraph 的 `cast_op.y.dtype` 到 destination 的 memref、到 `CastExtend<float>` 的
// `OutT` 推导，每一段都得把它带对。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_cast_store/LoadCastStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_cast_store::GetShapeConfig(load_cast_store::kOutputShapes, 0).shape;
  const auto input_shape = ToExpressions(load_cast_store::GetShapeConfig(load_cast_store::kInputShapes, 0).shape);
  AscGraphBuilder builder(load_cast_store::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0", input_shape, {});

  auto graph = builder.Cast("cast", "load0", af::DT_INT32)
                   .Store("store", "cast")
                   .Output("output", "store", 0, af::DT_INT32)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_cast_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 包装，不是标准调用：没有 `AscendC::` 限定，也没有 `RoundMode` 实参。模板实参取的是
  // **源**的 float，`OutT` 由 dst 推出来——按目的地取会实例化出一个参数对不上的重载。
  EXPECT_NE(kernel.find("CastExtend<float>("), std::string::npos);
  EXPECT_EQ(kernel.find("CastExtend<int32_t>("), std::string::npos);
  // **「没走标准调用」这件事在这里钉不了，一个写法都没有。** `cast.h` 的正文里
  // `AscendC::Cast(...)` 与 `AscendC::Cast<OutT, InT, false>(...)` 两种拼写都有
  // （包装最后转发的就是它），而整段头文本是被内联进 kernel 的——两条负例断言都当场
  // 红，试过了。这件事由 lit 那两份钉（IR 里建的是哪个 op、选中的是哪个 id），
  // 这里只钉正面证据。
  // 内联进来的是 mem 侧那份头，不是 reg 侧那份。两份的 include guard 不撞名。
  EXPECT_NE(kernel.find("__ASCENDC_API_CAST_H__"), std::string::npos);
  // 前导头：`cast.h` 用的 `MAX_REPEAT_TIMES` / `Rational` / `Ceiling` /
  // `ConvertToUint32` 只有它声明。
  EXPECT_NE(kernel.find("__ASCENDC_API_UTILS_H__"), std::string::npos);
  // 包装要 tmp buffer。
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
}
}  // namespace mlir::afir
