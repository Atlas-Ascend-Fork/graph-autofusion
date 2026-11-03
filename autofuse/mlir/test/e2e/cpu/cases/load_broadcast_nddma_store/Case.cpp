/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_broadcast_nddma_store/Case.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"
namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_broadcast_nddma_store;
  const auto output_shape = config::GetShapeConfig(config::kOutputShapes, 0).shape;
  const auto input_config = config::GetShapeConfig(config::kInputShapes, 0);
  const auto input_shape = ToExpressions(input_config.shape);
  // 广播轴由「输入该维是 1、输出不是」定出来。Load 在广播轴上 stride 取 0，其余维按输入自己的
  // 行主序算——与 legacy 的 nddma 用例（`optimize_st_v2.cpp` 的 `NddmaCase*`）同一写法。
  std::vector<int64_t> in_dims(input_config.shape.begin(), input_config.shape.end());
  std::vector<int64_t> out_dims(output_shape.begin(), output_shape.end());
  std::vector<int64_t> brc_axes;
  std::vector<af::Expression> load_strides(in_dims.size(), af::sym::kSymbolZero);
  int64_t stride = 1;
  for (size_t r = in_dims.size(); r > 0U; --r) {
    const size_t axis = r - 1U;
    if (in_dims[axis] == 1 && out_dims[axis] != 1) {
      brc_axes.push_back(static_cast<int64_t>(axis));
      continue;
    }
    load_strides[axis] = af::Symbol(stride);
    stride *= in_dims[axis];
  }
  AscGraphBuilder builder(config::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input_shape, {}, af::DT_FLOAT)
      .Load("load0", "input0", input_shape, load_strides);
  auto graph = builder.Broadcast("brc", "load0", brc_axes)
                   .Store("store", "brc")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  return graph;
}
void CheckGeneratedKernel(const std::string &kernel) {
  namespace config = load_broadcast_nddma_store;
  // 走的是 MLIR 那条路而不是回落 legacy：`GenerateForInductor` 在 MLIR 报不支持时会接着跑
  // legacy 并**返回成功**，所以这一条不是形式检查——少了它，整个用例在回落之后照样全绿。
  EXPECT_NE(kernel.find(std::string(config::kKernelName) + "_subgraph0_plan0"), std::string::npos);

  // 这张图在 reg 侧被调度成**四个模板**，计划选择代不出常数，四个都留给运行期选：
  // `template0` 走 `BroadcastExtend`、`template1` 走 `Duplicate`（两者都是 Load 与 Broadcast
  // 没有被融掉的那几种排布），`template2` / `template3` 才是 `NddmaTemplate` 融出来的
  // `Nddma` 节点。
  //
  // **所以这里不能断言「没有 BroadcastExtend」**：那不是回落的证据，是同一张图的另外几个
  // 候选。要断言的是 ND-DMA 那条路确实发出来了。
  EXPECT_NE(kernel.find("DataCopyNddma<"), std::string::npos);

  // 扩展头是**按注册表的键查、把头文本整份内联进 kernel**的，不是留一条 `#include`。
  // 所以两条一起查：包装的定义在（键查到了），以及没有留下解析不了的 include（键写成源
  // 文件名 `datacopy_nddma.h` 就会这样，设备编译器报找不到头文件 —— ODS 那段注释里有说明）。
  EXPECT_NE(kernel.find("void DataCopyNddma(const AscendC::LocalTensor<T> &dst"), std::string::npos);
  EXPECT_EQ(kernel.find("#include \"datacopy_nddma"), std::string::npos);

  // 三组尺寸各自一个栈上数组，名字带 `<src>_nddma_to_<dst>` 后缀，调用在三者之后。
  // **结构由 `test/lit/target/ascendc-data-copy-nddma-reg-base.mlir` 与
  // `test/lit/conversion/afir-nddma-reg-base.mlir` 钉**；这里钉的是这条真实路径上确实
  // 发出了它们，数值则由下面的 KernelTest 逐元素比出来。
  const size_t dims = kernel.find("const int64_t output_dims_");
  ASSERT_NE(dims, std::string::npos);
  const size_t outputStride = kernel.find("const int64_t output_stride_", dims);
  ASSERT_NE(outputStride, std::string::npos);
  const size_t inputStride = kernel.find("const int64_t input_stride_", outputStride);
  ASSERT_NE(inputStride, std::string::npos);
  EXPECT_LT(inputStride, kernel.find("DataCopyNddma<", inputStride));

  // **维数不在这里钉。** 同一张图的几个模板维数本来就不同——`[8,1] -> [8,128]` 实测
  // `template2` 是 `DataCopyNddma<float, 2>`（片上连续、GM 不连续，分维那一刀切出来的），
  // 而 `template3` 是 `DataCopyNddma<float, 1>`（广播轴排到 tile 外，整块 tile 重复读 GM
  // 同一个元素，一维是对的）。断言「不许有一维」会把后者判成错。
  //
  // 维数由 `test/lit/conversion/afir-nddma-reg-base.mlir` 在固定 descriptor 上钉；少切那一刀
  // 的后果（把 GM 上连续的 `行数 × 128` 个元素搬进来，而正确结果是同一个元素重复 128 次）
  // 由下面的 KernelTest 逐元素比出来。
}
}  // namespace mlir::afir
