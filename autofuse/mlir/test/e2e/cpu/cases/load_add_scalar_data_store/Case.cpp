/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> add(ScalarData) -> store CPU case -----*- C++ -*-===//
//
// **`ScalarData` 的第一个用例**，也是 kernel 形参里第一个按值传的标量。与同目录的
// `load_add_scalar_store` 只差一处：那边是编译期常量 `Scalar`（烧成
// `constexpr float c2_5_f32`），这边是运行期输入 `ScalarData`（排成一个 kernel 形参）。
//
// 它钉住的是**参数顺序**：`ScalarData` 与 `Data` 共用一个 index 序号空间，标量要按 index
// 交错排在指针形参中间，不是排到末尾。`KernelArgumentKind::ScalarInput` 的枚举值是 4
// （追加在末尾，ABI 兼容要求），所以一旦 `getKernelArgumentPermutation` 不把它折成
// `Input` 的秩，它会排到 output / workspace / tiling **之后**——下面那条参数类型序列
// 当场不同。
//
//===----------------------------------------------------------------------===//

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_add_scalar_data_store/LoadAddScalarDataStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace cfg = load_add_scalar_data_store;

  const auto output_shape = cfg::GetShapeConfig(cfg::kOutputShapes, 0).shape;
  const auto input_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 0).shape);
  AscGraphBuilder builder(cfg::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0", input_shape, {});

  // 与 load_add_scalar_store 唯一的差别：`Scalar("two", "2.5")` 换成 `ScalarData("scale", 1)`。
  // index 1 与 Data 的 index 0 是同一个序号空间。
  auto graph = builder.ScalarData("scale", 1, af::DT_FLOAT)
                   .Add("add", "load0", "scale")
                   .Store("store", "add")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

namespace {
// 取 `extern "C"  __global__` 那个 kernel 入口的形参**类型**序列，丢掉形参名。
//
// 丢名字是必须的：生成出来的是 SSA 编号（`v1` / `v2` …），随调度顺序漂移，钉不住。
// 而这条用例要钉的恰好只是类型序列——「第 2 个形参是按值的 float，不是指针」。
std::vector<std::string> KernelEntryParameterTypes(const std::string &kernel, const std::string &kernelName) {
  const std::string marker = "__global__ __aicore__ void " + kernelName + "(";
  const size_t start = kernel.find(marker);
  if (start == std::string::npos) {
    return {};
  }
  const size_t open = start + marker.size();
  const size_t close = kernel.find(')', open);
  if (close == std::string::npos) {
    return {};
  }
  std::vector<std::string> types;
  const std::string parameters = kernel.substr(open, close - open);
  for (size_t begin = 0; begin <= parameters.size();) {
    const size_t comma = parameters.find(", ", begin);
    const std::string parameter =
        parameters.substr(begin, comma == std::string::npos ? std::string::npos : comma - begin);
    const size_t space = parameter.rfind(' ');
    types.push_back(space == std::string::npos ? parameter : parameter.substr(0, space));
    if (comma == std::string::npos) {
      break;
    }
    begin = comma + 2U;
  }
  return types;
}
}  // namespace

void CheckGeneratedKernel(const std::string &kernel) {
  namespace cfg = load_add_scalar_data_store;
  EXPECT_NE(kernel.find(std::string(cfg::kKernelName) + "_subgraph0_plan0"), std::string::npos);

  // **这条是本用例的主判据。** index 0 的 `Data` 与 index 1 的 `ScalarData` 共用一个序号
  // 空间，所以标量排在两个指针形参中间；排错了这里就不是这个序列。
  const std::vector<std::string> expected{"__gm__ float*", "float", "__gm__ float*", "__gm__ uint8_t*",
                                          "AutofuseTilingData"};
  EXPECT_EQ(KernelEntryParameterTypes(kernel, cfg::kKernelName), expected);

  // 走的是 tensor_scalar 形态（`Adds`），与 `Scalar` 常量那条发的是同一个调用——标量来自
  // 块参数还是 `arith.constant`，conversion 层并不区分。模板实参里的 `1` 是 `isSetMask`。
  EXPECT_NE(kernel.find("AscendC::Adds<float, 1>("), std::string::npos);
  EXPECT_EQ(kernel.find("AscendC::Add<"), std::string::npos);
  // 标量不该有 `constexpr` 常量定义——那是 `Scalar` 的形态，不是这条路。
  EXPECT_EQ(kernel.find("constexpr float"), std::string::npos);

  // host 侧的 launch 包装器。**两处都要分流**：签名里标量是 `float input1` 而不是
  // `void* input1`，调用处传的是 `input1` 而不是 `(float)input1`。写法抄自 legacy 的
  // `GetScheduledResultInputOutput`——`AutofuseLaunchSignatureMatchesLegacy` 逐字比的
  // 就是下面这一行。
  //
  // `CheckGeneratedKernel` 拿到的是完整的 `result.kernel`；写进 kernel.cpp 的那一份被
  // `WriteSources` 在这个标记处截断了（E2E 直接调 kernel，不过 host 包装器），所以这条
  // 只能在这里钉，落到文件上就看不见了。
  EXPECT_NE(kernel.find("extern \"C\" int64_t AutofuseLaunch(uint32_t blockDim, void* stream, void* input0, "
                        "float input1, void* output0, void* workspace, AutofuseTilingData* tiling_data)"),
            std::string::npos);
  EXPECT_NE(kernel.find("input1, (__gm__ float*)output0,"), std::string::npos);
  EXPECT_EQ(kernel.find("(float)input1"), std::string::npos);
}
}  // namespace mlir::afir
