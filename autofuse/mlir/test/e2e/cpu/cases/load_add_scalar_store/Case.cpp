/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> add(scalar) -> store CPU case ---------*- C++ -*-===//
//
// **标量族的第一个编译型用例。L-1 说这张图跑不通，实测它一次就过。**
//
// L-1 记的现场是「`codegen_tiling.cpp:872` 的 `codegen_func_`（ATT 的 `CodegenTiling`）
// 返回 false，失败发生在 MLIR bridge 之前」。两处都不对：
//
// 一、这个 harness 里 `codegen_func_` **不是 ATT**，是
// `test/e2e/cpu/support/TilingStub.cpp` 的桩，无条件 `return false`。所以任何回落到
// legacy tiling 的图都会报那一行——那是 harness 的属性，与 `Scalar` 无关。
//
// 二、时序反了。`GenerateForInductor` 第一件事就是 `RunMlirCodegen`；只有它返回
// `kNotSupported` 才会走到 tiling。L-1 现场给的复现命令是手工跑 generator，既没有
// `--mlir_codegen_config=mlir`、也没有 cmake 烘进去的 `LD_LIBRARY_PATH`，两者缺任一个 MLIR 路径都
// 谢绝，于是必然落到那个桩上。**所以必须用 `check-e2e-cpu-<用例名>` 目标跑**（用例名带
// 形状后缀，例如 `check-e2e-cpu-load_add_scalar_store_202x40`）。
//
// 这个用例同时让此前无人走到的两支有了用例：`materializeOperandTiles` 的
// `Scheduled` 标量分支（埋点验过——改坏解析出来的标量值，逐元素断言当场红）
// 与 L-2 的 `isSetMask`——下面那条 `Adds<float, 1>` 里的 `1` 就是它。
//
// **L-2 曾经写着「要机器」，不成立**：那条在 CPU 模拟器上就能复现，只是当初所有用例的
// 形状都是 `[128]`，探不到。`isSetMask` 取 `false` 时 API 不设 mask、按 `repeatTimes = 1`
// 直发一条指令（`dav_c220` 的 Level 2 分支），一趟的 count 不超过一个 repeat（f32 是 64
// 个）时与取 `true` 结果相同——`[128]` 切完正好落在这个范围里，所以它一直是绿的。最初
// 变红的是同目录下的 `load_add_scalar_store_128x23x5`：14720 个元素里错 14528 个，
// 第一个错的就是下标 64。
//
// **2026-09-23 起整套用例都不是 `[128]` 了**（每条 2~4 轴、每轴 [1,512]），这类「只在一趟
// 算多于一个 repeat 时才错」的判据不再需要专门挑一条用例去覆盖。
//
// `getBufferizedElementwiseEffects` 里「标量不挂 `Read` effect」那一支**仍然没有用例
// 钉住**：把那条 `contributesShape` 判据删掉（标量也挂上 Read），这个用例、全部 156 条
// lit、AFIR 方言 UT 全都照样绿。
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_add_scalar_store/Case.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace cfg = load_add_scalar_store;

  const auto output_shape = cfg::GetShapeConfig(cfg::kOutputShapes, 0).shape;
  const auto input_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 0).shape);
  AscGraphBuilder builder(cfg::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0", input_shape, {});

  auto graph = builder.Scalar("two", "2.5", af::DT_FLOAT)
                   .Add("add", "load0", "two")
                   .Store("store", "add")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_add_scalar_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // **字面量写全**：emitter 打的是定长科学计数，写成 `(float)2.5` 会匹配上前缀，
  // 值改成 2.5e+01 也照样绿（纪律：新写钉常量的 check 前先 dump 一眼实际输出）。
  EXPECT_NE(kernel.find("constexpr float c2_5_f32 = (float)2.500000000e+00;"), std::string::npos);
  // 走的是 tensor_scalar 形态（`Adds`），不是 tensor_tensor（`Add`）。模板实参里那个
  // `1` 是 `isSetMask`，见文件头与 `Base.td` 的 `VecScalarL2Op`。**这条红成 `<float, 0>`
  // 就是那个缺省被改回了 `UnitAttr`，不要就地把 `1` 改回 `0`。**
  EXPECT_NE(kernel.find("AscendC::Adds<float, 1>("), std::string::npos);
  EXPECT_EQ(kernel.find("AscendC::Add<"), std::string::npos);
  // 标量是第三个实参、后面还有 count——不是最后一个。写反了 IR 与 C++ 都成立。
  EXPECT_NE(kernel.find("c2_5_f32, "), std::string::npos);
  // 标量不占 UB：搬入 / 搬出各一条队列，没有第三块。
  EXPECT_EQ(kernel.find("AscendC::TBuf<"), std::string::npos);
}
}  // namespace mlir::afir
