/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_TEST_E2E_CPU_SUPPORT_E2E_CASE_H_
#define AUTOFUSE_MLIR_TEST_E2E_CPU_SUPPORT_E2E_CASE_H_

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

#include "test/e2e/cpu/support/E2eCaseConfig.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"

namespace mlir::afir {
inline std::vector<af::Expression> ToExpressions(std::initializer_list<int64_t> shape) {
  std::vector<af::Expression> expressions;
  expressions.reserve(shape.size());
  for (auto dimension : shape) {
    expressions.push_back(af::Symbol(dimension));
  }
  return expressions;
}

// 生成的 kernel 里 `call` 那一处调用有几个顶层实参。
//
// 用它而不是按名字断言：kernel 里的实参是 SSA 编号（`v93` 之类），随调度顺序漂移；
// 而「几个实参」恰好就是同名重载之间唯一分得开的那件事——`AscendC::Sub` 的四参形态与
// 带 tmp buffer 的五参形态就是这么区分的。
//
// 只看第一处出现。 所以别拿它去数走包装的调用：包装的头会被内联进 kernel，
// 第一处出现的是那份定义（形参个数与调用点可以不同），不是调用点。标准
// `AscendC::` 调用没有这个问题——那些头不内联，kernel 里只有调用点。
//
// 找不到 `call` 返回 0，调用方那条 `EXPECT_EQ` 会自己报出来。
inline size_t CountCallArguments(const std::string &kernel, const std::string &call) {
  const size_t start = kernel.find(call);
  if (start == std::string::npos) {
    return 0U;
  }
  size_t depth = 0U;
  size_t arguments = 1U;
  for (size_t index = start + call.size() - 1U; index < kernel.size(); ++index) {
    const char character = kernel[index];
    if (character == '(') {
      ++depth;
    } else if (character == ')') {
      --depth;
      if (depth == 0U) {
        break;
      }
    } else if (character == ',' && depth == 1U) {
      ++arguments;
    }
  }
  return arguments;
}

// 广播轴与 Load 的 stride。广播轴由「输入该维是 1、输出不是」定出来；广播轴上 Load 的
// stride 取 0，其余维按输入自己的行主序算——与 legacy 的 nddma 用例（`optimize_st_v2.cpp`
// 的 `NddmaCase*`）同一写法，也与 `test/probe/CapabilityBoundaryProbe.cpp` 的 `BindBroadcast`
// 逐字同一条约定（测量工具与用例建的必须是同一张图，否则探针报 ok 的格在这里不成立）。
//
// 数值侧的另一面是 `E2eCaseConfig.h` 的 `BroadcastSourceIndex`。
struct BroadcastBinding {
  std::vector<af::Expression> inputShape;
  std::vector<af::Expression> outputShape;
  std::vector<af::Expression> loadStrides;
  // 广播轴，内 → 外。`Broadcast()` 只按下标改 repeats，与次序无关；两步广播要按这个次序
  // 切两半，所以记下来方向。
  std::vector<int64_t> brcAxes;
};

inline BroadcastBinding BindBroadcast(std::initializer_list<int64_t> out_shape,
                                      std::initializer_list<int64_t> in_shape) {
  const std::vector<int64_t> out(out_shape.begin(), out_shape.end());
  const std::vector<int64_t> in(in_shape.begin(), in_shape.end());
  BroadcastBinding binding;
  binding.inputShape = ToExpressions(in_shape);
  binding.outputShape = ToExpressions(out_shape);
  binding.loadStrides.assign(in.size(), af::sym::kSymbolZero);
  int64_t stride = 1;
  for (size_t position = in.size(); position > 0U; --position) {
    const size_t axis = position - 1U;
    if (in[axis] == 1 && out[axis] != 1) {
      binding.brcAxes.push_back(static_cast<int64_t>(axis));
      continue;
    }
    binding.loadStrides[axis] = af::Symbol(stride);
    stride *= in[axis];
  }
  return binding;
}

// `needle` 在 `haystack` 里出现几次。
//
// 数扩展头要数 `#define __ASCENDC_API_XXX_H__` 那一行而不是找子串：`#ifndef` / `#define`
// 一次内联就出现两次，找子串数不出「有没有被内联两遍」。
inline size_t CountOccurrences(const std::string &haystack, const std::string &needle) {
  size_t count = 0;
  for (size_t at = haystack.find(needle); at != std::string::npos; at = haystack.find(needle, at + needle.size())) {
    ++count;
  }
  return count;
}

// 动态形状用例的运行期尺寸，按图上登记 size var 的顺序排。
//
// 静态形状的用例（今天绝大多数）留空即可：它们的 `AutofuseTiling` 没有这些形参。图上有
// 运行期 size var 时，生成的 `AutofuseTiling` 会在 `tiling` 之前多出同样多个 `uint32_t`
// 形参，少传一个就是签名对不上、编译报错，不会悄悄算错。
//
// 用例在 `BuildGraph()` 里填它——那是唯一既知道图、又跑在 codegen 之前的地方。
extern std::vector<uint32_t> g_e2eRuntimeShapeArgs;

// Each generator links exactly one case, which supplies these two functions.
af::AscGraph BuildGraph();
void CheckGeneratedKernel(const std::string &kernel);

// Writes the device kernel, generated tiling data, and ATT host tiling source.
void WriteSources(const std::string &kernel, const std::string &tilingData, const std::string &tiling);
}  // namespace mlir::afir

#endif  // AUTOFUSE_MLIR_TEST_E2E_CPU_SUPPORT_E2E_CASE_H_
