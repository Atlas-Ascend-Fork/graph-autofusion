/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_TEST_E2E_CPU_SUPPORT_E2E_CASE_CONFIG_H_
#define AUTOFUSE_MLIR_TEST_E2E_CPU_SUPPORT_E2E_CASE_CONFIG_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace mlir::afir {
struct ShapeConfig {
  std::initializer_list<int64_t> shape;
  uint32_t element_count;
};

inline const ShapeConfig &GetShapeConfig(std::initializer_list<ShapeConfig> shapes, size_t index) {
  assert(index < shapes.size());
  return shapes.begin()[index];
}

// 输出线性下标 → 广播源的线性下标。广播轴由「输入该维是 1、输出不是」定出来，与
// `E2eCase.h` 的 `BindBroadcast`（建图侧）是同一条约定的两面。
//
// 放在这个头里而不是 `E2eCase.h`：`KernelTest` 跑在设备侧那个二进制里，只包得到
// `Case.h` → 本文件 + tikicpulib，包不了带 AscGraph 的那一份。
inline uint32_t BroadcastSourceIndex(std::initializer_list<int64_t> out_shape, std::initializer_list<int64_t> in_shape,
                                     uint32_t output_index) {
  assert(out_shape.size() == in_shape.size());
  const int64_t *out_dims = out_shape.begin();
  const int64_t *in_dims = in_shape.begin();
  uint32_t remaining = output_index;
  uint32_t source_index = 0;
  uint32_t source_stride = 1;
  for (size_t reverse = out_shape.size(); reverse > 0U; --reverse) {
    const size_t axis = reverse - 1U;
    const uint32_t coordinate = remaining % static_cast<uint32_t>(out_dims[axis]);
    remaining /= static_cast<uint32_t>(out_dims[axis]);
    if (in_dims[axis] != 1) {
      source_index += coordinate * source_stride;
    }
    source_stride *= static_cast<uint32_t>(in_dims[axis]);
  }
  return source_index;
}
}  // namespace mlir::afir

//===----------------------------------------------------------------------===//
// 用例 `Case.h` 的样板
//===----------------------------------------------------------------------===//
//
// 一个用例的 `Case.h` 原先 30 行里有 21 行是逐字样板：两个 stringify 宏（**每个用例各抄一
// 份**）、命名空间、三条 using、两条 static_assert、`kKernelName`。这里收成一对宏，用例侧只
// 剩「叫什么名字、几进几出」。
//
// 用法（`Case.h` 里，include guard 与许可证头照旧——宏生成不了 `#ifndef`）：
//
//     #include "CaseConfig.h"
//
//     AF_MLIR_E2E_CPU_CASE_BEGIN(load_abs_store, 1, 1);
//     AF_MLIR_E2E_CPU_CASE_END(load_abs_store)
//
// 用例自己的常量写在两个宏之间，与原先写在命名空间里逐字等价（例如 `load_axpy_half_store`
// 的 `kAlpha`）。
//
// `AF_MLIR_E2E_CPU_KERNEL_SYMBOL` 由 `add_mlir_e2e_cpu_case` 作为编译宏传进来，所以
// `kKernelName` 只能在这里取、不能写成参数。

#define AF_MLIR_E2E_CPU_STRINGIFY_IMPL(value) #value
#define AF_MLIR_E2E_CPU_STRINGIFY(value) AF_MLIR_E2E_CPU_STRINGIFY_IMPL(value)

// 两条 static_assert 的消息里带上用例名与期望个数：形状来自 CMake 那一侧，数量写错时报出来
// 的话要能一眼看出是哪个用例、期望几个。
#define AF_MLIR_E2E_CPU_CASE_BEGIN(case_name, input_count, output_count)                                               \
  namespace mlir::afir::case_name {                                                                                    \
  using ::mlir::afir::GetShapeConfig;                                                                                  \
  using ::mlir::afir::test_config::kInputShapes;                                                                       \
  using ::mlir::afir::test_config::kOutputShapes;                                                                      \
  static_assert(kInputShapes.size() == (input_count), #case_name " requires exactly " #input_count " input shape(s)"); \
  static_assert(kOutputShapes.size() == (output_count),                                                                \
                #case_name " requires exactly " #output_count " output shape(s)");                                     \
  constexpr const char *kKernelName = AF_MLIR_E2E_CPU_STRINGIFY(AF_MLIR_E2E_CPU_KERNEL_SYMBOL)

#define AF_MLIR_E2E_CPU_CASE_END(case_name) }  // namespace mlir::afir::case_name

#endif  // AUTOFUSE_MLIR_TEST_E2E_CPU_SUPPORT_E2E_CASE_CONFIG_H_
