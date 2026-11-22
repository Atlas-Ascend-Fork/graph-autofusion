// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `afir.arg_descs` 的覆盖范围门禁。
//
// 删掉 `afir.num_inputs` 之后，「这个数组该有多长」必须有一个独立于数组自身长度的
// 来源，否则消费端拿期望值去和同一个数组比长度就退化成恒真——少写描述符再也没有任何
// 东西能发现。第一版实现正是这样：物化后直接返回 `descriptors.size()`，于是给任意参数
// 挂一个 `#afir.kernel_arg<role = workspace>` 就能关掉整条门禁。
//
// 现在期望值 = 参数总数 − ABI 角色（workspace / tiling）参数个数。下面三条把这件事的
// 两侧都钉住：有 ABI 角色时仍然报、ABI 角色只抵扣自己那一个、以及物化前的老行为不变。

// RUN: af-opt --split-input-file --verify-diagnostics %s

// 3 个数据参数 + 1 个 workspace，`arg_descs` 只写 1 项。门禁必须报——这正是回归点。
module {
  // expected-error @below {{DescriptorCountMismatch: 'afir.arg_descs' expected 3 element but found 1}}
  func.func private @under_described(
      %a: memref<16xf16> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>},
      %b: memref<16xf16> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 1>},
      %c: memref<16xf16> {afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>},
      %ws: memref<?xi8> {afir.kernel_arg = #afir.kernel_arg<role = workspace>}) attributes {
    afir.arg_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
    afir.function_role = #afir.function_role<kernel_body>
  }
}

// -----

// 反例侧：同样带 ABI 角色，但描述符数目正确。ABI 角色只抵扣它自己那一个参数，
// 不是「一挂上就不查了」。
module {
  func.func private @fully_described(
      %a: memref<16xf16> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>},
      %b: memref<16xf16> {afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>},
      %ws: memref<?xi8> {afir.kernel_arg = #afir.kernel_arg<role = workspace>},
      %t: i64 {afir.kernel_arg = #afir.kernel_arg<role = tiling>}) attributes {
    afir.arg_descs = [#afir.tensor_desc<axis = [0], role = <input>>,
                      #afir.tensor_desc<axis = [0], role = <output>>],
    afir.function_role = #afir.function_role<kernel_body>
  }
}

// -----

// 物化前（没有任何 ABI 角色参数）：期望值是参数总数，与删除 `afir.num_inputs` 之前一致。
module {
  // expected-error @below {{DescriptorCountMismatch: 'afir.arg_descs' expected 2 element but found 1}}
  func.func private @pre_abi(%a: memref<16xf16>, %b: memref<16xf16>) attributes {
    afir.arg_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
    afir.function_role = #afir.function_role<kernel_body>
  }
}
