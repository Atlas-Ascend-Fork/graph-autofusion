// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `#afir.target` 的 negative 用例。
//
// 资源上限的这几条此前写在消费端（`artifact/TilingSource/TilingProgramSpec.cpp`）：
// 两个字段各判一段，而且只有走到生成 host tiling 那一步才会发现。现在值域在 Attr 的
// verifier，容器与类型在 AFIRDialect::verifyOperationAttribute——与 `afplan` 同一套分工。

// RUN: af-opt --split-input-file --verify-diagnostics %s

// soc 为空。它今天只写不读（dump 的目标标识），正因为没有读点，空串不会在任何下游被
// 发现——只写字段的唯一门禁就是产出时的这一道。
module attributes {afir.target = #afir.target<soc = "", aiv_num = 48, ub_size = 196608>} {
  // expected-error@-1 {{target soc must not be empty}}
}

// -----

// 核数为 0：分不出任何一个核，`ResLimit` 里的兜底上限也就不成立。
module attributes {afir.target = #afir.target<soc = "Ascend910B1", aiv_num = 0, ub_size = 196608>} {
  // expected-error@-1 {{target aiv_num expected a value in [1, 4294967295]}}
}

// -----

// 超出 uint32：这两个数最终落进生成代码里 `ResLimit` 的 uint32_t 字段。
module attributes {afir.target = #afir.target<soc = "Ascend910B1", aiv_num = 4294967296, ub_size = 196608>} {
  // expected-error@-1 {{target aiv_num expected a value in [1, 4294967295]}}
}

// -----

// UB 容量不超过 256：发射端写的是 `set_ub_size(limit->ub_size - 256U)`，256 字节留给
// tiling 自身，容量不超过它就没有可分配的空间。
module attributes {afir.target = #afir.target<soc = "Ascend910B1", aiv_num = 48, ub_size = 256>} {
  // expected-error@-1 {{target ub_size expected a value in [257, 4294967295]}}
}

// -----

// 挂错容器：消费者只在 plan module 上找它，挂在别处等于没挂。三类容器由三道各自的
// 闸门拦——func.func 是 `isKnownFunctionAttribute` 白名单，AFIR 自己的算子是
// `AFIRRejectUnknownAttributes` trait，其余落到 verifyOperationAttribute 的容器检查。
module {
  func.func @wrong_container() {
    // expected-error@+1 {{'afir.target' is only valid on builtin.module}}
    return {afir.target = #afir.target<soc = "Ascend910B1", aiv_num = 48, ub_size = 196608>}
  }
}

// -----

module {
  // expected-error@+1 {{unknown AFIR function attribute '"afir.target"'}}
  func.func private @not_a_module()
      attributes {afir.target = #afir.target<soc = "Ascend910B1", aiv_num = 48, ub_size = 196608>}
}

// -----

// 名字对但类型不对：还原成整数属性即退回迁移前的无类型编码。
// expected-error@+1 {{'afir.target' must be a #afir.target}}
module attributes {afir.target = 48 : i64} {
}
