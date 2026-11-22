// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms of
// the CANN Open Software License Agreement Version 2.0.
// RUN: not af-opt --afir-memory-allocate %s 2>&1 | FileCheck %s

// 输入已有 axis_binding，表示循环已生成，但函数签名不满足 canonical AscendC ABI。
// 阶段判据来自轴绑定结构，见 LoopGenerate.h，不依赖已删除的 loop_generated 属性。
module {
  func.func private @malformed_abi() attributes {
    afir.arg_descs = [],
    afir.asc_graph = #afir.asc_graph<
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"16">>],
      type = Compute>,
    afir.function_role = #afir.function_role<kernel_body>
  } {
    %axis = arith.constant {afir.axis_binding = #afir.axis_binding<axis = 0, role = axis_size>} 16 : index
    return
  }
}

// CHECK: canonical AscendC kernel ABI requires workspace and TilingData arguments
