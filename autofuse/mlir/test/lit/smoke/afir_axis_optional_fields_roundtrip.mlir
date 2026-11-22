// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `#afir.axis` 三个可缺省字段的 round-trip：align / from / split_pair_other_id 的
// 每一种在场组合都必须打印得出、读得回。
//
// 起因是一个真实缺口：三项在文本里都以同一个 `,` 开头，而 ODS 的可选组按组内首个
// 字面量判断该组是否出现，于是解析器一见到逗号就锁定 `align`。带 `from`、不带
// `align` 的轴——正是切分产出的轴的形态——打印出来再解析会报 "expected 'align'"。
// 打印方向一直是对的，所以症状只在「把 IR 打印出来再喂回去」时出现；bridge 产出的
// 轴始终带 `align`，因此在此之前没有任何用例走到过这条路径。
//
// 两条 RUN 缺一不可：第一条只验打印，第二条才是 round-trip。上面那个缺口在第一条
// 下完全看不出来。

// RUN: af-opt %s | FileCheck %s
// RUN: af-opt %s | af-opt | FileCheck %s

module {
  // 三项全缺：最朴素的原始轴。
  // CHECK-LABEL: func.func private @none
  // CHECK-SAME: afir.asc_graph = #afir.asc_graph<axes = [<id = 0, name = "z", axis_type = Original, size = <"s0">>], type = Compute>
  func.func private @none() attributes {
    afir.asc_graph = #afir.asc_graph<
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>],
      type = Compute>
  }

  // 只有 align。
  // CHECK-LABEL: func.func private @align_only
  // CHECK-SAME: axes = [<id = 0, name = "z", axis_type = Original, size = <"s0">, align = <"32">>]
  func.func private @align_only() attributes {
    afir.asc_graph = #afir.asc_graph<
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">, align = #afir.expr<"32">>],
      type = Compute>
  }

  // 只有 from——这就是此前打印得出、读不回的那一种。切分产出的轴就是这个形态：
  // `AscGraphImpl::DoSplit` 只写 from 与 split_pair_other_id，不写 align。
  // CHECK-LABEL: func.func private @from_only
  // CHECK-SAME: <id = 1, name = "zT", axis_type = TileOuter, size = <"s1">, from = [0]>
  func.func private @from_only() attributes {
    afir.asc_graph = #afir.asc_graph<
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>,
              <id = 1, name = "zT", axis_type = TileOuter, size = #afir.expr<"s1">, from = [0]>],
      type = Compute>
  }

  // 只有 split_pair_other_id：跳过前两项，尾部识别不能依赖「前一项出现过」。
  // CHECK-LABEL: func.func private @split_pair_only
  // CHECK-SAME: <id = 0, name = "z", axis_type = Original, size = <"s0">, split_pair_other_id = 1>
  // CHECK-SAME: <id = 1, name = "y", axis_type = Original, size = <"s1">, split_pair_other_id = 0>
  func.func private @split_pair_only() attributes {
    afir.asc_graph = #afir.asc_graph<
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">, split_pair_other_id = 1>,
              <id = 1, name = "y", axis_type = Original, size = #afir.expr<"s1">, split_pair_other_id = 0>],
      type = Compute>
  }

  // from + split_pair_other_id，无 align：一次 TileSplit 产出的完整一对。
  // CHECK-LABEL: func.func private @split_pair
  // CHECK-SAME: <id = 1, name = "zT", axis_type = TileOuter, size = <"Ceiling((s0 / (zt_size)))">, from = [0], split_pair_other_id = 2>
  // CHECK-SAME: <id = 2, name = "zt", axis_type = TileInner, size = <"zt_size">, from = [0], split_pair_other_id = 1>
  func.func private @split_pair() attributes {
    afir.asc_graph = #afir.asc_graph<
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>,
              <id = 1, name = "zT", axis_type = TileOuter, size = #afir.expr<"Ceiling((s0 / (zt_size)))">,
               from = [0], split_pair_other_id = 2>,
              <id = 2, name = "zt", axis_type = TileInner, size = #afir.expr<"zt_size">,
               from = [0], split_pair_other_id = 1>],
      type = Compute>
  }

  // 三项全在，外加 bind_block 与多来源 from（合轴的形态）。
  // CHECK-LABEL: func.func private @all
  // CHECK-SAME: <id = 2, name = "xy", axis_type = Merged, bind_block = true, size = <"(s0 * s1)">, align = <"32">, from = [0, 1], split_pair_other_id = 1>
  func.func private @all() attributes {
    afir.asc_graph = #afir.asc_graph<
      axes = [<id = 0, name = "x", axis_type = Original, size = #afir.expr<"s0">>,
              <id = 1, name = "y", axis_type = Original, size = #afir.expr<"s1">>,
              <id = 2, name = "xy", axis_type = Merged, bind_block = true, size = #afir.expr<"(s0 * s1)">,
               align = #afir.expr<"32">, from = [0, 1], split_pair_other_id = 1>],
      type = Compute>
  }
}
