// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `#afir.axis` 可缺省尾部的 negative 用例。
//
// custom 解析器必须拒绝未知关键字、字段乱序和重复字段，不能将其当成缺省项。
// 缺省本身合法，忽略这些错误会使实际轴属性偏离输入意图。

// RUN: af-opt --split-input-file --verify-diagnostics %s

// 拒绝拼错的 alignment 关键字，不能将该字段视为未指定。
#bad_keyword = #afir.asc_graph<
  axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">,
  // expected-error @below {{expected 'align', 'from' or 'split_pair_other_id'}}
  // expected-error @below {{failed to parse AscGraphAttrGroupsAttr parameter 'axes'}}
           alignment = #afir.expr<"32">>],
  type = Compute>

// -----

// 可选尾字段有固定顺序，乱序输入必须拒绝，保持序列化形式确定。
#wrong_order = #afir.asc_graph<
  axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">, from = [0],
  // expected-error @below {{expected 'align', 'from' or 'split_pair_other_id'}}
  // expected-error @below {{failed to parse AscGraphAttrGroupsAttr parameter 'axes'}}
           align = #afir.expr<"32">>],
  type = Compute>

// -----

// 拒绝重复字段，不能由后一项静默覆盖前一项。
#duplicate = #afir.asc_graph<
  axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">, from = [0],
  // expected-error @below {{expected 'align', 'from' or 'split_pair_other_id'}}
  // expected-error @below {{failed to parse AscGraphAttrGroupsAttr parameter 'axes'}}
           from = [1]>],
  type = Compute>

// -----

// align 的值不是 #afir.expr。两条诊断都要有：内层解析器指出语法错在哪个 token，
// 外层说明错的是哪个字段——只有后者能告诉写 IR 的人「align 要的是表达式属性」。
//
// 这里的 expected-error 必须写在属性**外面**并用行偏移：`#afir.asc_graph<...>` 是
// pretty dialect 形式，MLIR 按 `<` / `>` 配对扫它的文本范围，注释里一个落单的 `<`
// 会让配对失败。前三例的诊断文本不含 `<`，所以可以写在属性里。
// expected-error @+4 {{expected '<'}}
// expected-error @+3 {{expected an #afir.expr alignment expression}}
// expected-error @+2 {{failed to parse AscGraphAttrGroupsAttr parameter 'axes'}}
#bad_align = #afir.asc_graph<
  axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">, align = 32>],
  type = Compute>
