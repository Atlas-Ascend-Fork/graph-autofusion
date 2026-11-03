// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `#afir.axis` 可缺省尾部的 negative 用例。
//
// 尾部改由 custom 指令逐项识别关键字之后，「哪些写法应当被拒」不再由 ODS 生成的
// 解析器兜底，必须自己钉住。三种形态都会让下游读到一份与作者意图不同的轴：
// 拼错的关键字、颠倒的顺序、重复的项。放过任何一种，错的那一项会被当成「没写」，
// 而「没写」是合法的——所以失败是静默的。

// RUN: af-opt --split-input-file --verify-diagnostics %s

// 关键字写错（这里是写成了全称 alignment）。错的那一项会被当成「没写」，而「没写」是合法的——
// 放过它，失败就是静默的。
#bad_keyword = #afir.asc_graph<
  axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">,
  // expected-error @below {{expected 'align', 'from' or 'split_pair_other_id'}}
  // expected-error @below {{failed to parse AscGraphAttrGroupsAttr parameter 'axes'}}
           alignment = #afir.expr<"32">>],
  type = Compute>

// -----

// 顺序颠倒。尾部是有序的：允许乱序等于同一份轴有多种拼写，文本 diff 会失去意义。
#wrong_order = #afir.asc_graph<
  axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">, from = [0],
  // expected-error @below {{expected 'align', 'from' or 'split_pair_other_id'}}
  // expected-error @below {{failed to parse AscGraphAttrGroupsAttr parameter 'axes'}}
           align = #afir.expr<"32">>],
  type = Compute>

// -----

// 同一项写两遍。后一份会覆盖前一份，作者看到的与生效的不是同一个值。
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
