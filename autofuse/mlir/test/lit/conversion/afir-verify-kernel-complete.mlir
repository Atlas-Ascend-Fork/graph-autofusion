// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 完备性判据 C1–C4 的正负向用例（设计文档 5.2）。
//
// 把完备性从文档表格变成可运行的检查——否则「完备」永远是主观判断，也无法在 PR 里当
// 门禁。每条判据在这里都有一正一负：只有正向，检查器退化成「全放行」也能通过；只有
// 负向，它退化成「全拒绝」也能通过。
//
// 判据逐条可选。默认只跑 C3，因为另外三条在今天的生产 kernel 上还会红（5.2 的「当前」
// 列：C1 部分、C2 未满足、C4 覆盖面窄）——把它们一起设成默认，这个 pass 会立刻被加进
// 忽略名单。逐条可选之后，每条转绿的那一刻是一次显式提交。
//
// `--target=` 是必需的：C4 要查实现注册表，注册表按目标构建。

// RUN: af-opt --split-input-file --verify-diagnostics --target=Ascend910B1 \
// RUN:   --afir-verify-kernel-complete=criteria=c1,c2,c3,c4 %s

// 判据的**选择语义**本身也要钉住，它是 5.2 实施结论的核心主张：默认只跑 C3，显式点名是
// **替换**默认值而不是追加。后者依赖 MLIR `ListOption` 的实现细节——`handleOccurrence`
// 在首次出现时 `clear()` 掉默认值（PassOptions.h）——正因为它依赖上游实现，更该有用例
// 锁住：哪天上游改成追加，下面第二条 RUN 会立刻红。
//
// `--implicit-check-not` 作用于整段输出，因此「没报别的判据」是真的全局断言，
// 不受 split 顺序影响。

// RUN: not af-opt --split-input-file --target=Ascend910B1 --afir-verify-kernel-complete %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DEFAULT \
// RUN:       --implicit-check-not="Incomplete C1" --implicit-check-not="Incomplete C2" \
// RUN:       --implicit-check-not="Incomplete C4"

// RUN: not af-opt --split-input-file --target=Ascend910B1 --afir-verify-kernel-complete=criteria=c1 %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ONLYC1 \
// RUN:       --implicit-check-not="Incomplete C2" --implicit-check-not="Incomplete C3" \
// RUN:       --implicit-check-not="Incomplete C4"

// 不点名 criteria：只有 C3 生效，另外三条的负向输入一条都不该报。
// DEFAULT: Incomplete C3

// 点名 c1：只有 C1 生效。它同时证明了「替换」——若是追加，C3 也会报，
// 上面的 implicit-check-not 会拦下。
// ONLYC1: Incomplete C1

#in = #afir.tensor_desc<axis = [0], role = <input>>
#cp = #afir.tensor_desc<axis = [0], role = <compute>>
#out = #afir.tensor_desc<axis = [0], role = <output>>
#gm = #afir.buffer_desc<space = <gm>, life_time_axis_id = -1>
#ub = #afir.buffer_desc<space = <ub>, buffer_id = 0, life_time_axis_id = -1>
#id = affine_map<(d0) -> (d0)>

// 正向：四条判据同时成立的 kernel。每块 buffer 的地址空间写在类型上、alloc 有配对的
// dealloc（C1）；消费者用 producer 的结果值，没有缺失的顺序边（C2）；函数体只有 AFIR
// 与标准方言（C3）；用到的两个 AFIR Op 都能在实现注册表里查到（C4）。
module {
  func.func private @complete(%gm0: memref<16xf16, #afir.space<gm>> {afir.buffer_desc = #gm},
                              %gm1: memref<16xf16, #afir.space<gm>> {afir.buffer_desc = #gm}) attributes {
    afir.arg_descs = [#in, #out],
    afir.function_role = #afir.function_role<kernel_body>
  } {
    %tile = memref.alloc() {afir.buffer_desc = #ub} : memref<16xf16, #afir.space<ub>>
    %loaded = afir.load %gm0 outs(%tile) {
      afir.operand_descs = [#in], afir.result_descs = [#cp], afir.destination_descs = [#cp],
      indexing_maps = [#id, #id]
    } : memref<16xf16, #afir.space<gm>>, memref<16xf16, #afir.space<ub>> -> memref<16xf16, #afir.space<ub>>
    afir.store %loaded outs(%gm1) {
      afir.operand_descs = [#cp], afir.destination_descs = [#out],
      indexing_maps = [#id, #id]
    } : memref<16xf16, #afir.space<ub>>, memref<16xf16, #afir.space<gm>>
    memref.dealloc %tile : memref<16xf16, #afir.space<ub>>
    return
  }
}

// -----

#in = #afir.tensor_desc<axis = [0], role = <input>>
#cp = #afir.tensor_desc<axis = [0], role = <compute>>
#out = #afir.tensor_desc<axis = [0], role = <output>>
#gm = #afir.buffer_desc<space = <gm>, life_time_axis_id = -1>
#ub = #afir.buffer_desc<space = <ub>, buffer_id = 0, life_time_axis_id = -1>
#id = affine_map<(d0) -> (d0)>

// C1 负向：分配没有释放点。ADR-10 选 `memref.alloc` + `memref.dealloc` 而不是
// `alloca`，唯一的理由就是这条判据要求释放点可追踪；缺了它，所有权只能靠读代码推断。
module {
  func.func private @unreleased(%gm0: memref<16xf16, #afir.space<gm>> {afir.buffer_desc = #gm},
                                %gm1: memref<16xf16, #afir.space<gm>> {afir.buffer_desc = #gm}) attributes {
    afir.arg_descs = [#in, #out],
    afir.function_role = #afir.function_role<kernel_body>
  } {
    // expected-error @below {{Incomplete C1: allocation has no 'memref.dealloc'}}
    %tile = memref.alloc() {afir.buffer_desc = #ub} : memref<16xf16, #afir.space<ub>>
    %loaded = afir.load %gm0 outs(%tile) {
      afir.operand_descs = [#in], afir.result_descs = [#cp], afir.destination_descs = [#cp],
      indexing_maps = [#id, #id]
    } : memref<16xf16, #afir.space<gm>>, memref<16xf16, #afir.space<ub>> -> memref<16xf16, #afir.space<ub>>
    afir.store %loaded outs(%gm1) {
      afir.operand_descs = [#cp], afir.destination_descs = [#out],
      indexing_maps = [#id, #id]
    } : memref<16xf16, #afir.space<ub>>, memref<16xf16, #afir.space<gm>>
    return
  }
}

// -----

#in = #afir.tensor_desc<axis = [0], role = <input>>
#cp = #afir.tensor_desc<axis = [0], role = <compute>>
#out = #afir.tensor_desc<axis = [0], role = <output>>
#gm = #afir.buffer_desc<space = <gm>, life_time_axis_id = -1>
#ub = #afir.buffer_desc<space = <ub>, buffer_id = 0, life_time_axis_id = -1>
#id = affine_map<(d0) -> (d0)>

// C2 负向：消费者绕过 producer 的结果，直接读 destination。搬入与搬出之间因此没有任何
// 顺序边——「所有 consumer 的 wait 可验证」在这里不成立，而 IR 本身完全合法。
module {
  // expected-error @below {{Incomplete C2: 1 access pair(s) have no representable completion condition}}
  func.func private @unordered(%gm0: memref<16xf16, #afir.space<gm>> {afir.buffer_desc = #gm},
                               %gm1: memref<16xf16, #afir.space<gm>> {afir.buffer_desc = #gm}) attributes {
    afir.arg_descs = [#in, #out],
    afir.function_role = #afir.function_role<kernel_body>
  } {
    %tile = memref.alloc() {afir.buffer_desc = #ub} : memref<16xf16, #afir.space<ub>>
    // expected-error @below {{DataRace: RAW on the same storage between transfer_in and transfer_out}}
    %loaded = afir.load %gm0 outs(%tile) {
      afir.operand_descs = [#in], afir.result_descs = [#cp], afir.destination_descs = [#cp],
      indexing_maps = [#id, #id]
    } : memref<16xf16, #afir.space<gm>>, memref<16xf16, #afir.space<ub>> -> memref<16xf16, #afir.space<ub>>
    // expected-note @below {{the other access is here}}
    afir.store %tile outs(%gm1) {
      afir.operand_descs = [#cp], afir.destination_descs = [#out],
      indexing_maps = [#id, #id]
    } : memref<16xf16, #afir.space<ub>>, memref<16xf16, #afir.space<gm>>
    memref.dealloc %tile : memref<16xf16, #afir.space<ub>>
    return
  }
}

// -----

// C3 负向：AFIR 层出现了后端方言的 Op。它说明有一段语义 AFIR 表达不了，只能靠提前
// 下降到 `ascendc` 绕过去——那正是「可只用 AFIR 表示」不成立的形态。
module {
  func.func private @backend_leak() attributes {
    afir.function_role = #afir.function_role<kernel_body>
  } {
    // expected-error @below {{Incomplete C3: 'ascendc.pipe' is outside AFIR and the standard dialects}}
    %pipe = ascendc.pipe
    return
  }
}

// -----

// C4 负向：`afir.abs` 属于那 50 个「有 ODS 定义、没有注册实现」的 Op 之一。缺实现时
// 今天的失败是 `applyFullConversion` 的一句泛化的 "failed to legalize operation"；判据
// 要求的是「unsupported 编译期返回原因」，所以这里必须指名道姓，且在下降开始之前就报。
//
// 这条负例此前用的是同步族——同步族的 lowering 落地（2026-09-14）之后它不再缺实现，
// 用例如实地不再报错。之后换成 `afir.abs`，而 `afir.abs` 在 elementwise 切片的纵向
// 探针里补齐了实现，于是同样不再报错；再之后是 `afir.mean`，它随归约族补齐 max / min /
// mean / prod / any / all 一并有了实现。换成一个真没有实现的 Op，而不是把判据放宽：
// C4 转绿的路径就是逐个补实现，每补一个，这条负例就该换一个还没补的。
module {
  func.func private @unlowerable(%arg0: tensor<16x8xf32>) attributes {
    afir.function_role = #afir.function_role<kernel_body>
  } {
    // expected-error @below {{Incomplete C4: no Ascend implementation is registered for 'afir.concat'}}
    %0 = afir.concat %arg0, %arg0 {concat_axis = 0 : i32} : tensor<16x8xf32>, tensor<16x8xf32> -> tensor<32x8xf32>
    return
  }
}
