// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: env AF_AFIR_GEN_BUFFER=1 af-afir-gen | FileCheck %s

// The Schedule bridge's canonical path is memref destination-style AFIR. The
// legacy tensor demo remains covered by plan_bridge.mlir.
//
// 步 3 起 memref 自带地址空间：kernel 参数是 `#afir.space<gm>`，中间值是
// `#afir.space<ub>`。这两条断言同时也是「bridge 产出的 space 与 descriptor 一致」的
// 端到端证据——不一致时 `verifyCanonicalBufferDescriptors` 会先于 FileCheck 报错。
// CHECK: func.func private @stage0_template0(%arg0: memref<32x8xf16, #afir.space<gm>>
// 规划结果挂在参数自己的 arg attr 上，一个值一份（设计文档 5.1 步 8）。
// CHECK-SAME: {afir.buffer_desc = #afir.buffer_desc<space = <gm>, buffer_id = 0
// CHECK-SAME: %arg2: memref<32x8xf16, #afir.space<gm>>
// MLIR omits the empty result type in the printed function signature; the
// absence of a result is asserted by the following attributes/body checks.
// CHECK-SAME: ) attributes
// 局部 buffer 的定义点是它的 `memref.alloc`，规划结果挂在那里；此前它在每个
// 使用点的 `afir.*_buffer_descs` 数组里各存一份。
// CHECK: %[[ALLOC:.*]] = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(16 * s0)">, life_time_axis_id = -1>} : memref<32x8xf16, #afir.space<ub>>
// CHECK: {{.*}}afir.load %arg0 outs(%[[ALLOC]]) {{.*}}afir.destination_descs = [#afir.tensor_desc<axis = [0, 1],role = <compute>>]{{.*}}indexing_maps = [#{{.*}}, #{{.*}}]
// 中间值落在 UB：destination 的定义点上写的 space 和它的结果类型说的是同一件事。
// CHECK: %[[SUM_ALLOC:.*]] = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 5{{.*}}} : memref<32x8xf16, #afir.space<ub>>
// CHECK: {{.*}}afir.add %{{.*}}, %{{.*}} outs(%[[SUM_ALLOC]]) {{.*}}-> memref<32x8xf16, #afir.space<ub>>
// 输出搬回 GM：destination 是 kernel 的输出参数，它的 role 与 indexing_maps 仍然按
// 使用点存放（tensor descriptor 不在步 8 的迁移范围内）。
// CHECK: {{.*}}afir.store %{{.*}} outs(%arg2) {{.*}}afir.destination_descs = [#afir.tensor_desc<axis = [0, 1],role = <output>>]{{.*}}indexing_maps = [#{{.*}}, #{{.*}}]
// buffer 形态的 kernel 函数没有结果，legacy 的 `outputs =` 属性也不得出现。
// CHECK-NOT: outputs =

// 旧的 4 个按下标对齐的数组属性不再出现。
// RUN: env AF_AFIR_GEN_BUFFER=1 af-afir-gen | FileCheck %s --check-prefix=NOARRAY
// NOARRAY-NOT: afir.operand_buffer_descs
// NOARRAY-NOT: afir.result_buffer_descs
// NOARRAY-NOT: afir.destination_buffer_descs
// NOARRAY-NOT: afir.arg_buffer_descs
