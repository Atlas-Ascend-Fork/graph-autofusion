// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 步 3 的等价性门禁：space 在迁移期有两个载体（值的类型与 `#afir.buffer_desc` 的
// `space` 字段），两者必须逐点一致。
//
// 分别读取类型和 descriptor 的消费者必须得到相同空间，不能依赖查询入口。
// 负例在生产改写前固定一致性要求，避免双份编码分叉后仍通过 IR 校验（D2，设计文档 5.3）。

// RUN: af-opt --split-input-file --verify-diagnostics %s

#desc_1d = #afir.tensor_desc<axis = [0], role = #afir.role<compute>>
#ub = #afir.buffer_desc<space = #afir.space<ub>,>

// operand 的类型说 GM，它定义点上的 descriptor 说 UB。
func.func @operand_space_disagrees(%input: memref<4xf16, #afir.space<gm>> {afir.buffer_desc = #ub},
                                   %output: memref<4xf16, #afir.space<ub>> {afir.buffer_desc = #ub}) {
  // expected-error @+1 {{InvalidDescriptor: 'afir.operand_descs' element 0 memref memory space #afir.space<gm> disagrees with descriptor space #afir.space<ub>}}
  afir.store %input outs(%output) {
    afir.operand_descs = [#desc_1d],
    afir.destination_descs = [#desc_1d],
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
  } : memref<4xf16, #afir.space<gm>>, memref<4xf16, #afir.space<ub>>
  return
}

// -----

#desc_1d = #afir.tensor_desc<axis = [0], role = #afir.role<compute>>
#gm = #afir.buffer_desc<space = #afir.space<gm>,>
#ub = #afir.buffer_desc<space = #afir.space<ub>,>

// destination 的类型说 L1，它定义点上的 descriptor 说 UB。
func.func @destination_space_disagrees(%input: memref<4xf16, #afir.space<gm>> {afir.buffer_desc = #gm},
                                       %output: memref<4xf16, #afir.space<l1>> {afir.buffer_desc = #ub}) {
  // expected-error @+1 {{InvalidDescriptor: 'afir.destination_descs' element 0 memref memory space #afir.space<l1> disagrees with descriptor space #afir.space<ub>}}
  afir.store %input outs(%output) {
    afir.operand_descs = [#desc_1d],
    afir.destination_descs = [#desc_1d],
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
  } : memref<4xf16, #afir.space<gm>>, memref<4xf16, #afir.space<l1>>
  return
}

// -----

#desc_1d = #afir.tensor_desc<axis = [0], role = #afir.role<compute>>
#gm = #afir.buffer_desc<space = #afir.space<gm>,>

// 参数描述挂在 arg attr。被 AFIR op 使用的参数可在使用处检查；unused 参数
// 只能由函数属性检查发现。本例让 op 两端一致，单独验证 unused 参数的空间冲突。
// expected-error @+1 {{InvalidDescriptor: 'afir.arg_descs' element 2 memref memory space #afir.space<ub> disagrees with descriptor space #afir.space<gm>}}
func.func @argument_space_disagrees(%input: memref<4xf16, #afir.space<gm>> {afir.buffer_desc = #gm},
                                    %output: memref<4xf16, #afir.space<gm>> {afir.buffer_desc = #gm},
                                    %unused: memref<4xf16, #afir.space<ub>> {afir.buffer_desc = #gm})
    attributes {
  afir.arg_descs = [#desc_1d, #desc_1d, #desc_1d]
} {
  afir.store %input outs(%output) {
    afir.operand_descs = [#desc_1d],
    afir.destination_descs = [#desc_1d],
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
  } : memref<4xf16, #afir.space<gm>>, memref<4xf16, #afir.space<gm>>
  return
}
