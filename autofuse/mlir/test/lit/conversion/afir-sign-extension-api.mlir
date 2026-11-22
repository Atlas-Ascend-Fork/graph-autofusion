// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=CPP
// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s | FileCheck %s --check-prefix=SELECT

// `afir.sign` 的 mem 侧：`cast.h` + `sign.h`，要 tmp buffer。
//
// 这是第一个声明两个业务头的 op。 包装按 dtype 分流——f16 / f32 直接转发
// `AscendC::Sign`，其余先 `Cast`——所以两个头都得内联进 kernel。这条 lit 测不出头文件
// 内联（它不编译内联进来的文本），拦它的是 load_sign_store。这里能钉的是 IR 上建的是
// 哪个 op，以及选中的 id。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @sign_extend_f32(%input: !ascendc.local_tensor<?xf32>, %destination: !ascendc.local_tensor<?xf32>,
                             %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.sign %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @sign_extend_i32(%input: !ascendc.local_tensor<?xi32>, %destination: !ascendc.local_tensor<?xi32>,
                             %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.sign %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xi32>, !ascendc.local_tensor<?xi32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xi32>
    return
  }
}

// IR-LABEL: func.func @sign_extend_f32
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg1
// IR-NEXT: ascendc.sign_extend %arg1, %arg0, %arg2, %[[COUNT]]
// IR-NOT: ascendc.sign_extend_reg_base
// IR-NOT: afir.sign

// IR-LABEL: func.func @sign_extend_i32
// IR: ascendc.sign_extend

// 全局函数，不带 `AscendC::` 限定，元素类型是显式模板实参——与其余包装一致。
// CPP-LABEL: void sign_extend_f32(
// CPP: {{^ *}}SignExtend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// SELECT-COUNT-2: ascend.implementation = {id = "sign.tensor.extend", kind = "ascendc_api"}
