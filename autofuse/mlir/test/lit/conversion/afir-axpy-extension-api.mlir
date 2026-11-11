// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=CPP
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=CPP
// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s | FileCheck %s --check-prefix=SELECT
// RUN: af-opt --afir-select-implementation --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=SELECT

// `afir.axpy`：**第一个 tier D1 的 op**——两张支持表共用同一个扩展包装 `AxpyExtend`
// （`axpy.h`），`api_regbase/` 下没有 axpy。所以两个 `--target` 走的是同一条路、用同一组
// CHECK，与 `afir-binary-d1-both-targets.mlir` 一个形状，但那边共用的是**标准调用**、
// 这边共用的是扩展包装——`ApiFamily` 上是 `Any` 与 `Shared` 两档，正是这份用例钉的东西。
//
// 把 `Shared` 改回 `Any`，注册期 `checkFamilyMatchesHeader` 就会因为「标准调用不该带扩展头」
// 拒掉，整张注册表建不起来；改成 `MemBased` 或 `RegBased`，两条 SELECT 里必有一条落空。
//
// 另外两件只有这里钉得住的：
//
//   - alpha 的实参位置。真实签名是 `AxpyExtend(dst, src0, src1, alpha, tmp, count)`，
//     tmp 在 count 前面、alpha 在 tmp 前面。`OpVecBinaryExt.td` 里那个用不了的
//     `AscendC_AxpyExtendOp` 写的是 `$alpha, $count, $tmpBuf`——顺序错的那一版编得过、
//     只是调用打反。CPP 那条 check 逐个位置对，就是拦这个。
//   - alpha 恒建 f32。包装的形参是 `const float alpha`，不跟 `T` 走。下面 f16 那一份
//     拿到的仍然是 `f32` 常量——这与 `afir.leaky_relu` 恰好相反，那边的形参是 `const T&`。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @axpy_f32(%lhs: !ascendc.local_tensor<?xf32>, %rhs: !ascendc.local_tensor<?xf32>,
                      %destination: !ascendc.local_tensor<?xf32>, %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.axpy %lhs, %rhs outs(%destination) tmp(%tmp) {
      alpha = 5.000000e-01 : f32,
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
        !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @axpy_f16(%lhs: !ascendc.local_tensor<?xf16>, %rhs: !ascendc.local_tensor<?xf16>,
                      %destination: !ascendc.local_tensor<?xf16>, %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.axpy %lhs, %rhs outs(%destination) tmp(%tmp) {
      alpha = 5.000000e-01 : f32,
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16>),
        !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf16>
    return
  }
}

// IR-LABEL: func.func @axpy_f32
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg2
// IR: %[[ALPHA:.*]] = arith.constant 5.000000e-01 : f32
// IR: ascendc.axpy_extend_contiguous %arg2, %arg0, %arg1, %[[ALPHA]], %arg3, %[[COUNT]]
// IR-NOT: afir.axpy

// alpha 不跟 tile 走：f16 这一份拿到的仍然是 f32 常量。这一条红了就说明常量改成了按
// 元素类型建——那是 `afir.leaky_relu` 的规矩，`AxpyExtend` 的形参是 `const float`。
// IR-LABEL: func.func @axpy_f16
// IR: %[[ALPHA16:.*]] = arith.constant 5.000000e-01 : f32
// IR: ascendc.axpy_extend_contiguous %arg2, %arg0, %arg1, %[[ALPHA16]], %arg3, %{{.*}}

// 裸名，不带 `AscendC::` 限定——`axpy.h` 里的包装是全局函数。六个实参按位置对：
// dst, src0, src1, alpha, tmp, count。
//
// 常量的字面量写全（`5.000000000e-01` 就是 0.5）：emitter 打的是定长的科学计数形式，
// 写成 `(float)0.5` 不是它的前缀、对不上。两个函数里都是 `constexpr float`——alpha
// 不跟 tile 的元素类型走，f16 那份也是 f32。
// CPP-LABEL: void axpy_f32(
// CPP: constexpr float [[ALPHA:[A-Za-z0-9_]+]] = (float)5.000000000e-01
// CPP: {{^ *}}AxpyExtend({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, [[ALPHA]], {{[^,]*}}, static_cast<uint32_t>(
// CPP-LABEL: void axpy_f16(
// CPP: constexpr float [[ALPHA16:[A-Za-z0-9_]+]] = (float)5.000000000e-01
// CPP: {{^ *}}AxpyExtend({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, [[ALPHA16]], {{[^,]*}}, static_cast<uint32_t>(

// 两个 target 选中的是同一个 implementation，id 上没有 `_reg_base` 那一半。
// SELECT: ascend.implementation = {id = "axpy.tensor_tensor", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "axpy.tensor_tensor", kind = "ascendc_api"}
