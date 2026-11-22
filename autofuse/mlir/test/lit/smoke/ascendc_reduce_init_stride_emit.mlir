// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms of
// the CANN Open Software License Agreement Version 2.0 (the "License").
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --convert-afir-to-ascendc --target=Ascend910B1 %s | ascir-translate -mlir-to-ascendc | FileCheck %s --implicit-check-not=ReduceInit

// RUN: not ascir-translate -mlir-to-ascendc %s 2>&1 | FileCheck %s --check-prefix=UNLOWERED
// UNLOWERED: must be expanded by convert-afir-to-ascendc before emission

// Tail extent and physical pitch must remain separate through C++ emission.
func.func @reduce_init_physical_stride(%dst: !ascendc.local_tensor<1x64xf32>) {
  %a = arith.constant 1 : index
  %r = arith.constant 64 : index
  %actual = arith.constant 17 : index
  %pitch = arith.constant 32 : index
  ascendc.reduce_init %dst, %a, %r, %r, %actual, %pitch {reduceType = 2 : i32, isTailLast = false}
      : !ascendc.local_tensor<1x64xf32>, index, index, index, index, index
  return
}

// CHECK-LABEL: reduce_init_physical_stride
// CHECK: AscendC::Duplicate
