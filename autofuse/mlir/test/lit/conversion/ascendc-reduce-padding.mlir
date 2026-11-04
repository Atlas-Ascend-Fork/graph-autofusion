// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
// RUN: af-opt --convert-afir-to-ascendc --target=Ascend910B1 %s | FileCheck %s --implicit-check-not=ascendc.reduce_init
// RUN: af-opt --convert-afir-to-ascendc --target=Ascend910B1 %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=CPP --implicit-check-not=ReduceInit

// Nominal inner pitch is independent of the actual tail. This also exercises
// dynamic masks (not just a constant-folded fill of an entire buffer).
func.func @multi_r(%dst: !ascendc.local_tensor<?xf32>, %a: index, %r: index,
                   %current: index, %actual: index, %pitch: index) {
  ascendc.reduce_init %dst, %a, %r, %current, %actual, %pitch {reduceType = 2 : i32}
    : !ascendc.local_tensor<?xf32>, index, index, index, index, index
  return
}
// CHECK-LABEL: func.func @multi_r
// CHECK-SAME: %[[DST:[a-zA-Z0-9_]+]]: !ascendc.local_tensor<?xf32>, %[[A:[a-zA-Z0-9_]+]]: index, %[[R:[a-zA-Z0-9_]+]]: index,
// CHECK-SAME: %[[CURRENT:[a-zA-Z0-9_]+]]: index, %[[ACTUAL:[a-zA-Z0-9_]+]]: index, %[[PITCH:[a-zA-Z0-9_]+]]: index
// CHECK: %[[ROWS:[a-zA-Z0-9_]+]] = arith.divui %[[R]], %[[PITCH]]
// CHECK: arith.muli %[[A]], %[[ROWS]]
// CHECK: arith.remsi %[[ACTUAL]]
// CHECK: arith.shli {{.*}} : i64
// CHECK: ascendc.duplicate_l1
// CHECK-SAME: isSetMask
// CHECK: ascendc.duplicate_l0
// CHECK-SAME: isSetMask
// CHECK: ascendc.duplicate_l2
// CPP-LABEL: multi_r
// CPP: AscendC::Duplicate<float, 1>

func.func @half_max(%dst: !ascendc.local_tensor<?xf16>, %a: index, %r: index,
                    %current: index, %actual: index, %pitch: index) {
  ascendc.reduce_init %dst, %a, %r, %current, %actual, %pitch {reduceType = 1 : i32, isTailLast = true}
    : !ascendc.local_tensor<?xf16>, index, index, index, index, index
  return
}
// CHECK-LABEL: func.func @half_max
// CHECK: "emitc.constant"() {{.*}}#emitc.opaque<"AscendC::NumericLimits<half>::NegativeInfinity()">{{.*}} : () -> f16
// CHECK: ascendc.duplicate_l1
// CPP-LABEL: half_max
// CPP: AscendC::NumericLimits<half>::NegativeInfinity()
// CPP: AscendC::Duplicate<half, 1>
