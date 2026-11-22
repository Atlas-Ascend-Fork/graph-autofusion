// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-select-implementation --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=REG
// RUN: not af-opt --afir-select-implementation --target=Ascend910B1 %s 2>&1 | FileCheck %s --check-prefix=MEM

// REG: ascend.implementation = {id = "broadcast.scalar.duplicate_reg_base", kind = "ascendc_api"}
// MEM: 'afir.broadcast' op has no supported Ascend implementation

#scalar_src = #afir.tensor_desc<role = <compute>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

func.func @broadcast_scalar_i64(%destination: !ascendc.local_tensor<?xi64>) {
  %scalar = arith.constant 2 : i64
  %result = afir.broadcast %scalar outs(%destination) {
    afir.operand_descs = [#scalar_src],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : i64, !ascendc.local_tensor<?xi64> -> !ascendc.local_tensor<?xi64>
  return
}
