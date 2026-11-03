// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --target=Ascend910B1 --afir-select-implementation %s | FileCheck %s --check-prefix=SELECT
// RUN: af-opt --target=Ascend910B1 --afir-select-implementation --convert-afir-to-ascendc %s | FileCheck %s --check-prefix=LOWER
// RUN: not af-opt --target=Ascend910B1 --convert-afir-to-ascendc %s 2>&1 | FileCheck %s --check-prefix=MISSING

module {
  func.func @block_id() -> index {
    %id = afir.block_id {dimension = 0} : index
    return %id : index
  }
}

// SELECT-LABEL: func.func @block_id
// SELECT: afir.block_id
// SELECT-SAME: ascend.implementation = {id = "block_id", kind = "ascendc_api"}
// LOWER-LABEL: func.func @block_id
// LOWER-NOT: afir.block_id
// LOWER: ascendc.get_block_idx
// LOWER-NOT: afir.block_id
// LOWER: return
// MISSING: requires ascend.implementation dictionary attribute
