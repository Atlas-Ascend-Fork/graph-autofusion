// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --target=Ascend910B1 --afir-select-implementation --split-input-file --verify-diagnostics %s

module {
  func.func @unsupported_dimension() -> index {
    // expected-error @+1 {{has no supported Ascend implementation}}
    %id = afir.block_id {dimension = 1} : index
    return %id : index
  }
}

// -----

module {
  func.func @unregistered_selection() -> index {
    // expected-error @+1 {{selected Ascend implementation is not registered for this operation}}
    %id = afir.block_id {dimension = 0, ascend.implementation = {id = "invalid", kind = "ascendc_api"}} : index
    return %id : index
  }
}

// -----

module {
  func.func @stale_selection() -> index {
    // expected-error @+1 {{selected Ascend implementation is no longer supported for the current operation and target}}
    %id = afir.block_id {dimension = 1, ascend.implementation = {id = "block_id", kind = "ascendc_api"}} : index
    return %id : index
  }
}
