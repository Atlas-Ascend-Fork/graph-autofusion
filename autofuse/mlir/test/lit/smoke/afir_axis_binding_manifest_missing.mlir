// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.


// RUN: af-opt --afir-kernel-body-signature --afir-memory-allocate --split-input-file --verify-diagnostics %s
module {
  // expected-error@+1 {{axis binding was dropped after loop generation: axis 2, role axis_size}}
  func.func private @missing() attributes {
    afir.arg_descs = [], afir.function_role = #afir.function_role<kernel_body>,
    afir.axis_bindings = [#afir.axis_binding<axis = 0, role = axis_size>,
                          #afir.axis_binding<axis = 1, role = axis_size>,
                          #afir.axis_binding<axis = 2, role = axis_size>]
  } {
    %v = arith.constant {afir.axis_binding = [#afir.axis_binding<axis = 0, role = axis_size>,
                                            #afir.axis_binding<axis = 1, role = axis_size>]} 1 : index
    %tile = afir.tiling_param "tile_size" : index
    return
  }
}

// -----
// Having the axis under another role must not satisfy the manifest.
module {
  // expected-error@+1 {{axis binding was dropped after loop generation: axis 0, role actual_size}}
  func.func private @wrong_role() attributes {
    afir.arg_descs = [], afir.function_role = #afir.function_role<kernel_body>,
    afir.axis_bindings = [#afir.axis_binding<axis = 0, role = actual_size>]
  } {
    %v = arith.constant {afir.axis_binding = #afir.axis_binding<axis = 0, role = axis_size>} 1 : index
    %tile = afir.tiling_param "tile_size" : index
    return
  }
}

// -----
module {
  // expected-error@+1 {{afir.axis_bindings must contain only #afir.axis_binding entries}}
  func.func private @malformed_manifest() attributes {
    afir.arg_descs = [], afir.function_role = #afir.function_role<kernel_body>,
    afir.axis_bindings = [0 : i64]
  } {
    %v = arith.constant {afir.axis_binding = #afir.axis_binding<axis = 0, role = axis_size>} 1 : index
    %tile = afir.tiling_param "tile_size" : index
    return
  }
}

// -----
// A bound value with no SSA users survives allocation and satisfies the manifest.
module {
  func.func private @complete() attributes {
    afir.arg_descs = [], afir.function_role = #afir.function_role<kernel_body>,
    afir.axis_bindings = [#afir.axis_binding<axis = 0, role = axis_size>]
  } {
    %v = arith.constant {afir.axis_binding = #afir.axis_binding<axis = 0, role = axis_size>} 1 : index
    %tile = afir.tiling_param "tile_size" : index
    return
  }
}

// -----
// Hand-authored canonical functions without a manifest remain supported.
module {
  func.func private @no_manifest() attributes {
    afir.arg_descs = [], afir.function_role = #afir.function_role<kernel_body>
  } {
    %v = arith.constant {afir.axis_binding = #afir.axis_binding<axis = 0, role = axis_size>} 1 : index
    %tile = afir.tiling_param "tile_size" : index
    return
  }
}
