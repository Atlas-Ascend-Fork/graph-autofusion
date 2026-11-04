// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt --verify-diagnostics --split-input-file %s

// expected-error @+1 {{AFIR kernel body functions must be private}}
func.func @public_kernel_body() attributes {afir.function_role = #afir.function_role<kernel_body>} {
  return
}

// -----

// expected-error @+1 {{AFIR kernel entry functions must not be private}}
func.func private @private_entry() attributes {afir.function_role = #afir.function_role<kernel_entry>} {
  return
}

// -----

// expected-error @+1 {{'afir.function_role' is only valid on func.func}}
module attributes {afir.function_role = #afir.function_role<kernel_entry>} {}
