// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// RUN: not af-opt --afir-memory-allocate %s 2>&1 | FileCheck %s

module {
  func.func private @legacy_unsupported(%input: tensor<4xf16>) attributes {
    afir.function_role = #afir.function_role<kernel_body>
  } {
    %result = afir.add %input, %input {outputs = []} : (tensor<4xf16>, tensor<4xf16>) -> tensor<4xf16>
    return
  }
}

// CHECK: UnsupportedOperation: unknown AFIR operation attribute '"outputs"'
