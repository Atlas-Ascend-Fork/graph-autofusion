// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// RUN: not af-opt %s 2>&1 | FileCheck %s

module {
  func.func @legacy(%arg: memref<4xf16>) {
    %result = afir.add %arg, %arg {outputs = []} : (memref<4xf16>, memref<4xf16>) -> memref<4xf16>
    return
  }
}

// CHECK: UnsupportedOperation: unknown AFIR operation attribute '"outputs"'
