// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// RUN: not af-opt %s 2>&1 | FileCheck %s

module {
  func.func @legacy(%arg: memref<4xf16> {afir.tensor}) {
    return
  }
}

// CHECK: UnsupportedOperation: unknown AFIR function argument attribute '"afir.tensor"' at index 0
