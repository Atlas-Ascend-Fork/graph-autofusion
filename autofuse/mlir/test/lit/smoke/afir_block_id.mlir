// RUN: af-opt %s | FileCheck %s

module {
  func.func @block_id() -> index {
    %id = afir.block_id {dimension = 0} : index
    return %id : index
  }
}

// CHECK-LABEL: func.func @block_id
// CHECK: %{{.*}} = afir.block_id {dimension = 0 : i64} : index
