// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt %s | FileCheck %s

func.func @valid_shapes(%ranked: tensor<2x3x4xf32>, %unit: tensor<1x4xf32>, %dynamic: tensor<?x4xf32>)
    -> (tensor<2x3x4xf32>, tensor<2x3x4xf32>, tensor<3x4xf32>) {
  %0 = afir.add %ranked, %unit : (tensor<2x3x4xf32>, tensor<1x4xf32>) -> tensor<2x3x4xf32>
  %1 = afir.add %ranked, %dynamic : (tensor<2x3x4xf32>, tensor<?x4xf32>) -> tensor<2x3x4xf32>
  %2 = afir.sum %ranked {axis = 0 : i32} : tensor<2x3x4xf32> -> tensor<3x4xf32>
  return %0, %1, %2 : tensor<2x3x4xf32>, tensor<2x3x4xf32>, tensor<3x4xf32>
}

// CHECK-LABEL: func.func @valid_shapes
// CHECK: afir.add
// CHECK: afir.add
// CHECK: afir.sum
