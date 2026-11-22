// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt --verify-diagnostics --split-input-file %s

func.func @result_rank_too_small(%input: tensor<1x8xf16>) -> tensor<8xf16> {
  // expected-error @+1 {{result rank must be >= input rank for broadcast}}
  %result = afir.broadcast %input : tensor<1x8xf16> -> tensor<8xf16>
  return %result : tensor<8xf16>
}

// -----

func.func @incompatible_broadcast_dimension(%input: tensor<2x8xf16>) -> tensor<4x8xf16> {
  // expected-error @+1 {{input shape must be broadcastable to result shape}}
  %result = afir.broadcast %input : tensor<2x8xf16> -> tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}

// -----

func.func @unsupported_element_type(%input: tensor<1x8xf64>) -> tensor<4x8xf64> {
  // expected-error @+1 {{must be Tensor or MemRef of}}
  %result = afir.broadcast %input : tensor<1x8xf64> -> tensor<4x8xf64>
  return %result : tensor<4x8xf64>
}

// -----

func.func @memref_without_destination(%input: memref<1x8xf16>) -> memref<4x8xf16> {
  // expected-error @+1 {{logical form expects tensor input and result; memref form requires a destination}}
  %result = afir.broadcast %input : memref<1x8xf16> -> memref<4x8xf16>
  return %result : memref<4x8xf16>
}

// -----

func.func @mixed_buffer_types(%input: tensor<1x8xf16>, %output: memref<4x8xf16>) {
  // expected-error @+1 {{buffer form expects non-tensor input, destination, and result}}
  %result = afir.broadcast %input outs(%output) :
    tensor<1x8xf16>, memref<4x8xf16> -> memref<4x8xf16>
  return
}

// -----

func.func @buffer_destination_mismatch(%input: memref<1x8xf16>, %destination: memref<4x8xf16>) {
  // expected-error @+1 {{buffer result and destination must have the same type}}
  %result = afir.broadcast %input outs(%destination) :
    memref<1x8xf16>, memref<4x8xf16> -> memref<8x8xf16>
  return
}
