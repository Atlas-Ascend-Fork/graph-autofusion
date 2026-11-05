// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Reuse the valid reg-base graph so these failures isolate the dtype boundary.
// i64 is declared by legacy but its 8-byte base case is outside this MLIR implementation.
// Bool is likewise not a migrated numeric path; do not count the legacy skipped test as coverage.
// RUN: sed 's/bf16/i64/g' %S/afir-transpose-reg-base-bf16.mlir | not af-opt --afir-select-implementation --target=Ascend950DT_950x 2>&1 | FileCheck %s --check-prefix=I64
// RUN: sed 's/bf16/i1/g' %S/afir-transpose-reg-base-bf16.mlir | not af-opt --afir-select-implementation --target=Ascend950DT_950x 2>&1 | FileCheck %s --check-prefix=BOOL
// I64: error: 'afir.transpose' op operand #0 must be
// I64-SAME: but got 'memref<?x?x?x16x8xi64>'
// BOOL: error: 'afir.transpose' op operand #0 must be
// BOOL-SAME: but got 'memref<?x?x?x16x8xi1>'
