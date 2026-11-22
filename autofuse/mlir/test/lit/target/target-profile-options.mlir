// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt %s | FileCheck %s
// RUN: af-opt --target=Ascend910B1 %s | FileCheck %s
// RUN: af-opt --target-profile=%S/Inputs/AscendSynthetic.ini %s | FileCheck %s
// RUN: not af-opt --target=Ascend910B1 --target-profile=%S/Inputs/AscendSynthetic.ini %s 2>&1 | FileCheck %s --check-prefix=CONFLICT
// RUN: not af-opt --target=AscendUnknown %s 2>&1 | FileCheck %s --check-prefix=UNKNOWN
// RUN: not af-opt --target=../Ascend910B1 %s 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not af-opt --target-profile=%S/Inputs/missing.ini %s 2>&1 | FileCheck %s --check-prefix=MISSING

module {
  func.func @target_profile_options() {
    return
  }
}

// CHECK-LABEL: func.func @target_profile_options
// CONFLICT: TargetSelectionConflict: --target and --target-profile cannot be specified together
// UNKNOWN: AscendUnknown: unknown builtin target platform
// INVALID: ../Ascend910B1: invalid builtin target platform name
// MISSING: missing.ini: target profile is not a readable regular file
