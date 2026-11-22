// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms of
// the CANN Open Software License Agreement Version 2.0.
// RUN: af-opt --afir-pipeline-sync %s | FileCheck %s

module {
  func.func @existing_assert(%condition: i1) {
    cf.assert %condition, "condition must hold"
    return
  }
}

// CHECK-LABEL: func.func @existing_assert
// CHECK: cf.assert %{{.*}}, "condition must hold"
