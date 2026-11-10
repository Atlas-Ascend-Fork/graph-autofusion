/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_AUTOFUSE_VEC_CMPSEL_H
#define ASCIR_TARGET_ASC_AUTOFUSE_VEC_CMPSEL_H

#include "ascir/Target/Asc/Common.h"

namespace mlir {
namespace ascendc {

// 寄存器形态的一条 `Reg::Compare`（VF 里用的那个，不是 tile 级的 `CompareExtend` 包装）。
LogicalResult printOperation(CodeEmitter& emitter, ascendc::CompareRegOp op);

// 谓词寄存器的**声明**（`AscendC::Reg::MaskReg p;`）。`ascendc.mask_reg` 是上游 `Basic` 的
// op，但上游的 printer 表里没有它——今天只有 VF 这条路会建它（比较的目的地），所以印字
// 的那一份放在这个子树里，与 `Basic/Reg.h` 的 `RegTensorOp` 逐字同形（都只生成一行变量
// 声明），不去改 vendored 的那两个文件。
LogicalResult printOperation(CodeEmitter& emitter, ascendc::MaskRegOp op);

} // namespace ascendc
} // namespace mlir

#endif // ASCIR_TARGET_ASC_AUTOFUSE_VEC_CMPSEL_H
