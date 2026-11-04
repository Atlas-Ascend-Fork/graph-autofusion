/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_AUTOFUSE_OPS_TD
#define ASC_AUTOFUSE_OPS_TD

// Every operation Autofuse adds to the `ascendc` dialect. `../Ops.td` includes
// this file as its last line -- that include, one in `../Interfaces.td` and one
// in `../Core/Attributes.td` are the only three lines this subtree costs the
// upstream ODS. Adding an operation here costs nothing further.
//
// Files under `unused/` are deliberately not listed: they came in with the
// initial fork and have never been included, so none of their defs exist. See
// this directory's note in `api/ascendc/README.md`.

include "Autofuse/OpActivation.td"
include "Autofuse/OpDataCopy.td"
include "Autofuse/OpRegDataCopy.td"
include "Autofuse/OpVecBinary.td"
include "Autofuse/OpVecBroadcast.td"
include "Autofuse/OpVecCmpsel.td"
include "Autofuse/OpVecReduceND.td"
include "Autofuse/OpVecRemovePad.td"
include "Autofuse/OpVecUnary.td"
include "Autofuse/OpVecVconv.td"

#endif // ASC_AUTOFUSE_OPS_TD
