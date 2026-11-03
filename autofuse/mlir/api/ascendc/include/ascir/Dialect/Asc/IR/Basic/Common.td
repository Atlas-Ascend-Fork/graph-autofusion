/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_BASIC_COMMON_TD
#define ASC_BASIC_COMMON_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AscendC_DataCachePreloadOp
    : APIOp<"data_cache_preload", "DataCachePreload", [AscFunc]> {
  let description = "Preload data from a specific GM address into the data cache.";
  let arguments = (ins
    AscendC_GlobalTensor:$src,
    AnyInteger:$cacheOffset
  );
  let paramTypeLists = [0, 0];
}

def AscendC_ResetMaskOp : APIOp<"reset_mask", "ResetMask", [AscFunc]>
{
  let summary = "Reset mask ";
  let assemblyFormat = "attr-dict";
}

def AscendC_SetHF32ModeOp : APIOp<"set_hf32_mode", "SetHF32Mode", [AscFunc]> {
  let summary = "Set HF32 compute mode";
  let arguments = (ins I1:$mode);
  let assemblyFormat = "$mode attr-dict `:` type($mode)";
}

def AscendC_SetHF32TransModeOp : APIOp<"set_hf32_trans_mode", "SetHF32TransMode", [AscFunc]> {
  let summary = "Set HF32 transform mode";
  let arguments = (ins I1:$trans_mode);
  let assemblyFormat = "$trans_mode attr-dict `:` type($trans_mode)";
}

def AscendC_SetMaskCountOp : APIOp<"set_mask_count", "SetMaskCount", [AscFunc]> {
  let summary = "Set mask count";
  let assemblyFormat = "attr-dict";
}

def AscendC_SetMaskNormOp : APIOp<"set_mask_norm", "SetMaskNorm", [AscFunc]> {
  let summary = "Set mask normalization";
  let assemblyFormat = "attr-dict";
}

def AscendC_SetMMLayoutTransformOp : APIOp<"set_mm_layout_transform", "SetMMLayoutTransform", [AscFunc]> {
  let summary = "Set MM layout transform mode";
  let arguments = (ins I1:$mm_layout_mode);
  let assemblyFormat = "$mm_layout_mode attr-dict `:` type($mm_layout_mode)";
}

def AscendC_SetVectorMaskL0Op : VectorOp<"set_vector_mask_l0", "SetVectorMask", []> {
  let summary = "Set vector mask in continuous mode (L0 API, Normal or Counter mode)";
  let arguments = (ins
    AnyType:$len,
    TypeAttr:$dtype,
    AscendC_MaskModeAttr:$mode);
}

def AscendC_SetVectorMaskL1Op : VectorOp<"set_vector_mask_l1", "SetVectorMask", []> {
  let summary = "Set vector mask in bit mode (L1 API, Normal or Counter mode)";
  let arguments = (ins
      AnyType:$maskHigh,
      AnyType:$maskLow,
      TypeAttr:$dtype,
      AscendC_MaskModeAttr:$mode
  );
}

#endif // ASC_BASIC_OP_COMMON_TD
