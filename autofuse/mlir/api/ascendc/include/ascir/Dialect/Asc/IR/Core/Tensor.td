/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_CORE_TENSOR_TD
#define ASC_CORE_TENSOR_TD
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"

//===----------------------------------------------------------------------===//
// ShapeInfo operations
//===----------------------------------------------------------------------===//

def AscendC_ShapeInfoShapeOp : AscendC_Op<"shape_info.shape"> {
  let arguments = (ins AscendC_ShapeInfo:$base, AnySignlessIntegerOrIndex:$dim);
  let results = (outs AnySignlessIntegerOrIndex:$result);
  let assemblyFormat = "$base `,` $dim attr-dict type($dim) `,` type($result)";
}

def AscendC_ShapeInfoOriginalShapeOp : AscendC_Op<"shape_info.original_shape"> {
  let arguments = (ins AscendC_ShapeInfo:$base, AnySignlessIntegerOrIndex:$dim);
  let results = (outs AnySignlessIntegerOrIndex:$result);
  let assemblyFormat = "$base `,` $dim attr-dict type($dim) `,` type($result)";
}

def AscendC_GetShapeSizeOp : APIOp<"get_shape_size", "GetShapeSize", [AscFunc]> {
  let arguments = (ins AscendC_ShapeInfo:$shapeInfo);
  let results = (outs AnySignlessIntegerOrIndex:$result);
  let assemblyFormat = "$shapeInfo attr-dict `:` qualified(type($shapeInfo)) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// GlobalTensor operations
//===----------------------------------------------------------------------===//

def AscendC_GlobalTensorOp : AscendC_Op<"global_tensor", [AscConstructor]> {
  let summary = "Instantiate global tensor";
  let results = (outs AscendC_GlobalTensor:$tensor);
  let assemblyFormat = "attr-dict `:` qualified(type($tensor))";
  let hasCanonicalizeMethod = 1;
}

def AscendC_GlobalTensorBracketOp
    : AscendC_Op<"global_tensor.bracket", [Pure]> {
  let summary = "Call `AscendC::GlobalTensor::operator()` method";
  let arguments = (ins AscendC_GlobalTensor:$tensor, AnySignlessIntegerOrIndex:$index);
  let results = (outs AscendC_GlobalTensor:$result);
  let assemblyFormat = [{
    $tensor `(` $index `)`  attr-dict `:` qualified(type($tensor)) `,`
    type($index) `,` qualified(type($result))
  }];
}

def AscendC_GlobalTensorGetPhyAddrOp : APIOp<"global_tensor.get_phy_addr", "GetPhyAddr", [AscMemberFunc]> {
  let summary = "Call `AscendC::GlobalTensor::GetPhyAddr` method";
  let arguments = (ins AscendC_GlobalTensor:$tensor, Optional<UI64>:$offset);
  let results = (outs AnyRankedOrUnrankedMemRef:$result);
  let assemblyFormat = [{
    $tensor attr-dict (`,` $offset^)? `:` qualified(type($tensor))
    `,` qualified(type($result)) (`,` type($offset)^)?
  }];
}

def AscendC_GlobalTensorGetShapeInfoOp : APIOp<"global_tensor.get_shape_info", "GetShapeInfo", [AscMemberFunc]> {
  let summary = "Call `AscendC::GlobalTensor::GetShapeInfo` method";
  let arguments = (ins AscendC_GlobalTensor:$tensor);
  let results = (outs AscendC_ShapeInfo:$results);
  let assemblyFormat = [{
    $tensor attr-dict `:` qualified(type($tensor)) `,` type($results)
  }];
}

def AscendC_GlobalTensorGetSizeOp : APIOp<"global_tensor.get_size", "GetSize", [AscMemberFunc]> {
  let summary = "Call `AscendC::GlobalTensor::GetSize` method";
  let arguments = (ins AscendC_GlobalTensor:$tensor);
  let results = (outs UI64:$value);
}

def AscendC_GlobalTensorGetValueOp : APIOp<"global_tensor.get_value", "GetValue", [AscMemberFunc]> {
  let summary = "Call `AscendC::GlobalTensor::GetValue` method";
  let arguments = (ins AscendC_GlobalTensor:$tensor, AnyType:$offset);
  let results = (outs AnyType:$value);
  let assemblyFormat = [{
    $tensor `,` $offset attr-dict `:` qualified(type($tensor)) `,`
    qualified(type($offset)) `,` qualified(type($value))
  }];
}

def AscendC_GlobalTensorSetL2CacheHintOp : APIOp<"global_tensor.set_l2_cache_hint", "SetL2CacheHint", [AscMemberFunc]> {
  let summary = "Call `AscendC::GlobalTensor::SetL2CacheHint` method";
  let arguments = (ins AscendC_GlobalTensor:$tensor, AscendC_CacheModeAttr:$mode,
        AscendC_CacheRwModeAttr:$rwMode);
  let paramTypeLists = [0, -1, 3];
}

def AscendC_GlobalTensorSetShapeInfoOp : APIOp<"global_tensor.set_shape_info", "SetShapeInfo", [AscMemberFunc]> {
  let summary = "Call `AscendC::GlobalTensor::SetShapeInfo` method";
  let arguments = (ins AscendC_GlobalTensor:$tensor, AscendC_ShapeInfo:$shapeInfo);
}

def AscendC_GlobalTensorSetShapeInfoV2Op : APIOp<"global_tensor.set_shape_info_v2", "SetShapeInfo"> {
  let summary = "Call `AscendC::GlobalTensor::SetShapeInfo` method";
  let arguments = (ins AscendC_GlobalTensor:$tensor, Variadic<AnyInteger>:$dims);
}

def AscendC_GlobalTensorSetValueOp : APIOp<"global_tensor.set_value", "SetValue", [AscMemberFunc]> {
  let summary = "Call `AscendC::GlobalTensor::SetValue` method";
  let arguments = (ins AscendC_GlobalTensor:$tensor, UI64:$offset, AnyType:$value);
  let assemblyFormat = [{
    $tensor `,` $offset `,` $value attr-dict `:` qualified(type($tensor)) `,`
    qualified(type($offset)) `,` qualified(type($value))
  }];
}

def AscendC_GlobalTensorSubIndexOp
    : AscendC_Op<"global_tensor.subindex", [Pure]> {
  let summary = "Call `AscendC::GlobalTensor::operator[]` method";
  let arguments = (ins AscendC_GlobalTensor:$tensor, AnySignlessIntegerOrIndex:$index);
  let results = (outs AscendC_GlobalTensor:$result);
  let assemblyFormat = [{
    $tensor `[` $index `]`  attr-dict `:` qualified(type($tensor)) `,`
    type($index) `,` qualified(type($result))
  }];
  let hasFolder = 1;
}

def AscendC_GlobalTensorSetGlobalBufferOp : APIOp<"global_tensor.set_global_buffer", "SetGlobalBuffer", [AscMemberFunc]> {
  let summary = "Set data buffer of global tensor";
  let arguments = (ins AscendC_GlobalTensor:$tensor,
                       AnyRankedOrUnrankedMemRef:$buffer,
                       Optional<AnyInteger>:$size);
  let assemblyFormat = [{
    $tensor `,` $buffer attr-dict (`,` $size^)? `:` qualified(type($tensor)) `,`
    qualified(type($buffer)) (`,` type($size)^)?
  }];
}

//===----------------------------------------------------------------------===//
// LocalTensor operations
//===----------------------------------------------------------------------===//

def AscendC_LocalTensorOp : AscendC_Op<"local_tensor", [AscConstructor]> {
  let summary = "Instantiate local tensor";
  let results = (outs AscendC_LocalTensor:$tensor);
  let assemblyFormat = "attr-dict `:` qualified(type($tensor))";
  let hasCanonicalizeMethod = 1;
}

def AscendC_LocalTensorV2Op : AscendC_Op<"local_tensor_v2"> {
  let summary = "Instantiate local tensor";
  let arguments = (ins AscendC_TPositionAttr:$pos, UI32:$addr, UI32:$tileSize);
  let results = (outs AscendC_LocalTensor:$result);
  let assemblyFormat = "$pos `,` $addr `,` $tileSize attr-dict `:` qualified(type($result))";
}

def AscendC_LocalTensorV3Op : AscendC_Op<"local_tensor_v3"> {
  let summary = "Instantiate local tensor";
  let arguments = (ins AscendC_TPositionAttr:$pos, I32Attr:$addr, I32Attr:$tileSize);
  let results = (outs AscendC_LocalTensor:$result);
  let assemblyFormat = "$pos `,` $addr `,` $tileSize attr-dict `:` qualified(type($result))";
}

def AscendC_LocalTensorBracketOp
    : AscendC_Op<"local_tensor.bracket", [Pure]> {
  let summary = "Call `AscendC::LocalTensor::operator()` method";
  let arguments = (ins AscendC_LocalTensor:$tensor, AnySignlessIntegerOrIndex:$index);
  let results = (outs AscendC_LocalTensor:$result);
  let assemblyFormat = [{
    $tensor `(` $index `)`  attr-dict `:` qualified(type($tensor)) `,`
    type($index) `,` qualified(type($result))
  }];
}

def AscendC_LocalTensorGetLengthOp : APIOp<"local_tensor.get_length", "GetLength", [AscMemberFunc]> {
  let summary = "Call `AscendC::LocalTensor::GetLength` method";
  let arguments = (ins AscendC_LocalTensor:$tensor);
  let results = (outs UI32:$result);
  let assemblyFormat = [{
    $tensor attr-dict `:` qualified(type($tensor)) `,` type($result)
  }];
}

def AscendC_LocalTensorGetPhyAddrOp : APIOp<"local_tensor.get_phy_addr", "GetPhyAddr", [AscMemberFunc]> {
  let summary = "Call `AscendC::LocalTensor::GetPhyAddr` method";
  let arguments = (ins AscendC_LocalTensor:$tensor, Optional<UI32>:$offset);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $tensor attr-dict (`,` $offset^)? `:` qualified(type($tensor))
    `,` qualified(type($result)) (`,` type($offset)^)?
  }];
}

def AscendC_LocalTensorGetPhyAddrV2Op : APIOp<"local_tensor.get_phy_addr_v2", "GetPhyAddr"> {
  let summary = "Call `AscendC::LocalTensor::GetPhyAddr` method";
  let arguments = (ins AscendC_LocalTensor:$tensor);
  let results = (outs AnyRankedOrUnrankedMemRef:$result);
  let assemblyFormat = [{
    $tensor attr-dict `:` qualified(type($tensor))
    `,` qualified(type($result))
  }];
}

def AscendC_LocalTensorGetPositionOp : APIOp<"local_tensor.get_position", "GetPosition", [AscMemberFunc]> {
  let summary = "Call `AscendC::LocalTensor::GetPosition` method";
  let arguments = (ins AscendC_LocalTensor:$tensor);
  let results = (outs I32:$result);
  let assemblyFormat = [{
    $tensor attr-dict `:` qualified(type($tensor)) `,` type($result)
  }];
}

def AscendC_LocalTensorGetShapeInfoOp : APIOp<"local_tensor.get_shape_info", "GetShapeInfo", [AscMemberFunc]> {
  let summary = "Call `AscendC::LocalTensor::GetShapeInfo` method";
  let arguments = (ins AscendC_LocalTensor:$tensor);
  let results = (outs AscendC_ShapeInfo:$results);
  let assemblyFormat = [{
    $tensor attr-dict `:` qualified(type($tensor)) `,` qualified(type($results))
  }];
}

def AscendC_LocalTensorGetSizeOp : APIOp<"local_tensor.get_size", "GetSize", [AscMemberFunc]> {
  let summary = "Call `AscendC::LocalTensor::GetSize` method";
  let arguments = (ins AscendC_LocalTensor:$tensor);
  let results = (outs AnySignlessIntegerOrIndex:$result);
  let assemblyFormat = [{
    $tensor attr-dict `:` qualified(type($tensor)) `,` type($result)
  }];
}

def AscendC_LocalTensorGetUserTagOp : APIOp<"local_tensor.get_user_tag", "GetUserTag", [AscMemberFunc]> {
  let summary = "Call `AscendC::LocalTensor::GetUserTag` method";
  let arguments = (ins AscendC_LocalTensor:$tensor);
  let results = (outs AnySignlessIntegerOrIndex:$result);
  let assemblyFormat = [{
    $tensor attr-dict `:` qualified(type($tensor)) `,` type($result)
  }];
}

def AscendC_LocalTensorGetValueOp : APIOp<"local_tensor.get_value", "GetValue", [AscMemberFunc, OpWithSrcInterface]> {
  let summary = "Call `AscendC::LocalTensor::GetValue` method";
  let arguments = (ins AscendC_LocalTensor:$tensor, AnyType:$index);
  let results = (outs AnyType:$value);
  let assemblyFormat = [{
    $tensor `,` $index attr-dict `:` qualified(type($tensor)) `,`
    qualified(type($index)) `,` qualified(type($value))
  }];
  let extraClassDeclaration = extraClassDeclarationBase # [{
    SmallVector<Value> getSrcTensors() { return {getTensor()}; }
  }];
}

def AscendC_LocalTensorPrintOp
    : APIOp<"local_tensor.print", "Print", [AscMemberFunc]> {
  let summary = "Call `AscendC::LocalTensor::Print` method";
  let arguments = (ins AscendC_LocalTensor:$tensor, Optional<UI32>:$len);
}

def AscendC_LocalTensorReinterpretCastOp
    : APIOp<"reinterpret_cast", "ReinterpretCast", [Pure, DeclareOpInterfaceMethods<CastOpInterface>]> {
  let summary = "Call `AscendC::LocalTensor::ReinterpretCast` method";
  let arguments = (ins AscendC_LocalTensor:$in);
  let results = (outs AscendC_LocalTensor:$out);
  let assemblyFormat = [{
    $in attr-dict `:` qualified(type($in)) `to` qualified(type($out))
  }];
  let hasFolder = 1;
}

def AscendC_LocalTensorSetAddrWithOffsetOp
    : APIOp<"local_tensor.set_addr_with_offset", "SetAddrWithOffset", [AscMemberFunc]> {
  let summary = "Call `AscendC::LocalTensor::SetAddrWithOffset` method";
  let arguments = (ins AscendC_LocalTensor:$tensor, AscendC_LocalTensor:$src, UI32:$offset);
}

def AscendC_LocalTensorSetBufferLenOp
    : APIOp<"local_tensor.set_buffer_len", "SetBufferLen", [AscMemberFunc]> {
  let summary = "Call `AscendC::LocalTensor::SetBufferLen` method";
  let arguments = (ins AscendC_LocalTensor:$tensor, UI32:$dataLen);
}

def AscendC_LocalTensorSetShapeInfoOp : APIOp<"local_tensor.set_shape_info", "SetShapeInfo", [AscMemberFunc]> {
  let summary = "Call `AscendC::LocalTensor::SetShapeInfo` method";
  let arguments = (ins AscendC_LocalTensor:$tensor, AscendC_ShapeInfo:$shapeInfo);
}

def AscendC_LocalTensorSetSizeOp : APIOp<"local_tensor.set_size", "SetSize", [AscMemberFunc]> {
  let summary = "Call `AscendC::LocalTensor::SetSize` method";
  let arguments = (ins AscendC_LocalTensor:$tensor, UI32:$size);
}

def AscendC_LocalTensorSetUserTagOp : APIOp<"local_tensor.set_user_tag", "SetUserTag", [AscMemberFunc]> {
  let summary = "Call `AscendC::LocalTensor::SetUserTag` method";
  let arguments = (ins AscendC_LocalTensor:$tensor, I32:$tag);
}

def AscendC_LocalTensorSetValueOp : APIOp<"local_tensor.set_value", "SetValue", [AscMemberFunc, OpWithDstInterface]> {
  let summary = "Call `AscendC::LocalTensor::SetValue` method";
  let arguments = (ins AscendC_LocalTensor:$tensor, AnyType:$index, AnyType:$value);
  let assemblyFormat = [{
    $tensor `,` $index `,` $value attr-dict `:` qualified(type($tensor)) `,`
    type($index) `,` type($value)
  }];
  let extraClassDeclaration = extraClassDeclarationBase # [{
    Value getDst() { return {getTensor()}; }
  }];
}

def AscendC_LocalTensorSubIndexOp
    : AscendC_Op<"local_tensor.subindex", [Pure]> {
  let summary = "Call `AscendC::LocalTensor::operator[]` method";
  let arguments = (ins AscendC_LocalTensor:$tensor, AnySignlessIntegerOrIndex:$index);
  let results = (outs AscendC_LocalTensor:$result);
  let assemblyFormat = [{
    $tensor `[` $index `]`  attr-dict `:` qualified(type($tensor)) `,`
    type($index) `,` qualified(type($result))
  }];
  let hasFolder = 1;
}

def AscendC_LocalTensorToFileOp
    : APIOp<"local_tensor.to_file", "ToFile", [AscMemberFunc]> {
  let summary = "Call `AscendC::LocalTensor::ToFile` method";
  let arguments = (ins AscendC_LocalTensor:$tensor, StrAttr:$fileName);
}

#endif
