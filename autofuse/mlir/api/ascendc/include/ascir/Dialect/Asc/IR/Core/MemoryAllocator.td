/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_CORE_MEMORY_ALLOCATOR_TD
#define ASC_CORE_MEMORY_ALLOCATOR_TD

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

//===----------------------------------------------------------------------===//
// LocalMemAllocator operations
//===----------------------------------------------------------------------===//

def AscendC_LocalMemAllocatorOp : AscendC_Op<"local_mem_allocator", [AscConstructor]> {
  let summary = "Instantiate LocalMemAllocator with specified hardware position";
  let arguments = (ins I64:$hardware);
  let results = (outs AscendC_LocalMemAllocator:$allocator);
  let assemblyFormat = "$hardware attr-dict `:` type($hardware) `,` qualified(type($allocator))";
}

def AscendC_LocalMemAllocatorGetCurAddrOp : APIOp<"local_mem_allocator.get_cur_addr", "GetCurAddr", [AscMemberFunc]> {
  let summary = "Call `AscendC::LocalMemAllocator::GetCurAddr` method";
  let arguments = (ins AscendC_LocalMemAllocator:$allocator);
  let results = (outs UI32:$result);
  let assemblyFormat = [{
    $allocator attr-dict `:` qualified(type($allocator)) `,` type($result)
  }];
}

def AscendC_LocalMemAllocatorAllocOp : APIOp<"local_mem_allocator.alloc", "Alloc", [AscMemberFunc]> {
  let summary = "Call `AscendC::LocalMemAllocator::Alloc` method with tileSize as template parameter";
  let arguments = (ins
    AscendC_LocalMemAllocator:$allocator,
    AscendC_TPositionAttr:$pos,
    TypeAttr:$dtype,
    I32:$tileSize
  );
  let results = (outs AscendC_LocalTensor:$result);
  let paramTypeLists = [0, 3, 6, 4];
  let assemblyFormat = [{
    $allocator `,` $pos `,` $tileSize `,` $dtype attr-dict
    `:` qualified(type($allocator)) `,` type($tileSize) `,` type($result)
  }];
}

def AscendC_LocalMemAllocatorAllocDynamicOp : APIOp<"local_mem_allocator.alloc_dynamic", "Alloc", [AscMemberFunc]> {
  let summary = "Call `AscendC::LocalMemAllocator::Alloc` method with tileSize as runtime parameter";
  let arguments = (ins
    AscendC_LocalMemAllocator:$allocator,
    AscendC_TPositionAttr:$pos,
    TypeAttr:$dtype,
    I32:$tileSize
  );
  let results = (outs AscendC_LocalTensor:$result);
  let paramTypeLists = [0, 3, 6, 0];
  let assemblyFormat = [{
    $allocator `,` $pos `,` $tileSize `,` $dtype attr-dict
    `:` qualified(type($allocator)) `,` type($tileSize) `,` type($result)
  }];
}

#endif // ASC_CORE_MEMORY_ALLOCATOR_TD
