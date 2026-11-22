/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_BASIC_OP_PROPOSAL_TD
#define ASC_BASIC_OP_PROPOSAL_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AscendC_GetSortOffsetOp : VectorOp<"get_sort_offset", "GetSortOffset<float>", [AscFunc]> {
     let description = "Get sort offset in the sorted structure";
     let arguments = (ins UI32:$elemOffset);
     let results = (outs UI32:$result);
}


def AscendC_GetSortLenOp : VectorOp<"get_sort_len", "GetSortLen<float>", [AscFunc]> {
  let description = "Get sort length in the sorted structure";
  let arguments = (ins UI32:$elemCount);
  let results = (outs UI32:$result);
}

def AscendC_MrgSortSrcListOp : AscendC_Op<"mrg_sort_src_list", [AscConstructor]> {
  let summary = "Instantiate MrgSortSrcList";
  let results = (outs AscendC_MrgSortSrcList:$params);
  let assemblyFormat = "attr-dict `:` qualified(type($params))";
}

def AscendC_MrgSortOp : APIOp<"mrg_sort", "MrgSort", [AttrSizedOperandSegments]> {
  let summary = "Call `AscendC::MrgSort` method";
  let description = "`AscendC::MrgSort` merges sorted queues into one queue.\n";
  let arguments = (ins AnyType:$dst, AscendC_MrgSortSrcList:$sortList,
                   Variadic<UI16>:$elementCountList,
                   Variadic<UI32>:$sortedNum,
                   AnyType:$validBit, AnyType:$repeatTime,
                   UnitAttr:$isExhaustedSuspension);
}

def AscendC_MrgSort4Op : APIOp<"mrg_sort4", "MrgSort4", [AscFunc]> {
  let summary = "Call `AscendC::MrgSort4` method";
  let description = "`AscendC::MrgSort4` merges sorted Region Proposals queues into one queue.\n";
  let arguments = (ins AnyType:$dst, AscendC_MrgSortSrcList:$src,
                   AscendC_MrgSort4Info:$params);
}

def AscendC_MrgSortWithInfoOp : APIOp<"mrg_sort_with_info", "MrgSort", [AscFunc]> {
  let summary = "Call `AscendC::MrgSort` method with MrgSort4Info parameter";
  let description = "`AscendC::MrgSort` merges sorted queues into one queue using MrgSort4Info parameter.\n";
  let arguments = (ins AnyType:$dst, AscendC_MrgSortSrcList:$src,
                   AscendC_MrgSort4Info:$params);
}

def AscendC_ProposalConcatOp : VectorOp<"proposal_concat", "ProposalConcat", [AscFunc]> {
  let description = "Merge continuous elements into the corresponding positions within the Region Proposal";
  let arguments = (ins
    AscendC_LocalTensor:$dst,
    AscendC_LocalTensor:$src,
    AnyType:$repeatTime,
    AnyType:$modeNumber
  );
}

def AscendC_ProposalExtractOp : VectorOp<"proposal_extract", "ProposalExtract", [AscFunc]> {
  let description = "Extract the individual elements at the corresponding positions from the Region Proposals and then rearrange them.";
  let arguments = (ins
    AscendC_LocalTensor:$dst,
    AscendC_LocalTensor:$src,
    AnyType:$repeatTime,
    AnyType:$modeNumber
  );
}

def AscendC_RpSort16Op : APIOp<"rp_sort16", "RpSort16", [AscFunc]> {
  let summary = "Call `AscendC::RpSort16` method";
  let description = "`AscendC::RpSort16` sorts Region Proposals by score field (higher scores first), 16 proposals at a time.\n";
  let arguments = (ins AnyType:$dst, AnyType:$src, AnyType:$repeatTime);
}

def AscendC_Sort32Op : APIOp<"sort32", "Sort32", [AscFunc]> {
  let summary = "Call `AscendC::Sort32` method";
  let description = "`AscendC::Sort32` sorts 32 elements per iteration, storing results as (score, index) pairs.\n";
  let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1, AnyType:$repeatTime);
}

def AscendC_SortOp : APIOp<"sort", "Sort"> {
  let summary = "Call `AscendC::Sort` method";
  let description = "`AscendC::Sort` performs descending sort with different data layouts for different platforms.\n";
  let arguments = (ins AnyType:$dst, AnyType:$concat,
                   AnyType:$index, AnyType:$tmp,
                   AnyType:$repeatTime,
                   UnitAttr:$isFullSort);
}

def AscendC_GetMrgSortResultOp : VectorOp<"get_mrg_sort_result", "GetMrgSortResult"> {
     let description = "Get merge sort result";
     let results = (outs AnyType:$mrgSortList1, AnyType:$mrgSortList2, AnyType:$mrgSortList3, AnyType:$mrgSortList4);
  let assemblyFormat = "attr-dict `:` type($mrgSortList1) `,` type($mrgSortList2) `,`  type($mrgSortList3) `,` type($mrgSortList4)";
}

#endif //ASC_BASIC_OP_PROPOSAL_TD
