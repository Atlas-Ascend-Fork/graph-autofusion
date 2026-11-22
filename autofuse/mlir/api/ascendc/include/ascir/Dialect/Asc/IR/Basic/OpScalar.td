/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_BASIC_OP_SCALAR_TD
#define ASC_BASIC_OP_SCALAR_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AscendC_ScalarCastOp : APIOp<"scalar_cast", "ScalarCast", []> {
  let description = "Convert the data type of a scalar value.";
  let arguments = (ins
    AnyType:$valueIn,
    TypeAttr:$dtype,
    AscendC_RoundModeAttr:$roundMode
  );
  let results = (outs AnyType:$valueOut);
  let assemblyFormat = "$valueIn attr-dict `:` type($valueIn) `->` type($valueOut)";
}

def AscendC_ScalarGetSFFValueOp : APIOp<"scalar_get_sff_value", "ScalarGetSFFValue", [AscFunc]> {
    let description = "Get the position of the first 0 or 1 (from LSB) in a uint64_t value. Returns -1 if not found.";
    let arguments = (ins
        UI64:$valueIn,
        AnyInteger:$countValue
    );
    let results = (outs AnySignlessIntegerOrIndex:$position);
    let assemblyFormat = "operands attr-dict `:` qualified(type(operands)) `->` type($position)";
    let paramTypeLists = [0, 4];
}

def AscendC_ScalarGetCountOfValueOp : APIOp<"scalar_get_count_of_value", "ScalarGetCountOfValue", [AscFunc]> {
    let description = "Count the number of 0 or 1 bits in a uint64_t value. Returns the count of specified bit value.";
    let arguments = (ins
        UI64:$valueIn,
        I32:$countValue
    );
    let results = (outs I64:$count);
    let assemblyFormat = "operands attr-dict `:` qualified(type(operands)) `->` type($count)";
    let paramTypeLists = [0, 4];
}

def AscendC_ScalarCountLeadingZeroOp : APIOp<"scalar_count_leading_zero", "ScalarCountLeadingZero", [AscFunc]> {
    let description = "Count the number of leading zeros in a uint64_t value. Returns the count of leading zeros from MSB to the first 1.";
    let arguments = (ins
        UI64:$valueIn
    );
    let results = (outs I64:$count);
    let assemblyFormat = "operands attr-dict `:` qualified(type(operands)) `->` type($count)";
    let paramTypeLists = [0];
}

def AscendC_CountBitsCntSameAsSignBitOp : APIOp<"count_bits_cnt_same_as_sign_bit", "CountBitsCntSameAsSignBit", [AscFunc]> {
    let description = "Count the number of consecutive bits from the highest value bit that are the same as the sign bit in an int64_t value. Returns -1 if input is -1 (all 1s) or 0 (all 0s).";
    let arguments = (ins
        I64:$valueIn
    );
    let results = (outs I64:$count);
    let assemblyFormat = "operands attr-dict `:` qualified(type(operands)) `->` type($count)";
    let paramTypeLists = [0];
}

#endif //ASC_BASIC_OP_SCALAR_TD
