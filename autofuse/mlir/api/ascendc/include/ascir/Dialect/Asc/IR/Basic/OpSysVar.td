/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_BASIC_OP_SYSVAR_TD
#define ASC_BASIC_OP_SYSVAR_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AscendC_GetArchVersionOp : APIOp<"get_arch_version", "GetArchVersion", [Pure, AscFunc]> {
  let description = "Obtain the architecture version";
  let arguments = (ins AnyType:$coreVersion);
  let paramTypeLists = [0];
}

def AscendC_GetBlockIdxOp : APIOp<"get_block_idx", "GetBlockIdx", [Pure, AscFunc]> {
  let description = "Obtain an index of the current block";
  let results = (outs AnySignlessIntegerOrIndex:$value);
  let assemblyFormat = "attr-dict `:` type($value)";
}

def AscendC_GetBlockNumOp : APIOp<"get_block_num", "GetBlockNum", [Pure, AscFunc]> {
  let description = "Obtain the number of active blocks";
  let results = (outs AnySignlessIntegerOrIndex:$value);
  let assemblyFormat = "attr-dict `:` type($value)";
}

def AscendC_GetVecLenOp : APIOp<"get_vec_len", "GetVecLen", [Pure, AscFunc]> {
  let results = (outs AnySignlessIntegerOrIndex:$oneRepeatSize);
  let assemblyFormat = "attr-dict `:` type($oneRepeatSize)";
}

def AscendC_GetDataBlockSizeInBytesOp : APIOp<"get_data_block_size_in_bytes", "GetDataBlockSizeInBytes", [AscFunc]> {
  let description = "Obtain the size of a datablock in bytes for the current chip version";
  let results = (outs AnySignlessIntegerOrIndex:$value);
  let assemblyFormat = "attr-dict `:` type($value)";
}

def AscendC_GetProgramCounterOp : APIOp<"get_program_counter", "GetProgramCounter", [AscFunc]> {
  let description = "Get the program counter pointer";
  let results = (outs AnySignlessIntegerOrIndex:$value);
  let assemblyFormat = "attr-dict `:` type($value)";
}

def AscendC_GetSubBlockIdxOp : APIOp<"get_sub_block_idx", "GetSubBlockIdx", [Pure, AscFunc]> {
  let description = "Get the vector core ID in AI Core";
  let results = (outs AnySignlessIntegerOrIndex:$value);
  let assemblyFormat = "attr-dict `:` type($value)";
}

def AscendC_GetSubBlockNumOp : APIOp<"get_sub_block_num", "GetSubBlockNum", [Pure, AscFunc]> {
  let description = "Get the num of vector core in aicore";
  let results = (outs AnySignlessIntegerOrIndex:$value);
  let assemblyFormat = "attr-dict `:` type($value)";
}

def AscendC_GetSystemCycleOp : APIOp<"get_system_cycle", "GetSystemCycle", [AscFunc]> {
  let description = "Get the current system cycle count";
  let results = (outs AnySignlessIntegerOrIndex:$value);
  let assemblyFormat = "attr-dict `:` type($value)";
}

def AscendC_GetTaskRatioOp : APIOp<"get_task_ratio", "GetTaskRatio", [Pure, AscFunc]> {
  let description = "Get the ratio of task cores to AI Core, return 1 in coupled mode";
  let results = (outs I64:$value);
  let assemblyFormat = "attr-dict `:` type($value)";
}

def AscendC_TrapOp : APIOp<"trap", "Trap", [AscFunc]> {
  let description = "Interrupt AI Core execution in NPU mode, equivalent to assert in CPU mode";
  let assemblyFormat = "attr-dict";
}
#endif // ASC_BASIC_OP_SYSVAR_TD
