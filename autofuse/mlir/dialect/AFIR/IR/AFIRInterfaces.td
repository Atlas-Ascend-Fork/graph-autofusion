// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

//===- AFIRInterfaces.td - AFIR operation interfaces --------*- tablegen -*-===//

#ifndef AFIR_INTERFACES
#define AFIR_INTERFACES

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class AFIR_OpInterface<string name, list<Interface> bases = []>
    : OpInterface<name, bases> {
  let cppNamespace = "::mlir::afir";
}

def AFIRRoleInterface : AFIR_OpInterface<"AFIRRoleInterface"> {
  let description = "Provides the canonical role of an AFIR operation target";
  let methods = [InterfaceMethod<"Return canonical target role; fail when descriptors are absent or inconsistent",
                                 "::mlir::FailureOr<::mlir::afir::Role>", "getRole">];
}

def AFIRDataMoveInterface : AFIR_OpInterface<"AFIRDataMoveInterface", [MemoryEffectsOpInterface]> {
  let description = "Marks an AFIR data-movement operation and reuses MLIR memory effects";
}

// 操作数按 inputs | destinations | tmpBuffers 排列，getOperandPartition 通过本接口取得段长。
// AFIR_Bufferizable*OpBase 同时生成段长方法和 isBufferizedForm，派生 op 无需重复实现。
def AFIRDestinationStyleInterface : AFIR_OpInterface<"AFIRDestinationStyleInterface"> {
  let description = "Describes destination operands and tied results";
  let methods = [
    InterfaceMethod<"Return the number of destination operands", "unsigned", "getNumDestinations">,
    InterfaceMethod<"Return the result tied to a destination, if any",
                    "::mlir::FailureOr<unsigned>", "getTiedResultIndex",
                    (ins "unsigned":$destinationIndex)>,
    // 不提供默认 tmp buffer 数量，避免将未实现的计数误当为 0，造成操作数与 descriptor 错位。
    // 未继承上述 base 的 op 必须自行实现，遗漏时由链接错误暴露。
    InterfaceMethod<"Return the number of trailing tmp buffer operands",
                    "unsigned", "getNumTmpBuffers">,
    InterfaceMethod<"Whether the operation is in destination-passing (buffer) form",
                    "bool", "isBufferizedForm", (ins), /*methodBody=*/"",
                    /*defaultImplementation=*/[{ return $_op.getNumDestinations() != 0; }]>
  ];
}

def AFIRIndexingMapInterface : AFIR_OpInterface<"AFIRIndexingMapInterface"> {
  let description = "Verifies and exposes explicit indexing maps for data movement and reshape operations";
  let methods = [
    InterfaceMethod<"Return the explicit indexing maps without colliding with the ODS attribute accessor",
                    "::mlir::FailureOr<::llvm::SmallVector<::mlir::AffineMap>>",
                    "getIndexingMapList">,
    InterfaceMethod<"Verify the indexing maps", "::mlir::LogicalResult",
                    "verifyIndexingMaps">
  ];
}

def AFIRBroadcastInterface : AFIR_OpInterface<"AFIRBroadcastInterface"> {
  let description = "Verifies the shape relationship between a broadcast input and result";
  let methods = [
    InterfaceMethod<"Verify the broadcast input and result shape relationship",
                    "::mlir::LogicalResult", "verifyBroadcastShape">
  ];
}

// Direction is an ODS trait, not a second runtime enum. An operation must use
// at most one of these markers; concrete load/store and future data-move ops
// attach the appropriate trait when their ODS definitions are migrated.
def AFIR_TransferIn : NativeOpTrait<"AFIRTransferIn">;
def AFIR_TransferOut : NativeOpTrait<"AFIRTransferOut">;
def AFIR_RejectUnknownAttributes : NativeOpTrait<"AFIRRejectUnknownAttributes">;

#endif // AFIR_INTERFACES
