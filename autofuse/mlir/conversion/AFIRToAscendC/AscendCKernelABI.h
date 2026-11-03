/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_ASCENDCKERNELABI_H
#define AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_ASCENDCKERNELABI_H

#include <cstdint>
#include <string>

#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/IR/SymbolicBuilder.h"
#include "conversion/AFIRToAscendC/TilingSchema.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::ascend::kernel {

// 次序即 ABI 次序：`getKernelArgumentPermutation` 直接按这个枚举的值排序，所以
// `WorkspaceOffset` 必须排在 `Workspace` 之后、`Tiling` 之前——与 legacy 的
// `(..., GM_ADDR workspace, uint64_t workspace<ID>..., const T *t)` 同形。
// 这与 `afir::KernelArgRole` 的枚举**取值**无关，那边只是编码。
//
// `ScalarInput` 是例外，排在最后不代表它在 ABI 上最后：它与 `Input` **共用一个 index
// 序号空间**（legacy 的 `Data` / `ScalarData` 共用同一个 `index` 属性），所以两者在
// `getKernelArgumentKey` 下同键，`getKernelArgumentRank` 把它显式折成 `Input`，先后完全
// 由 `index` 决定。放在末尾只是为了不打乱上面那条「枚举次序 == ABI 次序」的读法。
enum class KernelArgumentKind { Input, Output, Workspace, WorkspaceOffset, Tiling, ScalarInput };

struct KernelArgumentIdentity {
  KernelArgumentKind kind;
  int64_t index = -1;

  bool operator==(const KernelArgumentIdentity &other) const {
    return kind == other.kind && index == other.index;
  }
};

FailureOr<KernelArgumentIdentity> getKernelArgumentIdentity(func::FuncOp function, unsigned index,
                                                            StringRef diagnosticPrefix = "GenKernelEntry");
std::string getKernelArgumentKey(const KernelArgumentIdentity &identity);

// Returns the old index for each new index after applying the canonical ABI order.
SmallVector<unsigned> getKernelArgumentPermutation(ArrayRef<KernelArgumentIdentity> arguments);

// AFIR 地址空间到 AscendC memory space 的**唯一**映射点（设计文档 3.1 / I4 / 约束 4）。
//
// AFIR 侧全程只出现 `#afir.space<*>`；整数编码只在这一个函数里出现，且取自 forked
// AscendC 方言自己的 `ascendc::AddressSpace` 枚举，而不是写死的魔数——此前
// `kAscendCGlobalMemorySpace = 22` 是把后端的 CCE-C 编码硬写在 AFIR 的转换层里。
//
// `space` 在 CCE-C 里没有对应地址空间时（L2 是 cache，不是地址空间）返回空属性。
Attribute getAscendCMemorySpaceAttr(MLIRContext *context, afir::Space space);

// GM 的便捷式。物化 kernel ABI 的几处调用都只关心 GM。
Attribute getAscendCGlobalMemorySpaceAttr(MLIRContext *context);
MemRefType getAscendCGlobalMemRefType(RankedTensorType tensorType);
MemRefType getAscendCWorkspaceType(MLIRContext *context);
// Workspace 字节偏移使用 ui64，与 TilingData 的 workspace<ID> 字段一致。
IntegerType getAscendCWorkspaceOffsetType(MLIRContext *context);
Type getAscendCTilingDataType(func::FuncOp func);

// 将 ranked memref 转为保留元素类型和地址空间的 unranked ABI；其他类型原样返回。
// 两层入口合并时使用，调用点恢复 ranked 视图；template 签名保留 afir.arg_descs 所需的秩。
// 调用方按参数角色排除标量和 workspace 字节基址。
Type getAscendCAbiArgumentType(Type type);

// Materialize the graph size symbols from the canonical AscendC TilingData
// argument. This is an ABI concern; AFIR loop generation consumes the returned
// bindings without knowing the concrete tiling-data representation.
FailureOr<afir::SymbolBindingMap> materializeAscendCTilingDataBindings(OpBuilder &builder, func::FuncOp func,
                                                                       ArrayRef<afir::ExprAttr> sizeVariables);

// Validate and return the canonical TilingData ABI argument.  This helper is
// intentionally safe for malformed canonical functions and always reports a
// diagnostic instead of indexing the function argument list speculatively.
FailureOr<Value> getCanonicalTilingArgument(func::FuncOp func);

// Check the pre-ABI external data arguments described by canonical AFIR.
// Only input/output GM descriptors and neutral/already-GM memrefs are accepted.
LogicalResult verifyCanonicalKernelArguments(func::FuncOp func, unsigned dataArgumentCount);

// Return false when no ABI has been materialized, true when the existing ABI
// is complete, and failure for partial or role-inconsistent ABI state.
FailureOr<bool> checkAscendCKernelABI(func::FuncOp func, afir::FunctionRole role);
void setAscendCKernelAttributes(func::FuncOp func, afir::FunctionRole role, unsigned inputCount);

}  // namespace mlir::ascend::kernel

#endif  // AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_ASCENDCKERNELABI_H
