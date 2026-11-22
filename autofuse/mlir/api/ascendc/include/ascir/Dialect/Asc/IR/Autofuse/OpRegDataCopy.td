/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_AUTOFUSE_OP_REG_DATACOPY_TD
#define ASC_AUTOFUSE_OP_REG_DATACOPY_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "Autofuse/Base.td"
include "Autofuse/Interfaces.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// UB <-> 向量寄存器：对齐搬运
//===----------------------------------------------------------------------===//
//
// **为什么不用上游的 `ascendc.data_copy_vld_reg` / `data_copy_vst_reg`**：那两个打出来的是
// `AscendC::Reg::DataCopy<T, Reg::LoadDist::...>(...)`，而**这一档 CANN 的 reg 搬运头里没有
// `DataCopy` 这个函数**——`asc/include/basic_api/reg_compute/kernel_reg_compute_datacopy_intf.h`
// 提供的是 `Load` / `LoadAlign` / `LoadUnAlign` / `Store` / `StoreAlign` / `StoreUnAlign`
// 这一组（`DataCopyMode` 只是其中几个重载的模板实参）。上游那两个 op 对的是另一代的
// 接口；照着生成的代码编不过，而且是在设备编译器上才报。
//
// 所以这里按**这一档头文件实际的签名**建两个 op，与 legacy 的 `vec_func_call` 生成的那一对
// 逐字同形（legacy 写 `MicroAPI::`，而 `namespace MicroAPI = Reg`，同一个函数）：
//
//     LoadAlign(U& dstReg, __ubuf__ T* srcAddr)
//     StoreAlign(__ubuf__ T* dstAddr, U& srcReg, MaskReg& mask)
//
// 两者都是 `__simd_callee__`：只能在 `__simd_vf__` 函数或 `__VEC_SCOPE__` 块里调。这条约束
// 由产生方（`AFIRVectorFuncToRegBase` 只在 `emitasc.vec_scope` 里建它们）保证，不在 verifier
// 里查——能在块外建出来的只有写 pass 的人，而那是 bug 不是能力不足。
//
// **`Load` 的那个 mask 去哪了**：`LoadAlign` 没有 mask 形参（整段取进来，尾部是垃圾也不碰
// 到结果，因为计算与 `StoreAlign` 都带 mask）。这与 legacy 的产物一致。
//
// **`dist` 是可选的**（M2 加的）。两个签名的第二个模板实参都有缺省值：
//
//     LoadAlign<T, LoadDist dist = DIST_NORM>(U& dstReg, __ubuf__ T* srcAddr)
//     StoreAlign<T, StoreDist dist = DIST_NORM>(__ubuf__ T* dstAddr, U& srcReg, MaskReg& mask)
//
// 同 dtype 的 VF 里一条都不写（`dist` 缺席，产物与 M1 逐字不变）；跨 dtype 的 VF 要靠它
// 把窄 dtype 的元素**取进宽的 lane 槽位**（`DIST_UNPACK_*`）、把宽 lane 的结果**压回窄
// dtype**（`DIST_PACK_*`）。不写就是 `DIST_NORM`：编得过、算得出，但每个 lane 读的是相邻
// 的几个窄元素而不是一个——**静默算错**。所以缺席与写上不是「省不省事」，而是两种语义；
// 选哪一个由 `AFIRVectorFuncToRegBase` 按边界 dtype 与 VF 的最宽 dtype 定，与 legacy 的
// `micro_load_api_call.cpp` / `micro_store_api_call.cpp` 同一套判据。

// **副作用必须声明。** 这两个 op 没有结果（目的地是操作数），所以不声明副作用时 MLIR 会
// 判它们是死代码：`--canonicalize` 会把整个 `__VEC_SCOPE__` 连同里面的计算一起抹掉，
// 而产物仍然编得过——两次 Load、一次 Store、中间什么都没有。实测踩过一次，M1 的那格 E2E
// 当时「接管成功」但 kernel 里一条计算都没有。
def AscendC_LoadAlignRegOp : AscendC_Op<"load_align_reg", []> {
  let summary = "Load one aligned vector of elements from UB into a register";
  let arguments = (ins Arg<AscendC_RegTensor, "", [MemWrite]>:$dstReg,
                       Arg<AnyRankedOrUnrankedMemRef, "", [MemRead]>:$src,
                       OptionalAttr<AscendC_LoadDist>:$dist);
  let assemblyFormat = [{
    $dstReg `,` $src attr-dict `:` type($dstReg) `,` type($src)
  }];
  let extraClassDeclaration = [{
    ::llvm::StringRef getAPIName() { return "LoadAlign"; }
  }];
}

def AscendC_StoreAlignRegOp : AscendC_Op<"store_align_reg", []> {
  let summary = "Store one aligned vector of elements from a register into UB";
  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "", [MemWrite]>:$dst,
                       Arg<AscendC_RegTensor, "", [MemRead]>:$srcReg, AscendC_MaskReg:$maskReg,
                       OptionalAttr<AscendC_StoreDist>:$dist);
  let assemblyFormat = [{
    $dst `,` $srcReg `,` $maskReg attr-dict `:` type($dst) `,` type($srcReg) `,` type($maskReg)
  }];
  let extraClassDeclaration = [{
    ::llvm::StringRef getAPIName() { return "StoreAlign"; }
  }];
}

#endif // ASC_AUTOFUSE_OP_REG_DATACOPY_TD
