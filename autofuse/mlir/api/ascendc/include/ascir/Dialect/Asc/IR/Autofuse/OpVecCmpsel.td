/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_AUTOFUSE_OP_VEC_CMPSEL_TD
#define ASC_AUTOFUSE_OP_VEC_CMPSEL_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "Autofuse/Base.td"
include "Autofuse/Interfaces.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Autofuse API extension wrappers -- selection / clip family
//===----------------------------------------------------------------------===//
//
// 这四个放在这里而不是 `OpVecTernaryExt.td`，是因为那个文件（连同另外六个 `*Ext.td`）
// 自最初集成起就没有被 `Ops.td` include 过，里面的 def 一个都没有被展开。放进去等于
// 没加。要用它们得先把整个文件接进来，而那会顺带带进 `where_ternary`（`AscFunc`，会
// 打成 `AscendC::Where`，而包装在全局命名空间）和 12 操作数的 `where_extend`——两个
// 今天没有 emitter、也没有任何东西会走到的 op。
//
// clip 族（clip_by_value）也放这里：它的 mem 侧包装就是拿 `Min` / `Max` 拼出来的，
// 与 cmpsel 同族，而且它与选择族共用同一个 emitter。
//
// 选择族（select / where）走的**连续 tile 形态**。两侧各一个：
//
//   mem:  Where(dst, cond, src0, src1, calCount, tmpBuffer)        duplicate.h + where.h
//   reg:  WhereExtend(dst, cond, src0, src1, calCount)             where_reg_base.h
//
// 上面那两个 op 都接不上这条路：`where_ternary` 带 `AscFunc`，会打成 `AscendC::Where`，
// 而这两个包装都在**全局**命名空间（legacy 的 `WhereApiCall` 直接打 `api_name_(`），而且
// 它没有 tmp buffer 操作数；`where_extend` 建模的是同名函数的另一个 overload——带
// firstAxis / lastAxis 与四个 second-to-last stride 的循环形态，那套参数 AFIR 的
// 「连续 tile + 元素个数」模型说不出来（与比较族的偏差 8 ① 是同一件事）。
//
// mem 侧要 tmp buffer，reg 侧不要。这不是「标准 vs 扩展」的函数——两侧都是 Autofuse 的
// 包装，只是 reg 侧那份自己在内部开寄存器，不问调用方要 UB。
//
// 两个 mem 侧包装（`duplicate.h` / `where.h` / `clipbyvalue.h`）都不是自足的翻译单元：
// 它们用 `KernelUtils::` 与 `MAX_REPEAT_TIME`，只有 `utils.h` 声明这些，所以都要声明
// `utils.h` 前导。漏掉不会在 MLIR 这一层报任何错——一路到 device 编译器才炸，报的还是
// 内联进来那段头文本的行号。reg 侧两个都自足，不声明前导。
// **后两个输入是 `AnyType` 而不是 `AscendC_LocalTensor`**：两族的包装都有标量重载
// （`where.h` 的场景 1/2/3、`clipbyvalue.h` / `clipbyvalue_reg_base.h` 的场景 2/3/4），
// 同名、实参位置相同，标量在哪个位置由 C++ 重载决议挑。发射侧打的是同一个位置上的
// SSA 值，tile 与标量没有区别，所以标量形态**不新建 def**，放宽这两位就够了。
// 第一个输入（mask / x）不放宽：那个重载两份头里都没有。
//
// dtype 校验不靠 ODS 这一层，在 conversion 的 capability 里（对标量取 `getElementTypeOrSelf`）。
def AscendC_WhereContiguousOp : VectorOp<"where_contiguous", "Where", [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse Where wrapper on a contiguous tile (mem-based API extension)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$condition,
                       AnyType:$src0, AnyType:$src1,
                       AnyType:$calCount, AscendC_LocalTensor:$sharedTmpBuffer);
  let extraClassDeclaration =
      extraClassDeclarationBase # ApiExtensionHeader<["duplicate.h", "where.h"], ["utils.h"]>.decl;
}

def AscendC_WhereContiguousRegBaseOp
    : VectorOp<"where_contiguous_reg_base", "WhereExtend", [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse WhereExtend wrapper on a contiguous tile (reg-based API extension)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$condition,
                       AnyType:$src0, AnyType:$src1,
                       AnyType:$calCount);
  let extraClassDeclaration =
      extraClassDeclarationBase # ApiExtensionHeader<["where_reg_base.h"]>.decl;
}

// clip 族（clip_by_value）。两侧同名 `ClipByValue`，与选择族同形：mem 侧要 tmp buffer，
// reg 侧不要（它转发给 `Clamp`，自己开寄存器）。
//
//   mem:  ClipByValue(dst, x, min, max, calCount, tmpBuffer)   clipbyvalue.h
//   reg:  ClipByValue(dst, x, min, max, calCount)              clipbyvalue_reg_base.h
//
// mem 侧 `clipbyvalue.h` 里原有的那个重载是 `(..., tmp_buf, calCount)`，与这一族其余包装
// （`where.h` / `true_div.h` 等）的顺序相反。那个重载没有动——legacy 仍按原顺序发；
// 头里另外补了一个 `(..., calCount, tmp_buf)` 的转发重载，让这条路上只有一种顺序，
// emitter 才能与选择族逐字共用。
def AscendC_ClipByValueContiguousOp : VectorOp<"clipbyvalue_contiguous", "ClipByValue", [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse ClipByValue wrapper on a contiguous tile (mem-based API extension)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$srcX,
                       AnyType:$srcMin, AnyType:$srcMax,
                       AnyType:$calCount, AscendC_LocalTensor:$sharedTmpBuffer);
  let extraClassDeclaration =
      extraClassDeclarationBase # ApiExtensionHeader<["clipbyvalue.h"], ["utils.h"]>.decl;
}

def AscendC_ClipByValueContiguousRegBaseOp
    : VectorOp<"clipbyvalue_contiguous_reg_base", "ClipByValue", [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse ClipByValue wrapper on a contiguous tile (reg-based API extension)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$srcX,
                       AnyType:$srcMin, AnyType:$srcMax,
                       AnyType:$calCount);
  let extraClassDeclaration =
      extraClassDeclarationBase # ApiExtensionHeader<["clipbyvalue_reg_base.h"]>.decl;
}

//===----------------------------------------------------------------------===//
// 三元的标准调用：`AscendC::Fma`
//===----------------------------------------------------------------------===//
//
// `Fma(dst, src0, src1, src2, sharedTmpBuffer, count)`（CANN 的 `adv_api/math/fma.h`，
// tmp 在 count **之前**）。放在这个文件里是因为三元那一族的 def 都在这里，而它与上面四个
// 共用 `TernaryElementwiseConversionPattern`——不是因为它属于 cmpsel。
//
// 与上面四个的差别只有一条：**它是标准调用**，没有包装头，所以不声明
// `ApiExtensionOpInterface`，而要进 `ElementwiseStockCalls.h` 的 `kIsStockCall`
// （漏了整张注册表建不起来）。与 `HypotL2Op` 是同一种情况的三元版本。
//
// **`arguments` 的顺序按 pattern 来，不按 `Fma` 来**：那条 pattern 建的是
// `TargetOp(dst, 第一个源, 第二个源, 第三个源, calCount, tmp)`，而这里的顺序是我们自己
// 定的。tmp 与 count 的真实次序由 emitter 负责（见 `Autofuse/VecTernary.h`），
// 两件事不混在一起。
//
// 三个源都是 `AscendC_LocalTensor` 而不是 `AnyType`：legacy 的
// `FmaAscIrCodegenImplV2::IsNodeValid` 用 `!IsNodeHasScalarInput(node)` 把标量挡在门外，
// 头里也没有标量重载。
def AscendC_FmaL2Op : VectorOp<"fma_l2", "Fma"> {
  let description = "`AscendC::Fma` is a vector ternary operation (L2 API) that takes a tmp buffer.\n";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src0,
                       AscendC_LocalTensor:$src1, AscendC_LocalTensor:$src2,
                       AnyType:$calCount, AscendC_LocalTensor:$sharedTmpBuffer);
}

//===----------------------------------------------------------------------===//
// 寄存器形态的一条比较（VF 里用的那个，不是上面两个 tile 级包装）
//===----------------------------------------------------------------------===//
//
// `AscendC::Reg::Compare<T, CMPMODE::EQ>(MaskReg& dst, U& src0, U& src1, MaskReg& mask)`
// （`asc/include/basic_api/reg_compute/kernel_reg_compute_vec_cmpsel_intf.h:34`）。
// 与 `ascendc.cast_reg` 同一个位置：上面那几个 `CompareExtend*` 包装是 **tile 级**的
// ——它们自己开 lane 循环、自己搬运；这一个是 VF 里的一条指令，由 `afir.vector_func`
// 的下降放进别人开好的 `__VEC_SCOPE__` 里。
//
// **结果是 `MaskReg` 而不是 `RegTensor`**：比较在寄存器形态下产出的是一条**谓词**，
// 一位一个 lane，不是一段按元素类型排布的值。这就是它不能放进 `BinaryRegOp` 那一族的
// 原因——那一族的 `dstReg` 是 `RegTensor`。谓词要变回值（legacy 在消费者不是 Select 时
// 补一条 `Reg::Duplicate(dst, 1, mask)`）是另一件事，今天不建模。
//
// 没有扩展头：`Reg::Compare` 是这一档 CANN 自带的寄存器 API，不是 Autofuse 的包装。
//
// 副作用与 `BinaryRegOp` 一族同理（目的地是操作数、没有结果，不声明就被 DCE），
// 见 `Base.td` 里那一段。
def AscendC_CompareRegOp : APIOp<"compare_reg", "Compare"> {
  let description = "`AscendC::Reg::Compare` on two vector registers, producing a predicate register";
  let arguments = (ins Arg<AscendC_MaskReg, "", [MemWrite]>:$dstMask,
                       Arg<AscendC_RegTensor, "", [MemRead]>:$src0Reg,
                       Arg<AscendC_RegTensor, "", [MemRead]>:$src1Reg,
                       AscendC_MaskReg:$maskReg, AscendC_CMPMODEAttr:$cmpMode);
  let assemblyFormat = [{
    $dstMask `,` $src0Reg `,` $src1Reg `,` $maskReg attr-dict `:` type($dstMask) `,` type($src0Reg) `,`
    type($src1Reg) `,` type($maskReg)
  }];
}

#endif // ASC_AUTOFUSE_OP_VEC_CMPSEL_TD
