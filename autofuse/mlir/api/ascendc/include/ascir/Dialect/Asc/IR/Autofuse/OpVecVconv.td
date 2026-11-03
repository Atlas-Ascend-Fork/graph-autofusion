/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_AUTOFUSE_OP_VEC_VCONV_TD
#define ASC_AUTOFUSE_OP_VEC_VCONV_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "Autofuse/Base.td"
include "Autofuse/Interfaces.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

// Autofuse 的 `CastExtend` 包装，两侧各一个。上面那三个 `AscendC_CastL*Op` **不是**这里
// 要的供给：它们发的是标准的 `AscendC::Cast`，要一个 `RoundMode` 实参、不收 tmp buffer，
// 而两张对照表要的都是包装——包装自己按 (InT, OutT) 推 `RoundMode`（`GetRoundMode` /
// `GetCastRoundMode`），还替中间要转两道的那些 dtype 对开循环。
//
// **两侧同名 `CastExtend`，但签名完全不同**，这是这一批 op 里第一次：
//
//   mem:  CastExtend(dst, src, tmp_buf, size)                        cast.h
//   reg:  CastExtend(dst, src, {dims}, {out_stride}, {in_stride})    cast_reg_base.h
//
// reg 那三个实参是 `const uint32_t (&)[dim]`，`dim` 由 brace-init 的长度推出来。连续
// 形态下 `dim` 恒为 1、两个 stride 恒为 1，所以它们不进 `arguments`——与
// `where_contiguous` / `clipbyvalue_contiguous` 同一个办法，「连续」写进助记符里，
// 由 emitter 把那三个字面量打出来。真要发 2 维形态那天，改的是再建一个 op，不是给
// 这个加操作数。
//
// mem 侧要 `utils.h` 前导：`cast.h` 用了 `MAX_REPEAT_TIMES` / `Rational` / `Ceiling` /
// `ConvertToUint32`，四个都只有 `utils.h` 声明（`sign` 那一对已经因为同一个 `cast.h`
// 声明过一次）。reg 那份自足，不声明前导。
def AscendC_CastExtendContiguousOp : VectorOp<"cast_extend_contiguous", "CastExtend", [
  ApiExtensionOpInterface, OpWithDstInterface, OpWithSrcInterface
]> {
  let description = "Call the Autofuse CastExtend wrapper on a contiguous tile (cast.h)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src,
                       AscendC_LocalTensor:$sharedTmpBuffer, AnyType:$calCount);
  let extraClassDeclaration =
      extraClassDeclarationBase # ApiExtensionHeader<["cast.h"], ["utils.h"]>.decl # [{
    SmallVector<Value> getSrcTensors() { return {getSrc()}; }
  }];
}

def AscendC_CastExtendContiguousRegBaseOp : VectorOp<"cast_extend_contiguous_reg_base", "CastExtend", [
  ApiExtensionOpInterface, OpWithDstInterface, OpWithSrcInterface
]> {
  let description = "Call the Autofuse CastExtend wrapper on a contiguous tile (cast_reg_base.h)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src, AnyType:$calCount);
  let extraClassDeclaration =
      extraClassDeclarationBase # ApiExtensionHeader<["cast_reg_base.h"]>.decl # [{
    SmallVector<Value> getSrcTensors() { return {getSrc()}; }
  }];
}

//===----------------------------------------------------------------------===//
// 寄存器形态的 Cast
//===----------------------------------------------------------------------===//
//
// `AscendC::Reg::Cast<OutT, InT, trait>(dstReg, srcReg, mask)`
// （`asc/include/basic_api/reg_compute/kernel_reg_compute_vec_vconv_intf.h:35`）。上面那两个
// `CastExtend*` 包装是 **tile 级**的——它们自己开 lane 循环、自己搬运；这一个是 VF 里的
// 一条指令，由 `afir.vector_func` 的下降放进别人开好的 `__VEC_SCOPE__` 里。
//
// **`trait` 为什么是个名字而不是四个枚举操作数**：第三个模板实参的类型是
// `const CastTrait&`——引用类型的非类型模板实参只能绑到有链接的对象上，绑不了临时量，
// 所以发不出 `Cast<float, half, {ZERO, NO_SAT, ZEROING, CAST_RINT}>` 这种写法。必须指一个
// 具名的 `constexpr CastTrait`。Autofuse 的那一组定义在 `api_regbase/utils.h` 里
// （`cast_trait_float_2_half` 一族），所以这个 op 声明 `utils_reg_base.h` 作为扩展头，
// 发射器把名字原样打出去。
//
// **(InT, OutT) -> 哪个名字** 这张判断表不在这里：它是 codegen 的选择，不是 API 的形状，
// 真源在 `AFIRVectorFuncToRegBase.cpp`（抄自 legacy 的
// `v35/codegen/micro_api_call/micro_cast_api_call.cpp:18`）。这个 op 只保证「名字原样打出、
// 而且定义它的那个头被内联进来」。
//
// 副作用与 `UnaryRegOp` 一族同理（目的地是操作数、没有结果，不声明就被 DCE），见
// `Base.td` 里那一段。
// `Ceil2IntExtend(dst, src, {count}, {1U}, {1U})`。与上面那个
// `CastExtendContiguousRegBaseOp` **逐项同形**（同一个 `cast_reg_base.h`、同样三个
// brace-init 数组、同样不打显式模板实参），只差函数名——包装体自己就是一句
// `CastExtend<InT, OutT, dim, RoundMode::CAST_CEIL>(...)`。
//
// 所以它不是「又一个 cast」：RoundMode 由**包装**写死，而 `CastExtend` 那个由 legacy 按
// (输入, 输出) 推。两者都发到 `cast_reg_base.h`，但 AFIR 这一层是两个不同的 op。
def AscendC_Ceil2IntExtendRegBaseOp : VectorOp<"ceil2int_extend_contiguous_reg_base", "Ceil2IntExtend", [
  ApiExtensionOpInterface, OpWithDstInterface, OpWithSrcInterface
]> {
  let description = "Call the Autofuse Ceil2IntExtend wrapper on a contiguous tile (cast_reg_base.h)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src, AnyType:$calCount);
  let extraClassDeclaration =
      extraClassDeclarationBase # ApiExtensionHeader<["cast_reg_base.h"]>.decl # [{
    SmallVector<Value> getSrcTensors() { return {getSrc()}; }
  }];
}

def AscendC_CastRegOp : APIOp<"cast_reg", "Cast", [ApiExtensionOpInterface]> {
  let description = "`AscendC::Reg::Cast` on one vector register, converting its element type";
  let arguments = (ins Arg<AscendC_RegTensor, "", [MemWrite]>:$dstReg,
                       Arg<AscendC_RegTensor, "", [MemRead]>:$srcReg, AscendC_MaskReg:$maskReg,
                       StrAttr:$trait);
  let assemblyFormat = [{
    $dstReg `,` $srcReg `,` $maskReg attr-dict `:` type($dstReg) `,` type($srcReg) `,` type($maskReg)
  }];
  let extraClassDeclaration = extraClassDeclarationBase # ApiExtensionHeader<["utils_reg_base.h"]>.decl;
}

#endif // ASC_AUTOFUSE_OP_VEC_VCONV_TD
