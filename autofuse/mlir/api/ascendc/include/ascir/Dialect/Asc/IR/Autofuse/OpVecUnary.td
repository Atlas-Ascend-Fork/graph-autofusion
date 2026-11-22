/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_AUTOFUSE_OP_VEC_UNARY_TD
#define ASC_AUTOFUSE_OP_VEC_UNARY_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "Autofuse/Base.td"
include "Autofuse/Interfaces.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

// The first of the Autofuse API extension wrappers to be wired end to end, and
// the template for the rest of them.
//
// Three things distinguish an extension op from a stock one, and all three are
// visible here:
//   - it declares `ApiExtensionOpInterface` and names the header it comes from,
//     which is how the kernel-source layer learns to bring the text in;
//   - the API name is the bare function name, because the wrappers live in the
//     GLOBAL namespace, not in `AscendC::` -- the emitter must not prepend the
//     namespace the way the stock unary emitter does;
//   - it takes a caller-provided tmp buffer, which the stock `AscendC::Abs`
//     does not.
// It deliberately does not carry `AscFunc`: the auto-generated emitter would
// print it as an `AscendC::` call.
def AscendC_AbsExtendOp : UnaryOp<"abs_extend", "AbsExtend", [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse AbsExtend wrapper (mem-based API extension)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src,
                       AscendC_LocalTensor:$sharedTmpBuffer, AnyType:$calCount);
  let extraClassDeclaration =
      extraClassDeclarationBase # ApiExtensionHeader<["abs.h"]>.decl # SrcTensorsFromSrc<>.decl;
}
// `NegExtend(dst, src, size)` is the one wrapper that both API families supply,
// under the same name and the same signature -- only the header differs. That is
// why there are two operations here rather than one carrying a header attribute:
// the header is what the kernel-source layer keys on, so it has to be a property
// of the operation the selected implementation built, not a value threaded
// through it.
class AscendC_NegExtendOpBase<string mnemonic, list<string> headers>
    : UnaryOp<mnemonic, "NegExtend", [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse NegExtend wrapper (" # !interleave(headers, ", ") # ")";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src, AnyType:$calCount);
  let extraClassDeclaration =
      extraClassDeclarationBase # ApiExtensionHeader<headers>.decl # SrcTensorsFromSrc<>.decl;
}
def AscendC_NegExtendOp : AscendC_NegExtendOpBase<"neg_extend", ["neg.h"]>;
def AscendC_NegExtendRegBaseOp : AscendC_NegExtendOpBase<"neg_extend_reg_base", ["neg_reg_base.h"]>;

// `SignExtend(dst, src, tmp_buf, size)`。与 `NegExtend` 一样两侧同名、签名也一样，多的是
// **两侧都要一块 tmp buffer**，所以 `arguments` 也进得了 base。
//
// **这是第一个真的声明两个业务头的 op**，也是前置 A（扩展头改成列表，`00cda3a0`）唯一
// 的用处兑现的地方：包装先按 dtype 分流，f16 / f32 直接转发 `AscendC::Sign`，其余 dtype
// 要先 `Cast`，于是 `cast.h` 与 `sign.h` 两个都得内联进 kernel。别拿 `prelude` 装第二个
// ——那一位的语义是「要先于 header 内联」，占掉它这个 op 就再也说不出自己还需要
// `utils.h`。
//
// 注意 `Adv/Math.td` 里那个 `def SignOp : UnaryMathOp<"sign", "Sign">` **不是**这里要的
// 可用实现：它生成的是 `AscendC::Sign`、不带 tmp，而且属于 `MathLibraryOp` 一类，被偏差 6 的
// `isReuseSource` 挡着。名字沾边，函数不是同一个。
// 两侧都要 `utils.h` 前导：两份 `sign.h` 都用了 `MAX_REPEAT_TIME`，`cast.h` 还用了
// `Rational`，两个都只有 `utils.h` 声明。**reg 侧要的也是这份 mem 的 `utils.h`**——
// `utils_reg_base.h` 不是它的 reg 版本，那是另一个文件（micro-API 的 `CastTrait`
// 常量表），这两个符号一个都没有。
//
// legacy 的 `LoadApiHeaderFiles` 里没写 `utils.h`，因为它走 `IncludeAndDefines` 那条
// 无条件内联的路；MLIR 这边得自己说出来。少了它 kernel 编不过（`MAX_REPEAT_TIME` /
// `Rational` 未声明），lit 测不出来——它不编译内联进来的文本，拦它的是 load_sign_store。
class AscendC_SignExtendOpBase<string mnemonic, list<string> headers>
    : UnaryOp<mnemonic, "SignExtend", [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse SignExtend wrapper (" # !interleave(headers, ", ") # ")";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src,
                       AscendC_LocalTensor:$sharedTmpBuffer, AnyType:$calCount);
  let extraClassDeclaration =
      extraClassDeclarationBase # ApiExtensionHeader<headers, ["utils.h"]>.decl # SrcTensorsFromSrc<>.decl;
}
// reg 侧点的是 `cast.h`，**不是** `cast_reg_base.h`——后者是存在的（reg 注册表里有这一
// 条），legacy 的 `SignAscIrCodegenImplV2::LoadApiHeaderFiles` 就是写的 `cast.h`。看着
// 像写漏了，但两张注册表灌的是同一个 `AscendCApiRegistry` 单例，reg 侧取得到，生成的
// kernel 里内联进来的确实是 mem 那份 `cast.h`。这里照抄 legacy，不替它订正。
def AscendC_SignExtendOp : AscendC_SignExtendOpBase<"sign_extend", ["cast.h", "sign.h"]>;
def AscendC_SignExtendRegBaseOp
    : AscendC_SignExtendOpBase<"sign_extend_reg_base", ["cast.h", "sign_reg_base.h"]>;

// `TanhExtend(dst, src, tmp_buf, size)` / `ErfExtend(dst, src, tmp_buf, size)`，
// `api_regbase/{tanh,erf}.h`。签名与 `SignExtend` 同形，但**只有 reg 侧有**：mem 侧
// 走的是标准的 `AscendC::Tanh` / `AscendC::Erf`（`Adv/Math.td` 里的 `TanhOp` / `ErfOp`），
// 没有包装、也没有扩展头，所以这里没有对应的 `_extend` 那一半。
//
// 两个包装体都只是把 algo / config 定死之后转发给同名的 `AscendC::` 调用，用到的
// `TanhConfig` / `ErfConfig` 由 CANN 的 `adv_api/math/{tanh,erf}.h` 声明，而那两个头
// `kernel_operator.h` 会无条件带进来（`adv_api/kernel_api.h` 第 58 / 73 行）。所以
// 不像 `SignExtend` 那样要 `utils.h` 前导——legacy 的 `LoadApiHeaderFiles` 里也只有
// 这一个头。
//
// reg 侧那块 tmp 的尺寸是 0（`CalcVoidTmpSizeV2`），与 `RemainderExtendRegBaseOp` 同一
// 种情况：形参在，所以照收不误。
// `headers` 是**列表**而不是单个头：这一类里有的包装要调另一个包装头里的东西，必须让它先
// 内联（`I0eExtend` 用 `modified_bessel_i0.h` 的 `I0_A` / `I0_B`）。**顺序就是内联顺序**，
// 照抄 legacy `LoadApiHeaderFiles` 的顺序。与 `AscendC_SignExtendOpBase` 的
// `["cast.h", "sign.h"]` 同一个办法。生成层按声明的相邻先后拓扑排
// （`Emit/ApiExtensionOrder.h`），所以**声明成什么顺序就生成什么顺序**，字典序反过来也行
// （`i0e_reg_base.h` 的字典序排在 `modified_bessel_i0_reg_base.h` 之前）。
//
// `preludes` 收那些**必须排在所有扩展头之前**的 utils 头。它表达的是比两两先后更强的约束：
// 排在**所有**扩展头前面。这一类里 utils 常被多个算子头共用（`modified_bessel_utils` 之于
// `i0` / `i0e` / `k0`），写进 `preludes` 就不必在每个 op 的 `headers` 里重复一遍先后。
class AscendC_UnaryTmpExtendRegBaseOpBase<string mnemonic, string apiName, list<string> headers,
                                          list<string> preludes = []>
    : UnaryOp<mnemonic, apiName, [ApiExtensionOpInterface]> {
  let description =
      "Call the Autofuse " # apiName # " reg-based wrapper (" # !interleave(headers, ", ") # ")";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src,
                       AscendC_LocalTensor:$sharedTmpBuffer, AnyType:$calCount);
  let extraClassDeclaration =
      extraClassDeclarationBase # ApiExtensionHeader<headers, preludes>.decl # SrcTensorsFromSrc<>.decl;
}
def AscendC_TanhExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"tanh_extend_reg_base", "TanhExtend", ["tanh_reg_base.h"]>;
def AscendC_ErfExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"erf_extend_reg_base", "ErfExtend", ["erf_reg_base.h"]>;
// `NdtrExtend` 与上面两个同形（`Xxx(dst, src, tmp, count)`、tmp 尺寸 0），**但 mem 侧
// 连对应的标准调用都没有**：`AscendC::Ndtr` 不存在，`ascir_builtin_ops_v1.cpp` 也不注册
// `Ndtr`，所以这里只有包装这一半，`Adv/Math.td` 那边不配对。
//
// 头名写的是**生成名** `ndtr_reg_base.h`，不是源文件名 `ndtr.h`——源在
// `autofuse/v35/ascendc/api_regbase/ndtr.h`，构建时生成到 `..._reg_base.h`，而
// `AddKernelIncludes` 生成的是生成名。写成 `ndtr.h` 在这一层不会红，内联那一步才解析不到。
def AscendC_NdtrExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"ndtr_extend_reg_base", "NdtrExtend", ["ndtr_reg_base.h"]>;
// 同族其余三个（第 2 批）。`AiryAiExtend` 是本类第一个要两个头的：legacy 的
// `LoadApiHeaderFiles` 给的就是 utils 在前、自己在后。
def AscendC_NdtriExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"ndtri_extend_reg_base", "NdtriExtend", ["ndtri_reg_base.h"]>;
def AscendC_LogNdtrExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"log_ndtr_extend_reg_base", "LogNdtrExtend",
                                          ["log_ndtr_reg_base.h"]>;
def AscendC_I0ExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"i0_extend_reg_base", "I0Extend", ["i0_reg_base.h"]>;
// `airy_ai.h` 调 `AutofuseSin` / `AutofuseCos`（定义在 trig utils 里），所以那个头走前导头。
def AscendC_AiryAiExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"airy_ai_extend_reg_base", "AiryAiExtend", ["airy_ai_reg_base.h"],
                                          ["trigonometric_function_utils_reg_base.h"]>;
// 第 3 批。两个包装都调 trig utils 的 `AutofuseSin` / `AutofuseCos`，并用 `bessel_j_utils`
// 里的常量 `BESSEL_J_SQRT_2_OVER_PI`；两个 utils 都进前导头。
def AscendC_BesselJ0ExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"bessel_j0_extend_reg_base", "BesselJ0Extend", ["bessel_j0_reg_base.h"],
                                          ["bessel_j_utils_reg_base.h",
                                           "trigonometric_function_utils_reg_base.h"]>;
def AscendC_BesselJ1ExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"bessel_j1_extend_reg_base", "BesselJ1Extend", ["bessel_j1_reg_base.h"],
                                          ["bessel_j_utils_reg_base.h",
                                           "trigonometric_function_utils_reg_base.h"]>;
// 第 4 批。本类第一个真的要**两个扩展头**的：`i0e.h` 用的常量表 `I0_A` / `I0_B` 只有
// `modified_bessel_i0.h` 定义，所以后者必须先内联；`ModifiedBesselImportData` /
// `ModifiedBesselExportData` 在 `modified_bessel_utils.h` 里，那个走前导头。
//
// 注意依赖序与字典序**相反**（`i0e_reg_base.h` < `modified_bessel_i0_reg_base.h`），这一格
// 到 2026-10-21 生成层改成拓扑排（f6f12395）之前是生成不出来的；前导头那一格也表达不了它，
// 前导说的是「排在所有扩展头之前」，不是两个扩展头**之间**的先后。
def AscendC_I0eExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"i0e_extend_reg_base", "I0eExtend",
                                          ["modified_bessel_i0_reg_base.h", "i0e_reg_base.h"],
                                          ["modified_bessel_utils_reg_base.h"]>;
def AscendC_I1eExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"i1e_extend_reg_base", "I1eExtend",
                                          ["modified_bessel_i1_reg_base.h", "i1e_reg_base.h"],
                                          ["modified_bessel_utils_reg_base.h"]>;
// 第 5 批。同类的另外四个，前导头都是 `modified_bessel_utils_reg_base.h`（`ImportData` /
// `ExportData` 在里面）。`K0` / `K1` 的包装调 `ModifiedBesselI0Extend` /
// `ModifiedBesselI1Extend`，所以各自要带上 `i0` / `i1` 那个头——与 `I0e` 同一种依赖，只是
// 这两对的依赖序**正好与字典序一致**（`modified_bessel_i0` < `modified_bessel_k0`）。
def AscendC_ModifiedBesselI0ExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"modified_bessel_i0_extend_reg_base", "ModifiedBesselI0Extend",
                                          ["modified_bessel_i0_reg_base.h"],
                                          ["modified_bessel_utils_reg_base.h"]>;
def AscendC_ModifiedBesselI1ExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"modified_bessel_i1_extend_reg_base", "ModifiedBesselI1Extend",
                                          ["modified_bessel_i1_reg_base.h"],
                                          ["modified_bessel_utils_reg_base.h"]>;
def AscendC_ModifiedBesselK0ExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"modified_bessel_k0_extend_reg_base", "ModifiedBesselK0Extend",
                                          ["modified_bessel_i0_reg_base.h", "modified_bessel_k0_reg_base.h"],
                                          ["modified_bessel_utils_reg_base.h"]>;
def AscendC_ModifiedBesselK1ExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"modified_bessel_k1_extend_reg_base", "ModifiedBesselK1Extend",
                                          ["modified_bessel_i1_reg_base.h", "modified_bessel_k1_reg_base.h"],
                                          ["modified_bessel_utils_reg_base.h"]>;

// `ReciprocalExtend(dst, src, tmp, size)` from autofuse/ascendc/api/reciprocal.h.
// Same shape as AbsExtend: the wrapper needs a tmp buffer for the ones-tensor it
// duplicates into, while the stock `AscendC::Reciprocal` needs none.
def AscendC_ReciprocalExtendOp
    : UnaryOp<"reciprocal_extend", "ReciprocalExtend", [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse ReciprocalExtend wrapper (mem-based API extension)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src,
                       AscendC_LocalTensor:$sharedTmpBuffer, AnyType:$calCount);
  let extraClassDeclaration =
      extraClassDeclarationBase # ApiExtensionHeader<["reciprocal.h"], ["utils.h"]>.decl # SrcTensorsFromSrc<>.decl;
}
// 浮点谓词族：输入是浮点，结果是一个字节的真值。两侧的差别不止 API 名和 tmp buffer——
// 标准调用写的是 `LocalTensor<bool>`，Autofuse 包装写的是 `LocalTensor<uint8_t>`。
// AFIR 那一层两侧都是 ui8，`ReinterpretCast` 是 lowering 补的一步。
class AscendC_PredicateExtendOpBase<string mnemonic, string apiName, list<string> headers>
    : UnaryOp<mnemonic, apiName, [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse " # apiName # " wrapper (mem-based API extension)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src,
                       AscendC_LocalTensor:$sharedTmpBuffer, AnyType:$calCount);
  let extraClassDeclaration =
      extraClassDeclarationBase # ApiExtensionHeader<headers, ["utils.h"]>.decl # SrcTensorsFromSrc<>.decl;
}
// 多一块**预先填好 1** 的 block 张量的那两个包装：
// `RsqrtExtend(dst, src, blk_tensor_with_value_1, tmp_buf, size)`（`api/rsqrt.h`）与
// `LogicalNot(dst, src, blk_tensor_with_value_1, tmp_buf, size)`（`api/logical_not.h`）。
//
// **这块张量不是第二块 tmp buffer。** tmp buffer 是未初始化的草稿纸，尺寸来自 ATT 的
// 聚合；这块是一个 32 字节的常量——包装把它当 `Div` / `Sub` / `Min` 的左操作数，配
// `src0BlkStride = 0` / `src0RepStride = 0` 反复读同一个 block。谁把 1 填进去是
// lowering 的事（见 `Implementations/ConstBlockTensor.h`），不是调度分配的事。
//
// **它的元素类型与 dst / src 无关**，由包装自己钉死：`RsqrtExtend` 收 `LocalTensor<float>`
// （中间结果一律先转 float），`LogicalNot` 收 `LocalTensor<half>`。所以显式模板实参仍然
// 只有一个 `T`（取自 src），这一块的类型是签名里写死的。
class AscendC_UnaryConstBlockExtendOpBase<string mnemonic, string apiName, list<string> headers>
    : UnaryOp<mnemonic, apiName, [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse " # apiName # " wrapper (" # !interleave(headers, ", ") # ")";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src,
                       AscendC_LocalTensor:$constBlock, AscendC_LocalTensor:$sharedTmpBuffer,
                       AnyType:$calCount);
  let extraClassDeclaration =
      extraClassDeclarationBase # ApiExtensionHeader<headers, ["utils.h"]>.decl # SrcTensorsFromSrc<>.decl;
}
// 两侧都要 `utils.h` 前导：两份包装都用了只有它声明的 `MAX_REPEAT_TIME`。
def AscendC_RsqrtExtendOp
    : AscendC_UnaryConstBlockExtendOpBase<"rsqrt_extend", "RsqrtExtend", ["rsqrt.h"]>;
// apiName 是裸名 `LogicalNot`，与 `AscendC::LogicalNot` 同名而不是同一个函数——包装是
// 全局的，生成的调用不带 `AscendC::` 限定，所以两者不会撞。mnemonic 带 `_extend`
// 后缀是为了与下面 reg 侧那个 `LogicalNotExtend` 分开。
def AscendC_LogicalNotExtendOp
    : AscendC_UnaryConstBlockExtendOpBase<"logical_not_extend", "LogicalNot", ["logical_not.h"]>;

// reg 侧的 `LogicalNotExtend(dst, src, count)`（`api_regbase/logical_not.h`）：既不要
// tmp buffer 也不要那块常量，形状与 `NegExtend` 逐字相同。**结果 dtype 也与 mem 侧不
// 一样**——包装的形参写死 `LocalTensor<uint8_t>`，而 mem 那份两端同 dtype。
//
// 不要前导：包装体只转发给 `AscendC::LogicalNot`，一个 `utils.h` 里的名字都没用。
def AscendC_LogicalNotExtendRegBaseOp
    : UnaryOp<"logical_not_extend_reg_base", "LogicalNotExtend", [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse LogicalNotExtend wrapper (logical_not_reg_base.h)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src, AnyType:$calCount);
  let extraClassDeclaration =
      extraClassDeclarationBase # ApiExtensionHeader<["logical_not_reg_base.h"]>.decl # SrcTensorsFromSrc<>.decl;
}

// step 4。`BitwiseNotExtend(dst, src, count)`（`api_regbase/bitwise_not.h`，键
// `bitwise_not_reg_base.h`——文件名与键不同，是 `where_reg_base.h` 那同一个命名惯例），
// 与 `LogicalNotExtend` 同形：不要 tmp、不要前导。legacy 的 `BitwiseNotAscIrCodegenImplV2`
// 另外还声明了一条 VF/MicroAPI 路（`GetMicroApiName` = `Not`），那条今天不建模——我们只接
// `UnaryApiCall` 那一支，与其余单点算子同一个范围。
def AscendC_BitwiseNotExtendRegBaseOp
    : UnaryOp<"bitwise_not_extend_reg_base", "BitwiseNotExtend", [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse BitwiseNotExtend wrapper (bitwise_not_reg_base.h)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src, AnyType:$calCount);
  let extraClassDeclaration =
      extraClassDeclarationBase # ApiExtensionHeader<["bitwise_not_reg_base.h"]>.decl # SrcTensorsFromSrc<>.decl;
}

// 第 6 批的一元三个，都只有 reg 侧（mem 侧 `ascir_builtin_ops_v1.cpp` 不注册它们）。
//
// `Erfinv` 在 `erfinv_reg_base.h` 里有 3 参和 4 参两个重载，legacy 走的是 `UnaryApiCall`，
// 即不传 tmp 的那个。**apiName 是裸名 `Erfinv`**，不带 `Extend`——与 `Pow` 同一种情况。
def AscendC_ErfinvExtendRegBaseOp
    : UnaryOp<"erfinv_extend_reg_base", "Erfinv", [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse Erfinv wrapper (erfinv_reg_base.h)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src, AnyType:$calCount);
  let extraClassDeclaration =
      extraClassDeclarationBase # ApiExtensionHeader<["erfinv_reg_base.h"]>.decl # SrcTensorsFromSrc<>.decl;
}
// `Exp2(dst, src, tmp_buf, calCount)` 要 tmp（包装体转发给 `Power<T, false, exp2_config>`），
// 所以走上面那个带 tmp 的 base；apiName 同样是裸名。
def AscendC_Exp2ExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"exp2_extend_reg_base", "Exp2", ["exp2_reg_base.h"]>;

// 第 7 批的一元 21 个，全部只有 reg 侧。19 个走上面那个带 tmp 的 base，`apiName` 一律是
// `XxxExtend`；另两个（`expm1` / `log1p`）不要 tmp，与 `Erfinv` 同形。
//
// 这 14 个是三角 / 双曲 / 取整那一串，每个只有自己一个头，法则与 `tanh` / `erf` 的 reg 那
// 半逐字相同。**不要用 `Adv/Math.td` 里同名的 `UnaryMathOp`**：那是 `AscendC::Acos` 等标准
// 调用，而 legacy 这 14 个的 `GetApiName` 给的都是 `XxxExtend`、`LoadApiHeaderFiles` 都给了
// `xxx_reg_base.h`，走的是 fork 自己的包装。同名的标准调用在这一批里只有 `digamma` /
// `erfc` / `trunc` 三个真的用得上（它们 legacy 一个包装头都不给）。
def AscendC_AcosExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"acos_extend_reg_base", "AcosExtend", ["acos_reg_base.h"]>;
def AscendC_AcoshExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"acosh_extend_reg_base", "AcoshExtend", ["acosh_reg_base.h"]>;
def AscendC_AsinExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"asin_extend_reg_base", "AsinExtend", ["asin_reg_base.h"]>;
def AscendC_AsinhExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"asinh_extend_reg_base", "AsinhExtend", ["asinh_reg_base.h"]>;
def AscendC_AtanExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"atan_extend_reg_base", "AtanExtend", ["atan_reg_base.h"]>;
def AscendC_AtanhExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"atanh_extend_reg_base", "AtanhExtend", ["atanh_reg_base.h"]>;
def AscendC_CeilExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"ceil_extend_reg_base", "CeilExtend", ["ceil_reg_base.h"]>;
def AscendC_CosExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"cos_extend_reg_base", "CosExtend", ["cos_reg_base.h"]>;
def AscendC_CoshExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"cosh_extend_reg_base", "CoshExtend", ["cosh_reg_base.h"]>;
def AscendC_ErfcxExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"erfcx_extend_reg_base", "ErfcxExtend", ["erfcx_reg_base.h"]>;
def AscendC_RoundExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"round_extend_reg_base", "RoundExtend", ["round_reg_base.h"]>;
def AscendC_SinExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"sin_extend_reg_base", "SinExtend", ["sin_reg_base.h"]>;
def AscendC_SinhExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"sinh_extend_reg_base", "SinhExtend", ["sinh_reg_base.h"]>;
def AscendC_TanExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"tan_extend_reg_base", "TanExtend", ["tan_reg_base.h"]>;

// Bessel Y 那一对与 `bessel_j0` / `bessel_j1` 的关系，与 `K0` / `K1` 对 `I0` / `I1` 的
// 关系同款：`bessel_y0.h` 的函数体直接调 `BesselJ0Extend`（小参数区复用 J0 的结果），所以
// 那个头要先内联。两个 utils（常量 `BESSEL_J_SQRT_2_OVER_PI` 与 `AutofuseSin` /
// `AutofuseCos`）走前导头，与已接通的 `BesselJ0Extend` 一致。
def AscendC_BesselY0ExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"bessel_y0_extend_reg_base", "BesselY0Extend",
                                          ["bessel_j0_reg_base.h", "bessel_y0_reg_base.h"],
                                          ["bessel_j_utils_reg_base.h",
                                           "trigonometric_function_utils_reg_base.h"]>;
def AscendC_BesselY1ExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"bessel_y1_extend_reg_base", "BesselY1Extend",
                                          ["bessel_j1_reg_base.h", "bessel_y1_reg_base.h"],
                                          ["bessel_j_utils_reg_base.h",
                                           "trigonometric_function_utils_reg_base.h"]>;
// `SphericalBesselJ0Extend` 只用 trig utils 的 `AutofuseSin`，**不调 `BesselJ0Extend`**
// （名字像而已），所以扩展头只有自己一个。前导两个照抄 legacy 的
// `LoadApiHeaderFiles`——`bessel_j_utils` 这一个它用不到，但多生成一个 include 不改语义，
// 而少生成会让「与 legacy 逐字同形」那条约束多一个例外。
def AscendC_SphericalBesselJ0ExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"spherical_bessel_j0_extend_reg_base", "SphericalBesselJ0Extend",
                                          ["spherical_bessel_j0_reg_base.h"],
                                          ["bessel_j_utils_reg_base.h",
                                           "trigonometric_function_utils_reg_base.h"]>;
// scaled 这一对要三个扩展头：包装调 `ModifiedBesselI0Extend`，并用 `modified_bessel_k0.h`
// 里的常量表 `K0_B`，所以 i0 与 k0 两个头都要先内联，依赖序正好与字典序一致。
def AscendC_ScaledModifiedBesselK0ExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"scaled_modified_bessel_k0_extend_reg_base",
                                          "ScaledModifiedBesselK0Extend",
                                          ["modified_bessel_i0_reg_base.h", "modified_bessel_k0_reg_base.h",
                                           "scaled_modified_bessel_k0_reg_base.h"],
                                          ["modified_bessel_utils_reg_base.h"]>;
def AscendC_ScaledModifiedBesselK1ExtendRegBaseOp
    : AscendC_UnaryTmpExtendRegBaseOpBase<"scaled_modified_bessel_k1_extend_reg_base",
                                          "ScaledModifiedBesselK1Extend",
                                          ["modified_bessel_i1_reg_base.h", "modified_bessel_k1_reg_base.h",
                                           "scaled_modified_bessel_k1_reg_base.h"],
                                          ["modified_bessel_utils_reg_base.h"]>;

// 不要 tmp 的两个，与 `Erfinv` 同形（三个实参）。`Log1p` 的 apiName 是**裸名**——
// legacy 的 `GetApiName` 给的就是 `Log1p`，而 `log1p_reg_base.h` 里确实有一个同名的
// 包装（不是 `AscendC::Log1p`，那个不存在）。
def AscendC_Expm1ExtendRegBaseOp : UnaryOp<"expm1_extend_reg_base", "Expm1Extend", [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse Expm1Extend reg-based wrapper (expm1_reg_base.h)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src, AnyType:$calCount);
  let extraClassDeclaration =
      extraClassDeclarationBase # ApiExtensionHeader<["expm1_reg_base.h"]>.decl # SrcTensorsFromSrc<>.decl;
}
def AscendC_Log1pExtendRegBaseOp : UnaryOp<"log1p_extend_reg_base", "Log1p", [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse Log1p reg-based wrapper (log1p_reg_base.h)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src, AnyType:$calCount);
  let extraClassDeclaration =
      extraClassDeclarationBase # ApiExtensionHeader<["log1p_reg_base.h"]>.decl # SrcTensorsFromSrc<>.decl;
}

def AscendC_IsnanExtendOp : AscendC_PredicateExtendOpBase<"isnan_extend", "IsnanExtend", ["isnan.h"]>;
def AscendC_IsFiniteExtendOp : AscendC_PredicateExtendOpBase<"isfinite_extend", "IsFiniteExtend", ["isfinite.h"]>;
def AscendC_IsInfExtendOp : AscendC_PredicateExtendOpBase<"isinf_extend", "IsInfExtend", ["is_inf.h"]>;

// 标准侧。只要 L2 一档——L0 / L1 的 mask+repeat 形态这一类用不到。
//
// 这三个声明在 CANN 的 `adv_api/math/is_{nan,inf,finite}.h` 里，不在 basic api 里，
// 所以值得写一句它们怎么到 kernel 的：`kernel_operator.h` →
// `interface/kernel_operator_intf.h` →（除 5101/5161/5165/5163 外无条件）
// `adv_api/kernel_api.h` → `math/is_nan.h`。V3510 不在排除名单里，所以 kernel 里只生成
// `kernel_operator.h` 就够，不需要像 legacy 的 `IncludeApiHeaderFiles()` 那样再点名。
// `AscendC::Sigmoid` 走的是同一条链。
def AscendC_IsNanL2Op : UnaryL2Op<"is_nan_l2", "IsNan">;
def AscendC_IsFiniteL2Op : UnaryL2Op<"is_finite_l2", "IsFinite">;
def AscendC_IsInfL2Op : UnaryL2Op<"is_inf_l2", "IsInf">;

// `AscendC::Log10`。与上面那三个一样是标准调用、只要 L2 一档，但它值得单独一句：
// **`Adv/Math.td` 里没有同名的 `UnaryMathOp`**，所以这不是「上游有、我们不用」，而是
// 上游那张数学库表本来就缺这一个。
//
// 为什么不去那张表里补：`UnaryMathOp` 带一个必填的 `isReuseSource` 操作数，emitter 会
// 把它当模板实参打印，于是「不要 tmp」的 `Log10` 就撞上
// `UnaryElementwiseConversionPattern` 那条 `static_assert`。按 `UnaryL2Op` 建则三个实参、
// 不打模板实参，与 legacy 的 `UnaryApiCall` 逐项同形。
def AscendC_Log10L2Op : UnaryL2Op<"log10_l2", "Log10">;

//===----------------------------------------------------------------------===//
// 多项式族：带一个 int64 模板实参的包装
//===----------------------------------------------------------------------===//

// `ChebyshevPolynomialTExtend<T, N>(dst, src, tmp, calCount)`。与
// `AscendC_UnaryTmpExtendRegBaseOpBase` 只差一件事——**多一个 int64 模板实参 `n`**，
// 而且它排在 dtype **之后**，所以 emitter 那边是另一条 body（模板实参列表写不出「跳过
// 前面的」）。这与标量二元那边的 `isScalarLatter` 是同一个形状：属性承载模板实参。
//
// **utils 走 `preludes` 而不是 `headers`，这一条非写不可。** legacy 的
// `LoadApiHeaderFiles` 给的是 utils 在前、自己在后（`XxxCal` 调 utils 里的
// `ChebyshevPolynomialCal`），而 `headers` 那一列要求**已按字典序排好**——
// `chebyshev_polynomial_t_reg_base.h` 与 `chebyshev_polynomial_u_reg_base.h` 都排在
// `chebyshev_polynomial_utils_reg_base.h` **前面**（`t` < `u`；`u_` 的 `_` < `ut` 的 `t`），
// 写进 `headers` 就是错的依赖序，而那不在这一层报错，是设备编译器在内联进来的文本里报。
// 八个一律走 preludes，不是只给对不上序的那两个——同族同形，分两种写法下一个人就得先
// 判字典序才敢加行。
class AscendC_UnaryTemplateAttrExtendRegBaseOpBase<string mnemonic, string apiName, string header,
                                                   string utilsHeader>
    : UnaryOp<mnemonic, apiName, [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse " # apiName # " reg-based wrapper (" # header # ")";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src,
                       AscendC_LocalTensor:$sharedTmpBuffer, AnyType:$calCount,
                       I64Attr:$n);
  let extraClassDeclaration =
      extraClassDeclarationBase # ApiExtensionHeader<[header], [utilsHeader]>.decl # SrcTensorsFromSrc<>.decl;
}

class AscendC_ChebyshevOpBase<string mnemonic, string apiName, string header>
    : AscendC_UnaryTemplateAttrExtendRegBaseOpBase<mnemonic, apiName, header,
                                                   "chebyshev_polynomial_utils_reg_base.h">;
def AscendC_ChebyshevPolynomialTExtendRegBaseOp
    : AscendC_ChebyshevOpBase<"chebyshev_polynomial_t_extend_reg_base", "ChebyshevPolynomialTExtend",
                              "chebyshev_polynomial_t_reg_base.h">;
def AscendC_ChebyshevPolynomialUExtendRegBaseOp
    : AscendC_ChebyshevOpBase<"chebyshev_polynomial_u_extend_reg_base", "ChebyshevPolynomialUExtend",
                              "chebyshev_polynomial_u_reg_base.h">;
def AscendC_ChebyshevPolynomialVExtendRegBaseOp
    : AscendC_ChebyshevOpBase<"chebyshev_polynomial_v_extend_reg_base", "ChebyshevPolynomialVExtend",
                              "chebyshev_polynomial_v_reg_base.h">;
def AscendC_ChebyshevPolynomialWExtendRegBaseOp
    : AscendC_ChebyshevOpBase<"chebyshev_polynomial_w_extend_reg_base", "ChebyshevPolynomialWExtend",
                              "chebyshev_polynomial_w_reg_base.h">;

class AscendC_ShiftedChebyshevOpBase<string mnemonic, string apiName, string header>
    : AscendC_UnaryTemplateAttrExtendRegBaseOpBase<mnemonic, apiName, header,
                                                   "shifted_chebyshev_polynomial_utils_reg_base.h">;
def AscendC_ShiftedChebyshevPolynomialTExtendRegBaseOp
    : AscendC_ShiftedChebyshevOpBase<"shifted_chebyshev_polynomial_t_extend_reg_base",
                                     "ShiftedChebyshevPolynomialTExtend",
                                     "shifted_chebyshev_polynomial_t_reg_base.h">;
def AscendC_ShiftedChebyshevPolynomialUExtendRegBaseOp
    : AscendC_ShiftedChebyshevOpBase<"shifted_chebyshev_polynomial_u_extend_reg_base",
                                     "ShiftedChebyshevPolynomialUExtend",
                                     "shifted_chebyshev_polynomial_u_reg_base.h">;
def AscendC_ShiftedChebyshevPolynomialVExtendRegBaseOp
    : AscendC_ShiftedChebyshevOpBase<"shifted_chebyshev_polynomial_v_extend_reg_base",
                                     "ShiftedChebyshevPolynomialVExtend",
                                     "shifted_chebyshev_polynomial_v_reg_base.h">;
def AscendC_ShiftedChebyshevPolynomialWExtendRegBaseOp
    : AscendC_ShiftedChebyshevOpBase<"shifted_chebyshev_polynomial_w_extend_reg_base",
                                     "ShiftedChebyshevPolynomialWExtend",
                                     "shifted_chebyshev_polynomial_w_reg_base.h">;

#endif // ASC_AUTOFUSE_OP_VEC_UNARY_TD
