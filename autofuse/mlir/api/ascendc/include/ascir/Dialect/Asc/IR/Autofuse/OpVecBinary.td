/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_AUTOFUSE_OP_VEC_BINARY_TD
#define ASC_AUTOFUSE_OP_VEC_BINARY_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "Autofuse/Base.td"
include "Autofuse/Interfaces.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

// 逐位与的两侧。这是二元一类里第一个两侧调用不同名的 op，所以两个 def 都在这里：
//
//   - mem 侧走包装 `BitwiseAndExtend(dst, src0, src1, tmp, size)`
//     （`autofuse/ascendc/api/bitwise_and.h`）。包装对 `int16_t` / `uint16_t` 直接转发
//     `AscendC::And`，其余 dtype 先 `ReinterpretCast<int16_t>` 再算——**tmp buffer 它
//     一次都没用**，但形参在签名里，少一个实参就调不上，所以这里照旧收。
//   - reg 侧是标准的 `AscendC::BitwiseAnd`，不带 tmp，也不引扩展头（它声明在 CANN 的
//     `adv_api/math/bitwise_and.h`，与 `IsNan` 一类同一条链，由 `kernel_operator.h`
//     无条件带进来）。
//
// 于是是两个 op 而不是一个带开关的 op：扩展头是 kernel source 层的键，得是「选中的
// implementation 建出来的那个 op」自己的属性。
def AscendC_BitwiseAndExtendOp
    : BinaryOp<"bitwise_and_extend", "BitwiseAndExtend", [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse BitwiseAndExtend wrapper (mem-based API extension)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src0, AscendC_LocalTensor:$src1,
                       AscendC_LocalTensor:$sharedTmpBuffer, AnyType:$calCount);
  let extraClassDeclaration = extraClassDeclarationBase # ApiExtensionHeader<["bitwise_and.h"]>.decl # [{
    SmallVector<Value> getSrcTensors() { return {getSrc0(), getSrc1()}; }
  }];
}
def AscendC_BitwiseAndL2Op : BinaryL2Op<"bitwise_and_l2", "BitwiseAnd">;

// `AscendC::Hypot(dst, src0, src1, tmp, calCount)`。标准调用，所以与 `BitwiseAndL2Op`
// 同一侧，但**多一块 tmp buffer**，而上游的 `BinaryL2Op` 的 `arguments` 是写死的四格
// （dst / src0 / src1 / calCount），装不下第五个实参。
//
// 所以这里不继承 `BinaryL2Op`，而是直接按 `BinaryOp` 展开 `arguments`——与
// `AscendC_BinaryTmpExtendRegBaseOpBase` 的 `arguments` 逐字相同，差别只在**不声明**
// `ApiExtensionOpInterface`：没有包装头可引，这一位的空与非空正是 `kIsStockCall` 查的
// 那个方向（见 `ElementwiseStockCalls.h`）。
def AscendC_HypotL2Op : BinaryOp<"hypot_l2", "Hypot"> {
  let description = "`AscendC::Hypot` is a vector binary operation (L2 API) that takes a tmp buffer.\n";
  let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1,
                       AnyType:$sharedTmpBuffer, AnyType:$calCount);
}

// `FloorDivExtend` 是二元里的 `NegExtend`：**两张支持表给的是同一个名字**，只有头文件
// 不同。所以与 NegExtend 一样写成一个 base 加两个 def——头是 kernel source 层的键，得是
// 选中的 implementation 建出来的那个 op 自己的属性，不能是穿过去的一个值。
//
// 与 NegExtend 不同的是这一对**还差一块 tmp buffer**：mem 侧那份要拿它装 `Div` 的中间
// 商再交给 `Floor`，reg 侧在寄存器里做完，一块都不要。所以两个 def 的 `arguments` 不同，
// base 只能共享到 trait 与描述这一层。
//
// mem 侧要 `utils.h` 前导：`floor_div.h` 用了 `MAX_REPEAT_TIME`，那是 fork 自己的常量，
// 只有 `utils.h` 声明。少了它生成的 kernel 编不过——`where.h` 当初就是栽在这条上。
class AscendC_FloorDivExtendOpBase<string mnemonic, list<Trait> traits = []>
    : BinaryOp<mnemonic, "FloorDivExtend", [ApiExtensionOpInterface] # traits> {
  let description = "Call the Autofuse FloorDivExtend wrapper";
}
def AscendC_FloorDivExtendOp : AscendC_FloorDivExtendOpBase<"floor_div_extend"> {
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src0, AscendC_LocalTensor:$src1,
                       AscendC_LocalTensor:$sharedTmpBuffer, AnyType:$calCount);
  let extraClassDeclaration =
      extraClassDeclarationBase # ApiExtensionHeader<["floor_div.h"], ["utils.h"]>.decl # [{
    SmallVector<Value> getSrcTensors() { return {getSrc0(), getSrc1()}; }
  }];
}
def AscendC_FloorDivExtendRegBaseOp : AscendC_FloorDivExtendOpBase<"floor_div_extend_reg_base"> {
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src0, AscendC_LocalTensor:$src1,
                       AnyType:$calCount);
  let extraClassDeclaration =
      extraClassDeclarationBase # ApiExtensionHeader<["floor_div_reg_base.h"]>.decl # [{
    SmallVector<Value> getSrcTensors() { return {getSrc0(), getSrc1()}; }
  }];
}

// `RemainderExtend` 比 `FloorDivExtend` 还整齐：两张表同名、**而且都要一块 tmp
// buffer**，于是连 `arguments` 都能进 base，两个 def 各自只剩一行扩展头。
//
// reg 侧那块 tmp 的尺寸是 0（legacy 的 `CalcVoidTmpSizeV2`）。它不是拿来算的——
// `remainder_reg_base.h` 里 `RemainderExtend` 有 4 参和 5 参两个重载，legacy 生成的是
// 5 参那个，少一个实参就挑到另一个去了。所以这里照收不误。
class AscendC_RemainderExtendOpBase<string mnemonic, list<string> headers>
    : BinaryOp<mnemonic, "RemainderExtend", [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse RemainderExtend wrapper (" # !interleave(headers, ", ") # ")";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src0, AscendC_LocalTensor:$src1,
                       AscendC_LocalTensor:$sharedTmpBuffer, AnyType:$calCount);
  let extraClassDeclaration = extraClassDeclarationBase # ApiExtensionHeader<headers>.decl # [{
    SmallVector<Value> getSrcTensors() { return {getSrc0(), getSrc1()}; }
  }];
}
def AscendC_RemainderExtendOp : AscendC_RemainderExtendOpBase<"remainder_extend", ["remainder.h"]>;
def AscendC_RemainderExtendRegBaseOp
    : AscendC_RemainderExtendOpBase<"remainder_extend_reg_base", ["remainder_reg_base.h"]>;

// 第 6 批的二元三个。`FmodExtend(dst, src1, src2, tmp_buf, size)` 与
// `RemainderExtend` 的 reg 那份逐字同形，所以同一种写法。
//
// **`Fmod` 与 `Mod` 两个调度节点共用这一个 def**：legacy 两边的 `GetApiName` 都是
// `FmodExtend`，头也都是 `fmod_reg_base.h`，与 `Gelu`（两张支持表同一个调用）同一种情况。
// 建两个 def 会让两条 implementation 指向同名的 AscendC op，没有意义。
def AscendC_FmodExtendRegBaseOp
    : BinaryOp<"fmod_extend_reg_base", "FmodExtend", [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse FmodExtend reg-based wrapper (fmod_reg_base.h)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src0, AscendC_LocalTensor:$src1,
                       AscendC_LocalTensor:$sharedTmpBuffer, AnyType:$calCount);
  let extraClassDeclaration = extraClassDeclarationBase # ApiExtensionHeader<["fmod_reg_base.h"]>.decl # [{
    SmallVector<Value> getSrcTensors() { return {getSrc0(), getSrc1()}; }
  }];
}
// `NextAfterExtend(dst, src, other, sharedTmpBuffer, calCount)` 同形。它还有一个
// `other` 是标量的重载，legacy 的 `IsNodeValid` 不收标量输入，所以这里也只有 tensor 那份。
def AscendC_NextAfterExtendRegBaseOp
    : BinaryOp<"next_after_extend_reg_base", "NextAfterExtend", [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse NextAfterExtend reg-based wrapper (next_after_reg_base.h)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src0, AscendC_LocalTensor:$src1,
                       AscendC_LocalTensor:$sharedTmpBuffer, AnyType:$calCount);
  let extraClassDeclaration = extraClassDeclarationBase # ApiExtensionHeader<["next_after_reg_base.h"]>.decl # [{
    SmallVector<Value> getSrcTensors() { return {getSrc0(), getSrc1()}; }
  }];
}


// `Pow` 与 `RemainderExtend` 同形——两张表同名、都要一块 tmp buffer，只有扩展头不同，
// 所以 `arguments` 照样进 base。差别只在名字：**apiName 是裸名 `Pow`，不带 `Extend`
// 后缀**，与 mem 侧的 `LogicalAnd` / 选择族的 `ClipByValue` 一样，照抄 legacy 两个
// `PowAscIrCodegenImpl` 的 `GetApiName()`。
//
// **`Pow` 不是 `AscendC::Pow`——`AscendC::` 里根本没有这个名字。** 标准调用叫
// `Power`（`Adv/Math.td` 的 `def PowerOp : BinaryMathOp<"power", "Power">`），两份包装
// 转发给的都是它；包装存在的理由是 dtype 分流与标量形态的展开。所以 mnemonic 上也不用
// `_extend` 那套后缀，直接 `pow` / `pow_reg_base`。
//
// **两侧都要 `utils.h` 前导**，理由与 `SignExtend` 那一对逐字相同：两份 `pow.h` 都用了
// `KernelUtils::BlkSize`，reg 那份还用了 `KernelUtils::BlkAlign` 与
// `IsIntegerWithinUint32Magnitude`，三个符号都只有 mem 的 `utils.h` 声明
// （`utils_reg_base.h` 是 micro-API 的 `CastTrait` 常量表，一个都没有）。
// 少了它 lit 全绿而 kernel 编不过——拦它的是 `load_pow_store`。
class AscendC_PowOpBase<string mnemonic, list<string> headers>
    : BinaryOp<mnemonic, "Pow", [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse Pow wrapper (" # !interleave(headers, ", ") # ")";
  // **两个 src 是 `AnyType` 而不是 `AscendC_LocalTensor`**：两份 `pow.h` 各有四个同名
  // 重载，(t,t) / (t,s) / (s,t) / (s,s)，标量在哪个位置由 C++ 重载决议挑。生成侧打的是
  // 同一个位置上的 SSA 值，tile 与标量没有区别（与 `CompareScalarExtend` 那一对注释里
  // 说的是同一件事），所以标量形态**不新建 def**，放宽这两位就够了。
  //
  // 放宽不会放掉 dtype 校验：那件事在 conversion 的 capability 里
  // （`areElementTypesSupported` 对标量取 `getElementTypeOrSelf`），不靠 ODS 这一层。
  let arguments = (ins AscendC_LocalTensor:$dst, AnyType:$src0, AnyType:$src1,
                       AscendC_LocalTensor:$sharedTmpBuffer, AnyType:$calCount);
  // 两个 src 里只有真是 tile 的那些才算「源张量」，标量不算——与 `SelectL0Op` 那一对的
  // 写法一致（`AnyType:$src1` 的那个 def 的 `getSrcTensors` 不返回 `src1`）。这里两位都
  // 可能是标量，所以只能运行时挑，不能写成静态的那一串。
  let extraClassDeclaration = extraClassDeclarationBase # ApiExtensionHeader<headers, ["utils.h"]>.decl # [{
    SmallVector<Value> getSrcTensors() {
      SmallVector<Value> tensors;
      for (Value src : {getSrc0(), getSrc1()}) {
        if (::mlir::isa<::mlir::ascendc::LocalTensorType>(src.getType())) {
          tensors.push_back(src);
        }
      }
      return tensors;
    }
  }];
}
def AscendC_PowOp : AscendC_PowOpBase<"pow", ["pow.h"]>;
def AscendC_PowRegBaseOp : AscendC_PowOpBase<"pow_reg_base", ["pow_reg_base.h"]>;

// `Sub` / `Div` 是这一批里第一对 **mem 侧不用新增 def** 的：它们的 tensor-tensor 形态
// legacy 生成的就是标准的 `AscendC::Sub` / `AscendC::Div`（`binary_api_call.cpp` 那条
// 不带包装的分支），`SubL2Op` / `DivL2Op` 本来就在上面。矩阵上那两个
// `scalar_sub.h` / `scalar_div.h` 是**标量形态**的头——包装叫 `Subs` / `Divs`，
// 带 `IS_SCALAR_LATTER` 模板实参，与 tensor-tensor 这条路无关。
//
// 只有 reg 侧要现写：那边连 tensor-tensor 都过一层包装（`SubExtend` / `DivExtend`），
// 函数体就是转发给同名的 `AscendC::` 调用，但头是 fork 自己的，得声明出来。两个包装
// 都不收 tmp buffer，所以 `arguments` 与 `FloorDivExtendRegBaseOp` 同形。
class AscendC_BinaryExtendRegBaseOpBase<string mnemonic, string apiName, string header>
    : BinaryOp<mnemonic, apiName, [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse " # apiName # " reg-based wrapper (" # header # ")";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src0, AscendC_LocalTensor:$src1,
                       AnyType:$calCount);
  let extraClassDeclaration = extraClassDeclarationBase # ApiExtensionHeader<[header]>.decl # [{
    SmallVector<Value> getSrcTensors() { return {getSrc0(), getSrc1()}; }
  }];
}
def AscendC_SubExtendRegBaseOp
    : AscendC_BinaryExtendRegBaseOpBase<"sub_extend_reg_base", "SubExtend", "sub_reg_base.h">;
def AscendC_DivExtendRegBaseOp
    : AscendC_BinaryExtendRegBaseOpBase<"div_extend_reg_base", "DivExtend", "div_reg_base.h">;
// 第 7 批的二元无 tmp 两个，与上面两个逐字同形。
def AscendC_CopySignExtendRegBaseOp
    : AscendC_BinaryExtendRegBaseOpBase<"copy_sign_extend_reg_base", "CopySignExtend", "copy_sign_reg_base.h">;
def AscendC_TruncDivExtendRegBaseOp
    : AscendC_BinaryExtendRegBaseOpBase<"trunc_div_extend_reg_base", "TruncDivExtend", "trunc_div_reg_base.h">;

// step 4。`BitwiseOr` / `BitwiseXor` 与上面四个同形（四参、无 tmp）。legacy 的 reg 侧
// （`BitwiseOrAscIrCodegenImplV2` / `BitwiseXorAscIrCodegenImplV2`）生成的不是标准调用
// `AscendC::BitwiseOr` / `BitwiseXor`，是 fork 的包装 `BitwiseOrExtend` /
// `BitwiseXorExtend`（`api_regbase/bitwise_or.h` / `bitwise_xor.h`，legacy 的
// `LoadApiHeaderFiles` 键是 `bitwise_or_reg_base.h` / `bitwise_xor_reg_base.h`——与
// `where_reg_base.h` 对应 `where.h` 同一个命名惯例，键不是文件名）。包装对 i16/ui16/i32/ui8
// 这几位只是转发给标准调用，但生成的调用必须与 legacy 的字面一致，照标准调用的名字挑会
// 生成一个 legacy 从不生成的调用（与 `afir.hypot` 那次同一种坑）。
def AscendC_BitwiseOrExtendRegBaseOp
    : AscendC_BinaryExtendRegBaseOpBase<"bitwise_or_extend_reg_base", "BitwiseOrExtend", "bitwise_or_reg_base.h">;
def AscendC_BitwiseXorExtendRegBaseOp
    : AscendC_BinaryExtendRegBaseOpBase<"bitwise_xor_extend_reg_base", "BitwiseXorExtend", "bitwise_xor_reg_base.h">;

// 第 7 批的二元带 tmp 四个。形状就是 `RemainderExtendRegBaseOp` / `FmodExtendRegBaseOp`
// 那一个——`XxxExtend(dst, src0, src1, tmp, calCount)`——但那两处把 apiName 写死在 base 里
// 或整段展开，表达不了「同一个形状、不同名字、而且头是一列」。`igamma` 偏偏两样都要：
// 六个头、依赖序固定。所以这里补一个与一元那边
// （`AscendC_UnaryTmpExtendRegBaseOpBase`）对称的 base，`headers` / `preludes` 两格的语义
// 也与那边同一套：`headers` 是两两先后，`preludes` 排在所有扩展头之前。
class AscendC_BinaryTmpExtendRegBaseOpBase<string mnemonic, string apiName, list<string> headers,
                                           list<string> preludes = []>
    : BinaryOp<mnemonic, apiName, [ApiExtensionOpInterface]> {
  let description =
      "Call the Autofuse " # apiName # " reg-based wrapper (" # !interleave(headers, ", ") # ")";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src0, AscendC_LocalTensor:$src1,
                       AscendC_LocalTensor:$sharedTmpBuffer, AnyType:$calCount);
  let extraClassDeclaration = extraClassDeclarationBase # ApiExtensionHeader<headers, preludes>.decl # [{
    SmallVector<Value> getSrcTensors() { return {getSrc0(), getSrc1()}; }
  }];
}
def AscendC_Atan2ExtendRegBaseOp
    : AscendC_BinaryTmpExtendRegBaseOpBase<"atan2_extend_reg_base", "Atan2Extend", ["atan2_reg_base.h"]>;
def AscendC_ZetaExtendRegBaseOp
    : AscendC_BinaryTmpExtendRegBaseOpBase<"zeta_extend_reg_base", "ZetaExtend", ["zeta_reg_base.h"]>;
// `igammac` 的四个 helper 住在**子目录**里，注册表的键带着那一段路径
// （`api_regbase/CMakeLists.txt` 生成的就是 `igammac_helper/series_reg_base.h`），所以这里
// 照原样写。四个 helper 被 `igammac` 与 `igamma` 共用，走前导头；`igamma` 的函数体还调
// `igammac_reg_base.h` 里的 `IgammaCImpl*`，所以它的扩展头是两个、igammac 在前。
def AscendC_IgammacExtendRegBaseOp
    : AscendC_BinaryTmpExtendRegBaseOpBase<"igammac_extend_reg_base", "IgammacExtend",
                                           ["igammac_reg_base.h"],
                                           ["igammac_helper/series_reg_base.h",
                                            "igammac_helper/continued_fraction_reg_base.h",
                                            "igammac_helper/asymptotic_series_reg_base.h",
                                            "igammac_helper/series_complement_reg_base.h"]>;
def AscendC_IgammaExtendRegBaseOp
    : AscendC_BinaryTmpExtendRegBaseOpBase<"igamma_extend_reg_base", "IgammaExtend",
                                           ["igammac_reg_base.h", "igamma_reg_base.h"],
                                           ["igammac_helper/series_reg_base.h",
                                            "igammac_helper/continued_fraction_reg_base.h",
                                            "igammac_helper/asymptotic_series_reg_base.h",
                                            "igammac_helper/series_complement_reg_base.h"]>;

// 标量形态的三个 def（`Subs` / `Divs` / `TrueDivExtends`）。**这一类与 tensor-tensor 那条
// 路没有共用的 def**，三件事都变了：
//
//   - **名字多一个 `s`**，而且是**全局**的 `Subs`，不是 `AscendC::Subs`。上面
//     `OpVecBinaryScalar.td` 的 `defm Subs : VecScalarL012Op<"subs", "Subs">` 建的是标准
//     API，走 `AscendC::` 前缀、不收 tmp、没有模板实参——**同名不同函数**，别点那个。
//     它占的助记符是 `subs_l0` / `_l1` / `_l2`，裸 `subs` 是空的，所以这里用得上。
//   - **多一个 bool 模板实参 `IS_SCALAR_LATTER`**：`Subs<T, false>` 表示标量在左。legacy
//     的 `binary_api_call.cpp` 按 `switch_scalar` 生成它。**这是这个文件里第一个用属性承载
//     模板实参的 bool**（比较族的 `CMPMODE` 是枚举，reg 侧也落在模板位，同一个形状）。
//   - **mem 侧要 tmp buffer**：`scalar_sub.h` / `scalar_div.h` 都先把标量 `Duplicate` 成一
//     个 block 再走 `BinaryRepeatParams`，那块中转就是调用方给的 tmp。tensor-tensor 那条
//     反而不用（见上面 `AFIR_SubOp` 的 ODS 注释）。
//
// `utils.h` 前导两个都要：两份头都用了 `MAX_REPEAT_TIME` 与 `KernelUtils::BlkSize` /
// `RptSize`。少了它 lit 全绿、kernel 编不过——与 `floor_div.h` / `sign.h` 同一条。
class AscendC_ScalarBinaryExtendOpBase<string mnemonic, string apiName, list<string> headers>
    : VecScalarOp<mnemonic, apiName, [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse " # apiName # " scalar wrapper (" # !interleave(headers, ", ") # ")";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src, AnyType:$scalar,
                       AscendC_LocalTensor:$sharedTmpBuffer, AnyType:$calCount,
                       DefaultValuedAttr<BoolAttr, "true">:$isScalarLatter);
  let extraClassDeclaration =
      extraClassDeclarationBase # ApiExtensionHeader<headers, ["utils.h"]>.decl # SrcTensorsFromSrc<>.decl;
}
def AscendC_SubsOp : AscendC_ScalarBinaryExtendOpBase<"subs", "Subs", ["scalar_sub.h"]>;
def AscendC_DivsOp : AscendC_ScalarBinaryExtendOpBase<"divs", "Divs", ["scalar_div.h"]>;

// `TrueDivExtends` 与上面两个差一个模板实参：它的 `dst` 与 `src` 可以是两种 dtype
// （i32 输入 → f32 输出），两个都写在模板位上，所以 emitter 那边是另一条 body。
// 两个扩展头与次序的理由见下面 `AscendC_TrueDivExtendOp`——标量这条正是那条注释里说
// 「今天生成不出来」的那个重载，现在能生成了。
def AscendC_TrueDivExtendsOp
    : AscendC_ScalarBinaryExtendOpBase<"true_div_extends", "TrueDivExtends", ["scalar_div.h", "true_div.h"]>;

// reg 侧只有 `Div` 一个走得到标量（`sub_reg_base.h` 里 `SubExtends` 也在，但 legacy v2 的
// `IsScalarInputSupported` 对 Sub 返回 false，调度产不出那种图，接了是死代码——矩阵的
// `reg_scalar` 列就是这个判据的摘要）。它不收 tmp：`div_reg_base.h` 那份自己在寄存器里
// 转存，外部符号一个都没用到，所以也不要前导头。
def AscendC_DivExtendsRegBaseOp
    : VecScalarOp<"div_extends_reg_base", "DivExtends", [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse DivExtends scalar wrapper (div_reg_base.h)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src, AnyType:$scalar,
                       AnyType:$calCount, DefaultValuedAttr<BoolAttr, "true">:$isScalarLatter);
  let extraClassDeclaration =
      extraClassDeclarationBase # ApiExtensionHeader<["div_reg_base.h"]>.decl # SrcTensorsFromSrc<>.decl;
}

// `TrueDivExtend` 只有 mem 侧一个 def：**reg 侧就是上面那个 `DivExtendRegBaseOp`**。
// v2 的 `TrueDivAscIrCodegenImplV2` 写的 apiName 是 `DivExtend`、头是 `div_reg_base.h`，
// 与 `Div` 那一版逐字相同——所以这里不新建 `true_div_extend_reg_base`，注册处直接点
// `AscendC_DivExtendRegBaseOp`。两侧共用一个 def 不是 tier D1（那是同一个**包装**两侧
// 通用，`ApiFamily::Shared`）：这里是两个不同的包装，只是 reg 那一个恰好已经有人建过。
//
// mem 侧那份（`autofuse/ascendc/api/true_div.h`）与 `Div` 真正的差别只有一行：
// `if constexpr (IsSameType<T, int32_t>)` 那一支先把两个 src `Cast` 成 `U` 再 `Div`，
// `else` 分支字面上就是 `AscendC::Div(dst, src1, src2, size)`。**i32 这一行就是这个 op
// 的全部理由**——丢掉它 `afir.truediv` 两侧都成了 `afir.div` 的逐字别名。
//
// 于是 `dst` 与两个 `src` 可以是两种 dtype（`LocalTensor<U>` vs `LocalTensor<T>`），
// 与 `LogicalAnd` 一类同形，别照着上面几个同 dtype 的 def 推。两个模板实参都从实参
// 推得出来，所以生成侧照旧不写模板实参。
//
// **两个扩展头**，而且次序就是依赖次序：`true_div.h` 的标量重载调 `Divs`，那是
// `scalar_div.h` 定义的。字典序恰好也是这个次序（前置 A 的形状在这里第二次兑现）。
//
// `utils.h` 前导：`scalar_div.h` 用了 `MAX_REPEAT_TIME` 与 `KernelUtils::BlkSize` /
// `RptSize`，只有 `utils.h` 声明。`TrueDivExtends` 这个重载这条路今天生成不出来，但头是
// 当作文本内联进来的、模板里那些名字又不依赖模板形参，少了前导当场编不过。
// 与 `floor_div.h` / `logical.h` / `sign.h` 同一条——**lit 一条都测不出来**。
def AscendC_TrueDivExtendOp
    : BinaryOp<"true_div_extend", "TrueDivExtend", [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse TrueDivExtend wrapper (mem-based API extension)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src0, AscendC_LocalTensor:$src1,
                       AscendC_LocalTensor:$sharedTmpBuffer, AnyType:$calCount);
  let extraClassDeclaration =
      extraClassDeclarationBase # ApiExtensionHeader<["scalar_div.h", "true_div.h"], ["utils.h"]>.decl # [{
    SmallVector<Value> getSrcTensors() { return {getSrc0(), getSrc1()}; }
  }];
}

// `LogicalAnd` / `LogicalOr` 是二元一类里第一对**输入与结果不同 dtype** 的 op。两侧的
// `REG_ASC_IR` 都是 `.Input("x1","T1").Input("x2","T1").Output("y","T2")`，而 `T2` 在
// mem 侧只有 `uint8`、reg 侧是 `{uint8, bool}`——四个包装的 `dst` 都写死
// `LocalTensor<uint8_t>`，不跟着 `T` 走。所以 `dst` 与两个 `src` 在这里是两种 dtype，
// 别照着上面几个同 dtype 的 def 推。
//
// 两侧差三样，每一样都落在 def 上：
//
//   - **apiName 不同**：mem 侧生成的是裸名 `LogicalAnd`（`logical.h` 里 fork 自己的包装，
//     名字就叫这个），reg 侧生成的是 `LogicalAndExtend`。于是 mem 那两个 def 的 mnemonic
//     带 `_extend` 说的是「走 ApiExtension 头」，**不是 apiName 的回声**——这是这个文件里
//     第一对两者对不上的 def，`BitwiseAndExtend` / `FloorDivExtend` / `SubExtend` 都是同名。
//   - **扩展头不同**：`logical.h` / `logical_reg_base.h`。
//   - **tmp buffer 要不要**：mem 侧那块是真用的（三条 `*LogicalCommon` 把两个输入 `Cast`
//     成 half 都落在里面，`Int64LogicalCommon` 还要切两段），reg 侧在寄存器里做完，一块
//     都不要。这与 `BitwiseAndExtend` 那种「形参在、但一次没用」不是一回事。
//
// mem 侧要 `utils.h` 前导：`logical.h` 的三条 `*LogicalCommon` 与三条
// `*LogicalCommonScalarExtend` 都用了 `MAX_REPEAT_TIME`，那是 fork 自己的常量，只有
// `utils.h` 声明。与 `floor_div.h` / `sign.h` 同一条——legacy 不写是因为它走
// `IncludeAndDefines` 无条件内联，这条路得自己说，而 lit 测不出来。
class AscendC_LogicalExtendMemOpBase<string mnemonic, string apiName>
    : BinaryOp<mnemonic, apiName, [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse " # apiName # " wrapper (mem-based API extension)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src0, AscendC_LocalTensor:$src1,
                       AscendC_LocalTensor:$sharedTmpBuffer, AnyType:$calCount);
  let extraClassDeclaration =
      extraClassDeclarationBase # ApiExtensionHeader<["logical.h"], ["utils.h"]>.decl # [{
    SmallVector<Value> getSrcTensors() { return {getSrc0(), getSrc1()}; }
  }];
}
def AscendC_LogicalAndExtendOp : AscendC_LogicalExtendMemOpBase<"logical_and_extend", "LogicalAnd">;
def AscendC_LogicalOrExtendOp : AscendC_LogicalExtendMemOpBase<"logical_or_extend", "LogicalOr">;

// reg 侧的 tensor-tensor 与 `SubExtend` / `DivExtend` 同形（四参、无 tmp），所以直接用
// 上面那个 base。
def AscendC_LogicalAndExtendRegBaseOp
    : AscendC_BinaryExtendRegBaseOpBase<"logical_and_extend_reg_base", "LogicalAndExtend",
                                        "logical_reg_base.h">;
def AscendC_LogicalOrExtendRegBaseOp
    : AscendC_BinaryExtendRegBaseOpBase<"logical_or_extend_reg_base", "LogicalOrExtend",
                                        "logical_reg_base.h">;
// 第 6 批。`LogicalXorExtend` 与上面两个逐字同形（四参无 tmp，dst 写死
// `LocalTensor<uint8_t>`，所以结果 dtype 与输入无关），只是头是自己的一份。
def AscendC_LogicalXorExtendRegBaseOp
    : AscendC_BinaryExtendRegBaseOpBase<"logical_xor_extend_reg_base", "LogicalXorExtend",
                                        "logical_xor_reg_base.h">;

// 标量形态。**这一类是第一个带 `ApiExtensionOpInterface` 的 `VecScalarOp`**，所以
// `SrcTensorsFromSrc` 那段注释说的情况到这里才第一次兑现：替掉
// `extraClassDeclaration` 去写扩展头，就得把基类那份 `getSrcTensors` 贴回来。
//
// 名字两侧各按各的规矩派生，**都不是 apiName 加个后缀那么直白**：
//
//   - mem 侧 `binary_api_tmp_call.cpp` 在标量分支里拼的是 `api_name_ + "ScalarExtend"`，
//     于是 `LogicalAndScalarExtend(dst, src, scalar, tmp, size)`——五参，tmp 照旧要。
//     那条分支还**按名字写死只认 `LogicalOr` / `LogicalAnd`**，是这一类专有的。
//   - reg 侧 `binary_api_call.cpp` 走的是通用的 `api_name_ + "s"` 分支（不是 `Sub` /
//     `Div` 那条带模板实参的），于是 `LogicalAndExtends(dst, src, scalar, size)`——四参。
//
// 只有三个 def：reg 侧的 `LogicalOr` 不收标量，理由见 `BinaryElementwise.cpp` 注册处。
class AscendC_LogicalScalarExtendMemOpBase<string mnemonic, string apiName>
    : VecScalarOp<mnemonic, apiName, [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse " # apiName # " wrapper (mem-based API extension)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src, AnyType:$scalar,
                       AscendC_LocalTensor:$sharedTmpBuffer, AnyType:$calCount);
  let extraClassDeclaration =
      extraClassDeclarationBase # ApiExtensionHeader<["logical.h"], ["utils.h"]>.decl # SrcTensorsFromSrc<>.decl;
}
def AscendC_LogicalAndScalarExtendOp
    : AscendC_LogicalScalarExtendMemOpBase<"logical_and_scalar_extend", "LogicalAndScalarExtend">;
def AscendC_LogicalOrScalarExtendOp
    : AscendC_LogicalScalarExtendMemOpBase<"logical_or_scalar_extend", "LogicalOrScalarExtend">;

def AscendC_LogicalAndExtendsRegBaseOp
    : VecScalarOp<"logical_and_extends_reg_base", "LogicalAndExtends", [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse LogicalAndExtends reg-based wrapper (logical_reg_base.h)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src, AnyType:$scalar,
                       AnyType:$calCount);
  let extraClassDeclaration =
      extraClassDeclarationBase # ApiExtensionHeader<["logical_reg_base.h"]>.decl # SrcTensorsFromSrc<>.decl;
}
// `AxpyExtend` 是这个文件里第一个 **tier D1** 的 op：两张支持表共用同一个包装。
// `autofuse/ascendc/api/axpy.h` 一份，`api_regbase/` 下没有 axpy——不是 legacy 漏写了
// 文件，是那边没有一份不同的实现。于是只有这一个 def，没有 `_reg_base` 那一半，
// 由 `ApiFamily::Shared` 在注册处说出「两侧都用它」。
//
// **不叫 `axpy_extend`**：`OpVecBinaryExt.td:39` 已经有一个同名的
// `AscendC_AxpyExtendOp`。那个文件从来不在 `Ops.td` 的 include 图里（见
// `check_ascendc_td_reachable.py` 的豁免表），所以今天不冲突；但它哪天被接进来的话
// 两个 def 会撞名，而那个 def 是**用不了的**——它是 `MathLibraryOp`（撞偏差 6 的
// `isReuseSource`），而且操作数顺序就是错的：写的是 `$alpha, $count, $tmpBuf`，
// 真实签名里 tmp 在 count 前面。`where_contiguous` 当初也是为了同一件事改的名。
//
//   AxpyExtend(dst, src_0, src_1, alpha, tmp_buf, count)    axpy.h
//
// `alpha` 是 `const float`，**不跟 `T` 走**——这与 `LeakyRelu` 的 `const T&` 恰好相反。
// tmp buffer 两个 dtype 分支都收（`CalcAxpyTmpSize`）：f16 那支真用它（切两段做
// `Cast` / `CAST_RINT`），f32 那支一次没用，但形参在签名里，少一个实参就调不上——
// 与 `BitwiseAndExtend` 同一种情况。
//
// 不声明 `utils.h` 前导：`axpy.h` 没用 `MAX_REPEAT_TIME` / `Rational`。里面的裸
// `LocalTensor<float>` / `RoundMode` 由 `AddKernelIncludes` 在有扩展头时生成的
// `using namespace AscendC;` 兜住。
def AscendC_AxpyExtendContiguousOp
    : BinaryOp<"axpy_extend_contiguous", "AxpyExtend", [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse AxpyExtend wrapper on a contiguous tile (shared by both API families)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src0, AscendC_LocalTensor:$src1,
                       AnyType:$alpha, AscendC_LocalTensor:$sharedTmpBuffer, AnyType:$calCount);
  let extraClassDeclaration = extraClassDeclarationBase # ApiExtensionHeader<["axpy.h"]>.decl # [{
    SmallVector<Value> getSrcTensors() { return {getSrc0(), getSrc1()}; }
  }];
}

// 比较族（Eq / Ne / Lt / Le / Gt / Ge）。**六个 AFIR op 共四个 def**：两侧各一个
// tensor-tensor、一个 tensor-scalar，比较模式是 `CMPMODE` 属性而不是六份 def。legacy
// 那边也是一份 emitter 服务六个节点类型（`CompareApiCall` / `CompareV2ApiCall` 把
// `GetApiName()` 直接拼进 `CMPMODE::`），这里沿用同一规则。
//
// **挂在这一类名下的「新调用形态」核完只剩一件。** 文档一直写着「`<T, dim, CMPMODE>`
// + brace-init 数组 + second-to-last stride」，对着两个 `*_api_call.cpp` 核过之后：
//
//   - second-to-last stride 只出现在**循环形态**（`outer_repeats_size != 0`）里——
//     mem 侧那一支来自 `compare_v2.h`，reg 侧是 `dim == 2`。这条路只生成连续形态，两侧
//     都够不着，与 `cast` / `where` / `clipbyvalue` 同一个理由；
//   - brace-init 数组 `Cast` 那一轮已经做掉了；
//   - 真正新的是 **同一个 `CMPMODE` 在 mem 侧是运行期实参、在 reg 侧是模板实参**。
//
// 两侧签名（都取连续形态）：
//
//   mem:  CompareExtend(dst, src0, src1, CMPMODE::EQ, count, tmp)                  compare.h
//         CompareScalarExtend(dst, src, scalar, CMPMODE::EQ, count, tmp)           compare.h
//   reg:  CompareExtend<InT, 1, CMPMODE::EQ>(dst, src0, src1, {count}, {1}, {1})   compare_reg_base.h
//         CompareScalarExtend<InT, 1, CMPMODE::EQ>(dst, src, scalar, {count}, {1}, {1})
//
// **mem 侧的 tmp buffer 在 count 之后**——这个文件里别的包装都是 tmp 在 count 之前
// （`printApiExtensionBinary` 就是按那个次序写死的）。所以这两个 def 的 `arguments`
// 按生成次序写，不照抄 `LogicalAndExtendOp`：次序写反了没有哪一层会报，生成的调用
// 把 count 当成 tmp。reg 侧一块 tmp 都不收（`GetCompareSizeV2` 照样分 32 字节，那是
// `TmpBufferUse::Unused` 那一档的事，见 `BinaryElementwise.cpp`）。
//
// **不叫 `compare_extend`**：`OpVecCompareExt.td:25` 已经有一个 `AscendC_CompareExtendOp`，
// 与 `AxpyExtendContiguousOp` 同一种情况——那个文件不在 `Ops.td` 的 include 图里，今天
// 不冲突，而且它建模的是另一个东西（apiName `"Compare"` 带 `AscFunc`，打出来是
// `AscendC::Compare`）。
//
// mem 侧要 `utils.h` 前导：`compare.h` 用了 `MAX_REPEAT_TIME(S)` 和 `KernelUtils::` 的
// 七个成员，只有 `utils.h` 声明。reg 那份自足（只用 `MicroAPI::` 与 `GetVecLen`）。
// 两个头 `["compare.h", "compare_v2.h"]` 照 legacy 的 `LoadApiHeaderFiles` 写全：连续
// 形态只用得上前一个，但这一类在支持表上的记录就是这两个，字典序恰好也是依赖序。
def AscendC_CompareExtendContiguousOp
    : BinaryOp<"compare_extend_contiguous", "CompareExtend", [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse CompareExtend wrapper on a contiguous tile (compare.h)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src0, AscendC_LocalTensor:$src1,
                       AscendC_CMPMODEAttr:$cmpMode, AnyType:$calCount, AscendC_LocalTensor:$sharedTmpBuffer);
  let extraClassDeclaration =
      extraClassDeclarationBase #
      ApiExtensionHeader<["compare.h", "compare_v2.h"], ["utils.h"]>.decl # [{
    SmallVector<Value> getSrcTensors() { return {getSrc0(), getSrc1()}; }
  }];
}

def AscendC_CompareExtendContiguousRegBaseOp
    : BinaryOp<"compare_extend_contiguous_reg_base", "CompareExtend", [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse CompareExtend wrapper on a contiguous tile (compare_reg_base.h)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src0, AscendC_LocalTensor:$src1,
                       AscendC_CMPMODEAttr:$cmpMode, AnyType:$calCount);
  let extraClassDeclaration =
      extraClassDeclarationBase # ApiExtensionHeader<["compare_reg_base.h"]>.decl # [{
    SmallVector<Value> getSrcTensors() { return {getSrc0(), getSrc1()}; }
  }];
}

def AscendC_CompareScalarExtendContiguousOp
    : VecScalarOp<"compare_scalar_extend_contiguous", "CompareScalarExtend", [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse CompareScalarExtend wrapper on a contiguous tile (compare.h)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src, AnyType:$scalar,
                       AscendC_CMPMODEAttr:$cmpMode, AnyType:$calCount, AscendC_LocalTensor:$sharedTmpBuffer);
  let extraClassDeclaration =
      extraClassDeclarationBase #
      ApiExtensionHeader<["compare.h", "compare_v2.h"], ["utils.h"]>.decl # SrcTensorsFromSrc<>.decl;
}

def AscendC_CompareScalarExtendContiguousRegBaseOp
    : VecScalarOp<"compare_scalar_extend_contiguous_reg_base", "CompareScalarExtend",
                  [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse CompareScalarExtend wrapper on a contiguous tile (compare_reg_base.h)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src, AnyType:$scalar,
                       AscendC_CMPMODEAttr:$cmpMode, AnyType:$calCount);
  let extraClassDeclaration =
      extraClassDeclarationBase # ApiExtensionHeader<["compare_reg_base.h"]>.decl # SrcTensorsFromSrc<>.decl;
}

// 内联广播（brc_inline）：MLIR 侧对等 legacy `BrcInlineTemplate` 的能力。设计见
// docs/development/2026-11-10-afir-broadcast-inline-design.md 功能需求 3。
//
// `BinaryBrcInlineApiWithTwoVectorizedAxisV2<T>(dst, src0, src1, shape0, shape1,
// is_input0_block_brc, is_input1_block_brc, first_axis_v_stride, &AscendC::<Api><T, false>)`
// （`autofuse/ascendc/api/brc_inline_api.h`）。**这个包装本身不计算**，只是按形状把广播
// 折进重复调用——真正做计算的是最后一个函数指针实参指向的标准 AscendC 调用（`Add` /
// `Mul` / ...），由 `wrappedApiName` 这个属性记住是哪一个；`getAPIName()`（走
// `ApiExtensionHeader` 那条既有机制）仍然是包装自己固定的名字，两者不是一回事。
//
// **形参表逐字照抄 C++，不省略恒定项**：五个标量都是独立的操作数，即使某次调用里某个
// 恒为编译期常量（如非广播侧的 `is_input*_block_brc` 恒为 1）——理由与 `BroadcastDim3Op`
// 保留恒为 0 的 `srcZ` 相同，省一个就得在 emitter 里补字面量，把「发的是什么」拆到两个
// 文件去读。
//
// **`<T, false>` 不是沿用 AGENTS 经验 3.3 说的危险缺省，与它的结论不冲突**：
// `BinaryBrcInlineApiWithTwoVectorizedAxisV2` 自己在 counter 模式下设 mask（内部的
// `SetMaskCount` / `SetVectorMask` / `SetMaskNorm`），调用方必须交出 mask 控制权，这正是
// `false` 的正当用途——与 `isSetMask` 缺省该是 `true` 的那条经验说的是另一种情况
// （没有任何代码会替调用方设 mask）。
//
// 只有 mem 侧：`autofuse/v35/ascendc/api_regbase/` 下没有对等的寄存器包装（实测见设计文档
// 「实施前的三项实测结论」第三条），所以这里不建 `_reg_base` 变体。
def AscendC_BinaryBrcInlineOp
    : BinaryOp<"binary_brc_inline", "BinaryBrcInlineApiWithTwoVectorizedAxisV2", [ApiExtensionOpInterface]> {
  let description = "Call the Autofuse BinaryBrcInlineApiWithTwoVectorizedAxisV2 wrapper, which folds a row "
                     "broadcast on one operand into the elementwise call itself (brc_inline_api.h).";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src0, AscendC_LocalTensor:$src1,
                       AnyType:$shape0, AnyType:$shape1,
                       AnyType:$isInput0BlockBrc, AnyType:$isInput1BlockBrc,
                       AnyType:$firstAxisVStride,
                       StrAttr:$wrappedApiName);
  let extraClassDeclaration = extraClassDeclarationBase # ApiExtensionHeader<["brc_inline_api.h"]>.decl # [{
    SmallVector<Value> getSrcTensors() { return {getSrc0(), getSrc1()}; }
  }];
}

#endif // ASC_AUTOFUSE_OP_VEC_BINARY_TD
