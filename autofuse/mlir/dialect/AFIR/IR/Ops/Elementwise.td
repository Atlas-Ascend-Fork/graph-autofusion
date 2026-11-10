// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

//===- Elementwise.td - AFIR elementwise operations --------*- tablegen -*-===//
//
// Unary / binary / compare / ternary math ops. Ported faithfully from
// Ascend-MLIR-gsr. Binary/compare/ternary use the upstream MLIR
// InferTensorTypeAdaptor trait for broadcasted return-type inference; unary
// ops use SameOperandsAndResultType.
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_MATH_OPS
#define AFIR_MATH_OPS

include "AFIRDialect.td"
include "AFIROpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/CommonTypeConstraints.td"

// Shared return-shape compatibility for elementwise operations of any arity.
// Logical tensor form and destination-style buffer form use the same contract:
// the result shape is the broadcast of the operand shapes, and two return types
// are interchangeable when their shapes are compatible.
class AFIR_ElementwiseShapeMethods {
  code elementwiseShapeMethodDecl = [{
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
  }];
  code elementwiseShapeMethodDef = [{
    bool $cppClass::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      if (l.size() != r.size() || l.size() != 1) return false;
      return succeeded(verifyCompatibleShape(l[0], r[0]));
    }
  }];
}

// Bases for elementwise operations that support both logical tensor form and
// destination-style buffer form. One per arity; `resultTypes` defaults to the
// input element types and is spelled out only by the operations whose result
// dtype differs (comparison and logical ops produce `ui8`).
//
// Every elementwise operation belongs on one of these three: they are what
// carries the destination-style interface, the optional tmp buffer, and the
// broadcast-compatible result contract, so an operation added here inherits
// the whole set rather than re-deriving it.
// 刻意不含 `SameOperandsAndResultType`：这一类的结果形状约定是「各输入形状的广播」
// 而不是「与输入逐字相同」，形状那一半由 `verifyLogicalForm` / `verifyBufferForm` 里的
// 广播检查管，dtype 那一半由下面的 `resultRule` 管。代价是无秩与动态形状之间比
// `SameOperandsAndResultType` 松（`tensor<4xf32> -> tensor<?xf32>` 现在合法）。

// destination-style 那一半，每个 bufferizable elementwise op 都要。
defvar AFIR_BufferizableDestinationStyleTraits =
    [AFIRDestinationStyleInterface,
     DeclareOpInterfaceMethods<AFIRDestinationStyleInterface>];

// 结果类型能从操作数推出来的那些，再加上 `InferTypeOpInterface`。
// `afir.cast` 不在内：它的结果元素类型既不是输入的，也不是一个常量，
// 而是调用方写在 op 上的那一个——`inferReturnTypes` 只拿得到操作数，推不出来。
// 声明了却推不准比不声明更坏：`verifyInferredResultTypes` 用的是
// `isCompatibleReturnTypes`，而这一类的那个只比形状，于是一个错的元素类型会被
// 静默放行。见 Cast.td。
defvar AFIR_BufferizableElementwiseTraits =
    AFIR_BufferizableDestinationStyleTraits # [DeclareOpInterfaceMethods<InferTypeOpInterface>];

// 一个 bufferizable elementwise Op 的六个方法，同一 arity 下逐字相同：verify 走共享
// 检查，tie 恒为 0，effects 与 speculatability 只看形态，结果类型要么取 destination
// 要么取输入的广播。在这里按 arity 生成一次，新增一个 op 就只剩三处改动——ODS 一行、
// 一个 AscendC implementation、一行 bridge registry。实现在 Ops/Elementwise.cpp，
// 接缝是 AFIRElementwiseSupport.h（生成的代码落在 AFIRDialect.cpp）。
// `logicalResultElementType` 只对 tensor 形态起作用：buffer 形态的结果类型恒等于
// destination。传空表示「取第一个输入的元素类型」，结果 dtype 与输入不同的族（谓词、
// 比较、cast）在这里说出它到底是什么。
//
// 拆成三段，因为 `afir.cast` 要的不是全套：
//
//   - `...VerifyMethodDef`：共享的形态检查。cast 在它之外还要查一张 (输入, 输出) 对表，
//     所以它自己拼一个 `verify()`，不用这一段。
//   - `...CommonMethodDef`：另外四个方法，每个 op 逐字相同。
//   - `...InferMethodDef`：`inferReturnTypes`。cast 不取——它的结果元素类型推不出来，
//     见上面 `AFIR_BufferizableElementwiseTraits` 那段。
//
// `bufferizableElementwiseMethodDef` 是三段拼起来的，原来的调用处一个字不用改。
class AFIR_BufferizableElementwiseMethods<code inputs, code adaptorInputs,
                                          code logicalResultElementType = [{ ::mlir::Type() }]> {
  code bufferizableElementwiseVerifyMethodDef = [{
    ::mlir::LogicalResult $cppClass::verify() {
      if (isBufferizedForm()) {
        return ::mlir::afir::elementwise::verifyBufferForm(getOperation(), }] # inputs # [{, getDestination());
      }
      return ::mlir::afir::elementwise::verifyLogicalForm(getOperation());
    }
  }];
  code bufferizableElementwiseCommonMethodDef = [{
    ::mlir::FailureOr<::mlir::afir::Role> $cppClass::getRole() {
      return ::mlir::afir::getCanonicalTargetRole(getOperation());
    }

    ::mlir::FailureOr<unsigned> $cppClass::getTiedResultIndex(unsigned destinationIndex) {
      if (destinationIndex != 0 || !getDestination() || getOperation()->getNumResults() != 1) {
        return ::mlir::failure();
      }
      return 0;
    }

    void $cppClass::getEffects(
        ::llvm::SmallVectorImpl<::mlir::afir::elementwise::MemoryEffect> &effects) {
      if (isBufferizedForm()) {
        ::mlir::afir::elementwise::getBufferFormEffects(getOperation(), getDestinationMutable(),
                                                        getTmpBufferMutable(), effects);
      }
    }

    ::mlir::Speculation::Speculatability $cppClass::getSpeculatability() {
      return isBufferizedForm() ? ::mlir::Speculation::NotSpeculatable : ::mlir::Speculation::Speculatable;
    }
  }];
  code bufferizableElementwiseInferMethodDef = [{
    ::mlir::LogicalResult $cppClass::inferReturnTypes(
        ::mlir::MLIRContext *context, ::std::optional<::mlir::Location>, ::mlir::ValueRange operands,
        ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
        ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
      Adaptor adaptor(operands, attributes, properties, regions);
      if (adaptor.getDestination()) {
        inferredReturnTypes.push_back(adaptor.getDestination().getType());
        return ::mlir::success();
      }
      return ::mlir::afir::elementwise::inferLogicalReturnType(
          }] # adaptorInputs # [{, }] # logicalResultElementType # [{, inferredReturnTypes);
    }
  }];
  code bufferizableElementwiseMethodDef = bufferizableElementwiseVerifyMethodDef #
      bufferizableElementwiseCommonMethodDef # bufferizableElementwiseInferMethodDef;
}

// The ODS element-type list is a union: `[I32, F16, F32, BF16]` on both sides admits
// `f16 -> f32`, which no elementwise operation means. `resultRule` puts
// the equality back; an operation whose result dtype genuinely differs from its
// inputs (comparison, cast) passes `resultTypes` and clears this bit.
// 把 `resultRule` 变成 `inferReturnTypes` 里那一句。规则只写在一个地方
// （`AFIRElementwiseSupport.h` 的 `ResultElementRule`），ODS 与 conversion 都读它：
// 原先 ODS 这一处是逐 op 手写的代码块、conversion 那一处是各自的枚举，两份编码分开写
// 就会分开错，而错法是静默的——一边推出 f32、另一边只认「目的地与输入同 dtype」，
// op 建得出来但永远选不出 implementation。
class AFIR_ResultElementTypeExpr<string inputExpr, string rule> {
  code expr = [{ ::mlir::afir::elementwise::inferResultElementType(context, }] # inputExpr #
              [{.getType(), ::mlir::afir::elementwise::ResultElementRule::}] # rule # [{) }];
}

class AFIR_BufferizableUnaryElementwiseOp<string mnemonic,
                                          list<Type> elementTypes,
                                          list<Type> resultTypes = elementTypes,
                                          string resultRule = "Same",
                                          dag extraAttrs = (ins)> :
    AFIR_BufferizableUnaryOpBase<mnemonic,
                                 AFIR_BufferizableElementwiseTraits #
                                     !if(!eq(resultRule, "Same"),
                                         [AllElementTypesMatch<["input", "result"]>], []),
                                 1,
                                 AFIR_BufferizableShapedType<elementTypes>,
                                 AFIR_BufferizableShapedType<resultTypes>,
                                 extraAttrs>,
    AFIR_ElementwiseShapeMethods,
    AFIR_OdsElementTypes<elementTypes>,
    AFIR_BufferizableElementwiseMethods<[{ {getInput()} }], [{ {adaptor.getInput()} }],
                                        AFIR_ResultElementTypeExpr<"adaptor.getInput()", resultRule>.expr> {
  let extraClassDeclaration = baseMethodDecl # elementwiseShapeMethodDecl # odsElementTypesDecl;
  let extraClassDefinition = baseMethods # elementwiseShapeMethodDef # bufferizableElementwiseMethodDef;
}

// `scalarOperand` 说**哪个操作数可以是标量**，取值三种：`"none"`（都不可以）、
// `"rhs"`（只有右边）、`"any"`（哪一边都可以，但不能两边都是）。它不新开一类 op：
// `afir.add %tile, %tile`、`afir.add %tile, %scalar`、`afir.add %scalar, %tile` 是同一个
// op 的三种操作数形态，由两个 implementation 接住（`add.tensor_tensor` /
// `add.tensor_scalar`）——后两种共用一个，因为这一类全是可交换的，见下。
//
// `"any"` 不等于「可交换」，它只说「两边都可以是标量」。 两类 op 都用它，区别在
// lowering 那一侧：
//
//   - 可交换的 8 个（add / mul / maximum / minimum / eq / ne / logical_and /
//     logical_or，矩阵上写 `2nd+swap(2nd)`）：`binary_api_call.cpp` 对它们生成的是不带模板
//     实参的 `Adds(y, x1, (T)scalar, count)`，交换与不交换逐字相同。lowering 只需要把
//     有形状的那个操作数放到第一个实参上。
//   - 不可交换的 3 个（sub / div / truediv）：legacy 把方向编码成一个 bool 模板实参
//     `IS_SCALAR_LATTER`（`Subs<T, false>` = 标量在左），所以 lowering 要把「有没有交换」
//     这件事生成，而不是生成完就忘。
//
// 两类都由 `ScalarOperand` 那一根轴的 `Any` 表达，因为 ODS 这一层管的只是类型约束
// （哪个操作数允许是标量）；「交换之后生成什么」是生成侧的事，`emissionOrder` 算出来的
// `tile` 下标同时服务两类——第一类丢掉它，第二类拿它当模板实参。
//
// lt / le / gt / ge 留 `"rhs"`：legacy 没给它们写
// `IsScalarInputSupportedIfExchangeInputs`，标量只能在右。
//
// 「两边都是标量」挡在 verifier 里，不在这里：ODS 的类型约束是逐个操作数的，
// 说不出「至多一个」。见 `verifyBufferizedElementwise`。
//
// `AllElementTypesMatch` 对标量仍然成立：它比的是 `getElementTypeOrSelf`，标量的
// 「元素类型」就是它自己。所以放宽这一位不会顺手放掉 `f16 + f32`。
// 非 `Same` 规则在二元这里清掉的不是整组绑定，只是 `result` 那一项。 一元的
// 谓词族清掉它就没有绑定可留（只有一个输入），二元不一样：`logical_and` / `logical_or`
// 的结果是 `ui8`、与输入无关，但两个输入之间仍然必须同 dtype——legacy 两侧的
// `REG_ASC_IR` 把它们绑在同一个 `T1` 上。整组一起丢的话 `afir.logical_and %a: f16,
// %b: i32` 就能过 ODS，一路走到 device 编译器才因为模板实参推导冲突报错。
//
// 所以这一位是在两组绑定之间选，不是「绑 / 不绑」：三元族那边的
// `elementTypeBinding` 传的是 trait 列表，理由是两类绑的名字各不相同；二元只有这两种
// 可能，用一位说得清，而且现有调用处一个都不用改。
class AFIR_BufferizableBinaryElementwiseOp<string mnemonic,
                                           list<Type> elementTypes,
                                           list<Type> resultTypes = elementTypes,
                                           string resultRule = "Same",
                                           string scalarOperand = "none",
                                           dag extraAttrs = (ins)> :
    AFIR_BufferizableBinaryOpBase<mnemonic,
                                  AFIR_BufferizableElementwiseTraits #
                                      !if(!eq(resultRule, "Same"),
                                          [AllElementTypesMatch<["lhs", "rhs", "result"]>],
                                          [AllElementTypesMatch<["lhs", "rhs"]>]),
                                  1,
                                  AFIR_BufferizableShapedType<elementTypes>,
                                  AFIR_BufferizableShapedType<resultTypes>,
                                  !if(!eq(scalarOperand, "none"),
                                      AFIR_BufferizableShapedType<elementTypes>,
                                      AnyTypeOf<[AFIR_BufferizableShapedType<elementTypes>] # elementTypes>),
                                  extraAttrs,
                                  !if(!eq(scalarOperand, "any"),
                                      AnyTypeOf<[AFIR_BufferizableShapedType<elementTypes>] # elementTypes>,
                                      AFIR_BufferizableShapedType<elementTypes>)>,
    AFIR_ElementwiseShapeMethods,
    AFIR_OdsElementTypes<elementTypes>,
    AFIR_BufferizableElementwiseMethods<[{ {getLhs(), getRhs()} }],
                                        [{ {adaptor.getLhs(), adaptor.getRhs()} }],
                                        AFIR_ResultElementTypeExpr<"adaptor.getLhs()", resultRule>.expr> {
  let extraClassDeclaration = baseMethodDecl # elementwiseShapeMethodDecl # odsElementTypesDecl;
  let extraClassDefinition = baseMethods # elementwiseShapeMethodDef # bufferizableElementwiseMethodDef;
}

// 三元 elementwise 的共享躯干。两个三元族（选择族与 clip 族）之间只有一处不同：
// 哪些操作数的元素类型被绑在一起。其余（形状契约、目的地风格接口、六个生成方法、
// 两行 `let`）逐字相同，所以在这里写一份，下面两个族各自只剩一个类头。
//
// `elementTypeBinding` 传的是 trait 列表而不是一位开关：绑定关系是「一组名字」，
// 用位表示就得在这里再写一次那组名字，而两类绑的名字本来就不一样。
//
// `scalarInputs` 说**后两个输入可不可以是标量**。三元这一类只需要一位，不像二元那边要三档：
// 两类的第一个操作数都恒是 tile（选择族的 mask、clip 族的 x），后两个各自可以是标量，而且
// 位置不能换——两侧的可用实现都是同名重载（`Where` 的场景 1/2/3、`ClipByValue` 的场景 2/3/4），
// 标量在哪个位置由 C++ 重载决议挑，没有模板实参承载方向。
//
// 为什么第一个操作数不给这一位：两份头里都没有那个重载。`where.h` 的三个标量重载
// `mask` 都是 `LocalTensor<uint8_t>`；`clipbyvalue.h` 只有「三个全是标量」那一个重载收标量
// 的 `src_x`，而那个重载没有 count / tmp 形参，`verifyBufferizedElementwise` 的「至少一个
// 输入有形状」本来就挡掉了全标量。所以放开它只会建出一个编不过的调用。
class AFIR_BufferizableTernaryElementwiseBase<string mnemonic,
                                              list<Trait> elementTypeBinding,
                                              list<Type> conditionTypes,
                                              list<Type> elementTypes,
                                              list<Type> resultTypes,
                                              bit scalarInputs = 0> :
    AFIR_BufferizableTernaryOpBase<mnemonic,
                                   AFIR_BufferizableElementwiseTraits # elementTypeBinding,
                                   1,
                                   AFIR_BufferizableShapedType<conditionTypes>,
                                   !if(scalarInputs,
                                       AnyTypeOf<[AFIR_BufferizableShapedType<elementTypes>] # elementTypes>,
                                       AFIR_BufferizableShapedType<elementTypes>),
                                   AFIR_BufferizableShapedType<resultTypes>>,
    AFIR_ElementwiseShapeMethods,
    AFIR_OdsElementTypes<elementTypes>,
    AFIR_BufferizableElementwiseMethods<[{ {getCondition(), getInput1(), getInput2()} }],
                                        [{ {adaptor.getCondition(), adaptor.getInput1(), adaptor.getInput2()} }]> {
  let extraClassDeclaration = baseMethodDecl # elementwiseShapeMethodDecl # odsElementTypesDecl;
  let extraClassDefinition = baseMethods # elementwiseShapeMethodDef # bufferizableElementwiseMethodDef;
}

// 选择族（select / where）：第一个操作数是 ui8 真值，元素类型集合与后两个
// 不同，所以 `conditionTypes` 单独给。`AllElementTypesMatch` 只把后两个输入与结果绑在
// 一起，第一个不进去——把 ui8 的条件也绑进去会要求整条链都是 ui8。
class AFIR_BufferizableTernaryElementwiseOp<string mnemonic,
                                            list<Type> conditionTypes,
                                            list<Type> elementTypes,
                                            list<Type> resultTypes = elementTypes,
                                            bit sameElementType = 1,
                                            bit scalarInputs = 0> :
    AFIR_BufferizableTernaryElementwiseBase<mnemonic,
                                            !if(sameElementType,
                                                [AllElementTypesMatch<["input1", "input2", "result"]>], []),
                                            conditionTypes, elementTypes, resultTypes, scalarInputs>;

// clip 族（clip_by_value）：三个输入与结果同 dtype，第一个操作数也进绑定。不绑它的话
// `clip_by_value %x: f16, %min: f32, %max: f32` 能过 ODS，一路走到 device 编译器才因为
// 模板实参推导冲突报错（legacy 在 codegen 里用 `GE_ASSERT_TRUE(x1_dtype == x2_dtype)`
// 兜这一条）。
//
// 躯干的操作数名是照选择族起的：clip 的 `condition` / `input1` / `input2` 读作
// x / min / max。改名要连 `arguments` 与 assemblyFormat 一起重写，等于把躯干复制
// 一份，不值当——这里用注释说清。
class AFIR_BufferizableTernaryHomoElementwiseOp<string mnemonic, list<Type> elementTypes,
                                               bit scalarInputs = 0> :
    AFIR_BufferizableTernaryElementwiseBase<mnemonic,
                                            [AllElementTypesMatch<["condition", "input1", "input2", "result"]>],
                                            elementTypes, elementTypes, elementTypes, scalarInputs>;

// 只有 tensor 形态的三个 math 族（`AFIR_UnaryMathOp` / `AFIR_BinaryMathOp` /
// `AFIR_BinaryHomoMathOp`）在这里存在过。最后一批实例是比较族那六行，换到
// `AFIR_BufferizableBinaryElementwiseOp` 之后三个类一个实例都不剩——理由与三元那两个
// base 逐字相同，见 `AFIROpBase.td`。tensor 形态没有消失：bufferizable 那条的
// destination 是 `Optional`，不给就是 tensor 形态。

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

// 两侧的并集：mem 侧 legacy 收 {f16, f32, i32}，reg 侧多收 i16 / ui8。ODS 是并集，
// 「这一侧供不供得出来」由注册处的集合说（`ElementTypeMatrixTest` 盯着它不比 legacy 窄）。
defvar AFIR_AbsElementTypes = [UI8, I16, I32, F16, F32, BF16];

def AFIR_AbsOp : AFIR_BufferizableUnaryElementwiseOp<"abs", AFIR_AbsElementTypes> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise absolute value of a tile";
  let description = [{
    逐元素取绝对值。输入与结果同形同 dtype。

    buffer 形态是 destination-style：结果写进 `outs` 给出的那块存储，同时作为结果值
    产出，消费者用结果值串依赖。

    `tmp` 是可选的片上临时缓冲。要不要它由目标决定而不是由本 Op 决定：提供
    `AscendC::Abs` 的平台直接调用，不需要 tmp buffer；只提供 `AbsExtend` 包装的平台需要
    一块，用于 int32 的取反暂存与原地计算。两种形态在 AFIR 这一层是同一个 Op，差异
    落在 lowering 选哪个 implementation（`abs.tensor` / `abs.tensor.extend`）。

    本 Op 在 `compute` pipe 上执行。
  }];
}
// 两套 API 家族对这几个 op 给的是同一个 `AscendC::` 调用，都不要 tmp buffer（对照表里的
// D0）。所以它们在 AFIR 这一层没有任何平台痕迹，也只有一个 implementation——和
// `afir.abs` 那种要挑 API 的 D2 不同。
def AFIR_ExpOp : AFIR_BufferizableUnaryElementwiseOp<"exp", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise natural exponential of a tile";
  let description = [{
    逐元素取自然指数。输入与结果同形同 dtype。

    buffer 形态是 destination-style：结果写进 `outs` 给出的那块存储，同时作为结果值
    产出，消费者用结果值串依赖。两套 API 家族给的是同一个调用，不需要 tmp buffer，因此
    只有一个 implementation（`exp.tensor`）。

    本 Op 在 `compute` pipe 上执行。
  }];
}
def AFIR_LnOp : AFIR_BufferizableUnaryElementwiseOp<"ln", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise natural logarithm of a tile";
  let description = [{
    逐元素取自然对数。其余与 `afir.exp` 相同：同形同 dtype、destination-style、
    单实现（`ln.tensor`）、`compute` pipe。
  }];
}
def AFIR_SqrtOp : AFIR_BufferizableUnaryElementwiseOp<"sqrt", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise square root of a tile";
  let description = [{
    逐元素取平方根。其余与 `afir.exp` 相同：同形同 dtype、destination-style、
    单实现（`sqrt.tensor`）、`compute` pipe。
  }];
}
// 原先它是 `AFIR_UnaryMathOp`（tensor-only，没有 buffer 形态），与 `gelu` / `sign` /
// `truediv` / `pow` 同一类：那一类的 op 都没有 implementation，写的是什么谁也没对过。
// 换基类的先例是 `4b2f2588`；手写的 `verify()` 一并删掉（见 `Ops/Elementwise.cpp`）。
def AFIR_RsqrtOp : AFIR_BufferizableUnaryElementwiseOp<"rsqrt", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise reciprocal square root of a tile";
  let description = [{
    逐元素取平方根的倒数。输入与结果同形同 dtype，只收 f16 / f32
    （reg 侧的 `REG_ASC_IR` 还收 bf16，AFIR 建模不出来）。

    buffer 形态是 destination-style。两个 implementation，两侧差得比一元里任何一对都远：

    - reg 侧是标准调用 `AscendC::Rsqrt(dst, src, count)`，不引扩展头、不要 tmp buffer；
    - mem 侧走包装 `RsqrtExtend(dst, src, blk_tensor_with_value_1, tmp_buf, size)`
      （`rsqrt.h`），**除 tmp buffer 外还多一块预先填好 1 的 32 字节常量张量**——包装
      算的是 `1 / sqrt(x)`，那个 1 是 `Div` 的左操作数。

    那块常量**不是第二块 tmp buffer**，所以它不在操作数里：tmp buffer 是调度分配的未初始化
    草稿纸，而它是 lowering 在 kernel 序言里 `Duplicate` 出来的一个常量，按 (值, dtype)
    在整个 kernel 里共用一份（legacy 的 `GetApiExtractDupSet` 也是这么记的）。
    见 `Implementations/ConstBlockTensor.h`。本 Op 在 `compute` pipe 上执行。
  }];
}
// `afir.reciprocal` 与 `afir.abs` 同形：一侧有标准调用且不要 tmp buffer，另一侧只有
// `ReciprocalExtend` 包装且要一块（用来 Duplicate 出除数 1）。
def AFIR_ReciprocalOp : AFIR_BufferizableUnaryElementwiseOp<"reciprocal", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise reciprocal of a tile";
  let description = [{
    逐元素取倒数。输入与结果同形同 dtype。

    buffer 形态是 destination-style。`tmp` 是可选的片上临时缓冲，要不要由目标决定：
    提供 `AscendC::Reciprocal` 的平台不需要，只提供 `ReciprocalExtend` 包装的平台需要
    一块存放全 1 张量。差异落在 lowering 选哪个 implementation
    （`reciprocal.tensor` / `reciprocal.tensor.extend`）。

    本 Op 在 `compute` pipe 上执行。
  }];
}
// `afir.erf` / `afir.tanh` 是 **`MathLibraryOp` 一类的头两个**。它们与 `afir.sigmoid`
// 同形（两侧都要 tmp buffer，一侧标准调用、一侧包装），只有一处不同：**标准调用在 mem
// 侧**，reg 侧才是包装（`ErfExtend` / `TanhExtend`）。`abs` / `reciprocal` / 谓词族都是
// 反过来的，别照着它们的方向猜。
//
// 两个原先是 `AFIR_UnaryMathOp`（tensor-only，手写 `verify()`）。换成 buffer 形态的基类
// 时那两个手写 `verify()` 必须一起删——不删是链接期重复定义，编译期一声不吭。
def AFIR_ErfOp : AFIR_BufferizableUnaryElementwiseOp<"erf", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise error function of a tile";
  let description = [{
    逐元素取高斯误差函数。输入与结果同形同 dtype。

    buffer 形态是 destination-style，两套 API 家族都要一块 tmp buffer：mem 侧走标准的
    `AscendC::Erf`，reg 侧走 `ErfExtend` 包装（`erf_reg_base.h`）；对应两个 implementation
    （`erf.tensor` / `erf.tensor.extend_reg_base`）。

    只收 f16 / f32。reg 侧的 `REG_ASC_IR(Erf)` 还收 bf16，AFIR 建模不出来。

    本 Op 在 `compute` pipe 上执行。
  }];
}

def AFIR_TanhOp : AFIR_BufferizableUnaryElementwiseOp<"tanh", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise hyperbolic tangent of a tile";
  let description = [{
    逐元素取双曲正切。其余与 `afir.erf` 相同：同形同 dtype、destination-style、两侧都要
    tmp buffer、mem 侧标准调用而 reg 侧 `TanhExtend` 包装（`tanh_reg_base.h`），对应
    `tanh.tensor` / `tanh.tensor.extend_reg_base`。

    只收 f16 / f32，理由同 `afir.erf`。
  }];
}

// `afir.ndtr` 是**特殊函数一元族的第一个**，也是这条路上第一个 **legacy 只有 reg 侧有**
// 的 op：`REG_ASC_IR(Ndtr)` 只在 `v35/ascir/generator/ascir_builtin_ops_v2.cpp` 里，
// `ascir_builtin_ops_v1.cpp`（mem 侧）那 89 个注册里没有它，对照表
// `ascir_op_lowering_matrix.csv` 记的是 `mem_present=0` / 分档 D3。
//
// 所以它与 `afir.erf` 的差别不是「mem 侧还没接」而是「mem 侧没有这个算子」：
// `BridgeOperationRegistryTest.EveryElementwiseOpCoversBothApiFamilies` 那条「两侧都要
// 有」的断言对它不成立，按那条用例自己写的办法走——加一张显式的「缺哪一侧 + 为什么」
// 表并配反向断言，不放宽那个循环。
def AFIR_NdtrOp : AFIR_BufferizableUnaryElementwiseOp<"ndtr", [F32]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise standard normal CDF of a tile";
  let description = [{
    逐元素取标准正态分布的累积分布函数。输入与结果同形同 dtype。

    buffer 形态是 destination-style。只有 reg 侧一个 implementation
    （`ndtr.tensor.extend_reg_base`）：走 `NdtrExtend` 包装（`ndtr_reg_base.h`），
    签名与 `ErfExtend` 同形，**tmp buffer 那个形参在、但尺寸是 0**
    （legacy 的 `CalcTmpBufSize` 返回 `CalcVoidTmpSizeV2`），与 `TanhExtend` /
    `RemainderExtend` 的 reg 侧是同一种情况——形参在就照收，所以注册时
    `TakesTmpBuffer=true`。

    只收 f32：`REG_ASC_IR(Ndtr)` 的类型域就只有 `DT_FLOAT` 一个，不是 AFIR 建模不出来。

    本 Op 在 `compute` pipe 上执行。
  }];
}

// 同族其余三个（第 2 批）。四个 op 在这一层逐项相同——1 入 1 出、同形同 dtype、只收
// f32、destination-style、tmp 形参在而尺寸为 0、只有 reg 侧一个 implementation——区别只有
// 包装的函数名与扩展头。所以这里不重复 `afir.ndtr` 那段说明，各自只记自己的那一行事实。
def AFIR_NdtriOp : AFIR_BufferizableUnaryElementwiseOp<"ndtri", [F32]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise inverse of the standard normal CDF of a tile";
  let description = [{
    逐元素取标准正态分布累积分布函数的反函数。其余与 `afir.ndtr` 相同：reg-only、
    `NdtriExtend` 包装（`ndtri_reg_base.h`）、implementation `ndtri.tensor.extend_reg_base`。
  }];
}

def AFIR_LogNdtrOp : AFIR_BufferizableUnaryElementwiseOp<"log_ndtr", [F32]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise log of the standard normal CDF of a tile";
  let description = [{
    逐元素取标准正态分布累积分布函数的自然对数。其余与 `afir.ndtr` 相同：reg-only、
    `LogNdtrExtend` 包装（`log_ndtr_reg_base.h`）、
    implementation `log_ndtr.tensor.extend_reg_base`。
  }];
}

def AFIR_I0Op : AFIR_BufferizableUnaryElementwiseOp<"i0", [F32]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise modified Bessel function of the first kind, order zero";
  let description = [{
    逐元素取零阶第一类修正贝塞尔函数。其余与 `afir.ndtr` 相同：reg-only、`I0Extend` 包装
    （`i0_reg_base.h`）、implementation `i0.tensor.extend_reg_base`。
  }];
}

def AFIR_AiryAiOp : AFIR_BufferizableUnaryElementwiseOp<"airy_ai", [F32]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise Airy function Ai of a tile";
  let description = [{
    逐元素取艾里函数 Ai。其余与 `afir.ndtr` 相同，只有一处不同：**它的包装要两个扩展头**
    （`trigonometric_function_utils_reg_base.h` 在前、`airy_ai_reg_base.h` 在后，顺序即内联
    顺序，照抄 legacy 的 `LoadApiHeaderFiles`）。implementation `airy_ai.tensor.extend_reg_base`。
  }];
}

// 第 3 批。与前面几个同形，只是包装要两个前导 utils 头。
def AFIR_BesselJ0Op : AFIR_BufferizableUnaryElementwiseOp<"bessel_j0", [F32]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise Bessel function of the first kind, order zero";
  let description = [{
    逐元素取零阶第一类贝塞尔函数。其余与 `afir.ndtr` 相同：reg-only、`BesselJ0Extend`
    包装、implementation `bessel_j0.tensor.extend_reg_base`。
  }];
}

def AFIR_BesselJ1Op : AFIR_BufferizableUnaryElementwiseOp<"bessel_j1", [F32]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise Bessel function of the first kind, order one";
  let description = [{
    逐元素取一阶第一类贝塞尔函数。其余同 `afir.bessel_j0`,
    implementation `bessel_j1.tensor.extend_reg_base`。
  }];
}

// 第 4 批。与同族前几个同形，差别只在包装要两个扩展头且依赖序与字典序相反，
// 那一层在 `OpVecUnary.td` 的 `AscendC_I0eExtendRegBaseOp` 上说明。
def AFIR_I0eOp : AFIR_BufferizableUnaryElementwiseOp<"i0e", [F32]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise exponentially scaled modified Bessel function of the first kind, order zero";
  let description = [{
    逐元素取指数缩放的零阶第一类修正贝塞尔函数（`i0e(x) = exp(-|x|) * i0(x)`）。其余与
    `afir.i0` 相同：reg-only、`I0eExtend` 包装、implementation `i0e.tensor.extend_reg_base`。
  }];
}

def AFIR_I1eOp : AFIR_BufferizableUnaryElementwiseOp<"i1e", [F32]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise exponentially scaled modified Bessel function of the first kind, order one";
  let description = [{
    逐元素取指数缩放的一阶第一类修正贝塞尔函数。其余同 `afir.i0e`,
    implementation `i1e.tensor.extend_reg_base`。
  }];
}

// 第 5 批。同族另外四个。与 `afir.i0e` 的差别有两处，都不影响这一层的建模：
// legacy 的 `CalcTmpBufSize` 给的是非 0 尺寸（`CalcModifiedBesselI0TmpSizeV2` 固定 512
// 字节），而 AFIR 只带 tmp 的「有没有」、不带尺寸，尺寸落在 ATT 那一侧（与 `afir.axpy`
// 同一种情况）；`K0` / `K1` 的 legacy impl 基类是 `AscIrCodegenV2` 而不是
// `SimtFloatUnaryAscIrCodegenImplV2`，即没有 SIMT 标量那一路——`afir.airy_ai` /
// `afir.sigmoid` / `afir.abs` 都已经是这样。
def AFIR_ModifiedBesselI0Op : AFIR_BufferizableUnaryElementwiseOp<"modified_bessel_i0", [F32]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise modified Bessel function of the first kind, order zero";
  let description = [{
    逐元素取零阶第一类修正贝塞尔函数。其余与 `afir.i0e` 相同：reg-only、
    `ModifiedBesselI0Extend` 包装、implementation
    `modified_bessel_i0.tensor.extend_reg_base`。
  }];
}

def AFIR_ModifiedBesselI1Op : AFIR_BufferizableUnaryElementwiseOp<"modified_bessel_i1", [F32]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise modified Bessel function of the first kind, order one";
  let description = [{
    逐元素取一阶第一类修正贝塞尔函数。其余同 `afir.modified_bessel_i0`,
    implementation `modified_bessel_i1.tensor.extend_reg_base`。
  }];
}

def AFIR_ModifiedBesselK0Op : AFIR_BufferizableUnaryElementwiseOp<"modified_bessel_k0", [F32]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise modified Bessel function of the second kind, order zero";
  let description = [{
    逐元素取零阶第二类修正贝塞尔函数。包装调 `ModifiedBesselI0Extend`，所以它声明**两个**
    扩展头（见 `OpVecUnary.td`）；其余同 `afir.modified_bessel_i0`,
    implementation `modified_bessel_k0.tensor.extend_reg_base`。
  }];
}

def AFIR_ModifiedBesselK1Op : AFIR_BufferizableUnaryElementwiseOp<"modified_bessel_k1", [F32]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise modified Bessel function of the second kind, order one";
  let description = [{
    逐元素取一阶第二类修正贝塞尔函数。其余同 `afir.modified_bessel_k0`,
    implementation `modified_bessel_k1.tensor.extend_reg_base`。
  }];
}

// 第 6 批（一元）。这一批七个在 legacy 的 mem 侧（`ascir_builtin_ops_v1.cpp`）都没注册，
// 全是 reg-only，与 ndtr 那一类同款——`EveryElementwiseOpCoversBothApiFamilies` 那条「两侧
// 都要有」的断言靠 `kRegOnlyElementwiseOps` 那张显式表放行。
//
// dtype 只声明 `kF16 | kF32`（位运算那几个是 `kUI8 | kI16 | kUI16 | kI32`）。legacy 的域里
// 还有 bf16 / i8 / i64 / ui32：i8 / i64 / ui32 不在 AFIR 支持表里，bf16 在表里但留给专门
// 一批连数值验证一起放——这里多声明一位的后果是把那些格子从干净回落变成硬失败。
def AFIR_ErfinvOp : AFIR_BufferizableUnaryElementwiseOp<"erfinv", [F32]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise inverse error function of a tile";
  let description = [{
    逐元素取误差函数的反函数。reg-only，包装的 apiName 是裸名 `Erfinv`（不带 `Extend`),
    走 3 参那个重载、不要 tmp；implementation `erfinv.tensor.extend_reg_base`。
  }];
}

def AFIR_Exp2Op : AFIR_BufferizableUnaryElementwiseOp<"exp2", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise base-2 exponential of a tile";
  let description = [{
    逐元素取 2 的幂。reg-only，包装 `Exp2` 转发给 `Power<T, false, exp2_config>`，要一块
    tmp；implementation `exp2.tensor.extend_reg_base`。
  }];
}

def AFIR_Log2Op : AFIR_BufferizableUnaryElementwiseOp<"log2", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise base-2 logarithm of a tile";
  let description = [{
    逐元素取以 2 为底的对数。**没有包装**，走 CANN 自己的 `AscendC::Log2`
    （`adv_api/math/log.h`），所以没有扩展头；要一块 tmp（legacy 的
    `CalcLog2TmpSizeV2`）。implementation `log2.tensor`。
  }];
}

def AFIR_FloorOp : AFIR_BufferizableUnaryElementwiseOp<"floor", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise floor of a tile";
  let description = [{
    逐元素向下取整。走标准调用 `AscendC::Floor`，tmp 形参在但尺寸是 0
    （`CalcVoidTmpSizeV2`）——与 `afir.ndtr` 同一种情况，形参在就照收。
    implementation `floor.tensor`。
  }];
}

def AFIR_LgammaOp : AFIR_BufferizableUnaryElementwiseOp<"lgamma", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise log-gamma of a tile";
  let description = [{
    逐元素取 gamma 函数的自然对数。走标准调用 `AscendC::Lgamma`，要一块 tmp
    （`CalcLgammaTmpSizeV2`）。implementation `lgamma.tensor`。
  }];
}


// 第 7 批（一元 24 个）。与第 6 批同款，全部只有 reg 侧（legacy 的 mem 侧
// `ascir_builtin_ops_v1.cpp` 都不注册），靠 `kRegOnlyElementwiseOps` 那张显式表放行
// `EveryElementwiseOpCoversBothApiFamilies`。
//
// dtype 一律取「legacy 域 ∩ AFIR 已建模的 7 位」，所以含 bf16：`ElementTypeMatrixTest`
// 要求注册集合不许比 legacy 窄。Bessel / Igamma 那几个 legacy 只给 `DT_FLOAT`，就只写 F32。
//
// **这一批里 19 个在 `Adv/Math.td` 有同名的 `UnaryMathOp`，但只有 `digamma` / `erfc` /
// `trunc` 三个真的走它**：另外那些 legacy 的 `GetApiName` 给的是 `XxxExtend`、
// `LoadApiHeaderFiles` 给了 `xxx_reg_base.h`，走 fork 自己的包装。按名字猜会全答反。

def AFIR_AcosOp : AFIR_BufferizableUnaryElementwiseOp<"acos", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise arc cosine of a tile";
  let description = [{
    逐元素取反余弦。走 `AcosExtend` 包装（`acos_reg_base.h`），要一块 tmp。
    implementation `acos.tensor.extend_reg_base`。
  }];
}

def AFIR_AcoshOp : AFIR_BufferizableUnaryElementwiseOp<"acosh", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise inverse hyperbolic cosine of a tile";
  let description = [{
    逐元素取反双曲余弦。走 `AcoshExtend` 包装（`acosh_reg_base.h`）。
    implementation `acosh.tensor.extend_reg_base`。
  }];
}

def AFIR_AsinOp : AFIR_BufferizableUnaryElementwiseOp<"asin", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise arc sine of a tile";
  let description = [{
    逐元素取反正弦。走 `AsinExtend` 包装（`asin_reg_base.h`）。
    implementation `asin.tensor.extend_reg_base`。
  }];
}

def AFIR_AsinhOp : AFIR_BufferizableUnaryElementwiseOp<"asinh", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise inverse hyperbolic sine of a tile";
  let description = [{
    逐元素取反双曲正弦。走 `AsinhExtend` 包装（`asinh_reg_base.h`）。
    implementation `asinh.tensor.extend_reg_base`。
  }];
}

def AFIR_AtanOp : AFIR_BufferizableUnaryElementwiseOp<"atan", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise arc tangent of a tile";
  let description = [{
    逐元素取反正切。走 `AtanExtend` 包装（`atan_reg_base.h`）。
    implementation `atan.tensor.extend_reg_base`。
  }];
}

def AFIR_AtanhOp : AFIR_BufferizableUnaryElementwiseOp<"atanh", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise inverse hyperbolic tangent of a tile";
  let description = [{
    逐元素取反双曲正切。走 `AtanhExtend` 包装（`atanh_reg_base.h`）。
    implementation `atanh.tensor.extend_reg_base`。
  }];
}

def AFIR_CeilOp : AFIR_BufferizableUnaryElementwiseOp<"ceil", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise ceiling of a tile";
  let description = [{
    逐元素向上取整。**不走标准调用** `AscendC::Ceil`：legacy 的 `GetApiName` 给的是
    `CeilExtend`、头是 `ceil_reg_base.h`，与 `afir.floor` 相反（那一个才是标准调用）。
    implementation `ceil.tensor.extend_reg_base`。
  }];
}

def AFIR_CosOp : AFIR_BufferizableUnaryElementwiseOp<"cos", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise cosine of a tile";
  let description = [{
    逐元素取余弦。走 `CosExtend` 包装（`cos_reg_base.h`）。
    implementation `cos.tensor.extend_reg_base`。
  }];
}

def AFIR_CoshOp : AFIR_BufferizableUnaryElementwiseOp<"cosh", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise hyperbolic cosine of a tile";
  let description = [{
    逐元素取双曲余弦。走 `CoshExtend` 包装（`cosh_reg_base.h`）。
    implementation `cosh.tensor.extend_reg_base`。
  }];
}

def AFIR_ErfcxOp : AFIR_BufferizableUnaryElementwiseOp<"erfcx", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise scaled complementary error function of a tile";
  let description = [{
    逐元素取缩放后的互补误差函数（`exp(x^2) * erfc(x)`）。走 `ErfcxExtend` 包装
    （`erfcx_reg_base.h`）。implementation `erfcx.tensor.extend_reg_base`。
  }];
}

def AFIR_RoundOp : AFIR_BufferizableUnaryElementwiseOp<"round", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise round-half-to-even of a tile";
  let description = [{
    逐元素四舍六入五成双。走 `RoundExtend` 包装（`round_reg_base.h`），与 `afir.ceil`
    同一种情况——同名的标准调用在这一批里用不上。
    implementation `round.tensor.extend_reg_base`。
  }];
}

def AFIR_SinOp : AFIR_BufferizableUnaryElementwiseOp<"sin", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise sine of a tile";
  let description = [{
    逐元素取正弦。走 `SinExtend` 包装（`sin_reg_base.h`）。
    implementation `sin.tensor.extend_reg_base`。
  }];
}

def AFIR_SinhOp : AFIR_BufferizableUnaryElementwiseOp<"sinh", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise hyperbolic sine of a tile";
  let description = [{
    逐元素取双曲正弦。走 `SinhExtend` 包装（`sinh_reg_base.h`）。
    implementation `sinh.tensor.extend_reg_base`。
  }];
}

def AFIR_TanOp : AFIR_BufferizableUnaryElementwiseOp<"tan", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise tangent of a tile";
  let description = [{
    逐元素取正切。走 `TanExtend` 包装（`tan_reg_base.h`）。
    implementation `tan.tensor.extend_reg_base`。
  }];
}

def AFIR_BesselY0Op : AFIR_BufferizableUnaryElementwiseOp<"bessel_y0", [F32]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise Bessel function of the second kind, order zero";
  let description = [{
    逐元素取第二类零阶 Bessel 函数。包装体小参数区要复用 `BesselJ0Extend` 的结果，
    所以 `bessel_j0_reg_base.h` 要先内联——与 `afir.i0e` 同一种依赖。
    implementation `bessel_y0.tensor.extend_reg_base`。
  }];
}

def AFIR_BesselY1Op : AFIR_BufferizableUnaryElementwiseOp<"bessel_y1", [F32]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise Bessel function of the second kind, order one";
  let description = [{
    逐元素取第二类一阶 Bessel 函数。与 `afir.bessel_y0` 同构，复用的是 `BesselJ1Extend`。
    implementation `bessel_y1.tensor.extend_reg_base`。
  }];
}

def AFIR_SphericalBesselJ0Op : AFIR_BufferizableUnaryElementwiseOp<"spherical_bessel_j0", [F32]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise spherical Bessel function of the first kind, order zero";
  let description = [{
    逐元素取第一类零阶球 Bessel 函数（`sin(x) / x`）。名字像 `bessel_j0`，但包装体**不调**
    `BesselJ0Extend`，只用 trig utils 的 `AutofuseSin`，所以扩展头只有自己一个。
    implementation `spherical_bessel_j0.tensor.extend_reg_base`。
  }];
}

def AFIR_ScaledModifiedBesselK0Op : AFIR_BufferizableUnaryElementwiseOp<"scaled_modified_bessel_k0", [F32]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise exponentially scaled modified Bessel function K0";
  let description = [{
    逐元素取指数缩放后的第二类零阶修正 Bessel 函数。包装调 `ModifiedBesselI0Extend`
    并用 `modified_bessel_k0.h` 的常量表，所以 i0 / k0 两个头都要先内联。
    implementation `scaled_modified_bessel_k0.tensor.extend_reg_base`。
  }];
}

def AFIR_ScaledModifiedBesselK1Op : AFIR_BufferizableUnaryElementwiseOp<"scaled_modified_bessel_k1", [F32]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise exponentially scaled modified Bessel function K1";
  let description = [{
    逐元素取指数缩放后的第二类一阶修正 Bessel 函数，与 K0 同构。
    implementation `scaled_modified_bessel_k1.tensor.extend_reg_base`。
  }];
}

def AFIR_DigammaOp : AFIR_BufferizableUnaryElementwiseOp<"digamma", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise digamma function of a tile";
  let description = [{
    逐元素取 digamma（gamma 的对数导数）。**没有包装**，走标准调用 `AscendC::Digamma`
    （`adv_api/math/digamma.h`），要一块 tmp——与 `afir.lgamma` 同形。
    implementation `digamma.tensor`。
  }];
}

def AFIR_ErfcOp : AFIR_BufferizableUnaryElementwiseOp<"erfc", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise complementary error function of a tile";
  let description = [{
    逐元素取互补误差函数。走标准调用 `AscendC::Erfc`，要一块 tmp。注意它与
    `afir.erf` 不同族：`erf` 两侧都有供给，这一个只有 reg 侧。
    implementation `erfc.tensor`。
  }];
}

def AFIR_TruncOp : AFIR_BufferizableUnaryElementwiseOp<"trunc", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise truncation toward zero of a tile";
  let description = [{
    逐元素向零取整。走标准调用 `AscendC::Trunc`，要一块 tmp。
    implementation `trunc.tensor`。
  }];
}

def AFIR_Expm1Op : AFIR_BufferizableUnaryElementwiseOp<"expm1", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise exp(x) - 1 of a tile";
  let description = [{
    逐元素取 `exp(x) - 1`。走 `Expm1Extend` 包装（`expm1_reg_base.h`），**不要 tmp**
    ——与 `afir.erfinv` 同形。implementation `expm1.tensor.extend_reg_base`。
  }];
}

def AFIR_Log1pOp : AFIR_BufferizableUnaryElementwiseOp<"log1p", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise log(1 + x) of a tile";
  let description = [{
    逐元素取 `log(1 + x)`。走 `log1p_reg_base.h` 里的包装，不要 tmp；apiName 是**裸名**
    `Log1p`（不带 `Extend` 后缀，也不是 `AscendC::Log1p`——那个不存在）。
    implementation `log1p.tensor.extend_reg_base`。
  }];
}

// 第 8 批（一元 1 个）。`log10` 是第 6 批那段注释里点名留下的那一个：它走标准调用
// `AscendC::Log10`（`adv_api/math/log.h`）**却不要 tmp**，而当时这条路上「标准调用 +
// 不要 tmp」的组合一个实例都没有——`UnaryElementwiseConversionPattern` 的
// `static_assert` 挡的正是 `MaterializesReuseSource && !TakesTmpBuffer`。
//
// 解法不是放开那个 `static_assert`，而是别让它成为 `MathLibraryOp`：那一类的
// `isReuseSource` 是**必填**操作数，才逼出「要物化一个模板实参」这件事。`Log10` 在
// `Adv/Math.td` 里本来就没有同名的 `UnaryMathOp`（那个文件与上游逐字相同，不动），
// 所以在 Autofuse 子树里按 `UnaryL2Op` 建一个就够——与 `BitwiseAndL2Op` 同款，
// 标准调用、三个实参、不打模板实参。
def AFIR_Log10Op : AFIR_BufferizableUnaryElementwiseOp<"log10", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise base-10 logarithm of a tile";
  let description = [{
    逐元素取以 10 为底的对数。**没有包装**，走 `AscendC::Log10`
    （`adv_api/math/log.h`），所以没有扩展头；与 `afir.log2` 的区别是**不要 tmp**
    ——legacy 的 `Log10AscIrCodegenImplV2` 不重写 `CalcTmpBufSize`，发的是
    `UnaryApiCall` 而不是 `UnaryApiTmpCall`。implementation `log10.tensor`。
  }];
}

// 第 9 批（一元 8 个，多项式族）。这一批引入的是「属性落在模板实参位」这根轴，不是
// 八个新算子——八个的形状逐项相同，legacy 的八个 codegen impl 也只差一个名字：
//
//   ChebyshevPolynomialTExtend<float, 3>(y[off], x[off], tmp_0, size);
//
// `n` 是节点自有的 `ir_attr`（`REG_ASC_IR(...).Attr<int64_t>("n")`），**不是操作数**：
// 不进三段划分、不占缓冲。它与 `afir.leaky_relu` 的 `negative_slop` 是同一类东西，
// 差别在落点——那个落在实参位上，这八个落在模板实参位上。模板位要的是编译期
// 常量，所以这一支不能复用 `UnaryScalarAttrConversionPattern`：建常量在那里是对的，
// 在这里会把一个 `arith.constant` 打到 `<>` 里去。
//
// 元素类型只有 f32，而且是 legacy 自己的域（八个的 `REG_ASC_IR` 都是
// `{{"T", TensorType{DT_FLOAT}}}`），不是我们收窄的结果；包装体里还有一条
// `static_assert(SupportType<T, float>())` 兜着。
class AFIR_PolynomialOp<string mnemonic> :
    AFIR_BufferizableUnaryElementwiseOp<mnemonic, [F32], [F32], /*resultRule=*/"Same",
                                        /*extraAttrs=*/(ins I64Attr:$n)> {
  let afirStability = AFIR_Experimental;
}

def AFIR_ChebyshevPolynomialTOp : AFIR_PolynomialOp<"chebyshev_polynomial_t"> {
  let summary = "Element-wise Chebyshev polynomial of the first kind";
  let description = [{
    逐元素取第一类 Chebyshev 多项式 `T_n(x)`。走 `ChebyshevPolynomialTExtend` 包装
    （`chebyshev_polynomial_t_reg_base.h`，前导 `chebyshev_polynomial_utils_reg_base.h`），
    要一块 tmp；`n` 在模板实参位。
    implementation `chebyshev_polynomial_t.tensor.extend_reg_base`。
  }];
}

def AFIR_ChebyshevPolynomialUOp : AFIR_PolynomialOp<"chebyshev_polynomial_u"> {
  let summary = "Element-wise Chebyshev polynomial of the second kind";
  let description = [{
    逐元素取第二类 Chebyshev 多项式 `U_n(x)`。与 `afir.chebyshev_polynomial_t` 同形，
    只换包装名与头。implementation `chebyshev_polynomial_u.tensor.extend_reg_base`。
  }];
}

def AFIR_ChebyshevPolynomialVOp : AFIR_PolynomialOp<"chebyshev_polynomial_v"> {
  let summary = "Element-wise Chebyshev polynomial of the third kind";
  let description = [{
    逐元素取第三类 Chebyshev 多项式 `V_n(x)`。
    implementation `chebyshev_polynomial_v.tensor.extend_reg_base`。
  }];
}

def AFIR_ChebyshevPolynomialWOp : AFIR_PolynomialOp<"chebyshev_polynomial_w"> {
  let summary = "Element-wise Chebyshev polynomial of the fourth kind";
  let description = [{
    逐元素取第四类 Chebyshev 多项式 `W_n(x)`。
    implementation `chebyshev_polynomial_w.tensor.extend_reg_base`。
  }];
}

def AFIR_ShiftedChebyshevPolynomialTOp : AFIR_PolynomialOp<"shifted_chebyshev_polynomial_t"> {
  let summary = "Element-wise shifted Chebyshev polynomial of the first kind";
  let description = [{
    逐元素取平移后的第一类 Chebyshev 多项式。与未平移的那四个同形，另一套包装与前导头。
    implementation `shifted_chebyshev_polynomial_t.tensor.extend_reg_base`。
  }];
}

def AFIR_ShiftedChebyshevPolynomialUOp : AFIR_PolynomialOp<"shifted_chebyshev_polynomial_u"> {
  let summary = "Element-wise shifted Chebyshev polynomial of the second kind";
  let description = [{
    逐元素取平移后的第二类 Chebyshev 多项式。
    implementation `shifted_chebyshev_polynomial_u.tensor.extend_reg_base`。
  }];
}

def AFIR_ShiftedChebyshevPolynomialVOp : AFIR_PolynomialOp<"shifted_chebyshev_polynomial_v"> {
  let summary = "Element-wise shifted Chebyshev polynomial of the third kind";
  let description = [{
    逐元素取平移后的第三类 Chebyshev 多项式。
    implementation `shifted_chebyshev_polynomial_v.tensor.extend_reg_base`。
  }];
}

def AFIR_ShiftedChebyshevPolynomialWOp : AFIR_PolynomialOp<"shifted_chebyshev_polynomial_w"> {
  let summary = "Element-wise shifted Chebyshev polynomial of the fourth kind";
  let description = [{
    逐元素取平移后的第四类 Chebyshev 多项式。
    implementation `shifted_chebyshev_polynomial_w.tensor.extend_reg_base`。
  }];
}

// step 2a（一元 2 个，取整到 i32）。这一批引入的是 `ResultElementRule::Int32`——结果
// 恒为 signless i32、与输入 dtype 无关，与谓词族那条 `Uint8` 逐项同形。
//
// 两个的输入域都是 legacy 自己的 `{f32, f16, bf16}`、输出都写死 `DT_INT32`
// （`REG_ASC_IR(...)` 的 `{{"T1", ...}, {"T2", TensorType{DT_INT32}}}`）。
//
// 可用实现不一样，而且对照表那一列会骗人：它的 `reg_api_name` 是 legacy 的
// `GetApiName()`，对这一类**不是生成的函数名**——CANN 全树里没有 `AscendC::FloorToInt`。
// 实际生成的是：
//
//   ceil2int      Ceil2IntExtend(dst, src, {count}, {1U}, {1U})        fork 的 cast.h
//   floor_to_int  AscendC::Cast(dst, src, RoundMode::CAST_FLOOR, count) 标准调用
//
// `round_to_int` / `trunc_to_int` 与 `floor_to_int` 同一种可用实现（只换 RoundMode），但它们的
// 合法组合是一张逐对对表，不在这一批，见剩余工作总表里 step 2b 那一节。
def AFIR_Ceil2IntOp : AFIR_BufferizableUnaryElementwiseOp<"ceil2int", [F16, F32, BF16], [I32],
                                                          /*resultRule=*/"Int32"> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise ceiling of a tile, converted to i32";
  let description = [{
    逐元素向上取整并转成 i32。走 `Ceil2IntExtend` 包装（`cast_reg_base.h`），它自己转发给
    `CastExtend<..., RoundMode::CAST_CEIL>`，所以与 `afir.cast` 的 reg 那一半同一个头、同一种
    三数组实参，**不要 tmp**。implementation `ceil2int.tensor.extend_reg_base`。
  }];
}

def AFIR_FloorToIntOp : AFIR_BufferizableUnaryElementwiseOp<"floor_to_int", [F16, F32, BF16], [I32],
                                                            /*resultRule=*/"Int32"> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise floor of a tile, converted to i32";
  let description = [{
    逐元素向下取整并转成 i32。**没有包装**，走标准调用
    `AscendC::Cast(dst, src, AscendC::RoundMode::CAST_FLOOR, count)`——与 `afir.cast` 的区别
    是 RoundMode 不是推出来的而是这个 op 固定的。implementation `floor_to_int.tensor`。
  }];
}

def AFIR_ReluOp : AFIR_BufferizableUnaryElementwiseOp<"relu", [I32, F16, F32]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise rectified linear unit of a tile";
  let description = [{
    逐元素取 `max(x, 0)`。其余与 `afir.exp` 相同：同形同 dtype、destination-style、
    单实现（`relu.tensor`）、`compute` pipe。
  }];
}
// 两套 API 家族都只有 `NegExtend` 包装，签名也一样，只有头文件不同（`neg.h` /
// `neg_reg_base.h`）。所以它既不是 D0（有平台差异）也不像 `afir.abs`（差异在标准
// 与扩展之间）：差异只在扩展头来自哪张支持表。两侧都不要 tmp buffer。
// dtype 表去掉了 I16：`NegExtend` 的 `@tparam` 只声称 int32_t / float / half，两张
// 支持表的包装都是这一份。ODS 收下 I16 而 lowering 接不住，表现是 bridge 建得出来、
// 走到 SelectImplementation 才报「没有可用实现」——比在这里就拒掉晚，也更难看懂。
def AFIR_NegOp : AFIR_BufferizableUnaryElementwiseOp<"neg", [I16, I32, F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise negation of a tile";
  let description = [{
    逐元素取相反数。输入与结果同形同 dtype。

    buffer 形态是 destination-style。两套 API 家族给的都是 `NegExtend` 包装，都不要
    tmp buffer，区别只在扩展头由哪张供给表提供；对应两个 implementation
    （`neg.tensor.extend` / `neg.tensor.extend_reg_base`）。

    本 Op 在 `compute` pipe 上执行。
  }];
}
// `afir.sigmoid` 是把「要不要 tmp buffer」与「用标准还是扩展」拆开的那个例子：两侧都要
// 一块 tmp buffer，但一侧走 `AscendC::Sigmoid`，另一侧走 `SigmoidExtend` 包装。
def AFIR_SigmoidOp : AFIR_BufferizableUnaryElementwiseOp<"sigmoid", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise sigmoid of a tile";
  let description = [{
    逐元素取 `1 / (1 + exp(-x))`。输入与结果同形同 dtype。

    buffer 形态是 destination-style。两套 API 家族都需要一块 tmp buffer，差异只在调用的
    是标准 API 还是 Autofuse 包装；对应两个 implementation
    （`sigmoid.tensor` / `sigmoid.tensor.extend`）。

    本 Op 在 `compute` pipe 上执行。
  }];
}

// `gelu` / `sign` 原先写在二元那一段，是 `AFIR_BinaryMathOp`。 两个都是一元：
// legacy 两侧的 `REG_ASC_IR` 都只有一个 `.Input("x", "T")`，矩阵两侧也都是
// `UnaryApiTmpCall`。写错了没被发现，是因为它们此前一个 implementation 都没有——与
// `floor_div` 的元素类型集合同一类：ODS 上没有实例的那一行，谁也没对过。
def AFIR_GeluOp : AFIR_BufferizableUnaryElementwiseOp<"gelu", [F16, F32]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise GELU of a tile";
  let description = [{
    逐元素取高斯误差线性单元。输入与结果同形同 dtype。

    两张供给表给的是**同一个** `AscendC::Gelu`，也不引扩展头，所以只有一个 implementation
    （`gelu.tensor`），不问任何 feature。但它要一块 tmp buffer——这一点与 exp / ln / relu /
    sqrt 那一档不同，别按「同一个调用」就以为不要。

    只收 f16 / f32，两侧的 `REG_ASC_IR(Gelu)` 都只注册了这两个。
  }];
}

def AFIR_SignOp : AFIR_BufferizableUnaryElementwiseOp<"sign", [F16, F32, I32, UI8, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise sign of a tile";
  let description = [{
    逐元素取符号。输入与结果同形同 dtype。

    两侧都走包装、名字还一样（`SignExtend`）、签名也一样、都要一块 tmp buffer，只有扩展头
    和收的元素类型不同——一元里的 `afir.remainder`。

    **这里的元素类型集合是两侧供给的并集。** mem 侧是 f16 / f32 / i32，reg 侧多一个 ui8
    （它还收 i64 / bf16，AFIR 建模不出来）。分档是 lowering 的事，两个 implementation 各自
    把自己那一半说出来。

    包装按 dtype 分流：f16 / f32 直接转发 `AscendC::Sign`，其余先 `Cast`，所以两个业务头
    （`cast.h` 与 `sign*.h`）都要内联进 kernel——**它是第一个真的声明两个扩展头的 op**。
  }];
}
// `afir.logical_not` 的 buffer 形态。**它与 `afir.cast` 一样不在
// `AFIR_BufferizableUnaryElementwiseOp` 上**，理由也一样：`InferTypeOpInterface` 推不出
// 它的结果元素类型。两侧给的关系不同——mem 侧 `T -> T`，reg 侧 `T -> ui8`——而这条接口
// 只拿得到操作数，说不出「看选中的是哪一侧」。声明了却推不准比不声明更坏，那条判据在
// `Ops/Cast.td` 的基类头上写全了。
//
// 于是这里只取 destination-style 那一半，`verify()` 自己拼：共享的形态检查 +
// `verifyLogicalNotPair`（并集：结果等于输入，或者结果是 `ui8`）。
class AFIR_BufferizableLogicalNotOp<string mnemonic, list<Type> inputTypes, list<Type> resultTypes> :
    AFIR_BufferizableUnaryOpBase<mnemonic,
                                 AFIR_BufferizableDestinationStyleTraits,
                                 1,
                                 AFIR_BufferizableShapedType<inputTypes>,
                                 AFIR_BufferizableShapedType<resultTypes>>,
    AFIR_ElementwiseShapeMethods,
    AFIR_OdsElementTypes<inputTypes>,
    AFIR_BufferizableElementwiseMethods<[{ {getInput()} }], [{ {adaptor.getInput()} }]> {
  let extraClassDeclaration = baseMethodDecl # elementwiseShapeMethodDecl # odsElementTypesDecl;
  let extraClassDefinition = baseMethods # elementwiseShapeMethodDef #
      bufferizableElementwiseCommonMethodDef # [{
    ::mlir::LogicalResult $cppClass::verify() {
      const ::mlir::LogicalResult form = isBufferizedForm()
          ? ::mlir::afir::elementwise::verifyBufferForm(getOperation(), {getInput()}, getDestination())
          : ::mlir::afir::elementwise::verifyLogicalForm(getOperation());
      if (::mlir::failed(form)) {
        return form;
      }
      return ::mlir::afir::elementwise::verifyLogicalNotPair(getOperation(), getInput().getType(),
                                                             getResult().getType());
    }
  }];
}

// 输入列表是两侧的并集里 AFIR 建模得出来的那些：mem 侧 legacy 的 `kLogicalNotTypePairs`
// 是 {f16, f32, ui8, i16, i32}，reg 侧那一版多出 i8 / i64 / bf16 / ui16 / ui32 / ui64 /
// bool，其中 `ui16` 建得出来，其余建不出来。结果列表只多一个 `ui8`，它已经在里面了。
def AFIR_LogicalNotOp : AFIR_BufferizableLogicalNotOp<"logical_not", [UI8, UI16, I16, I32, F16, F32, BF16],
                                                      [UI8, UI16, I16, I32, F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise logical negation of a tile";
  let description = [{
    逐元素逻辑取反：非零变 0，零变 1。

    **这是一元里第一个两侧结果 dtype 关系不同的 op**（二元那边是 `afir.truediv`）：

    - mem 侧走包装 `LogicalNot(dst, src, blk_tensor_with_value_1, tmp_buf, size)`
      （`logical_not.h`，裸名不是 `LogicalNotExtend`），两端同 dtype，要一块 tmp buffer
      **外加一块预先填好 1 的 32 字节常量张量**（包装算的是 `1 - min(1, |x|)`）；
    - reg 侧走包装 `LogicalNotExtend(dst, src, count)`（`logical_not_reg_base.h`），
      结果恒为 `ui8`，既不要 tmp buffer 也不要那块常量。

    那块常量不是第二块 tmp buffer，理由与 `afir.rsqrt` 逐字相同，见那一条。
    ODS 这一层给的是输入与结果各自的并集，逐侧的关系由 lowering 的 `ResultElementType`
    说（`Same` / `Uint8Logical`）。本 Op 在 `compute` pipe 上执行。
  }];
}

// step 4。`bitwise_not`：逐位非。与 `afir.bitwise_and` 共用同一张元素类型集合
// （`AFIR_BitwiseElementTypes`，定义在下面二元那一段），只有 reg 一侧——legacy 的
// `BitwiseNotAscIrCodegenImplV2` 只在 v2 那张表上注册。
def AFIR_BitwiseNotOp : AFIR_BufferizableUnaryElementwiseOp<"bitwise_not", [UI8, I16, UI16, I32]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise bitwise NOT of a tile";
  let description = [{
    逐位取反。输入与结果同形同 dtype。

    走 fork 的包装 `BitwiseNotExtend(dst, src, count)`（`api_regbase/bitwise_not.h`，
    `LoadApiHeaderFiles` 键 `bitwise_not_reg_base.h`），不要 tmp。legacy 另外声明了一条
    走 `MicroAPI::Not` 的 VF 融合路，本 Op 不建模那一条，与其余单点算子同一个范围。
    只有 reg 一侧。implementation `bitwise_not.tensor`。
  }];
}

// 浮点谓词族：输入浮点、结果一个字节的真值——这一类是非 `Same` 规则
// 第一次被清掉的地方。`AllElementTypesMatch` 一开就会把 `f32 -> ui8` 判死，而这正是
// 它们要表达的东西；同形仍然要求，只是不再要求同 dtype。
class AFIR_PredicateElementwiseOp<string mnemonic> :
    AFIR_BufferizableUnaryElementwiseOp<mnemonic, [F16, F32, BF16], [UI8], /*resultRule=*/"Uint8"> {
  let afirStability = AFIR_Experimental;
}

def AFIR_IsnanOp : AFIR_PredicateElementwiseOp<"isnan"> {
  let summary = "Element-wise NaN predicate of a tile";
  let description = [{
    逐元素判断是否为 NaN。输入与结果同形，结果 dtype 是 `ui8`。

    buffer 形态是 destination-style。`tmp` 是可选的片上临时缓冲：提供标准
    `AscendC::IsNan` 的平台不需要，只提供 `IsnanExtend` 包装的平台需要一块。两个
    implementation（`isnan.tensor` / `isnan.tensor.extend`）。

    本 Op 在 `compute` pipe 上执行。
  }];
}
def AFIR_IsFiniteOp : AFIR_PredicateElementwiseOp<"isfinite"> {
  let summary = "Element-wise finiteness predicate of a tile";
  let description = [{
    逐元素判断是否有限。其余与 `afir.isnan` 相同：同形、结果 `ui8`、
    destination-style、两个 implementation（`isfinite.tensor` / `isfinite.tensor.extend`）、
    `compute` pipe。
  }];
}
def AFIR_IsInfOp : AFIR_PredicateElementwiseOp<"isinf"> {
  let summary = "Element-wise infinity predicate of a tile";
  let description = [{
    逐元素判断是否为无穷。其余与 `afir.isnan` 相同：同形、结果 `ui8`、
    destination-style、两个 implementation（`isinf.tensor` / `isinf.tensor.extend`）、
    `compute` pipe。
  }];
}

// `negative_slop` 是这一类里第一个「ODS 属性变成 API 实参」的 op：生成的调用是
// `AscendC::LeakyRelu(dst, src, slope, calCount)`，slope 不是操作数而是属性，conversion
// 里现建一个与 tile 元素类型同型的常量。legacy 也是这么打的（`leaky_relu_api_call.cpp`
// 把 `(dtype)` 强转写进调用文本）。
def AFIR_LeakyReluOp : AFIR_BufferizableUnaryElementwiseOp<"leaky_relu", [F16, F32], [F16, F32],
                                                           /*resultRule=*/"Same",
                                                           /*extraAttrs=*/(ins F32Attr:$negative_slop)>;

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

defvar AFIR_AddElementTypes = [I16, I32, F16, F32, BF16];

def AFIR_AddOp : AFIR_BufferizableBinaryElementwiseOp<"add", AFIR_AddElementTypes, AFIR_AddElementTypes,
                                                      /*resultRule=*/"Same", /*scalarOperand=*/"any"> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise addition of two tiles";
  let description = [{
    逐元素相加。两个操作数与结果同形；广播由独立的 `afir.broadcast` 表达，不折进本 Op
    ——一个 Op 一件事，广播的合法性才有单独的检查点。

    buffer 形态下是 destination-style：结果写进 `outs` 给出的那块存储，同时作为结果值
    产出。与 `afir.load` / `afir.store` 同理，消费者用结果值串依赖。

    本 Op 在 `compute` pipe 上执行。

    **它是 54 个 elementwise Op 里唯一有注册 AscendC 实现的那个**，因此也是本族里唯一
    满足 2.5 分档判据 (b) 的。其余同族 Op 的 ODS 定义已经在，缺的是实现——见 5.4 的
    并行工作线。
  }];
}
def AFIR_SubOp : AFIR_BufferizableBinaryElementwiseOp<"sub", [F16, F32, I16, I32, UI8, UI16, BF16],
                                                      [F16, F32, I16, I32, UI8, UI16, BF16],
                                                      /*resultRule=*/"Same", /*scalarOperand=*/"any"> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise difference of two tiles";
  let description = [{
    逐元素相减。与 `afir.div` 同形，两个一起看：**mem 侧发的是标准的 `AscendC::Sub`、
    却仍然拿着一块 tmp buffer**。

    这块 tmp 不是这条调用要的。legacy 的 `SubAscIrCodegenImpl::CalcTmpBufSize` 对
    tensor-tensor 与标量两种形态都给了一块（`CalcBinaryApiTmpSize`：tensor 一支 32 字节、
    标量一支 `CalcDefaultTmpSize`），而真正用它的只有标量那条——`Subs<T, IS_SCALAR_LATTER>`
    要拿它转存（`scalar_sub.h`）。tensor-tensor 这条发的是不带 tmp 的四参调用
    （`binary_api_call.cpp` 的 else 分支）。调度照样分了这块 UB，AFIR 就得照样收下它，
    否则 bridge 那条「给了 tmp 却落到装不下它的 op 上」直接报错。

    reg 侧没有这回事：`SubExtend` 是 fork 自己的包装（`sub_reg_base.h`），转发给同名的
    `AscendC::Sub`，不要 tmp。

    元素类型两侧不同：mem 侧的 `REG_ASC_IR(Sub)` 就是这四个，reg 侧那一版还收
    bf16 / i8 / i64 / ui8 / ui16 / ui32 / ui64。这里只写两侧的交集加 AFIR 建模得出来的
    部分——与 `afir.add` 同一个口径，那边的 reg 侧也宽，ODS 也没跟着放。
  }];
}
// Mul / Maximum / Minimum 的 tensor-tensor 形态在两套家族上是同一个 `AscendC::` 调用，
// 都不要 tmp buffer。对照表把它们列成 D1 是因为标量变体两侧共用一个扩展头
// （`scalar_mul.h` / `scalar_maximum.h` / `scalar_minimum.h`），那条路在 E3。
def AFIR_MulOp : AFIR_BufferizableBinaryElementwiseOp<"mul", [F16, F32, I16, I32, BF16], [F16, F32, I16, I32, BF16],
                                                      /*resultRule=*/"Same", /*scalarOperand=*/"any"> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise product of two tiles";
  let description = [{
    逐元素相乘。形状按广播兼容，dtype 两输入与结果一致。其余与 `afir.add` 相同：
    destination-style、单实现（`mul.tensor_tensor`）、`compute` pipe。
  }];
}
def AFIR_DivOp : AFIR_BufferizableBinaryElementwiseOp<"div", [F16, F32], [F16, F32],
                                                      /*resultRule=*/"Same", /*scalarOperand=*/"any"> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise quotient of two tiles";
  let description = [{
    逐元素相除。与 `afir.sub` 逐字同形——legacy 那两个 impl 类连 `IsScalarInputSupported`
    和 `IsInplaceSupported` 都一样，只有头名 `scalar_div.h` / `scalar_sub.h` 不同。
    mem 侧带着一块用不上的 tmp buffer 这件事见 `afir.sub` 的说明。

    元素类型两侧都只有 f16 / f32，是这一批里少见的两侧相等。
  }];
}
def AFIR_MinimumOp : AFIR_BufferizableBinaryElementwiseOp<"minimum", [I16, I32, F16, F32, BF16], [I16, I32, F16, F32, BF16],
                                                          /*resultRule=*/"Same", /*scalarOperand=*/"any"> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise minimum of two tiles";
  let description = [{
    逐元素取较小值。其余与 `afir.mul` 相同；单实现 `minimum.tensor_tensor`。
  }];
}
def AFIR_MaximumOp : AFIR_BufferizableBinaryElementwiseOp<"maximum", [I16, I32, F16, F32, BF16], [I16, I32, F16, F32, BF16],
                                                          /*resultRule=*/"Same", /*scalarOperand=*/"any"> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise maximum of two tiles";
  let description = [{
    逐元素取较大值。其余与 `afir.mul` 相同；单实现 `maximum.tensor_tensor`。
  }];
}
// 真除。二元一类里第一个结果 dtype 是「一张 (输入, 输出) 对表」的 op——逻辑族那种
// 「结果恒为 ui8」是它的特例，这里是逐行的：mem 侧 `kTrueDivTypePairs` 写的是
// `{f16->f16, f32->f32, i32->f32}`，两个浮点跟着输入、i32 的商是 f32。
//
// 于是规则不能是 `Same`（i32 输入配 f32 结果过不了 `AllElementTypesMatch`），
// 而 `elementTypes` / `resultTypes` 是并集写法，与逻辑族采用同一规则：这一层只说
// 「输入收哪些、结果可以是哪些」，逐行那张表由 lowering 的 `ResultElementType` 说。
// 两个输入之间仍然绑死同 dtype——legacy 两侧都把它们绑在同一个 `T1` 上。
//
// `logicalResultElementType` 这里不能传空：传空是「取第一个输入的元素类型」，i32 输入
// 会推出 i32 结果，而 `resultTypes` 里没有 i32。
//
// 原先它是 `AFIR_BinaryMathOp`（tensor-only，没有 buffer 形态），与 `gelu` / `sign`
// 同一类：那一类的 op 都没有 implementation，写的是什么谁也没对过。换基类的先例是
// `4b2f2588`。注意助记符是 `truediv`，没有下划线，与 `floor_div` 不同。
def AFIR_TrueDivOp : AFIR_BufferizableBinaryElementwiseOp<"truediv", [F16, F32, I32, BF16], [F16, F32, BF16],
                                                          /*resultRule=*/"SameOrF32ForI32",
                                                          /*scalarOperand=*/"any"> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise true division of two tiles";
  let description = [{
    逐元素真除：整数输入也按浮点算，商不截断。这就是它与 `afir.div` 的**唯一**差别——
    f16 / f32 两行两侧发的都是同一个 `AscendC::Div`，只有 i32 那一行 mem 侧的包装
    `TrueDivExtend` 先把两个输入 `Cast` 成 f32 再除。

    元素类型不是一个集合而是一张对表，mem 侧 `{f16->f16, f32->f32, i32->f32}`。
    ODS 这一层给的是输入与结果各自的并集，逐行的对应关系在 lowering 那边
    （`BinaryElementwise.cpp` 的 `ResultElementType::SameOrF32ForI32`）。

    buffer 形态是 destination-style，两个 implementation：mem 侧走包装
    `TrueDivExtend(dst, x1, x2, tmp, size)`（`true_div.h` + `scalar_div.h`，要 tmp
    buffer——i32 那支拿它装两份 cast 结果，f32 那支一次没用但形参在签名里）；
    **reg 侧发的是 `DivExtend`（`div_reg_base.h`），与 `afir.div` 的 reg 侧逐字相同**，
    因为 v2 那边 TrueDiv 与 Div 的 impl 类本来就一样。所以 reg 侧不收 i32，只有
    f16 / f32——bf16 那一行 AFIR 不建模。

    没有标量形态：legacy mem 侧收任意位置的标量（对照表那一格是 `any`），但那要 L-4
    的 swap 方向，这一轮只接 tensor-tensor。本 Op 在 `compute` pipe 上执行。
  }];
}
// 原先它是 `AFIR_BinaryMathOp`（tensor-only，没有 buffer 形态），与 `truediv` /
// `gelu` / `sign` 同一类。换基类的先例是 `4b2f2588`；手写的 `verify()` 与
// `AFIR_INFER_BROADCAST(PowOp)` 两处一起删（见 `Ops/Elementwise.cpp`）。
def AFIR_PowOp : AFIR_BufferizableBinaryElementwiseOp<"pow", [F16, F32, I16, I32, UI8, UI16, BF16],
                                                      [F16, F32, I16, I32, UI8, UI16, BF16],
                                                      /*resultRule=*/"Same", /*scalarOperand=*/"any"> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise power of two tiles";
  let description = [{
    逐元素乘幂：`y = x1 ^ x2`。

    元素类型是两侧的并集，而这一对**差得比前面任何一对都远**：mem 侧 legacy 的
    `REG_ASC_IR(Pow)` 只给了 {i32, f16, f32}，reg 侧那一版是 {i8, ui8, ui16, ui32,
    i16, i32, f16, f32, bf16}。i8 / ui32 / bf16 AFIR 建模不出来，剩下的都收进来了，
    于是这里比 `bitwise_and` 那一族还宽一档（多了 f16 / f32）。分档是 lowering 的事。

    两侧都走包装、**都叫裸名 `Pow`**（不是 `PowExtend`）、都要一块 tmp buffer，只有
    扩展头不同（`pow.h` vs `pow_reg_base.h`）——与 `afir.remainder` 同形。注意
    `AscendC::Power` 才是标准调用，两份包装转发给它的都是它；`Pow` 这个名字在
    `AscendC::` 里没有对应物。

    没有标量形态：legacy **两侧**都收标量（mem 侧只拒全标量，reg 侧那一格是 `any`），
    但那要 L-4 的 swap 方向，这一轮只接 tensor-tensor。本 Op 在 `compute` pipe 上执行。
  }];
}
defvar AFIR_BitwiseElementTypes = [UI8, I16, UI16, I32];

def AFIR_BitwiseAndOp : AFIR_BufferizableBinaryElementwiseOp<"bitwise_and", AFIR_BitwiseElementTypes,
                                                             AFIR_BitwiseElementTypes> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise bitwise AND of two tiles";
  let description = [{
    逐位与。元素类型集合与算术那几个不同——这里是**位模式**，浮点不在内；反过来
    `ui8` / `ui16` 在内，而它们不属于算术族。

    `scalarOperand` 是 `"none"`：legacy 的 `IsNodeValid` 明写不支持标量输入（v1 / v2 两侧
    都是），所以右操作数只有 tile 一种形态。

    两侧的供给不是同一个函数：mem 侧走包装 `BitwiseAndExtend`（`bitwise_and.h`，带一块
    调用方给的 tmp buffer——**包装其实没用它**，但形参在那里，见下），reg 侧是标准的
    `AscendC::BitwiseAnd`，不带 tmp。所以是两个 implementation，不是一个。
  }];
}

// step 4。`bitwise_or` / `bitwise_xor` 与 `bitwise_and` 同一张元素类型集合，但只有
// reg 一侧（legacy 的 mem 侧 `ascir_builtin_ops_v1.cpp` 没注册这两个），而且 reg 侧生成的
// 不是标准调用——legacy 的 `BitwiseOrAscIrCodegenImplV2` / `BitwiseXorAscIrCodegenImplV2`
// 走的是 fork 的包装 `BitwiseOrExtend` / `BitwiseXorExtend`
// （`api_regbase/bitwise_or.h` / `bitwise_xor.h`），不带 tmp。与 `bitwise_not` 同一种情况。
def AFIR_BitwiseOrOp : AFIR_BufferizableBinaryElementwiseOp<"bitwise_or", AFIR_BitwiseElementTypes,
                                                            AFIR_BitwiseElementTypes> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise bitwise OR of two tiles";
  let description = [{
    逐位或。两个输入同形同 dtype，结果同形同 dtype。

    走 fork 的包装 `BitwiseOrExtend(dst, s0, s1, count)`（`api_regbase/bitwise_or.h`，
    `LoadApiHeaderFiles` 键 `bitwise_or_reg_base.h`），不要 tmp。`scalarOperand` 是
    `"none"`：legacy 的 `IsNodeValid` 明写不支持标量输入。只有 reg 一侧。
    implementation `bitwise_or.tensor`。
  }];
}

def AFIR_BitwiseXorOp : AFIR_BufferizableBinaryElementwiseOp<"bitwise_xor", AFIR_BitwiseElementTypes,
                                                             AFIR_BitwiseElementTypes> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise bitwise XOR of two tiles";
  let description = [{
    逐位异或。两个输入同形同 dtype，结果同形同 dtype。

    走 fork 的包装 `BitwiseXorExtend(dst, s0, s1, count)`（`api_regbase/bitwise_xor.h`，
    `LoadApiHeaderFiles` 键 `bitwise_xor_reg_base.h`），不要 tmp。`scalarOperand` 是
    `"none"`：legacy 的 `IsNodeValid` 明写不支持标量输入。只有 reg 一侧。
    implementation `bitwise_xor.tensor`。
  }];
}
// 没有 BF16，而且与 `afir.sub` 在这一位上不一样：`ascir_op_lowering_matrix.csv` 里
// `Sub` 的 `reg_dtypes` 有 bf16，`FloorDiv` 的没有。两者 reg 侧共用
// `kArithmeticRegElementTypes` 那个常量，2026-10-20 往那个常量上加 `| kBF16` 时把这里一起
// 放宽了；常量已改回不带 bf16，bf16 写在 `sub` 自己的注册处。
def AFIR_FloorDivOp : AFIR_BufferizableBinaryElementwiseOp<"floor_div", [F16, F32, I16, I32, UI8, UI16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise floor division of two tiles";
  let description = [{
    逐元素相除后向下取整。

    **这里的元素类型集合是两侧供给的并集，不是任何一侧自己的。** mem 侧只收 f16 / f32
    （legacy v1 的 `REG_ASC_IR(FloorDiv)` 就只注册了这两个），reg 侧那一版在 f16 / f32
    之外还认整数。分档是 lowering 的事：ODS 这一层给并集，两个 implementation 各自把
    自己那一半说出来（见 `BinaryElementwise.cpp` 的 `ElementDomain`）。

    改造前这里写的是 `[I16, I32]`——与 mem 侧**完全不相交**、又是 reg 侧的真子集，
    两边都对不上。它没被发现是因为这个 op 当时还没有任何 implementation。

    两侧都走包装、名字还一样（`FloorDivExtend`），只有扩展头和收不收 tmp 不同——与
    `afir.neg` 同形，只是抬到了二元。没有标量形态：legacy 两侧的 `IsNodeValid` 都明写
    拒绝标量输入。
  }];
}
// 第 6 批（二元）。七个里 `logical_xor` 走 `AFIR_LogicalBinaryOp`（结果恒 `ui8`，与
// `logical_and` / `logical_or` 同一张表），位运算两个走 `AFIR_BitwiseElementTypes`，
// 其余四个是浮点族。全部 reg-only。
def AFIR_FmodOp : AFIR_BufferizableBinaryElementwiseOp<"fmod", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise floating-point remainder of two tiles";
  let description = [{
    逐元素取浮点取模。走 `FmodExtend` 包装（`fmod_reg_base.h`），形状与
    `afir.remainder` 的 reg 那一半逐字相同。implementation
    `fmod.tensor.extend_reg_base`。
  }];
}

def AFIR_ModOp : AFIR_BufferizableBinaryElementwiseOp<"mod", [F16, F32, BF16, I16, UI8]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise modulo of two tiles";
  let description = [{
    逐元素取模。**与 `afir.fmod` 落到同一个 AscendC op**——legacy 两个调度节点的
    `GetApiName` 都是 `FmodExtend`、头都是 `fmod_reg_base.h`，与 `afir.gelu`
    （两张供给表同一个调用）同一种情况。implementation `mod.tensor.extend_reg_base`。
  }];
}

def AFIR_NextAfterOp : AFIR_BufferizableBinaryElementwiseOp<"next_after", [F32]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise next representable value after a tile toward another";
  let description = [{
    逐元素取从 `x` 朝 `y` 方向的下一个可表示浮点数。走 `NextAfterExtend` 包装
    （`next_after_reg_base.h`）。包装另有一个 `other` 为标量的重载，legacy 的
    `IsNodeValid` 不收标量输入，所以这里也只有 tensor-tensor。只收 f32：
    `REG_ASC_IR(NextAfter)` 的类型域只有 `DT_FLOAT`。
    implementation `next_after.tensor.extend_reg_base`。
  }];
}


// 第 7 批（二元 6 个）。全部 reg-only、两输入同 dtype。
//
// 混合 dtype 那一类（legacy 的 `HermitePolynomialH` / `He` / `PolyGamma`：`x` 是 `T`、
// `n` 是 `U`）**不在这一批**：`BinaryElementwiseCapabilityProvider` 的
// `areElementTypesSupported` 开头就要求两个输入元素类型逐字相同，接它们要先补那一支。

def AFIR_CopySignOp : AFIR_BufferizableBinaryElementwiseOp<"copy_sign", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise copy of the sign of one tile onto another";
  let description = [{
    逐元素把 `x2` 的符号搬到 `x1` 的幅值上。走 `CopySignExtend` 包装
    （`copy_sign_reg_base.h`），**不要 tmp**——与 `afir.sub` 的 reg 那一半同形。
    implementation `copy_sign.tensor.extend_reg_base`。
  }];
}

def AFIR_TruncDivOp : AFIR_BufferizableBinaryElementwiseOp<"trunc_div", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise division truncated toward zero";
  let description = [{
    逐元素相除后向零取整。走 `TruncDivExtend` 包装（`trunc_div_reg_base.h`），不要 tmp。
    与 `afir.floor_div` 的区别只是取整方向。
    implementation `trunc_div.tensor.extend_reg_base`。
  }];
}

def AFIR_Atan2Op : AFIR_BufferizableBinaryElementwiseOp<"atan2", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise two-argument arc tangent";
  let description = [{
    逐元素取 `atan2(y, x)`。走 `Atan2Extend` 包装（`atan2_reg_base.h`），要一块 tmp。
    implementation `atan2.tensor.extend_reg_base`。
  }];
}

def AFIR_ZetaOp : AFIR_BufferizableBinaryElementwiseOp<"zeta", [F32]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise Hurwitz zeta function";
  let description = [{
    逐元素取 Hurwitz zeta。走 `ZetaExtend` 包装（`zeta_reg_base.h`），要一块 tmp。
    implementation `zeta.tensor.extend_reg_base`。
  }];
}

def AFIR_IgammaOp : AFIR_BufferizableBinaryElementwiseOp<"igamma", [F32]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise regularized lower incomplete gamma function";
  let description = [{
    逐元素取规范化下不完全 gamma。包装体调 `igammac_reg_base.h` 里的 `IgammaCImpl*`，
    所以那个头要先内联；四个 `igammac_helper/` 下的 helper 走前导头。
    implementation `igamma.tensor.extend_reg_base`。
  }];
}

def AFIR_IgammacOp : AFIR_BufferizableBinaryElementwiseOp<"igammac", [F32]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise regularized upper incomplete gamma function";
  let description = [{
    逐元素取规范化上不完全 gamma，即 `1 - igamma`。与 `afir.igamma` 共用那四个 helper。
    implementation `igammac.tensor.extend_reg_base`。
  }];
}

// 第 8 批（二元 1 个）。`hypot` 与第 7 批的 `zeta` / `igamma` 同形——reg-only、两输入
// 同 dtype、要一块 tmp——只有一处不同：它走标准调用。legacy 的
// `HypotAscIrCodegenImplV2` 不重写 `LoadApiHeaderFiles`，生成的是 `AscendC::Hypot`
// （`adv_api/math/hypot.h`），所以没有扩展头、要进 `kIsStockCall`。
//
// fork 里确实有一份 `v35/ascendc/api_regbase/hypot.h`（里面是 `HypotExtend`），但
// legacy 一处都不加载它——照着文件名挑就会选错可用实现，生成一个 legacy 从不生成的调用。
// 这与第 7 批那 19 个「`Adv/Math.td` 有同名 `UnaryMathOp` 但不是可用实现」是反向的同一种坑。
def AFIR_HypotOp : AFIR_BufferizableBinaryElementwiseOp<"hypot", [F16, F32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise Euclidean norm of two tiles";
  let description = [{
    逐元素取 `sqrt(x1^2 + x2^2)`。走标准调用 `AscendC::Hypot(dst, s0, s1, tmp, size)`，
    要一块 tmp（legacy 的 `CalcVoidTmpSizeV2`，尺寸是 0 但形参在就照收，与
    `afir.remainder` 的 reg 那一半同一种情况）。implementation `hypot.tensor`。
  }];
}

def AFIR_RemainderOp : AFIR_BufferizableBinaryElementwiseOp<"remainder", [F16, F32, I32, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise remainder of two tiles";
  let description = [{
    逐元素取余。**这是本切片里第一个 ODS 上原本就没有、现建的 op**——它在方案的
    「ODS 还缺 3 个」名单上。

    元素类型仍然是两侧的并集：mem 侧 legacy 只注册了 f32 / i32，reg 侧那一版是
    f16 / f32 / bf16 / i32。**bf16 不在这里**，因为 AFIR 今天整个不建模 bf16——
    把它加进来就不只是这一个 op 的事了，得先有类型、有搬运、有测试。

    两侧都走包装、都叫 `RemainderExtend`、都要一块 tmp buffer，只有扩展头不同——比
    `afir.floor_div` 还整齐（那一对还差 tmp）。reg 侧那块 tmp 的尺寸是 0
    （legacy 的 `CalcVoidTmpSizeV2`）：它不是拿来算的，是拿来挑重载的——`RemainderExtend`
    在 reg 侧有 4 参和 5 参两个重载，legacy 发的是 5 参那个。

    没有标量形态：mem 侧的 `IsNodeValid` 明写拒绝标量输入。
  }];
}
// 逻辑与 / 逻辑或。二元一类里第一对结果 dtype 与输入无关的 op——一元那边的谓词族
// （isnan / isfinite / isinf）已经是这个形状，二元这里到这一对才第一次出现，所以
// 「非 `Same` 规则」在二元的语义也是到这里才定下来（见类定义处）。
//
// 结果恒为 `ui8`：两侧的 `REG_ASC_IR` 都写 `.Output("y", "T2")` 而 `T2` 只有 `uint8`
// （reg 侧还收 `bool`，AFIR 不建模）。四个包装的 `dst` 也都写死 `LocalTensor<uint8_t>`。
//
// 输入集合是两侧的交集规则：mem 侧 `REG_ASC_IR` 给的就是这五个，reg 侧更宽
// （bf16 / i8 / i64 / ui32 / ui16 / ui64 / bool），AFIR 建模不出来。分档是 lowering 的事，
// 各 implementation 各说各的一半。
//
// 有标量形态，哪一边都可以（矩阵 mem 侧两个都是 `2nd+swap(2nd)`，两个 op 都可
// 交换）。但不是四个 implementation 都有：reg 侧的 `logical_or` 一个标量都不收。
// 理由与名字都在 `BinaryElementwise.cpp` 注册处——ODS 写的是两侧的并集。
class AFIR_LogicalBinaryOp<string mnemonic> :
    AFIR_BufferizableBinaryElementwiseOp<mnemonic, [UI8, UI16, I16, I32, F16, F32, BF16], [UI8],
                                         /*resultRule=*/"Uint8", /*scalarOperand=*/"any"> {
  let afirStability = AFIR_Experimental;
}

def AFIR_LogicalOrOp : AFIR_LogicalBinaryOp<"logical_or"> {
  let summary = "Element-wise logical OR of two tiles";
  let description = [{
    逐元素逻辑或。两个输入同形同 dtype，结果同形、dtype 恒为 `ui8`。

    buffer 形态是 destination-style。mem 侧走包装 `LogicalOr(dst, s0, s1, tmp, size)`
    （`logical.h`）并且**真的用那块 tmp**——它把两个输入 `Cast` 成 half 再算；reg 侧走
    `LogicalOrExtend(dst, s0, s1, size)`（`logical_reg_base.h`），在寄存器里做完，不要 tmp。
    两个 implementation（`logical_or.tensor_tensor` / `logical_or.tensor_tensor.extend_reg_base`）。

    标量形态只有 mem 侧有（`logical_or.tensor_scalar`）。

    本 Op 在 `compute` pipe 上执行。
  }];
}

def AFIR_LogicalAndOp : AFIR_LogicalBinaryOp<"logical_and"> {
  let summary = "Element-wise logical AND of two tiles";
  let description = [{
    逐元素逻辑与。其余与 `afir.logical_or` 逐条相同——同形、结果 `ui8`、
    destination-style、mem 侧要 tmp 而 reg 侧不要、`compute` pipe。

    与 `logical_or` 的唯一差别是**标量形态两侧都有**
    （`logical_and.tensor_scalar` / `logical_and.tensor_scalar.extend_reg_base`）。
  }];
}

def AFIR_LogicalXorOp : AFIR_LogicalBinaryOp<"logical_xor"> {
  let summary = "Element-wise logical XOR of two tiles";
  let description = [{
    逐元素逻辑异或。与 `afir.logical_and` / `afir.logical_or` 同形、结果恒 `ui8`,
    差别是**只有 reg 侧**（legacy mem 侧不注册 `LogicalXor`），所以只有
    `logical_xor.tensor_tensor_reg_base`，也没有标量那一半。
  }];
}

// `alpha` 是二元这一类里第一个「ODS 属性变成 API 实参」的 op，与 `afir.leaky_relu`
// 同类但常量的建法相反：`AxpyExtend` 的形参是 `const float alpha`，不跟 `T` 走，
// 所以那个常量恒建 f32；`LeakyRelu` 的是 `const T&`，得跟着 tile 的元素类型建。
// 照抄另一个就会在 f16 上错——一边让 `T` 推不出来，一边多一次舍入。
def AFIR_AxpyOp : AFIR_BufferizableBinaryElementwiseOp<"axpy", [F16, F32, BF16], [F16, F32, BF16],
                                                       /*resultRule=*/"Same", /*scalarOperand=*/"none",
                                                       /*extraAttrs=*/(ins F32Attr:$alpha)> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise fused scale-and-add of two tiles";
  let description = [{
    `y = x1 + alpha * x2`，逐元素。两个输入与结果同形同 dtype，`alpha` 是 op 自己的
    属性、不是操作数——它不进三段划分，也不占一块缓冲。

    buffer 形态是 destination-style，**两套 API 家族共用同一个包装**
    `AxpyExtend(dst, x1, x2, alpha, tmp, size)`（`axpy.h`）：`api_regbase/` 下没有
    axpy，reg 侧不是另有一份实现，而是就用这一份。所以只有一个 implementation
    （`axpy.tensor_tensor`，`ApiFamily::Shared`），不按 target 分流。

    tmp buffer 两个 dtype 分支都收：f16 那支真用它（切两段做 `Cast` / `CAST_RINT`），
    f32 那支一次没用，但形参在签名里，少一个实参就调不上。

    元素类型只有 f16 / f32：两侧的 `REG_ASC_IR(Axpy)` 给的 `T` 就是这两个。
    没有标量形态——对照表两格都是 `no`。本 Op 在 `compute` pipe 上执行。
  }];
}

//===----------------------------------------------------------------------===//
// Compare Operations
//===----------------------------------------------------------------------===//

// 比较族。六个 op 的 ODS 一字不差——差别只有 `CMPMODE`，而那不是 AFIR 这一层的
// 东西：它由「是哪个 op」定死，在 lowering 注册处说出来。所以这里是一个类加六行。
//
// 结果恒为 `ui8`，与输入无关（两侧的 `REG_ASC_IR` 都是 `.Output("y","T2")` 而 `T2`
// 在 mem 侧只有 `uint8`，reg 侧是 `{uint8, bool}`，AFIR 不建模 bool），与
// `AFIR_LogicalBinaryOp` 逐字同形；两个输入之间仍然绑在一起（`T1`）。
// 与 `afir.cast` / `afir.logical_not` 不同，这一类的结果类型推得出来——它是个常量，
// 所以照常留在 `AFIR_BufferizableBinaryElementwiseOp` 上，不用另起基类。
//
// 原先这六行是 `AFIR_BinaryHomoMathOp`（tensor-only），而且六行里有四行是错的:
// 三个写了 `F64`、四个写了 `I64`、`lt` 少了 `I32`。两侧的 legacy 注册里从来没有
// `float64`；`int64` 两侧都有，但 AFIR 这一层的 `ElementTypes` 位掩码拼不出它，
// 于是与 `pow` / `logical_not` 采用同一规则——ODS 写「两侧并集里建模得出来的那些」。
// 六行各写各的没有被发现，是因为这个 op 当时没有任何 implementation
// （与 `floor_div` / `cast` 那两轮同一种）。
//
// 元素类型：mem 侧 `T1` 是 {f16, f32, i32, i64}（`lt` 没有 i64），reg 侧是
// {bf16, f16, f32, i8, i16, i32, i64, ui8, ui16, ui32, ui64, bool}。并集 ∩ 建模得出来的
// = 下面这六个。分档是 lowering 的事，两个 implementation 各说各的一半。
//
// 有标量形态，两侧六个 op 全都有——`OnlySecondInputSupportScalar`。
// swap 方向只有 Eq / Ne 有（它们可交换，`lt` / `le` / `gt` / `ge` 不可），所以
// `scalarOperand` 由实例传：那两个是 `"any"`，其余四个留默认 `"rhs"`。这与 legacy 的
// `IsScalarInputSupportedIfExchangeInputs` 只覆写在 Eq / Ne 上是同一条判据。
class AFIR_CompareOp<string mnemonic, string scalarOperand = "rhs"> :
    AFIR_BufferizableBinaryElementwiseOp<mnemonic, [F16, F32, I16, I32, UI8, UI16, BF16], [UI8],
                                         /*resultRule=*/"Uint8", scalarOperand> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise " # mnemonic # " comparison of two tiles";
  let description = [{
    逐元素比较。两个输入同形同 dtype，结果同形、dtype 恒为 `ui8`（真 1 假 0）。

    buffer 形态是 destination-style。两侧都走包装、**都叫 `CompareExtend`**，差的是
    比较模式落在哪里：mem 侧 `CompareExtend(dst, s0, s1, CMPMODE::EQ, size, tmp)`
    （`compare.h`，tmp 在 size **之后**）；reg 侧
    `CompareExtend<T, 1, CMPMODE::EQ>(dst, s0, s1, {size}, {1}, {1})`
    （`compare_reg_base.h`，不收 tmp）。两侧调度都分一块 tmp buffer，reg 侧那块是
    32 字节的死数、调用收不下——`TmpBufferUse::Unused` 那一档。

    标量形态（右操作数是一个标量）两侧都有，建的是 `CompareScalarExtend`。
    本 Op 在 `compute` pipe 上执行。
  }];
}

def AFIR_EqOp : AFIR_CompareOp<"eq", /*scalarOperand=*/"any">;
def AFIR_NeOp : AFIR_CompareOp<"ne", /*scalarOperand=*/"any">;
def AFIR_LtOp : AFIR_CompareOp<"lt">;
def AFIR_LeOp : AFIR_CompareOp<"le">;
def AFIR_GtOp : AFIR_CompareOp<"gt">;
def AFIR_GeOp : AFIR_CompareOp<"ge">;

//===----------------------------------------------------------------------===//
// Ternary Operations
//===----------------------------------------------------------------------===//
// `clip_by_value(x, min, max)`：逐元素夹到 [min, max]。三个输入与结果同 dtype，所以走
// homo 那一档——三元躯干的操作数名 `condition` / `input1` / `input2` 在这个 op 上读作
// x / min / max。
// mem 侧 legacy 只收 f16 / f32，reg 侧到 ui16。ODS 取并集，两侧各自收哪些写在注册处。
def AFIR_ClipByValueOp :
    AFIR_BufferizableTernaryHomoElementwiseOp<"clip_by_value", [F16, F32, I16, I32, UI8, UI16, BF16],
                                              /*scalarInputs=*/1>;

// step 3。`fma(x1, x2, x3) = x1 * x2 + x3`。三个输入与结果同 dtype，所以与
// `clip_by_value` 同走 homo 那一档——躯干的 `condition` / `input1` / `input2` 在这个 op 上
// 读作 x1 / x2 / x3，三个源是对称的，没有「条件」那层语义。
//
// `scalarInputs` 不给：legacy 的 `FmaAscIrCodegenImplV2::IsNodeValid` 用
// `!IsNodeHasScalarInput(node)` 明确拒标量，`adv_api/math/fma.h` 里也没有标量重载。
//
// 元素类型取 legacy `REG_ASC_IR(Fma)` 的 `T` 去掉 i8（AFIR 的那张表里没有这一位）。
// 其中 bf16 / i16 / ui8 在 legacy 侧会被 `dtype_consistency` 按
// `GetConversionDtype` 先换成 f32 / f16 再生成 `Fma`，但「这个节点类型收哪几种」是注册处
// 与对照表比对的那一列，照 `REG_ASC_IR` 写——比 legacy 窄才是真损失。
def AFIR_FmaOp : AFIR_BufferizableTernaryHomoElementwiseOp<"fma", [F16, F32, I16, UI8, BF16]> {
  let afirStability = AFIR_Experimental;
  let summary = "Element-wise fused multiply-add of three tiles";
  let description = [{
    逐元素算 `x1 * x2 + x3`。三个输入同形同 dtype，结果同形同 dtype。

    走标准调用 `AscendC::Fma(dst, s0, s1, s2, tmp, size)`（`adv_api/math/fma.h`），
    没有包装、没有扩展头，要一块 tmp（legacy 的 `CalcVoidTmpSizeV2`，尺寸是 0 但形参
    在就照收，与 `afir.hypot` 同一种情况）。**tmp 在 count 之前**，与这一族其余调用相反。

    只有 reg 一侧：legacy 的 `REG_ASC_IR(Fma)` 只在 v2 那张注册表上。
    implementation `fma.tensor`。
  }];
}

#endif // AFIR_MATH_OPS
