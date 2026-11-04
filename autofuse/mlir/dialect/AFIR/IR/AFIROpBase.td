// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

//===- AFIROpBase.td - AFIR operation base classes ----------*- tablegen -*-===//
//
// Op base classes for the AFIR dialect (Nary/Nullary/Unary/Binary/Ternary/
// Variadic). Ported faithfully from Ascend-MLIR-gsr. Type-inference traits are
// added per-op-family in the op .td files (e.g. math.td), not here.
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_BASE
#define AFIR_BASE

include "AFIRDialect.td"
include "AFIRAttrs.td"
include "AFIRInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

// 目标方言在 lowering 过程中引入的 shaped 值（例如 AscendC 的 GlobalTensor /
// LocalTensor）是迁移期的类型化中间态：它们实现 ShapedTypeInterface，但既不是
// Tensor 也不是 MemRef。这里保持它们可解析，同时把规范的元素类型约束继续施加在
// 标准 tensor / buffer 形态上。
def AFIR_TargetShapedType : Type<
    And<[IsShapedTypePred,
         CPred<"!::llvm::isa<::mlir::TensorType, ::mlir::BaseMemRefType>($_self)">]>,
    "target dialect shaped type", "::mlir::ShapedType">;

class AFIR_BufferizableShapedType<list<Type> elementTypes> :
    AnyTypeOf<[TensorOrMemRef<elementTypes>, AFIR_TargetShapedType],
              "", "::mlir::ShapedType">;

// 片上临时缓冲。相当一部分 AscendC API 要求调用方额外交一块 tmp buffer（mem-based 侧
// 47 个共有 elementwise Op 里有 34 个要），语义上是「一块字节」而不是某个 dtype 的
// 张量，所以固定成一维 `ui8` memref。
//
// 它的**字节数不在 AFIR 里算**：实际分配值是 ATT 侧按 `max(各节点 Σ) ⊔ 8KB` 聚合出
// 来的 `b{id}_size`，AFIR 只能引用。因此承载它的 `memref.alloc` 上挂的是
// `#afir.buffer_desc<size_from_resource = true>`，尺寸由 MemoryAllocate 绑到那个
// canonical 资源名上——见 docs/design/2026-09-17 的 tmp buffer 一节。
// 约束沿用其余可 bufferize 操作数的那一套而不是只写 `memref`：下降途中 tmp buffer 会
// 先变成目标方言的张量（`!ascendc.local_tensor<?xui8>`），此时 Op 本身还没被重写。
def AFIR_TmpBufferType : AFIR_BufferizableShapedType<[UI8]>;

//===----------------------------------------------------------------------===//
// Common per-op attributes
//===----------------------------------------------------------------------===//
//
// 删除旧 `ascir` 的 `ir_attr_def`（DictionaryAttr）属性：AFIR Core 不提供任意键值对的透传通道，
// 否则任何尚未建模的概念都可以绕过 ODS 与 verifier 悄悄进入 IR，稳定性分层和
// 对外文档随之失效（设计文档 I1 / 3.7 / ADR-7）。
//
// Schedule 模块迁移期若确有尚未建模的字段需要透传，承载方是 adapter 命名空间下的
// func 级具名属性（如 `#af_migration.pending<>`），不是 Core 的 Op 属性；且每项
// 载荷必须记录删除条件。`ir_attr_def` 的删除条件是：Schedule 迁移完成、其字段全部
// 转为强类型 Attr。届时字段已知，应直接建强类型 Attr 而非复用字典。
//
// 删除 ODS 属性 `loop_axis`（步 3b）：它声明在全部 54 个 Op 上，生成的 `getLoopAxis()`
// **零调用**。它唯一的作用是让 `AFIR_RejectUnknownAttributes` 的「去前缀后比对已注册名」
// 逻辑放行同名的 discardable 属性 `afir.loop_axis`——而后者是 i64、这里声明的是 I32，
// 于是 ODS 里写的类型与 IR 里实际出现的值对不上，等于用一个假声明当白名单。
//
// 现在最内层循环轴由强类型的 `#afir.sched` 承载（`afir.sched`，登记在 `isKnownAFIR`），
// 这个假声明随之取消。
//
// 删除 ODS 属性 `operand_buffer_descs` / `result_buffer_descs` / `destination_buffer_descs`
// （步 8）：值的规划结果改挂在值的定义点上，一个值一份（`afir.buffer_desc`），使用点不再
// 有与操作数按下标对齐的数组。留着这三个声明会让 `AFIR_RejectUnknownAttributes` 继续
// 放行同名的 discardable 属性——旧写法写回来不报错，等于迁移没有闸门。
//
// tensor descriptor（`operand_descs` / `result_descs` / `destination_descs`）保留：轴与
// role 是**使用点**的事实，同一个 GM 值在不同使用点可以有不同的轴解释。
class AFIR_CommonAttrs {
  dag commonArgs = (ins
    OptionalAttr<AffineMapArrayAttr>:$indexing_maps,
    OptionalAttr<ArrayAttr>:$operand_descs,
    OptionalAttr<ArrayAttr>:$result_descs,
    OptionalAttr<ArrayAttr>:$destination_descs
  );
}

//===----------------------------------------------------------------------===//
// Side-effect resources
//===----------------------------------------------------------------------===//
//
// 同步原语（`afir.event_record` / `afir.event_wait` / `afir.pipe_barrier`）作用在
// 「pipe 之间的执行顺序」这件全局状态上，而不是某个操作数或结果。它们既没有操作数也
// 没有可依赖的结果，因此必须显式声明副作用——否则「不会被删掉」这份安全是**隐式**的：
// 今天成立只是因为没有 MemoryEffect 接口即视为未知副作用，一旦有人给基类加上 `Pure`，
// `afir.event_wait` 会被静默 DCE 掉，而失败形态是偶发的错数据。
def AFIR_PipelineResource : Resource<"::mlir::afir::PipelineResource">;

//===----------------------------------------------------------------------===//
// Stability tiers
//===----------------------------------------------------------------------===//
//
// 稳定性分层见 docs/design/2026-09-09-afir-public-dsl-architecture-design.md 2.5：
//
//   stable        对外公开契约，外部可依赖；破坏性变更需大版本 + 迁移诊断
//   experimental  对外公开但可能变；可在小版本内变更
//   internal      不对外；afplan 与 adapter 全部在此，可自由演进
//
// 默认 internal。Op 在自己的定义体里显式升档：
//
//   def AFIR_AddOp : AFIR_BufferizableBinaryElementwiseOp<"add", ...> {
//     let afirStability = AFIR_Stable;
//     let summary = "...";
//     let description = [{ ... }];
//   }
//
// stable 档强制要求非空 summary / description——它们会直接进入 `-gen-dialect-doc`
// 产出的对外文档。该约束由 tools/af-mlir-tools/check_afir_ods_docs.py 在 PR 门禁中
// 检查，缺失即拦截，使文档质量随重构单调上升。
defvar AFIR_Internal = "internal";
defvar AFIR_Experimental = "experimental";
defvar AFIR_Stable = "stable";

//===----------------------------------------------------------------------===//
// Base AFIR Operation
//===----------------------------------------------------------------------===//
class AFIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<AFIR_Dialect, mnemonic, traits # [AFIR_RejectUnknownAttributes]> {
  // 稳定性档位，取值见上方 AFIR_Internal / AFIR_Experimental / AFIR_Stable。
  // 该字段只被文档门禁脚本读取，不参与任何代码生成。
  string afirStability = AFIR_Internal;
}

//===----------------------------------------------------------------------===//
// Base AFIR operation carrying a verifier
//===----------------------------------------------------------------------===//
class AFIR_BaseOp<string mnemonic, list<Trait> traits> : AFIR_Op<mnemonic, traits> {
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// N-ary op base
//===----------------------------------------------------------------------===//
class AFIR_NaryOp<string mnemonic,
                          list<Trait> traits,
                          int numOperands,
                          int numResults = 1,
                          bit finalize = 0> :
    AFIR_BaseOp<mnemonic, traits> {
  code baseMethodDecl = [{
  int getNumberOfOperands();
  int getNumberOfResults();
  }];
  code baseMethods = [{
      int $cppClass::getNumberOfOperands() { return }] # numOperands # [{; }
      int $cppClass::getNumberOfResults() { return }] # numResults # [{; }
    }];
  let hasVerifier = 1;
  let extraClassDeclaration = !if(finalize, baseMethodDecl, "");
  let extraClassDefinition = !if(finalize, baseMethods, "");
}

//===----------------------------------------------------------------------===//
// ODS 元素类型列表的可机读副本
//===----------------------------------------------------------------------===//
//
// **ODS 这一份元素类型列表是推导得出来的，不是独立事实。**
//
// 按设计（`docs/development/2026-10-16-afir-elementwise-broadcast-closeout-design.md`
// 的功能需求 3 第一步）：「AFIR 能不能表达某个 dtype」= 两侧 implementation「供不供得起」
// 的**并集**。所以 ODS 的列表应当等于注册处那两份位掩码的并集（投影到 AFIR 的供给表上）。
//
// 在这个 mixin 之前，这件事没有任何机器查过，于是两个方向都出过事，而且**两个方向都不会
// 红**：
//
//   - 注册处写了而 ODS 没有：那一位是句没人读的假话。op 建不出来，implementation 永远
//     选不到——`ElementTypeMatrixTest` 的文件头记着 2026-09-23 一次清掉 17 处就是这一类。
//   - ODS 有而注册处没有：op 建得出来、选不出 implementation，整图干净回落。不报错，只是
//     白回落，而接管表上这一格的首因会指向「没有 Ascend 实现」而不是「这一位没注册」。
//
// 发出来的是 CSV 拼写（`i16;ui8;bf16`），与 `ascir_op_lowering_matrix.csv` 和
// `support/AfirElementTypes.h` 的 `CSV` 列同一套拼写。能直接这么发是因为 MLIR 内建类型
// 记录的名字小写之后恰好就是那套拼写（`I16` → `i16`、`BF16` → `bf16`）；哪天这个巧合不
// 成立，门禁会报「不认识的拼写」而不是静默少查一位。
class AFIR_OdsElementTypes<list<Type> types> {
  code odsElementTypesDecl = [{
  /// 这个 op 的 ODS **输入**元素类型列表，CSV 拼写、分号分隔。
  ///
  /// 唯一的读点是 `ElementTypeMatrixTest.OdsElementTypesMatchRegistrations`。判据按输入列表来，是因为
  /// implementation 的位掩码查的就是输入的元素类型（`areElementTypesSupported` 把
  /// `allowed` 用在 `sourceElement` 上）；结果那一侧由 `ResultElementRule` 管。
  static ::llvm::StringRef getOdsElementTypeSpellings() {
    return "}] # !interleave(!foreach(t, types, !tolower(!cast<string>(t))), ";") # [{";
  }
  }];
}

//===----------------------------------------------------------------------===//
// Bufferizable op bases
//===----------------------------------------------------------------------===//
//
// These bases keep the logical tensor form and the post-allocation buffer form
// under one operation name. The optional destination distinguishes the forms:
// logical operations have no destination, while buffer-form operations do.
// The shapedType constraint keeps canonical Tensor/MemRef element types in
// ODS while concrete descriptor validation remains shared by the buffer-form
// operation verifiers.
//
// `resultShapedType` defaults to `shapedType` and only differs for operations
// whose result element type is not the input's -- the comparison and logical
// operations produce `ui8` from float inputs, and `afir.cast` by definition
// changes type. The destination shares it, not the input's: it is the storage
// the result is written into.
// `extraAttrs` 收这个 op 自己的属性（`afir.leaky_relu` 的 `negative_slop`）。属性进
// `attr-dict`，所以 assemblyFormat 一个字不用改；它也不是操作数，三段划分不受影响。
class AFIR_BufferizableUnaryOpBase<string mnemonic,
                                   list<Trait> traits,
                                   bit finalize = 0,
                                   Type shapedType,
                                   Type resultShapedType = shapedType,
                                   dag extraAttrs = (ins)> :
    AFIR_BaseOp<mnemonic,
                traits # [AttrSizedOperandSegments,
                          AFIRRoleInterface,
                          DeclareOpInterfaceMethods<AFIRRoleInterface>,
                          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                          DeclareOpInterfaceMethods<ConditionallySpeculatable>]>,
    AFIR_CommonAttrs {
  let arguments = !con(
    (ins shapedType:$input, Optional<resultShapedType>:$destination,
         Optional<AFIR_TmpBufferType>:$tmpBuffer),
    extraAttrs,
    commonArgs
  );
  let results = (outs resultShapedType:$result);
  let assemblyFormat = [{
    $input (`outs` `(` $destination^ `)`)? (`tmp` `(` $tmpBuffer^ `)`)? attr-dict `:`
    qualified(type($input)) (`,` qualified(type($destination))^)?
    (`,` qualified(type($tmpBuffer))^)? `->`
    qualified(type($result))
  }];
  code baseMethodDecl = [{
  int getNumberOfOperands();
  int getNumberOfResults();
  }];
  code baseMethods = [{
    int $cppClass::getNumberOfOperands() { return getOperation()->getNumOperands(); }
    int $cppClass::getNumberOfResults() { return getOperation()->getNumResults(); }
    unsigned $cppClass::getNumDestinations() { return getDestination() ? 1 : 0; }
    unsigned $cppClass::getNumTmpBuffers() { return getTmpBuffer() ? 1 : 0; }
  }];
  let hasVerifier = 1;
  let extraClassDeclaration = !if(finalize, baseMethodDecl, "");
  let extraClassDefinition = !if(finalize, baseMethods, "");
}

// Bufferizable unary op whose buffer form may be side-effect-only. The
// optional result keeps logical tensor form compatible while allowing
// store-like operations to omit the tied result after bufferization.
//
// Deliberately without `$tmpBuffer`, unlike the other bufferizable bases: the
// only operation on this base is `afir.store`, and neither Store nor Load needs
// a tmp buffer on either API family (see the op lowering matrix in
// docs/guides/). Adding an operand that can never legitimately be filled would
// buy uniformity by making every store carry `operandSegmentSizes` for it.
class AFIR_BufferizableUnaryOptionalResultOpBase<string mnemonic,
                                                  list<Trait> traits,
                                                  bit finalize = 0,
                                                  Type shapedType> :
    AFIR_BaseOp<mnemonic,
                traits # [AFIRRoleInterface,
                          DeclareOpInterfaceMethods<AFIRRoleInterface>,
                          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                          DeclareOpInterfaceMethods<ConditionallySpeculatable>]>,
    AFIR_CommonAttrs {
  let arguments = !con(
    (ins shapedType:$input, Optional<shapedType>:$destination),
    commonArgs
  );
  let results = (outs Optional<shapedType>:$result);
  let assemblyFormat = [{
    $input (`outs` `(` $destination^ `)`)? attr-dict `:`
    qualified(type($input)) (`,` qualified(type($destination))^)?
    (`->` qualified(type($result))^)?
  }];
  code baseMethodDecl = [{
  int getNumberOfOperands();
  int getNumberOfResults();
  }];
  code baseMethods = [{
    int $cppClass::getNumberOfOperands() { return getOperation()->getNumOperands(); }
    int $cppClass::getNumberOfResults() { return getOperation()->getNumResults(); }
    unsigned $cppClass::getNumDestinations() { return getDestination() ? 1 : 0; }
    unsigned $cppClass::getNumTmpBuffers() { return 0; }
  }];
  let hasVerifier = 1;
  let extraClassDeclaration = !if(finalize, baseMethodDecl, "");
  let extraClassDefinition = !if(finalize, baseMethods, "");
}

// `lhsType` / `rhsType` 默认都与 `shapedType` 相同。它们各占一个参数，是为了让
// tensor-scalar 形态的 op 把**那一个**操作数放宽成「一块 tile 或者一个标量」——放宽
// 的只是被指名的位置，别的一律不动。两边都能放宽：矩阵上 8 个 op 的标量格是
// `2nd+swap(2nd)`，标量在第一个操作数上那半张格子由 `lhsType` 说出来。
//
// **「两个都是标量」不由类型约束挡**：ODS 的约束是逐个操作数的，说不出「至多一个」。
// 那一条在 verifier 里（`verifyBufferizedElementwise` 要求至少一个输入有形状）。
//
// `extraAttrs` 与一元那边同义：这个 op 自己的属性（`afir.axpy` 的 `alpha`）。属性进
// `attr-dict`，所以 assemblyFormat 一个字不用改；它也不是操作数，三段划分不受影响。
class AFIR_BufferizableBinaryOpBase<string mnemonic,
                                    list<Trait> traits,
                                    bit finalize = 0,
                                    Type shapedType,
                                    Type resultShapedType = shapedType,
                                    Type rhsType = shapedType,
                                    dag extraAttrs = (ins),
                                    Type lhsType = shapedType> :
    AFIR_BaseOp<mnemonic,
                traits # [AttrSizedOperandSegments,
                          AFIRRoleInterface,
                          DeclareOpInterfaceMethods<AFIRRoleInterface>,
                          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                          DeclareOpInterfaceMethods<ConditionallySpeculatable>]>,
    AFIR_CommonAttrs {
  let arguments = !con(
    (ins lhsType:$lhs, rhsType:$rhs, Optional<resultShapedType>:$destination,
         Optional<AFIR_TmpBufferType>:$tmpBuffer),
    extraAttrs,
    commonArgs
  );
  let results = (outs resultShapedType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs (`outs` `(` $destination^ `)`)? (`tmp` `(` $tmpBuffer^ `)`)? attr-dict `:`
    `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)`
    (`,` qualified(type($destination))^)?
    (`,` qualified(type($tmpBuffer))^)? `->` qualified(type($result))
  }];
  code baseMethodDecl = [{
  int getNumberOfOperands();
  int getNumberOfResults();
  }];
  code baseMethods = [{
    int $cppClass::getNumberOfOperands() { return getOperation()->getNumOperands(); }
    int $cppClass::getNumberOfResults() { return getOperation()->getNumResults(); }
    unsigned $cppClass::getNumDestinations() { return getDestination() ? 1 : 0; }
    unsigned $cppClass::getNumTmpBuffers() { return getTmpBuffer() ? 1 : 0; }
  }];
  let hasVerifier = 1;
  let extraClassDeclaration = !if(finalize, baseMethodDecl, "");
  let extraClassDefinition = !if(finalize, baseMethods, "");
}

// 三元的 bufferizable base。与二元的差别只在多一个输入：destination 与 tmpBuffer 的
// 形态、三段划分、assemblyFormat 的可选块都逐字相同。
//
// 这一段在 E0-3 写过一次又删掉了——当时没有任何 def 用到它，而没有实例的 tablegen class
// 不会被展开，也就从来没被检查过。现在 Select / Where / ClipByValue 用得上
// 了才加回来，所以它是被编译、被用例走到的。
//
// `condition` 与 `input1` / `input2` 分开给类型：选择族的第一个操作数是 ui8 真值，后两个
// 是数据，两者的元素类型集合本来就不同。
class AFIR_BufferizableTernaryOpBase<string mnemonic,
                                     list<Trait> traits,
                                     bit finalize = 0,
                                     Type conditionType,
                                     Type shapedType,
                                     Type resultShapedType = shapedType> :
    AFIR_BaseOp<mnemonic,
                traits # [AttrSizedOperandSegments,
                          AFIRRoleInterface,
                          DeclareOpInterfaceMethods<AFIRRoleInterface>,
                          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                          DeclareOpInterfaceMethods<ConditionallySpeculatable>]>,
    AFIR_CommonAttrs {
  let arguments = !con(
    (ins conditionType:$condition, shapedType:$input1, shapedType:$input2,
         Optional<resultShapedType>:$destination,
         Optional<AFIR_TmpBufferType>:$tmpBuffer),
    commonArgs
  );
  let results = (outs resultShapedType:$result);
  let assemblyFormat = [{
    $condition `,` $input1 `,` $input2 (`outs` `(` $destination^ `)`)? (`tmp` `(` $tmpBuffer^ `)`)? attr-dict `:`
    `(` qualified(type($condition)) `,` qualified(type($input1)) `,` qualified(type($input2)) `)`
    (`,` qualified(type($destination))^)?
    (`,` qualified(type($tmpBuffer))^)? `->` qualified(type($result))
  }];
  code baseMethodDecl = [{
  int getNumberOfOperands();
  int getNumberOfResults();
  }];
  code baseMethods = [{
    int $cppClass::getNumberOfOperands() { return getOperation()->getNumOperands(); }
    int $cppClass::getNumberOfResults() { return getOperation()->getNumResults(); }
    unsigned $cppClass::getNumDestinations() { return getDestination() ? 1 : 0; }
    unsigned $cppClass::getNumTmpBuffers() { return getTmpBuffer() ? 1 : 0; }
  }];
  let hasVerifier = 1;
  let extraClassDeclaration = !if(finalize, baseMethodDecl, "");
  let extraClassDefinition = !if(finalize, baseMethods, "");
}

//===----------------------------------------------------------------------===//
// Nullary op base (0 inputs, 1 result)
//===----------------------------------------------------------------------===//
class AFIR_NullaryOpBase<string mnemonic,
                       list<Trait> traits,
                       list<Type> types,
                       bit finalize = 0> :
    AFIR_NaryOp<mnemonic, traits, 0, 1, finalize>, AFIR_CommonAttrs {
  let arguments = commonArgs;
  let results = (outs TensorOf<types>:$result);
  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Unary op base (1 input, 1 result)
//===----------------------------------------------------------------------===//
class AFIR_UnaryOpBase<string mnemonic,
                       list<Trait> traits,
                       list<Type> inputTypes,
                       list<Type> resultTypes,
                       bit finalize = 0> :
    AFIR_NaryOp<mnemonic, traits, 1, 1, finalize>, AFIR_CommonAttrs {
  let arguments = !con(
    (ins TensorOf<inputTypes>:$input),
    commonArgs
  );
  let results = (outs TensorOf<resultTypes>:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  let hasVerifier = 1;
}

// 只有 tensor 形态的二元 base（`AFIR_BinaryOpBase` / `AFIR_BinaryOp` /
// `AFIR_BinaryHomoOp`）也在这里存在过，最后一批实例是比较族那六行。它们换到
// `AFIR_BufferizableBinaryElementwiseOp` 之后三个类一个实例都不剩，理由与下面三元那
// 两个逐字相同。`AFIR_UnaryOpBase` 留着——`Reduce.td` 还在用它；只有它上面那层薄封装
// `AFIR_UnaryOp` 没有实例了，一起删掉。

// 只有 tensor 形态的三元 base（`AFIR_TernaryOpBase` / `AFIR_TernaryOp`）在这里存在过，
// 唯一的实例是 `afir.clip_by_value`。它换到 `AFIR_BufferizableTernaryHomoElementwiseOp`
// 之后两个类一个实例都不剩——没有实例的 tablegen class 不会被展开，留着等于带一段没被
// 检查过的代码，所以删掉。三元的 tensor 形态没有消失：bufferizable 那条的 destination
// 是 `Optional`，不给就是 tensor 形态。

//===----------------------------------------------------------------------===//
// Variadic op base (N inputs, 1 result)
//===----------------------------------------------------------------------===//
class AFIR_VariadicOpBase<string mnemonic,
                         list<Trait> traits,
                         list<Type> inputTypes,
                         list<Type> resultTypes,
                         bit finalize = 0> :
    AFIR_BaseOp<mnemonic, traits>, AFIR_CommonAttrs {
  let arguments = !con(
    (ins Variadic<TensorOf<inputTypes>>:$inputs),
    commonArgs
  );
  let results = (outs TensorOf<resultTypes>:$result);
  let assemblyFormat = [{
    $inputs attr-dict `:` type($inputs) `->` type($result)
  }];
  code baseMethodDecl = [{
  int getNumberOfOperands();
  int getNumberOfResults();
  }];
  code baseMethods = [{
      int $cppClass::getNumberOfOperands() { return getInputs().size(); }
      int $cppClass::getNumberOfResults() { return 1; }
    }];

  let hasVerifier = 1;
  let extraClassDeclaration = !if(finalize, baseMethodDecl, "");
  let extraClassDefinition = !if(finalize, baseMethods, "");
}

class AFIR_VariadicOp<string mnemonic,
                      list<Trait> traits = [],
                      list<Type> types,
                      bit finalize = 0> :
    AFIR_VariadicOpBase<mnemonic, traits, types, types, finalize>;

#endif // AFIR_BASE
