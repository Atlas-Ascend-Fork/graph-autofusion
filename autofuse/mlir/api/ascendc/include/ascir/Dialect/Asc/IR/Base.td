/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_BASE_TD
#define ASC_BASE_TD

include "Dialect.td"
include "Interfaces.td"
include "Core/Attributes.td"
include "Core/Types.td"

// `Arg<..., [MemWrite]>`（寄存器计算那一类要声明的副作用）在这个头里。
include "mlir/Interfaces/SideEffectInterfaces.td"

def AscConstructor : NativeOpTrait<"AscConstructorTrait">;
def AscMemberFunc : NativeOpTrait<"AscMemberFuncTrait">;
def AscFunc : NativeOpTrait<"AscFuncTrait">;

class AscendC_Op<string mnemonic, list<Trait> traits = []>
    : Op<AscendC_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::ascendc";
  let assemblyFormat = "operands attr-dict `:` qualified(type(operands))";
  // `genEmitter` is a boolean flag indicating whether this operation can auto code generation
  // for it's emitter function. It is set to true if any of the following traits are present
  // - AscConstructor: The operation represents a constructor call
  // - AscMemberFunc: The operation represents a member function call
  // - AscFunc: The operation represents a regular function call
  // The flag defaulting to false if none of these traits are present.
  // When true, the operation's auto emitter function will be generated during  code emission phase
  bit genEmitter = !foldl(
    0, traits, init, trait,
    !or(init,
        !eq(trait, AscConstructor),
        !eq(trait, AscMemberFunc),
        !eq(trait, AscFunc)
    )
  );
  // `paramTypeLists` defines the mapping between operation arguments and parameters.
  // The list corresponds one-to-one with the operation's arguments, where each value specifies
  // how the arguments should be treated for template parameters generation:
  // -3: Pointer type parameter
  // -2: Pointer to int type parameter
  // -1: The argument is an attributes (not a template)
  //  0: Regular parameter (pass directly as a function parameter)
  //  1: Extract the template type from the argument at this position (e.g., <typename T> form function(T arg))
  //  2: Extract the element type from a template type (e.g., LocalTensor<T> -> T)
  //  3：Non-type template parameter (enum value)
  //  4: Non-type template parameter (regular value)
  //  5: Type template paramter (eg., <typename T> function())
  //  6: Type attribute as template parameter (e.g., TypeAttr for <typename T>, no function parameter)
  // This allows flexible template argument generation from operation arguments while supporting
  // various patterns like type extraction and non-type template
  list<int> paramTypeLists = [];
}

//===----------------------------------------------------------------------===//
// Base operations
//===----------------------------------------------------------------------===//

class APIOp<string mnemonic, string apiName, list<Trait> traits = []>
    : AscendC_Op<mnemonic, [APIOpInterface] # traits> {
  let summary = "Call `AscendC::" # apiName # "` function";
  string comment = "";
  code extraClassDeclarationBase = [{
    static StringRef getAPIName() { return "}] # apiName # [{"; }
    static StringRef getComment() { return "}] # comment # [{"; }
  }];
  let extraClassDeclaration = extraClassDeclarationBase;
}

class DataCopyOp<string mnemonic, string apiName, list<Trait> traits = []>
    : APIOp<mnemonic, apiName, [DataCopyOpInterface] # traits>;

class CopyToL0Op<string mnemonic, string apiName, list<Trait> traits = []>
    : DataCopyOp<mnemonic, apiName, [CopyToL0OpInterface] # traits>;

class VectorOp<string mnemonic, string apiName, list<Trait> traits = []>
    : APIOp<mnemonic, apiName, [VectorOpInterface] # traits>;

//===----------------------------------------------------------------------===//
// Vector unary operations
//===----------------------------------------------------------------------===//

class UnaryOp<string mnemonic, string apiName, list<Trait> traits = []>
    : VectorOp<mnemonic, apiName, [UnaryOpInterface] # traits> {
  let description = "`AscendC::" # apiName # "` is a vector unary operation.\n";
  let extraClassDeclaration = extraClassDeclarationBase # [{
    SmallVector<Value> getSrcTensors() { return {getSrc()}; }
  }];
}

class UnaryL0Op<string mnemonic, string apiName, list<Trait> traits = []>
    : UnaryOp<mnemonic, apiName, [UnaryL0OpInterface] # traits> {
  let description = "`AscendC::" # apiName # "` is a vector unary operation (L0 API).\n";
  let arguments = (ins AnyType:$dst, AnyType:$src, AnyType:$mask,
                   AnyType:$repeatTimes, AscendC_UnaryRepeatParams:$repeatParams,
                   UnitAttr:$isSetMask);
}

class UnaryL1Op<string mnemonic, string apiName, list<Trait> traits = []>
    : UnaryOp<mnemonic, apiName, traits> {
  let description = "`AscendC::" # apiName # "` is a vector unary operation (L1 API).\n";
  let arguments = (ins AnyType:$dst, AnyType:$src, Variadic<UI64>:$mask,
                   AnyType:$repeatTimes, AscendC_UnaryRepeatParams:$repeatParams,
                   UnitAttr:$isSetMask);
}

class UnaryL2Op<string mnemonic, string apiName, list<Trait> traits = []>
    : UnaryOp<mnemonic, apiName, [UnaryL2OpInterface] # traits> {
  let description = "`AscendC::" # apiName # "` is a vector unary operation (L2 API).\n";
  let arguments = (ins AnyType:$dst, AnyType:$src, AnyType:$calCount);
}

multiclass UnaryL012Op<string mnemonic, string apiName, list<Trait> traits = []> {
  def L0Op : UnaryL0Op<mnemonic # "_l0", apiName, traits>;
  def L1Op : UnaryL1Op<mnemonic # "_l1", apiName, traits>;
  def L2Op : UnaryL2Op<mnemonic # "_l2", apiName, traits>;
}

//===----------------------------------------------------------------------===//
// Vector binary operations
//===----------------------------------------------------------------------===//

class BinaryOp<string mnemonic, string apiName, list<Trait> traits = []>
    : VectorOp<mnemonic, apiName, [BinaryOpInterface] # traits> {
  let description = "`AscendC::" # apiName # "` is a vector binary operation.\n";
  let extraClassDeclaration = extraClassDeclarationBase # [{
    SmallVector<Value> getSrcTensors() { return {getSrc0(), getSrc1()}; }
  }];
}

class BinaryL0Op<string mnemonic, string apiName, list<Trait> traits = []>
    : BinaryOp<mnemonic, apiName, [BinaryL0OpInterface] # traits> {
  let description = "`AscendC::" # apiName # "` is a vector binary operation (L0 API).\n";
  let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1,
                   AnyType:$mask, AnyType:$repeatTimes,
                  AscendC_BinaryRepeatParams:$repeatParams);
}

class BinaryL1Op<string mnemonic, string apiName, list<Trait> traits = []>
    : BinaryOp<mnemonic, apiName, traits> {
  let description = "`AscendC::" # apiName # "` is a vector binary operation (L1 API).\n";
  let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1,
                   Variadic<UI64>:$mask, AnyType:$repeatTimes,
                   AscendC_BinaryRepeatParams:$repeatParams);
}

class BinaryL2Op<string mnemonic, string apiName, list<Trait> traits = []>
    : BinaryOp<mnemonic, apiName, [BinaryL2OpInterface] # traits> {
  let description = "`AscendC::" # apiName # "` is a vector binary operation (L2 API).\n";
  let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1,
                   AnyType:$calCount);
}

class BinaryTemplateL0Op<string mnemonic, string apiName, list<Trait> traits = []>
    : BinaryOp<mnemonic, apiName, [BinaryL0OpInterface] # traits> {
  let description = "`AscendC::" # apiName # "` is a vector binary operation (L0 API).\n";
  let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1,
                   AnyType:$mask, AnyType:$repeatTimes,
                  AscendC_BinaryRepeatParams:$repeatParams, UnitAttr:$isSetMask);
}

class BinaryTemplateL1Op<string mnemonic, string apiName, list<Trait> traits = []>
    : BinaryOp<mnemonic, apiName, traits> {
  let description = "`AscendC::" # apiName # "` is a vector binary operation (L1 API).\n";
  let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1,
                   Variadic<UI64>:$mask, AnyType:$repeatTimes,
                   AscendC_BinaryRepeatParams:$repeatParams, UnitAttr:$isSetMask);
}

class BinaryTemplateL2Op<string mnemonic, string apiName, list<Trait> traits = []>
    : BinaryOp<mnemonic, apiName, [BinaryL2OpInterface] # traits> {
  let description = "`AscendC::" # apiName # "` is a vector binary operation (L2 API).\n";
  let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1,
                   AnyType:$calCount, UnitAttr:$isSetMask);
}

class BinaryCastL0Op<string mnemonic, string apiName, list<Trait> traits = []>
    : BinaryOp<mnemonic, apiName, [BinaryL0OpInterface] # traits> {
  let description = "`AscendC::" # apiName # "` is a vector binary operation (L0 API).\n";
  let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1,
                   AnyType:$mask, AnyType:$repeatTimes,
                  AscendC_BinaryRepeatParams:$repeatParams, UnitAttr:$isSetMask);
}

class BinaryCastL1Op<string mnemonic, string apiName, list<Trait> traits = []>
    : BinaryOp<mnemonic, apiName, traits> {
  let description = "`AscendC::" # apiName # "` is a vector binary operation (L1 API).\n";
  let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1,
                   Variadic<UI64>:$mask, AnyType:$repeatTimes,
                   AscendC_BinaryRepeatParams:$repeatParams, UnitAttr:$isSetMask);
}

class BinaryCastL2Op<string mnemonic, string apiName, list<Trait> traits = []>
    : BinaryOp<mnemonic, apiName, [BinaryL2OpInterface] # traits> {
  let description = "`AscendC::" # apiName # "` is a vector binary operation (L2 API).\n";
  let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1,
                   AnyType:$calCount, UnitAttr:$isSetMask);
}

class BinaryL3Op<string mnemonic, string apiName, list<Trait> traits = []>
    : BinaryOp<mnemonic, apiName, [BinaryL3OpInterface] # traits> {
  let summary = "Call `LocalTensor::" # apiName # "` method";
  let description = "`LocalTensor::" # apiName # "` performs a vector binary operation (L3 API).\n";
  let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1);
}

multiclass BinaryL012Op<string baseMnemonic, string apiName, list<Trait> traits = []> {
  def L0Op : BinaryL0Op<baseMnemonic # "_l0", apiName, traits>;
  def L1Op : BinaryL1Op<baseMnemonic # "_l1", apiName, traits>;
  def L2Op : BinaryL2Op<baseMnemonic # "_l2", apiName, traits>;
}

multiclass BinaryTemplateL012Op<string baseMnemonic, string apiName, list<Trait> traits = []> {
  def L0Op : BinaryTemplateL0Op<baseMnemonic # "_l0", apiName, traits>;
  def L1Op : BinaryTemplateL1Op<baseMnemonic # "_l1", apiName, traits>;
  def L2Op : BinaryL2Op<baseMnemonic # "_l2", apiName, traits>;
}

multiclass BinaryCastL012Op<string baseMnemonic, string apiName, list<Trait> traits = []> {
  def L0Op : BinaryCastL0Op<baseMnemonic # "_l0", apiName, traits>;
  def L1Op : BinaryCastL1Op<baseMnemonic # "_l1", apiName, traits>;
  def L2Op : BinaryL2Op<baseMnemonic # "_l2", apiName, traits>;
}

multiclass BinaryL0123Op<string baseMnemonic, string apiName, string l3operator,
                        list<Trait> traits = []> {
  defm "" : BinaryL012Op<baseMnemonic, apiName, traits>;
  def L3Op : BinaryL3Op<baseMnemonic # "_l3", l3operator, traits>;
}

multiclass BinaryTemplateL0123Op<string baseMnemonic, string apiName, string l3operator,
                        list<Trait> traits = []> {
  defm "" : BinaryTemplateL012Op<baseMnemonic, apiName, traits>;
  def L3Op : BinaryL3Op<baseMnemonic # "_l3", l3operator, traits>;
}

//===----------------------------------------------------------------------===//
// Vector-scalar operations
//===----------------------------------------------------------------------===//

class VecScalarOp<string mnemonic, string apiName, list<Trait> traits = []>
    : VectorOp<mnemonic, apiName, [VecScalarOpInterface] # traits> {
  let description = "`AscendC::" # apiName # "` is a vector-scalar operation.\n";
  let extraClassDeclaration = extraClassDeclarationBase # [{
    SmallVector<Value> getSrcTensors() { return {getSrc()}; }
  }];
}

class VecScalarL0Op<string mnemonic, string apiName, list<Trait> traits = []>
    : VecScalarOp<mnemonic, apiName, [VecScalarL0OpInterface] # traits> {
  let description = "`AscendC::" # apiName # "` is a vector-scalar operation (L0 API).\n";
  let arguments = (ins AnyType:$dst, AnyType:$src, AnyType:$scalar,
                   AnyType:$mask, AnyType:$repeatTimes, AscendC_UnaryRepeatParams:$repeatParams,
                   UnitAttr:$isSetMask);
}

class VecScalarL1Op<string mnemonic, string apiName, list<Trait> traits = []>
    : VecScalarOp<mnemonic, apiName, traits> {
  let description = "`AscendC::" # apiName # "` is a vector-scalar operation (L1 API).\n";
  let arguments = (ins AnyType:$dst, AnyType:$src, AnyType:$scalar,
                   Variadic<UI64>:$mask, AnyType:$repeatTimes, AscendC_UnaryRepeatParams:$repeatParams,
                   UnitAttr:$isSetMask);
}

class VecScalarL2Op<string mnemonic, string apiName, list<Trait> traits = []>
    : VecScalarOp<mnemonic, apiName, [VecScalarL2OpInterface] # traits> {
  let description = "`AscendC::" # apiName # "` is a vector-scalar operation (L2 API).\n";
  // `isSetMask` 缺省必须是 `true`，与 AscendC 声明上的缺省一致——**不能用 `UnitAttr`**。
  //
  // count 形态里这个模板实参不是「要不要多设一次 mask」的微优化，是**由谁设 mask** 的
  // 分工：取 `true` 时 API 自己 `SetMaskCount()` + `SetMask(0, count)` + 收尾复位；取
  // `false` 时 API 整段跳过，按 `repeatTimes = 1` 直生成一条指令，靠调用方**事先**把
  // mask 置成 count 模式并设好 count（见 `dav_c220/kernel_operator_vec_binary_scalar_impl.h`
  // 的 Level 2 分支，m200 / c100 同形）。所以 `false` 是个有用的取值——一串同 count 的
  // L2 调用可以把那四条提到循环外——但它有前置条件。
  //
  // `UnitAttr` 的缺省（属性不写 = `false`）正好落在**有前置条件**的那一侧：谁建 op 时
  // 忘了写，谁就静默声称「mask 我已经设好了」。今天没有任何 pass 提前设 mask，于是
  // 生成的 `Adds<float, 0>` 只算一个 repeat，count 根本没被读——128 个 f32 只有前 64
  // 个是对的（`load_add_scalar_store_128x23x5` 钉的就是这条）。
  let arguments = (ins AnyType:$dst, AnyType:$src, AnyType:$scalar,
                   AnyType:$calCount, DefaultValuedAttr<BoolAttr, "true">:$isSetMask);
}

multiclass VecScalarL012Op<string baseMnemonic, string apiName, list<Trait> traits = []> {
  def L0Op : VecScalarL0Op<baseMnemonic # "_l0", apiName, traits>;
  def L1Op : VecScalarL1Op<baseMnemonic # "_l1", apiName, traits>;
  def L2Op : VecScalarL2Op<baseMnemonic # "_l2", apiName, traits>;
}

//===----------------------------------------------------------------------===//
// Register API operations
//===----------------------------------------------------------------------===//

class RegOp<string mnemonic, string apiName, list<Trait> traits = []>
    : APIOp<mnemonic, "Reg::" # apiName, [RegOpInterface] # traits> {
  let description = "`AscendC::Reg::" # apiName # "` is a register API operation.\n";
}

// 目的地是**操作数**（`dstReg`）而不是结果，所以这一类 op 没有任何结果。**副作用必须
// 声明**：不声明时 MLIR 判它是死代码，`--canonicalize` 会把整段寄存器计算抹掉，而产物
// 仍然编得过（搬运还在，计算没了）。Autofuse 这一侧实测踩过一次，见
// `conversion/AFIRToAscendC/AFIRVectorFuncToRegBase.cpp`。
class BinaryRegOp<string baseMnemonic, string apiName, list<Trait> traits = []>
    : RegOp<baseMnemonic # "_reg", apiName, [BinaryRegOpInterface] # traits> {
  let description = "`AscendC::Reg::" # apiName # "` is a binary register API operation.\n";
  let arguments = (ins Arg<AscendC_RegTensor, "", [MemWrite]>:$dstReg,
                   Arg<AscendC_RegTensor, "", [MemRead]>:$src0Reg,
                   Arg<AscendC_RegTensor, "", [MemRead]>:$src1Reg, AscendC_MaskReg:$maskReg);
}

// 与二元那一类同理（副作用必须声明），见上。
class UnaryRegOp<string baseMnemonic, string apiName, list<Trait> traits = []>
    : RegOp<baseMnemonic # "_reg", apiName, [UnaryRegOpInterface] # traits> {
  let description = "`AscendC::Reg::" # apiName # "` is a unary register API operation.\n";
  let arguments = (ins Arg<AscendC_RegTensor, "", [MemWrite]>:$dstReg,
                   Arg<AscendC_RegTensor, "", [MemRead]>:$srcReg,
                   AscendC_MaskReg:$maskReg);
}

class VecScalarRegOp<string baseMnemonic, string apiName, list<Trait> traits = []>
    : RegOp<baseMnemonic # "_reg", apiName, [VecScalarRegOpInterface] # traits> {
  let description = "`AscendC::Reg::" # apiName # "` is a vector scalar register API operation.\n";
  let arguments = (ins AscendC_RegTensor:$dstReg, AscendC_RegTensor:$srcReg, AnyType: $scalar,
                   AscendC_MaskReg:$maskReg);
}

//===----------------------------------------------------------------------===//
// Math library operations
//===----------------------------------------------------------------------===//

class MathLibraryOp<string mnemonic, string apiName, list<Trait> traits = []>
    : VectorOp<mnemonic, apiName, [MathLibraryOpInterface] # traits> {
  let description = "`AscendC::" # apiName # "` is a math library operation.\n";
}

class UnaryMathOp<string mnemonic, string apiName, list<Trait> traits = []>
    : MathLibraryOp<mnemonic, apiName, [UnaryMathOpInterface, AttrSizedOperandSegments] # traits> {
  let description = "`AscendC::" # apiName # "` is a unary math library operation.\n";
  let arguments = (ins AscendC_LocalTensor:$dst, AnyType:$src, Optional<AnyType>:$sharedTmpBuffer,
                   Optional<AnyType>:$calCount, AnyType:$isReuseSource);
  let extraClassDeclaration = extraClassDeclarationBase # [{
    SmallVector<Value> getDstTensors()
    {
        SmallVector<Value> dstTensors{getDst()};
        if (auto tensor = getSharedTmpBuffer())
            dstTensors.emplace_back(tensor);
        return dstTensors;
    }
    SmallVector<Value> getSrcTensors() { return {getSrc()}; }
  }];
}

class BinaryMathOp<string mnemonic, string apiName, list<Trait> traits = []>
    : MathLibraryOp<mnemonic, apiName, [BinaryMathOpInterface, AttrSizedOperandSegments] # traits> {
  let description = "`AscendC::" # apiName # "` is a binary math library operation.\n";
  let arguments = (ins AscendC_LocalTensor:$dst, AnyType:$src0, AnyType:$src1,
                   Optional<AnyType>:$sharedTmpBuffer, Optional<AnyType>:$calCount, AnyType:$isReuseSource);
  let extraClassDeclaration = extraClassDeclarationBase # [{
    SmallVector<Value> getDstTensors()
    {
        SmallVector<Value> dstTensors{getDst()};
        if (auto tensor = getSharedTmpBuffer())
            dstTensors.emplace_back(tensor);
        return dstTensors;
    }
    SmallVector<Value> getSrcTensors() { return {getSrc0(), getSrc1()}; }
  }];
}

//===----------------------------------------------------------------------===//
// Data Conversion operations
//===----------------------------------------------------------------------===//

class DataConversionOp<string mnemonic, string apiName, list<Trait> traits = []>
    : APIOp<mnemonic, apiName, [DataConversionOpInterface] # traits> {
  let description = "`AscendC::" # apiName # "` is a data conversion operation.\n";
}

class TransposeOp<string mnemonic, string apiName, list<Trait> traits = []>
    : DataConversionOp<mnemonic, apiName, [AscFunc] # traits> {
}

class TransposeSimpleOp<string mnemonic, string apiName, list<Trait> traits = []>
    : TransposeOp<mnemonic, apiName, traits> {
  let summary = "Call `AscendC::" # apiName # "` method (simple version)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src);
}

class TransposeExtOp<string mnemonic, string apiName, list<Trait> traits = []>
    : TransposeOp<mnemonic, apiName, traits> {
  let summary = "Call `AscendC::" # apiName # "` method (extended version)";
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src,
                       AscendC_LocalTensor:$sharedTmpBuffer,
                       AscendC_TransposeParamsExt:$params);
}

class TransDataTo5HDTensorListOp<string mnemonic, string apiName, list<Trait> traits = []>
    : DataConversionOp<mnemonic, apiName, [AttrSizedOperandSegments] # traits> {
  let summary = "Call `AscendC::" # apiName # "` with LocalTensor lists";
  let arguments = (ins Variadic<AscendC_LocalTensor>:$dstList,
                       Variadic<AscendC_LocalTensor>:$srcList,
                       AscendC_TransDataTo5HDParams:$params);
}

class TransDataTo5HDUintListOp<string mnemonic, string apiName, list<Trait> traits = []>
    : DataConversionOp<mnemonic, apiName, [AttrSizedOperandSegments] # traits> {
  let summary = "Call `AscendC::" # apiName # "` with uint64_t address lists";
  let arguments = (ins Variadic<AnyUnsignedInteger>:$dstList,
                       Variadic<AnyUnsignedInteger>:$srcList,
                       AscendC_TransDataTo5HDParams:$params);
}

class TransDataTo5HDSingleOp<string mnemonic, string apiName, list<Trait> traits = []>
    : DataConversionOp<mnemonic, apiName, traits> {
  let summary = "Call `AscendC::" # apiName # "` with LocalTensor<uint64_t>";
  let arguments = (ins AscendC_LocalTensor:$dst,
                       AscendC_LocalTensor:$src,
                       AscendC_TransDataTo5HDParams:$params);
}

#endif // ASC_BASE_TD
