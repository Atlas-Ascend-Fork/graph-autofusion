/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- AFIRVectorFuncToRegBase.cpp - afir.vector_func 下降到寄存器形态 ------===//
//
// 相位 C.5：把 `afir.vector_func` 展开成
//
//     emitasc.vec_scope { emitasc.vf_for { load_align_reg / <op>_reg / store_align_reg } }
//
// 也就是 `__VEC_SCOPE__ { for (...) { AscendC::Reg::... } }`。**中间结果不落 UB**：它们是
// `ascendc.reg_tensor` 产出的向量寄存器，在 region 内按算子顺序串起来。这正是 VF 的收益，
// 也是展平路线（把子图摊成普通 tile 级算子）做不到的——那条路每个中间结果都要一块 UB，
// 而 UB 预算由 legacy 的 buffer 分配与 ATT 定，MLIR 只读 tiling 里的 `q*_size`。
//
// 设计见 docs/development/2026-10-18-afir-vector-func-implementation-design.md 的功能需求 3。
// **与设计文档的一处偏差**：那里写的是「换成 `ascvf.vf_group` 再跑移植来的 `LowerToReg`」。
// 实测对不上——上游 `LowerToReg` 的 pattern 匹配的是 `ascendc::*L2Op` 那一族 stock 算子，
// 而我们这条路下降出来的是 Autofuse 的包装算子（`ascendc.sub_extend_reg_base` 之类），
// **一条都匹配不到**；而且它发的 `Reg::DataCopy` 在这一档 CANN 的 reg 搬运头里不存在
// （那里是 `LoadAlign` / `StoreAlign` 一族）。所以这一步改成直接在**我们自己的算子集**上
// 下降，上游那份留作结构参照。
//
// 能力判定在**这个 pass 的入口**，不在 pattern 里：pattern 失败是硬失败（而且报在 root op
// 上，看不出是哪一条），而「这一期收不了这种 VF」必须是能力不足、干净回落给 legacy。
// 判据见 `checkSupported`。
//
//===----------------------------------------------------------------------===//

#include "conversion/AFIRToAscendC/AFIRToAscendCPasses.h"

#include <memory>
#include <optional>

#include "dialect/AFIR/IR/AFIR.h"
#include "support/CapabilityShortfall.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

namespace mlir::ascend::kernel {
namespace {

// AFIR 算子 -> 寄存器形态算子。**这张表就是「有没有寄存器形态」的真源**：能力判定查它，
// 下降也查它，不存在「查了一张表、按另一张表发」的可能。
//
// **加一行要带一格用例。** 一行声明的是「这个 AscendC::Reg API 与这个 AFIR 算子同义」，
// 而同义是个会错的判断（mask 语义、饱和模式、以及 bf16 这类要模板实参的 dtype）。M1 收
// 两格 E2E 真正走到的那三个（`add` / `div` / `abs`）；M2 加 `cast` / `exp` / `relu`，
// 随后补 `neg`（带 `load_abs_neg_reg_base_store` 那一格 E2E，探针上 T5c 那 10 格也是它）；
// M3 加 `eq` / `where` 这一对（带 `load_eq_where_reg_base_store`）；M4 加 `mul`
// （带 `load_mul_abs_reg_base_store`）。
// 其余按后续分期加，每期一条判据。
enum class RegOpArity : uint8_t { kUnary, kBinary, kTernary };

// 这个算子在寄存器形态下**会不会换元素类型**。只有 `cast` 会。这一位不是装饰：它决定
// 能力层查的是「操作数与结果同类型」还是「这一对 dtype 有没有 trait」，也决定建 op 时要
// 不要带那个 `trait` 实参。
enum class RegOpElementTypes : uint8_t { kUniform, kConverting };

// M3：这个算子在寄存器形态下产出的是什么。
//
// **`kPredicate` 不是「结果是 ui8」的另一种写法。** 比较在寄存器形态下产出的是一条
// **谓词**——`MaskReg`，一位一个 lane，没有元素类型；AFIR 那一侧把它写成 `ui8` 的张量
// 只是因为 AFIR 没有谓词类型。两者的区别是可观测的：谓词进不了 `Reg::Add` 这一族，
// 也存不回 UB；要把它变回一段值，legacy 会补一条 `Reg::Duplicate(dst, 1, mask)`
// （`micro_compare_api_call.cpp:48`），那条今天不建模。
//
// 这一位决定三件事：建的是 `ascendc.mask_reg` 还是 `ascendc.reg_tensor`、这个算子
// 「算在哪个元素类型上」看结果还是看操作数、以及它的结果允许被谁消费。
enum class RegOpResult : uint8_t { kValue, kPredicate };

// 哪一个操作数是谓词（`-1` 表示没有）。`afir.where` 的 0 号操作数是条件。
constexpr int64_t kNoPredicateOperand = -1;

// 建 op 要的一组值。写成一个结构体而不是一串形参：`cast` 多一个 `trait`，而表里只能有
// 一个签名——否则「加一行只动这张表」就不成立了。
struct RegOpOperands {
  Value dstReg;
  ValueRange srcRegs;
  Value maskReg;
  // 只有 `kConverting` 的那一行读它。
  llvm::StringRef castTrait;
};

struct RegOpMapping {
  llvm::StringRef afirName;
  RegOpArity arity;
  RegOpElementTypes elementTypes;
  RegOpResult result;
  int64_t predicateOperand;
  // 建 op 的那一下。表里存函数指针而不是在下降处写一串 `if`：加一行只动这张表。
  Operation *(*build)(OpBuilder &, Location, const RegOpOperands &);
};

template <typename RegOpT>
Operation *buildUnaryRegOp(OpBuilder &builder, Location loc, const RegOpOperands &operands) {
  return builder.create<RegOpT>(loc, operands.dstReg, operands.srcRegs[0], operands.maskReg);
}

template <typename RegOpT>
Operation *buildBinaryRegOp(OpBuilder &builder, Location loc, const RegOpOperands &operands) {
  return builder.create<RegOpT>(loc, operands.dstReg, operands.srcRegs[0], operands.srcRegs[1], operands.maskReg);
}

// M3：比较。目的地是一条谓词（`MaskReg`），`maskReg` 仍是那张 lane 活性掩码。
// 比较模式由**行**定死——它是 AFIR 算子名的一部分（`afir.eq` 就是 `CMPMODE::EQ`），
// 不是运行期的选择，所以写成模板实参而不是 `RegOpOperands` 的一个字段：表里只能有一个
// 签名，否则「加一行只动这张表」就不成立了。
template <ascendc::CMPMODE mode>
Operation *buildCompareRegOp(OpBuilder &builder, Location loc, const RegOpOperands &operands) {
  return builder.create<ascendc::CompareRegOp>(loc, operands.dstReg, operands.srcRegs[0], operands.srcRegs[1],
                                               operands.maskReg, mode);
}

// M3：选择。`afir.where` 的操作数是 (条件, then, else)，而
// `Reg::Select(dst, src0, src1, mask)` 把谓词排在**最后**、`src0` 是谓词为真时取的那一路
// ——与 legacy 的 `micro_where_api_call.cpp:45`（`Select(out, T1, T2, mask)`，mask 来自
// 比较）逐字同序。
//
// **这里传的 mask 是谓词，不是 lane 活性掩码**，与这张表其余各行都不同。那是对的：
// legacy 也这么发，活着的 lane 由存回去那一步的 `StoreAlign` 挡。
Operation *buildSelectRegOp(OpBuilder &builder, Location loc, const RegOpOperands &operands) {
  return builder.create<ascendc::SelectRegOp>(loc, operands.dstReg, operands.srcRegs[1], operands.srcRegs[2],
                                              operands.srcRegs[0]);
}

Operation *buildCastRegOp(OpBuilder &builder, Location loc, const RegOpOperands &operands) {
  return builder.create<ascendc::CastRegOp>(loc, operands.dstReg, operands.srcRegs[0], operands.maskReg,
                                            builder.getStringAttr(operands.castTrait));
}

llvm::ArrayRef<RegOpMapping> regOpMappings() {
  static const RegOpMapping mappings[] = {
      {afir::AddOp::getOperationName(), RegOpArity::kBinary, RegOpElementTypes::kUniform, RegOpResult::kValue,
       kNoPredicateOperand, buildBinaryRegOp<ascendc::AddRegOp>},
      {afir::DivOp::getOperationName(), RegOpArity::kBinary, RegOpElementTypes::kUniform, RegOpResult::kValue,
       kNoPredicateOperand, buildBinaryRegOp<ascendc::DivRegOp>},
      // `truediv` 在 reg 侧与 `div` 发同一个调用，**这不是偷懒**：legacy 的
      // `ascir_op_lowering_matrix.csv` 里 `TrueDiv` 的 `reg_api_name` 就是 `DivExtend`
      // （mem 侧才是 `TrueDivExtend` 加 tmp buffer，那是另一条路）。所以这一行与 legacy
      // 逐字一致，不是我们这一侧选了个近似。
      {afir::TrueDivOp::getOperationName(), RegOpArity::kBinary, RegOpElementTypes::kUniform, RegOpResult::kValue,
       kNoPredicateOperand, buildBinaryRegOp<ascendc::DivRegOp>},
      {afir::AbsOp::getOperationName(), RegOpArity::kUnary, RegOpElementTypes::kUniform, RegOpResult::kValue,
       kNoPredicateOperand, buildUnaryRegOp<ascendc::AbsRegOp>},
      {afir::ExpOp::getOperationName(), RegOpArity::kUnary, RegOpElementTypes::kUniform, RegOpResult::kValue,
       kNoPredicateOperand, buildUnaryRegOp<ascendc::ExpRegOp>},
      {afir::ReluOp::getOperationName(), RegOpArity::kUnary, RegOpElementTypes::kUniform, RegOpResult::kValue,
       kNoPredicateOperand, buildUnaryRegOp<ascendc::ReluRegOp>},
      {afir::NegOp::getOperationName(), RegOpArity::kUnary, RegOpElementTypes::kUniform, RegOpResult::kValue,
       kNoPredicateOperand, buildUnaryRegOp<ascendc::NegRegOp>},
      {afir::CastOp::getOperationName(), RegOpArity::kUnary, RegOpElementTypes::kConverting, RegOpResult::kValue,
       kNoPredicateOperand, buildCastRegOp},
      // M4：`afir.mul`。reg 侧接管表上 `load_brc_test` 报的就是这一行
      // （`'afir.mul' has no register-form lowering`），带一格 E2E
      // `load_mul_abs_reg_base_store`。
      {afir::MulOp::getOperationName(), RegOpArity::kBinary, RegOpElementTypes::kUniform, RegOpResult::kValue,
       kNoPredicateOperand, buildBinaryRegOp<ascendc::MulRegOp>},
      // M3：谓词这一对。两行一起加、一格 E2E（`load_eq_where_reg_base_store`）同时走到
      // ——单独加任一行都没有用：比较产出的谓词今天只有 `where` 消费得了，而 `where` 的
      // 条件今天只能由比较产出（从 UB 取一段 ui8 当条件要补一条 `CompareScalar ... NE 0`，
      // 那是 legacy 的另一条路，不在本期）。
      {afir::EqOp::getOperationName(), RegOpArity::kBinary, RegOpElementTypes::kUniform, RegOpResult::kPredicate,
       kNoPredicateOperand, buildCompareRegOp<ascendc::CMPMODE::EQ>},
      // Legacy LoadGEWhereTransposeCase uses the same predicate path with GE mode.
      {afir::GeOp::getOperationName(), RegOpArity::kBinary, RegOpElementTypes::kUniform, RegOpResult::kPredicate,
       kNoPredicateOperand, buildCompareRegOp<ascendc::CMPMODE::GE>},
      {afir::WhereOp::getOperationName(), RegOpArity::kTernary, RegOpElementTypes::kUniform, RegOpResult::kValue,
       /*predicateOperand=*/0, buildSelectRegOp},
  };
  return mappings;
}

const RegOpMapping *findRegOpMapping(Operation *op) {
  for (const RegOpMapping &mapping : regOpMappings()) {
    if (op->getName().getStringRef() == mapping.afirName) {
      return &mapping;
    }
  }
  return nullptr;
}

size_t operandCountFor(RegOpArity arity) {
  switch (arity) {
    case RegOpArity::kUnary:
      return 1U;
    case RegOpArity::kBinary:
      return 2U;
    case RegOpArity::kTernary:
      return 3U;
  }
  return 0U;
}

std::string listRegOpMappings() {
  std::string names;
  for (const RegOpMapping &mapping : regOpMappings()) {
    names += (names.empty() ? "" : ", ") + mapping.afirName.str();
  }
  return names;
}

//===----------------------------------------------------------------------===//
// M2：跨 dtype 的那三件事
//===----------------------------------------------------------------------===//
//
// 一个 VF 里可以中途换 dtype（`afir.cast`）。换 dtype 不是「多发一条指令」那么简单，因为
// **向量寄存器是定宽的**：`RegTensor<float>` 一段装 `VL/4` 个元素，`RegTensor<half>` 装
// `VL/2` 个。要把 half 转成 float，那些 half 必须先被放进 **4 字节间隔的 lane 槽位**里，
// 否则转出来的是另一组元素。所以跨 dtype 的 VF 有三件事要定：
//
//   1. **lane 的宽度**由这个 VF 里**最宽**的元素类型定（下面的 `widestElementType`）：
//      一段 lane 装 `GetVecLen() / sizeof(max)` 个元素，`UpdateMask<max>` 也按它算。
//      与 legacy 的 `max_dtype_size_` 同一个取法（`vec_func_call.cpp:342`）。
//   2. **窄边界的搬运要带 dist**：取进来用 `DIST_UNPACK_*`（撑进宽槽位）、存回去用
//      `DIST_PACK_*`（压回窄 dtype）。不带就是 `DIST_NORM`——编得过、算得出，**静默算错**。
//   3. **`Reg::Cast` 的 `trait`**：那是个 `const CastTrait&` 模板实参，只能指一个具名的
//      `constexpr`，而挑哪一个取决于 (输入, 输出) 这一对（主要是 RoundMode 与 SatMode）。

// (输入, 输出) -> `api_regbase/utils.h` 里那个具名 `constexpr CastTrait`。
//
// **这张表抄自 legacy 的 `v35/codegen/micro_api_call/micro_cast_api_call.cpp:18`**，一行不改、
// 一行不加——它是「哪一对 dtype 该用哪种舍入/饱和」这件事的既有答案，而那是个会错的判断，
// 不该在这条路上重新发明一遍。常量的**定义**仍然只有一份（`utils_reg_base.h` 被
// `ascendc.cast_reg` 声明为扩展头内联进 kernel），这里复制的只是「挑哪一个」。
//
// **表外的 dtype 对报能力不足，不回落到缺省 trait。** legacy 在查不到时用
// `cast_trait_none`，而那条路上「没列过」与「列过且是 none」分不开；这里宁可干净回落给
// legacy——放开一道闸门要能验数值，而没验过的 dtype 对给不出数值依据。
struct CastTraitMapping {
  // 两侧都用 `ge::DataType` 的那套名字（与被抄的那张表逐行对得上），MLIR 类型先归一成
  // 这个标签再查，避免「f16 与 half 是不是一回事」这种问题散在查表处。
  llvm::StringRef fromDtype;
  llvm::StringRef toDtype;
  llvm::StringRef trait;
};

llvm::ArrayRef<CastTraitMapping> castTraitMappings() {
  static const CastTraitMapping mappings[] = {
      {"float", "float16", "cast_trait_float_2_half"}, {"float", "int64", "cast_trait_float_2_int64"},
      {"float", "int16", "cast_trait_float_2_int64"},  {"float", "int32", "cast_trait_float_2_int32"},
      {"float", "bf16", "cast_trait_float_2_bf16"},    {"float16", "uint8", "cast_trait_float_2_int64"},
      {"float16", "int8", "cast_trait_half_2_int8"},   {"float16", "float", "cast_trait_bf16_2_float"},
      {"int32", "float", "cast_trait_int32_2_float"},  {"int32", "int16", "cast_trait_int32_2_int16"},
      {"int64", "int32", "cast_trait_int32_2_int16"},  {"int64", "float", "cast_trait_int64_2_float"},
      {"bf16", "float", "cast_trait_bf16_2_float"},    {"uint8", "float16", "cast_trait_bf16_2_float"},
      {"int8", "float16", "cast_trait_bf16_2_float"},  {"int8", "int16", "cast_trait_bf16_2_float"},
      {"int16", "float", "cast_trait_bf16_2_float"},   {"int16", "uint8", "cast_trait_int16_2_uint8"},
  };
  return mappings;
}

// MLIR 元素类型 -> 上面那张表用的标签。认不出来的返回空串，查表自然落空。
llvm::StringRef castDtypeTag(Type elementType) {
  if (isa<Float32Type>(elementType)) {
    return "float";
  }
  if (isa<Float16Type>(elementType)) {
    return "float16";
  }
  if (isa<BFloat16Type>(elementType)) {
    return "bf16";
  }
  auto intType = dyn_cast<IntegerType>(elementType);
  if (!intType) {
    return {};
  }
  const bool unsignedInt = intType.isUnsigned();
  switch (intType.getWidth()) {
    case 8U:
      return unsignedInt ? "uint8" : "int8";
    case 16U:
      return unsignedInt ? llvm::StringRef{} : llvm::StringRef{"int16"};
    case 32U:
      return unsignedInt ? llvm::StringRef{} : llvm::StringRef{"int32"};
    case 64U:
      return unsignedInt ? llvm::StringRef{} : llvm::StringRef{"int64"};
    default:
      return {};
  }
}

std::optional<llvm::StringRef> castTraitFor(Type from, Type to) {
  const llvm::StringRef fromTag = castDtypeTag(from);
  const llvm::StringRef toTag = castDtypeTag(to);
  if (fromTag.empty() || toTag.empty()) {
    return std::nullopt;
  }
  for (const CastTraitMapping &mapping : castTraitMappings()) {
    if (mapping.fromDtype == fromTag && mapping.toDtype == toTag) {
      return mapping.trait;
    }
  }
  return std::nullopt;
}

// 这个算子「算在哪个元素类型上」。产值的看结果；产谓词的看它第一个值操作数——谓词没有
// 元素类型（AFIR 那一侧写成 `ui8` 只是因为 AFIR 没有谓词类型），拿 `ui8` 去跟 lane 宽度
// 比会把每一格比较都判成「算在比 lane 窄的 dtype 上」。
Type computeElementType(Operation *inner, const RegOpMapping &mapping) {
  if (mapping.result == RegOpResult::kValue) {
    return getElementTypeOrSelf(inner->getResult(0).getType());
  }
  for (auto [index, operand] : llvm::enumerate(inner->getOperands())) {
    if (static_cast<int64_t>(index) != mapping.predicateOperand) {
      return getElementTypeOrSelf(operand.getType());
    }
  }
  return {};
}

// 这个 VF 里最宽的元素类型。数的是**边界加 region 内每个结果**，与 legacy 一致——legacy 的
// 子图里 `Load` / `Store` 也是节点，它们的输出 dtype 就是边界 dtype，所以两边数的是同一个
// 集合。
//
// **谓词的那个 `ui8` 也数进去，这与 legacy 一致**（`vec_func_call.cpp:342` 的
// `max_dtype_size` 把比较输出一起数了）。它永远不会成为最宽的那一个——1 字节是这张表里
// 最窄的一档——所以数不数都是同一个答案，留着是为了与被抄的那一侧逐字对得上。
Type widestElementType(afir::VectorFuncOp op) {
  Type widest;
  const auto consider = [&widest](Type elementType) {
    if (!widest || ascendc::getTypeSize(elementType) > ascendc::getTypeSize(widest)) {
      widest = elementType;
    }
  };
  for (Value boundary : llvm::concat<Value>(op.getSrcs(), op.getDsts())) {
    consider(cast<ShapedType>(boundary.getType()).getElementType());
  }
  for (Operation &inner : op.getBodyBlock().without_terminator()) {
    for (Value result : inner.getResults()) {
      consider(getElementTypeOrSelf(result.getType()));
    }
  }
  return widest;
}

// 一个窄边界要用哪种 `LoadDist` 才能把元素撑进 `widest` 的 lane 槽位。等宽返回「不带 dist」
// （`std::nullopt` 包着的空 optional 在这里不好读，所以分两层：外层 `failed` 表示这一对
// 宽度**这一期撑不出来**，内层空表示不需要 dist）。
//
// 判据抄自 legacy 的 `DetermineLoadDistAndUnPackSequence`（`micro_load_api_call.cpp:29`），
// **但只收一步到位的那几种**：legacy 在一条 dist 不够时会再补一串 `Reg::UnPack`，那要多两个
// op（`unpack_reg` / `pack_reg`）与一个临时寄存器，归后续。今天「一步不够」只发生在最宽是
// 8 字节的 VF 上，那一类整图干净回落。
FailureOr<std::optional<ascendc::LoadDist>> loadDistFor(Type boundary, Type widest) {
  const uint32_t from = ascendc::getTypeSize(boundary);
  const uint32_t to = ascendc::getTypeSize(widest);
  if (from == to) {
    return std::optional<ascendc::LoadDist>{};
  }
  if (from == 1U && to == 2U) {
    return std::optional{ascendc::LoadDist::DIST_UNPACK_B8};
  }
  if (from == 1U && to == 4U) {
    return std::optional{ascendc::LoadDist::DIST_UNPACK4_B8};
  }
  if (from == 2U && to == 4U) {
    return std::optional{ascendc::LoadDist::DIST_UNPACK_B16};
  }
  return failure();
}

// 存回去那一侧，与上面对称。抄自 legacy 的 `DetermineStoreDistAndPackSequence`
// （`micro_store_api_call.cpp:28`），同样只收一步到位的那几种。
FailureOr<std::optional<ascendc::StoreDist>> storeDistFor(Type boundary, Type widest) {
  const uint32_t to = ascendc::getTypeSize(boundary);
  const uint32_t from = ascendc::getTypeSize(widest);
  if (from == to) {
    return std::optional<ascendc::StoreDist>{};
  }
  if (from == 2U && to == 1U) {
    return std::optional{ascendc::StoreDist::DIST_PACK_B16};
  }
  if (from == 4U && to == 1U) {
    return std::optional{ascendc::StoreDist::DIST_PACK4_B32};
  }
  if (from == 4U && to == 2U) {
    return std::optional{ascendc::StoreDist::DIST_PACK_B32};
  }
  return failure();
}

// 本阶段（M2）收哪些 VF。不满足的**全部报能力不足**，整图干净回落给 legacy——这些都是
// 「还没做」而不是「坏了」。每一条都对应分期表上的一项，见设计文档功能需求 5。
//
// **判据的顺序就是诊断说哪一条的顺序**，而一张图可能同时违反几条（最宽 8 字节那一类必然
// 同时违反取与存两条，因为 8 字节不在 `afir.vector_func` 的边界 dtype 域里）。要改顺序就
// 要同时改 `test/lit/conversion/afir-vector-func-unsupported.mlir` 的预期。
WalkResult checkRegisterElementTypes(Operation *inner, const RegOpMapping *mapping, Type widest) {
  const Type resultType = computeElementType(inner, *mapping);
  if (mapping->elementTypes == RegOpElementTypes::kConverting) {
    const Type sourceType = getElementTypeOrSelf(inner->getOperand(0).getType());
    if (!castTraitFor(sourceType, resultType)) {
      afir::emitCapabilityShortfall(inner->getLoc())
          << "UnsupportedOperation: '" << inner->getName().getStringRef() << "' from " << sourceType << " to "
          << resultType << " has no register-form cast trait";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  }
  // 换 dtype 的只有 `cast`，其余算子的操作数与结果必须是同一个元素类型——`Reg::` 那一族
  // 二元/一元 API 只有一个元素类型。**谓词那一位不在其中**：它没有元素类型。
  for (auto [index, operand] : llvm::enumerate(inner->getOperands())) {
    if (static_cast<int64_t>(index) == mapping->predicateOperand) {
      continue;
    }
    if (getElementTypeOrSelf(operand.getType()) != resultType) {
      afir::emitCapabilityShortfall(inner->getLoc())
          << "UnsupportedOperation: '" << inner->getName().getStringRef()
          << "' mixes operand and result element types, which has no register-form lowering";
      return WalkResult::interrupt();
    }
  }
  // **算的那一步只在最宽的那个 dtype 上做。** 窄 dtype 的算子不收：它的寄存器里元素是按
  // 宽槽位排的（窄边界被 `DIST_UNPACK_*` 撑开过），而 `Reg::Abs(RegTensor<half>&, mask)`
  // 读的是 `VL/2` 个 half，与 `UpdateMask<float>` 给的那张 mask 对不上几个 lane 活着
  // ——这件事只能在设备上量，量不出来就不该在能力层答「支持」。今天 17 格里跨 dtype 的
  // 那几格都是「先 cast 到宽的、算完再 cast 回去」，所以这一条不挡它们。
  if (resultType != widest) {
    afir::emitCapabilityShortfall(inner->getLoc())
        << "UnsupportedOperation: '" << inner->getName().getStringRef() << "' computes in " << resultType
        << " while the vector function's lanes are " << widest << " wide, which has no register-form lowering yet";
    return WalkResult::interrupt();
  }
  return WalkResult::advance();
}

LogicalResult checkSupported(afir::VectorFuncOp op) {
  if (op.getDsts().size() != 1U) {
    return afir::emitCapabilityShortfall(op.getLoc())
           << "UnsupportedOperation: afir.vector_func with " << op.getDsts().size()
           << " destinations has no register-form lowering yet; only a single destination is modeled";
  }

  // lane 的宽度由最宽的那个元素类型定，窄边界靠 `LoadDist` / `StoreDist` 撑进/压回那个
  // 槽位。撑不出来的（最宽 8 字节那一类，legacy 要再补一串 `Reg::UnPack`）干净回落。
  const Type widest = widestElementType(op);
  for (Value src : op.getSrcs()) {
    const Type boundary = cast<ShapedType>(src.getType()).getElementType();
    if (failed(loadDistFor(boundary, widest))) {
      return afir::emitCapabilityShortfall(op.getLoc())
             << "UnsupportedOperation: afir.vector_func loads " << boundary << " into " << widest
             << " lanes, which needs an unpack sequence that has no register-form lowering yet";
    }
  }
  const Type destinationType = cast<ShapedType>(op.getDsts().front().getType()).getElementType();
  if (failed(storeDistFor(destinationType, widest))) {
    return afir::emitCapabilityShortfall(op.getLoc())
           << "UnsupportedOperation: afir.vector_func stores " << widest << " lanes into " << destinationType
           << ", which needs a pack sequence that has no register-form lowering yet";
  }

  WalkResult walk = op.getBody().walk([&](Operation *inner) {
    if (isa<afir::VectorFuncYieldOp>(inner)) {
      return WalkResult::advance();
    }
    const RegOpMapping *mapping = findRegOpMapping(inner);
    if (mapping == nullptr) {
      afir::emitCapabilityShortfall(inner->getLoc())
          << "UnsupportedOperation: '" << inner->getName().getStringRef()
          << "' has no register-form lowering; the modeled operations are " << listRegOpMappings();
      return WalkResult::interrupt();
    }
    const size_t expected = operandCountFor(mapping->arity);
    if (inner->getNumOperands() != expected || inner->getNumResults() != 1U) {
      afir::emitCapabilityShortfall(inner->getLoc())
          << "UnsupportedOperation: '" << inner->getName().getStringRef() << "' has " << inner->getNumOperands()
          << " operands and " << inner->getNumResults()
          << " results inside a vector function, which the register-form lowering does not model";
      return WalkResult::interrupt();
    }
    // M3：谓词的来路与去处。两条都是「还没建模」而不是「非法」，所以都报能力不足。
    //
    // 一、谓词只能喂给声明了谓词操作数的那一行。legacy 在消费者不是 `Select` 的 mask 时
    //     补一条 `Reg::Duplicate(dst, static_cast<T>(1), mask)` 把它物化回一段值
    //     （`micro_compare_api_call.cpp:48`），本期不建模——包括「VF 直接把比较结果存
    //     回 UB」那一种（`afir.vector_func_yield` 不是谓词消费者，所以也落在这里）。
    //     判据与 legacy 的 `ShouldInitAsMaskReg`（`vec_func_call.cpp:264`）同一条：
    //     **所有**消费者都得是 Select 的 mask 那一位，有一个不是就不能当谓词用。
    if (mapping->result == RegOpResult::kPredicate) {
      for (OpOperand &use : inner->getResult(0).getUses()) {
        const RegOpMapping *consumer = findRegOpMapping(use.getOwner());
        if (consumer == nullptr || consumer->predicateOperand == kNoPredicateOperand ||
            use.getOperandNumber() != static_cast<unsigned>(consumer->predicateOperand)) {
          afir::emitCapabilityShortfall(inner->getLoc())
              << "UnsupportedOperation: '" << inner->getName().getStringRef()
              << "' produces a predicate register that is consumed as a value, which needs a register-form "
                 "materialization that is not modeled yet";
          return WalkResult::interrupt();
        }
      }
    }
    // 二、谓词操作数只能来自 region 内产谓词的那一行。来自块参数（从 UB 取进来的一段
    //     ui8 真值）时 legacy 补一条 `Reg::CompareScalar<..., NE>(tmp, reg, 0, pReg)`
    //     把它转成谓词（`micro_where_api_call.cpp:30`），本期同样不建模。
    if (mapping->predicateOperand != kNoPredicateOperand) {
      Operation *producer = inner->getOperand(static_cast<unsigned>(mapping->predicateOperand)).getDefiningOp();
      const RegOpMapping *producerMapping = producer == nullptr ? nullptr : findRegOpMapping(producer);
      if (producerMapping == nullptr || producerMapping->result != RegOpResult::kPredicate) {
        afir::emitCapabilityShortfall(inner->getLoc())
            << "UnsupportedOperation: '" << inner->getName().getStringRef()
            << "' takes a condition that is not a predicate register, which needs a register-form comparison "
               "against zero that is not modeled yet";
        return WalkResult::interrupt();
      }
    }

    return checkRegisterElementTypes(inner, mapping, widest);
  });
  if (walk.wasInterrupted()) {
    return failure();
  }
  return success();
}

// UB 侧的首地址。`local_tensor.get_phy_addr_v2` 打出来是
// `reinterpret_cast<__ubuf__ T*>(tensor.GetPhyAddr())`，与 legacy 的
// `(__local_mem__ float *)local_5[0].GetPhyAddr()` 是同一件事（`__local_mem__` 与
// `__ubuf__` 在这一代上同义）。
FailureOr<Value> makeUbufAddress(OpBuilder &builder, Location loc, Value tensor, Type elementType) {
  auto tensorType = dyn_cast<ascendc::LocalTensorType>(tensor.getType());
  if (!tensorType) {
    return emitError(loc) << "afir.vector_func boundary is not an AscendC local tensor; the register-form lowering "
                             "runs after memory allocation, where it must be";
  }
  auto addressType = MemRefType::get(tensorType.getShape(), elementType, MemRefLayoutAttrInterface{},
                                     builder.getI64IntegerAttr(static_cast<int64_t>(ascendc::AddressSpace::ubuf)));
  return builder.create<ascendc::LocalTensorGetPhyAddrV2Op>(loc, addressType, tensor).getResult();
}

LogicalResult lowerVectorFunc(afir::VectorFuncOp op) {
  if (failed(checkSupported(op))) {
    return failure();
  }
  OpBuilder builder(op);
  const Location loc = op.getLoc();
  MLIRContext *ctx = builder.getContext();
  Value destination = op.getDsts().front();
  const Type destinationType = cast<ShapedType>(destination.getType()).getElementType();
  // lane 的宽度由 VF 里最宽的元素类型定，不由 destination 定——跨 dtype 的 VF 里
  // destination 可能是窄的那一侧（`cast_abs` 那一类：half 进、half 出，中间走 float）。
  const Type widest = widestElementType(op);

  // 作用域外：首地址、lane 一段多少个元素、循环次数、以及给 `UpdateMask` 用的那个计数器。
  //
  // **首地址按各自的边界 dtype 取**，不按 lane 的宽度：`LoadAlign<T>` 的 `T` 就是 UB 里
  // 那块的元素类型，撑不撑进宽槽位是 `dist` 的事。
  SmallVector<Value> sourceAddresses;
  sourceAddresses.reserve(op.getSrcs().size());
  for (Value src : op.getSrcs()) {
    FailureOr<Value> address = makeUbufAddress(builder, loc, src, cast<ShapedType>(src.getType()).getElementType());
    if (failed(address)) {
      return failure();
    }
    sourceAddresses.push_back(*address);
  }
  FailureOr<Value> destinationAddress = makeUbufAddress(builder, loc, destination, destinationType);
  if (failed(destinationAddress)) {
    return failure();
  }

  const Type indexType = builder.getIndexType();
  Value elementCount = builder.create<ascendc::LocalTensorGetSizeOp>(loc, indexType, destination).getResult();
  Value vectorLength = builder.create<ascendc::GetVecLenOp>(loc, indexType).getResult();
  Value elementBytes = builder.create<arith::ConstantIndexOp>(loc, ascendc::getTypeSize(widest)).getResult();
  // 一段 lane 装多少个元素 = GetVecLen() / sizeof(最宽的那个)。与 legacy 的
  // `ELEMENT_PER_VECTOR_LENGTH`（`SIZE_OF_DTYPE = sizeof(max_dtype)`）逐字同义。
  Value elementsPerVector = builder.create<arith::DivUIOp>(loc, vectorLength, elementBytes).getResult();
  // 向上取整而不是截断：尾段那一趟由 mask 挡住多出来的 lane，不发就会漏算尾巴。
  //
  // 写成 `(count + elems - 1) / elems` 而不是一条 ceildiv，理由是发射器：
  // `arith.ceildivui` 没有打印（走到发射那一步才报 unable to find printer），
  // 而 `index.ceildivu` 有打印但 `ascir-translate` 没注册 `index` 方言，lit 里解析不了。
  // 这个写法与 legacy 的 `(element_count + ELEMENT_PER_VECTOR_LENGTH - 1) /
  // ELEMENT_PER_VECTOR_LENGTH` 逐字同形。
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1).getResult();
  Value lastLane = builder.create<arith::SubIOp>(loc, elementsPerVector, one).getResult();
  Value roundedUp = builder.create<arith::AddIOp>(loc, elementCount, lastLane).getResult();
  Value loopCount = builder.create<arith::DivUIOp>(loc, roundedUp, elementsPerVector).getResult();

  // `UpdateMask` 吃的是一个**可变计数器的引用**（`UpdateMask(uint32_t&)`），每调一次它自己
  // 减一段。所以这里建的是变量而不是值——legacy 那边是 `uint32_t sreg_0 = element_count;`。
  // 变量的元素类型写 `ui32`、初值直接给那个 index 值：发射出来是
  // `uint32_t v[1]{<count>}`，而 `UpdateMask(uint32_t&)` 收的正是 `*v`。
  // **不插 `arith.index_castui`**：那个算子的结果必须是 signless，给不出 `ui32`。
  auto counterType = MemRefType::get({1}, builder.getIntegerType(32U, false));
  Value counter = builder.create<emitasc::VariableOp>(loc, counterType, TypedAttr{}, elementCount).getResult();

  auto vecScope = builder.create<emitasc::VecScopeOp>(loc);
  Region &scopeRegion = vecScope.getRegion();
  Block *scopeBlock = scopeRegion.empty() ? builder.createBlock(&scopeRegion) : &scopeRegion.front();
  builder.setInsertionPointToEnd(scopeBlock);

  // 寄存器声明放在 `__VEC_SCOPE__` 里、循环外——与 legacy 的产物同形（循环里重复声明同一个
  // 寄存器在 C++ 上也合法，但那样每趟都多一条声明，产物与 legacy 对不起来）。
  //
  // **每个值按自己的元素类型声明**，不按 lane 的宽度：`RegTensor<half>` 与
  // `RegTensor<float>` 是同一个物理寄存器的两种读法，而 `Reg::Cast<OutT, InT, ...>` 要的正是
  // 两个类型各自对得上的那两个引用。
  const auto regTensorFor = [&](Type elementType) { return ascendc::RegTensorType::get(ctx, elementType); };
  llvm::DenseMap<Value, Value> regForValue;
  SmallVector<Value> sourceRegs;
  sourceRegs.reserve(op.getSrcs().size());
  for (BlockArgument argument : op.getBodyBlock().getArguments()) {
    Value reg =
        builder.create<ascendc::RegTensorOp>(loc, regTensorFor(getElementTypeOrSelf(argument.getType()))).getResult();
    sourceRegs.push_back(reg);
    regForValue[argument] = reg;
  }
  SmallVector<Operation *> bodyOps;
  for (Operation &inner : op.getBodyBlock().without_terminator()) {
    // 产谓词的那一行声明的是 `MaskReg` 而不是 `RegTensor<T>`——与 legacy 的
    // `MicroApiTensor::Define()` 按 `init_as_mask_reg_` 分两种声明同一回事
    // （`micro_api_call.cpp:47` / `:51`）。能力层已经核过这个结果只被 Select 的 mask 消费。
    const RegOpMapping *mapping = findRegOpMapping(&inner);
    Value reg;
    if (mapping->result == RegOpResult::kPredicate) {
      reg = builder.create<ascendc::MaskRegOp>(loc, ascendc::MaskRegType::get(ctx)).getResult();
    } else {
      reg = builder.create<ascendc::RegTensorOp>(loc, regTensorFor(getElementTypeOrSelf(inner.getResult(0).getType())))
                .getResult();
    }
    regForValue[inner.getResult(0)] = reg;
    bodyOps.push_back(&inner);
  }

  auto loop = builder.create<emitasc::VFForOp>(loc, loopCount);
  builder.setInsertionPointToStart(loop.getBody());

  // `UpdateMask` 按 lane 的宽度算，而不是按 destination 的 dtype：一趟活着的 lane 数是
  // `VL/sizeof(最宽的那个)`。与 legacy 的 `UpdateMask<max_dtype_size_>` 同一条
  // （`vf_loop.cpp:397`）。
  Value mask =
      builder.create<ascendc::UpdateMaskOp>(loc, ascendc::MaskRegType::get(ctx), counter, TypeAttr::get(widest))
          .getResult();
  Value offset = builder.create<arith::MulIOp>(loc, loop.getInductionVar(), elementsPerVector).getResult();

  for (auto [src, address, reg] : llvm::zip(op.getSrcs(), sourceAddresses, sourceRegs)) {
    Value sourcePointer =
        builder.create<emitasc::PtrOffsetOp>(loc, address.getType(), address, IntegerAttr{}, offset).getResult();
    // `dist` 由（边界 dtype, lane 宽度）定；等宽时不带（缺席即 `DIST_NORM`，与 M1 的产物
    // 逐字相同）。能力层已经核过这一对撑得出来，所以这里的 `failed` 是 pass 自己的 bug。
    FailureOr<std::optional<ascendc::LoadDist>> dist =
        loadDistFor(cast<ShapedType>(src.getType()).getElementType(), widest);
    if (failed(dist)) {
      return op.emitOpError() << "source element type has no load distribution, which the capability layer accepted";
    }
    builder.create<ascendc::LoadAlignRegOp>(loc, reg, sourcePointer,
                                            *dist ? ascendc::LoadDistAttr::get(ctx, **dist) : ascendc::LoadDistAttr{});
  }

  for (Operation *inner : bodyOps) {
    const RegOpMapping *mapping = findRegOpMapping(inner);
    SmallVector<Value> operandRegs;
    operandRegs.reserve(inner->getNumOperands());
    for (Value operand : inner->getOperands()) {
      auto reg = regForValue.find(operand);
      if (reg == regForValue.end()) {
        return inner->emitOpError() << "operand is not produced inside the vector function";
      }
      operandRegs.push_back(reg->second);
    }
    RegOpOperands operands{regForValue[inner->getResult(0)], operandRegs, mask, {}};
    if (mapping->elementTypes == RegOpElementTypes::kConverting) {
      // 能力层已经核过这一对有 trait。
      operands.castTrait = *castTraitFor(getElementTypeOrSelf(inner->getOperand(0).getType()),
                                         getElementTypeOrSelf(inner->getResult(0).getType()));
    }
    mapping->build(builder, loc, operands);
  }

  Value yielded = op.getYield().getResults().front();
  auto resultReg = regForValue.find(yielded);
  if (resultReg == regForValue.end()) {
    return op.emitOpError() << "yielded value is not produced inside the vector function";
  }
  Value destinationPointer =
      builder
          .create<emitasc::PtrOffsetOp>(loc, destinationAddress->getType(), *destinationAddress, IntegerAttr{}, offset)
          .getResult();
  FailureOr<std::optional<ascendc::StoreDist>> storeDist = storeDistFor(destinationType, widest);
  if (failed(storeDist)) {
    return op.emitOpError() << "destination element type has no store distribution, which the capability layer "
                               "accepted";
  }
  builder.create<ascendc::StoreAlignRegOp>(
      loc, destinationPointer, resultReg->second, mask,
      *storeDist ? ascendc::StoreDistAttr::get(ctx, **storeDist) : ascendc::StoreDistAttr{});

  // `emitasc.vf_for` 的终结算子由它自己的 builder 补（`VFForOp::build` 里的
  // `ensureTerminator`），所以这里只补 `vec_scope` 那一个——后者没有自定义 builder，
  // region 建出来是空的。
  builder.setInsertionPointToEnd(scopeBlock);
  builder.create<emitasc::YieldOp>(loc);

  // 结果就是那块 destination：下游按结果值串依赖，而数据已经由上面的 store 写进去了。
  op.getResult(0).replaceAllUsesWith(destination);
  op.erase();
  return success();
}

// 模块级而不是函数级：流水线的这一相位是挂在模块上的 pass manager 上（与
// `ConvertAFIRToAscendC` 同一相位），函数级 pass 加进去会在运行时报
// 「trying to schedule a pass on an unsupported operation」。
struct AFIRVectorFuncToRegBasePass : public PassWrapper<AFIRVectorFuncToRegBasePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AFIRVectorFuncToRegBasePass)

  llvm::StringRef getArgument() const final {
    return "afir-vector-func-to-reg-base";
  }
  llvm::StringRef getDescription() const final {
    return "Lower afir.vector_func into a register-form vector scope";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<afir::AFIRDialect, arith::ArithDialect, ascendc::AscendCDialect, emitasc::EmitAscDialect,
                    func::FuncDialect, index::IndexDialect, memref::MemRefDialect>();
  }

  void runOnOperation() final {
    SmallVector<afir::VectorFuncOp> vectorFuncs;
    getOperation().walk([&](afir::VectorFuncOp op) { vectorFuncs.push_back(op); });
    for (afir::VectorFuncOp op : vectorFuncs) {
      if (failed(lowerVectorFunc(op))) {
        signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createAFIRVectorFuncToRegBasePass() {
  return std::make_unique<AFIRVectorFuncToRegBasePass>();
}

void registerAFIRVectorFuncToRegBasePass() {
  PassRegistration<AFIRVectorFuncToRegBasePass>();
}

}  // namespace mlir::ascend::kernel
