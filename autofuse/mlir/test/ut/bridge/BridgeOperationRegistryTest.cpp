/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- BridgeOperationRegistryTest.cpp -------------------------*- C++ -*-===//
//
// Pins AFIR_BRIDGE_OPERATION_REGISTRY as the one place that decides what the
// Schedule bridge can build. Adding an operation should mean a row here, an ODS
// definition and an AscendC implementation -- three places. These tests fail if
// a fourth appears: a second list of names, or a form predicate that answers
// something the table does not say.
//
//===----------------------------------------------------------------------===//

#include "bridge/AfirBuilder/AfirBuilder.h"
#include "bridge/ScheduleToAfir/ScheduleToAfir.h"

#include "conversion/AFIRToAscendC/Implementations/ElementwiseApiFamily.h"
#include "conversion/AFIRToAscendC/RegisterAscendCImplementations.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "target/Ascend/Capability/ImplementationRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include <gtest/gtest.h>

namespace mlir::afir::bridge {
namespace {

TEST(BridgeOperationRegistryTest, EveryRowNamesARegisteredAfirOperation) {
  MLIRContext context;
  context.getOrLoadDialect<AFIRDialect>();
  for (const AFIRBridgeOpMapping &mapping : getAFIRBridgeableOperations()) {
    EXPECT_TRUE(context.isOperationRegistered(mapping.afir_operation_name))
        << mapping.scheduler_name.str() << " maps to unregistered operation " << mapping.afir_operation_name.str();
  }
}

TEST(BridgeOperationRegistryTest, SchedulerNamesAreUnique) {
  llvm::StringSet<> seen;
  for (const AFIRBridgeOpMapping &mapping : getAFIRBridgeableOperations()) {
    EXPECT_TRUE(seen.insert(mapping.scheduler_name).second)
        << "duplicate scheduler name " << mapping.scheduler_name.str();
  }
}

// The buffer-form predicate must read the table's form column rather than
// carry its own list: before the column existed the two had already drifted
// (the table listed 13 operations, the predicate accepted 4).
TEST(BridgeOperationRegistryTest, BufferFormSupportFollowsTheFormColumn) {
  for (const AFIRBridgeOpMapping &mapping : getAFIRBridgeableOperations()) {
    const bool expected = mapping.form == BridgeOpForm::Buffer || mapping.form == BridgeOpForm::Both;
    EXPECT_EQ(isAFIRBufferFormOperationSupported(mapping.scheduler_name), expected) << mapping.scheduler_name.str();
  }
}

TEST(BridgeOperationRegistryTest, UnlistedOperationIsNotBuildable) {
  EXPECT_FALSE(isAFIRBufferFormOperationSupported("NoSuchScheduleNode"));
}

TEST(BridgeOperationRegistryTest, ArgMaxDialectOperationsAreNotRegistered) {
  MLIRContext context;
  context.getOrLoadDialect<AFIRDialect>();
  for (llvm::StringRef name : {"afir.argmax", "afir.argmax_phase1", "afir.argmax_phase2"}) {
    EXPECT_FALSE(context.isOperationRegistered(name)) << name.str();
  }
  for (llvm::StringRef name : {"ArgMax", "ArgMaxMultiRPhase1", "ArgMaxMultiRPhase2"}) {
    EXPECT_FALSE(isAFIRBufferFormOperationSupported(name)) << name.str();
  }
}

// The production path builds buffer form only, so this is the set that decides
// which fused graphs the MLIR codegen path can take at all. Kept as an explicit
// expectation so growing it is a deliberate, reviewed change rather than a
// side effect of touching the table.
TEST(BridgeOperationRegistryTest, BufferFormSetIsTheProductionCoverage) {
  llvm::StringSet<> bufferForm;
  for (const AFIRBridgeOpMapping &mapping : getAFIRBridgeableOperations()) {
    if (mapping.form == BridgeOpForm::Buffer || mapping.form == BridgeOpForm::Both) {
      bufferForm.insert(mapping.scheduler_name);
    }
  }
  EXPECT_EQ(bufferForm.size(), 124u);
  EXPECT_FALSE(bufferForm.contains("ArgMax"));
  EXPECT_FALSE(bufferForm.contains("ArgMaxMultiRPhase1"));
  EXPECT_FALSE(bufferForm.contains(
      "ArgMaxMultiRPhase2"));  // `Nddma` 是搬运族的第三个（2026-10-10）：搬入与广播融成一条指令，调度在 reg 侧把每个
  // `Load → Broadcast` 都换成它（`dialect/AFIR/IR/Ops/Data.td`）。它与 `Load` / `Store` 一样
  // **不进 `registeredFamilies()` 那张侧表**，所以下面那条按族覆盖的断言不覆盖它——理由写在
  // `Implementations/Nddma.cpp` 的注册处。
  //
  // `Ndtr` 是特殊函数一元族的第一个（2026-10-20），也是**第一个 legacy 只有 reg 侧有**的
  // op（mem 侧的 `ascir_builtin_ops_v1.cpp` 不注册它）。它进侧表，但只进 reg 那一列，
  // 由下面那条按族覆盖的断言连同 `kRegOnlyElementwiseOps` 一起管。
  //
  // `RemovePad` / `Transpose` / `Pad` 是布局变换族（2026-09-30 / 2026-10-15 / 2026-10-19）：
  // 同一组形状契约（Transpose 放宽为同 rank 重排），另一套 lowering，见
  // `dialect/AFIR/IR/Ops/Layout.td`。
  for (llvm::StringRef name : {"Load",
                               "Store",
                               "Broadcast",
                               "Add",
                               "Abs",
                               "Exp",
                               "Ln",
                               "Relu",
                               "Sqrt",
                               "Mul",
                               "Maximum",
                               "Minimum",
                               "Neg",
                               "Reciprocal",
                               "Sigmoid",
                               "Isnan",
                               "IsFinite",
                               "IsInf",
                               "Select",
                               "Where",
                               "ClipByValue",
                               "LeakyRelu",
                               "BitwiseAnd",
                               "FloorDiv",
                               "Remainder",
                               "Sub",
                               "Div",
                               "Gelu",
                               "Sign",
                               "LogicalAnd",
                               "LogicalOr",
                               "Axpy",
                               "TrueDiv",
                               "Tanh",
                               "Erf",
                               "Pow",
                               "Cast",
                               "Rsqrt",
                               "LogicalNot",
                               "Eq",
                               "Ne",
                               "Lt",
                               "Le",
                               "Gt",
                               "Ge",
                               "RemovePad",
                               "Transpose",
                               "Pad",
                               "Nddma",
                               "Sum",
                               "Max",
                               "Min",
                               "Mean",
                               "Prod",
                               "Any",
                               "All",
                               "Ndtr",
                               "Ndtri",
                               "LogNdtr",
                               "I0",
                               "AiryAi",
                               "BesselJ0",
                               "BesselJ1",
                               "I0e",
                               "I1e",
                               "ModifiedBesselI0",
                               "ModifiedBesselI1",
                               "ModifiedBesselK0",
                               "ModifiedBesselK1",
                               "LogicalXor",
                               "Log2",
                               "Exp2",
                               "Floor",
                               "Lgamma",
                               "Erfinv",
                               "Fmod",
                               "Mod",
                               "NextAfter",
                               "Acos",
                               "Acosh",
                               "Asin",
                               "Asinh",
                               "Atan",
                               "Atanh",
                               "Ceil",
                               "Cos",
                               "Cosh",
                               "Erfcx",
                               "Round",
                               "Sin",
                               "Sinh",
                               "Tan",
                               "BesselY0",
                               "BesselY1",
                               "SphericalBesselJ0",
                               "ScaledModifiedBesselK0",
                               "ScaledModifiedBesselK1",
                               "Digamma",
                               "Erfc",
                               "Trunc",
                               "Expm1",
                               "Log1p",
                               "CopySign",
                               "TruncDiv",
                               "Atan2",
                               "Zeta",
                               "Igamma",
                               "Igammac",
                               "Log10",
                               "Hypot",
                               "ChebyshevPolynomialT",
                               "ChebyshevPolynomialU",
                               "ChebyshevPolynomialV",
                               "ChebyshevPolynomialW",
                               "ShiftedChebyshevPolynomialT",
                               "ShiftedChebyshevPolynomialU",
                               "ShiftedChebyshevPolynomialV",
                               "ShiftedChebyshevPolynomialW",
                               "Ceil2Int",
                               "FloorToInt",
                               "Fma",
                               "BitwiseNot",
                               "BitwiseOr",
                               "BitwiseXor"}) {
    EXPECT_TRUE(bufferForm.contains(name)) << name.str();
  }
}

// bridge 说「这个 op 能建 buffer 形态」，conversion 就得有实现接得住。两个集合分在
// 两个目录里，谁也不引用谁，只有这条断言把它们绑在一起；不绑的话，多出来的那一个
// 要一路走到 conversion 才炸，报的还是「kernel 不完备」而不是「bridge 不该建它」。
TEST(BridgeOperationRegistryTest, BufferFormSetIsCoveredByAscendCImplementations) {
  // **今天没有豁免，所以这张表是空的。** 最后一行是 `Broadcast`，2026-09-23 接通常量源
  // 一档（`Implementations/Broadcast.cpp`）之后删掉。**注意它只覆盖 legacy 四条分支里的
  // 一条**——其余形态在 capability 里报 unsupported、整张图回落 legacy，而这条断言问的
  // 是「有没有实现」，答不到「覆盖了几条分支」。想让它答得到，就得让 registry 记下形态，
  // 那是另一件事。
  //
  // 下一个真的一条 pattern 都没有的 op 来的时候，把名字加回这张表，下面那段反向断言
  // 会一起管住它。
  const llvm::StringSet<> knownWithoutImplementation = {};

  MLIRContext context;
  context.getOrLoadDialect<AFIRDialect>();
  ascend::instruction::ImplementationRegistry registry(context);
  ASSERT_TRUE(succeeded(ascend::kernel::registerAscendCImplementations(registry)));
  ASSERT_TRUE(succeeded(registry.freeze()));

  for (const AFIRBridgeOpMapping &mapping : getAFIRBridgeableOperations()) {
    if (mapping.form == BridgeOpForm::Tensor || knownWithoutImplementation.contains(mapping.scheduler_name)) {
      continue;
    }
    OperationName sourceOp(mapping.afir_operation_name, &context);
    EXPECT_TRUE(registry.containsSourceOp(sourceOp))
        << mapping.scheduler_name.str() << " is buildable in buffer form but has no AscendC implementation";
  }

  // 豁免名单自己也要是真的：名字写错、或者实现补上了却忘了删这一行，都会让上面
  // 那个循环白白跳过一个本该检查的 op。
  for (llvm::StringRef name : knownWithoutImplementation.keys()) {
    const AFIRBridgeOpMapping *row = nullptr;
    for (const AFIRBridgeOpMapping &mapping : getAFIRBridgeableOperations()) {
      if (mapping.scheduler_name == name) {
        row = &mapping;
      }
    }
    ASSERT_NE(row, nullptr) << name.str() << " is exempted but not in the table";
    OperationName sourceOp(row->afir_operation_name, &context);
    EXPECT_FALSE(registry.containsSourceOp(sourceOp))
        << name.str() << " now has an implementation; drop it from the exemption list";
  }
}

// 上面那条断言答的是「有没有实现」，答不到「哪一侧」——`containsSourceOp` 对只注册了
// 一侧的 op 同样返回 true。两侧各是一行 `registerXxx`，删掉一行照样编得过，于是一个
// 在另一块目标上根本选不出实现的 op 不会让任何东西变红。
//
// 这一条按族补上：elementwise 的每个 op 要么有一个两侧通用的 implementation（`Any` 或
// `Shared`），要么 mem / reg 各至少一个。
//
// **两条断言是接力，不是重复**：一侧都没有的 op 根本不出现在家族侧表里，由上面那条的
// 豁免名单接住；一侧有一侧没有的，由这一条接住。
//
// 粒度是「每个 afir op」，不是「每个形态」。所以 `logical_or` 的标量形态只有 mem 侧
// 这件事这里查不出来——它的 tensor-tensor 形态两侧都在。那一条是 legacy 判据本身就
// 不让 reg 侧选中，写在 `BinaryElementwise.cpp` 的注册处。
TEST(BridgeOperationRegistryTest, EveryElementwiseOpCoversBothApiFamilies) {
  namespace elementwise = mlir::ascend::kernel::elementwise;

  // 曾经有过一行 `afir.masked_fill`（reg 侧没接），2026-09-23 查清原因不是缺供给、而是
  // `MaskedFillInputReorderPass` 在调度之前就把 `MaskedFill` 改写成了 `Select`，于是那个
  // op 连同 ODS、bridge 行与 mem 侧 implementation 一起删掉了，表也就空了。
  //
  // **2026-10-20 第一个真的只接得了一侧的 op 来了**，按上面那段原话办——加这张显式表加
  // 反向断言，下面那条循环一点没放宽。
  //
  // 收进来的理由必须是「legacy 那一侧根本没有这个算子」，不能是「那一侧还没接」：
  // 「还没接」的正确做法是接完再进表。两者的区别查得到——`ascir_op_lowering_matrix.csv`
  // 的 `mem_present` / `reg_present` 两列就是这件事，`0` 是没有、`1` 是有。
  struct RegOnlyRow {
    llvm::StringRef afirOpName;
    llvm::StringRef reason;
  };
  //
  // 特殊函数一元族的 5 个全是同一条理由，只是算子名不同：对照表那一行 `mem_present=0`,
  // 源只在 `v35/ascendc/api_regbase/` 下，mem 侧那张 `ascir_builtin_ops_v1.cpp` 里没有。
  constexpr llvm::StringRef kRegOnlyReason =
      "legacy 的 mem 侧 ascir_builtin_ops_v1.cpp 那 89 个注册里没有这个算子（对照表 mem_present=0、分档 D3），"
      "对应的 AscendC:: 标准调用也不存在；补一个 mem 侧 implementation 会让它在 mem 目标上被选中，"
      "发出一个根本不存在的调用";
  const RegOnlyRow kRegOnlyElementwiseOps[] = {
      {"afir.ndtr", kRegOnlyReason},
      {"afir.ndtri", kRegOnlyReason},
      {"afir.log_ndtr", kRegOnlyReason},
      {"afir.i0", kRegOnlyReason},
      {"afir.airy_ai", kRegOnlyReason},
      {"afir.bessel_j0", kRegOnlyReason},
      {"afir.bessel_j1", kRegOnlyReason},
      {"afir.i0e", kRegOnlyReason},
      {"afir.i1e", kRegOnlyReason},
      {"afir.modified_bessel_i0", kRegOnlyReason},
      {"afir.modified_bessel_i1", kRegOnlyReason},
      {"afir.modified_bessel_k0", kRegOnlyReason},
      {"afir.modified_bessel_k1", kRegOnlyReason},
      // 第 6 批。这十四个在 legacy 的 mem 侧（`ascir_builtin_ops_v1.cpp`）都没注册——注意
      // `BitwiseAnd` / `LogicalAnd` / `LogicalOr` 是两侧都有的，不能照它们看这一族。
      {"afir.logical_xor", kRegOnlyReason},
      {"afir.log2", kRegOnlyReason},
      {"afir.exp2", kRegOnlyReason},
      {"afir.floor", kRegOnlyReason},
      {"afir.lgamma", kRegOnlyReason},
      {"afir.erfinv", kRegOnlyReason},
      {"afir.fmod", kRegOnlyReason},
      {"afir.mod", kRegOnlyReason},
      {"afir.next_after", kRegOnlyReason},
      // 第 7 批的三十个，legacy 的 mem 侧同样一个都不注册。**`erfc` 与 `trunc` 不能照
      // `erf` 看**：`erf` 两侧都有供给（mem 侧标准调用、reg 侧包装），这两个只有 reg 侧。
      {"afir.acos", kRegOnlyReason},
      {"afir.acosh", kRegOnlyReason},
      {"afir.asin", kRegOnlyReason},
      {"afir.asinh", kRegOnlyReason},
      {"afir.atan", kRegOnlyReason},
      {"afir.atanh", kRegOnlyReason},
      {"afir.ceil", kRegOnlyReason},
      {"afir.cos", kRegOnlyReason},
      {"afir.cosh", kRegOnlyReason},
      {"afir.erfcx", kRegOnlyReason},
      {"afir.round", kRegOnlyReason},
      {"afir.sin", kRegOnlyReason},
      {"afir.sinh", kRegOnlyReason},
      {"afir.tan", kRegOnlyReason},
      {"afir.bessel_y0", kRegOnlyReason},
      {"afir.bessel_y1", kRegOnlyReason},
      {"afir.spherical_bessel_j0", kRegOnlyReason},
      {"afir.scaled_modified_bessel_k0", kRegOnlyReason},
      {"afir.scaled_modified_bessel_k1", kRegOnlyReason},
      {"afir.digamma", kRegOnlyReason},
      {"afir.erfc", kRegOnlyReason},
      {"afir.trunc", kRegOnlyReason},
      {"afir.expm1", kRegOnlyReason},
      {"afir.log1p", kRegOnlyReason},
      {"afir.copy_sign", kRegOnlyReason},
      {"afir.trunc_div", kRegOnlyReason},
      {"afir.atan2", kRegOnlyReason},
      {"afir.zeta", kRegOnlyReason},
      {"afir.igamma", kRegOnlyReason},
      {"afir.igammac", kRegOnlyReason},
      // 第 8 批的两个。`log10` 不能照 `log2` 之外的 `ln` 看、`hypot` 也不能照 `pow` 看：
      // 那两个两侧都有供给，这两个的 `mem_present` 是 0。
      {"afir.log10", kRegOnlyReason},
      {"afir.hypot", kRegOnlyReason},
      // 第 9 批的八个。多项式族在 legacy 的 mem 侧一个都没注册，而且**连对应的
      // `AscendC::` 标准调用都不存在**——只有 `v35/ascendc/api_regbase/` 下那份包装。
      {"afir.chebyshev_polynomial_t", kRegOnlyReason},
      {"afir.chebyshev_polynomial_u", kRegOnlyReason},
      {"afir.chebyshev_polynomial_v", kRegOnlyReason},
      {"afir.chebyshev_polynomial_w", kRegOnlyReason},
      {"afir.shifted_chebyshev_polynomial_t", kRegOnlyReason},
      {"afir.shifted_chebyshev_polynomial_u", kRegOnlyReason},
      {"afir.shifted_chebyshev_polynomial_v", kRegOnlyReason},
      {"afir.shifted_chebyshev_polynomial_w", kRegOnlyReason},
      // step 2a 的两个。取整到整数那一族在 legacy 的 mem 侧一个都没注册。**`ceil2int` 不能
      // 照 `afir.cast` 看**：cast 两侧都有供给，这两个只有 reg 侧。
      {"afir.ceil2int", kRegOnlyReason},
      {"afir.floor_to_int", kRegOnlyReason},
      // step 3。`AscendC::Fma` 这个标准调用是存在的（`adv_api/math/fma.h`），所以这一行
      // 的理由只成立前半句：`REG_ASC_IR(Fma)` 只在 v2 那张表上，mem 侧没有它。与
      // `afir.hypot` 同一种情况。
      {"afir.fma", kRegOnlyReason},
      // step 4。三个都只有 reg 一侧——mem 侧 `ascir_builtin_ops_v1.cpp` 一个都没注册。
      // **不能照 `afir.bitwise_and` 看**：and 两侧都有供给（mem 走包装、reg 走标准调用），
      // 这三个一个标准调用都没有，reg 侧也走包装。
      {"afir.bitwise_not", kRegOnlyReason},
      {"afir.bitwise_or", kRegOnlyReason},
      {"afir.bitwise_xor", kRegOnlyReason},
  };
  auto findRegOnly = [&](llvm::StringRef afirOpName) -> const RegOnlyRow * {
    for (const RegOnlyRow &row : kRegOnlyElementwiseOps) {
      if (row.afirOpName == afirOpName) {
        return &row;
      }
    }
    return nullptr;
  };

  MLIRContext context;
  context.getOrLoadDialect<AFIRDialect>();
  ascend::instruction::ImplementationRegistry registry(context);
  ASSERT_TRUE(succeeded(ascend::kernel::registerAscendCImplementations(registry)));
  ASSERT_TRUE(succeeded(registry.freeze()));

  struct Coverage {
    bool shared = false;
    bool mem = false;
    bool reg = false;
  };
  llvm::StringMap<Coverage> coverage;
  for (const elementwise::RegisteredFamily &row : elementwise::registeredFamilies()) {
    Coverage &entry = coverage[row.afirOpName];
    switch (row.family) {
      case elementwise::ApiFamily::Any:
      case elementwise::ApiFamily::Shared:
        entry.shared = true;
        break;
      case elementwise::ApiFamily::MemBased:
        entry.mem = true;
        break;
      case elementwise::ApiFamily::RegBased:
        entry.reg = true;
        break;
    }
  }
  // 侧表空了说明注册路径没走到这里，整条断言就退化成了恒真。
  ASSERT_FALSE(coverage.empty());

  for (const auto &entry : coverage) {
    const Coverage &sides = entry.getValue();
    // 表里那几行：不放过，改成对快照——缺的必须正好是 mem 那一侧。哪天 `Shared` 或 mem
    // 侧冒出来了，这里会红，逼着动的人回来重读 `reason` 还成不成立。
    if (const RegOnlyRow *regOnly = findRegOnly(entry.getKey())) {
      EXPECT_TRUE(sides.reg) << entry.getKey().str() << " 连 reg 侧都没有了：回 kRegOnlyElementwiseOps 重读 "
                             << regOnly->reason.str();
      EXPECT_FALSE(sides.mem || sides.shared)
          << entry.getKey().str() << " 现在两侧都有了，删掉 kRegOnlyElementwiseOps 里这一行让它回到下面那条判据";
      continue;
    }
    EXPECT_TRUE(sides.shared || (sides.mem && sides.reg))
        << entry.getKey().str() << " 只注册了一侧：要么把另一侧补上，要么按本用例上面那段说明加一张显式豁免表";
  }

  // 反向断言：表里的名字必须真的还在侧表里。名字写错、或者 op 整个去掉了却忘了删行的话，
  // 那一行就只是一段没人走得到的文本，而下一个同名 op 进来时它会替那个 op 挡着判据。
  for (const RegOnlyRow &row : kRegOnlyElementwiseOps) {
    EXPECT_TRUE(coverage.contains(row.afirOpName))
        << row.afirOpName.str() << " 已经不在 elementwise 侧表里了，删掉 kRegOnlyElementwiseOps 这一行";
  }
}

// 具名属性表与操作表是两张表，谁也不引用谁。一个节点类型在属性表里而不在操作表里
// （拼错、或者那个 op 后来被去掉了），bridge 会照常去读一个永远读不到的属性名，而
// 那条路只有跑到那个 op 的用例才走得到。
TEST(BridgeOperationRegistryTest, FloatIrAttrRowsNameABridgeableOperation) {
  for (const FloatIrAttrMapping &attr : getFloatIrAttrMappings()) {
    EXPECT_NE(findFloatIrAttr(attr.scheduler_name), nullptr) << attr.scheduler_name.str();
    bool listed = false;
    for (const AFIRBridgeOpMapping &mapping : getAFIRBridgeableOperations()) {
      listed = listed || mapping.scheduler_name == attr.scheduler_name;
    }
    EXPECT_TRUE(listed) << attr.scheduler_name.str() << " carries an ir_attr but is not a bridgeable operation";
    EXPECT_FALSE(attr.ascgraph_attr_name.empty()) << attr.scheduler_name.str();
    EXPECT_FALSE(attr.afir_attr_name.empty()) << attr.scheduler_name.str();
  }
  EXPECT_EQ(findFloatIrAttr("NoSuchScheduleNode"), nullptr);
  // 大多数 op 不带自有属性——这一条挡的是「表变成了对所有 op 都答是」。
  EXPECT_EQ(findFloatIrAttr("Add"), nullptr);
}

// int64 那张表的同一条判据。两张表是分开的，所以这条断言也得分开写——合起来写就得先
// switch 一次类型，而那正是当初不合表的理由。
TEST(BridgeOperationRegistryTest, IntIrAttrRowsNameABridgeableOperation) {
  EXPECT_FALSE(getIntIrAttrMappings().empty()) << "表空了说明这条断言退化成了恒真";
  for (const IntIrAttrMapping &attr : getIntIrAttrMappings()) {
    EXPECT_NE(findIntIrAttr(attr.scheduler_name), nullptr) << attr.scheduler_name.str();
    bool listed = false;
    for (const AFIRBridgeOpMapping &mapping : getAFIRBridgeableOperations()) {
      listed = listed || mapping.scheduler_name == attr.scheduler_name;
    }
    EXPECT_TRUE(listed) << attr.scheduler_name.str() << " carries an ir_attr but is not a bridgeable operation";
    EXPECT_FALSE(attr.ascgraph_attr_name.empty()) << attr.scheduler_name.str();
    EXPECT_FALSE(attr.afir_attr_name.empty()) << attr.scheduler_name.str();
  }
  EXPECT_EQ(findIntIrAttr("NoSuchScheduleNode"), nullptr);
  EXPECT_EQ(findIntIrAttr("Add"), nullptr);
  // **两张表不许收同一个节点类型**：`buildBufferNodeOp` 对两张表各挂一次属性，同一个
  // 节点同时在两张表上就会挂两个属性，而 `NodeInfo` 只带得了一个值——另一个必然是
  // `nullopt`，于是那个 op 永远在 bridge 上报「carries no ...」。
  for (const IntIrAttrMapping &intAttr : getIntIrAttrMappings()) {
    EXPECT_EQ(findFloatIrAttr(intAttr.scheduler_name), nullptr)
        << intAttr.scheduler_name.str() << " 同时在 float 与 int 两张 ir_attr 表上";
  }
}

}  // namespace
}  // namespace mlir::afir::bridge
