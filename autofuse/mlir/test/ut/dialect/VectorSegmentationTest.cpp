/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 向量化轴合并（阶段 2 合轴）的覆盖。
//
// **为什么是 UT 而不是 LIT**：`collapseVectorizedAxes` 的产物是一个结构体，不是 IR。
// LIT 只能隔着下降结果间接看它——段数变了会表现成循环层数变了，但「段的 stride 取的是
// 最内轴还是最外轴」「extent 是推出来的还是挑出来的」这类判据在输出文本里没有形状。
// 这里逐条构造判据的正反两侧。
//
// **载体用真的 AFIR op**：`afir.operand_descs` / `afir.destination_descs` 的长度由
// `AFIRDialect.cpp:533` `getExpectedDescriptorCount` 按 `getOperandPartition` 的
// `inputCount` / `destinationCount` 校验，所以这两个属性挂不到任意 op 上。
// 一元用 `afir.abs`（1 进 1 目的），二元用 `afir.add`，另一侧角色不是 `compute` 的形态
// 用 `afir.store`。descriptor 本身照 `AFIRElementwiseContractTest` 的 buffer 形态模板写。

#include "dialect/AFIR/Analysis/VectorSegmentation.h"

#include <string>

#include "dialect/AFIR/IR/AFIR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::afir;

namespace {

class VectorSegmentationTest : public testing::Test {
 protected:
  void SetUp() override {
    context.getOrLoadDialect<AFIRDialect>();
    context.getOrLoadDialect<func::FuncDialect>();
    context.getOrLoadDialect<memref::MemRefDialect>();
  }

  // `preamble` 放 `#src` / `#dst` 这些 descriptor 别名，`body` 是函数体里那一条 AFIR op。
  // `rank` 是载体 memref 的秩——`AFIRDialect` 校验「descriptor 的 axis 个数必须等于 shaped
  // 秩」，所以它必须跟用例里 descriptor 的轴数一致。维度用全动态（`?`），这样 descriptor
  // 的 repeats 写字面量还是符号都不受 memref 形状牵连。
  FailureOr<VectorPlan> Collapse(llvm::StringRef preamble, llvm::StringRef body, unsigned rank = 1) {
    std::string dimensions;
    std::string argAxes;
    for (unsigned dimension = 0; dimension < rank; ++dimension) {
      dimensions += "?x";
      argAxes += (dimension == 0 ? "" : ", ") + std::to_string(dimension);
    }
    const std::string memrefType = "memref<" + dimensions + "f16, #afir.space<ub>>";
    const std::string source = std::string(R"mlir(
#tile = #afir.buffer_desc<space = <ub>, buffer_id = 0, life_time_axis_id = -1>
#arg = #afir.tensor_desc<axis = [)mlir") +
                               argAxes + "], role = <compute>>\n" + preamble.str() +
                               "\nfunc.func private @kernel(\n"
                               "    %lhs: " +
                               memrefType +
                               " {afir.buffer_desc = #tile},\n"
                               "    %rhs: " +
                               memrefType +
                               " {afir.buffer_desc = #tile},\n"
                               "    %out: " +
                               memrefType +
                               " {afir.buffer_desc = #tile}) attributes {\n"
                               "  afir.function_role = #afir.function_role<kernel_body>,\n"
                               "  afir.arg_descs = [#arg, #arg, #arg]\n} {\n" +
                               body.str() + "  return\n}\n";
    module = parseSourceString<ModuleOp>(source, &context);
    if (!module) {
      ADD_FAILURE() << "failed to parse:\n" << source;
      return failure();
    }
    Operation *carrier = nullptr;
    module->walk([&](Operation *op) {
      if (op->hasAttr(kOperandDescsAttrName)) {
        carrier = op;
      }
    });
    if (carrier == nullptr) {
      ADD_FAILURE() << "no carrier op";
      return failure();
    }
    return collapseVectorizedAxes(carrier);
  }

  static std::string MemRef(unsigned rank) {
    std::string dimensions;
    for (unsigned dimension = 0; dimension < rank; ++dimension) {
      dimensions += "?x";
    }
    return "memref<" + dimensions + "f16, #afir.space<ub>>";
  }

  // 一元载体：一个操作数、一个目的地。
  static std::string Unary(llvm::StringRef operandAlias, llvm::StringRef destinationAlias, unsigned rank = 1) {
    const std::string type = MemRef(rank);
    return "  %0 = afir.abs %lhs outs(%out) {afir.operand_descs = [" + operandAlias.str() +
           "], afir.destination_descs = [" + destinationAlias.str() + "], afir.result_descs = [" +
           destinationAlias.str() + "]} : " + type + ", " + type + " -> " + type + "\n";
  }

  // 二元载体：两个操作数、一个目的地。用来看 operand 侧的下标记账。
  static std::string Binary(llvm::StringRef firstAlias, llvm::StringRef secondAlias, llvm::StringRef destinationAlias,
                            unsigned rank = 1) {
    const std::string type = MemRef(rank);
    return "  %0 = afir.add %lhs, %rhs outs(%out) {afir.operand_descs = [" + firstAlias.str() + ", " +
           secondAlias.str() + "], afir.destination_descs = [" + destinationAlias.str() + "], afir.result_descs = [" +
           destinationAlias.str() + "]} : (" + type + ", " + type + "), " + type + " -> " + type + "\n";
  }

  static std::string Text(ExprAttr expression) {
    return expression.getValue().getValue().str();
  }

  MLIRContext context;
  OwningOpRef<ModuleOp> module;
};

// 一根向量化轴：一段，extent 就是那根轴的 extent。
TEST_F(VectorSegmentationTest, SingleAxisYieldsSingleSegment) {
  auto plan = Collapse(R"mlir(
#t = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"128">], strides = [#afir.expr<"1">],
                       vectorized_axis = [0], vectorized_strides = [#afir.expr<"1">], role = <compute>>
)mlir",
                       Unary("#t", "#t"));
  ASSERT_TRUE(succeeded(plan));
  ASSERT_EQ(plan->segments.size(), 1U);
  EXPECT_EQ(plan->segments[0].axes, SmallVector<int64_t>{0});
  EXPECT_EQ(Text(plan->segments[0].extent), "128");
  // 两个张量（operand 与 destination 各一份），所以两个 stride。
  ASSERT_EQ(plan->tensors.size(), 2U);
  ASSERT_EQ(plan->segments[0].strides.size(), 2U);
  EXPECT_EQ(Text(plan->segments[0].strides[0]), "1");
}

// 两根连续轴：`stride[外] == stride[内] × extent[内]` 对两个张量都成立，合成一段，
// extent 是乘积。
TEST_F(VectorSegmentationTest, MergesContiguousAxes) {
  auto plan = Collapse(R"mlir(
#t = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"4">, #afir.expr<"32">],
                       strides = [#afir.expr<"32">, #afir.expr<"1">],
                       vectorized_axis = [0, 1],
                       vectorized_strides = [#afir.expr<"32">, #afir.expr<"1">], role = <compute>>
)mlir",
                       Unary("#t", "#t", 2), 2);
  ASSERT_TRUE(succeeded(plan));
  ASSERT_EQ(plan->segments.size(), 1U);
  EXPECT_EQ(plan->segments[0].axes, (SmallVector<int64_t>{0, 1}));
  EXPECT_EQ(Text(plan->segments[0].extent), "128");
  // 段的 stride 取**最内**轴的，不是最外轴的。取错会变成 32。
  EXPECT_EQ(Text(plan->segments[0].strides[0]), "1");
}

// 核心判据：只要**一个** operand 否决就封段。
//
// 这就是 `afir-broadcast-multi-axis-unsupported.mlir` 里那对 descriptor：dst 侧
// `8 == 1 × 8` 成立，src 侧 `0 == 1 × 8` 不成立，所以不能合。少了「所有张量都要同意」
// 这一条，这里会合成一段，下降出来的 `Broadcast` 会把广播轴当普通轴算。
TEST_F(VectorSegmentationTest, OneDissentingOperandClosesTheSegment) {
  auto plan = Collapse(R"mlir(
#src = #afir.tensor_desc<axis = [2, 5], repeats = [#afir.expr<"1">, #afir.expr<"8">],
                         strides = [#afir.expr<"0">, #afir.expr<"1">],
                         vectorized_axis = [2, 5],
                         vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [2, 5], repeats = [#afir.expr<"zt_size">, #afir.expr<"8">],
                         strides = [#afir.expr<"8">, #afir.expr<"1">],
                         vectorized_axis = [2, 5],
                         vectorized_strides = [#afir.expr<"8">, #afir.expr<"1">], role = <compute>>
)mlir",
                       Unary("#src", "#dst", 2), 2);
  ASSERT_TRUE(succeeded(plan));
  ASSERT_EQ(plan->segments.size(), 2U);
  // 外段是广播段：extent 取非 1 的那一侧（dst 的 zt_size），src 侧 stride 为 0。
  EXPECT_EQ(plan->segments[0].axes, SmallVector<int64_t>{2});
  EXPECT_EQ(Text(plan->segments[0].extent), "zt_size");
  EXPECT_EQ(Text(plan->segments[0].strides[0]), "0");
  EXPECT_EQ(Text(plan->segments[0].strides[1]), "8");
  // 内段是普通段。
  EXPECT_EQ(plan->segments[1].axes, SmallVector<int64_t>{5});
  EXPECT_EQ(Text(plan->segments[1].extent), "8");
  EXPECT_EQ(Text(plan->segments[1].strides[0]), "1");
}

// 异议者不在第一位也要生效。
//
// 上一格里不肯合的恰好是 `plan.tensors[0]`，所以「只看第一个张量」这种实现也能通过它。
// 这一格把关系倒过来：operand 侧连续（`8 == 1 × 8`），destination 侧不连续（stride 为 0），
// 异议来自 `plan.tensors[1]`。把判据从「所有张量」缩成「第一个张量」，只有这一格会红。
TEST_F(VectorSegmentationTest, ADissentingSecondTensorAlsoClosesTheSegment) {
  auto plan = Collapse(R"mlir(
#src = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"4">, #afir.expr<"8">],
                         strides = [#afir.expr<"8">, #afir.expr<"1">],
                         vectorized_axis = [0, 1],
                         vectorized_strides = [#afir.expr<"8">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"8">],
                         strides = [#afir.expr<"0">, #afir.expr<"1">],
                         vectorized_axis = [0, 1],
                         vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
)mlir",
                       Unary("#src", "#dst", 2), 2);
  ASSERT_TRUE(succeeded(plan));
  ASSERT_EQ(plan->segments.size(), 2U);
  // 异议者是第二个张量：它在外段上的 stride 是 0。
  ASSERT_EQ(plan->tensors.size(), 2U);
  EXPECT_EQ(Text(plan->segments[0].strides[0]), "8");
  EXPECT_EQ(Text(plan->segments[0].strides[1]), "0");
}

// 相邻的两根广播轴要能合并——`0 == 0 × e` 恒真。这不是特例，是同一条判据。
TEST_F(VectorSegmentationTest, MergesAdjacentBroadcastAxes) {
  auto plan = Collapse(R"mlir(
#src = #afir.tensor_desc<axis = [0, 1, 2],
                         repeats = [#afir.expr<"1">, #afir.expr<"1">, #afir.expr<"8">],
                         strides = [#afir.expr<"0">, #afir.expr<"0">, #afir.expr<"1">],
                         vectorized_axis = [0, 1, 2],
                         vectorized_strides = [#afir.expr<"0">, #afir.expr<"0">, #afir.expr<"1">],
                         role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1, 2],
                         repeats = [#afir.expr<"3">, #afir.expr<"4">, #afir.expr<"8">],
                         strides = [#afir.expr<"32">, #afir.expr<"8">, #afir.expr<"1">],
                         vectorized_axis = [0, 1, 2],
                         vectorized_strides = [#afir.expr<"32">, #afir.expr<"8">, #afir.expr<"1">],
                         role = <compute>>
)mlir",
                       Unary("#src", "#dst", 3), 3);
  ASSERT_TRUE(succeeded(plan));
  ASSERT_EQ(plan->segments.size(), 2U);
  EXPECT_EQ(plan->segments[0].axes, (SmallVector<int64_t>{0, 1}));
  EXPECT_EQ(Text(plan->segments[0].extent), "12");
  EXPECT_EQ(plan->segments[1].axes, SmallVector<int64_t>{2});
}

// extent 为 1 的轴整根丢掉，**而且丢轴发生在逐对比较之前**——否则它会把两侧本来连续的
// 两根轴割开。这里外轴与内轴本来可合，中间插一根 extent 为 1 的轴。
TEST_F(VectorSegmentationTest, DropsUnitAxisWithoutSplittingItsNeighbours) {
  auto plan = Collapse(R"mlir(
#t = #afir.tensor_desc<axis = [0, 1, 2],
                       repeats = [#afir.expr<"4">, #afir.expr<"1">, #afir.expr<"32">],
                       strides = [#afir.expr<"32">, #afir.expr<"32">, #afir.expr<"1">],
                       vectorized_axis = [0, 1, 2],
                       vectorized_strides = [#afir.expr<"32">, #afir.expr<"32">, #afir.expr<"1">],
                       role = <compute>>
)mlir",
                       Unary("#t", "#t", 3), 3);
  ASSERT_TRUE(succeeded(plan));
  ASSERT_EQ(plan->segments.size(), 1U);
  // 轴 1 不在列表里，轴 0 与轴 2 合成一段。
  EXPECT_EQ(plan->segments[0].axes, (SmallVector<int64_t>{0, 2}));
  EXPECT_EQ(Text(plan->segments[0].extent), "128");
}

// 符号 stride 可证相等时要能合：`z1t_size == 1 × z1t_size`。这一格钉住符号那条路真的通，
// 不是只会处理字面量。
TEST_F(VectorSegmentationTest, MergesSymbolicContiguousAxes) {
  auto plan = Collapse(R"mlir(
#t = #afir.tensor_desc<axis = [0, 1],
                       repeats = [#afir.expr<"4">, #afir.expr<"z1t_size">],
                       strides = [#afir.expr<"z1t_size">, #afir.expr<"1">],
                       vectorized_axis = [0, 1],
                       vectorized_strides = [#afir.expr<"z1t_size">, #afir.expr<"1">],
                       role = <compute>>
)mlir",
                       Unary("#t", "#t", 2), 2);
  ASSERT_TRUE(succeeded(plan));
  ASSERT_EQ(plan->segments.size(), 1U);
  EXPECT_EQ(plan->segments[0].axes, (SmallVector<int64_t>{0, 1}));
}

// 符号证不了就封段，不是就当相等。错误方向是多一层循环，不是算错地址。
TEST_F(VectorSegmentationTest, KeepsSegmentsApartWhenEqualityIsUnprovable) {
  auto plan = Collapse(R"mlir(
#t = #afir.tensor_desc<axis = [0, 1],
                       repeats = [#afir.expr<"4">, #afir.expr<"b">],
                       strides = [#afir.expr<"a">, #afir.expr<"1">],
                       vectorized_axis = [0, 1],
                       vectorized_strides = [#afir.expr<"a">, #afir.expr<"1">],
                       role = <compute>>
)mlir",
                       Unary("#t", "#t", 2), 2);
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(plan->segments.size(), 2U);
}

// 各张量谈论的不是同一串向量化轴 → 失败。两串不同的轴之间没有「相邻」可言。
TEST_F(VectorSegmentationTest, FailsOnMismatchedVectorizedAxisLists) {
  auto plan = Collapse(R"mlir(
#a = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"4">, #afir.expr<"8">],
                       strides = [#afir.expr<"8">, #afir.expr<"1">],
                       vectorized_axis = [0, 1],
                       vectorized_strides = [#afir.expr<"8">, #afir.expr<"1">], role = <compute>>
#b = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"4">, #afir.expr<"8">],
                       strides = [#afir.expr<"8">, #afir.expr<"1">],
                       vectorized_axis = [1],
                       vectorized_strides = [#afir.expr<"1">], role = <compute>>
)mlir",
                       Unary("#a", "#b", 2), 2);
  EXPECT_TRUE(failed(plan));
}

// 同一根轴上两个不同的非 1 extent → 失败，不挑一个用。广播那一侧是 1，所以这不是广播，
// 是矛盾的 descriptor；采用其中一个会让后面所有 stride 算式建立在猜测上。
TEST_F(VectorSegmentationTest, FailsOnConflictingExtents) {
  auto plan = Collapse(R"mlir(
#a = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"8">], strides = [#afir.expr<"1">],
                       vectorized_axis = [0], vectorized_strides = [#afir.expr<"1">], role = <compute>>
#b = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"16">], strides = [#afir.expr<"1">],
                       vectorized_axis = [0], vectorized_strides = [#afir.expr<"1">], role = <compute>>
)mlir",
                       Unary("#a", "#b"));
  EXPECT_TRUE(failed(plan));
}

// 每根向量化轴的 extent 都是 1：没有段，但不是失败。常量源广播的 `Duplicate` 一档自己
// 认这个形态。
TEST_F(VectorSegmentationTest, ReportsNoSegmentsWhenEveryExtentIsOne) {
  auto plan = Collapse(R"mlir(
#t = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"1">],
                       strides = [#afir.expr<"0">, #afir.expr<"0">],
                       vectorized_axis = [0, 1],
                       vectorized_strides = [#afir.expr<"0">, #afir.expr<"0">], role = <compute>>
)mlir",
                       Unary("#t", "#t", 2), 2);
  ASSERT_TRUE(succeeded(plan));
  EXPECT_TRUE(plan->segments.empty());
  EXPECT_FALSE(plan->tensors.empty());
}

// 角色过滤：只有 `compute` 的张量参与。Load 的 GM 侧角色是 `input`、没有向量化轴，
// 把它算进去会让「各张量同一串向量化轴」这条立刻不成立。
TEST_F(VectorSegmentationTest, IgnoresNonComputeDescriptors) {
  auto plan = Collapse(R"mlir(
#gm = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"128">], strides = [#afir.expr<"1">],
                        role = <input>>
#ub = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"128">], strides = [#afir.expr<"1">],
                        vectorized_axis = [0], vectorized_strides = [#afir.expr<"1">], role = <compute>>
)mlir",
                       Unary("#gm", "#ub"));
  ASSERT_TRUE(succeeded(plan));
  ASSERT_EQ(plan->tensors.size(), 1U);
  EXPECT_EQ(plan->tensors[0].side, VectorTensor::Side::Destination);
  ASSERT_EQ(plan->segments.size(), 1U);
  EXPECT_EQ(plan->segments[0].strides.size(), 1U);
}

// 标量操作数：角色是 `compute` 但一个向量化轴都没有，**跳过**而不是报失败。
//
// 调度给标量的 descriptor 就是这个形态（`repeats = [1, 1], strides = [0, 0]`，无
// `vectorized_axis`）。把它算进来会让「各张量同一串向量化轴」那条立刻不成立，
// 整个标量二元图就回落 legacy 了。
TEST_F(VectorSegmentationTest, SkipsScalarOperandsWithoutVectorizedAxes) {
  auto plan = Collapse(R"mlir(
#scalar = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"1">], strides = [#afir.expr<"0">],
                            role = <compute>>
#ub = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"128">], strides = [#afir.expr<"1">],
                        vectorized_axis = [0], vectorized_strides = [#afir.expr<"1">], role = <compute>>
)mlir",
                       Binary("#scalar", "#ub", "#ub"));
  ASSERT_TRUE(succeeded(plan));
  // 标量不在列表里：operand 侧只剩下标 1 那一个。
  ASSERT_EQ(plan->tensors.size(), 2U);
  EXPECT_EQ(plan->tensors[0].side, VectorTensor::Side::Operand);
  EXPECT_EQ(plan->tensors[0].index, 1U);
  EXPECT_EQ(plan->tensors[1].side, VectorTensor::Side::Destination);
  ASSERT_EQ(plan->segments.size(), 1U);
  EXPECT_EQ(plan->segments[0].strides.size(), 2U);
}

// 一个 `compute` 张量都没有 → 失败。没有片上侧就没有向量化轴可谈。
TEST_F(VectorSegmentationTest, FailsWithoutComputeDescriptors) {
  auto plan = Collapse(R"mlir(
#gm = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"128">], strides = [#afir.expr<"1">],
                        role = <input>>
)mlir",
                       Unary("#gm", "#gm"));
  EXPECT_TRUE(failed(plan));
}

// 张量出处要带上：下降侧按段的 stride 给对应那个 operand 发偏移，按数组下标硬对是错的。
TEST_F(VectorSegmentationTest, RecordsTensorProvenance) {
  auto plan = Collapse(R"mlir(
#gm = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"128">], strides = [#afir.expr<"1">],
                        role = <input>>
#ub = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"128">], strides = [#afir.expr<"1">],
                        vectorized_axis = [0], vectorized_strides = [#afir.expr<"1">], role = <compute>>
)mlir",
                       Binary("#gm", "#ub", "#ub"));
  ASSERT_TRUE(succeeded(plan));
  ASSERT_EQ(plan->tensors.size(), 2U);
  // operand 侧那个 compute 张量的下标是 1，不是 0——#gm 被角色过滤掉了，但它占着下标 0。
  EXPECT_EQ(plan->tensors[0].side, VectorTensor::Side::Operand);
  EXPECT_EQ(plan->tensors[0].index, 1U);
  EXPECT_EQ(plan->tensors[1].side, VectorTensor::Side::Destination);
  EXPECT_EQ(plan->tensors[1].index, 0U);
}

}  // namespace
