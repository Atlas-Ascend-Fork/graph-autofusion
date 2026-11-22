/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 分段落成 IR 的覆盖。
//
// 断言的是结构不是文本：这里要钉的是「几层循环」「偏移是哪几个值乘加出来的」
// 「窗口 extent 取的是哪一段」。这些在打印出来的文本里要靠正则去凑，而结构断言是直接的。
// 文本那一层由 LIT 在算子接通之后覆盖。
//
// Tail bindings are covered by pipeline tests; these tests use nominal extents.

#include "conversion/AFIRToAscendC/VectorizedLoops.h"

#include <string>

#include "dialect/AFIR/Analysis/VectorSegmentation.h"
#include "dialect/AFIR/IR/AFIR.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::ascend::kernel;

namespace {

class VectorizedLoopsTest : public testing::Test {
 protected:
  void SetUp() override {
    context.getOrLoadDialect<afir::AFIRDialect>();
    context.getOrLoadDialect<ascendc::AscendCDialect>();
    context.getOrLoadDialect<arith::ArithDialect>();
    context.getOrLoadDialect<func::FuncDialect>();
    context.getOrLoadDialect<memref::MemRefDialect>();
    context.getOrLoadDialect<scf::SCFDialect>();
  }

  // 一个带 descriptor 的 `afir.abs` 载体，外加两块 `local_tensor` 供偏移。
  // `rank` 要与 descriptor 的轴数一致（`AFIRDialect` 校验 axis 个数等于 shaped 秩）。
  bool Build(llvm::StringRef preamble, unsigned rank) {
    std::string dimensions;
    std::string argAxes;
    for (unsigned dimension = 0; dimension < rank; ++dimension) {
      dimensions += "?x";
      argAxes += (dimension == 0 ? "" : ", ") + std::to_string(dimension);
    }
    const std::string memrefType = "memref<" + dimensions + "f16, #afir.space<ub>>";
    const std::string source = std::string(
                                   "#tile = #afir.buffer_desc<space = <ub>, buffer_id = 0, "
                                   "life_time_axis_id = -1>\n#arg = #afir.tensor_desc<axis = [") +
                               argAxes + "], role = <compute>>\n" + preamble.str() + R"mlir(
func.func private @kernel(
    %in: )mlir" + memrefType + R"mlir( {afir.buffer_desc = #tile}, %out: )mlir" +
                               memrefType +
                               R"mlir( {afir.buffer_desc = #tile}) attributes {
  afir.function_role = #afir.function_role<kernel_body>,
  afir.arg_descs = [#arg, #arg]
} {
  %q = ascendc.queue : <vecin, 1>
  %ta = ascendc.que_bind.alloc_tensor %q : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<32xf16>
  %tb = ascendc.que_bind.alloc_tensor %q : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<32xf16>
  %0 = afir.abs %in outs(%out) {
    afir.operand_descs = [#src], afir.destination_descs = [#dst], afir.result_descs = [#dst]
  } : )mlir" + memrefType + ", " +
                               memrefType + " -> " + memrefType + R"mlir(
  return
}
)mlir";
    module = parseSourceString<ModuleOp>(source, &context);
    if (!module) {
      ADD_FAILURE() << "failed to parse:\n" << source;
      return false;
    }
    module->walk([&](Operation *op) {
      if (isa<afir::AbsOp>(op)) {
        carrier = op;
      }
      if (auto alloc = dyn_cast<ascendc::TQueBindAllocTensorOp>(op)) {
        localTensors.push_back(alloc.getResult());
      }
    });
    return carrier != nullptr && localTensors.size() == 2;
  }

  // 在载体处按标称范围物化分段，不做尾块替换。
  FailureOr<VectorizedWindow> Run(unsigned apiWindowSize) {
    FailureOr<afir::VectorPlan> plan = afir::collapseVectorizedAxes(carrier);
    if (failed(plan)) {
      ADD_FAILURE() << "collapseVectorizedAxes failed";
      return failure();
    }
    segmentCount = plan->segments.size();
    OpBuilder builder(carrier);
    return materializeVectorizedSegments(carrier, builder, *plan, afir::ExtentSource::Nominal, localTensors,
                                         apiWindowSize);
  }

  unsigned CountLoops() {
    unsigned count = 0;
    module->walk([&](scf::ForOp) { ++count; });
    return count;
  }

  MLIRContext context;
  OwningOpRef<ModuleOp> module;
  Operation *carrier = nullptr;
  SmallVector<Value> localTensors;
  size_t segmentCount = 0;
};

constexpr llvm::StringLiteral kTwoSegments = R"mlir(
#src = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"8">],
                         strides = [#afir.expr<"0">, #afir.expr<"1">],
                         vectorized_axis = [0, 1],
                         vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"4">, #afir.expr<"8">],
                         strides = [#afir.expr<"8">, #afir.expr<"1">],
                         vectorized_axis = [0, 1],
                         vectorized_strides = [#afir.expr<"8">, #afir.expr<"1">], role = <compute>>
)mlir";

constexpr llvm::StringLiteral kSingleSegment = R"mlir(
#src = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"128">], strides = [#afir.expr<"1">],
                         vectorized_axis = [0], vectorized_strides = [#afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"128">], strides = [#afir.expr<"1">],
                         vectorized_axis = [0], vectorized_strides = [#afir.expr<"1">], role = <compute>>
)mlir";

// 单段：一层循环都不建，偏移落成常量 0，窗口 extent 就是那一段。
//
// 这一格是「当前的路径逐字不变」的锚点——今天单段生成的就是「常量 0 + subindex」。
TEST_F(VectorizedLoopsTest, SingleSegmentBuildsNoLoop) {
  ASSERT_TRUE(Build(kSingleSegment, 1));
  auto window = Run(/*apiWindowSize=*/1);
  ASSERT_TRUE(succeeded(window));
  EXPECT_EQ(segmentCount, 1U);
  EXPECT_EQ(CountLoops(), 0U);
  ASSERT_EQ(window->tiles.size(), 2U);
  ASSERT_EQ(window->extents.size(), 1U);

  // 两块 tile 都生成了 subindex，偏移是常量 0。
  for (Value tile : window->tiles) {
    auto subIndex = tile.getDefiningOp<ascendc::LocalTensorSubIndexOp>();
    ASSERT_TRUE(subIndex);
    auto constant = subIndex->getOperand(1).getDefiningOp<arith::ConstantIndexOp>();
    ASSERT_TRUE(constant);
    EXPECT_EQ(constant.value(), 0);
  }
  auto extent = window->extents[0].getDefiningOp<arith::ConstantIndexOp>();
  ASSERT_TRUE(extent);
  EXPECT_EQ(extent.value(), 128);
}

// 两段、窗口一段：外段变一层循环，上界是外段 extent，窗口 extent 是内段。
TEST_F(VectorizedLoopsTest, OuterSegmentBecomesALoop) {
  ASSERT_TRUE(Build(kTwoSegments, 2));
  auto window = Run(/*apiWindowSize=*/1);
  ASSERT_TRUE(succeeded(window));
  EXPECT_EQ(segmentCount, 2U);
  ASSERT_EQ(CountLoops(), 1U);

  scf::ForOp loop;
  module->walk([&](scf::ForOp op) { loop = op; });
  auto upperBound = loop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  ASSERT_TRUE(upperBound);
  // 外段 extent 取非 1 的那一侧（dst 的 4），不是 src 的 1。
  EXPECT_EQ(upperBound.value(), 4);

  ASSERT_EQ(window->extents.size(), 1U);
  auto innerExtent = window->extents[0].getDefiningOp<arith::ConstantIndexOp>();
  ASSERT_TRUE(innerExtent);
  EXPECT_EQ(innerExtent.value(), 8);
}

// 偏移是 `iv × stride`，而且逐张量各取各的 stride：src 在广播段上 stride 是 0，
// dst 是 8。两块 tile 共用一个偏移的实现会让这一格红。
TEST_F(VectorizedLoopsTest, OffsetsUsePerTensorStrides) {
  ASSERT_TRUE(Build(kTwoSegments, 2));
  auto window = Run(/*apiWindowSize=*/1);
  ASSERT_TRUE(succeeded(window));
  ASSERT_EQ(window->tiles.size(), 2U);

  const int64_t expectedStrides[] = {0, 8};
  for (auto [index, tile] : llvm::enumerate(window->tiles)) {
    auto subIndex = tile.getDefiningOp<ascendc::LocalTensorSubIndexOp>();
    ASSERT_TRUE(subIndex) << "tile " << index;
    auto product = subIndex->getOperand(1).getDefiningOp<arith::MulIOp>();
    ASSERT_TRUE(product) << "tile " << index << " offset is not iv * stride";
    // 左边是归纳变量（循环体参数），右边是 stride 常量。
    EXPECT_TRUE(isa<BlockArgument>(product.getLhs())) << "tile " << index;
    auto stride = product.getRhs().getDefiningOp<arith::ConstantIndexOp>();
    ASSERT_TRUE(stride) << "tile " << index;
    EXPECT_EQ(stride.value(), expectedStrides[index]) << "tile " << index;
  }
}

// 窗口要两段（广播那一档）时，两段都进窗口、一层循环都不建。
TEST_F(VectorizedLoopsTest, TwoSegmentWindowLeavesNoLoop) {
  ASSERT_TRUE(Build(kTwoSegments, 2));
  auto window = Run(/*apiWindowSize=*/2);
  ASSERT_TRUE(succeeded(window));
  EXPECT_EQ(CountLoops(), 0U);
  ASSERT_EQ(window->extents.size(), 2U);
  auto outer = window->extents[0].getDefiningOp<arith::ConstantIndexOp>();
  auto inner = window->extents[1].getDefiningOp<arith::ConstantIndexOp>();
  ASSERT_TRUE(outer && inner);
  // 外→内。
  EXPECT_EQ(outer.value(), 4);
  EXPECT_EQ(inner.value(), 8);
}

// 段数少于窗口 → 失败，不是凑合生成。
TEST_F(VectorizedLoopsTest, FailsWhenWindowIsWiderThanThePlan) {
  ASSERT_TRUE(Build(kSingleSegment, 1));
  FailureOr<afir::VectorPlan> plan = afir::collapseVectorizedAxes(carrier);
  ASSERT_TRUE(succeeded(plan));
  OpBuilder builder(carrier);
  EXPECT_TRUE(
      failed(materializeVectorizedSegments(carrier, builder, *plan, afir::ExtentSource::Nominal, localTensors, 2)));
}

// `tiles` 与 `plan.tensors` 不等长 → 失败。少了这道检查，下面按下标取 stride 会越界读。
TEST_F(VectorizedLoopsTest, FailsOnTileCountMismatch) {
  ASSERT_TRUE(Build(kTwoSegments, 2));
  FailureOr<afir::VectorPlan> plan = afir::collapseVectorizedAxes(carrier);
  ASSERT_TRUE(succeeded(plan));
  OpBuilder builder(carrier);
  SmallVector<Value> tooFew{localTensors.front()};
  EXPECT_TRUE(failed(materializeVectorizedSegments(carrier, builder, *plan, afir::ExtentSource::Nominal, tooFew, 1)));
}

}  // namespace
