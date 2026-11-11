/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// `SymbolicBuilder` 的表达式构造工具。
//
// 为什么要钉住序列化出来的那个串：这些表达式要和 descriptor 里由调度器算好、经
// bridge 原样传进来的 stride 比较（`provablyContiguous` 先比规范串，比不出来才上证明器）。
// 两边的规范形式一旦不一致，判据会静默判否——表现成「布局证不出来、候选被拒」，而不是
// 报错。所以这里直接对照真实 descriptor 里出现过的字面串。

#include "dialect/AFIR/IR/SymbolicBuilder.h"

#include <string>

#include "dialect/AFIR/IR/AFIR.h"

#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::afir;

namespace {

class SymbolicBuilderTest : public testing::Test {
 protected:
  void SetUp() override {
    context.getOrLoadDialect<AFIRDialect>();
  }

  ExprAttr Expr(llvm::StringRef text) {
    FailureOr<ExprAttr> attr = makeExprAttr(&context, text);
    return succeeded(attr) ? *attr : ExprAttr();
  }

  std::string AlignUp(llvm::StringRef extent, int64_t alignNum) {
    ExprAttr value = Expr(extent);
    if (!value) {
      return "<bad extent>";
    }
    FailureOr<ExprAttr> aligned = alignUpExpr(&context, value, alignNum);
    if (failed(aligned)) {
      return "<failed>";
    }
    return aligned->getValue().getValue().str();
  }

  MLIRContext context;
};

// 这个串来自真实产物：`afir-to-ascendc-reduce-sum.mlir` 的 operand descriptor 里，A 轴的
// `vectorized_strides` 就是 `(8 * Ceiling((Rational(1 , 8) * z1t_size)))`。对不上就说明
// 我们构造的对齐式子和调度器的不是同一个规范形式。
TEST_F(SymbolicBuilderTest, AlignUpMatchesTheSchedulerSerialization) {
  EXPECT_EQ(AlignUp("z1t_size", 8), "(8 * Ceiling((Rational(1 , 8) * z1t_size)))");
}

// 常量能被 SymEngine 直接算掉，不该留下 `Ceiling`。
TEST_F(SymbolicBuilderTest, AlignUpFoldsConstantExtents) {
  EXPECT_EQ(AlignUp("8", 8), "8");
  EXPECT_EQ(AlignUp("5", 8), "8");
  EXPECT_EQ(AlignUp("17", 8), "24");
  EXPECT_EQ(AlignUp("32", 8), "32");
}

// `alignNum == 1` 时对齐是恒等变换，直接把原式带回去——不要绕一圈 Ceiling 产生一个
// 形状不同但语义相同的串，那会让规范串比较失效。
TEST_F(SymbolicBuilderTest, AlignUpToOneIsIdentity) {
  EXPECT_EQ(AlignUp("z1t_size", 1), "z1t_size");
}

TEST_F(SymbolicBuilderTest, AlignUpRejectsNonPositiveAlignment) {
  EXPECT_EQ(AlignUp("z1t_size", 0), "<failed>");
  EXPECT_EQ(AlignUp("z1t_size", -8), "<failed>");
  EXPECT_TRUE(failed(alignUpExpr(&context, ExprAttr(), 8)));
}

// 对齐后的式子必须能当 `provablyContiguous` 的 `innerExtent` 用：这是它唯一的消费形态
// ——「外侧 stride == 内侧 stride × 对齐后的内侧长度」。
TEST_F(SymbolicBuilderTest, AlignedExtentProvesContiguityAgainstSchedulerStride) {
  FailureOr<ExprAttr> aligned = alignUpExpr(&context, Expr("z1t_size"), 8);
  ASSERT_TRUE(succeeded(aligned));
  // A 轴 stride 取自真实 descriptor；R 最内轴 stride 为 1。
  EXPECT_TRUE(provablyContiguous(&context, Expr("(8 * Ceiling((Rational(1 , 8) * z1t_size)))"), Expr("1"), *aligned));
  // 没对齐的裸长度证不过去——这正是少了 `alignUpExpr` 时会发生的静默判否。
  EXPECT_FALSE(
      provablyContiguous(&context, Expr("(8 * Ceiling((Rational(1 , 8) * z1t_size)))"), Expr("1"), Expr("z1t_size")));
}

}  // namespace
