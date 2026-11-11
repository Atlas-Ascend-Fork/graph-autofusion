/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// `alignment_bytes` 的聚合规则。
//
// 这几条只能在 UT 里钉。 这个字段当前的实际取值只有 -1（没有要求）与 32
// （unary 算子的 tmp buffer，`AfirBuilder` 的 `kTmpBufferAlignmentBytes`），两个值上
// 「取 max」与「取最小公倍数」同解，而「取先看到的那个」又依赖 `values` 的遍历顺序——
// 于是 lit（`afir_memory_allocate_buffer_alignment_merge.mlir`）与 E2E
// （`load_abs_diamond_store`）都只钉得住「要求没被丢掉」这一半，钉不住规则本身。
// 直接调这个函数是唯一能把规则钉死的地方。
//
// 规则的含义：`alignment_bytes` 不是资源的身份，而是某个张量对这块存储字节数的要求
// （`alignResourceSize` 把 size 向上取到它的倍数）。多个张量共用一块存储时这块存储要同时
// 满足它们，所以是最小公倍数而不是最大值：48 与 32 的公共解是 96，取 48 会让 32 落空。

#include "dialect/AFIR/Analysis/ResourceIds.h"

#include <gtest/gtest.h>

namespace {

// 两边都没有要求：结果仍是「没有要求」，而不是退化成 1。下游 `alignResourceSize` 按 -1
// 原样返回 size、不生成向上取整，折成 1 会多生成一串恒等算术。
TEST(CanonicalResourceFactsTest, NoRequirementOnEitherSideStaysUnconstrained) {
  EXPECT_EQ(mlir::afir::aggregateAlignmentBytes(-1, -1), -1);
}

// 一侧有要求：要求必须活下来，两种次序都要。这一条是真正要命的那个方向——丢掉之后
// 合并出来的 TBuf 装不下带要求的那个成员，而且不报错。
// 「取先看到的那个」会在第二组上给出 -1，于是这一条把它判死了（lit 做不到：那里带 32 的
// descriptor 正好先注册）。
TEST(CanonicalResourceFactsTest, SingleRequirementSurvivesEitherOrder) {
  EXPECT_EQ(mlir::afir::aggregateAlignmentBytes(32, -1), 32);
  EXPECT_EQ(mlir::afir::aggregateAlignmentBytes(-1, 32), 32);
}

// 两侧都有要求且互不整除：取最小公倍数。这一条把「取 max」判死了——取 max 会给 48，
// 而 48 不是 32 的倍数，32 那一条要求就落空了。
// 今天 `alignment_bytes` 取不到 48，所以这一条防的是取值变多之后，而不是现在的某张图。
TEST(CanonicalResourceFactsTest, TwoRequirementsTakeLeastCommonMultiple) {
  EXPECT_EQ(mlir::afir::aggregateAlignmentBytes(48, 32), 96);
  EXPECT_EQ(mlir::afir::aggregateAlignmentBytes(32, 48), 96);
}

// 一侧整除另一侧：结果是较大的那个，取 max 与取最小公倍数在这里同解。
// 单独列出来是因为这正是目前唯一会出现的形状（-1/32，以及将来可能的 32/64），
// 看到上面那条 48 的断言时容易以为聚合总会放大。
TEST(CanonicalResourceFactsTest, DivisibleRequirementsKeepTheCoarserOne) {
  EXPECT_EQ(mlir::afir::aggregateAlignmentBytes(64, 32), 64);
  EXPECT_EQ(mlir::afir::aggregateAlignmentBytes(32, 32), 32);
}

// `alignment_bytes` **不在**相等清单里：只差这一个字段的两份 descriptor 必须判成「一致」，
// 否则合法调度被拒（unary 算子的 tmp buffer 与某个中间结果落到同一个 `b<id>` 就是这个
// 形状）。这一条与上面的聚合是一对：清单放过去、聚合把要求收起来，少一半都不对。
TEST(CanonicalResourceFactsTest, AlignmentIsNotPartOfResourceIdentity) {
  mlir::afir::CanonicalResourceFacts tmpBuffer;
  tmpBuffer.space = mlir::afir::Space::UB;
  tmpBuffer.lifeTimeAxisId = -1;
  tmpBuffer.sizeFromResource = true;

  mlir::afir::CanonicalResourceFacts computeBuffer = tmpBuffer;
  EXPECT_TRUE(mlir::afir::canonicalResourceFactsAgree(tmpBuffer, computeBuffer));

  // `queue_depth` 仍然在清单里：它说的是这条队列轮转几块，是资源自己的属性，两个成员各说
  // 一个数就是矛盾。拿它做对照，免得这一条退化成「什么都判一致」也能绿。
  computeBuffer.queueDepth = 2;
  EXPECT_FALSE(mlir::afir::canonicalResourceFactsAgree(tmpBuffer, computeBuffer));
}

}  // namespace
