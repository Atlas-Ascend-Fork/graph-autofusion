/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 步 3（space 进 memref memory space）的访问器契约。
//
// 这几条断言看起来平凡，但它们锁的是「值的地址空间只从值的类型读」这条规则的边界：
// 读不到时必须返回"没有"，而不是编出一个默认值。整个 space 迁移的正确性依赖于
// 消费端能区分"这个值在 GM"和"这个值的类型没说它在哪"——把缺失折叠成 GM 会让
// 迁移期未改造的旧形态静默地被当成 GM，且不报错。

#include "dialect/AFIR/IR/AFIR.h"

#include <gtest/gtest.h>

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace {

class AFIRMemorySpaceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context.loadDialect<afir::AFIRDialect, memref::MemRefDialect>();
  }

  MLIRContext context;
};

TEST_F(AFIRMemorySpaceTest, ReadsEverySpaceFromTheType) {
  for (afir::Space space : {afir::Space::GM, afir::Space::UB, afir::Space::L1, afir::Space::L2, afir::Space::L0A,
                            afir::Space::L0B, afir::Space::L0C}) {
    MemRefType type = MemRefType::get({8}, Float16Type::get(&context), MemRefLayoutAttrInterface{},
                                      afir::SpaceAttr::get(&context, space));
    ASSERT_TRUE(afir::getMemorySpace(type).has_value());
    EXPECT_EQ(*afir::getMemorySpace(type), space);
    EXPECT_EQ(afir::getMemorySpaceAttr(type), afir::SpaceAttr::get(&context, space));
  }
}

// 缺失必须表现为"没有"。迁移前 bridge 产出的就是这种无 memory space 的 memref。
TEST_F(AFIRMemorySpaceTest, MissingSpaceIsNotDefaultedToGm) {
  MemRefType type = MemRefType::get({8}, Float16Type::get(&context));
  EXPECT_FALSE(afir::getMemorySpace(type).has_value());
  EXPECT_FALSE(afir::getMemorySpaceAttr(type));
}

// 其他方言的地址空间编码（AscendC 物化后的 CCE-C 魔数）不是 AFIR 的 space，
// 必须表现为"读不到"，而不是被当成某个 AFIR 取值。
TEST_F(AFIRMemorySpaceTest, ForeignMemorySpaceIsNotAnAfirSpace) {
  Attribute foreign = IntegerAttr::get(IntegerType::get(&context, 32), 22);
  MemRefType type = MemRefType::get({8}, Float16Type::get(&context), MemRefLayoutAttrInterface{}, foreign);
  EXPECT_FALSE(afir::getMemorySpace(type).has_value());
  EXPECT_FALSE(afir::getMemorySpaceAttr(type));
}

// tensor 形态没有地址空间可言。
TEST_F(AFIRMemorySpaceTest, TensorTypeHasNoSpace) {
  EXPECT_FALSE(afir::getMemorySpace(RankedTensorType::get({8}, Float16Type::get(&context))).has_value());
}

// withMemorySpace 只换 space，shape / 元素类型 / layout 原样保留。
TEST_F(AFIRMemorySpaceTest, WithMemorySpaceKeepsShapeElementTypeAndLayout) {
  auto layout = StridedLayoutAttr::get(&context, 16, {8, 1});
  MemRefType source = MemRefType::get({4, 8}, Float16Type::get(&context), layout);

  MemRefType result = afir::withMemorySpace(source, afir::Space::L1);

  EXPECT_EQ(result.getShape(), source.getShape());
  EXPECT_EQ(result.getElementType(), source.getElementType());
  EXPECT_EQ(result.getLayout(), source.getLayout());
  ASSERT_TRUE(afir::getMemorySpace(result).has_value());
  EXPECT_EQ(*afir::getMemorySpace(result), afir::Space::L1);
}

// 换 space 是可覆盖的：已有 space 的类型换成另一个 space，不会叠加也不会被忽略。
TEST_F(AFIRMemorySpaceTest, WithMemorySpaceReplacesAnExistingSpace) {
  MemRefType source = MemRefType::get({8}, Float16Type::get(&context), MemRefLayoutAttrInterface{},
                                      afir::SpaceAttr::get(&context, afir::Space::GM));

  MemRefType result = afir::withMemorySpace(source, afir::Space::UB);

  ASSERT_TRUE(afir::getMemorySpace(result).has_value());
  EXPECT_EQ(*afir::getMemorySpace(result), afir::Space::UB);
}

}  // namespace
