/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "dialect/AFIR/IR/AFIR.h"

#include <gtest/gtest.h>

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/AsmParser/AsmParser.h"

using namespace mlir;

namespace {

// `#afir.layout` 的契约不是它的文本形态，而是它给出的 affine map——LIT 只能断言前者。
// 本文件锁住后者：逻辑下标到物理下标空间的映射，以及 `MemRefLayoutAttrInterface` 的
// 其余几个查询（设计文档 3.1、ADR-16、5.1 步 6）。

class AFIRLayoutTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context.getOrLoadDialect<afir::AFIRDialect>();
  }

  afir::LayoutAttr layout(afir::LayoutOrder outer, afir::TileOrder tiling, ArrayRef<int64_t> tile, int64_t rank) {
    return afir::LayoutAttr::get(&context, outer, tiling, tile, rank);
  }

  // 把期望写成可读的 affine_map 文本再解析，比手工拼 AffineExpr 更像「断言」。
  AffineMap parseMap(StringRef text) {
    auto attr = dyn_cast_or_null<AffineMapAttr>(parseAttribute(text, &context));
    return attr ? attr.getAffineMap() : AffineMap();
  }

  MLIRContext context;
};

// ND：连续行优先，与不写 layout 等价。
TEST_F(AFIRLayoutTest, RowMajorUntiledIsIdentity) {
  afir::LayoutAttr nd = layout(afir::LayoutOrder::RowMajor, afir::TileOrder::None, {}, 2);
  EXPECT_EQ(nd.getAffineMap(), parseMap("affine_map<(d0, d1) -> (d0, d1)>"));
  EXPECT_TRUE(nd.isIdentity());
}

// DN：最后两维互换。
TEST_F(AFIRLayoutTest, ColMajorUntiledTransposesTrailingDims) {
  afir::LayoutAttr dn = layout(afir::LayoutOrder::ColMajor, afir::TileOrder::None, {}, 2);
  EXPECT_EQ(dn.getAffineMap(), parseMap("affine_map<(d0, d1) -> (d1, d0)>"));
  EXPECT_FALSE(dn.isIdentity());
}

// FRACTAL_NZ：外层按列（N）、块内按行（Z）。物理下标空间即 [N1][M1][M0][N0]。
TEST_F(AFIRLayoutTest, ColMajorGridRowMajorTileIsFractalNz) {
  afir::LayoutAttr nz = layout(afir::LayoutOrder::ColMajor, afir::TileOrder::RowMajor, {16, 16}, 2);
  EXPECT_EQ(nz.getAffineMap(),
            parseMap("affine_map<(d0, d1) -> (d1 floordiv 16, d0 floordiv 16, d0 mod 16, d1 mod 16)>"));
  EXPECT_FALSE(nz.isIdentity());
}

// FRACTAL_ZN：与 NZ 只差两个正交维的取值，网格与块内的顺序同时翻转。
TEST_F(AFIRLayoutTest, RowMajorGridColMajorTileIsFractalZn) {
  afir::LayoutAttr zn = layout(afir::LayoutOrder::RowMajor, afir::TileOrder::ColMajor, {16, 16}, 2);
  EXPECT_EQ(zn.getAffineMap(),
            parseMap("affine_map<(d0, d1) -> (d0 floordiv 16, d1 floordiv 16, d1 mod 16, d0 mod 16)>"));
}

// 外层与分块是两个独立的维：四种组合各自不同，这正是不用扁平枚举的理由。
TEST_F(AFIRLayoutTest, OuterAndTileOrdersAreIndependent) {
  AffineMap zz = layout(afir::LayoutOrder::RowMajor, afir::TileOrder::RowMajor, {16, 16}, 2).getAffineMap();
  AffineMap nz = layout(afir::LayoutOrder::ColMajor, afir::TileOrder::RowMajor, {16, 16}, 2).getAffineMap();
  AffineMap zn = layout(afir::LayoutOrder::RowMajor, afir::TileOrder::ColMajor, {16, 16}, 2).getAffineMap();
  AffineMap nn = layout(afir::LayoutOrder::ColMajor, afir::TileOrder::ColMajor, {16, 16}, 2).getAffineMap();
  EXPECT_NE(zz, nz);
  EXPECT_NE(zz, zn);
  EXPECT_NE(nz, nn);
  EXPECT_NE(zn, nn);
}

// 基块尺寸是参数：换 dtype 只换数字，不新增枚举项。
TEST_F(AFIRLayoutTest, TileExtentsAreParameters) {
  afir::LayoutAttr i8 = layout(afir::LayoutOrder::ColMajor, afir::TileOrder::RowMajor, {16, 32}, 2);
  EXPECT_EQ(i8.getAffineMap(),
            parseMap("affine_map<(d0, d1) -> (d1 floordiv 32, d0 floordiv 16, d0 mod 16, d1 mod 32)>"));
}

// 前导维原样透传：双缓冲的 stage 维不参与矩阵解释。
TEST_F(AFIRLayoutTest, LeadingDimsPassThrough) {
  afir::LayoutAttr staged = layout(afir::LayoutOrder::ColMajor, afir::TileOrder::RowMajor, {16, 16}, 3);
  EXPECT_EQ(staged.getAffineMap(),
            parseMap("affine_map<(d0, d1, d2) -> (d0, d2 floordiv 16, d1 floordiv 16, d1 mod 16, d2 mod 16)>"));
}

// memref 构造期就把 rank 与基块整除性钉死——布局进类型之后白拿的闸门。
TEST_F(AFIRLayoutTest, MemRefAcceptsAndRejectsLayoutByShape) {
  Type f16 = Float16Type::get(&context);
  afir::LayoutAttr nz = layout(afir::LayoutOrder::ColMajor, afir::TileOrder::RowMajor, {16, 16}, 2);

  EXPECT_TRUE(
      MemRefType::getChecked([&] { return emitError(UnknownLoc::get(&context)); }, {64, 32}, f16, nz, Attribute()));

  ScopedDiagnosticHandler handler(&context, [](Diagnostic &) { return success(); });
  // 基块除不尽。
  EXPECT_FALSE(
      MemRefType::getChecked([&] { return emitError(UnknownLoc::get(&context)); }, {64, 30}, f16, nz, Attribute()));
  // rank 不符。
  EXPECT_FALSE(
      MemRefType::getChecked([&] { return emitError(UnknownLoc::get(&context)); }, {2, 64, 32}, f16, nz, Attribute()));
}

// 动态维在编译期无从判断整除性，放行而不是误报。
TEST_F(AFIRLayoutTest, DynamicDimSkipsDivisibilityCheck) {
  Type f16 = Float16Type::get(&context);
  afir::LayoutAttr nz = layout(afir::LayoutOrder::ColMajor, afir::TileOrder::RowMajor, {16, 16}, 2);
  EXPECT_TRUE(MemRefType::getChecked([&] { return emitError(UnknownLoc::get(&context)); }, {ShapedType::kDynamic, 32},
                                     f16, nz, Attribute()));
}

// 布局与地址空间正交，各占 memref 的一个扩展点。
TEST_F(AFIRLayoutTest, LayoutAndMemorySpaceAreIndependent) {
  Type f16 = Float16Type::get(&context);
  afir::LayoutAttr nz = layout(afir::LayoutOrder::ColMajor, afir::TileOrder::RowMajor, {16, 16}, 2);
  auto type = MemRefType::get({64, 32}, f16, nz, afir::SpaceAttr::get(&context, afir::Space::L0A));
  EXPECT_EQ(afir::getLayoutAttr(type), nz);
  EXPECT_EQ(afir::getMemorySpace(type), afir::Space::L0A);

  MemRefType plain = afir::withLayout(type, {});
  EXPECT_FALSE(afir::getLayoutAttr(plain));
  EXPECT_TRUE(plain.getLayout().isIdentity());
  EXPECT_EQ(afir::getMemorySpace(plain), afir::Space::L0A);
}

// 取不到 LayoutAttr 不等于「没有布局」：null layout 与显式 identity 都表示连续 ND，
// 判据是 isIdentity()，不是「有没有 LayoutAttr」。
TEST_F(AFIRLayoutTest, GetLayoutAttrIsNotTheIdentityPredicate) {
  Type f16 = Float16Type::get(&context);
  auto implicit = MemRefType::get({64, 32}, f16);
  EXPECT_FALSE(afir::getLayoutAttr(implicit));
  EXPECT_TRUE(implicit.getLayout().isIdentity());

  auto explicitNd =
      MemRefType::get({64, 32}, f16, layout(afir::LayoutOrder::RowMajor, afir::TileOrder::None, {}, 2), Attribute());
  EXPECT_TRUE(afir::getLayoutAttr(explicitNd));
  EXPECT_TRUE(explicitNd.getLayout().isIdentity());
}

}  // namespace
