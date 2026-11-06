/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- LegacyGraphGuardSt.cpp - legacy 输入图的功能看护 ---------*- C++ -*-===//
//
// **搬的是输入图与形状，不是用例本体。** legacy 自己的 e2e 用例在
// `tests/st/backend_e2e`（mem）与 `tests/v35/st/backend_e2e_v2`（reg）里，一个用例三四个文件、
// 带自己的数值参考；它们的图全部来自 `tests/framework/share_graph` 的
// `ShareGraph::Xxx` 工厂。要的是那套覆盖面，所以这里只把**图和形状**按
// `AscGraphBuilder` 重写一遍，一个图一条用例，跑的是我们自己的那条链。
//
// 范围按接管表定，不按名字猜：reg 侧 200 格里 63 格、mem 侧 38 格里 **13** 格是
// `takeover`（2026-10-26 两张表），映射到 **74 个去重后的输入图**。`fallback` 那些 MLIR 今天
// 还收不了，放进来只会是红的，所以不放——它们是缺口清单，不是看护对象。
//
// （mem 那个数原先写的是 20，是错的：数表本身得到 13，见
// `docs/development/2026-10-26-legacy-takeover-table-mem-f4.txt`。接管表随后走到 reg 67，
// 重筛后是 78 个图 / 剩余 105，见 `2026-10-26-feature7-remaining-graphs-rescreen.txt`；
// 本文件的看护集合仍是当初那 74 个那一版，新增的 4 个未搬。）
//
// **为什么用严格模式**：允许回落的那一档（`--mlir_codegen_config=fallback`）在 MLIR 收不了时回落给
// legacy，`GenerateForInductor` 照样返回 SUCCESS——那样这条用例在「MLIR 不再接管这张图」时
// 仍然是绿的，看护不到任何东西。严格模式把回落也判成失败，于是 ASSERT SUCCESS 的含义变成
// 「这张图确实是 MLIR 发的」。
//
// **入口是 `GenerateForInductor`**，与接管表那边的 `Codegen::Generate`（kTensorFlow）不同：
// 场景这一位会传进 ATT 的候选准备闸门，而设计范围里的生产入口是 inductor 那个。接管表自己
// 记了这条口径限制，这里顺手把它对齐。
//
// 两侧的平台位走两条路，不要混：
//   mem  `CodegenOptions{.soc_version = "Ascend910B1"}`；
//   reg  `af::InstallV35SocPerfRuntimeStub()`（抄 `v35/st/backend_e2e_v2` 的 fixture），
//        一个 stub 把 soc 名与 NpuArch 一起带上，`CodegenOptions` 用缺省。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "codegen.h"
#include "common/platform_context.h"
#include "optimize.h"
// **两份 `runtime_stub.h` 里取 `st/common` 那一份**：接管表量的 `backend_e2e_v2` 按 include
// 路径解析到的就是它，而两份 `RuntimeStubV2` 分头演化过（补 `aclrtGetSocName` 那次只补了
// `v35/st/optimize` 那份，表因此整片量在了错的平台上）。取另一份会让这里的平台位与表对不上。
#include "st/common/runtime_stub.h"
#include "test/Support/TestEnvironment.h"
#include "tests/framework/easy_asc_graph/asc_graph_builder.h"

namespace {

using af::testing::AscGraphBuilder;
using af::testing::Sym;

// 一条看护用例的全部动作。`soc_version` 为空表示走 reg 侧那条（由 stub 供 soc 名）。
void ExpectMlirOwnsGraph(af::AscGraph graph, const std::string &soc_version) {
  mlir::test::ScopedEnvVar strict("AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=mlir");

  // **不设图级 tiling key。** 一个 stage 调度出多个模板时（`ContinuesBrc` 有 5 个），图级的
  // `AscGraph::SetTilingKey` 识别不了模板，产物装配当场报错并要求去掉它；不设的话每个模板的
  // key 默认取模板下标，正是想要的。
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused;
  ASSERT_EQ(optimizer.Optimize(graph, fused), af::SUCCESS);
  ASSERT_FALSE(fused.node_idx_to_scheduled_results.empty());

  codegen::CodegenOptions options;
  if (!soc_version.empty()) {
    options.soc_version = soc_version;
  }
  codegen::Codegen codegen(options);
  codegen::CodegenResult result;
  // 严格模式下这一条等价于「MLIR 接管了这张图」——回落会让它返回失败。
  ASSERT_EQ(codegen.GenerateForInductor(fused, result), af::SUCCESS);
  EXPECT_FALSE(result.kernel.empty());
  EXPECT_NE(result.kernel.find("_subgraph0_plan"), std::string::npos);
  EXPECT_FALSE(result.tiling.empty());
  EXPECT_FALSE(result.tiling_data.empty());
}

// Legacy optimizer success does not imply an MLIR candidate exists. Keep this
// boundary explicit instead of accepting a silent legacy fallback as migration.
void ExpectMlirRejection(af::AscGraph graph, const std::string &reason, const std::string &gate) {
  mlir::test::ScopedEnvVar strict("AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=mlir");
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused;
  ASSERT_EQ(optimizer.Optimize(graph, fused), af::SUCCESS);
  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen::CodegenResult result;
  mlir::test::ScopedStderrCapture capture;
  const auto status = codegen.GenerateForInductor(fused, result);
  const auto diagnostic = capture.Text();
  EXPECT_EQ(status, af::FAILED);
  EXPECT_NE(diagnostic.find(reason), std::string::npos) << diagnostic;
  EXPECT_NE(diagnostic.find("gate=" + gate), std::string::npos) << diagnostic;
}

// mem 侧（NpuArch 2201）。
class LegacyGraphGuardMemSt : public testing::Test {
 protected:
  void SetUp() override {
    ge::PlatformContext::GetInstance().Reset();
  }
};

// reg 侧（NpuArch 3510 / Ascend950PR）。fixture 抄 `v35/st/backend_e2e_v2` 的那一份。
class LegacyGraphGuardRegSt : public testing::Test {
 protected:
  void SetUp() override {
    ge::PlatformContext::GetInstance().Reset();
    af::InstallV35SocPerfRuntimeStub();
  }
  void TearDown() override {
    ge::RuntimeStub::Reset();
  }
};

//===----------------------------------------------------------------------===//
// 按接管表的 takeover 定侧：一个图在哪一侧被接管就在那一侧发用例，两侧都被接管就发两条。
//===----------------------------------------------------------------------===//

// legacy backend_e2e/sub_transpose_abs_test 与其 PGO 变体共用的输入图。
TEST_F(LegacyGraphGuardMemSt, SubTransposeAbs) {
  ExpectMlirOwnsGraph(AscGraphBuilder("sub_transpose_abs")
                          .Loops({32, 64, 512})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Data("data1", 1, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Load("load1", "data1")
                          .Sub("sub", "load0", "load1")
                          .Transpose("transpose", "sub", {1, 0, 2})
                          .Abs("abs", "transpose")
                          .Store("store", "abs")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      "Ascend910B1");
}

// legacy v35 optimize_st_v2.cpp 的同名图；只在 CPU 仿真环境生成。
TEST_F(LegacyGraphGuardRegSt, NddmaCaseTranspose021OutputWithSingleRef) {
  ExpectMlirOwnsGraph(AscGraphBuilder("transpose021")
                          .Loops({32, 64, 16})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Transpose("transpose", "load0", {0, 2, 1})
                          .Store("store", "transpose")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      "");
}

TEST_F(LegacyGraphGuardRegSt, LoadCastTransposeCase) {
  ExpectMlirOwnsGraph(AscGraphBuilder("cast_transpose")
                          .Loops({64, 64})
                          .Data("data0", 0, af::DT_FLOAT16)
                          .Load("load0", "data0")
                          .Cast("cast", "load0", af::DT_FLOAT)
                          .Transpose("transpose", "cast", {1, 0})
                          .Store("store", "transpose")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      "");
}

// Optimizer-only input, not a legacy Inductor migration case. The comparison
// layout is currently unsupported; keep its strict rejection visible.
TEST_F(LegacyGraphGuardRegSt, LoadGEWhereTransposeCaseLoweringBoundary) {
  const auto s0 = Sym(41);
  const auto s1 = Sym(54);
  const auto s2 = Sym(38);
  const auto s3 = Sym(55);
  ExpectMlirRejection(AscGraphBuilder("where_transpose")
                          .Loops({s0, s1, s2, s3})
                          .Data("data0", 0)
                          .Load("load0", "data0")
                          .Data("data1", 1)
                          .Load("load1", "data1")
                          .Ge("ge", "load0", "load1")
                          .Where("where", "ge", "load0", "load1")
                          .Transpose("transpose", "where", {0, 3, 1, 2})
                          .Store("store", "transpose")
                          .Output("output", "store", 0)
                          .Build(),
                      "'afir.ge' op has no supported Ascend implementation", "lowering");
}

TEST_F(LegacyGraphGuardRegSt, TransposeTwoAxisSplitCaseNeedInputAlignCandidateBoundary) {
  auto s0 = Sym(347);
  auto s2 = Sym(15);
  auto s3 = Sym(49);
  ExpectMlirRejection(
      AscGraphBuilder("transpose_broadcast_mul")
          .Loops({s0, s2, s3})
          .Data("data0", 0, af::DT_FLOAT)
          .Load("load0", "data0", {}, {s2 * s3, s2, af::sym::kSymbolOne})
          .Transpose("transpose0", "load0", {0, 2, 1})
          .Data("data1", 1, af::DT_FLOAT)
          .Load("load1", "data1", {s0, af::sym::kSymbolOne, s3}, {s3, af::sym::kSymbolZero, af::sym::kSymbolOne})
          .Broadcast("brc0", "load1", {1})
          .Mul("mul0", "brc0", "transpose0")
          .Store("store0", "mul0")
          .Output("output", "store0", 0, af::DT_FLOAT)
          .Build(),
      "elementwise-broadcast", "candidate-preparation");
}

// 对应 `ShareGraph::AcosBf16FusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, AcosBf16) {
  ExpectMlirOwnsGraph(AscGraphBuilder("acosbf16")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_BF16)
                          .Load("load0", "data0")
                          .Op<af::ascir_op::Acos>("acos", {"load0"})
                          .Store("store", "acos")
                          .Output("output", "store", 0, af::DT_BF16)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::AcoshBf16FusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, AcoshBf16) {
  ExpectMlirOwnsGraph(AscGraphBuilder("acoshbf16")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_BF16)
                          .Load("load0", "data0")
                          .Op<af::ascir_op::Acosh>("acosh", {"load0"})
                          .Store("store", "acosh")
                          .Output("output", "store", 0, af::DT_BF16)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::AddAbsFusedGraph(3)`。
TEST_F(LegacyGraphGuardMemSt, AddAbs) {
  ExpectMlirOwnsGraph(AscGraphBuilder("addabs")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Data("data1", 1, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Load("load1", "data1")
                          .Op<af::ascir_op::Add>("add", {"load0", "load1"})
                          .Op<af::ascir_op::Abs>("abs", {"add"})
                          .Store("store", "abs", {}, {})
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      "Ascend910B1");
}

// 对应 `ShareGraph::AddAbsFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, AddAbs) {
  ExpectMlirOwnsGraph(AscGraphBuilder("addabs")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Data("data1", 1, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Load("load1", "data1")
                          .Op<af::ascir_op::Add>("add", {"load0", "load1"})
                          .Op<af::ascir_op::Abs>("abs", {"add"})
                          .Store("store", "abs", {}, {})
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::AddExp2FusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, AddExp2) {
  ExpectMlirOwnsGraph(AscGraphBuilder("addexp2")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Data("data1", 1, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Load("load1", "data1")
                          .Op<af::ascir_op::Add>("add", {"load0", "load1"})
                          .Op<af::ascir_op::Exp2>("exp2", {"add"})
                          .Store("store", "exp2")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::AddFloorFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, AddFloor) {
  ExpectMlirOwnsGraph(AscGraphBuilder("addfloor")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Data("data1", 1, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Load("load1", "data1")
                          .Op<af::ascir_op::Add>("add", {"load0", "load1"})
                          .Op<af::ascir_op::Floor>("floor", {"add"})
                          .Store("store", "floor")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::AddGeluFusedGraph(3)`。
TEST_F(LegacyGraphGuardMemSt, AddGelu) {
  ExpectMlirOwnsGraph(AscGraphBuilder("addgelu")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Data("data1", 1, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Load("load1", "data1")
                          .Op<af::ascir_op::Add>("add", {"load0", "load1"})
                          .Op<af::ascir_op::Gelu>("gelu", {"add"})
                          .Store("store", "gelu")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      "Ascend910B1");
}

// 对应 `ShareGraph::AddNegFusedGraph(3)`。
TEST_F(LegacyGraphGuardMemSt, AddNeg) {
  ExpectMlirOwnsGraph(AscGraphBuilder("addneg")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Data("data1", 1, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Load("load1", "data1")
                          .Op<af::ascir_op::Add>("add", {"load0", "load1"})
                          .Op<af::ascir_op::Neg>("neg", {"add"})
                          .Store("store", "neg")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      "Ascend910B1");
}

// 对应 `ShareGraph::AddRsqrtFusedGraph(3)`。
TEST_F(LegacyGraphGuardMemSt, AddRsqrt) {
  ExpectMlirOwnsGraph(AscGraphBuilder("addrsqrt")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Data("data1", 1, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Load("load1", "data1")
                          .Op<af::ascir_op::Add>("add", {"load0", "load1"})
                          .Op<af::ascir_op::Rsqrt>("rsqrt", {"add"})
                          .Store("store", "rsqrt")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      "Ascend910B1");
}

// 对应 `ShareGraph::AsinBf16FusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, AsinBf16) {
  ExpectMlirOwnsGraph(AscGraphBuilder("asinbf16")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_BF16)
                          .Load("load0", "data0")
                          .Cast("cast", "load0", af::DT_FLOAT)
                          .Op<af::ascir_op::Asin>("asin", {"cast"})
                          .Cast("cast_back", "asin", af::DT_BF16)
                          .Store("store", "cast_back")
                          .Output("output", "store", 0, af::DT_BF16)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::AsinhBf16FusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, AsinhBf16) {
  ExpectMlirOwnsGraph(AscGraphBuilder("asinhbf16")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_BF16)
                          .Load("load0", "data0")
                          .Cast("cast", "load0", af::DT_FLOAT)
                          .Op<af::ascir_op::Asinh>("asinh", {"cast"})
                          .Cast("cast_back", "asinh", af::DT_BF16)
                          .Store("store", "cast_back")
                          .Output("output", "store", 0, af::DT_BF16)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::Atan2Bf16FusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, Atan2Bf16) {
  ExpectMlirOwnsGraph(AscGraphBuilder("atan2bf16")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_BF16)
                          .Data("data1", 1, af::DT_BF16)
                          .Load("load0", "data0")
                          .Load("load1", "data1")
                          .Op<af::ascir_op::Atan2>("atan2", {"load0", "load1"})
                          .Store("store", "atan2")
                          .Output("output", "store", 0, af::DT_BF16)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::AtanBf16FusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, AtanBf16) {
  ExpectMlirOwnsGraph(AscGraphBuilder("atanbf16")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_BF16)
                          .Load("load0", "data0")
                          .Op<af::ascir_op::Atan>("atan", {"load0"})
                          .Store("store", "atan")
                          .Output("output", "store", 0, af::DT_BF16)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::AtanhBf16FusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, AtanhBf16) {
  ExpectMlirOwnsGraph(AscGraphBuilder("atanhbf16")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_BF16)
                          .Load("load0", "data0")
                          .Cast("cast", "load0", af::DT_FLOAT)
                          .Op<af::ascir_op::Atanh>("atanh", {"cast"})
                          .Cast("cast_back", "atanh", af::DT_BF16)
                          .Store("store", "cast_back")
                          .Output("output", "store", 0, af::DT_BF16)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::BesselJ0StoreFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, BesselJ0Store) {
  ExpectMlirOwnsGraph(AscGraphBuilder("besselj0store")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Op<af::ascir_op::BesselJ0>("bessel_j0", {"load0"})
                          .Store("store", "bessel_j0")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::BesselJ1StoreFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, BesselJ1Store) {
  ExpectMlirOwnsGraph(AscGraphBuilder("besselj1store")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Op<af::ascir_op::BesselJ1>("bessel_j1", {"load0"})
                          .Store("store", "bessel_j1")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::BesselY0StoreFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, BesselY0Store) {
  ExpectMlirOwnsGraph(AscGraphBuilder("bessely0store")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Op<af::ascir_op::BesselY0>("bessel_y0", {"load0"})
                          .Store("store", "bessel_y0")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::BesselY1StoreFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, BesselY1Store) {
  ExpectMlirOwnsGraph(AscGraphBuilder("bessely1store")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Op<af::ascir_op::BesselY1>("bessel_y1", {"load0"})
                          .Store("store", "bessel_y1")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::CastCastNanFusedGraph(3, DT_FLOAT, DT_UINT8)`。
//
// **这个图的两个 dtype 是工厂的形参，不是图里的字面量**，所以要去调用点看：
// `cast_nan_test` 传的是 (f32 进, ui8 出)。`Isnan` 的输出恒为一个字节的真值，走建图器的专用
// 方法而不是通用 `Op<>`——后者会把输出 dtype 从输入继承成 f32，报
// "'afir.isnan' op operand #1 must be ... 8-bit unsigned integer"。
TEST_F(LegacyGraphGuardRegSt, CastCastNan) {
  ExpectMlirOwnsGraph(AscGraphBuilder("castcastnan")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Load("load", "data0")
                          .Isnan("IsNan", "load")
                          .Store("store", "IsNan")
                          .Output("output", "store", 0, af::DT_UINT8)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::CeilBf16FusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, CeilBf16) {
  ExpectMlirOwnsGraph(AscGraphBuilder("ceilbf16")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data", 0, af::DT_BF16)
                          .Load("load", "data")
                          .Op<af::ascir_op::Ceil>("ceil", {"load"})
                          .Store("store", "ceil")
                          .Output("output", "store", 0, af::DT_BF16)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::CopysignBf16FusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, CopysignBf16) {
  ExpectMlirOwnsGraph(AscGraphBuilder("copysignbf16")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_BF16)
                          .Data("data1", 1, af::DT_BF16)
                          .Load("load0", "data0")
                          .Load("load1", "data1")
                          .Op<af::ascir_op::CopySign>("copysign", {"load0", "load1"})
                          .Store("store", "copysign")
                          .Output("output", "store", 0, af::DT_BF16)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::CosBf16FusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, CosBf16) {
  ExpectMlirOwnsGraph(AscGraphBuilder("cosbf16")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data", 0, af::DT_BF16)
                          .Load("load", "data")
                          .Op<af::ascir_op::Cos>("cos", {"load"})
                          .Store("store", "cos")
                          .Output("output", "store", 0, af::DT_BF16)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::CoshBf16FusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, CoshBf16) {
  ExpectMlirOwnsGraph(AscGraphBuilder("coshbf16")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_BF16)
                          .Load("load0", "data0")
                          .Cast("cast", "load0", af::DT_FLOAT)
                          .Op<af::ascir_op::Cosh>("cosh", {"cast"})
                          .Cast("cast_back", "cosh", af::DT_BF16)
                          .Store("store", "cast_back")
                          .Output("output", "store", 0, af::DT_BF16)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::DigammaBf16FusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, DigammaBf16) {
  ExpectMlirOwnsGraph(AscGraphBuilder("digammabf16")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_BF16)
                          .Load("load0", "data0")
                          .Cast("cast", "load0", af::DT_FLOAT)
                          .Op<af::ascir_op::Digamma>("digamma", {"cast"})
                          .Cast("cast_back", "digamma", af::DT_BF16)
                          .Store("store", "cast_back")
                          .Output("output", "store", 0, af::DT_BF16)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::DivAbsFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, DivAbs) {
  ExpectMlirOwnsGraph(AscGraphBuilder("divabs")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Data("data1", 1, af::DT_FLOAT)
                          .Load("load1", "data1")
                          .Op<af::ascir_op::Div>("div", {"load0", "load1"})
                          .Op<af::ascir_op::Abs>("abs", {"div"})
                          .Store("store", "abs")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::ErfBf16FusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, ErfBf16) {
  ExpectMlirOwnsGraph(AscGraphBuilder("erfbf16")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data", 0, af::DT_BF16)
                          .Load("load", "data")
                          .Op<af::ascir_op::Erf>("erf", {"load"})
                          .Store("store", "erf")
                          .Output("output", "store", 0, af::DT_BF16)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::ErfcBf16FusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, ErfcBf16) {
  ExpectMlirOwnsGraph(AscGraphBuilder("erfcbf16")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_BF16)
                          .Load("load0", "data0")
                          .Cast("cast", "load0", af::DT_FLOAT)
                          .Op<af::ascir_op::Erfc>("erfc", {"cast"})
                          .Cast("cast_back", "erfc", af::DT_BF16)
                          .Store("store", "cast_back")
                          .Output("output", "store", 0, af::DT_BF16)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::ErfcxTestFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, ErfcxTest) {
  ExpectMlirOwnsGraph(AscGraphBuilder("erfcxtest")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Op<af::ascir_op::Erfcx>("erfcx", {"load0"})
                          .Store("store", "erfcx")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::Expm1Bf16FusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, Expm1Bf16) {
  ExpectMlirOwnsGraph(AscGraphBuilder("expm1bf16")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data", 0, af::DT_BF16)
                          .Load("load", "data")
                          .Op<af::ascir_op::Expm1>("expm1", {"load"})
                          .Store("store", "expm1")
                          .Output("output", "store", 0, af::DT_BF16)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::FloorDivInt32StoreFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, FloorDivInt32Store) {
  ExpectMlirOwnsGraph(AscGraphBuilder("floordivint32store")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_INT32)
                          .Data("data1", 1, af::DT_INT32)
                          .Load("load0", "data0")
                          .Load("load1", "data1")
                          .Op<af::ascir_op::FloorDiv>("floor_div", {"load0", "load1"})
                          .Store("store", "floor_div")
                          .Output("output", "store", 0, af::DT_INT32)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::FmodFloatFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, FmodFloat) {
  ExpectMlirOwnsGraph(AscGraphBuilder("fmodfloat")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Data("data1", 1, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Load("load1", "data1")
                          .Op<af::ascir_op::Fmod>("fmod", {"load0", "load1"})
                          .Store("store", "fmod")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::I0StoreFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, I0Store) {
  ExpectMlirOwnsGraph(AscGraphBuilder("i0store")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Op<af::ascir_op::I0>("i0", {"load0"})
                          .Store("store", "i0")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::I0eStoreFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, I0eStore) {
  ExpectMlirOwnsGraph(AscGraphBuilder("i0estore")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Op<af::ascir_op::I0e>("i0e", {"load0"})
                          .Store("store", "i0e")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::I1eStoreFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, I1eStore) {
  ExpectMlirOwnsGraph(AscGraphBuilder("i1estore")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Op<af::ascir_op::I1e>("i1e", {"load0"})
                          .Store("store", "i1e")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::LgammaFloatFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, LgammaFloat) {
  ExpectMlirOwnsGraph(AscGraphBuilder("lgammafloat")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Op<af::ascir_op::Lgamma>("lgamma", {"load0"})
                          .Store("store", "lgamma")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::LoadAiryAiStoreFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, LoadAiryAiStore) {
  ExpectMlirOwnsGraph(AscGraphBuilder("loadairyaistore")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Op<af::ascir_op::AiryAi>("airy_ai", {"load0"})
                          .Store("store", "airy_ai")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::LoadCompareStoreFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, LoadCompareStore) {
  ExpectMlirOwnsGraph(AscGraphBuilder("loadcomparestore")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Data("data1", 1, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Load("load1", "data1")
                          .Ge("ge0", "load0", "load1")
                          .Cast("cast0", "ge0", af::DT_FLOAT)
                          .Store("store0", "cast0")
                          .Output("output", "store0", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::LoadErfinvStoreFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, LoadErfinvStore) {
  ExpectMlirOwnsGraph(AscGraphBuilder("loaderfinvstore")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Op<af::ascir_op::Erfinv>("erfinv", {"load0"})
                          .Store("store", "erfinv")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::LoadLog2StoreFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, LoadLog2Store) {
  ExpectMlirOwnsGraph(AscGraphBuilder("loadlog2store")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Op<af::ascir_op::Log2>("log2", {"load0"})
                          .Store("store", "log2")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::LoadLogicalNotStoreFusedGraph(3)`。
TEST_F(LegacyGraphGuardMemSt, LoadLogicalNotStore) {
  ExpectMlirOwnsGraph(AscGraphBuilder("loadlogicalnotstore")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Op<af::ascir_op::LogicalNot>("logical_not", {"load0"})
                          .Store("store", "logical_not")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      "Ascend910B1");
}

// 对应 `ShareGraph::LoadModifiedBesselI0StoreFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, LoadModifiedBesselI0Store) {
  ExpectMlirOwnsGraph(AscGraphBuilder("loadmodifiedbesseli0store")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Op<af::ascir_op::ModifiedBesselI0>("modified_bessel_i0", {"load0"})
                          .Store("store", "modified_bessel_i0")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::LoadModifiedBesselI1StoreFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, LoadModifiedBesselI1Store) {
  ExpectMlirOwnsGraph(AscGraphBuilder("loadmodifiedbesseli1store")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Op<af::ascir_op::ModifiedBesselI1>("modified_bessel_i1", {"load0"})
                          .Store("store", "modified_bessel_i1")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::LoadModifiedBesselK0StoreFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, LoadModifiedBesselK0Store) {
  ExpectMlirOwnsGraph(AscGraphBuilder("loadmodifiedbesselk0store")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Op<af::ascir_op::ModifiedBesselK0>("modified_bessel_k0", {"load0"})
                          .Store("store", "modified_bessel_k0")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::LoadModifiedBesselK1StoreFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, LoadModifiedBesselK1Store) {
  ExpectMlirOwnsGraph(AscGraphBuilder("loadmodifiedbesselk1store")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Op<af::ascir_op::ModifiedBesselK1>("modified_bessel_k1", {"load0"})
                          .Store("store", "modified_bessel_k1")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::LoadTanhStoreFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, LoadTanhStore) {
  ExpectMlirOwnsGraph(AscGraphBuilder("loadtanhstore")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Op<af::ascir_op::Tanh>("sigmoid0", {"load0"})
                          .Store("store", "sigmoid0")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::Log1pBfloat16FusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, Log1pBfloat16) {
  ExpectMlirOwnsGraph(AscGraphBuilder("log1pbfloat16")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_BF16)
                          .Load("load0", "data0")
                          .Op<af::ascir_op::Log1p>("log1p", {"load0"})
                          .Store("store", "log1p")
                          .Output("output", "store", 0, af::DT_BF16)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::LogNdtrStoreFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, LogNdtrStore) {
  ExpectMlirOwnsGraph(AscGraphBuilder("logndtrstore")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Op<af::ascir_op::LogNdtr>("log_ndtr", {"load0"})
                          .Store("store", "log_ndtr")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::LogicalXorFloatFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, LogicalXorFloat) {
  ExpectMlirOwnsGraph(AscGraphBuilder("logicalxorfloat")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Data("data1", 1, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Load("load1", "data1")
                          .LogicalXor("logicalxor", "load0", "load1")
                          .Store("store", "logicalxor")
                          .Output("output", "store", 0, af::DT_UINT8)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::ModFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, Mod) {
  ExpectMlirOwnsGraph(AscGraphBuilder("mod")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Data("data1", 1, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Load("load1", "data1")
                          .Op<af::ascir_op::Mod>("mod", {"load0", "load1"})
                          .Store("store", "mod")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::NdtrStoreFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, NdtrStore) {
  ExpectMlirOwnsGraph(AscGraphBuilder("ndtrstore")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Op<af::ascir_op::Ndtr>("ndtr", {"load0"})
                          .Store("store", "ndtr")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::NdtriStoreFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, NdtriStore) {
  ExpectMlirOwnsGraph(AscGraphBuilder("ndtristore")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Op<af::ascir_op::Ndtri>("ndtri", {"load0"})
                          .Store("store", "ndtri")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::NextAfterStoreFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, NextAfterStore) {
  ExpectMlirOwnsGraph(AscGraphBuilder("nextafterstore")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Data("data1", 1, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Load("load1", "data1")
                          .Op<af::ascir_op::NextAfter>("next_after", {"load0", "load1"})
                          .Store("store", "next_after")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::PowBf16FusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, PowBf16) {
  ExpectMlirOwnsGraph(AscGraphBuilder("powbf16")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_BF16)
                          .Data("data1", 1, af::DT_BF16)
                          .Load("load0", "data0")
                          .Load("load1", "data1")
                          .Op<af::ascir_op::Pow>("pow", {"load0", "load1"})
                          .Store("store", "pow")
                          .Output("output", "store", 0, af::DT_BF16)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::RemainderInt32StoreFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, RemainderInt32Store) {
  ExpectMlirOwnsGraph(AscGraphBuilder("remainderint32store")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_INT32)
                          .Data("data1", 1, af::DT_INT32)
                          .Load("load0", "data0")
                          .Load("load1", "data1")
                          .Op<af::ascir_op::Remainder>("remainder", {"load0", "load1"})
                          .Store("store", "remainder")
                          .Output("output", "store", 0, af::DT_INT32)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::RoundBf16FusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, RoundBf16) {
  ExpectMlirOwnsGraph(AscGraphBuilder("roundbf16")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data", 0, af::DT_BF16)
                          .Load("load", "data")
                          .Op<af::ascir_op::Round>("round", {"load"})
                          .Store("store", "round")
                          .Output("output", "store", 0, af::DT_BF16)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::ScaledModifiedBesselK0StoreFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, ScaledModifiedBesselK0Store) {
  ExpectMlirOwnsGraph(AscGraphBuilder("scaledmodifiedbesselk0store")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Op<af::ascir_op::ScaledModifiedBesselK0>("scaled_modified_bessel_k0", {"load0"})
                          .Store("store", "scaled_modified_bessel_k0")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::ScaledModifiedBesselK1StoreFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, ScaledModifiedBesselK1Store) {
  ExpectMlirOwnsGraph(AscGraphBuilder("scaledmodifiedbesselk1store")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Op<af::ascir_op::ScaledModifiedBesselK1>("scaled_modified_bessel_k1", {"load0"})
                          .Store("store", "scaled_modified_bessel_k1")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::SignBf16FusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, SignBf16) {
  ExpectMlirOwnsGraph(AscGraphBuilder("signbf16")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data", 0, af::DT_BF16)
                          .Load("load", "data")
                          .Op<af::ascir_op::Sign>("sign", {"load"})
                          .Store("store", "sign")
                          .Output("output", "store", 0, af::DT_BF16)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::SphericalBesselJ0StoreFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, SphericalBesselJ0Store) {
  ExpectMlirOwnsGraph(AscGraphBuilder("sphericalbesselj0store")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Op<af::ascir_op::SphericalBesselJ0>("spherical_bessel_j0", {"load0"})
                          .Store("store", "spherical_bessel_j0")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::ZetaStoreFusedGraph(3)`。
TEST_F(LegacyGraphGuardRegSt, ZetaStore) {
  ExpectMlirOwnsGraph(AscGraphBuilder("zetastore")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Data("data1", 1, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Load("load1", "data1")
                          .Op<af::ascir_op::Zeta>("zeta", {"load0", "load1"})
                          .Store("store", "zeta")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

//===----------------------------------------------------------------------===//
// 手工译的图：带自有属性、Scalar / Where、或多输出
//
// 这些不走生成器：属性（Axpy 的 alpha、LeakyRelu 的 negative_slope）、输出 dtype 不跟输入走、
// 多 Store / 多 Output，都要按建图器的专用方法写。
//===----------------------------------------------------------------------===//

// 对应 `ShareGraph::AbsClipFusedGraph(3)`。ClipByValue 三输入、输出跟 x1，通用 `Op<>` 够用。
TEST_F(LegacyGraphGuardRegSt, AbsClip) {
  ExpectMlirOwnsGraph(AscGraphBuilder("absclip")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Data("data1", 1, af::DT_FLOAT)
                          .Data("data2", 2, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Load("load1", "data1")
                          .Load("load2", "data2")
                          .Abs("abs", "load0")
                          .Op<af::ascir_op::ClipByValue>("clipbyvalue", {"abs", "load1", "load2"})
                          .Store("store", "clipbyvalue")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::AxpyAbsFusedGraph(3)`。alpha 是 Axpy 自有的 ir_attr，通用通道碰不到。
TEST_F(LegacyGraphGuardMemSt, AxpyAbs) {
  ExpectMlirOwnsGraph(AscGraphBuilder("axpyabs")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Data("data1", 1, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Load("load1", "data1")
                          .Axpy("axpy", "load0", "load1", 0.8F)
                          .Abs("abs", "axpy")
                          .Store("store", "abs")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      "Ascend910B1");
}

// 对应 `ShareGraph::LoadLeakyReluStoreFusedGraph(3)`。negative_slope 同上。
TEST_F(LegacyGraphGuardRegSt, LoadLeakyReluStore) {
  ExpectMlirOwnsGraph(AscGraphBuilder("loadleakyrelustore")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .LeakyRelu("leakyrelu0", "load0", 0.8F)
                          .Store("store", "leakyrelu0")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      /*soc_version=*/"");
}

// 对应 `ShareGraph::LoadToStoreAndAbsFusedGraph(3)`。两个 Store / 两个 Output：一条 load 既被
// Abs 吃掉又直接落盘，看护的是「一个中间张量喂两条出边」这个形状。
TEST_F(LegacyGraphGuardMemSt, LoadToStoreAndAbs) {
  ExpectMlirOwnsGraph(AscGraphBuilder("loadtostoreandabs")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Abs("abs", "load0")
                          .Store("store", "abs")
                          .Store("store2", "load0")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Output("output2", "store2", 1, af::DT_FLOAT)
                          .Build(),
                      "Ascend910B1");
}

// 对应 `ShareGraph::CastCastFusedGraph(3, DT_FLOAT16, DT_FLOAT)`。两个 dtype 是工厂形参，
// 要去调用点看。Cast 夹一个 Abs，再 Cast 回去。
TEST_F(LegacyGraphGuardRegSt, CastCast) {
  ExpectMlirOwnsGraph(AscGraphBuilder("castcast")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT16)
                          .Load("load", "data0")
                          .Cast("cast0", "load", af::DT_FLOAT)
                          .Abs("abs", "cast0")
                          .Cast("cast1", "abs", af::DT_FLOAT16)
                          .Store("store", "cast1")
                          .Output("output", "store", 0, af::DT_FLOAT16)
                          .Build(),
                      /*soc_version=*/"");
}

//===----------------------------------------------------------------------===//
// Compare 族：一个工厂三种实参组合，各是一张不同的图（`CompareFusedGraph(2, ...)`，二维）
//===----------------------------------------------------------------------===//

// `(2, false, DT_FLOAT, "eq")`——第二个输入是 Scalar 而不是张量。
TEST_F(LegacyGraphGuardRegSt, CompareEqScalar) {
  ExpectMlirOwnsGraph(AscGraphBuilder("compare_eq_scalar")
                          .Loops({Sym("s0"), Sym("s1")})
                          .Data("compare_data0", 0, af::DT_FLOAT)
                          .Load("load0", "compare_data0")
                          .Scalar("scalar0", "1", af::DT_FLOAT)
                          .Eq("eq", "load0", "scalar0")
                          .Store("store", "eq")
                          .Output("output", "store", 0, af::DT_UINT8)
                          .Build(),
                      /*soc_version=*/"");
}

// `(2, true, DT_FLOAT, "gt")`——两个输入都是张量。
TEST_F(LegacyGraphGuardRegSt, CompareGtTensorFloat) {
  ExpectMlirOwnsGraph(AscGraphBuilder("compare_gt_tensor_float")
                          .Loops({Sym("s0"), Sym("s1")})
                          .Data("compare_data0", 0, af::DT_FLOAT)
                          .Data("compare_data1", 1, af::DT_FLOAT)
                          .Load("load0", "compare_data0")
                          .Load("load2", "compare_data1")
                          .Gt("gt", "load0", "load2")
                          .Store("store", "gt")
                          .Output("output", "store", 0, af::DT_UINT8)
                          .Build(),
                      /*soc_version=*/"");
}

// `(2, true, DT_INT32, "gt")`。
TEST_F(LegacyGraphGuardRegSt, CompareGtTensorInt32) {
  ExpectMlirOwnsGraph(AscGraphBuilder("compare_gt_tensor_int32")
                          .Loops({Sym("s0"), Sym("s1")})
                          .Data("compare_data0", 0, af::DT_INT32)
                          .Data("compare_data1", 1, af::DT_INT32)
                          .Load("load0", "compare_data0")
                          .Load("load2", "compare_data1")
                          .Gt("gt", "load0", "load2")
                          .Store("store", "gt")
                          .Output("output", "store", 0, af::DT_UINT8)
                          .Build(),
                      /*soc_version=*/"");
}

//===----------------------------------------------------------------------===//
// Where 族：掩码是 ui8，x2 / x3 可以是张量也可以是 UB 上的 Scalar
//
// 走 `Op<>` 但 **follow_index 必须是 1**：输出 dtype 跟 x2 / x3，不跟 x1 那个 ui8 掩码。
//===----------------------------------------------------------------------===//

// 对应 `ShareGraph::LoadWhereStoreFusedGraph(3)`——三个都是张量。
TEST_F(LegacyGraphGuardMemSt, LoadWhereStore) {
  ExpectMlirOwnsGraph(AscGraphBuilder("loadwherestore")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_UINT8)
                          .Data("data1", 1, af::DT_FLOAT)
                          .Data("data2", 2, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Load("load1", "data1")
                          .Load("load2", "data2")
                          .Op<af::ascir_op::Where>("where", {"load0", "load1", "load2"}, 1U)
                          .Store("store", "where")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      "Ascend910B1");
}

// 对应 `ShareGraph::LoadWhereX2IsUbscalarStoreFusedGraph(3)`——x2 是 Scalar。
TEST_F(LegacyGraphGuardMemSt, LoadWhereX2IsUbscalarStore) {
  ExpectMlirOwnsGraph(AscGraphBuilder("loadwherex2isubscalarstore")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_UINT8)
                          .Data("data1", 1, af::DT_FLOAT)
                          .Scalar("scalar0", "100", af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Load("load1", "data1")
                          // **继承源必须是张量**：`follow_index` 同时决定 dtype 与轴，指向
                          // Scalar 会让输出拿到空轴集，掩码就广播不到输出，候选枚举按
                          // elementwise-broadcast 判据拒收（实测）。这里指向 x3 的 `load1`。
                          .Op<af::ascir_op::Where>("where", {"load0", "scalar0", "load1"}, 2U)
                          .Store("store", "where")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      "Ascend910B1");
}

// 对应 `ShareGraph::LoadWhereX2X3IsUbscalarStoreFusedGraph(3)`——x2 与 x3 都是 Scalar。
TEST_F(LegacyGraphGuardMemSt, LoadWhereX2X3IsUbscalarStore) {
  ExpectMlirOwnsGraph(AscGraphBuilder("loadwherex2x3isubscalarstore")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_UINT8)
                          .Scalar("scalar0", "100", af::DT_FLOAT)
                          .Scalar("scalar1", "200", af::DT_FLOAT)
                          .Load("load0", "data0")
                          // 两个分支都是 UB 上的 Scalar，图里唯一带轴的输入是掩码，所以轴只能
                          // 跟掩码；而 `afir.where` 的 verifier 要求两个分支与结果同 dtype，
                          // 也就是 dtype 不能跟掩码。两者要分开，所以走 `OpWithDtype`。
                          .OpWithDtype<af::ascir_op::Where>("where", {"load0", "scalar0", "scalar1"}, 0U, af::DT_FLOAT)
                          .Store("store", "where")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      "Ascend910B1");
}

// 对应 `ShareGraph::LoadWhereX3IsUbscalarStoreFusedGraph(3)`——x3 是 Scalar。
TEST_F(LegacyGraphGuardMemSt, LoadWhereX3IsUbscalarStore) {
  ExpectMlirOwnsGraph(AscGraphBuilder("loadwherex3isubscalarstore")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_UINT8)
                          .Data("data1", 1, af::DT_FLOAT)
                          .Scalar("scalar0", "200", af::DT_FLOAT)
                          .Load("load0", "data0")
                          .Load("load1", "data1")
                          .Op<af::ascir_op::Where>("where", {"load0", "load1", "scalar0"}, 1U)
                          .Store("store", "where")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      "Ascend910B1");
}

// 对应 `ShareGraph::IsInfMaskedFillFusedGraph(3)`。MaskedFill 的三个入是 x / mask / value，
// 按位置走通用通道；输出跟 x（f32），所以 follow_index 用 0。
TEST_F(LegacyGraphGuardMemSt, IsInfMaskedFill) {
  ExpectMlirOwnsGraph(AscGraphBuilder("isinfmaskedfill")
                          .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                          .Data("data0", 0, af::DT_FLOAT)
                          .Load("load0", "data0")
                          .IsInf("is_inf", "load0")
                          .Scalar("scalar0", "-1.0", af::DT_FLOAT)
                          .Op<af::ascir_op::MaskedFill>("masked_fill", {"load0", "is_inf", "scalar0"}, 0U)
                          .Store("store", "masked_fill")
                          .Output("output", "store", 0, af::DT_FLOAT)
                          .Build(),
                      "Ascend910B1");
}

//===----------------------------------------------------------------------===//
// 带显式 view 的图：Load / Broadcast 的 repeats / strides 要照着原图写
//===----------------------------------------------------------------------===//

// 对应 `ShareGraph::LoadPowAllInputIsScalarStoreFusedGraph`（二维）。两个输入都是 1x1、步长全 0
// 的「标量张量」，各自 Broadcast 到 {s0, s1} 之后再 Pow。看护的是「两个操作数都靠广播铺开」
// 这个形状——它和 `afir.broadcast` 的尾轴对齐不变量是同一条路上的。
TEST_F(LegacyGraphGuardMemSt, LoadPowAllInputIsScalarStore) {
  ExpectMlirOwnsGraph(
      AscGraphBuilder("loadpowallinputisscalarstore")
          .Loops({Sym("s0"), Sym("s1")})
          .Data("scalar1", 0, af::DT_FLOAT)
          .Data("scalar2", 1, af::DT_FLOAT)
          .Load("load1", "scalar1", {Sym(int64_t{1}), Sym(int64_t{1})}, {Sym(int64_t{0}), Sym(int64_t{0})})
          .Broadcast("brc1", "load1", {Sym("s0"), Sym("s1")})
          .Load("load2", "scalar2", {Sym(int64_t{1}), Sym(int64_t{1})}, {Sym(int64_t{0}), Sym(int64_t{0})})
          .Broadcast("brc2", "load2", {Sym("s0"), Sym("s1")})
          .Pow("pow", "brc1", "brc2")
          .Store("store", "pow")
          .Output("output", "store", 0, af::DT_FLOAT)
          .Build(),
      "Ascend910B1");
}

// 对应 `ShareGraph::ContinuesBrcFusedGraph`（五维）。看护的是**链式广播**：load1 的 view 是
// 1 x s1 x 1 x s3 x 1，经三个 Broadcast 逐个把第 0、2、4 轴铺开，广播结果既直接落盘（store1 /
// y1），又与 load0 相加后落另一条出边（store / y）。一个中间张量喂两条出边、而且其中一条是
// 广播链的终点——这是 `afir.broadcast` 那条尾轴不变量最容易出事的形状。
TEST_F(LegacyGraphGuardRegSt, ContinuesBrc) {
  ExpectMlirOwnsGraph(
      AscGraphBuilder("continuesbrc")
          .Loops({Sym("s0"), Sym("s1"), Sym("s2"), Sym("s3"), Sym("s4")})
          .Data("x0", 0, af::DT_FLOAT)
          .Data("x1", 1, af::DT_FLOAT)
          .Load("load0", "x0")
          .Load("load1", "x1", {Sym(int64_t{1}), Sym("s1"), Sym(int64_t{1}), Sym("s3"), Sym(int64_t{1})},
                {Sym(int64_t{0}), Sym("s3"), Sym(int64_t{0}), Sym(int64_t{1}), Sym(int64_t{0})})
          .Broadcast("brc0", "load1", {Sym("s0"), Sym("s1"), Sym(int64_t{1}), Sym("s3"), Sym(int64_t{1})})
          .Broadcast("brc1", "brc0", {Sym("s0"), Sym("s1"), Sym("s2"), Sym("s3"), Sym(int64_t{1})})
          .Broadcast("brc2", "brc1", {Sym("s0"), Sym("s1"), Sym("s2"), Sym("s3"), Sym("s4")})
          .Store("store1", "brc2")
          .Output("y1", "store1", 1, af::DT_FLOAT)
          .Add("add0", "load0", "brc2")
          .Store("store", "add0")
          .Output("y", "store", 0, af::DT_FLOAT)
          .Build(),
      /*soc_version=*/"");
}

// 对应 `ShareGraph::ConcatAscGraph(graph, {"s0", "s1"})`（二维，int32）。两条 load 沿最内轴
// 拼接。`concat.x` 是变参输入，建图器的 `Concat` 直接收一个列表。
TEST_F(LegacyGraphGuardMemSt, Concat) {
  ExpectMlirOwnsGraph(AscGraphBuilder("concat")
                          .Loops({Sym("s0"), Sym("s1")})
                          .Data("concat_data0", 0, af::DT_INT32)
                          .Data("concat_data1", 1, af::DT_INT32)
                          .Load("load0", "concat_data0")
                          .Load("load1", "concat_data1")
                          .Concat("concat", {"load0", "load1"})
                          .Store("store", "concat")
                          .Output("output", "store", 0, af::DT_INT32)
                          .Build(),
                      "Ascend910B1");
}

//===----------------------------------------------------------------------===//
// 没搬的那一个
//
// `ShareGraph::LoadWhereReduceStoreFusedGraph`（reg 侧 3 格用它）**没搬**，不是漏了：
//
//   - 它是手工拼出来的不规则图：`Data` 只有 {z0, z1} 两根轴而 `Load` 有三根，`Store` 吃的是
//     `where.y` 却声明了归约后的 view（{s0, s1, 1}），而 `max` 的输出没有任何人消费。
//   - 照着它写需要逐节点摆 axis / repeats / strides，而建图器那条路要先拿到 `Loops` 建出来的
//     axis id；凑出来的东西未必是原图，那就不是「搬输入图」而是现编一张。
//   - 它比 `LoadWhereX2X3IsUbscalarStore` 多出来的料只有 reduce，而 reduce 不在本轮
//     elementwise / broadcast 的范围内。
//
// 所以这一格留给做 reduce 那一轮，连同它那三个 legacy 用例一起看。
//===----------------------------------------------------------------------===//

}  // namespace
