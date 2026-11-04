/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// `ReducePlan` 的覆盖（reduce 跨作用域累加设计 3.2.1、7.2）。
//
// **为什么不是 LIT**：`buildReducePlan` 不改 IR，它的输出是一组推论——累加轴、AR/RA、
// 两块 scratch 谁是谁。这些在任何 pass 的输出文本里都没有形状，LIT 只能间接地从「最终
// 生成了什么」倒推。而这里正反两侧都要钉：既要钉住 S0 推出 `[z1T]`，也要钉住 elementwise
// 推不出任何累加轴——后者一旦失守，LoopGenerate 会把普通算子的跨作用域结果也放行，
// 那是静默算错值，不是报错。
//
// 轴编号沿用调度器 S0 通用模板的真实编号，便于与 `afir_loop_generate_reduce_accumulation.mlir`
// 对照：0/1 原始轴，2/3 = z0T/z0t，4/5 = z1T/z1t，6/7 = z0TB/z0Tb。被归约的是 z1 一侧。

#include "dialect/AFIR/Analysis/ReducePlan.h"
#include "dialect/AFIR/Analysis/ReduceScratchLifetime.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include <string>

#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/IR/ReduceScratch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
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

class ReducePlanTest : public testing::Test {
 protected:
  void SetUp() override {
    context.getOrLoadDialect<AFIRDialect>();
    context.getOrLoadDialect<func::FuncDialect>();
    context.getOrLoadDialect<memref::MemRefDialect>();
    context.getOrLoadDialect<arith::ArithDialect>();
    context.getOrLoadDialect<scf::SCFDialect>();
  }

  // 两块 GM 是「跨阶段 workspace」的载体：id 是它唯一跨 func 对得上的身份。
  static constexpr llvm::StringLiteral kPrelude = R"mlir(
#ub0 = #afir.buffer_desc<space = <ub>, buffer_id = 0, life_time_axis_id = -1>
#ub1 = #afir.buffer_desc<space = <ub>, buffer_id = 1, life_time_axis_id = -1>
#ub2 = #afir.buffer_desc<space = <ub>, buffer_id = 2, life_time_axis_id = -1>
#ub3 = #afir.buffer_desc<space = <ub>, buffer_id = 3, life_time_axis_id = -1>
#ub4 = #afir.buffer_desc<space = <ub>, buffer_id = 4, life_time_axis_id = -1>
#ub5 = #afir.buffer_desc<space = <ub>, buffer_id = 5, life_time_axis_id = -1>
#gm8 = #afir.buffer_desc<space = <gm>, buffer_id = 8, life_time_axis_id = -1>
#gm9 = #afir.buffer_desc<space = <gm>, buffer_id = 9, life_time_axis_id = -1>
)mlir";

  // 整个 module 自己给（多 func 的跨阶段形态需要），只共用上面那组别名。
  OwningOpRef<ModuleOp> ParseModule(llvm::StringRef source) {
    return parseSourceString<ModuleOp>(kPrelude.str() + source.str(), &context);
  }

  OwningOpRef<ModuleOp> Parse(llvm::StringRef body, llvm::StringRef graph = {}) {
    std::string source = kPrelude.str() + R"mlir(
module {
  func.func private @kernel() attributes {
    afir.function_role = #afir.function_role<kernel_body>)mlir" +
                         (graph.empty() ? std::string() : ",\n    afir.asc_graph = " + graph.str()) + R"mlir(
  } {
)mlir" + body.str() +
                         R"mlir(
    return
  }
}
)mlir";
    return parseSourceString<ModuleOp>(source, &context);
  }

  // 函数体里唯一那个被测算子。
  Operation *Subject(ModuleOp module, llvm::StringRef name) {
    Operation *found = nullptr;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == name) {
        found = op;
      }
    });
    return found;
  }

  MLIRContext context;
};

// S0 的形状：输入 `[z0TB, z0Tb, z0t, z1T, z1t]`，destination 在 z1T / z1t 两维 stride 为 0。
// 调度路径是 `[6, 7, 4]`（到 loop_axis 为止），其中只有 z1T(4) 是归约维——z1t(5) 落在
// loop_axis 之内，由一次 API 调用算完，不跨轮，所以不进累加轴。
constexpr llvm::StringLiteral kS0Body = R"mlir(
    %in = memref.alloc() {afir.buffer_desc = #ub0} : memref<2x2x2x4x8xf32>
    %api = memref.alloc() {afir.buffer_desc = #ub1} : memref<256xui8>
    %acc = memref.alloc() {afir.buffer_desc = #ub2} : memref<256xui8>
    %dst = memref.alloc() {afir.buffer_desc = #ub3} : memref<2x2x2x1x1xf32>
    %sum = afir.sum %in outs(%dst) scratch(%api, %acc) {afir.scratch_roles = ["api_scratch", "round_result"],
      afir.operand_descs = [
        #afir.tensor_desc<axis = [6, 7, 3, 4, 5],
                          strides = [<"128">, <"64">, <"32">, <"8">, <"1">],
                          vectorized_axis = [3, 5], vectorized_strides = [<"8">, <"1">],
                          role = <compute>>],
      afir.destination_descs = [
        #afir.tensor_desc<axis = [6, 7, 3, 4, 5],
                          strides = [<"4">, <"2">, <"1">, <"0">, <"0">],
                          vectorized_axis = [3, 5], vectorized_strides = [<"1">, <"0">],
                          role = <compute>>],
      afir.result_descs = [
        #afir.tensor_desc<axis = [6, 7, 3, 4, 5],
                          strides = [<"4">, <"2">, <"1">, <"0">, <"0">],
                          vectorized_axis = [3, 5], vectorized_strides = [<"1">, <"0">],
                          role = <compute>>],
      afir.sched = #afir.sched<axis = [6, 7, 4, 3, 5], loop_axis = 4>
    } : memref<2x2x2x4x8xf32>, memref<2x2x2x1x1xf32>, scratch(memref<256xui8>, memref<256xui8>)
      -> memref<2x2x2x1x1xf32>
)mlir";

TEST_F(ReducePlanTest, S0AccumulatesAlongTheTileOuterAxisOnly) {
  auto module = Parse(kS0Body);
  ASSERT_TRUE(module);
  FailureOr<ReducePlan> plan = buildReducePlan(Subject(*module, "afir.sum"));
  ASSERT_TRUE(succeeded(plan));

  EXPECT_TRUE(plan->isAccumulating());
  ASSERT_EQ(plan->accumulationAxes.size(), 1U);
  EXPECT_EQ(plan->accumulationAxes.front(), 4);
}

// Real ARAR case[3] merges both reduced outer axes into one scheduler axis.
// Accumulation follows axis provenance, not direct id equality: merged(4, 5)
// is reduced, while a mixed A/R merge would not be.
TEST_F(ReducePlanTest, MergedReductionAxisAccumulates) {
  std::string body = kS0Body.str();
  const std::string direct = "#afir.sched<axis = [6, 7, 4, 3, 5], loop_axis = 4>";
  const auto at = body.find(direct);
  ASSERT_NE(at, std::string::npos);
  body.replace(at, direct.size(), "#afir.sched<axis = [6, 7, 8, 3, 5], loop_axis = 8>");

  auto module = Parse(body, R"mlir(#afir.asc_graph<tiling_key = 0,axes = [
      <id = 4, name = "r0", axis_type = Original, size = <"4">, align = <"1">>,
      <id = 5, name = "r1", axis_type = Original, size = <"8">, align = <"1">>,
      <id = 8, name = "r0r1", axis_type = Merged, size = <"32">, align = <"1">, from = [4, 5]>],
      type = Compute, size_var = [<"unused">]>)mlir");
  ASSERT_TRUE(module);
  FailureOr<ReducePlan> plan = buildReducePlan(Subject(*module, "afir.sum"));
  ASSERT_TRUE(succeeded(plan));
  ASSERT_EQ(plan->accumulationAxes.size(), 1U);
  EXPECT_EQ(plan->accumulationAxes.front(), 8);
}

TEST_F(ReducePlanTest, MixedKeptAndReducedMergedAxisDoesNotAccumulate) {
  std::string body = kS0Body.str();
  const std::string direct = "#afir.sched<axis = [6, 7, 4, 3, 5], loop_axis = 4>";
  const auto at = body.find(direct);
  ASSERT_NE(at, std::string::npos);
  body.replace(at, direct.size(), "#afir.sched<axis = [6, 7, 8, 3, 5], loop_axis = 8>");

  auto module = Parse(body, R"mlir(#afir.asc_graph<tiling_key = 0,axes = [
      <id = 3, name = "a", axis_type = Original, size = <"2">, align = <"1">>,
      <id = 5, name = "r", axis_type = Original, size = <"8">, align = <"1">>,
      <id = 8, name = "ar", axis_type = Merged, size = <"16">, align = <"1">, from = [3, 5]>],
      type = Compute, size_var = [<"unused">]>)mlir");
  ASSERT_TRUE(module);
  FailureOr<ReducePlan> plan = buildReducePlan(Subject(*module, "afir.sum"));
  ASSERT_TRUE(succeeded(plan));
  EXPECT_TRUE(plan->accumulationAxes.empty());
}

// destination 的末项向量化 stride 为 0 -> 被归约的是末维 -> AR，`[first, last]` 就是
// `[保留维, 归约维]`。
TEST_F(ReducePlanTest, S0IsArWithKeptAxisFirst) {
  auto module = Parse(kS0Body);
  ASSERT_TRUE(module);
  FailureOr<ReducePlan> plan = buildReducePlan(Subject(*module, "afir.sum"));
  ASSERT_TRUE(succeeded(plan));

  EXPECT_EQ(plan->pattern, ReducePattern::AR);
  ASSERT_EQ(plan->first.axes.size(), 1U);
  EXPECT_EQ(plan->first.axes.front(), 3);
  ASSERT_EQ(plan->last.axes.size(), 1U);
  EXPECT_EQ(plan->last.axes.front(), 5);
  EXPECT_TRUE(plan->tailLast);
}

// 全轴归约：保留轴是调度器造出来的长度 1 退化轴，它在**两侧**的 vectorized stride 都是 0。
//
// A/R 分类必须两侧一起看——落点侧 stride 为 0 **且**输入侧不为 0 才是归约维，两侧都为 0 是
// 退化/广播维。只看落点侧会把这根退化的保留轴也判成归约维，`plan.first.axes` 于是为空，
// `buildReducePlan` 整个失败；调用点 `redirectAccumulatedOperand` 把这个失败报成一句与真因
// 无关的「tensor SSA value crosses sibling/outer schedule scopes」，全轴归约在真机上就是这样
// 一条都下降不了的。判据与 `collectReducedAxes` 用的是同一条（那里读的是 `strides`）。
TEST_F(ReducePlanTest, AllReduceKeepsTheDegenerateAxisAsTheKeptSide) {
  std::string body = kS0Body.str();
  // 输入侧：退化的保留轴 stride 0，归约轴 stride 1。
  const std::string inputAxes = "vectorized_axis = [3, 5], vectorized_strides = [<\"8\">, <\"1\">]";
  const std::string inputDegenerate = "vectorized_axis = [3, 5], vectorized_strides = [<\"0\">, <\"1\">]";
  ASSERT_NE(body.find(inputAxes), std::string::npos);
  body.replace(body.find(inputAxes), inputAxes.size(), inputDegenerate);
  // 落点侧：两根都是 0。
  const std::string axes = "vectorized_axis = [3, 5], vectorized_strides = [<\"1\">, <\"0\">]";
  const std::string bothZero = "vectorized_axis = [3, 5], vectorized_strides = [<\"0\">, <\"0\">]";
  for (size_t position = body.find(axes); position != std::string::npos; position = body.find(axes, position)) {
    body.replace(position, axes.size(), bothZero);
  }

  auto module = Parse(body);
  ASSERT_TRUE(module);
  FailureOr<ReducePlan> plan = buildReducePlan(Subject(*module, "afir.sum"));
  ASSERT_TRUE(succeeded(plan)) << "退化的保留轴不是归约维，A 侧不该被判空";

  EXPECT_EQ(plan->pattern, ReducePattern::AR);
  ASSERT_EQ(plan->first.axes.size(), 1U);
  EXPECT_EQ(plan->first.axes.front(), 3);
  ASSERT_EQ(plan->last.axes.size(), 1U);
  EXPECT_EQ(plan->last.axes.front(), 5);
}

// 成对的反例：两根向量化轴**都真的**是归约维（落点 0、输入非 0）时仍然拒绝。
//
// 只写上一条的话，把 `plan.first.axes.empty()` 那道门删掉也能全绿，而那会让一个没有 A 侧的
// tile 被当成二维 AR 发出去——高阶 API 的两个维度有一个是空的，算出来的是错值。
TEST_F(ReducePlanTest, RejectsATileWhoseVectorizedAxesAreAllReduced) {
  std::string body = kS0Body.str();
  const std::string axes = "vectorized_axis = [3, 5], vectorized_strides = [<\"1\">, <\"0\">]";
  const std::string bothZero = "vectorized_axis = [3, 5], vectorized_strides = [<\"0\">, <\"0\">]";
  for (size_t position = body.find(axes); position != std::string::npos; position = body.find(axes, position)) {
    body.replace(position, axes.size(), bothZero);
  }

  auto module = Parse(body);
  ASSERT_TRUE(module);
  // 输入侧两根都非 0（`kS0Body` 原样的 `8` 与 `1`），于是两根都是真归约维。
  //
  // 原因也要一并钉住：下面这五条失败用例走的是五个语义完全不同的失败点，而它们此前在调用方
  // 那里只剩一个裸 `failure()`，彼此不可区分。
  std::string reason;
  EXPECT_TRUE(failed(buildReducePlan(Subject(*module, "afir.sum"), &reason)))
      << "A 侧真的为空时必须拒绝，不能当成二维 AR 发出去";
  EXPECT_NE(reason.find("no kept axis"), std::string::npos) << reason;
}

TEST_F(ReducePlanTest, MultipleContiguousAxesMergeIntoAr) {
  std::string body = kS0Body.str();
  const std::string inputAxes = "vectorized_axis = [3, 5], vectorized_strides = [<\"8\">, <\"1\">]";
  const std::string inputAarr =
      "vectorized_axis = [3, 7, 4, 5], vectorized_strides = [<\"256\">, <\"32\">, <\"8\">, <\"1\">]";
  const auto inputAt = body.find(inputAxes);
  ASSERT_NE(inputAt, std::string::npos);
  body.replace(inputAt, inputAxes.size(), inputAarr);
  const std::string axes = "vectorized_axis = [3, 5], vectorized_strides = [<\"1\">, <\"0\">]";
  const std::string aarr = "vectorized_axis = [3, 7, 4, 5], vectorized_strides = [<\"32\">, <\"8\">, <\"0\">, <\"0\">]";
  for (auto at = body.find(axes); at != std::string::npos; at = body.find(axes)) {
    body.replace(at, axes.size(), aarr);
  }

  auto module = Parse(body);
  ASSERT_TRUE(module);
  FailureOr<ReducePlan> plan = buildReducePlan(Subject(*module, "afir.sum"));
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(plan->pattern, ReducePattern::AR);
  EXPECT_EQ(plan->first.axes, (llvm::SmallVector<int64_t, 2>{3, 7}));
  EXPECT_EQ(plan->last.axes, (llvm::SmallVector<int64_t, 2>{4, 5}));
  EXPECT_FALSE(plan->tailLast);
}

// RA 且 R 组有两根轴。计划**建得出来**——分析层对轴数没有意见，A/R 两段仍然各自连续。
// 拒绝它的是下降侧的 `isSupportedReduceLayout`（`Reduce.cpp`），理由是 `srcShape` 的外维
// 是一个扁平行数，多根外维轴时中间可能隔着尾块留下的空行。这条用例的作用是把两件事分开：
// lit 里那条 `afir-to-ascendc-reduce-sum-ra-rejected.mlir` 报的是「轴数不支持」，不是
// 「计划建不出来」——两者在能力查询处都映射成 `UnsupportedShape`，光看诊断分不出来。
TEST_F(ReducePlanTest, RaWithMultipleReducedAxesStillBuildsAPlan) {
  std::string body = kS0Body.str();
  const std::string inputAxes = "vectorized_axis = [3, 5], vectorized_strides = [<\"8\">, <\"1\">]";
  const std::string inputRraa =
      "vectorized_axis = [3, 7, 4, 5], vectorized_strides = [<\"256\">, <\"32\">, <\"8\">, <\"1\">]";
  const auto inputAt = body.find(inputAxes);
  ASSERT_NE(inputAt, std::string::npos);
  body.replace(inputAt, inputAxes.size(), inputRraa);
  const std::string axes = "vectorized_axis = [3, 5], vectorized_strides = [<\"1\">, <\"0\">]";
  const std::string rraa = "vectorized_axis = [3, 7, 4, 5], vectorized_strides = [<\"0\">, <\"0\">, <\"8\">, <\"1\">]";
  for (auto at = body.find(axes); at != std::string::npos; at = body.find(axes)) {
    body.replace(at, axes.size(), rraa);
  }

  auto module = Parse(body);
  ASSERT_TRUE(module);
  FailureOr<ReducePlan> plan = buildReducePlan(Subject(*module, "afir.sum"));
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(plan->pattern, ReducePattern::RA);
  EXPECT_EQ(plan->last.axes, (llvm::SmallVector<int64_t, 2>{3, 7}));
  EXPECT_EQ(plan->first.axes, (llvm::SmallVector<int64_t, 2>{4, 5}));
}

// AR 且 R 组有**三根**轴。与上面那条 RA 用例同一个分工：分析层照样建得出计划——A/R 两段
// 各自连续，轴数不是它的判据——拒绝它的是下降侧的 `isSupportedReduceLayout`。
//
// 这条用例存在的理由和 RA 那条一样：把「计划建不出来」和「轴数不支持」分开。两者在能力
// 查询处都映射成 `UnsupportedShape`，光看诊断分不出是哪一种。
//
// 为什么三根就不行（两根可以）：`ReduceInit` 按行填 `[dim_r_current, dim_r)`，是每行的一段
// **后缀**。两根轴时后缀恰好就是「整行 i₀ ∈ [A₀, N₀)」，行内尾巴由 `inner_r` 掩码补上，两支
// 盖满。三根起，`dim_r_current = align(N₂)·A₀·A₁` 不再是任何一条轴边界的后缀：中间那根轴
// 取尾块时，既有垃圾行落在填充区**之前**没被填到，也有有效数据落在填充区**之内**被清零。
TEST_F(ReducePlanTest, ArWithThreeReducedAxesStillBuildsAPlan) {
  std::string body = kS0Body.str();
  const std::string inputAxes = "vectorized_axis = [3, 5], vectorized_strides = [<\"8\">, <\"1\">]";
  const std::string inputArrr =
      "vectorized_axis = [3, 7, 4, 5], vectorized_strides = [<\"256\">, <\"32\">, <\"8\">, <\"1\">]";
  const auto inputAt = body.find(inputAxes);
  ASSERT_NE(inputAt, std::string::npos);
  body.replace(inputAt, inputAxes.size(), inputArrr);
  const std::string axes = "vectorized_axis = [3, 5], vectorized_strides = [<\"1\">, <\"0\">]";
  const std::string arrr = "vectorized_axis = [3, 7, 4, 5], vectorized_strides = [<\"1\">, <\"0\">, <\"0\">, <\"0\">]";
  for (auto at = body.find(axes); at != std::string::npos; at = body.find(axes)) {
    body.replace(at, axes.size(), arrr);
  }

  auto module = Parse(body);
  ASSERT_TRUE(module);
  FailureOr<ReducePlan> plan = buildReducePlan(Subject(*module, "afir.sum"));
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(plan->pattern, ReducePattern::AR);
  EXPECT_EQ(plan->first.axes, (llvm::SmallVector<int64_t, 2>{3}));
  EXPECT_EQ(plan->last.axes, (llvm::SmallVector<int64_t, 2>{7, 4, 5}));
  EXPECT_FALSE(plan->tailLast);
}

// RCore phase 2 的 descriptor。这条不是从调度器跑出来抓的，而是**照着构造它的那段代码**
// 抄的：`optimize/task_generator/reduce_schedule_case_generator.cpp` 的
// `RMulticorePhase2Graph::CreateVarAxis` 建两根轴 `Rm`(id 0)、`A`(id 1)，
// `CompleteNodeAttr` 把 reduce 前的张量写成 `strides = {A_org_size, 1}`、
// `repeats = {Rm_org_size, A_org_size}`，reduce 后写成 `strides = {0, 1}`、
// `repeats = {1, A_org_size}`——被归约的恒是**首维** `Rm`，而且恒**只有一根**。
//
// 也就是说 phase 2 不是「碰巧可能是 RA」，是构造上只能是 RA、且落在本期开放的
// 「R 组单轴」范围内（见设计文档 3.3）。这正是 RA 作为 RCore 前置能力的那一句证明。
//
// `vectorized_*` 在 phase 2 构造完成时还不存在——它由后续 tiling 产生。这里按未切分的
// 二维 UB 张量补成两根轴全向量化，是**重建**不是抓取；能抓取的真实 `[R, A]` 调度产物在
// `test/lit/conversion/afir-to-ascendc-reduce-sum-ra.mlir`，下降侧的 `layout = RA`、
// 不发 `ReduceInit` 由那条用例钉。为什么本环境拿不到 phase 2 自己的下降产物，见设计
// 文档 6.7。
constexpr llvm::StringLiteral kPhase2Body = R"mlir(
    %in = memref.alloc() {afir.buffer_desc = #ub0} : memref<8x16xf32>
    %api = memref.alloc() {afir.buffer_desc = #ub1} : memref<256xui8>
    %acc = memref.alloc() {afir.buffer_desc = #ub2} : memref<256xui8>
    %dst = memref.alloc() {afir.buffer_desc = #ub3} : memref<1x16xf32>
    %sum = afir.sum %in outs(%dst) scratch(%api, %acc) {afir.scratch_roles = ["api_scratch", "round_result"],
      afir.operand_descs = [
        #afir.tensor_desc<axis = [0, 1],
                          repeats = [<"Rm_org_size">, <"A_org_size">],
                          strides = [<"A_org_size">, <"1">],
                          vectorized_axis = [0, 1],
                          vectorized_strides = [<"A_org_size">, <"1">],
                          role = <compute>>],
      afir.destination_descs = [
        #afir.tensor_desc<axis = [0, 1],
                          repeats = [<"1">, <"A_org_size">],
                          strides = [<"0">, <"1">],
                          vectorized_axis = [0, 1],
                          vectorized_strides = [<"0">, <"1">],
                          role = <compute>>],
      afir.result_descs = [
        #afir.tensor_desc<axis = [0, 1],
                          repeats = [<"1">, <"A_org_size">],
                          strides = [<"0">, <"1">],
                          vectorized_axis = [0, 1],
                          vectorized_strides = [<"0">, <"1">],
                          role = <compute>>],
      afir.sched = #afir.sched<axis = [0, 1], loop_axis = 0>
    } : memref<8x16xf32>, memref<1x16xf32>, scratch(memref<256xui8>, memref<256xui8>)
      -> memref<1x16xf32>
)mlir";

TEST_F(ReducePlanTest, PhaseTwoDescriptorIsRaWithASingleReducedAxis) {
  auto module = Parse(kPhase2Body);
  ASSERT_TRUE(module);
  FailureOr<ReducePlan> plan = buildReducePlan(Subject(*module, "afir.sum"));
  ASSERT_TRUE(succeeded(plan));

  EXPECT_EQ(plan->pattern, ReducePattern::RA);
  // `last` 是被归约的那一组：Rm，一根轴——`Reduce.cpp` 的 `isSupportedReduceLayout`
  // 放行 RA 的判据就是这个 `size() == 1`。
  EXPECT_EQ(plan->last.axes, (llvm::SmallVector<int64_t, 2>{0}));
  EXPECT_EQ(plan->first.axes, (llvm::SmallVector<int64_t, 2>{1}));
}

TEST_F(ReducePlanTest, RejectsAlternatingVectorizedAxes) {
  std::string body = kS0Body.str();
  const std::string inputAxes = "vectorized_axis = [3, 5], vectorized_strides = [<\"8\">, <\"1\">]";
  const std::string inputArar =
      "vectorized_axis = [3, 4, 7, 5], vectorized_strides = [<\"256\">, <\"32\">, <\"8\">, <\"1\">]";
  const auto inputAt = body.find(inputAxes);
  ASSERT_NE(inputAt, std::string::npos);
  body.replace(inputAt, inputAxes.size(), inputArar);
  const std::string axes = "vectorized_axis = [3, 5], vectorized_strides = [<\"1\">, <\"0\">]";
  const std::string arar = "vectorized_axis = [3, 4, 7, 5], vectorized_strides = [<\"32\">, <\"0\">, <\"8\">, <\"0\">]";
  for (auto at = body.find(axes); at != std::string::npos; at = body.find(axes)) {
    body.replace(at, axes.size(), arar);
  }

  auto module = Parse(body);
  ASSERT_TRUE(module);
  std::string reason;
  EXPECT_TRUE(failed(buildReducePlan(Subject(*module, "afir.sum"), &reason)));
  EXPECT_NE(reason.find("alternate more than once"), std::string::npos) << reason;
}

// Roles remain attached to operands even when the scheduler changes their order.
TEST_F(ReducePlanTest, ScratchRolesFollowExplicitMetadata) {
  std::string body = kS0Body.str();
  auto at = body.find("scratch(%api, %acc)");
  body.replace(at, std::string("scratch(%api, %acc)").size(), "scratch(%acc, %api)");
  const std::string roles = R"(["api_scratch", "round_result"])";
  at = body.find(roles);
  body.replace(at, roles.size(), R"(["round_result", "api_scratch"])");
  auto module = Parse(body);
  ASSERT_TRUE(module);
  auto plan = buildReducePlan(Subject(*module, "afir.sum"));
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(plan->apiScratchIndex, 1U);
  ASSERT_TRUE(plan->accumulationScratchIndex.has_value());
  EXPECT_EQ(*plan->accumulationScratchIndex, 0U);
}

TEST_F(ReducePlanTest, RejectsInvalidScratchRoles) {
  auto module = Parse(kS0Body);
  ASSERT_TRUE(module);
  auto *op = Subject(*module, "afir.sum");
  Builder builder(&context);
  const SmallVector<std::pair<SmallVector<StringRef>, StringRef>> invalid = {
      {{}, "one role per scratch operand"},
      {{"api_scratch"}, "one role per scratch operand"},
      {{"api_scratch", "api_scratch"}, "duplicate reduce scratch role"},
      {{"api_scratch", "unknown"}, "unknown reduce scratch role"},
      {{"round_result", "round_result"}, "duplicate reduce scratch role"}};
  std::string message;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
    llvm::raw_string_ostream os(message);
    diagnostic.print(os);
    return success();
  });
  for (const auto &[roles, expected] : invalid) {
    message.clear();
    op->setAttr(kReduceScratchRolesAttr, builder.getStrArrayAttr(roles));
    EXPECT_TRUE(failed(verify(op)));
    EXPECT_NE(message.find(expected.str()), std::string::npos) << message;
    std::string reason;
    EXPECT_TRUE(failed(buildReducePlan(op, &reason)));
    EXPECT_NE(reason.find(expected.str()), std::string::npos) << reason;
  }
  op->removeAttr(kReduceScratchRolesAttr);
  EXPECT_TRUE(failed(verify(op)));
  EXPECT_TRUE(failed(buildReducePlan(op)));
  op->setAttr(kReduceScratchRolesAttr,
              builder.getArrayAttr({builder.getI64IntegerAttr(0), builder.getStringAttr("round_result")}));
  EXPECT_TRUE(failed(verify(op)));
}

TEST_F(ReducePlanTest, RejectsScratchRolesOnTensorFormWithoutScratchOperands) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
module {
  func.func @tensor_reduce(%input: tensor<2x8xf32>) -> tensor<2xf32> {
    %result = afir.sum %input {axis = 1 : i32} : tensor<2x8xf32> -> tensor<2xf32>
    return %result : tensor<2xf32>
  }
})mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto *op = Subject(*module, "afir.sum");
  Builder builder(&context);
  op->setAttr(kReduceScratchRolesAttr, builder.getStrArrayAttr({"api_scratch"}));
  std::string message;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
    llvm::raw_string_ostream os(message);
    diagnostic.print(os);
    return success();
  });
  EXPECT_TRUE(failed(verify(op)));
  EXPECT_NE(message.find("tensor form cannot carry scratch buffers or afir.scratch_roles"), std::string::npos)
      << message;
}

TEST_F(ReducePlanTest, RejectsScratchRolesOnNonReduceOperation) {
  auto module = Parse(kS0Body);
  ASSERT_TRUE(module);
  auto *op = Subject(*module, "afir.sum");
  Operation &alloc = op->getBlock()->front();
  ASSERT_TRUE(isa<memref::AllocOp>(alloc));
  alloc.setAttr(kReduceScratchRolesAttr, op->getAttr(kReduceScratchRolesAttr));
  std::string message;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
    llvm::raw_string_ostream os(message);
    diagnostic.print(os);
    return success();
  });
  EXPECT_TRUE(failed(verify(&alloc)));
  EXPECT_NE(message.find("InvalidDescriptor: afir.scratch_roles is only valid on reduce operations"), std::string::npos)
      << message;
}

// 累加要用第二块 scratch 存本轮的段结果。只给一块时不能「凑合着用 API 那块」——
// 那会在 API 调用期间被覆盖。这里必须失败，由调用方报 `MissingInformation`。
TEST_F(ReducePlanTest, RejectsAccumulationWithoutItsOwnScratch) {
  std::string body = kS0Body.str();
  const std::string twoScratch = "scratch(%api, %acc)";
  const auto at = body.find(twoScratch);
  ASSERT_NE(at, std::string::npos);
  body.replace(at, twoScratch.size(), "scratch(%api)");
  const std::string twoTypes = "scratch(memref<256xui8>, memref<256xui8>)";
  const auto typesAt = body.find(twoTypes);
  ASSERT_NE(typesAt, std::string::npos);
  body.replace(typesAt, twoTypes.size(), "scratch(memref<256xui8>)");
  const std::string roles = R"(["api_scratch", "round_result"])";
  body.replace(body.find(roles), roles.size(), R"(["api_scratch"])");
  auto module = Parse(body);
  ASSERT_TRUE(module);
  std::string reason;
  EXPECT_TRUE(failed(buildReducePlan(Subject(*module, "afir.sum"), &reason)));
  EXPECT_NE(reason.find("round_result"), std::string::npos) << reason;
}

// 以下两个用例钉的是**依赖方向**，不是某个具体推论：一个归约沿哪些轴累加，与「调度器给了
// 几块 scratch」和「这块布局能不能表示成二维 AR/RA」都无关。两者此前压在同一个
// `FailureOr` 里，后果是 scratch 少给一块时 `LoopGenerate` 报 `InvalidSchedule`——把资源
// 问题归因成调度标注破了。
//
// 判据刻意不是「LoopGenerate 不调某个函数」：那约束的是代码长相，缓存一份完整计划就能
// 绕过。这里改成问同一个语义在不同资源/布局条件下答案是否一致。
TEST_F(ReducePlanTest, AccumulationAnalysisIgnoresScratchCount) {
  // 基线（两块 scratch）的累加轴，就是下面两个变体必须复现的答案。
  auto baseline = Parse(kS0Body);
  ASSERT_TRUE(baseline);
  FailureOr<ReduceAccumulationInfo> expected = analyzeReduceAccumulation(Subject(*baseline, "afir.sum"));
  ASSERT_TRUE(succeeded(expected));
  ASSERT_EQ(expected->accumulationAxes.size(), 1U);
  EXPECT_EQ(expected->accumulationAxes.front(), 4);

  // scratch 减到一块、再减到零块：descriptor 与 schedule 一字未改，所以语义答案必须不变，
  // 而完整计划必须按原规则拒绝。
  struct Variant {
    const char *operands;
    const char *types;
    const char *planReason;
  };
  // `scratch` 在 ODS 里是锚在变参上的**可选组**（`Reduce.td:55,57`），零块时整组连同它前面
  // 那个逗号一起消失——所以类型那一侧的替换必须把逗号算进来，否则留下悬空逗号，报的是
  // `expected 'scratch'`，看着像语法不支持零 scratch。
  const Variant variants[] = {
      {"scratch(%api)", ", scratch(memref<256xui8>)", "round_result"},
      {"", "", "afir.scratch_roles"},
  };
  for (const Variant &variant : variants) {
    std::string body = kS0Body.str();
    const std::string twoScratch = "scratch(%api, %acc)";
    const auto at = body.find(twoScratch);
    ASSERT_NE(at, std::string::npos);
    body.replace(at, twoScratch.size(), variant.operands);
    const std::string twoTypes = ", scratch(memref<256xui8>, memref<256xui8>)";
    const auto typesAt = body.find(twoTypes);
    ASSERT_NE(typesAt, std::string::npos);
    body.replace(typesAt, twoTypes.size(), variant.types);
    const std::string roles = R"(afir.scratch_roles = ["api_scratch", "round_result"],)";
    body.replace(body.find(roles), roles.size(), *variant.operands ? R"(afir.scratch_roles = ["api_scratch"], )" : "");

    auto module = Parse(body);
    ASSERT_TRUE(module) << variant.operands;
    Operation *subject = Subject(*module, "afir.sum");
    ASSERT_NE(subject, nullptr) << variant.operands;

    FailureOr<ReduceAccumulationInfo> actual = analyzeReduceAccumulation(subject);
    ASSERT_TRUE(succeeded(actual)) << "scratch 个数不该影响累加语义: " << variant.operands;
    ASSERT_EQ(actual->accumulationAxes.size(), expected->accumulationAxes.size()) << variant.operands;
    for (auto [index, axisId] : llvm::enumerate(expected->accumulationAxes)) {
      EXPECT_EQ(actual->accumulationAxes[index], axisId) << variant.operands;
    }

    std::string reason;
    EXPECT_TRUE(failed(buildReducePlan(subject, &reason))) << "资源不足仍要由完整计划拒绝";
    EXPECT_NE(reason.find(variant.planReason), std::string::npos) << reason;
  }
}

// A/R 交替超过一次：语义上完全清楚（归约维就是那两根 stride 为 0 的），只是**当前**高阶
// API 的二维 AR/RA 表示不了。这类限制不该让循环生成拿不到累加轴。
TEST_F(ReducePlanTest, AccumulationAnalysisIgnoresTwoDimensionalExpressibility) {
  std::string body = kS0Body.str();
  const std::string inputAxes = "vectorized_axis = [3, 5], vectorized_strides = [<\"8\">, <\"1\">]";
  const std::string inputArar =
      "vectorized_axis = [3, 4, 7, 5], vectorized_strides = [<\"256\">, <\"32\">, <\"8\">, <\"1\">]";
  const auto inputAt = body.find(inputAxes);
  ASSERT_NE(inputAt, std::string::npos);
  body.replace(inputAt, inputAxes.size(), inputArar);
  const std::string axes = "vectorized_axis = [3, 5], vectorized_strides = [<\"1\">, <\"0\">]";
  const std::string arar = "vectorized_axis = [3, 4, 7, 5], vectorized_strides = [<\"32\">, <\"0\">, <\"8\">, <\"0\">]";
  for (auto at = body.find(axes); at != std::string::npos; at = body.find(axes)) {
    body.replace(at, axes.size(), arar);
  }

  auto module = Parse(body);
  ASSERT_TRUE(module);
  Operation *subject = Subject(*module, "afir.sum");
  ASSERT_NE(subject, nullptr);

  FailureOr<ReduceAccumulationInfo> accumulation = analyzeReduceAccumulation(subject);
  ASSERT_TRUE(succeeded(accumulation)) << "二维可表达性不该影响累加语义";
  ASSERT_EQ(accumulation->accumulationAxes.size(), 1U);
  EXPECT_EQ(accumulation->accumulationAxes.front(), 4);

  std::string reason;
  EXPECT_TRUE(failed(buildReducePlan(subject, &reason))) << "二维化不成立仍要由完整计划拒绝";
  EXPECT_NE(reason.find("alternate more than once"), std::string::npos) << reason;
}

// 语义信息本身缺失时，语义查询必须失败——它不是「永远成功」的宽松版本。否则
// LoopGenerate 会把标注破了的 op 当成合法非累加，放行一个静默算错值的重定向。
TEST_F(ReducePlanTest, AccumulationAnalysisStillRejectsBrokenSemantics) {
  std::string body = kS0Body.str();
  const std::string sched = "afir.sched = #afir.sched<axis = [6, 7, 4, 3, 5], loop_axis = 4>";
  const auto at = body.find(sched);
  ASSERT_NE(at, std::string::npos);
  body.replace(at, sched.size(), "afir.sched = #afir.sched<axis = [6, 7, 4, 3, 5], loop_axis = -1>");

  auto module = Parse(body);
  ASSERT_TRUE(module);
  std::string reason;
  EXPECT_TRUE(failed(analyzeReduceAccumulation(Subject(*module, "afir.sum"), &reason)));
  EXPECT_NE(reason.find("loop axis"), std::string::npos) << reason;
}

// S2 全载：`loop_axis` 就在归约维之内，一次算完，不跨轮。累加轴为空，因而也不需要
// 第二块 scratch——`isAccumulating()` 是「要不要跨作用域」的唯一判据，两处都问它。
TEST_F(ReducePlanTest, AllLoadFormDoesNotAccumulate) {
  std::string body = kS0Body.str();
  const std::string s0Sched = "#afir.sched<axis = [6, 7, 4, 3, 5], loop_axis = 4>";
  const auto at = body.find(s0Sched);
  ASSERT_NE(at, std::string::npos);
  body.replace(at, s0Sched.size(), "#afir.sched<axis = [6, 7, 3, 4, 5], loop_axis = 7>");

  auto module = Parse(body);
  ASSERT_TRUE(module);
  FailureOr<ReducePlan> plan = buildReducePlan(Subject(*module, "afir.sum"));
  ASSERT_TRUE(succeeded(plan));

  EXPECT_FALSE(plan->isAccumulating());
  EXPECT_TRUE(plan->accumulationAxes.empty());
  EXPECT_FALSE(plan->accumulationScratchIndex.has_value());
}

// 归约维落在前一维 -> RA。`first` / `last` 跟着换位：`last` 始终是被归约的那一维。
TEST_F(ReducePlanTest, ReducedLeadingAxisIsRa) {
  std::string body = kS0Body.str();
  const std::string ar = "vectorized_axis = [3, 5], vectorized_strides = [<\"1\">, <\"0\">]";
  const std::string ra = "vectorized_axis = [3, 5], vectorized_strides = [<\"0\">, <\"1\">]";
  for (auto at = body.find(ar); at != std::string::npos; at = body.find(ar)) {
    body.replace(at, ar.size(), ra);
  }

  auto module = Parse(body);
  ASSERT_TRUE(module);
  FailureOr<ReducePlan> plan = buildReducePlan(Subject(*module, "afir.sum"));
  ASSERT_TRUE(succeeded(plan));

  EXPECT_EQ(plan->pattern, ReducePattern::RA);
  ASSERT_EQ(plan->first.axes.size(), 1U);
  EXPECT_EQ(plan->first.axes.front(), 5);
  ASSERT_EQ(plan->last.axes.size(), 1U);
  EXPECT_EQ(plan->last.axes.front(), 3);
}

// 两维都被归约：既不是 AR 也不是 RA，高阶 API 的二维布局表达不了，本期拒绝。
// 判不出形态时必须失败而不是猜一个——猜错的方向是静默算错。
TEST_F(ReducePlanTest, RejectsBothVectorizedAxesReduced) {
  std::string body = kS0Body.str();
  const std::string ar = "vectorized_axis = [3, 5], vectorized_strides = [<\"1\">, <\"0\">]";
  const std::string both = "vectorized_axis = [3, 5], vectorized_strides = [<\"0\">, <\"0\">]";
  for (auto at = body.find(ar); at != std::string::npos; at = body.find(ar)) {
    body.replace(at, ar.size(), both);
  }

  auto module = Parse(body);
  ASSERT_TRUE(module);
  std::string reason;
  EXPECT_TRUE(failed(buildReducePlan(Subject(*module, "afir.sum"), &reason)));
  // 两根向量化轴都被归约，落的是「保留轴为空」那一支，不是 `collectReducedAxes`——
  // 那一支要等到下面 elementwise 那条才走得到。
  EXPECT_NE(reason.find("no kept axis"), std::string::npos) << reason;
}

// elementwise：destination 每一维都前进，没有归约维。这条是 LoopGenerate 放行判据的
// 反例侧——`buildReducePlan` 在这里必须失败，跨作用域的 `afir.add` 才会保留原来那条
// 「tensor SSA value crosses ...」诊断，而不是被当成累加放行。
TEST_F(ReducePlanTest, ElementwiseHasNoReductionAxis) {
  auto module = Parse(R"mlir(
    %lhs = memref.alloc() {afir.buffer_desc = #ub0} : memref<2x4xf32>
    %rhs = memref.alloc() {afir.buffer_desc = #ub1} : memref<2x4xf32>
    %dst = memref.alloc() {afir.buffer_desc = #ub2} : memref<2x4xf32>
    %out = afir.add %lhs, %rhs outs(%dst) {
      afir.operand_descs = [
        #afir.tensor_desc<axis = [0, 1], strides = [<"4">, <"1">],
                          vectorized_axis = [0, 1], vectorized_strides = [<"4">, <"1">],
                          role = <compute>>,
        #afir.tensor_desc<axis = [0, 1], strides = [<"4">, <"1">],
                          vectorized_axis = [0, 1], vectorized_strides = [<"4">, <"1">],
                          role = <compute>>],
      afir.destination_descs = [
        #afir.tensor_desc<axis = [0, 1], strides = [<"4">, <"1">],
                          vectorized_axis = [0, 1], vectorized_strides = [<"4">, <"1">],
                          role = <compute>>],
      afir.result_descs = [
        #afir.tensor_desc<axis = [0, 1], strides = [<"4">, <"1">],
                          vectorized_axis = [0, 1], vectorized_strides = [<"4">, <"1">],
                          role = <compute>>],
      afir.sched = #afir.sched<axis = [0, 1], loop_axis = 0>
    } : (memref<2x4xf32>, memref<2x4xf32>), memref<2x4xf32> -> memref<2x4xf32>
  )mlir");
  ASSERT_TRUE(module);
  std::string reason;
  EXPECT_TRUE(failed(buildReducePlan(Subject(*module, "afir.add"), &reason)));
  EXPECT_NE(reason.find("no axis is reduced"), std::string::npos) << reason;
}

//===----------------------------------------------------------------------===//
// Mean 的完整逻辑归约域
//===----------------------------------------------------------------------===//

// kS0Body 的归约维是 z1T(4) 与 z1t(5)，两根都是 z1 切出来的。域必须收敛成**一根** z1，
// 尺寸取它的 `size`——而不是两根分别的尺寸、更不是它们的乘积。
//
// 这条是整个 Mean 修正的地基：切分出来的 inner/outer 指向同一个来源，重复计数会让分母变成
// `R^2` 量级，结果小到近乎 0；而取其中一根的 tile 尺寸则是「均值的均值」。
TEST_F(ReducePlanTest, MeanDomainDeduplicatesTheSplitSource) {
  auto module = Parse(kS0Body, R"mlir(#afir.asc_graph<tiling_key = 0,axes = [
      <id = 1, name = "z1", axis_type = Original, size = <"32">, align = <"-1">>,
      <id = 4, name = "z1T", axis_type = TileOuter, size = <"4">, align = <"1">, from = [1]>,
      <id = 5, name = "z1t", axis_type = TileInner, size = <"8">, align = <"1">, from = [1]>],
      type = Compute, size_var = [<"unused">]>)mlir");
  ASSERT_TRUE(module);
  std::string reason;
  FailureOr<LogicalReductionDomain> domain = analyzeLogicalReductionDomain(Subject(*module, "afir.sum"), &reason);
  ASSERT_TRUE(succeeded(domain)) << reason;
  ASSERT_EQ(domain->originalAxes.size(), 1U);
  EXPECT_EQ(domain->originalAxes.front(), 1);
  ASSERT_EQ(domain->sizes.size(), 1U);
  EXPECT_EQ(domain->sizes.front().getValue().getValue(), "32");
}

// 多根独立的 R 轴：域是它们的并集，`N` 是乘积。顺序按 destination 的轴序，不按哈希序——
// 顺序进产物（`afir.symbolic_eval` 的操作数序），哈希序会让同一份输入编出不同的 kernel。
TEST_F(ReducePlanTest, MeanDomainKeepsEveryIndependentReducedAxisInDescriptorOrder) {
  auto module = Parse(kS0Body, R"mlir(#afir.asc_graph<tiling_key = 0,axes = [
      <id = 4, name = "r0", axis_type = Original, size = <"4">, align = <"1">>,
      <id = 5, name = "r1", axis_type = Original, size = <"8">, align = <"1">>],
      type = Compute, size_var = [<"unused">]>)mlir");
  ASSERT_TRUE(module);
  FailureOr<LogicalReductionDomain> domain = analyzeLogicalReductionDomain(Subject(*module, "afir.sum"));
  ASSERT_TRUE(succeeded(domain));
  ASSERT_EQ(domain->originalAxes.size(), 2U);
  // destination 的轴序是 [6, 7, 3, 4, 5]，所以 4 在 5 前面。
  EXPECT_EQ(domain->originalAxes[0], 4);
  EXPECT_EQ(domain->originalAxes[1], 5);
  EXPECT_EQ(domain->sizes[0].getValue().getValue(), "4");
  EXPECT_EQ(domain->sizes[1].getValue().getValue(), "8");
}

// 符号尺寸照样是合法的域：`N` 在运行期才知道值，但它是**调度输入**的符号，不随 tile 变。
TEST_F(ReducePlanTest, MeanDomainAcceptsASymbolicOriginalSize) {
  auto module = Parse(kS0Body, R"mlir(#afir.asc_graph<tiling_key = 0,axes = [
      <id = 1, name = "z1", axis_type = Original, size = <"R_org_size">, align = <"-1">>,
      <id = 4, name = "z1T", axis_type = TileOuter, size = <"4">, align = <"1">, from = [1]>,
      <id = 5, name = "z1t", axis_type = TileInner, size = <"8">, align = <"1">, from = [1]>],
      type = Compute, size_var = [<"unused">]>)mlir");
  ASSERT_TRUE(module);
  FailureOr<LogicalReductionDomain> domain = analyzeLogicalReductionDomain(Subject(*module, "afir.sum"));
  ASSERT_TRUE(succeeded(domain));
  ASSERT_EQ(domain->sizes.size(), 1U);
  EXPECT_EQ(domain->sizes.front().getValue().getValue(), "R_org_size");
}

// 来源丢失：`z1t` 既不是 Original、`from` 又为空。必须**失败**，不能退回它自己的尺寸——
// 退回去就是拿一轮的元素数当总数。
TEST_F(ReducePlanTest, MeanDomainRejectsLostProvenance) {
  auto module = Parse(kS0Body, R"mlir(#afir.asc_graph<tiling_key = 0,axes = [
      <id = 4, name = "z1T", axis_type = TileOuter, size = <"4">, align = <"1">>,
      <id = 5, name = "z1t", axis_type = TileInner, size = <"8">, align = <"1">>],
      type = Compute, size_var = [<"unused">]>)mlir");
  ASSERT_TRUE(module);
  std::string reason;
  EXPECT_TRUE(failed(analyzeLogicalReductionDomain(Subject(*module, "afir.sum"), &reason)));
  EXPECT_NE(reason.find("provenance is lost"), std::string::npos) << reason;
}

// 归约维不在轴表里：同样失败。这一档与上一档的原因不同（一个是链断了、一个是轴压根没声明），
// 诊断也要分开说，否则查的人会去翻 `from` 而不是翻轴表。
TEST_F(ReducePlanTest, MeanDomainRejectsAnUndeclaredReducedAxis) {
  auto module = Parse(kS0Body, R"mlir(#afir.asc_graph<tiling_key = 0,axes = [
      <id = 4, name = "z1T", axis_type = Original, size = <"4">, align = <"1">>],
      type = Compute, size_var = [<"unused">]>)mlir");
  ASSERT_TRUE(module);
  std::string reason;
  EXPECT_TRUE(failed(analyzeLogicalReductionDomain(Subject(*module, "afir.sum"), &reason)));
  EXPECT_NE(reason.find("not declared in the asc_graph axis list"), std::string::npos) << reason;
}

// 字面量尺寸非正：失败。这一档必须在**本函数**里失败，而不是留给下降侧的物化去挡——
// 能力查询只调本函数，两处判据不一致就会出现「能力说支持、下降才失败」，候选被丢掉而
// 诊断停在「下降失败」这一层，查的人看不到真因。
TEST_F(ReducePlanTest, MeanDomainRejectsANonPositiveLiteralSize) {
  auto module = Parse(kS0Body, R"mlir(#afir.asc_graph<tiling_key = 0,axes = [
      <id = 1, name = "z1", axis_type = Original, size = <"0">, align = <"-1">>,
      <id = 4, name = "z1T", axis_type = TileOuter, size = <"4">, align = <"1">, from = [1]>,
      <id = 5, name = "z1t", axis_type = TileInner, size = <"8">, align = <"1">, from = [1]>],
      type = Compute, size_var = [<"unused">]>)mlir");
  ASSERT_TRUE(module);
  std::string reason;
  EXPECT_TRUE(failed(analyzeLogicalReductionDomain(Subject(*module, "afir.sum"), &reason)));
  EXPECT_NE(reason.find("non-positive literal size"), std::string::npos) << reason;
}

// 没有轴表：失败。缺这张表时连「哪根轴是 Original」都答不了。
TEST_F(ReducePlanTest, MeanDomainRejectsAMissingAxisList) {
  auto module = Parse(kS0Body);
  ASSERT_TRUE(module);
  std::string reason;
  EXPECT_TRUE(failed(analyzeLogicalReductionDomain(Subject(*module, "afir.sum"), &reason)));
  EXPECT_NE(reason.find("no asc_graph axis list"), std::string::npos) << reason;
}

// 域分析与**累加语义**是两件事，不能互相牵连：这份输入跨轮累加（`isAccumulating()` 为真），
// 而域照样求得出来。反过来 AllLoad 不累加，域也一样求得出来——分母与「分几轮算」无关，
// 这正是本次修正的核心。
TEST_F(ReducePlanTest, MeanDomainIsIndependentOfAccumulation) {
  auto module = Parse(kS0Body, R"mlir(#afir.asc_graph<tiling_key = 0,axes = [
      <id = 1, name = "z1", axis_type = Original, size = <"32">, align = <"-1">>,
      <id = 4, name = "z1T", axis_type = TileOuter, size = <"4">, align = <"1">, from = [1]>,
      <id = 5, name = "z1t", axis_type = TileInner, size = <"8">, align = <"1">, from = [1]>],
      type = Compute, size_var = [<"unused">]>)mlir");
  ASSERT_TRUE(module);
  Operation *subject = Subject(*module, "afir.sum");
  FailureOr<ReducePlan> plan = buildReducePlan(subject);
  ASSERT_TRUE(succeeded(plan));
  ASSERT_TRUE(plan->isAccumulating());
  FailureOr<LogicalReductionDomain> domain = analyzeLogicalReductionDomain(subject);
  ASSERT_TRUE(succeeded(domain));
  ASSERT_EQ(domain->sizes.size(), 1U);
  EXPECT_EQ(domain->sizes.front().getValue().getValue(), "32");
}

//===----------------------------------------------------------------------===//
// 归约域的来源证明：属性的写（annotate）与读（getProven）
//===----------------------------------------------------------------------===//

// 本节的 IR 是 **bridge 产物形态**：reduce 的输入是上一个算子的 **SSA 结果**。这一点是整件
// 事的前提——`AFIRMemoryAllocate` 之后消费者改读 destination，SSA 边断掉，来源就再也查不出
// 来了，所以证明必须在这个形态上做一次、写进 IR。
//
// 下面每一条都只换 mean 的**输入来源**，mean 自己（描述子、轴图）逐字相同。这样「有没有写
// 属性」的差别只可能来自来源，不会与「描述子是不是齐全」混在一起。

// 归约域：z1(32) 被切成 z1T/z1t。与上一节同一张轴图。
constexpr llvm::StringLiteral kDomainGraph = R"mlir(#afir.asc_graph<tiling_key = 0,axes = [
      <id = 1, name = "z1", axis_type = Original, size = <"32">, align = <"-1">>,
      <id = 4, name = "z1T", axis_type = TileOuter, size = <"4">, align = <"1">, from = [1]>,
      <id = 5, name = "z1t", axis_type = TileInner, size = <"8">, align = <"1">, from = [1]>],
      type = Compute, size_var = [<"unused">]>)mlir";

// `[6, 7, 3, 4, 5]` 五维，z1T(4)/z1t(5) 在 destination 上 stride 为 0，即被归约的两维。
constexpr llvm::StringLiteral kFiveDimCompute = R"mlir(
        #afir.tensor_desc<axis = [6, 7, 3, 4, 5],
                          strides = [<"128">, <"64">, <"32">, <"8">, <"1">],
                          vectorized_axis = [3, 5], vectorized_strides = [<"8">, <"1">],
                          role = <compute>>)mlir";
constexpr llvm::StringLiteral kFiveDimReduced = R"mlir(
        #afir.tensor_desc<axis = [6, 7, 3, 4, 5],
                          strides = [<"4">, <"2">, <"1">, <"0">, <"0">],
                          vectorized_axis = [3, 5], vectorized_strides = [<"1">, <"0">],
                          role = <compute>>)mlir";

// 被测的那个 mean。`$INPUT` 由各用例替换成不同来源的值。
std::string MeanOver(llvm::StringRef input) {
  std::string descs = "      afir.operand_descs = [" + kFiveDimCompute.str() + "],\n" +
                      "      afir.destination_descs = [" + kFiveDimReduced.str() + "],\n" +
                      "      afir.result_descs = [" + kFiveDimReduced.str() + "],\n" +
                      "      afir.sched = #afir.sched<axis = [6, 7, 4, 3, 5], loop_axis = 4>";
  return R"mlir(
    %api = memref.alloc() {afir.buffer_desc = #ub1} : memref<256xui8>
    %acc = memref.alloc() {afir.buffer_desc = #ub2} : memref<256xui8>
    %dst = memref.alloc() {afir.buffer_desc = #ub3} : memref<2x2x2x1x1xf32>
    %mean = afir.mean )mlir" +
         input.str() + R"mlir( outs(%dst) scratch(%api, %acc) {afir.scratch_roles = ["api_scratch", "round_result"],
)mlir" + descs +
         R"mlir(
    } : memref<2x2x2x4x8xf32>, memref<2x2x2x1x1xf32>, scratch(memref<256xui8>, memref<256xui8>)
      -> memref<2x2x2x1x1xf32>
)mlir";
}

// 一次归约，结果与输入同形（调度后被归约的维保留、stride 为 0），好让它直接喂给 mean。
// `reducedDescs` 决定它归约的是哪几根轴——「归约了另一根轴」与「归约了同一根轴」是两件不同
// 的事，用例要分开说。
std::string UpstreamReduce(llvm::StringRef result, llvm::StringRef reducedDescs) {
  std::string descs = "      afir.operand_descs = [" + kFiveDimCompute.str() + "],\n" +
                      "      afir.destination_descs = [" + reducedDescs.str() + "],\n" + "      afir.result_descs = [" +
                      reducedDescs.str() + "],\n" +
                      "      afir.sched = #afir.sched<axis = [6, 7, 4, 3, 5], loop_axis = 4>";
  return R"mlir(
    %psrc = memref.alloc() {afir.buffer_desc = #ub0} : memref<2x2x2x4x8xf32>
    %papi = memref.alloc() {afir.buffer_desc = #ub1} : memref<256xui8>
    %pacc = memref.alloc() {afir.buffer_desc = #ub2} : memref<256xui8>
    %pdst = memref.alloc() {afir.buffer_desc = #ub3} : memref<2x2x2x4x8xf32>
    )mlir" +
         result.str() +
         R"mlir( = afir.sum %psrc outs(%pdst) scratch(%papi, %pacc) {afir.scratch_roles = ["api_scratch", "round_result"],
)mlir" + descs +
         R"mlir(
    } : memref<2x2x2x4x8xf32>, memref<2x2x2x4x8xf32>, scratch(memref<256xui8>, memref<256xui8>)
      -> memref<2x2x2x4x8xf32>
)mlir";
}

// 归约 z0t(3) 那一根——与 mean 归约的 z1T/z1t 不是同一根。对应
// `x -> Sum(dim=C) -> Mean(dim=B)`：上游是**已经算完**的另一次归约。
constexpr llvm::StringLiteral kFiveDimOtherAxisReduced = R"mlir(
        #afir.tensor_desc<axis = [6, 7, 3, 4, 5],
                          strides = [<"32">, <"16">, <"0">, <"8">, <"1">],
                          vectorized_axis = [3, 5], vectorized_strides = [<"8">, <"1">],
                          role = <compute>>)mlir";

// 一次 elementwise，同样同形。
std::string ElementwiseOver(llvm::StringRef input, llvm::StringRef result, llvm::StringRef destination) {
  std::string descs = "      afir.operand_descs = [" + kFiveDimCompute.str() + ", " + kFiveDimCompute.str() + "],\n" +
                      "      afir.destination_descs = [" + kFiveDimCompute.str() + "],\n" +
                      "      afir.result_descs = [" + kFiveDimCompute.str() + "],\n" +
                      "      afir.sched = #afir.sched<axis = [6, 7, 4, 3, 5], loop_axis = 4>";
  return "\n    " + result.str() + " = afir.add " + input.str() + ", " + input.str() + " outs(" + destination.str() +
         ") {\n" + descs + R"mlir(
    } : (memref<2x2x2x4x8xf32>, memref<2x2x2x4x8xf32>), memref<2x2x2x4x8xf32> -> memref<2x2x2x4x8xf32>
)mlir";
}

std::string OneKernel(llvm::StringRef body) {
  return R"mlir(
module {
  func.func private @kernel() attributes {
    afir.function_role = #afir.function_role<kernel_body>,
    afir.asc_graph = )mlir" +
         kDomainGraph.str() + R"mlir(
  } {
)mlir" + body.str() +
         R"mlir(
    return
  }
}
)mlir";
}

// 输入直接来自一块干净的缓冲（对应真实产物里 `afir.load` 搬进来的 tile）：来源一路没有归约，
// 轴图说的就是全部，属性必须被写出，且与轴图逐项一致。
TEST_F(ReducePlanTest, ReductionDomainIsProvenForANonReducedInput) {
  std::string body = R"mlir(
    %in = memref.alloc() {afir.buffer_desc = #ub0} : memref<2x2x2x4x8xf32>
)mlir" + MeanOver("%in");
  auto module = ParseModule(OneKernel(body));
  ASSERT_TRUE(module);
  annotateReductionDomains(*module);

  Operation *mean = Subject(*module, "afir.mean");
  ASSERT_TRUE(mean);
  std::string reason;
  FailureOr<LogicalReductionDomain> domain = getProvenReductionDomain(mean, &reason);
  ASSERT_TRUE(succeeded(domain)) << reason;
  ASSERT_EQ(domain->originalAxes.size(), 1U);
  EXPECT_EQ(domain->originalAxes.front(), 1);
  ASSERT_EQ(domain->sizes.size(), 1U);
  EXPECT_EQ(domain->sizes.front().getValue().getValue(), "32");
}

// 上游是用户图里**已经算完**的另一次归约（`x -> Sum(dim=C) -> Mean(dim=B)`）：必须**放行**，
// 而且分母是 mean 自己的域，不是上游那次的。
//
// 这一条守的是「判据不是算子类别」。同一个 func 内轴图的 `from` 链原封不动，mean 的归约轴顺
// 着它走到的就是前端图里真正的 Original 轴；核内跨轮累加在单个 reduce 算子**内部**完成，
// 不是两个算子之间，所以上游有没有归约过都不影响这张轴图的结论。
//
// 本次改动之前这种图是支持的（旧判据只看阶段号）。按算子类别判会把它判成未证明——那是一次
// 能力回退，而不是更安全。
TEST_F(ReducePlanTest, ReductionDomainIsProvenAfterACompletedUpstreamReduction) {
  std::string body = UpstreamReduce("%upstream", kFiveDimOtherAxisReduced) + MeanOver("%upstream");
  auto module = ParseModule(OneKernel(body));
  ASSERT_TRUE(module);
  annotateReductionDomains(*module);

  Operation *mean = Subject(*module, "afir.mean");
  ASSERT_TRUE(mean);
  std::string reason;
  FailureOr<LogicalReductionDomain> domain = getProvenReductionDomain(mean, &reason);
  ASSERT_TRUE(succeeded(domain)) << reason;
  // 分母是 z1(32)——mean 自己的域。不是上游归约的那根轴，也不是两者的乘积。
  ASSERT_EQ(domain->originalAxes.size(), 1U);
  EXPECT_EQ(domain->originalAxes.front(), 1);
  EXPECT_EQ(domain->sizes.front().getValue().getValue(), "32");
}

// 输入是一次 **elementwise** 的中间结果：来源里没有归约，必须照样算证明得出。
// 这一条守的是方案点名不要的那种保守——「中间结果一律拒」会把合法形态一起挡掉。
TEST_F(ReducePlanTest, ReductionDomainIsProvenThroughAnElementwiseProducer) {
  std::string body = R"mlir(
    %in = memref.alloc() {afir.buffer_desc = #ub0} : memref<2x2x2x4x8xf32>
    %edst = memref.alloc() {afir.buffer_desc = #ub3} : memref<2x2x2x4x8xf32>
)mlir" + ElementwiseOver("%in", "%el", "%edst") +
                     MeanOver("%el");
  auto module = ParseModule(OneKernel(body));
  ASSERT_TRUE(module);
  annotateReductionDomains(*module);
  Operation *mean = Subject(*module, "afir.mean");
  ASSERT_TRUE(mean);
  EXPECT_TRUE(mean->hasAttr(kReductionDomainAttrName));
}

// **跨阶段**：partial 不经 SSA、经一块 GM workspace 传给下一个 func，两端之间只有 buffer id
// 这一个共同身份。写那块 workspace 的算子在另一个 func 里，所以这件事只有在 module 这一层
// 才看得见。
//
// 与上一条对照着看：同样是「上游有一次归约」，在**同一个 func** 内放行，跨 workspace 则拒。
// 差别不在算子类别，在于跨过去之后轴图是分期改写新合成的，`from` 链不再指回前端图。
//
// 跨阶段这一侧的判据**是**保守的：写进 workspace 的只要经过归约就拒，不区分「算完的结果」
// 与「等待合并的 partial」。要分开这两者，判据得来自分期改写自己的语义身份，不能由这里从算子
// 类别反猜；今天唯一的分期形态是 RCore，其第二阶段是 `Sum`，所以这一档不丢现有能力。
TEST_F(ReducePlanTest, ReductionDomainIsNotProvenAcrossAWorkspaceHoldingAPartial) {
  std::string producer = UpstreamReduce("%partial", kFiveDimReduced) +
                         R"mlir(
    %ws = memref.alloc() {afir.buffer_desc = #gm9} : memref<2x2x2x4x8xf32>
)mlir" + ElementwiseOver("%partial", "%copy", "%ws");
  std::string consumer = R"mlir(
    %ws = memref.alloc() {afir.buffer_desc = #gm9} : memref<2x2x2x4x8xf32>
)mlir" + MeanOver("%ws");
  auto module = ParseModule(R"mlir(
module {
  func.func private @stage0() attributes {
    afir.function_role = #afir.function_role<kernel_body>,
    afir.asc_graph = )mlir" +
                            kDomainGraph.str() + R"mlir(
  } {
)mlir" + producer +
                            R"mlir(
    return
  }
  func.func private @stage1() attributes {
    afir.function_role = #afir.function_role<kernel_body>,
    afir.asc_graph = )mlir" +
                            kDomainGraph.str() + R"mlir(
  } {
)mlir" + consumer +
                            R"mlir(
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  annotateReductionDomains(*module);
  Operation *mean = Subject(*module, "afir.mean");
  ASSERT_TRUE(mean);
  EXPECT_FALSE(mean->hasAttr(kReductionDomainAttrName));
}

// 同一个形状的对照：workspace 里装的是 **elementwise** 的结果而不是 partial。判据是「写它的
// 是什么」，不是「它在不在 GM」——否则合法的 `Mean(workspace tensor)` 会被误拒。
TEST_F(ReducePlanTest, ReductionDomainIsProvenAcrossAWorkspaceHoldingANonPartial) {
  std::string producer = R"mlir(
    %in = memref.alloc() {afir.buffer_desc = #ub0} : memref<2x2x2x4x8xf32>
    %ws = memref.alloc() {afir.buffer_desc = #gm9} : memref<2x2x2x4x8xf32>
)mlir" + ElementwiseOver("%in", "%copy", "%ws");
  std::string consumer = R"mlir(
    %ws = memref.alloc() {afir.buffer_desc = #gm9} : memref<2x2x2x4x8xf32>
)mlir" + MeanOver("%ws");
  auto module = ParseModule(R"mlir(
module {
  func.func private @stage0() attributes {
    afir.function_role = #afir.function_role<kernel_body>,
    afir.asc_graph = )mlir" +
                            kDomainGraph.str() + R"mlir(
  } {
)mlir" + producer +
                            R"mlir(
    return
  }
  func.func private @stage1() attributes {
    afir.function_role = #afir.function_role<kernel_body>,
    afir.asc_graph = )mlir" +
                            kDomainGraph.str() + R"mlir(
  } {
)mlir" + consumer +
                            R"mlir(
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  annotateReductionDomains(*module);
  Operation *mean = Subject(*module, "afir.mean");
  ASSERT_TRUE(mean);
  EXPECT_TRUE(mean->hasAttr(kReductionDomainAttrName));
}

// 幂等：写两遍得到同一个属性。bridge 侧今天只调一次，但这条判据让「再跑一遍」是安全的。
TEST_F(ReducePlanTest, AnnotatingTwiceIsIdempotent) {
  std::string body = R"mlir(
    %in = memref.alloc() {afir.buffer_desc = #ub0} : memref<2x2x2x4x8xf32>
)mlir" + MeanOver("%in");
  auto module = ParseModule(OneKernel(body));
  ASSERT_TRUE(module);
  annotateReductionDomains(*module);
  Operation *mean = Subject(*module, "afir.mean");
  ASSERT_TRUE(mean);
  Attribute first = mean->getAttr(kReductionDomainAttrName);
  annotateReductionDomains(*module);
  EXPECT_EQ(first, mean->getAttr(kReductionDomainAttrName));
}

// 读侧：属性**缺席即未证明**。不得退回 `analyzeLogicalReductionDomain`——那条路在跨阶段图上
// 给的正是错分母。
TEST_F(ReducePlanTest, ProvenDomainRejectsAMeanWithoutTheAttribute) {
  std::string body = R"mlir(
    %in = memref.alloc() {afir.buffer_desc = #ub0} : memref<2x2x2x4x8xf32>
)mlir" + MeanOver("%in");
  auto module = ParseModule(OneKernel(body));
  ASSERT_TRUE(module);
  // 故意不调 annotate。
  std::string reason;
  EXPECT_TRUE(failed(getProvenReductionDomain(Subject(*module, "afir.mean"), &reason)));
  EXPECT_NE(reason.find("afir.reduction_domain"), std::string::npos) << reason;
}

// 读侧：属性与轴图**都**答得出来而彼此不一致，说明中间有一方被改写过而另一方没跟上。
// 此时拒绝，不挑一个信。
TEST_F(ReducePlanTest, ProvenDomainRejectsADriftedAttribute) {
  std::string body = R"mlir(
    %in = memref.alloc() {afir.buffer_desc = #ub0} : memref<2x2x2x4x8xf32>
)mlir" + MeanOver("%in");
  auto module = ParseModule(OneKernel(body));
  ASSERT_TRUE(module);
  Operation *mean = Subject(*module, "afir.mean");
  ASSERT_TRUE(mean);
  // 轴图说 32，属性说 64。
  mean->setAttr(kReductionDomainAttrName,
                ReductionDomainAttr::get(&context, {1}, {ExprAttr::get(&context, StringAttr::get(&context, "64"))}));
  std::string reason;
  EXPECT_TRUE(failed(getProvenReductionDomain(mean, &reason)));
  EXPECT_NE(reason.find("no longer agrees"), std::string::npos) << reason;
}

// 读侧：轴图答不出来**不是**错误——属性正是为那种图存在的（跨阶段的合成图里，完整域的那几
// 根轴压根不在本函数的轴图里）。这一条钉的是「校验绑定」不等于「要求轴图也能算出来」。
TEST_F(ReducePlanTest, ProvenDomainAcceptsADomainTheAxisGraphCannotConfirm) {
  std::string body = R"mlir(
    %in = memref.alloc() {afir.buffer_desc = #ub0} : memref<2x2x2x4x8xf32>
)mlir" + MeanOver("%in");
  // 不给 asc_graph：轴图无从回答。
  auto module = ParseModule(R"mlir(
module {
  func.func private @kernel() attributes {
    afir.function_role = #afir.function_role<kernel_body>
  } {
)mlir" + body +
                            R"mlir(
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  Operation *mean = Subject(*module, "afir.mean");
  ASSERT_TRUE(mean);
  ASSERT_TRUE(failed(analyzeLogicalReductionDomain(mean)));
  mean->setAttr(kReductionDomainAttrName,
                ReductionDomainAttr::get(&context, {1}, {ExprAttr::get(&context, StringAttr::get(&context, "32"))}));
  std::string reason;
  FailureOr<LogicalReductionDomain> domain = getProvenReductionDomain(mean, &reason);
  ASSERT_TRUE(succeeded(domain)) << reason;
  EXPECT_EQ(domain->sizes.front().getValue().getValue(), "32");
}

}  // namespace

namespace {
scf::ForOp WrapReduceInLoop(Operation *op, int64_t axis) {
  OpBuilder builder(op);
  Value zero = builder.create<arith::ConstantIndexOp>(op->getLoc(), 0);
  Value two = builder.create<arith::ConstantIndexOp>(op->getLoc(), 2);
  Value one = builder.create<arith::ConstantIndexOp>(op->getLoc(), 1);
  auto loop = builder.create<scf::ForOp>(op->getLoc(), zero, two, one);
  loop->setAttr(kAxisBindingAttrName, AxisBindingAttr::get(op->getContext(), axis, AxisValueRole::Loop));
  auto reduce = cast<SumOp>(op);
  reduce.getResult().replaceAllUsesWith(reduce.getDestination());
  op->moveBefore(loop.getBody()->getTerminator());
  return loop;
}
}  // namespace

TEST_F(ReducePlanTest, ScratchUseScopeDoesNotChangePhysicalLifetimeOrDestination) {
  auto module = Parse(kS0Body);
  ASSERT_TRUE(module);
  auto reduce = cast<SumOp>(Subject(*module, "afir.sum"));
  auto func = reduce->getParentOfType<func::FuncOp>();
  auto destinationOwner = reduce.getDestination().getDefiningOp();
  auto loop = WrapReduceInLoop(reduce, 4);
  ASSERT_TRUE(succeeded(scopeReduceScratchAllocations(func)));
  for (Value scratch : reduce.getScratch()) {
    EXPECT_EQ(scratch.getDefiningOp()->getBlock(), loop.getBody());
    EXPECT_EQ(scratch.getDefiningOp()->getAttrOfType<IntegerAttr>(kReduceScratchScopeAttr).getInt(), 4);
    EXPECT_EQ(getBufferDescAttr(scratch).getLifeTimeAxisId(), -1);
  }
  EXPECT_EQ(destinationOwner->getBlock(), &func.front());
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(ReducePlanTest, SharedScratchAcrossSiblingLoopsKeepsCommonScope) {
  auto module = Parse(kS0Body);
  ASSERT_TRUE(module);
  auto reduce = cast<SumOp>(Subject(*module, "afir.sum"));
  auto func = reduce->getParentOfType<func::FuncOp>();
  OpBuilder builder(reduce);
  auto clone = cast<SumOp>(builder.clone(*reduce));
  WrapReduceInLoop(reduce, 4);
  WrapReduceInLoop(clone, 4);
  ASSERT_TRUE(succeeded(scopeReduceScratchAllocations(func)));
  for (Value scratch : reduce.getScratch()) {
    EXPECT_EQ(scratch.getDefiningOp()->getBlock(), &func.front());
    EXPECT_EQ(scratch.getDefiningOp()->getAttrOfType<IntegerAttr>(kReduceScratchScopeAttr).getInt(), -1);
  }
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(ReducePlanTest, RejectsScratchScopeEscape) {
  auto module = Parse(kS0Body);
  ASSERT_TRUE(module);
  auto reduce = cast<SumOp>(Subject(*module, "afir.sum"));
  auto owner = reduce.getScratch()[0].getDefiningOp();
  owner->setAttr(kReduceScratchScopeAttr, IntegerAttr::get(IntegerType::get(&context, 64), 4));
  ScopedDiagnosticHandler handler(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(verifyReduceScratchScope(reduce.getScratch()[0], reduce)));
}

TEST_F(ReducePlanTest, RejectsOverlappingApiAndRoundStorage) {
  auto module = Parse(kS0Body);
  ASSERT_TRUE(module);
  auto reduce = cast<SumOp>(Subject(*module, "afir.sum"));
  WrapReduceInLoop(reduce, 4);
  setBufferDescAttr(reduce.getScratch()[1].getDefiningOp(), getBufferDescAttr(reduce.getScratch()[0]));
  ScopedDiagnosticHandler handler(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(scopeReduceScratchAllocations(reduce->getParentOfType<func::FuncOp>())));
}

TEST_F(ReducePlanTest, UnconsumedRoundResultKeepsPhysicalAllocation) {
  std::string body = kS0Body.str();
  const std::string oldSchedule = "loop_axis = 4";
  body.replace(body.find(oldSchedule), oldSchedule.size(), "loop_axis = 7");
  auto module = Parse(body);
  ASSERT_TRUE(module);
  auto reduce = cast<SumOp>(Subject(*module, "afir.sum"));
  auto func = reduce->getParentOfType<func::FuncOp>();
  auto loop = WrapReduceInLoop(reduce, 7);
  ASSERT_TRUE(succeeded(scopeReduceScratchAllocations(func)));
  EXPECT_EQ(reduce.getScratch()[0].getDefiningOp()->getBlock(), loop.getBody());
  EXPECT_EQ(reduce.getScratch()[1].getDefiningOp()->getBlock(), &func.front());
  EXPECT_FALSE(reduce.getScratch()[1].getDefiningOp()->hasAttr(kReduceScratchScopeAttr));
}

TEST_F(ReducePlanTest, MissingLoopBindingCannotBecomeScratchScope) {
  auto module = Parse(kS0Body);
  ASSERT_TRUE(module);
  auto reduce = cast<SumOp>(Subject(*module, "afir.sum"));
  auto func = reduce->getParentOfType<func::FuncOp>();
  auto loop = WrapReduceInLoop(reduce, 4);
  loop->removeAttr(kAxisBindingAttrName);
  ScopedDiagnosticHandler handler(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(scopeReduceScratchAllocations(func)));
}

TEST_F(ReducePlanTest, AccumulationRequiresItsMaterializedLoop) {
  auto module = Parse(kS0Body);
  ASSERT_TRUE(module);
  auto reduce = cast<SumOp>(Subject(*module, "afir.sum"));
  WrapReduceInLoop(reduce, 7);
  std::string message;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
    llvm::raw_string_ostream os(message);
    diagnostic.print(os);
    return success();
  });
  EXPECT_TRUE(failed(scopeReduceScratchAllocations(reduce->getParentOfType<func::FuncOp>())));
  EXPECT_NE(message.find("accumulation axis 4 has no enclosing materialized loop"), std::string::npos) << message;
}
