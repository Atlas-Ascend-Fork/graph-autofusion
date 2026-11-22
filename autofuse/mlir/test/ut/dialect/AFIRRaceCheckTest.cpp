/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 竞争检查的覆盖（设计文档 5.1 步 8b、5.3 测试策略表的「数据竞争」一行）。
//
// 为什么不是 LIT：竞争的失败形态是「IR 完全合法，但两次访问之间少一条
// happens-before」。LIT 断言的是输出文本——一条缺失的顺序边在文本里没有任何形状，
// 所以 LIT 只能验证「检查器报了什么」，验证不了「检查器该报什么」。这里逐个构造
// 已知有竞争 / 已知无竞争的形态，正反两侧都钉住；没有反例侧，检查器退化成「全报」
// 也能通过。

#include "dialect/AFIR/Analysis/RaceCheck.h"

#include <string>
#include <utility>

#include "llvm/ADT/STLExtras.h"

#include "dialect/AFIR/Analysis/BufferAlias.h"
#include "dialect/AFIR/IR/AFIR.h"
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
using namespace mlir::afir;

namespace {

constexpr llvm::StringLiteral kPreamble = R"mlir(
#in = #afir.tensor_desc<axis = [0], role = <input>>
#cp = #afir.tensor_desc<axis = [0], role = <compute>>
#out = #afir.tensor_desc<axis = [0], role = <output>>
#gm = #afir.buffer_desc<space = <gm>, life_time_axis_id = -1>
#ub = #afir.buffer_desc<space = <ub>, buffer_id = 0, life_time_axis_id = -1>
#id = affine_map<(d0) -> (d0)>
)mlir";

class AFIRRaceCheckTest : public testing::Test {
 protected:
  void SetUp() override {
    context.getOrLoadDialect<AFIRDialect>();
    context.getOrLoadDialect<arith::ArithDialect>();
    context.getOrLoadDialect<func::FuncDialect>();
    context.getOrLoadDialect<memref::MemRefDialect>();
    context.getOrLoadDialect<scf::SCFDialect>();
  }

  // `body` 是 kernel 函数体；两个 GM 参数固定为一进一出。
  OwningOpRef<ModuleOp> Parse(llvm::StringRef body) {
    std::string source = kPreamble.str() + R"mlir(
module {
  func.func private @kernel(%gm0: memref<16xf16> {afir.buffer_desc = #gm},
                            %gm1: memref<16xf16> {afir.buffer_desc = #gm}) attributes {
    afir.arg_descs = [#in, #out],
    afir.function_role = #afir.function_role<kernel_body>
  } {
)mlir" + body.str() +
                         R"mlir(
    return
  }
}
)mlir";
    return parseSourceString<ModuleOp>(source, &context);
  }

  llvm::SmallVector<RaceFinding> Check(ModuleOp module) {
    auto func = module.lookupSymbol<func::FuncOp>("kernel");
    EXPECT_TRUE(func);
    RaceChecker checker;
    return checker.check(func);
  }

  MLIRContext context;
};

// 未覆盖的 producer：`afir.load` 写 `%tile` 并把「写过了」表达成结果 `%loaded`，
// 消费者却绕回去读 `%tile`。两条流之间因此没有任何顺序边——lowering 之后就是搬入
// 还没完成、搬出已经开始读。
TEST_F(AFIRRaceCheckTest, ReportsProducerThatConsumerBypasses) {
  auto module = Parse(R"mlir(
    %tile = memref.alloc() {afir.buffer_desc = #ub} : memref<16xf16>
    %loaded = afir.load %gm0 outs(%tile) {
      afir.operand_descs = [#in], afir.result_descs = [#cp], afir.destination_descs = [#cp],
      indexing_maps = [#id, #id]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    afir.store %tile outs(%gm1) {
      afir.operand_descs = [#cp], afir.destination_descs = [#out],
      indexing_maps = [#id, #id]
    } : memref<16xf16>, memref<16xf16>
  )mlir");
  ASSERT_TRUE(module);
  auto findings = Check(*module);

  ASSERT_EQ(findings.size(), 1U);
  EXPECT_EQ(findings.front().kind, HazardKind::ReadAfterWrite);
  EXPECT_EQ(getPipe(findings.front().first), Pipe::TransferIn);
  EXPECT_EQ(getPipe(findings.front().second), Pipe::TransferOut);
  EXPECT_FALSE(findings.front().iterationDistance.has_value());
}

// 反例侧：同一段 IR，消费者改用结果值。值依赖就是 AFIR 层的 happens-before，检查器
// 必须一条都不报——否则它对「写对了的 IR」没有任何判别力。
TEST_F(AFIRRaceCheckTest, AcceptsProducerThreadedThroughItsResult) {
  auto module = Parse(R"mlir(
    %tile = memref.alloc() {afir.buffer_desc = #ub} : memref<16xf16>
    %loaded = afir.load %gm0 outs(%tile) {
      afir.operand_descs = [#in], afir.result_descs = [#cp], afir.destination_descs = [#cp],
      indexing_maps = [#id, #id]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    afir.store %loaded outs(%gm1) {
      afir.operand_descs = [#cp], afir.destination_descs = [#out],
      indexing_maps = [#id, #id]
    } : memref<16xf16>, memref<16xf16>
  )mlir");
  ASSERT_TRUE(module);

  EXPECT_TRUE(Check(*module).empty());
}

// 显式同步边：同样是「消费者绕过 producer 的结果」的写法，但中间放了一对
// `afir.event_record` / `afir.event_wait`。这条边就是放弃 `TQue` 自动同步之后
// AFIR 自己表达顺序的手段（设计文档 3.2 / ADR-12），检查器必须认。
// tmp buffer 操作数排在 destination 之后。按「destination 是最后几个操作数」去切段的
// 写法会把 destination 数成输入（读）、把 tmp buffer 数成 destination（写），于是真正
// 的写被记成了读——这个方向是漏报，写对写错的 IR 都一条不报。
TEST_F(AFIRRaceCheckTest, ReportsBypassedProducerThatCarriesATmpBufferOperand) {
  auto module = Parse(R"mlir(
    %tile = memref.alloc() {afir.buffer_desc = #ub} : memref<16xf16>
    %dst = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1,>} : memref<16xf16>
    %tmp_buffer = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 7, size_from_resource = true,>} : memref<1024xui8>
    %loaded = afir.load %gm0 outs(%tile) {
      afir.operand_descs = [#in], afir.result_descs = [#cp], afir.destination_descs = [#cp],
      indexing_maps = [#id, #id]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    %abs = afir.abs %loaded outs(%dst) tmp(%tmp_buffer) {
      afir.operand_descs = [#cp], afir.result_descs = [#cp], afir.destination_descs = [#cp]
    } : memref<16xf16>, memref<16xf16>, memref<1024xui8> -> memref<16xf16>
    afir.store %dst outs(%gm1) {
      afir.operand_descs = [#cp], afir.destination_descs = [#out],
      indexing_maps = [#id, #id]
    } : memref<16xf16>, memref<16xf16>
  )mlir");
  ASSERT_TRUE(module);
  auto findings = Check(*module);

  ASSERT_EQ(findings.size(), 1U);
  EXPECT_EQ(findings.front().kind, HazardKind::ReadAfterWrite);
  EXPECT_EQ(getPipe(findings.front().second), Pipe::TransferOut);
}

// 反例侧：同一段 IR，store 改用 abs 的结果值。tmp buffer 在不在不影响这条判定。
TEST_F(AFIRRaceCheckTest, AcceptsTmpBufferCarryingProducerThreadedThroughItsResult) {
  auto module = Parse(R"mlir(
    %tile = memref.alloc() {afir.buffer_desc = #ub} : memref<16xf16>
    %dst = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1,>} : memref<16xf16>
    %tmp_buffer = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 7, size_from_resource = true,>} : memref<1024xui8>
    %loaded = afir.load %gm0 outs(%tile) {
      afir.operand_descs = [#in], afir.result_descs = [#cp], afir.destination_descs = [#cp],
      indexing_maps = [#id, #id]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    %abs = afir.abs %loaded outs(%dst) tmp(%tmp_buffer) {
      afir.operand_descs = [#cp], afir.result_descs = [#cp], afir.destination_descs = [#cp]
    } : memref<16xf16>, memref<16xf16>, memref<1024xui8> -> memref<16xf16>
    afir.store %abs outs(%gm1) {
      afir.operand_descs = [#cp], afir.destination_descs = [#out],
      indexing_maps = [#id, #id]
    } : memref<16xf16>, memref<16xf16>
  )mlir");
  ASSERT_TRUE(module);

  EXPECT_TRUE(Check(*module).empty());
}

TEST_F(AFIRRaceCheckTest, AcceptsProducerOrderedByAnExplicitEvent) {
  auto module = Parse(R"mlir(
    %tile = memref.alloc() {afir.buffer_desc = #ub} : memref<16xf16>
    %loaded = afir.load %gm0 outs(%tile) {
      afir.operand_descs = [#in], afir.result_descs = [#cp], afir.destination_descs = [#cp],
      indexing_maps = [#id, #id]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    %t = afir.event_record : !afir.token<from = transfer_in, to = transfer_out>
    afir.event_wait %t : !afir.token<from = transfer_in, to = transfer_out>
    afir.store %tile outs(%gm1) {
      afir.operand_descs = [#cp], afir.destination_descs = [#out],
      indexing_maps = [#id, #id]
    } : memref<16xf16>, memref<16xf16>
  )mlir");
  ASSERT_TRUE(module);

  EXPECT_TRUE(Check(*module).empty());
}

// 反例侧：边的方向反了。`from = transfer_out, to = transfer_in` 说的是「搬出先于
// 搬入」，而这里要挡的依赖是搬入先于搬出。方向写反在无类型 token 下是合法 IR，失败
// 方式是静默的数据竞争——这正是 ADR-19 让 token 带 pipe 对的理由，检查器必须仍然报。
TEST_F(AFIRRaceCheckTest, ReportsProducerWhoseEventEdgePointsTheWrongWay) {
  auto module = Parse(R"mlir(
    %tile = memref.alloc() {afir.buffer_desc = #ub} : memref<16xf16>
    %loaded = afir.load %gm0 outs(%tile) {
      afir.operand_descs = [#in], afir.result_descs = [#cp], afir.destination_descs = [#cp],
      indexing_maps = [#id, #id]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    %t = afir.event_record : !afir.token<from = transfer_out, to = transfer_in>
    afir.event_wait %t : !afir.token<from = transfer_out, to = transfer_in>
    afir.store %tile outs(%gm1) {
      afir.operand_descs = [#cp], afir.destination_descs = [#out],
      indexing_maps = [#id, #id]
    } : memref<16xf16>, memref<16xf16>
  )mlir");
  ASSERT_TRUE(module);
  auto findings = Check(*module);

  ASSERT_EQ(findings.size(), 1U);
  EXPECT_EQ(findings.front().kind, HazardKind::ReadAfterWrite);
}

// 同一条流内程序序即执行序，所以两次搬入写同一块存储不是竞争。
TEST_F(AFIRRaceCheckTest, AcceptsTwoWritesOnTheSameStream) {
  auto module = Parse(R"mlir(
    %tile = memref.alloc() {afir.buffer_desc = #ub} : memref<16xf16>
    %first = afir.load %gm0 outs(%tile) {
      afir.operand_descs = [#in], afir.result_descs = [#cp], afir.destination_descs = [#cp],
      indexing_maps = [#id, #id]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    %second = afir.load %gm0 outs(%tile) {
      afir.operand_descs = [#in], afir.result_descs = [#cp], afir.destination_descs = [#cp],
      indexing_maps = [#id, #id]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
  )mlir");
  ASSERT_TRUE(module);

  EXPECT_TRUE(Check(*module).empty());
}

// 跨 stage 的读写重叠：两个 stage 的切片在下标空间里真的不相交时不得误报。
TEST_F(AFIRRaceCheckTest, AcceptsDisjointStageSlices) {
  auto module = Parse(R"mlir(
    %pool = memref.alloc() {afir.buffer_desc = #ub} : memref<2x16xf16>
    %a = memref.subview %pool[0, 0] [1, 16] [1, 1] {afir.buffer_desc = #ub}
        : memref<2x16xf16> to memref<16xf16, strided<[1]>>
    %b = memref.subview %pool[1, 0] [1, 16] [1, 1] {afir.buffer_desc = #ub}
        : memref<2x16xf16> to memref<16xf16, strided<[1], offset: 16>>
    %loaded = afir.load %gm0 outs(%a) {
      afir.operand_descs = [#in], afir.result_descs = [#cp], afir.destination_descs = [#cp],
      indexing_maps = [#id, #id]
    } : memref<16xf16>, memref<16xf16, strided<[1]>> -> memref<16xf16, strided<[1]>>
    afir.store %b outs(%gm1) {
      afir.operand_descs = [#cp], afir.destination_descs = [#out],
      indexing_maps = [#id, #id]
    } : memref<16xf16, strided<[1], offset: 16>>, memref<16xf16>
  )mlir");
  ASSERT_TRUE(module);

  EXPECT_TRUE(Check(*module).empty());
}

// 同一段 IR，两个切片落回同一个 stage：别名回来了，竞争也回来了。
TEST_F(AFIRRaceCheckTest, ReportsOverlappingStageSlices) {
  auto module = Parse(R"mlir(
    %pool = memref.alloc() {afir.buffer_desc = #ub} : memref<2x16xf16>
    %a = memref.subview %pool[0, 0] [1, 16] [1, 1] {afir.buffer_desc = #ub}
        : memref<2x16xf16> to memref<16xf16, strided<[1]>>
    %b = memref.subview %pool[0, 0] [1, 16] [1, 1] {afir.buffer_desc = #ub}
        : memref<2x16xf16> to memref<16xf16, strided<[1]>>
    %loaded = afir.load %gm0 outs(%a) {
      afir.operand_descs = [#in], afir.result_descs = [#cp], afir.destination_descs = [#cp],
      indexing_maps = [#id, #id]
    } : memref<16xf16>, memref<16xf16, strided<[1]>> -> memref<16xf16, strided<[1]>>
    afir.store %b outs(%gm1) {
      afir.operand_descs = [#cp], afir.destination_descs = [#out],
      indexing_maps = [#id, #id]
    } : memref<16xf16, strided<[1]>>, memref<16xf16>
  )mlir");
  ASSERT_TRUE(module);
  auto findings = Check(*module);

  ASSERT_EQ(findings.size(), 1U);
  EXPECT_EQ(findings.front().kind, HazardKind::ReadAfterWrite);
  EXPECT_FALSE(findings.front().iterationDistance.has_value());
}

// 双缓冲 WAR 冒险。循环体内的值依赖只在同一次迭代里成立；消费者读第 i 次的数据时，
// 生产者已经可以开始写第 i + d 次的同一块存储。d 由轮转级数给出——这正是多缓冲做的
// 事：把冒险推远，推不掉。检查器必须报出这条边，并给出正确的 d。
TEST_F(AFIRRaceCheckTest, ReportsLoopCarriedWriteAfterReadAtTheRotationDistance) {
  constexpr llvm::StringLiteral kLoopBody = R"mlir(
    %pool = memref.alloc() {afir.buffer_desc = #ub} : memref<STAGESx16xf16>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %stages = arith.constant STAGES : index
    %n = arith.constant 8 : index
    scf.for %i = %c0 to %n step %c1 {
      %slot = arith.remsi %i, %stages : index
      %tile = memref.subview %pool[%slot, 0] [1, 16] [1, 1] {afir.buffer_desc = #ub}
          : memref<STAGESx16xf16> to memref<16xf16, strided<[1], offset: ?>>
      %loaded = afir.load %gm0 outs(%tile) {
        afir.operand_descs = [#in], afir.result_descs = [#cp], afir.destination_descs = [#cp],
        indexing_maps = [#id, #id]
      } : memref<16xf16>, memref<16xf16, strided<[1], offset: ?>>
          -> memref<16xf16, strided<[1], offset: ?>>
      afir.store %loaded outs(%gm1) {
        afir.operand_descs = [#cp], afir.destination_descs = [#out],
        indexing_maps = [#id, #id]
      } : memref<16xf16, strided<[1], offset: ?>>, memref<16xf16>
    }
  )mlir";

  for (int64_t stages : {1, 2, 3}) {
    SCOPED_TRACE("stages=" + std::to_string(stages));
    std::string body = kLoopBody.str();
    for (size_t pos = 0; (pos = body.find("STAGES", pos)) != std::string::npos;) {
      body.replace(pos, 6, std::to_string(stages));
    }
    auto module = Parse(body);
    ASSERT_TRUE(module);
    auto findings = Check(*module);

    ASSERT_EQ(findings.size(), 1U);
    EXPECT_EQ(findings.front().kind, HazardKind::WriteAfterRead);
    EXPECT_EQ(getPipe(findings.front().first), Pipe::TransferOut);
    EXPECT_EQ(getPipe(findings.front().second), Pipe::TransferIn);
    ASSERT_TRUE(findings.front().iterationDistance.has_value());
    EXPECT_EQ(*findings.front().iterationDistance, stages);
  }
}

// 双缓冲 WAR 的防护形态：token 由 `scf.for` 的 iter_arg 携带，于是第 i+1 次迭代的
// wait 等的是第 i 次迭代的 record——边跨的正好是一次迭代，正是上一条用例报出的那条
// 循环携带 WAR 所缺的东西。
//
// 这条用例同时钉住 SSA 形态本身：wait 消费的是块参数而不是 record 的结果，两者之间
// 没有同迭代的 use-def 边。写成同迭代的一对（record 在前、wait 在后），挡住的是同一次
// 迭代的依赖，跨迭代的那条依然是竞争——下一条用例。
TEST_F(AFIRRaceCheckTest, AcceptsLoopCarriedWriteAfterReadGuardedByACarriedEvent) {
  auto module = Parse(R"mlir(
    %pool = memref.alloc() {afir.buffer_desc = #ub} : memref<2x16xf16>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %stages = arith.constant 2 : index
    %n = arith.constant 8 : index
    %seed = afir.event_record : !afir.token<from = transfer_out, to = transfer_in>
    %last = scf.for %i = %c0 to %n step %c1 iter_args(%w = %seed)
        -> (!afir.token<from = transfer_out, to = transfer_in>) {
      %slot = arith.remsi %i, %stages : index
      %tile = memref.subview %pool[%slot, 0] [1, 16] [1, 1] {afir.buffer_desc = #ub}
          : memref<2x16xf16> to memref<16xf16, strided<[1], offset: ?>>
      afir.event_wait %w : !afir.token<from = transfer_out, to = transfer_in>
      %loaded = afir.load %gm0 outs(%tile) {
        afir.operand_descs = [#in], afir.result_descs = [#cp], afir.destination_descs = [#cp],
        indexing_maps = [#id, #id]
      } : memref<16xf16>, memref<16xf16, strided<[1], offset: ?>>
          -> memref<16xf16, strided<[1], offset: ?>>
      afir.store %loaded outs(%gm1) {
        afir.operand_descs = [#cp], afir.destination_descs = [#out],
        indexing_maps = [#id, #id]
      } : memref<16xf16, strided<[1], offset: ?>>, memref<16xf16>
      %next = afir.event_record : !afir.token<from = transfer_out, to = transfer_in>
      scf.yield %next : !afir.token<from = transfer_out, to = transfer_in>
    }
  )mlir");
  ASSERT_TRUE(module);

  EXPECT_TRUE(Check(*module).empty());
}

// 反例侧：同一对 record/wait，但在同一次迭代内配对（record 在前、wait 在后，token
// 不经过 iter_arg）。它挡住的是同迭代的 RAW，跨迭代的 WAR 一点没挡——第 i+1 次迭代的
// 搬入仍然可能追上第 i 次迭代还没读完的那个槽。放过它就是把「同步写了但写错位置」
// 当成安全。
TEST_F(AFIRRaceCheckTest, ReportsLoopCarriedWriteAfterReadWhenTheEventPairsWithinOneIteration) {
  auto module = Parse(R"mlir(
    %pool = memref.alloc() {afir.buffer_desc = #ub} : memref<2x16xf16>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %stages = arith.constant 2 : index
    %n = arith.constant 8 : index
    scf.for %i = %c0 to %n step %c1 {
      %slot = arith.remsi %i, %stages : index
      %tile = memref.subview %pool[%slot, 0] [1, 16] [1, 1] {afir.buffer_desc = #ub}
          : memref<2x16xf16> to memref<16xf16, strided<[1], offset: ?>>
      %loaded = afir.load %gm0 outs(%tile) {
        afir.operand_descs = [#in], afir.result_descs = [#cp], afir.destination_descs = [#cp],
        indexing_maps = [#id, #id]
      } : memref<16xf16>, memref<16xf16, strided<[1], offset: ?>>
          -> memref<16xf16, strided<[1], offset: ?>>
      %t = afir.event_record : !afir.token<from = transfer_in, to = transfer_out>
      afir.event_wait %t : !afir.token<from = transfer_in, to = transfer_out>
      afir.store %tile outs(%gm1) {
        afir.operand_descs = [#cp], afir.destination_descs = [#out],
        indexing_maps = [#id, #id]
      } : memref<16xf16, strided<[1], offset: ?>>, memref<16xf16>
    }
  )mlir");
  ASSERT_TRUE(module);
  auto findings = Check(*module);

  ASSERT_EQ(findings.size(), 1U);
  EXPECT_EQ(findings.front().kind, HazardKind::WriteAfterRead);
  ASSERT_TRUE(findings.front().iterationDistance.has_value());
  EXPECT_EQ(*findings.front().iterationDistance, 2);
}

// 兄弟循环各自轮转：两个下标看起来都是「mod 2」，但 i 不是同一个 i，两个循环各自
// 遍历了全部槽位。按相位差判「永不撞同一个槽」是漏报——别名分析必须先确认两侧由同一
// 个归纳变量驱动。
TEST_F(AFIRRaceCheckTest, ReportsRaceBetweenSiblingLoopsThatRotateIndependently) {
  auto module = Parse(R"mlir(
    %pool = memref.alloc() {afir.buffer_desc = #ub} : memref<2x16xf16>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %n = arith.constant 8 : index
    scf.for %i = %c0 to %n step %c1 {
      %slot = arith.remsi %i, %c2 : index
      %tile = memref.subview %pool[%slot, 0] [1, 16] [1, 1] {afir.buffer_desc = #ub}
          : memref<2x16xf16> to memref<16xf16, strided<[1], offset: ?>>
      %loaded = afir.load %gm0 outs(%tile) {
        afir.operand_descs = [#in], afir.result_descs = [#cp], afir.destination_descs = [#cp],
        indexing_maps = [#id, #id]
      } : memref<16xf16>, memref<16xf16, strided<[1], offset: ?>>
          -> memref<16xf16, strided<[1], offset: ?>>
    }
    scf.for %j = %c0 to %n step %c1 {
      %shifted = arith.addi %j, %c1 : index
      %slot = arith.remsi %shifted, %c2 : index
      %tile = memref.subview %pool[%slot, 0] [1, 16] [1, 1] {afir.buffer_desc = #ub}
          : memref<2x16xf16> to memref<16xf16, strided<[1], offset: ?>>
      afir.store %tile outs(%gm1) {
        afir.operand_descs = [#cp], afir.destination_descs = [#out],
        indexing_maps = [#id, #id]
      } : memref<16xf16, strided<[1], offset: ?>>, memref<16xf16>
    }
  )mlir");
  ASSERT_TRUE(module);
  auto findings = Check(*module);

  ASSERT_EQ(findings.size(), 1U);
  EXPECT_EQ(findings.front().kind, HazardKind::ReadAfterWrite);
  // 不在同一个循环里，没有「跨迭代距离」可言。
  EXPECT_FALSE(findings.front().iterationDistance.has_value());
}

// 两维各自轮转：同时撞上要等到两个同余式都满足，第一次在 d = 5（反方向 d = 7）。
// 搜索上界取最大级数（4）会在这之前收工，把真竞争判成「永不重叠」。
TEST_F(AFIRRaceCheckTest, ReportsLoopCarriedRaceThatOnlyAppearsAtTheCommonMultiple) {
  auto module = Parse(R"mlir(
    %pool = memref.alloc() {afir.buffer_desc = #ub} : memref<3x4x16xf16>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %n = arith.constant 32 : index
    scf.for %i = %c0 to %n step %c1 {
      %outer = arith.remsi %i, %c3 : index
      %inner = arith.remsi %i, %c4 : index
      %written = memref.subview %pool[%outer, %inner, 0] [1, 1, 16] [1, 1, 1] {afir.buffer_desc = #ub}
          : memref<3x4x16xf16> to memref<16xf16, strided<[1], offset: ?>>
      %shiftedOuter = arith.addi %i, %c1 : index
      %shiftedInner = arith.addi %i, %c3 : index
      %readOuter = arith.remsi %shiftedOuter, %c3 : index
      %readInner = arith.remsi %shiftedInner, %c4 : index
      %read = memref.subview %pool[%readOuter, %readInner, 0] [1, 1, 16] [1, 1, 1] {afir.buffer_desc = #ub}
          : memref<3x4x16xf16> to memref<16xf16, strided<[1], offset: ?>>
      %loaded = afir.load %gm0 outs(%written) {
        afir.operand_descs = [#in], afir.result_descs = [#cp], afir.destination_descs = [#cp],
        indexing_maps = [#id, #id]
      } : memref<16xf16>, memref<16xf16, strided<[1], offset: ?>>
          -> memref<16xf16, strided<[1], offset: ?>>
      afir.store %read outs(%gm1) {
        afir.operand_descs = [#cp], afir.destination_descs = [#out],
        indexing_maps = [#id, #id]
      } : memref<16xf16, strided<[1], offset: ?>>, memref<16xf16>
    }
  )mlir");
  ASSERT_TRUE(module);
  auto findings = Check(*module);

  ASSERT_EQ(findings.size(), 2U);
  llvm::SmallVector<int64_t> distances;
  for (const RaceFinding &finding : findings) {
    ASSERT_TRUE(finding.iterationDistance.has_value());
    distances.push_back(*finding.iterationDistance);
  }
  llvm::sort(distances);
  EXPECT_EQ(distances[0], 5);
  EXPECT_EQ(distances[1], 7);
  // 同一次迭代里两个切片落在不同槽，不该报。
  EXPECT_TRUE(llvm::all_of(findings, [](const RaceFinding &finding) { return finding.iterationDistance.has_value(); }));
}

// 步长不是 1 时槽位每次迭代前进 step，`d ≡ Δ (mod S)` 的算术对不上。与其算错距离，
// 不如不认这个轮转——退回保守侧。本例里 `%i` 取 0,2,4,…，`%i mod 2` 恒为 0，实际就是
// 单缓冲，保守答案同时也是正确答案。
TEST_F(AFIRRaceCheckTest, TreatsNonUnitLoopStepConservatively) {
  auto module = Parse(R"mlir(
    %pool = memref.alloc() {afir.buffer_desc = #ub} : memref<2x16xf16>
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %n = arith.constant 8 : index
    scf.for %i = %c0 to %n step %c2 {
      %slot = arith.remsi %i, %c2 : index
      %tile = memref.subview %pool[%slot, 0] [1, 16] [1, 1] {afir.buffer_desc = #ub}
          : memref<2x16xf16> to memref<16xf16, strided<[1], offset: ?>>
      %loaded = afir.load %gm0 outs(%tile) {
        afir.operand_descs = [#in], afir.result_descs = [#cp], afir.destination_descs = [#cp],
        indexing_maps = [#id, #id]
      } : memref<16xf16>, memref<16xf16, strided<[1], offset: ?>>
          -> memref<16xf16, strided<[1], offset: ?>>
      afir.store %loaded outs(%gm1) {
        afir.operand_descs = [#cp], afir.destination_descs = [#out],
        indexing_maps = [#id, #id]
      } : memref<16xf16, strided<[1], offset: ?>>, memref<16xf16>
    }
  )mlir");
  ASSERT_TRUE(module);
  auto findings = Check(*module);

  ASSERT_EQ(findings.size(), 1U);
  EXPECT_EQ(findings.front().kind, HazardKind::WriteAfterRead);
  ASSERT_TRUE(findings.front().iterationDistance.has_value());
  EXPECT_EQ(*findings.front().iterationDistance, 1);
}

// 别名分析单独的正反例。`resolveBufferSlice` 追根时要穿过 destination-style 的 tie
// ——结果与它绑定的 destination 是同一块存储（设计文档 5.1 步 8）；漏了这一条，
// 「消费者用结果值」的形态会被当成另一块存储，竞争全线漏报。
TEST_F(AFIRRaceCheckTest, ResolvesSliceThroughDestinationTieAndSubView) {
  auto module = Parse(R"mlir(
    %pool = memref.alloc() {afir.buffer_desc = #ub} : memref<2x16xf16>
    %a = memref.subview %pool[0, 0] [1, 16] [1, 1] {afir.buffer_desc = #ub}
        : memref<2x16xf16> to memref<16xf16, strided<[1]>>
    %b = memref.subview %pool[1, 0] [1, 16] [1, 1] {afir.buffer_desc = #ub}
        : memref<2x16xf16> to memref<16xf16, strided<[1], offset: 16>>
    %loaded = afir.load %gm0 outs(%a) {
      afir.operand_descs = [#in], afir.result_descs = [#cp], afir.destination_descs = [#cp],
      indexing_maps = [#id, #id]
    } : memref<16xf16>, memref<16xf16, strided<[1]>> -> memref<16xf16, strided<[1]>>
    afir.store %loaded outs(%gm1) {
      afir.operand_descs = [#cp], afir.destination_descs = [#out],
      indexing_maps = [#id, #id]
    } : memref<16xf16, strided<[1]>>, memref<16xf16>
  )mlir");
  ASSERT_TRUE(module);
  auto func = module->lookupSymbol<func::FuncOp>("kernel");
  ASSERT_TRUE(func);

  Value pool, sliceA, sliceB, loaded;
  func.walk([&](Operation *op) {
    if (isa<memref::AllocOp>(op)) {
      pool = op->getResult(0);
    } else if (isa<memref::SubViewOp>(op)) {
      (sliceA ? sliceB : sliceA) = op->getResult(0);
    } else if (isa<LoadOp>(op)) {
      loaded = op->getResult(0);
    }
  });
  ASSERT_TRUE(pool && sliceA && sliceB && loaded);

  BufferSlice resolvedA = resolveBufferSlice(sliceA);
  BufferSlice resolvedB = resolveBufferSlice(sliceB);
  BufferSlice resolvedResult = resolveBufferSlice(loaded);
  BufferSlice resolvedPool = resolveBufferSlice(pool);

  EXPECT_EQ(resolvedA.base, pool);
  EXPECT_EQ(resolvedB.base, pool);
  // 结果值顺着 tie 追到 destination，再追到同一个 pool，且带着 destination 的段信息。
  EXPECT_EQ(resolvedResult.base, pool);
  EXPECT_EQ(aliasSlices(resolvedResult, resolvedA), AliasResult::MayAlias);
  EXPECT_EQ(aliasSlices(resolvedA, resolvedB), AliasResult::NoAlias);
  // 整块与它的任意切片一定重叠。
  EXPECT_TRUE(resolvedPool.wholeBase);
  EXPECT_EQ(aliasSlices(resolvedPool, resolvedA), AliasResult::MayAlias);
  // 不同的根不可能重叠：AFIR 里没有跨定义点构造别名的手段。
  BufferSlice argument = resolveBufferSlice(func.getArgument(0));
  EXPECT_EQ(aliasSlices(argument, resolvedPool), AliasResult::NoAlias);
}

// 轮转下标的周期直接决定跨迭代第一次撞上的距离，这是双缓冲分析的算术核心，单独钉住。
TEST(AFIRBufferAliasTest, RotatingOffsetsFirstCollideAtTheirCommonPeriod) {
  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  OpBuilder builder(&context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module->getBody());
  // 别名判定只看 Value 的身份：一个常量当存储的占位，另一个当归纳变量的占位。
  Value shared = builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), 0).getResult();
  Value inductionVariable = builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), 1).getResult();
  Value otherInductionVariable = builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), 2).getResult();

  auto rotating = [&](Value iv, llvm::ArrayRef<std::pair<int64_t, int64_t>> dimensions) {
    BufferSlice slice;
    slice.base = shared;
    for (auto [modulus, phase] : dimensions) {
      slice.offsets.push_back(DimensionOffset::getRotating(iv, modulus, phase));
      slice.sizes.push_back(1);
    }
    return slice;
  };

  BufferSlice singleA = rotating(inductionVariable, {{1, 0}});
  BufferSlice singleB = rotating(inductionVariable, {{1, 0}});
  EXPECT_EQ(firstAliasingIterationDistance(singleA, singleB), std::optional<int64_t>(1));

  BufferSlice doubleA = rotating(inductionVariable, {{2, 0}});
  BufferSlice doubleB = rotating(inductionVariable, {{2, 0}});
  EXPECT_EQ(aliasSlices(doubleA, doubleB, /*iterationDistance=*/1), AliasResult::NoAlias);
  EXPECT_EQ(aliasSlices(doubleA, doubleB, /*iterationDistance=*/2), AliasResult::MayAlias);
  EXPECT_EQ(firstAliasingIterationDistance(doubleA, doubleB), std::optional<int64_t>(2));

  // 相位差 1 的两个访问在相邻迭代上就会撞——级数不是唯一的因素。
  BufferSlice shifted = rotating(inductionVariable, {{2, 1}});
  EXPECT_EQ(firstAliasingIterationDistance(shifted, doubleB), std::optional<int64_t>(1));

  // 两维各自轮转时，同时撞上要等到两个同余式都满足——周期是最小公倍数，不是最大
  // 级数。按最大级数截断会在 d ≤ 4 里找不到解就收工，把真竞争报成「永不重叠」。
  BufferSlice twoDimensionalA = rotating(inductionVariable, {{3, 0}, {4, 0}});
  BufferSlice twoDimensionalB = rotating(inductionVariable, {{3, 1}, {4, 3}});
  EXPECT_EQ(firstAliasingIterationDistance(twoDimensionalA, twoDimensionalB), std::optional<int64_t>(5));
  EXPECT_EQ(firstAliasingIterationDistance(twoDimensionalB, twoDimensionalA), std::optional<int64_t>(7));

  // 相位差只有在两侧由同一个归纳变量驱动时才有意义。换一个归纳变量，两个「轮转」之间
  // 没有共同的 i，任何距离上都必须按可能重叠处理。
  BufferSlice foreign = rotating(otherInductionVariable, {{2, 1}});
  EXPECT_EQ(aliasSlices(doubleA, foreign, /*iterationDistance=*/1), AliasResult::MayAlias);
  EXPECT_EQ(firstAliasingIterationDistance(doubleA, foreign), std::optional<int64_t>(1));

  // 距离所属的循环也必须是驱动这个下标的那个循环，否则是两套坐标系。
  EXPECT_EQ(aliasSlices(doubleA, doubleB, /*iterationDistance=*/1, otherInductionVariable), AliasResult::MayAlias);
}

}  // namespace
