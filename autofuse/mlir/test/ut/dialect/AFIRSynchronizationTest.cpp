/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 同步原语的副作用声明（设计文档 3.6）。
//
// 为什么值得单独一个用例：这三个 Op 的「不会被优化掉」此前是隐式的——没有
// MemoryEffect 接口即视为未知副作用，于是 DCE / CSE 都不动它们。那份安全依赖于
// 「没人给它们加 `Pure`」，而 `afir.event_wait` 无结果、无可见产物，一旦被误加
// `Pure` 就会被静默删掉，失败形态是偶发的错数据——LIT 看不出来，因为被删掉的 IR
// 本身仍然合法。
//
// 现在三个 Op 都显式声明作用在 `PipelineResource` 上。本用例钉住这个声明的效果
// 而不是它的写法：MLIR 的优化按 `isMemoryEffectFree` / `wouldOpBeTriviallyDead` 决策，
// 所以直接问这两个函数。

#include "dialect/AFIR/IR/AFIR.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::afir;

namespace {

class AFIRSynchronizationTest : public testing::Test {
 protected:
  void SetUp() override {
    context.getOrLoadDialect<AFIRDialect>();
    context.getOrLoadDialect<func::FuncDialect>();
  }

  OwningOpRef<ModuleOp> Parse(llvm::StringRef body) {
    std::string source = std::string("module {\n  func.func @k() {\n") + body.str() + "\n    return\n  }\n}\n";
    return parseSourceString<ModuleOp>(source, &context);
  }

  MLIRContext context;
};

TEST_F(AFIRSynchronizationTest, SynchronizationPrimitivesAreNotOptimizableAway) {
  auto module = Parse(R"mlir(
    %t = afir.event_record : !afir.token<from = transfer_in, to = vector>
    afir.event_wait %t : !afir.token<from = transfer_in, to = vector>
    afir.pipe_barrier vector
  )mlir");
  ASSERT_TRUE(module);

  unsigned checked = 0;
  module->walk([&](Operation *op) {
    if (!isa<EventRecordOp, EventWaitOp, PipeBarrierOp>(op)) {
      return;
    }
    ++checked;
    EXPECT_FALSE(isMemoryEffectFree(op)) << op->getName().getStringRef().str()
                                         << " must declare its effect on pipeline ordering";
    EXPECT_FALSE(wouldOpBeTriviallyDead(op))
        << op->getName().getStringRef().str() << " must not be dead-code eliminated";
  });
  EXPECT_EQ(checked, 3U);
}

// 两条 `afir.event_record` 即使类型完全相同也是两个不同的事件，不能被 CSE 合并成
// 一条——合并之后第二个 wait 等的是第一个 record，跨的那段依赖就没被挡住。
TEST_F(AFIRSynchronizationTest, IdenticalEventRecordsAreNotInterchangeable) {
  auto module = Parse(R"mlir(
    %a = afir.event_record : !afir.token<from = transfer_in, to = vector>
    %b = afir.event_record : !afir.token<from = transfer_in, to = vector>
    afir.event_wait %a : !afir.token<from = transfer_in, to = vector>
    afir.event_wait %b : !afir.token<from = transfer_in, to = vector>
  )mlir");
  ASSERT_TRUE(module);

  llvm::SmallVector<Operation *> records;
  module->walk([&](EventRecordOp op) { records.push_back(op); });
  ASSERT_EQ(records.size(), 2U);
  // CSE 只合并 `isMemoryEffectFree` 的算子；声明了副作用即不可合并。
  EXPECT_FALSE(isMemoryEffectFree(records[0]));
  EXPECT_FALSE(isMemoryEffectFree(records[1]));
}

}  // namespace
