/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 同步族下降产出的 AscendC 算子必须能翻译成 C++ 源码。
//
// 这条不是多余的：`Target/AscendC/Translation.cpp` 的分发表 `PrintableOpTypes` 是手写
// 的，`ascendc::SetFlagOp` 并不在手写部分里——它由 ODS 生成的 `GET_OP_TYPE_LIST` 补进来。
// 也就是说「能不能翻译」取决于一个生成列表的内容，而不是某处显式的登记。LIT 只能断言
// 下降产出了哪些算子，断言不了它们能翻译；这里直接调 `translateToAscendC` 把这一段钉住。

#include <string>

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Target/Asc/Translation.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

namespace mlir::ascend::kernel {
namespace {

class SynchronizationTranslationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context.getOrLoadDialect<func::FuncDialect>();
    context.getOrLoadDialect<ascendc::AscendCDialect>();
  }

  std::string Translate(llvm::StringRef source) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(source, &context);
    EXPECT_TRUE(module);
    if (!module) {
      return {};
    }
    std::string text;
    llvm::raw_string_ostream os(text);
    EXPECT_TRUE(succeeded(translateToAscendC(module->getOperation(), os)));
    return os.str();
  }

  MLIRContext context;
};

TEST_F(SynchronizationTranslationTest, LoweredSynchronizationOpsReachAscendCSource) {
  std::string source = Translate(R"mlir(
      module {
        func.func @kernel() {
          %pipe = ascendc.pipe
          %id = ascendc.pipe.fetch_event_id %pipe, v_mte3 : i8
          ascendc.set_flag v_mte3, %id : i8
          ascendc.wait_flag v_mte3, %id : i8
          ascendc.pipe_barrier pipe_v
          return
        }
      }
    )mlir");
  // 方向必须原样出现在 HardEvent 模板参数上——写反在源码里是看得见的。
  EXPECT_NE(source.find("HardEvent::V_MTE3"), std::string::npos) << source;
  EXPECT_NE(source.find("SetFlag"), std::string::npos) << source;
  EXPECT_NE(source.find("WaitFlag"), std::string::npos) << source;
  EXPECT_NE(source.find("PipeBarrier"), std::string::npos) << source;
}

TEST_F(SynchronizationTranslationTest, SyncAllHardEmitsAivOnlyProtocol) {
  // 下降产出的 `ascendc.sync_all_hard`（`Implementations/Synchronization.cpp` 显式置
  // `isAIVOnly`）走 tablegen 自动 emitter，打出 `AscendC::SyncAll()`——缺省
  // `isAIVOnly = true`，即 AIV-only 协议，恰是非 cube（纯 AIV）范围内唯一正确的取值；
  // 需 AIC 参与的混模协议 `SyncAll<false>` 在纯 AIV 生成路径下死锁（910B3 实测）。auto
  // emitter 结构上打不出 `<false>`，属性只作 IR 层意图声明，不参与打印。
  std::string source = Translate(R"mlir(
      module {
        func.func @kernel() {
          ascendc.sync_all_hard {isAIVOnly}
          return
        }
      }
    )mlir");
  EXPECT_NE(source.find("AscendC::SyncAll()"), std::string::npos) << source;
  EXPECT_EQ(source.find("SyncAll<false>"), std::string::npos) << source;
}

}  // namespace
}  // namespace mlir::ascend::kernel
