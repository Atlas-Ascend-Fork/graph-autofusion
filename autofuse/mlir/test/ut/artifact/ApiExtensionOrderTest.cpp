/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "artifact/KernelSource/Emit/ApiExtensionOrder.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {
using mlir::ascend::kernel::ApiExtensionOrder;

// 声明的来源 op 只被诊断用到（报在哪个 op 上），排序本身不看它，所以这里拿空 module 当占位。
// 这也是这个单元能被直接测的原因：真正成环的声明组合今天一个都不存在——`resolve` 的失败路径
// 是给将来写错 ODS 的人准备的护栏，从已注册的 op 构不出来。
class ApiExtensionOrderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mlir::OpBuilder builder(&context);
    declaringOp = mlir::ModuleOp::create(builder.getUnknownLoc());
    context.getDiagEngine().registerHandler([this](mlir::Diagnostic &diagnostic) {
      diagnostics += diagnostic.str();
      return mlir::success();
    });
  }

  mlir::Operation *op() {
    return declaringOp.get();
  }

  std::vector<std::string> resolve(const ApiExtensionOrder &order) {
    auto resolved = order.resolve();
    EXPECT_TRUE(succeeded(resolved)) << diagnostics;
    std::vector<std::string> names;
    if (succeeded(resolved)) {
      for (llvm::StringRef header : *resolved) {
        names.push_back(header.str());
      }
    }
    return names;
  }

  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> declaringOp;
  std::string diagnostics;
};

// 全部边都是正字典序时输出精确等于字典序。这一条钉住的是「2026-10-21 之前接通的那些 op 的
// include 块一个字节都不变」：它们的声明全是正字典序，所以换成拓扑排不许改动它们的输出。
TEST_F(ApiExtensionOrderTest, MatchesDictionaryOrderWhenEveryEdgeRunsForward) {
  ApiExtensionOrder order;
  order.declare(op(), {"duplicate.h", "where.h"});
  order.declare(op(), {"compare.h", "compare_v2.h"});
  order.declare(op(), {"broadcast.h", "duplicate.h"});
  order.declare(op(), {"scalar_div.h", "true_div.h"});
  order.declare(op(), {"cast.h", "sign.h"});
  order.declare(op(), {"abs.h"});
  EXPECT_EQ(resolve(order),
            (std::vector<std::string>{"abs.h", "broadcast.h", "cast.h", "compare.h", "compare_v2.h", "duplicate.h",
                                      "scalar_div.h", "sign.h", "true_div.h", "where.h"}));
}

// 反字典序的那条边要按声明顺序生成。`i0e.h` 用的 `I0_A` / `I0_B` 只有 `modified_bessel_i0.h` 定义，
// 而 `i0e_reg_base.h` 的字典序在它前面——按字典序生成的是一份编不过的 kernel。
TEST_F(ApiExtensionOrderTest, HonorsAnEdgeThatRunsAgainstDictionaryOrder) {
  ApiExtensionOrder order;
  order.declare(op(), {"modified_bessel_i0_reg_base.h", "i0e_reg_base.h"});
  EXPECT_EQ(resolve(order), (std::vector<std::string>{"modified_bessel_i0_reg_base.h", "i0e_reg_base.h"}));
}

// 跨声明的约束，而且与声明次序无关。这一格是按首见次序生成（legacy 的做法）保不住的：
// `airy_ai` 把 `trigonometric_function_utils` 放在列表首项，`bessel_j0` 要 `bessel_j_utils`
// 排在它前面，谁在前就取决于两个 op 哪个先被 walk 到，也就是取决于 plan。
TEST_F(ApiExtensionOrderTest, HonorsCrossDeclarationConstraintsInEitherDeclarationSequence) {
  const std::vector<std::string> expected = {"bessel_j_utils_reg_base.h", "trigonometric_function_utils_reg_base.h"};

  ApiExtensionOrder airyAiFirst;
  airyAiFirst.declare(op(), {"trigonometric_function_utils_reg_base.h"});
  airyAiFirst.declare(op(), {"bessel_j_utils_reg_base.h", "trigonometric_function_utils_reg_base.h"});
  EXPECT_EQ(resolve(airyAiFirst), expected);

  ApiExtensionOrder besselJ0First;
  besselJ0First.declare(op(), {"bessel_j_utils_reg_base.h", "trigonometric_function_utils_reg_base.h"});
  besselJ0First.declare(op(), {"trigonometric_function_utils_reg_base.h"});
  EXPECT_EQ(resolve(besselJ0First), expected);
}

// 传递约束：声明列表只记相邻的两两先后，`a` 要排在 `c` 前面是靠拓扑排传出来的。
TEST_F(ApiExtensionOrderTest, PropagatesOrderTransitivelyAcrossDeclarations) {
  ApiExtensionOrder order;
  order.declare(op(), {"modified_bessel_utils_reg_base.h", "modified_bessel_i1_reg_base.h", "i1e_reg_base.h"});
  EXPECT_EQ(resolve(order), (std::vector<std::string>{"modified_bessel_utils_reg_base.h",
                                                      "modified_bessel_i1_reg_base.h", "i1e_reg_base.h"}));
}

// 排不出来时报的是整个环，按前进方向、首尾相同，读的人不用自己把环接上。
TEST_F(ApiExtensionOrderTest, ReportsTheFullCyclePath) {
  ApiExtensionOrder order;
  order.declare(op(), {"a.h", "b.h"});
  order.declare(op(), {"b.h", "a.h"});
  EXPECT_TRUE(failed(order.resolve()));
  EXPECT_NE(diagnostics.find("'a.h' -> 'b.h' -> 'a.h'"), std::string::npos) << diagnostics;
  EXPECT_NE(diagnostics.find("the declared order is cyclic"), std::string::npos) << diagnostics;
}

// 三个头的环也要打完整，而不是只报其中两个。
TEST_F(ApiExtensionOrderTest, ReportsEveryHeaderOnALongerCycle) {
  ApiExtensionOrder order;
  order.declare(op(), {"a.h", "b.h", "c.h"});
  order.declare(op(), {"c.h", "a.h"});
  EXPECT_TRUE(failed(order.resolve()));
  EXPECT_NE(diagnostics.find("'a.h' -> 'b.h' -> 'c.h' -> 'a.h'"), std::string::npos) << diagnostics;
}

// 同一个列表里的重名走的也是这条路，打出来的路径正好点出重复的那个名字。
TEST_F(ApiExtensionOrderTest, ReportsARepeatedHeaderWithinOneDeclarationAsACycle) {
  ApiExtensionOrder order;
  order.declare(op(), {"a.h", "b.h", "a.h"});
  EXPECT_TRUE(failed(order.resolve()));
  EXPECT_NE(diagnostics.find("'a.h' -> 'b.h' -> 'a.h'"), std::string::npos) << diagnostics;
}

// 紧挨着的重名是个自环，路径退成一条边。
TEST_F(ApiExtensionOrderTest, ReportsAdjacentRepetitionAsASelfCycle) {
  ApiExtensionOrder order;
  order.declare(op(), {"a.h", "a.h"});
  EXPECT_TRUE(failed(order.resolve()));
  EXPECT_NE(diagnostics.find("'a.h' -> 'a.h'"), std::string::npos) << diagnostics;
}

// 空声明不建节点：没有扩展头的 kernel 一个字节都不变，靠的就是这一条。
TEST_F(ApiExtensionOrderTest, StaysEmptyForDeclarationsWithoutHeaders) {
  ApiExtensionOrder order;
  EXPECT_TRUE(order.empty());
  order.declare(op(), {});
  EXPECT_TRUE(order.empty());
  EXPECT_FALSE(order.contains("abs.h"));
  EXPECT_TRUE(resolve(order).empty());

  order.declare(op(), {"abs.h"});
  EXPECT_FALSE(order.empty());
  EXPECT_TRUE(order.contains("abs.h"));
  EXPECT_FALSE(order.contains("neg.h"));
}
}  // namespace
