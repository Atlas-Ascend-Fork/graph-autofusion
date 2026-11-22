/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 依赖方向单向（设计文档 2.2、验收 1）的运行期那一半。
//
// 静态那一半在 `tools/af-mlir-tools/check_afir_layering.py`：源码里不出现 `afplan`、
// CMake 里不链接 `AFPlanDialect`。两半各管一件事，谁都替代不了谁——脚本看得见依赖但
// 看不见语义，本用例看得见语义但看不见一个没被走到的 `#include`。
//
// 这里问的是：`afir` 的解析与校验路径会不会自己去碰 afplan。做法是把 `afplan`
// 从 context 里拿掉：一个只加载了 `afir` 的 context 能完整解析并校验一个 kernel body，
// 且解析结束后 `afplan` 仍然没被加载。若哪天 `afir` 的某个 verifier 开始读候选身份
// （`afplan.variant` 之类），它要么在这里崩、要么把 `afplan` 拉进 context——两种都会
// 让本用例变红，而 LIT 不会：LIT 的 af-opt 把所有方言都注册好了。

#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFPlan/IR/AFPlan.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace mlir;

namespace {

// 一个带完整规划结果的 kernel body：函数角色、轴表、buffer descriptor、同步 token
// 与运行期尺寸都在场。它是 `afir` 这一层能表达的东西的横切面，不是最小用例——最小
// 用例走不到那些 verifier，而本用例要问的正是「哪个 verifier 会去碰 afplan」。
constexpr llvm::StringLiteral kKernelBody = R"mlir(
module {
  func.func private @kernel(%input: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>}) attributes {
    afir.arg_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
    afir.asc_graph = #afir.asc_graph<
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>],
      type = Compute,
      size_var = [#afir.expr<"s0">]>,
    afir.function_role = #afir.function_role<kernel_body>
  } {
    %size = afir.tiling_param "s0" : index
    %tile = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, slot_count = 2, queue_depth = 2, size_from_resource = true, alignment_bytes = 32, life_time_axis_id = -1>} : memref<16xf16>
    %token = afir.event_record : !afir.token<from = transfer_in, to = vector>
    afir.event_wait %token : !afir.token<from = transfer_in, to = vector>
    memref.dealloc %tile : memref<16xf16>
    return
  }
}
)mlir";

TEST(AFIRLayeringTest, AfirParsesAndVerifiesWithoutTheAFPlanDialect) {
  MLIRContext context;
  context.getOrLoadDialect<afir::AFIRDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  ASSERT_EQ(context.getLoadedDialect<afplan::AFPlanDialect>(), nullptr);

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(kKernelBody, &context);
  ASSERT_TRUE(module);
  EXPECT_TRUE(succeeded(verify(*module)));
  // 解析与校验都没有把 `afplan` 拉进来。这一条比「解析成功」更强：MLIR 会按需加载
  // 已注册的方言，所以一个悄悄冒出来的依赖会表现为这里多了一个已加载方言，而不是失败。
  EXPECT_EQ(context.getLoadedDialect<afplan::AFPlanDialect>(), nullptr);
}

TEST(AFIRLayeringTest, CandidateIdentityIsNotPartOfAfir) {
  MLIRContext context;
  context.getOrLoadDialect<afir::AFIRDialect>();
  context.getOrLoadDialect<func::FuncDialect>();

  // 反向的那一条：候选身份确实住在另一个命名空间里。只加载 `afir` 时它解析不出来，
  // 说明它不是被 `afir` 顺带注册进来的一组字段——否则「迁到 afplan」就只是换了个前缀。
  constexpr llvm::StringLiteral kWithCandidate = R"mlir(
module attributes {afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>} {
}
)mlir";
  EXPECT_FALSE(parseSourceString<ModuleOp>(kWithCandidate, &context));

  // 同一份 IR 在加载了 `afplan` 之后解析得出来——上面的失败是「没注册」，不是写法错。
  context.getOrLoadDialect<afplan::AFPlanDialect>();
  EXPECT_TRUE(parseSourceString<ModuleOp>(kWithCandidate, &context));
}

}  // namespace
