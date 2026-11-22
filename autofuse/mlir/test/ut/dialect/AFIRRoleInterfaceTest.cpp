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

void LoadDialects(MLIRContext &context) {
  context.getOrLoadDialect<AFIRDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
}

TEST(AFIRRoleInterfaceTest, ReturnsCanonicalTargetRole) {
  MLIRContext context;
  LoadDialects(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(
module {
  func.func @test(%input: memref<4xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>,>},
                  %output: memref<4xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>,>}) attributes {
    afir.arg_descs = [#afir.tensor_desc<role = <input>>, #afir.tensor_desc<role = <output>>]
  } {
    %tmp = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>,>} : memref<4xf16>
    %loaded = afir.load %input outs(%tmp) {
      afir.operand_descs = [#afir.tensor_desc<role = <input>>],
      afir.destination_descs = [#afir.tensor_desc<role = <compute>>],
      afir.result_descs = [#afir.tensor_desc<role = <compute>>],
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<4xf16>, memref<4xf16> -> memref<4xf16>
    afir.store %loaded outs(%output) {
      afir.operand_descs = [#afir.tensor_desc<role = <compute>>],
      afir.destination_descs = [#afir.tensor_desc<role = <output>>],
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<4xf16>, memref<4xf16>
    return
  }
}
)mlir",
                                            &context);
  ASSERT_TRUE(module);

  Operation *load = nullptr;
  Operation *store = nullptr;
  module->walk([&](Operation *operation) {
    if (isa<LoadOp>(operation)) {
      load = operation;
    }
    if (isa<StoreOp>(operation)) {
      store = operation;
    }
  });
  ASSERT_NE(load, nullptr);
  ASSERT_NE(store, nullptr);
  auto loadInterface = dyn_cast<AFIRRoleInterface>(load);
  auto storeInterface = dyn_cast<AFIRRoleInterface>(store);
  ASSERT_TRUE(loadInterface);
  ASSERT_TRUE(storeInterface);
  auto loadRole = loadInterface.getRole();
  auto storeRole = storeInterface.getRole();
  ASSERT_TRUE(succeeded(loadRole));
  ASSERT_TRUE(succeeded(storeRole));
  EXPECT_EQ(*loadRole, Role::Compute);
  EXPECT_EQ(*storeRole, Role::Output);
}

TEST(AFIRRoleInterfaceTest, RejectsMissingOrConflictingTargetDescriptors) {
  MLIRContext context;
  LoadDialects(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(
module {
  func.func @missing(%lhs: tensor<4xf16>, %rhs: tensor<4xf16>) -> tensor<4xf16> {
    %missing = afir.add %lhs, %rhs : (tensor<4xf16>, tensor<4xf16>) -> tensor<4xf16>
    return %missing : tensor<4xf16>
  }
  func.func @conflict(%lhs: memref<4xf16> {afir.buffer_desc = #afir.buffer_desc<space = <ub>,>},
                      %rhs: memref<4xf16> {afir.buffer_desc = #afir.buffer_desc<space = <ub>,>},
                      %dst: memref<4xf16> {afir.buffer_desc = #afir.buffer_desc<space = <ub>,>}) attributes {
    afir.arg_descs = [#afir.tensor_desc<role = <compute>>, #afir.tensor_desc<role = <compute>>,
                      #afir.tensor_desc<role = <compute>>]
  } {
    %conflict = afir.add %lhs, %rhs outs(%dst) {
      afir.operand_descs = [#afir.tensor_desc<role = <compute>>, #afir.tensor_desc<role = <compute>>],
      afir.destination_descs = [#afir.tensor_desc<role = <compute>>],
      afir.result_descs = [#afir.tensor_desc<role = <output>>]
    } : (memref<4xf16>, memref<4xf16>), memref<4xf16> -> memref<4xf16>
    return
  }
}
)mlir",
                                            &context);
  ASSERT_TRUE(module);

  Operation *missing = nullptr;
  Operation *conflict = nullptr;
  module->walk([&](Operation *operation) {
    if (!isa<AddOp>(operation)) {
      return;
    }
    if (operation->getAttr(kResultDescsAttrName)) {
      conflict = operation;
    } else {
      missing = operation;
    }
  });
  ASSERT_NE(missing, nullptr);
  ASSERT_NE(conflict, nullptr);
  auto missingInterface = dyn_cast<AFIRRoleInterface>(missing);
  auto conflictInterface = dyn_cast<AFIRRoleInterface>(conflict);
  ASSERT_TRUE(missingInterface);
  ASSERT_TRUE(conflictInterface);
  auto missingRole = missingInterface.getRole();
  auto conflictRole = conflictInterface.getRole();
  EXPECT_TRUE(failed(missingRole));
  EXPECT_TRUE(failed(conflictRole));
}

}  // namespace
