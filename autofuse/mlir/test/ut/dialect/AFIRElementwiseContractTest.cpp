/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- AFIRElementwiseContractTest.cpp -------------------------*- C++ -*-===//
//
// elementwise ODS 上两条**写对了但没人读**的轴。两条都不是 LIT 能钉住的，理由不同：
//
//   - 结果元素类型（`logicalResultElementType`）只在 `inferReturnTypes` 的「没有目的地」
//     那一支里读。而两种造 tensor 形态 op 的办法都不走那一支：汇编里把结果类型写全了，
//     `AfirBuilder::createLogicalNodeOp` 也是拿 `TypeRange{resultTy}` 建的。于是这条轴
//     只能由直接调用 `inferReturnTypes` 的用例来钉。
//   - 标量操作数不带 `Read` 这条，观察后果是访存排序；而标量是 `arith.constant`
//     （`Pure`），多挂一个 `Read` 在今天任何一张图上都改不了调度结论。所以钉它的办法
//     不是造一张会重排的图，而是直接断言 effect 列表。
//
//===----------------------------------------------------------------------===//

#include "dialect/AFIR/IR/AFIR.h"

#include <string>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::afir;

namespace {

class AFIRElementwiseContractTest : public testing::Test {
 protected:
  void SetUp() override {
    context.getOrLoadDialect<AFIRDialect>();
    context.getOrLoadDialect<arith::ArithDialect>();
    context.getOrLoadDialect<func::FuncDialect>();
    context.getOrLoadDialect<memref::MemRefDialect>();
  }

  OwningOpRef<ModuleOp> Parse(llvm::StringRef source) {
    return parseSourceString<ModuleOp>(source, &context);
  }

  // 模块里第一个 `OpTy`。用例都只放一个。
  template <typename OpTy>
  OpTy FindFirst(ModuleOp module) {
    OpTy found;
    module.walk([&found](OpTy op) {
      found = op;
      return WalkResult::interrupt();
    });
    return found;
  }

  // 拿这个 op 自己的操作数 / 属性 / properties 去问它的 `inferReturnTypes`，**跳过汇编
  // 里写着的那个结果类型**。这是让「没有目的地」那一支唯一走得到的办法。
  template <typename OpTy>
  llvm::SmallVector<Type> InferReturnTypes(OpTy op) {
    llvm::SmallVector<Type> inferred;
    Operation *operation = op.getOperation();
    if (failed(OpTy::inferReturnTypes(&context, operation->getLoc(), operation->getOperands(),
                                      operation->getAttrDictionary(), operation->getPropertiesStorage(),
                                      operation->getRegions(), inferred))) {
      return {};
    }
    return inferred;
  }

  MLIRContext context;
};

//===----------------------------------------------------------------------===//
// `logicalResultElementType`
//===----------------------------------------------------------------------===//

// `truediv` 存在的全部理由：i32 ⊕ i32 的结果是 f32。这条规则写在
// `trueDivResultElementType` 里，ODS 与 lowering 两处共读它，但 tensor 形态这一处
// 至今没有用例——把它改回「取第一个输入的元素类型」，今天一个用例都不会红。
TEST_F(AFIRElementwiseContractTest, TrueDivInfersF32FromI32Inputs) {
  OwningOpRef<ModuleOp> module = Parse(R"mlir(
    func.func @truediv_i32(%lhs: tensor<4xi32>, %rhs: tensor<4xi32>) -> tensor<4xf32> {
      %0 = afir.truediv %lhs, %rhs : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  )mlir");
  ASSERT_TRUE(module);
  auto op = FindFirst<TrueDivOp>(*module);
  ASSERT_TRUE(op);

  llvm::SmallVector<Type> inferred = InferReturnTypes(op);
  ASSERT_EQ(inferred.size(), 1u);
  EXPECT_EQ(inferred[0], RankedTensorType::get({4}, Float32Type::get(&context)));
}

// 另一侧：f32 输入时结果就是 f32，不是「一律 f32」也不是「一律换类型」。
// 少了这一条，把 `trueDivResultElementType` 写成恒返回 f32 也能通过上一条。
TEST_F(AFIRElementwiseContractTest, TrueDivKeepsF32InputsUnchanged) {
  OwningOpRef<ModuleOp> module = Parse(R"mlir(
    func.func @truediv_f32(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>) -> tensor<4xf32> {
      %0 = afir.truediv %lhs, %rhs : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  )mlir");
  ASSERT_TRUE(module);
  auto op = FindFirst<TrueDivOp>(*module);
  ASSERT_TRUE(op);

  llvm::SmallVector<Type> inferred = InferReturnTypes(op);
  ASSERT_EQ(inferred.size(), 1u);
  EXPECT_EQ(inferred[0], RankedTensorType::get({4}, Float32Type::get(&context)));
}

// 比较族：结果恒为 ui8，与输入的 dtype 无关。这一族六个 op 共用 ODS 里的同一行
// （`AFIR_CompareOp` 的那个 `ui8`），六个手写的 `inferReturnTypeComponents` 是被它替掉的
// ——但替掉前后这条轴的覆盖都是零。
TEST_F(AFIRElementwiseContractTest, CompareFamilyInfersUint8Result) {
  OwningOpRef<ModuleOp> module = Parse(R"mlir(
    func.func @eq_f16(%lhs: tensor<4xf16>, %rhs: tensor<4xf16>) -> tensor<4xui8> {
      %0 = afir.eq %lhs, %rhs : (tensor<4xf16>, tensor<4xf16>) -> tensor<4xui8>
      return %0 : tensor<4xui8>
    }
  )mlir");
  ASSERT_TRUE(module);
  auto op = FindFirst<EqOp>(*module);
  ASSERT_TRUE(op);

  llvm::SmallVector<Type> inferred = InferReturnTypes(op);
  ASSERT_EQ(inferred.size(), 1u);
  EXPECT_EQ(inferred[0], RankedTensorType::get({4}, IntegerType::get(&context, 8, IntegerType::Unsigned)));
}

// 对照组：结果 dtype 与输入相同的那一族，`logicalResultElementType` 传的是空。
// 没有这一条，把空的那一支也改成 ui8 不会有任何用例红。
TEST_F(AFIRElementwiseContractTest, AddInfersInputElementType) {
  OwningOpRef<ModuleOp> module = Parse(R"mlir(
    func.func @add_f16(%lhs: tensor<4xf16>, %rhs: tensor<4xf16>) -> tensor<4xf16> {
      %0 = afir.add %lhs, %rhs : (tensor<4xf16>, tensor<4xf16>) -> tensor<4xf16>
      return %0 : tensor<4xf16>
    }
  )mlir");
  ASSERT_TRUE(module);
  auto op = FindFirst<AddOp>(*module);
  ASSERT_TRUE(op);

  llvm::SmallVector<Type> inferred = InferReturnTypes(op);
  ASSERT_EQ(inferred.size(), 1u);
  EXPECT_EQ(inferred[0], RankedTensorType::get({4}, Float16Type::get(&context)));
}

//===----------------------------------------------------------------------===//
// buffer 形态的 effect 列表
//===----------------------------------------------------------------------===//

constexpr llvm::StringLiteral kScalarAddModule = R"mlir(
#tile = #afir.buffer_desc<space = <ub>, buffer_id = 0, life_time_axis_id = -1>
#compute = #afir.tensor_desc<axis = [0], role = <compute>>

func.func private @add_scalar(
    %lhs: memref<16xf16, #afir.space<ub>> {afir.buffer_desc = #tile},
    %out: memref<16xf16, #afir.space<ub>> {afir.buffer_desc = #tile}) attributes {
  afir.function_role = #afir.function_role<kernel_body>,
  afir.arg_descs = [#compute, #compute]
} {
  %scalar = arith.constant 2.000000e+00 : f16
  %sum = afir.add %lhs, %scalar outs(%out) {
    afir.destination_descs = [#compute],
    afir.operand_descs = [#compute, #compute],
    afir.result_descs = [#compute]
  } : (memref<16xf16, #afir.space<ub>>, f16),
      memref<16xf16, #afir.space<ub>> -> memref<16xf16, #afir.space<ub>>
  return
}
)mlir";

// 标量操作数**不带 `Read`**。这条分支在 `getBufferizedElementwiseEffects` 里，删掉那句
// `contributesShape` 判断后所有 lit 用例与 E2E 仍然全绿：标量是 `arith.constant`
// （`Pure`），多一个 `Read` 在今天的图上改不了任何调度结论。
//
// 所以这里直接断言 effect 列表：tile 一个 `Read`，目的地一个 `Write`，标量一条都没有。
TEST_F(AFIRElementwiseContractTest, ScalarOperandCarriesNoReadEffect) {
  OwningOpRef<ModuleOp> module = Parse(kScalarAddModule);
  ASSERT_TRUE(module);
  auto op = FindFirst<AddOp>(*module);
  ASSERT_TRUE(op);
  ASSERT_TRUE(op.isBufferizedForm());

  auto effectOp = dyn_cast<MemoryEffectOpInterface>(op.getOperation());
  ASSERT_TRUE(effectOp);
  llvm::SmallVector<MemoryEffects::EffectInstance> effects;
  effectOp.getEffects(effects);

  const Value tile = op.getOperation()->getOperand(0);
  const Value scalar = op.getOperation()->getOperand(1);
  ASSERT_TRUE(llvm::isa<FloatType>(scalar.getType()));

  unsigned reads = 0;
  unsigned writes = 0;
  for (const MemoryEffects::EffectInstance &effect : effects) {
    EXPECT_NE(effect.getValue(), scalar) << "标量挂上了内存 effect";
    if (llvm::isa<MemoryEffects::Read>(effect.getEffect())) {
      ++reads;
      EXPECT_EQ(effect.getValue(), tile);
    } else if (llvm::isa<MemoryEffects::Write>(effect.getEffect())) {
      ++writes;
      EXPECT_EQ(effect.getValue(), op.getDestination());
    }
  }
  // 个数一起钉：只查「标量没有 effect」的话，把整个循环删掉也能过。
  EXPECT_EQ(reads, 1u);
  EXPECT_EQ(writes, 1u);
}

}  // namespace
