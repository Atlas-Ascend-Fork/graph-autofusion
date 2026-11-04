/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/AFIRToAscendC/AFIRToAscendCPasses.h"
#include "conversion/AFIRToAscendC/AscendCKernelABI.h"

#include "dialect/AFIR/IR/AFIR.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::afir;
using namespace mlir::ascend::kernel;

namespace {

constexpr StringLiteral kBufferKernel = R"mlir(module {
  func.func private @buffer_form_copy(
      %input: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>},
      %output: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>}) attributes {
    afir.arg_descs = [
      #afir.tensor_desc<axis = [0], role = <input>>,
      #afir.tensor_desc<axis = [0], role = <output>>
    ],
    afir.asc_graph = #afir.asc_graph<
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>],
      type = Compute,
      size_var = [#afir.expr<"s0">]>,
    afir.function_role = #afir.function_role<kernel_body>
  } {
    %tmp = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * s0)">, alignment_bytes = 32, life_time_axis_id = -1>} : memref<16xf16>
    %loaded = afir.load %input outs(%tmp) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    afir.store %loaded outs(%output) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <output>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16>
    return
  }
}
)mlir";

std::string Print(Operation *op) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  op->print(stream);
  return text;
}

class KernelBodySignatureTest : public testing::Test {
 protected:
  void SetUp() override {
    context.loadDialect<AFIRDialect, func::FuncDialect, memref::MemRefDialect, bufferization::BufferizationDialect,
                        emitasc::EmitAscDialect>();
    module = parseSourceString<ModuleOp>(kBufferKernel, &context);
    ASSERT_TRUE(module);
    function = module->lookupSymbol<func::FuncOp>("buffer_form_copy");
    ASSERT_TRUE(function);
  }

  LogicalResult Run() {
    diagnostic.clear();
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic &message) {
      llvm::raw_string_ostream stream(diagnostic);
      message.print(stream);
      return success();
    });
    PassManager manager(&context);
    manager.addPass(createAFIRKernelBodySignaturePass());
    auto result = manager.run(*module);
    function = module->lookupSymbol<func::FuncOp>("buffer_form_copy");
    return result;
  }

  void ExpectFailureWithoutChanges(StringRef expected) {
    const auto before = Print(*module);
    EXPECT_TRUE(failed(Run()));
    EXPECT_NE(diagnostic.find(expected.str()), std::string::npos) << diagnostic;
    EXPECT_EQ(Print(*module), before);
  }

  // kernel 参数的规划结果挂在它自己的 arg attr 上，一个值一份（设计文档 5.1 步 8），
  // 改一个参数不再需要重建整个数组。
  void SetArgumentBuffer(Attribute buffer, unsigned index = 0) {
    function.setArgAttr(index, kBufferDescAttrName, buffer);
  }

  // 在 `index` 处插一个按值标量 kernel 参数（AscGraph 的 `ScalarData`）：类型是元素类型
  // 本身、角色是 `scalar_input`、`afir.arg_descs` 里跟着插一格退化的描述子、不挂
  // `afir.buffer_desc`。`type` 可以传别的，用来构造反向判据要拦的形态。
  void InsertScalarArgument(unsigned index, Type type = {}) {
    if (!type) {
      type = Float16Type::get(&context);
    }
    NamedAttrList attrs;
    attrs.append(kKernelArgAttrName, KernelArgAttr::get(&context, KernelArgRole::ScalarInput, /*index=*/1));
    function.insertArgument(index, type, attrs.getDictionary(&context), function.getLoc());

    auto descs = function->getAttrOfType<ArrayAttr>(kArgDescsAttrName);
    SmallVector<Attribute> updated(descs.begin(), descs.end());
    updated.insert(updated.begin() + index, parseAttribute("#afir.tensor_desc<role = <input>>", &context));
    function->setAttr(kArgDescsAttrName, ArrayAttr::get(&context, updated));
  }

  MLIRContext context;
  OwningOpRef<ModuleOp> module;
  func::FuncOp function;
  std::string diagnostic;
};

TEST_F(KernelBodySignatureTest, MaterializesOnlyExternalBuffersAndPreservesInternalQueue) {
  ASSERT_TRUE(succeeded(Run())) << diagnostic;
  ASSERT_EQ(function.getNumArguments(), 4U);
  EXPECT_EQ(cast<MemRefType>(function.getArgument(0).getType()).getMemorySpace(),
            getAscendCGlobalMemorySpaceAttr(&context));
  EXPECT_EQ(function.getArgument(2).getType(), getAscendCWorkspaceType(&context));
  EXPECT_TRUE(getTilingSchemaField(function.getArgument(3).getType(), "q2_size"));
  EXPECT_FALSE(getTilingSchemaField(function.getArgument(3).getType(), "b2_size"));
  function.walk([&](memref::AllocOp alloc) { EXPECT_FALSE(alloc.getType().getMemorySpace()); });
  EXPECT_TRUE(succeeded(verify(*module)));
  const auto once = Print(*module);
  ASSERT_TRUE(succeeded(Run())) << diagnostic;
  EXPECT_EQ(Print(*module), once);
}

TEST_F(KernelBodySignatureTest, RejectsUbArgumentWithoutRetypingEarlierGmArgument) {
  SetArgumentBuffer(parseAttribute("#afir.buffer_desc<space = <ub>, buffer_id = 7, life_time_axis_id = -1>", &context),
                    1);
  ExpectFailureWithoutChanges("must describe external GM storage");
}

TEST_F(KernelBodySignatureTest, RejectsWrongDescriptorType) {
  SetArgumentBuffer(StringAttr::get(&context, "invalid"));
  ExpectFailureWithoutChanges("BufferDescAttr");
}

TEST_F(KernelBodySignatureTest, RejectsNonBoundaryTensorRole) {
  auto attrs = function->getAttrOfType<ArrayAttr>(kArgDescsAttrName);
  function->setAttr(
      kArgDescsAttrName,
      ArrayAttr::get(&context, {parseAttribute("#afir.tensor_desc<role = <compute>>", &context), attrs[1]}));
  ExpectFailureWithoutChanges("must have input or output role");
}

TEST_F(KernelBodySignatureTest, RejectsConflictingExplicitMemorySpace) {
  auto types = llvm::to_vector(function.getArgumentTypes());
  types[0] = MemRefType::get({16}, Float16Type::get(&context), MemRefLayoutAttrInterface{},
                             IntegerAttr::get(IntegerType::get(&context, 32), 3));
  function.getArgument(0).setType(types[0]);
  function.setType(FunctionType::get(&context, types, {}));
  ExpectFailureWithoutChanges("conflicting memory space");
}

// GM kernel 参数按 AscendC ABI 是连续 ND 存储。步 6 让 memref 能表达 NZ / fractal
// 之后，这条闸门保证这些片上布局不会无声地漏到 kernel 边界上——那里没有对应的 ABI
// 约定，错了不报错、只算错。
TEST_F(KernelBodySignatureTest, RejectsNonIdentityLayoutAtAbiBoundary) {
  auto types = llvm::to_vector(function.getArgumentTypes());
  auto nz = cast<MemRefLayoutAttrInterface>(
      parseAttribute("#afir.layout<outer = col_major, tiling = row_major, tile = [16, 16], rank = 2>", &context));
  types[0] = MemRefType::get({16, 16}, Float16Type::get(&context), nz, afir::SpaceAttr::get(&context, afir::Space::GM));
  function.getArgument(0).setType(types[0]);
  function.setType(FunctionType::get(&context, types, {}));
  ExpectFailureWithoutChanges("must have an identity layout at the AscendC ABI boundary");
}

TEST_F(KernelBodySignatureTest, RejectsGmQueueFields) {
  SetArgumentBuffer(parseAttribute(
      "#afir.buffer_desc<space = <gm>, slot_count = 2, queue_depth = 2, life_time_axis_id = -1>", &context));
  ExpectFailureWithoutChanges("cannot carry queue fields");
}

TEST_F(KernelBodySignatureTest, RevalidatesDescriptorsOnAlreadyMaterializedAbi) {
  ASSERT_TRUE(succeeded(Run()));
  SetArgumentBuffer(parseAttribute("#afir.buffer_desc<space = <ub>, buffer_id = 7, life_time_axis_id = -1>", &context));
  ExpectFailureWithoutChanges("must describe external GM storage");
}

TEST_F(KernelBodySignatureTest, RejectsMaterializedDeclarationWithoutAccessingMissingBlockArguments) {
  ASSERT_TRUE(succeeded(Run()));
  function.getBody().getBlocks().clear();
  ExpectFailureWithoutChanges("requires a function body with matching arguments");
}

// 按值标量形参穿过签名物化时**类型不变**：它不是一块存储，没有地址空间可换。物化前这里
// 是 `cast<MemRefType>`，遇到标量是 assert 失败而不是诊断。
// 按值标量与 GM 张量共用一个 index 序号空间，这条钉住由此而来的两件事。
//
// **形状是「标量夹在两个输入中间」**，比 CPU E2E 那条强：E2E 只有一个标量，它落在
// index 1、后面只剩 output / workspace / tiling，虽然也能照出分秩（`ScalarInput` 的
// 枚举值追加在末尾，比 `Output` 大），但照不出「标量与指针之间按 index 排先后」。
TEST(KernelArgumentIdentityTest, ScalarInputSharesTheInputIndexSpace) {
  // 同键：同一个 index 上的两类输入是**冲突**，不是两个各自合法的参数。
  EXPECT_EQ(getKernelArgumentKey({KernelArgumentKind::ScalarInput, 1}),
            getKernelArgumentKey({KernelArgumentKind::Input, 1}));
  EXPECT_NE(getKernelArgumentKey({KernelArgumentKind::ScalarInput, 1}),
            getKernelArgumentKey({KernelArgumentKind::Input, 2}));

  // 同秩：标量按 index 排在两个指针输入中间，不是整体排到它们之后。
  const SmallVector<KernelArgumentIdentity> arguments{{KernelArgumentKind::Output, 0},
                                                      {KernelArgumentKind::Input, 2},
                                                      {KernelArgumentKind::ScalarInput, 1},
                                                      {KernelArgumentKind::Input, 0},
                                                      {KernelArgumentKind::Workspace, -1}};
  SmallVector<std::string> ordered;
  for (unsigned old : getKernelArgumentPermutation(arguments)) {
    ordered.push_back(getKernelArgumentKey(arguments[old]));
  }
  EXPECT_EQ(ordered, (SmallVector<std::string>{"input:0", "input:1", "input:2", "output:0", "workspace"}));
}

TEST_F(KernelBodySignatureTest, MaterializesScalarInputArgumentWithoutRetypingIt) {
  InsertScalarArgument(1);
  ASSERT_TRUE(succeeded(Run())) << diagnostic;
  EXPECT_EQ(function.getArgument(1).getType(), Float16Type::get(&context));
  EXPECT_EQ(cast<MemRefType>(function.getArgument(0).getType()).getMemorySpace(),
            getAscendCGlobalMemorySpaceAttr(&context));
  EXPECT_TRUE(succeeded(verify(*module)));
}

// 角色与类型是一条**双向**不变量，这条测反方向：声明了 `scalar_input` 却给了 memref。
// 只判「放行标量」不判这一侧的话，两种参数就可以互相冒充。
TEST_F(KernelBodySignatureTest, RejectsScalarInputRoleOnShapedArgument) {
  InsertScalarArgument(1, MemRefType::get({16}, Float16Type::get(&context), MemRefLayoutAttrInterface{},
                                          SpaceAttr::get(&context, Space::GM)));
  ExpectFailureWithoutChanges("cannot be a shaped type");
}

// 按值标量没有存储，挂 `afir.buffer_desc` 说明有人把它当成了一块缓冲。
TEST_F(KernelBodySignatureTest, RejectsScalarInputArgumentCarryingBufferDescriptor) {
  InsertScalarArgument(1);
  SetArgumentBuffer(
      parseAttribute("#afir.buffer_desc<space = <gm>, size_bytes_expr = <\"(2 * s0)\">, life_time_axis_id = -1>",
                     &context),
      1);
  ExpectFailureWithoutChanges("cannot carry");
}

TEST_F(KernelBodySignatureTest, RejectsScalarDataArgumentOnAlreadyMaterializedAbi) {
  ASSERT_TRUE(succeeded(Run()));
  auto types = llvm::to_vector(function.getArgumentTypes());
  types[0] = IntegerType::get(&context, 64);
  function.getArgument(0).setType(types[0]);
  function.setType(FunctionType::get(&context, types, {}));
  ExpectFailureWithoutChanges("must be a GM memref");
}

TEST_F(KernelBodySignatureTest, AcceptsCompletedAbiAfterAfirOperationsHaveBeenLowered) {
  ASSERT_TRUE(succeeded(Run()));
  auto &block = function.front();
  while (&block.front() != block.getTerminator()) {
    block.getTerminator()->getPrevNode()->erase();
  }
  EXPECT_TRUE(succeeded(verify(*module)));
  const auto before = Print(*module);
  ASSERT_TRUE(succeeded(Run())) << diagnostic;
  EXPECT_EQ(Print(*module), before);
}

TEST_F(KernelBodySignatureTest, FailureInLaterFunctionDoesNotModifyEarlierFunction) {
  auto invalid = cast<func::FuncOp>(function->clone());
  invalid.setName("invalid_kernel");
  module->push_back(invalid);
  invalid->setAttr(kArgDescsAttrName, ArrayAttr::get(&context, {}));
  ExpectFailureWithoutChanges("descriptor arrays");
}

TEST_F(KernelBodySignatureTest, RejectsExistingCallersWithoutLeavingMismatchedCallSignatures) {
  OpBuilder builder(&context);
  auto caller = func::FuncOp::create(function.getLoc(), "caller", function.getFunctionType());
  module->push_back(caller);
  builder.setInsertionPointToStart(caller.addEntryBlock());
  builder.create<func::CallOp>(function.getLoc(), function, caller.getArguments());
  builder.create<func::ReturnOp>(function.getLoc());
  ASSERT_TRUE(succeeded(verify(*module)));
  ExpectFailureWithoutChanges("before creating symbol uses");
}

TEST_F(KernelBodySignatureTest, RejectsMissingResourceIdBeforePublishingWorkspaceArgument) {
  function.walk([&](Operation *op) {
    if (!op->hasTrait<OpTrait::AFIRTransferIn>()) {
      return;
    }
    // 规划结果只在值的定义点存一份：要让搬运落点缺 buffer_id，改的是 destination 的
    // 定义 Op，而不是使用点上的两份拷贝。
    Value destination = op->getOperand(op->getNumOperands() - 1);
    if (Operation *definingOp = destination.getDefiningOp()) {
      definingOp->setAttr(kBufferDescAttrName,
                          parseAttribute("#afir.buffer_desc<space = <ub>, life_time_axis_id = -1>", &context));
    }
  });
  ExpectFailureWithoutChanges("missing buffer_id");
}

// 队列身份只看 descriptor：`%staged` 带着 slot_count / queue_depth，但它的使用点只有
// 一个计算 Op（`afir.store` 取的是 `%scaled` 这个结果值，不是它），一个搬运边界都没有。
// 判据曾经还有「被搬运族读写就算队列」一条，那时这块存储要靠 slot_count 才勉强归对；
// 现在它是唯一判据，`q4_size` 与 `b4_size` 只可能出一个。
TEST_F(KernelBodySignatureTest, QueueDescriptorsWithoutTransferTraitsDoNotInventBufferFields) {
  std::string source = kBufferKernel.str();
  constexpr StringLiteral kStoreLoaded = "afir.store %loaded outs(%output)";
  const size_t store = source.find(kStoreLoaded);
  ASSERT_NE(store, std::string::npos);
  source.replace(
      store, kStoreLoaded.size(),
      R"mlir(%staged = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 4, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * s0)">, alignment_bytes = 32, life_time_axis_id = -1>} : memref<16xf16>
    %scaled = afir.add %loaded, %loaded outs(%staged) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>, #afir.tensor_desc<axis = [0], role = <compute>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : (memref<16xf16>, memref<16xf16>), memref<16xf16> -> memref<16xf16>
    afir.store %scaled outs(%output))mlir");
  module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(Run())) << diagnostic;
  EXPECT_TRUE(getTilingSchemaField(function.getArgument(3).getType(), "q4_size"));
  EXPECT_FALSE(getTilingSchemaField(function.getArgument(3).getType(), "b4_size"));
}

TEST_F(KernelBodySignatureTest, RejectsShapeAndResourceFieldNameCollision) {
  std::string source = kBufferKernel.str();
  for (size_t pos = 0; (pos = source.find("s0", pos)) != std::string::npos; pos += 7) {
    source.replace(pos, 2, "q2_size");
  }
  module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  ExpectFailureWithoutChanges("resource size field conflicts with a shape field");
}

TEST_F(KernelBodySignatureTest, RejectsCanonicalCountThatOmitsExternalOutputs) {
  ASSERT_TRUE(succeeded(Run()));
  OpBuilder builder(&context);
  function->setAttr("ascendc.num_inputs", builder.getI32IntegerAttr(1));
  ExpectFailureWithoutChanges("must include every external data argument");
}

// 身份查询只剩两种失败：下标越界，和参数压根没声明身份。此前还要挡「属性不是整数」
// 「整数位宽超过 64」「input_index 与 output_index 同时挂」——强类型化之后这三种形态
// 在解析期就进不来，属性 verifier 也不会放行（见 smoke/afir_kernel_arg_invalid.mlir）。
TEST_F(KernelBodySignatureTest, ArgumentIdentityRejectsOutOfBoundsAndUndeclaredArguments) {
  ScopedDiagnosticHandler handler(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(getKernelArgumentIdentity(function, 10U)));
  function.removeArgAttr(0U, mlir::afir::kKernelArgAttrName);
  EXPECT_TRUE(failed(getKernelArgumentIdentity(function, 0U)));
}

TEST_F(KernelBodySignatureTest, TensorFormWithDescriptorsRetainsOutputDestinationsAndOriginalInputCount) {
  module = parseSourceString<ModuleOp>(R"mlir(
      module {
        func.func private @tensor_kernel(%lhs: tensor<4xf16>, %rhs: tensor<4xf16>) -> tensor<4xf16>
          attributes {
            afir.function_role = #afir.function_role<kernel_body>,
            afir.arg_descs = [#afir.tensor_desc<role = <input>>, #afir.tensor_desc<role = <input>>],
            afir.result_descs = [#afir.tensor_desc<role = <output>>]
          } {
          %result = afir.add %lhs, %rhs : (tensor<4xf16>, tensor<4xf16>) -> tensor<4xf16>
          return %result : tensor<4xf16>
        }
      }
    )mlir",
                                       &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(Run())) << diagnostic;
  auto kernel = module->lookupSymbol<func::FuncOp>("tensor_kernel");
  ASSERT_EQ(kernel.getNumArguments(), 5U);
  EXPECT_EQ(kernel->getAttrOfType<IntegerAttr>("ascendc.num_inputs").getInt(), 2);
  // ABI 追加出来的两个参数自己声明身份，而不是留给消费端按位置与类型反推。
  auto workspace = mlir::afir::getKernelArgAttr(kernel, 3U);
  ASSERT_TRUE(workspace);
  EXPECT_EQ(workspace.getRole(), mlir::afir::KernelArgRole::Workspace);
  auto tiling = mlir::afir::getKernelArgAttr(kernel, 4U);
  ASSERT_TRUE(tiling);
  EXPECT_EQ(tiling.getRole(), mlir::afir::KernelArgRole::Tiling);
  EXPECT_EQ(kernel.getNumResults(), 0U);
  // 结果被搬成尾随的目的地参数之后，描述符也要跟着并进 arg_descs，
  // 否则数量校验会以为少写了描述符。
  auto argDescs = kernel->getAttrOfType<ArrayAttr>(mlir::afir::kArgDescsAttrName);
  ASSERT_TRUE(argDescs);
  EXPECT_EQ(argDescs.size(), 3U);
  EXPECT_FALSE(kernel->hasAttr(mlir::afir::kResultDescsAttrName));
  auto once = Print(*module);
  ASSERT_TRUE(succeeded(Run())) << diagnostic;
  EXPECT_EQ(Print(*module), once);
}

}  // namespace
