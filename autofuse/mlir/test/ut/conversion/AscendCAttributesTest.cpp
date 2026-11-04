/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/AFIRToAscendC/AscendCAttributes.h"
#include "conversion/AFIRToAscendC/AscendCKernelABI.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "target/Ascend/AscendTargetProfile.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include <optional>

#include "gtest/gtest.h"

namespace mlir::afir {
using namespace mlir::ascend::kernel;
namespace {

TEST(AscendCAttributesTest, MapsEveryAFIRSpaceOntoTheTargetVocabulary) {
  // afir::Space and ascend::MemorySpace describe the same physical spaces but
  // live in layers that must not depend on each other. Pin the translation so
  // adding an enumerator on either side cannot silently diverge.
  EXPECT_EQ(*toTargetMemorySpace(afir::Space::GM), ascend::MemorySpace::GM);
  EXPECT_EQ(*toTargetMemorySpace(afir::Space::UB), ascend::MemorySpace::UB);
  EXPECT_EQ(*toTargetMemorySpace(afir::Space::L1), ascend::MemorySpace::L1);
  EXPECT_EQ(*toTargetMemorySpace(afir::Space::L2), ascend::MemorySpace::L2);
  EXPECT_EQ(*toTargetMemorySpace(afir::Space::L0A), ascend::MemorySpace::L0A);
  EXPECT_EQ(*toTargetMemorySpace(afir::Space::L0B), ascend::MemorySpace::L0B);
  EXPECT_EQ(*toTargetMemorySpace(afir::Space::L0C), ascend::MemorySpace::L0C);
}

TEST(AscendCAttributesTest, ReadsMemorySpaceFromTheValueDefinition) {
  // 空间是**值**的事实，问的是值本身，不再是「某个 Op 的某个数组的第 N 项」
  // （设计文档 D3 / 5.1 步 8）。值的定义点有两种形态：kernel 参数在 arg attr 上，
  // 局部值在定义 Op 上。
  MLIRContext context;
  context.loadDialect<func::FuncDialect, afir::AFIRDialect, memref::MemRefDialect>();
  OpBuilder builder(&context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(builder.getUnknownLoc());
  auto memrefType = MemRefType::get({4}, builder.getF16Type());
  auto func =
      func::FuncOp::create(builder.getUnknownLoc(), "kernel", builder.getFunctionType({memrefType, memrefType}, {}));
  module->push_back(func);
  Block &entry = *func.addEntryBlock();

  auto makeBufferDesc = [&](afir::Space space) -> afir::BufferDescAttr {
    return afir::BufferDescAttr::get(&context, afir::SpaceAttr::get(&context, space),
                                     /*buffer_id=*/0, /*reuse_id=*/-1, /*slot_count=*/-1,
                                     /*queue_depth=*/-1, /*size_bytes_expr=*/afir::ExprAttr{},
                                     /*size_from_resource=*/false, /*alignment_bytes=*/-1,
                                     /*life_time_axis_id=*/-1);
  };

  // 没有值，或者定义点上根本没有 descriptor。
  EXPECT_TRUE(failed(getValueMemorySpace(Value{})));
  EXPECT_TRUE(failed(getValueMemorySpace(entry.getArgument(0))));

  func.setArgAttr(0, afir::kBufferDescAttrName, makeBufferDesc(afir::Space::GM));
  func.setArgAttr(1, afir::kBufferDescAttrName, makeBufferDesc(afir::Space::L1));

  auto source = getValueMemorySpace(entry.getArgument(0));
  ASSERT_TRUE(succeeded(source));
  EXPECT_EQ(*source, ascend::MemorySpace::GM);

  // LocalTensor 的落点不必然是 UB：L1 要如实报成 L1，能力查询问的才是这条搬运边。
  auto destination = getValueMemorySpace(entry.getArgument(1));
  ASSERT_TRUE(succeeded(destination));
  EXPECT_EQ(*destination, ascend::MemorySpace::L1);

  // 局部值的定义点是定义它的 Op。
  builder.setInsertionPointToStart(&entry);
  auto alloc = builder.create<memref::AllocOp>(builder.getUnknownLoc(), memrefType);
  EXPECT_TRUE(failed(getValueMemorySpace(alloc.getResult())));
  afir::setBufferDescAttr(alloc.getOperation(), makeBufferDesc(afir::Space::UB));
  auto local = getValueMemorySpace(alloc.getResult());
  ASSERT_TRUE(succeeded(local));
  EXPECT_EQ(*local, ascend::MemorySpace::UB);

  // 类型不对的 descriptor 不能被当成 descriptor 读出来。
  func.setArgAttr(0, afir::kBufferDescAttrName, builder.getUnitAttr());
  EXPECT_TRUE(failed(getValueMemorySpace(entry.getArgument(0))));
}

// `memref` 的空间由类型自带时，它是一手事实，优先于定义点的 descriptor。这与
// `CanonicalBindings.cpp` 的 `resolveValueSpace` 是同一条判据——两份判据此前不一样，
// 同一个值在规划侧与取用侧可能得到不同的空间，而没有任何断言拦得住。
TEST(AscendCAttributesTest, PrefersTheMemorySpaceCarriedByTheType) {
  MLIRContext context;
  context.loadDialect<func::FuncDialect, afir::AFIRDialect, memref::MemRefDialect>();
  OpBuilder builder(&context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(builder.getUnknownLoc());
  auto spaced = MemRefType::get({4}, builder.getF16Type(), MemRefLayoutAttrInterface{},
                                afir::SpaceAttr::get(&context, afir::Space::UB));
  auto func = func::FuncOp::create(builder.getUnknownLoc(), "kernel", builder.getFunctionType({spaced}, {}));
  module->push_back(func);
  Block &entry = *func.addEntryBlock();

  // descriptor 一个都没有，光凭类型就能答。
  auto fromType = getValueMemorySpace(entry.getArgument(0));
  ASSERT_TRUE(succeeded(fromType));
  EXPECT_EQ(*fromType, ascend::MemorySpace::UB);

  // descriptor 说的是别的空间时，以类型为准，不以先到先得为准。
  func.setArgAttr(0, afir::kBufferDescAttrName,
                  afir::BufferDescAttr::get(&context, afir::SpaceAttr::get(&context, afir::Space::GM),
                                            /*buffer_id=*/0, /*reuse_id=*/-1, /*slot_count=*/-1,
                                            /*queue_depth=*/-1, /*size_bytes_expr=*/afir::ExprAttr{},
                                            /*size_from_resource=*/false, /*alignment_bytes=*/-1,
                                            /*life_time_axis_id=*/-1));
  auto stillType = getValueMemorySpace(entry.getArgument(0));
  ASSERT_TRUE(succeeded(stillType));
  EXPECT_EQ(*stillType, ascend::MemorySpace::UB);
}

TEST(AscendCAttributesTest, ReadsAndWritesExecutionSpecs) {
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  OpBuilder builder(&context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(builder.getUnknownLoc());
  auto func = func::FuncOp::create(builder.getUnknownLoc(), "kernel", builder.getFunctionType({}, {}));
  module->push_back(func);

  const ascend::ExecutionSpec simdMemoryBased{ascend::ProcessorType::AICore, ascend::AICoreUnit::Vector,
                                              ascend::ExecutionModel::SIMD, ascend::InstructionForm::MemoryBased};
  const ascend::ExecutionSpec simdRegBased{ascend::ProcessorType::AICore, ascend::AICoreUnit::Vector,
                                           ascend::ExecutionModel::SIMD, ascend::InstructionForm::RegBased};
  const ascend::ExecutionSpec simt{ascend::ProcessorType::AICore, ascend::AICoreUnit::Vector,
                                   ascend::ExecutionModel::SIMT, std::nullopt};

  EXPECT_TRUE(failed(getExecutionSpec(nullptr)));
  EXPECT_TRUE(failed(setExecutionSpec(nullptr, simdMemoryBased)));
  EXPECT_TRUE(failed(getExecutionSpec(func)));

  for (const ascend::ExecutionSpec &expected : {simdMemoryBased, simdRegBased, simt}) {
    ASSERT_TRUE(succeeded(setExecutionSpec(func, expected)));
    FailureOr<ascend::ExecutionSpec> actual = getExecutionSpec(func);
    ASSERT_TRUE(succeeded(actual));
    EXPECT_EQ(actual->processor, expected.processor);
    EXPECT_EQ(actual->unit, expected.unit);
    EXPECT_EQ(actual->executionModel, expected.executionModel);
    EXPECT_EQ(actual->instructionForm, expected.instructionForm);
  }
  EXPECT_FALSE(func->hasAttr("ascendc.instruction_form"));
}

TEST(AscendCAttributesTest, RejectsMalformedExecutionSpecsWithoutPartialWrites) {
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  OpBuilder builder(&context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(builder.getUnknownLoc());
  auto func = func::FuncOp::create(builder.getUnknownLoc(), "kernel", builder.getFunctionType({}, {}));
  module->push_back(func);

  const ascend::ExecutionSpec valid{ascend::ProcessorType::AICore, ascend::AICoreUnit::Vector,
                                    ascend::ExecutionModel::SIMD, ascend::InstructionForm::MemoryBased};
  ASSERT_TRUE(succeeded(setExecutionSpec(func, valid)));

  const ascend::ExecutionSpec unsupportedUnit{ascend::ProcessorType::AICore, ascend::AICoreUnit::Cube, std::nullopt,
                                              std::nullopt};
  EXPECT_TRUE(failed(setExecutionSpec(func, unsupportedUnit)));
  EXPECT_EQ(func->getAttrOfType<StringAttr>("ascendc.unit").getValue(), "AiCore.Vector");
  EXPECT_EQ(func->getAttrOfType<StringAttr>("ascendc.execution_model").getValue(), "SIMD");
  EXPECT_EQ(func->getAttrOfType<StringAttr>("ascendc.instruction_form").getValue(), "MemoryBased");

  const ascend::ExecutionSpec simdWithoutForm{ascend::ProcessorType::AICore, ascend::AICoreUnit::Vector,
                                              ascend::ExecutionModel::SIMD, std::nullopt};
  const ascend::ExecutionSpec simtWithForm{ascend::ProcessorType::AICore, ascend::AICoreUnit::Vector,
                                           ascend::ExecutionModel::SIMT, ascend::InstructionForm::RegBased};
  EXPECT_TRUE(failed(setExecutionSpec(func, simdWithoutForm)));
  EXPECT_TRUE(failed(setExecutionSpec(func, simtWithForm)));
  EXPECT_EQ(func->getAttrOfType<StringAttr>("ascendc.execution_model").getValue(), "SIMD");
  EXPECT_EQ(func->getAttrOfType<StringAttr>("ascendc.instruction_form").getValue(), "MemoryBased");

  func->setAttr("ascendc.unit", builder.getStringAttr("unknown"));
  EXPECT_TRUE(failed(getExecutionSpec(func)));
  func->setAttr("ascendc.unit", builder.getStringAttr("AiCore.Vector"));
  func->setAttr("ascendc.execution_model", builder.getStringAttr("unknown"));
  EXPECT_TRUE(failed(getExecutionSpec(func)));
  func->setAttr("ascendc.execution_model", builder.getStringAttr("SIMD"));
  func->setAttr("ascendc.instruction_form", builder.getStringAttr("unknown"));
  EXPECT_TRUE(failed(getExecutionSpec(func)));
  func->removeAttr("ascendc.instruction_form");
  EXPECT_TRUE(failed(getExecutionSpec(func)));
  func->setAttr("ascendc.execution_model", builder.getStringAttr("SIMT"));
  func->setAttr("ascendc.instruction_form", builder.getStringAttr("RegBased"));
  EXPECT_TRUE(failed(getExecutionSpec(func)));
}

TEST(AscendCAttributesTest, ReadsAndWritesKernelAttributes) {
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  OpBuilder builder(&context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(builder.getUnknownLoc());
  auto func = func::FuncOp::create(builder.getUnknownLoc(), "kernel", builder.getFunctionType({}, {}));
  module->push_back(func);

  EXPECT_TRUE(failed(hasAscendCAiCoreAttr(nullptr)));
  EXPECT_TRUE(failed(getAscendCNumInputs(nullptr)));
  EXPECT_TRUE(failed(hasAscendCGlobalAttr(nullptr)));
  FailureOr<bool> aiCoreAttrState = hasAscendCAiCoreAttr(func);
  FailureOr<IntegerAttr> inputCountAttr = getAscendCNumInputs(func);
  FailureOr<bool> globalAttrState = hasAscendCGlobalAttr(func);
  ASSERT_TRUE(succeeded(aiCoreAttrState));
  ASSERT_TRUE(succeeded(inputCountAttr));
  ASSERT_TRUE(succeeded(globalAttrState));
  EXPECT_FALSE(*aiCoreAttrState);
  EXPECT_FALSE(*inputCountAttr);
  EXPECT_FALSE(*globalAttrState);

  setAscendCAiCore(func);
  setAscendCNumInputs(func, 3);
  setAscendCGlobal(func, true);

  aiCoreAttrState = hasAscendCAiCoreAttr(func);
  inputCountAttr = getAscendCNumInputs(func);
  globalAttrState = hasAscendCGlobalAttr(func);
  ASSERT_TRUE(succeeded(aiCoreAttrState));
  ASSERT_TRUE(succeeded(inputCountAttr));
  ASSERT_TRUE(succeeded(globalAttrState));
  EXPECT_TRUE(*aiCoreAttrState);
  ASSERT_TRUE(*inputCountAttr);
  EXPECT_EQ((*inputCountAttr).getInt(), 3);
  EXPECT_TRUE(*globalAttrState);

  func->setAttr("ascendc.aicore", builder.getStringAttr("invalid"));
  func->setAttr("ascendc.num_inputs", builder.getStringAttr("invalid"));
  func->setAttr("ascendc.global", builder.getStringAttr("invalid"));
  EXPECT_TRUE(failed(hasAscendCAiCoreAttr(func)));
  EXPECT_TRUE(failed(getAscendCNumInputs(func)));
  EXPECT_TRUE(failed(hasAscendCGlobalAttr(func)));

  setAscendCGlobal(func, false);
  globalAttrState = hasAscendCGlobalAttr(func);
  ASSERT_TRUE(succeeded(globalAttrState));
  EXPECT_FALSE(*globalAttrState);
}

TEST(AscendCAttributesTest, KernelABIRejectsPartialAndMalformedAttributes) {
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  ScopedDiagnosticHandler diagnosticHandler(&context, [](Diagnostic &) { return success(); });
  OpBuilder builder(&context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(builder.getUnknownLoc());
  auto func = func::FuncOp::create(builder.getUnknownLoc(), "kernel", builder.getFunctionType({}, {}));
  module->push_back(func);

  FailureOr<bool> noABI = checkAscendCKernelABI(func, FunctionRole::KernelEntry);
  ASSERT_TRUE(succeeded(noABI));
  EXPECT_FALSE(*noABI);

  setAscendCGlobal(func, true);
  EXPECT_TRUE(failed(checkAscendCKernelABI(func, FunctionRole::KernelEntry)));

  setAscendCGlobal(func, false);
  func->setAttr("ascendc.aicore", builder.getStringAttr("invalid"));
  EXPECT_TRUE(failed(checkAscendCKernelABI(func, FunctionRole::KernelEntry)));
}

// 后端 ABI 属性齐备、但参数一个都没声明身份，是「物化到一半」——此前这条由
// `afir.num_inputs` 在场与否回答，而那个属性只是 `ascendc.num_inputs` 的第二份编码。
// 现在判据是参数自己有没有 `#afir.kernel_arg`（设计文档 3.6 / D2）。
TEST(AscendCAttributesTest, KernelABIRejectsBackendAttributesWithoutArgumentRoles) {
  MLIRContext context;
  context.loadDialect<afir::AFIRDialect, emitasc::EmitAscDialect, func::FuncDialect>();
  ScopedDiagnosticHandler diagnosticHandler(&context, [](Diagnostic &) { return success(); });
  OpBuilder builder(&context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(builder.getUnknownLoc());
  RankedTensorType tensorType = RankedTensorType::get({4}, builder.getF16Type());
  Type inputType = getAscendCGlobalMemRefType(tensorType);
  Type workspaceType = getAscendCWorkspaceType(&context);
  auto func = func::FuncOp::create(builder.getUnknownLoc(), "kernel", builder.getFunctionType({}, {}));
  module->push_back(func);
  Type tilingType = getAscendCTilingDataType(func);
  func.setType(builder.getFunctionType({inputType, inputType, workspaceType, tilingType}, {}));
  setAscendCAiCore(func);
  setAscendCNumInputs(func, 2);

  EXPECT_TRUE(failed(checkAscendCKernelABI(func, FunctionRole::KernelBody)));

  // 补齐声明之后同一个函数就能通过：失败的确实是「缺声明」，不是别的什么。
  afir::setKernelArgAttr(func, 0U, afir::KernelArgRole::Input, 0);
  afir::setKernelArgAttr(func, 1U, afir::KernelArgRole::Output, 0);
  afir::setKernelArgAttr(func, 2U, afir::KernelArgRole::Workspace);
  afir::setKernelArgAttr(func, 3U, afir::KernelArgRole::Tiling);
  func.addEntryBlock();
  FailureOr<bool> complete = checkAscendCKernelABI(func, FunctionRole::KernelBody);
  ASSERT_TRUE(succeeded(complete));
  EXPECT_TRUE(*complete);
}

// AFIR 地址空间到 CCE-C 编码的单点映射（设计文档 3.1 / I4：AFIR 侧全程不出现魔数）。
// 生产路径目前只走 GM 一条臂——`getAscendCGlobalMemorySpaceAttr` 是唯一调用者——其余
// 五条以及 L2 的空返回值没有任何调用点覆盖。按 5.3 纪律，这里把整张映射表对着
// `ascir/Dialect/Asc/IR/Core/Attributes.td` 的枚举编码钉住，免得 Cube 路径接上来时
// 才发现某一条写错了：错的映射不会报错，只会把张量写到别的存储上。
TEST(AscendCKernelABITest, MapsEveryAfirSpaceToItsCceCEncoding) {
  MLIRContext context;
  auto encoding = [&](Space space) -> std::optional<int64_t> {
    auto attr = dyn_cast_or_null<IntegerAttr>(getAscendCMemorySpaceAttr(&context, space));
    if (!attr) {
      return std::nullopt;
    }
    EXPECT_TRUE(attr.getType().isSignlessInteger(32));
    return attr.getInt();
  };

  EXPECT_EQ(encoding(Space::GM), 22);
  EXPECT_EQ(encoding(Space::L0A), 23);
  EXPECT_EQ(encoding(Space::L0B), 24);
  EXPECT_EQ(encoding(Space::L0C), 25);
  EXPECT_EQ(encoding(Space::UB), 26);
  EXPECT_EQ(encoding(Space::L1), 27);
  // L2 是 cache，不是 CCE-C 的地址空间：没有可映射的编码，返回空属性。调用方必须把
  // 空返回值当作「这个 space 不能物化到 kernel ABI 上」，而不是当作 Default(0)。
  EXPECT_EQ(encoding(Space::L2), std::nullopt);

  // GM 的便捷包装与走完整映射得到的是同一个属性。
  EXPECT_EQ(getAscendCGlobalMemorySpaceAttr(&context), getAscendCMemorySpaceAttr(&context, Space::GM));
}

}  // namespace
}  // namespace mlir::afir
