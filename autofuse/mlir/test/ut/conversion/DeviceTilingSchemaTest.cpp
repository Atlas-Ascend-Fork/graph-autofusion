/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/AFIRToAscendC/TilingSchema.h"
#include "conversion/AFIRToAscendC/AscendCKernelABI.h"
#include "artifact/KernelSource/Emit/TilingDataHeaderEmitter.h"

#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "dialect/AFIR/IR/AFIRAttributes.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::ascend::kernel;

namespace {

class TilingSchemaTest : public testing::Test {
 protected:
  MLIRContext context;
  OpBuilder builder{&context};

  void SetUp() override {
    context.getOrLoadDialect<emitasc::EmitAscDialect>();
  }
};

TEST_F(TilingSchemaTest, PreservesFieldOrderWidthsAndNestedTypes) {
  auto names = builder.getArrayAttr({builder.getStringAttr("second"), builder.getStringAttr("first")});
  auto types = builder.getArrayAttr({TypeAttr::get(builder.getI64Type()), TypeAttr::get(builder.getI16Type())});
  auto stage = TilingSchema::get(&context, builder.getStringAttr("Stage"), types, names);
  ASSERT_TRUE(stage);
  EXPECT_EQ(stage.getNamesAttr(), names);
  EXPECT_EQ(stage.getTypesAttr(), types);
  EXPECT_EQ(getTilingSchemaField(stage, "second"), builder.getI64Type());
  EXPECT_EQ(getTilingSchemaField(stage, "first"), builder.getI16Type());
  EXPECT_FALSE(getTilingSchemaField(stage, "missing"));

  auto root = TilingSchema::get(&context, builder.getStringAttr("Root"), builder.getArrayAttr({TypeAttr::get(stage)}),
                                builder.getArrayAttr({builder.getStringAttr("stage")}));
  ASSERT_TRUE(root);
  EXPECT_EQ(getTilingSchemaField(root, "stage"), stage.getType());
  EXPECT_EQ(getTilingSchema(root).getType(), root.getType());

  std::string text;
  llvm::raw_string_ostream stream(text);
  ASSERT_TRUE(succeeded(emitTilingDataHeader({root}, stream)));
  EXPECT_LT(text.find("uint64_t second;"), text.find("uint16_t first;"));
  EXPECT_NE(text.find("Stage stage;"), std::string::npos);
}

TEST_F(TilingSchemaTest, RejectsMalformedArraysAndDuplicateFields) {
  auto name = builder.getStringAttr("Stage");
  auto field = builder.getStringAttr("value");
  auto type = TypeAttr::get(builder.getI32Type());
  auto empty = builder.getArrayAttr(ArrayRef<Attribute>{});
  EXPECT_FALSE(TilingSchema::get(&context, name, builder.getArrayAttr({type}), empty));
  EXPECT_FALSE(
      TilingSchema::get(&context, name, builder.getArrayAttr({type, type}), builder.getArrayAttr({field, field})));
  EXPECT_FALSE(TilingSchema::get(&context, name, builder.getArrayAttr({field}), builder.getArrayAttr({field})));
  EXPECT_FALSE(TilingSchema::get(&context, name, builder.getArrayAttr({type}), builder.getArrayAttr({type})));
  EXPECT_FALSE(TilingSchema::get(&context, name, builder.getArrayAttr({type}),
                                 builder.getArrayAttr({builder.getStringAttr("")})));
  EXPECT_FALSE(TilingSchema::get(&context, builder.getStringAttr(""), empty, empty));
  EXPECT_TRUE(TilingSchema::get(&context, name, empty, empty));
}

TEST_F(TilingSchemaTest, RejectsMalformedExistingAndNestedSchemas) {
  auto malformed = emitasc::PyStructType::get(&context, builder.getStringAttr("Bad"),
                                              builder.getArrayAttr({TypeAttr::get(builder.getI32Type())}),
                                              builder.getArrayAttr(ArrayRef<Attribute>{}));
  EXPECT_FALSE(getTilingSchema(malformed));
  EXPECT_FALSE(getTilingSchemaField(malformed, "value"));
  auto rootTypes = builder.getArrayAttr({TypeAttr::get(malformed)});
  auto rootNames = builder.getArrayAttr({builder.getStringAttr("nested")});
  EXPECT_FALSE(TilingSchema::get(&context, builder.getStringAttr("Root"), rootTypes, rootNames));
  auto malformedRoot = emitasc::PyStructType::get(&context, builder.getStringAttr("Root"), rootTypes, rootNames);
  EXPECT_FALSE(getTilingSchema(malformedRoot));
  EXPECT_FALSE(getTilingSchema(Type{}));
  EXPECT_FALSE(getTilingSchema(builder.getI32Type()));
  EXPECT_FALSE(getTilingSchemaField(builder.getI32Type(), "value"));
}

TEST_F(TilingSchemaTest, RejectsMixedContexts) {
  MLIRContext otherContext;
  auto foreignName = StringAttr::get(&otherContext, "Foreign");
  auto empty = builder.getArrayAttr(ArrayRef<Attribute>{});
  EXPECT_FALSE(TilingSchema::get(&context, foreignName, empty, empty));
}

TEST_F(TilingSchemaTest, ApiFieldLookupRequiresCanonicalArgumentAndExistingField) {
  context.getOrLoadDialect<afir::AFIRDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  auto schema = TilingSchema::get(&context, builder.getStringAttr("Stage"),
                                  builder.getArrayAttr({TypeAttr::get(builder.getI32Type())}),
                                  builder.getArrayAttr({builder.getStringAttr("api_field")}));
  auto input = getAscendCGlobalMemRefType(RankedTensorType::get({4}, builder.getF32Type()));
  auto workspace = getAscendCWorkspaceType(&context);
  auto func = func::FuncOp::create(builder.getUnknownLoc(), "kernel",
                                   builder.getFunctionType({input, input, workspace, schema.getType()}, {}));
  auto module = ModuleOp::create(builder.getUnknownLoc());
  module.push_back(func);
  builder.setInsertionPointToStart(func.addEntryBlock());
  auto user = builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), 1);
  EXPECT_FALSE(getApiTilingFieldType(user, "api_field"));
  afir::setKernelArgAttr(func, 0U, afir::KernelArgRole::Input, 0);
  afir::setKernelArgAttr(func, 1U, afir::KernelArgRole::Output, 0);
  afir::setKernelArgAttr(func, 2U, afir::KernelArgRole::Workspace);
  afir::setKernelArgAttr(func, 3U, afir::KernelArgRole::Tiling);
  EXPECT_EQ(getApiTilingFieldType(user, "api_field"), builder.getI32Type());
  EXPECT_FALSE(getApiTilingFieldType(user, "missing"));
}

TEST_F(TilingSchemaTest, EmitterRejectsMissingSchemaAndUnsupportedScalar) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  EXPECT_TRUE(failed(emitTilingDataHeader({TilingSchema{}}, stream)));
  auto schema = TilingSchema::get(&context, builder.getStringAttr("Stage"),
                                  builder.getArrayAttr({TypeAttr::get(builder.getF32Type())}),
                                  builder.getArrayAttr({builder.getStringAttr("value")}));
  ASSERT_TRUE(schema);
  EXPECT_TRUE(failed(emitTilingDataHeader({schema}, stream)));
}

TEST_F(TilingSchemaTest, ExternalApiTilingTypesAreIncludedNotDefined) {
  // ATT 给 api-tiling（Pad / ConfusionTranspose）发的字段类型，C++ 定义在官方
  // kernel_tiling.h 里；生成头只引用（成员声明）并把 include 打在文件头，
  // 自己再定义一份就是与官方头的重定义冲突。
  auto apiTiling =
      TilingSchema::get(&context, builder.getStringAttr("PadTiling"), builder.getArrayAttr(ArrayRef<Attribute>{}),
                        builder.getArrayAttr(ArrayRef<Attribute>{}));
  ASSERT_TRUE(apiTiling);
  auto stage =
      TilingSchema::get(&context, builder.getStringAttr("Stage"), builder.getArrayAttr({TypeAttr::get(apiTiling)}),
                        builder.getArrayAttr({builder.getStringAttr("pad_tiling_0")}));
  ASSERT_TRUE(stage);
  auto root = TilingSchema::get(&context, builder.getStringAttr("Root"), builder.getArrayAttr({TypeAttr::get(stage)}),
                                builder.getArrayAttr({builder.getStringAttr("stage_0")}));
  ASSERT_TRUE(root);

  std::string text;
  llvm::raw_string_ostream stream(text);
  ASSERT_TRUE(succeeded(emitTilingDataHeader({root}, stream)));
  EXPECT_EQ(text.find("struct PadTiling"), std::string::npos);
  EXPECT_NE(text.find("PadTiling pad_tiling_0;"), std::string::npos);
  const size_t includeAt = text.find("#include \"kernel_tiling/kernel_tiling.h\"");
  EXPECT_NE(includeAt, std::string::npos);
  EXPECT_LT(includeAt, text.find("struct Stage"));
  EXPECT_GT(includeAt, text.find("#include <cstdint>"));
}

TEST_F(TilingSchemaTest, SingleStageWrapperForwardsScalarAccessors) {
  auto stage = TilingSchema::get(&context, builder.getStringAttr("Stage"),
                                 builder.getArrayAttr({TypeAttr::get(builder.getI32Type())}),
                                 builder.getArrayAttr({builder.getStringAttr("block_dim")}));
  auto wrapper = TilingSchema::get(&context, builder.getStringAttr("Group"),
                                   builder.getArrayAttr({TypeAttr::get(stage.getType())}),
                                   builder.getArrayAttr({builder.getStringAttr("stage_data")}));
  std::string text;
  llvm::raw_string_ostream stream(text);
  ASSERT_TRUE(succeeded(emitTilingDataHeader({wrapper}, stream)));
  EXPECT_NE(text.find("stage_data.set_block_dim(value)"), std::string::npos);
  EXPECT_NE(text.find("return stage_data.get_block_dim()"), std::string::npos);
  EXPECT_NE(text.find("return stage_data.get_addr_block_dim()"), std::string::npos);
}

}  // namespace
