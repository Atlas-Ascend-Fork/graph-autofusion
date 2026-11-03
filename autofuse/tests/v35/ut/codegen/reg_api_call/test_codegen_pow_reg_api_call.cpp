/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "ascir/generator/v2_ascir_codegen_impl.h"
#include "codegen_kernel.h"
#include "common_utils.h"
#include "elewise/pow_api_call.h"
#include "optimize/pre_process/pre_process_config.h"
#include "utils/api_call_factory.h"

using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;

namespace {

// 构造 Load -> Pow(tensor底数, scalar指数) 的测试图，返回 Pow 节点
af::AscNodePtr BuildPowGraph(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  Data x_op("x", graph);
  Load load_op("load");
  Pow pow_op("pow");
  Scalar constant_op("constant");
  constant_op.ir_attr.SetValue("3.0");
  graph.AddNode(load_op);
  graph.AddNode(pow_op);
  graph.AddNode(constant_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id};
  *load_op.y.axis = {z0.id};
  *load_op.y.repeats = {s0};
  *load_op.y.strides = {One};
  pow_op.x1 = load_op.y;
  pow_op.x2 = constant_op.y;
  *pow_op.y.axis = {z0.id};
  *pow_op.y.repeats = {s0};
  *pow_op.y.strides = {One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto constant = graph.FindNode("constant");
  constant->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeInvalid;
  constant->outputs[0].attr.mem.tensor_id = 1;
  constant->outputs[0].attr.mem.position = af::Position::kPositionInvalid;
  constant->outputs[0].attr.dtype = DT_FLOAT;

  auto pow = graph.FindNode("pow");
  pow->attr.api.compute_type = af::ComputeType::kComputeElewise;
  pow->attr.api.type = af::ApiType::kAPITypeCompute;
  pow->attr.api.unit = af::ComputeUnit::kUnitVector;
  pow->attr.sched.loop_axis = z0.id;
  pow->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  pow->outputs[0].attr.vectorized_axis = {z0.id};
  pow->outputs[0].attr.vectorized_strides = {One};
  pow->outputs[0].attr.dtype = DT_FLOAT;
  pow->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  pow->outputs[0].attr.mem.tensor_id = 2;
  pow->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  pow->outputs[0].attr.que.id = 2;
  pow->outputs[0].attr.opt.merge_scope = af::kIdNone;
  return pow;
}

}  // namespace

// 默认走 Power 高精度实现；配置 Pow 或 all 精度提升黑名单后切换为 PowFast 快速幂路径，
// ApiCall、头文件与 tmp 依赖两条路径保持一致
TEST(CodegenKernel, TestPowBlacklistCodegenBasicInfo) {
  unsetenv("AUTOFUSE_FLAGS");
  af::pre_process::PreProcessConfig::Instance().Reset();

  af::AscGraph graph("test_pow_graph");
  auto pow = BuildPowGraph(graph);

  af::ascir::PowAscIrCodegenImplV2 codegen;
  EXPECT_EQ(codegen.GetApiCallName(), "PowApiCall");

  const auto headers = codegen.LoadApiHeaderFiles(false);
  ASSERT_EQ(headers.size(), 1U);
  EXPECT_EQ(headers[0], "pow_reg_base.h");

  EXPECT_FALSE(codegen.CalcTmpBufSize(*pow).empty());

  EXPECT_EQ(codegen.GetApiName(), "Pow");

  setenv("AUTOFUSE_FLAGS", "--autofuse_enhance_precision_blacklist=Pow", 1);
  af::pre_process::PreProcessConfig::Instance().Reset();
  EXPECT_EQ(codegen.GetApiName(), "PowFast");

  setenv("AUTOFUSE_FLAGS", "--autofuse_enhance_precision_blacklist=all", 1);
  af::pre_process::PreProcessConfig::Instance().Reset();
  EXPECT_EQ(codegen.GetApiName(), "PowFast");

  unsetenv("AUTOFUSE_FLAGS");
  af::pre_process::PreProcessConfig::Instance().Reset();
}

// 默认生成 Pow 调用文本；黑名单下 GetApiName 切换为 PowFast，参数列表保持一致
TEST(CodegenKernel, TestPowBlacklistGenerateCallText) {
  unsetenv("AUTOFUSE_FLAGS");
  af::pre_process::PreProcessConfig::Instance().Reset();

  af::AscGraph graph("test_pow_call_text_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  Data x_op("x", graph);
  Load load_op("load");
  Pow pow_op("pow");
  Scalar constant_op("constant");
  constant_op.ir_attr.SetValue("3.0");
  graph.AddNode(load_op);
  graph.AddNode(pow_op);
  graph.AddNode(constant_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id};
  *load_op.y.axis = {z0.id};
  *load_op.y.repeats = {s0};
  *load_op.y.strides = {One};
  pow_op.x1 = load_op.y;
  pow_op.x2 = constant_op.y;
  *pow_op.y.axis = {z0.id};
  *pow_op.y.repeats = {s0};
  *pow_op.y.strides = {One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto constant = graph.FindNode("constant");
  constant->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeInvalid;
  constant->outputs[0].attr.mem.tensor_id = 1;
  constant->outputs[0].attr.mem.position = af::Position::kPositionInvalid;
  constant->outputs[0].attr.dtype = DT_FLOAT;

  auto pow = graph.FindNode("pow");
  pow->attr.api.compute_type = af::ComputeType::kComputeElewise;
  pow->attr.api.type = af::ApiType::kAPITypeCompute;
  pow->attr.api.unit = af::ComputeUnit::kUnitVector;
  pow->attr.sched.loop_axis = z0.id;
  pow->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  pow->outputs[0].attr.vectorized_axis = {z0.id};
  pow->outputs[0].attr.vectorized_strides = {One};
  pow->outputs[0].attr.dtype = DT_FLOAT;
  pow->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  pow->outputs[0].attr.mem.tensor_id = 2;
  pow->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  pow->outputs[0].attr.que.id = 2;
  pow->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  EXPECT_EQ(tpipe.AddTensor(load->outputs[0]), 0);
  EXPECT_EQ(tpipe.AddTensor(pow->outputs[0]), 0);
  tpipe.AddTensor("3.0", constant->outputs[0], "const_y");

  tiler.AddAxis(z0);
  tiler.AddSizeVar(af::SizeVar(s0));

  codegen::ApiTensor x1, x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = constant->outputs[0].attr.mem.tensor_id;

  af::ascir::PowAscIrCodegenImplV2 codegen;
  EXPECT_EQ(codegen.GetApiName(), "Pow");

  codegen::PowApiCall default_call(codegen.GetApiName());
  EXPECT_EQ(default_call.Init(pow), 0);
  default_call.inputs.push_back(&x1);
  default_call.inputs.push_back(&x2);
  std::string default_result;
  EXPECT_EQ(default_call.Generate(tpipe, std::vector<af::AxisId>{}, default_result), 0);
  EXPECT_EQ(default_result, std::string{"Pow(local_2[0], local_0[0], scalar_1, tmp_buf_0, local_0_actual_size);\n"});
  EXPECT_EQ(default_result.find("PowFast"), std::string::npos);

  setenv("AUTOFUSE_FLAGS", "--autofuse_enhance_precision_blacklist=Pow", 1);
  af::pre_process::PreProcessConfig::Instance().Reset();
  EXPECT_EQ(codegen.GetApiName(), "PowFast");

  codegen::PowApiCall blacklist_call(codegen.GetApiName());
  EXPECT_EQ(blacklist_call.Init(pow), 0);
  blacklist_call.inputs.push_back(&x1);
  blacklist_call.inputs.push_back(&x2);
  std::string blacklist_result;
  EXPECT_EQ(blacklist_call.Generate(tpipe, std::vector<af::AxisId>{}, blacklist_result), 0);
  EXPECT_EQ(blacklist_result,
            std::string{"PowFast(local_2[0], local_0[0], scalar_1, tmp_buf_0, local_0_actual_size);\n"});

  unsetenv("AUTOFUSE_FLAGS");
  af::pre_process::PreProcessConfig::Instance().Reset();
}
