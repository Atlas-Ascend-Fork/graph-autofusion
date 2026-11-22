/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "node_utils_ex.h"
#include "graph_utils.h"
#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "common_utils.h"
#include "utils/api_call_factory.h"
#include "../common.h"
#include "../../../../common/att/soc_perf_test_config.h"
#include "codegen.h"
#include "optimize.h"

using namespace af::ops;
using namespace codegen;
using namespace af::ascir_op;
using namespace testing;
using namespace codegen;

namespace af {

static void CreateMatmulElewiseBrcGraph(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(32);
  auto s1 = graph.CreateSizeVar(32);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = af::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0.y.repeats = {s0, s1};
  *data0.y.strides = {s1, af::ops::One};
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.y.dtype = af::DT_FLOAT16;
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, af::ops::One};

  af::ascir_op::Data data1("data1", graph);
  data1.attr.sched.axis = {z0.id, z1.id};
  data1.y.dtype = af::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data1.y.repeats = {s0, s1};
  *data1.y.strides = {s1, af::ops::One};
  data1.ir_attr.SetIndex(1);

  af::ascir_op::Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = af::DT_FLOAT16;
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, af::ops::One};

  af::ascir_op::MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = af::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, af::ops::One};
  matmul.ir_attr.SetTranspose_x1(1);
  matmul.ir_attr.SetTranspose_x2(0);
  matmul.ir_attr.SetHas_relu(0);
  matmul.ir_attr.SetEnable_hf32(0);
  matmul.ir_attr.SetOffset_x(0);

  af::ascir_op::Data data2("data2", graph);
  data2.y.dtype = af::DT_FLOAT;
  data2.attr.sched.axis = {z0.id, z1.id};
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data2.y.repeats = {af::ops::One, af::ops::One};
  *data2.y.strides = {af::ops::Zero, af::ops::Zero};
  data2.ir_attr.SetIndex(2);

  af::ascir_op::Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.y.dtype = af::DT_FLOAT;
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.repeats = {af::ops::One, af::ops::One};
  *load2.y.strides = {af::ops::Zero, af::ops::Zero};

  af::ascir_op::Broadcast broadcast0("broadcast0");
  broadcast0.x = load2.y;
  broadcast0.attr.sched.axis = {z0.id, z1.id};
  *broadcast0.y.axis = {z0.id, z1.id};
  broadcast0.y.dtype = af::DT_FLOAT;
  *broadcast0.y.repeats = {af::ops::One, s1};
  *broadcast0.y.strides = {af::ops::Zero, af::ops::One};

  af::ascir_op::Broadcast broadcast1("broadcast1");
  broadcast1.x = broadcast0.y;
  broadcast1.attr.sched.axis = {z0.id, z1.id};
  *broadcast1.y.axis = {z0.id, z1.id};
  broadcast1.y.dtype = af::DT_FLOAT;
  *broadcast1.y.repeats = {s0, s1};
  *broadcast1.y.strides = {s1, af::ops::One};

  af::ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id};
  add_op.x1 = matmul.y;
  add_op.x2 = broadcast1.y;
  add_op.y.dtype = af::DT_FLOAT;
  *add_op.y.axis = {z0.id, z1.id};
  *add_op.y.repeats = {s0, s1};
  *add_op.y.strides = {s1, af::ops::One};

  af::ascir_op::Data data3("data3", graph);
  data3.y.dtype = af::DT_FLOAT;
  data3.attr.sched.axis = {z0.id, z1.id};
  *data3.y.axis = {z0.id, z1.id};
  data3.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data3.y.repeats = {s0, af::ops::One};
  *data3.y.strides = {af::ops::One, af::ops::Zero};
  data3.ir_attr.SetIndex(3);

  af::ascir_op::Load load3("load3");
  load3.x = data3.y;
  load3.attr.sched.axis = {z0.id, z1.id};
  load3.y.dtype = af::DT_FLOAT;
  *load3.y.axis = {z0.id, z1.id};
  *load3.y.repeats = {s0, af::ops::One};
  *load3.y.strides = {af::ops::One, af::ops::Zero};

  af::ascir_op::Broadcast broadcast2("broadcast2");
  broadcast2.x = load3.y;
  broadcast2.attr.sched.axis = {z0.id, z1.id};
  *broadcast2.y.axis = {z0.id, z1.id};
  broadcast2.y.dtype = af::DT_FLOAT;
  *broadcast2.y.repeats = {s0, s1};
  *broadcast2.y.strides = {s1, af::ops::One};

  af::ascir_op::Sigmoid sigmoid0("sigmoid0");
  sigmoid0.x = broadcast2.y;
  sigmoid0.attr.sched.axis = {z0.id, z1.id};
  *sigmoid0.y.axis = {z0.id, z1.id};
  sigmoid0.y.dtype = af::DT_FLOAT;
  *sigmoid0.y.repeats = {s0, s1};
  *sigmoid0.y.strides = {s1, af::ops::One};

  af::ascir_op::Mul mul("mul");
  mul.attr.sched.axis = {z0.id, z1.id};
  mul.x1 = add_op.y;
  mul.x2 = sigmoid0.y;
  mul.y.dtype = af::DT_FLOAT;
  *mul.y.axis = {z0.id, z1.id};
  *mul.y.repeats = {s0, s1};
  *mul.y.strides = {s1, af::ops::One};

  af::ascir_op::Data data4("data4", graph);
  data4.y.dtype = af::DT_FLOAT;
  data4.attr.sched.axis = {z0.id, z1.id};
  *data4.y.axis = {z0.id, z1.id};
  data4.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data4.y.repeats = {af::ops::One, s1};
  *data4.y.strides = {af::ops::Zero, af::ops::One};
  data4.ir_attr.SetIndex(4);

  af::ascir_op::Load load4("load4");
  load4.x = data4.y;
  load4.attr.sched.axis = {z0.id, z1.id};
  load4.y.dtype = af::DT_FLOAT;
  *load4.y.axis = {z0.id, z1.id};
  *load4.y.repeats = {af::ops::One, s1};
  *load4.y.strides = {af::ops::Zero, af::ops::One};

  af::ascir_op::Broadcast broadcast3("broadcast3");
  broadcast3.x = load4.y;
  broadcast3.attr.sched.axis = {z0.id, z1.id};
  *broadcast3.y.axis = {z0.id, z1.id};
  broadcast3.y.dtype = af::DT_FLOAT;
  *broadcast3.y.repeats = {s0, s1};
  *broadcast3.y.strides = {s1, af::ops::One};

  af::ascir_op::Rsqrt rsqrt0("rsqrt0");
  rsqrt0.x = broadcast3.y;
  rsqrt0.attr.sched.axis = {z0.id, z1.id};
  *rsqrt0.y.axis = {z0.id, z1.id};
  rsqrt0.y.dtype = af::DT_FLOAT;
  *rsqrt0.y.repeats = {s0, s1};
  *rsqrt0.y.strides = {s1, af::ops::One};

  af::ascir_op::Sub sub("sub");
  sub.attr.sched.axis = {z0.id, z1.id};
  sub.x1 = mul.y;
  sub.x2 = rsqrt0.y;
  sub.y.dtype = af::DT_FLOAT;
  *sub.y.axis = {z0.id, z1.id};
  *sub.y.repeats = {s0, s1};
  *sub.y.strides = {s1, af::ops::One};

  af::ascir_op::Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = sub.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = af::DT_FLOAT;
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1, af::ops::One};

  af::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = af::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

static af::ComputeGraphPtr LoadMatmulElewiseBrcFusedGraph() {
  auto builder = GraphBuilder("load_matmul_elewise_brc_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  af::AttrUtils::SetInt(data2->GetOpDescBarePtr(), "_parent_node_index", 2);
  auto data3 = builder.AddNode("data3", "Data", 0, 1);
  af::AttrUtils::SetInt(data3->GetOpDescBarePtr(), "_parent_node_index", 3);
  auto data4 = builder.AddNode("data4", "Data", 0, 1);
  af::AttrUtils::SetInt(data4->GetOpDescBarePtr(), "_parent_node_index", 4);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 5, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(data2, 0, ascbc, 2);
  builder.AddDataEdge(data3, 0, ascbc, 3);
  builder.AddDataEdge(data4, 0, ascbc, 4);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_matmul_elewise_brc_store");
  CreateMatmulElewiseBrcGraph(sub_graph);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateMatmulCompareScalarGraph(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(32);
  auto s1 = graph.CreateSizeVar(32);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = af::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0.y.repeats = {s0, s1};
  *data0.y.strides = {s1, af::ops::One};
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.y.dtype = af::DT_FLOAT16;
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, af::ops::One};

  af::ascir_op::Data data1("data1", graph);
  data1.attr.sched.axis = {z0.id, z1.id};
  data1.y.dtype = af::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data1.y.repeats = {s0, s1};
  *data1.y.strides = {s1, af::ops::One};
  data1.ir_attr.SetIndex(1);

  af::ascir_op::Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = af::DT_FLOAT16;
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, af::ops::One};

  af::ascir_op::MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = af::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, af::ops::One};
  matmul.ir_attr.SetTranspose_x1(1);
  matmul.ir_attr.SetTranspose_x2(0);
  matmul.ir_attr.SetHas_relu(0);
  matmul.ir_attr.SetEnable_hf32(0);
  matmul.ir_attr.SetOffset_x(0);

  af::ascir_op::Data data2("data2", graph);
  data2.y.dtype = af::DT_FLOAT;
  data2.attr.sched.axis = {z0.id, z1.id};
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data2.y.repeats = {s0, s1};
  *data2.y.strides = {s1, af::ops::One};
  data2.ir_attr.SetIndex(2);

  af::ascir_op::Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.y.dtype = af::DT_FLOAT;
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.repeats = {s0, s1};
  *load2.y.strides = {s1, af::ops::One};

  af::ascir_op::Data data3("data3", graph);
  data3.y.dtype = af::DT_FLOAT;
  data3.attr.sched.axis = {z0.id, z1.id};
  *data3.y.axis = {z0.id, z1.id};
  data3.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data3.y.repeats = {af::ops::One, af::ops::One};
  *data3.y.strides = {af::ops::Zero, af::ops::Zero};
  data3.ir_attr.SetIndex(3);

  af::ascir_op::Load load3("load3");
  load3.x = data3.y;
  load3.attr.sched.axis = {z0.id, z1.id};
  load3.y.dtype = af::DT_FLOAT;
  *load3.y.axis = {z0.id, z1.id};
  *load3.y.repeats = {af::ops::One, af::ops::One};
  *load3.y.strides = {af::ops::Zero, af::ops::Zero};

  af::ascir_op::Eq eq0("eq0");
  eq0.x1 = load2.y;
  eq0.x2 = load3.y;
  eq0.attr.sched.axis = {z0.id, z1.id};
  eq0.y.dtype = af::DT_UINT8;
  *eq0.y.axis = {z0.id, z1.id};
  *eq0.y.repeats = {s0, s1};
  *eq0.y.strides = {s1, af::ops::One};

  af::ascir_op::Scalar scalar0("scalar0", graph);
  scalar0.ir_attr.SetValue("1");

  af::ascir_op::Where where0("where");
  where0.x1 = eq0.y;
  where0.x2 = matmul.y;
  where0.x3 = scalar0.y;
  where0.attr.sched.axis = {z0.id, z1.id};
  where0.y.dtype = af::DT_FLOAT;
  *where0.y.axis = {z0.id, z1.id};
  *where0.y.repeats = {s0, s1};
  *where0.y.strides = {s1, af::ops::One};

  af::ascir_op::Store store_op("store");
  store_op.x = where0.y;
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.y.dtype = af::DT_FLOAT;
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1, af::ops::One};

  af::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = af::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

static af::ComputeGraphPtr LoadMatmulCompareScalarFusedGraph() {
  auto builder = GraphBuilder("load_matmul_compare_scalar_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  af::AttrUtils::SetInt(data2->GetOpDescBarePtr(), "_parent_node_index", 2);
  auto data3 = builder.AddNode("data3", "Data", 0, 1);
  af::AttrUtils::SetInt(data3->GetOpDescBarePtr(), "_parent_node_index", 3);
  auto data4 = builder.AddNode("data4", "Data", 0, 1);
  af::AttrUtils::SetInt(data4->GetOpDescBarePtr(), "_parent_node_index", 4);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 5, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(data2, 0, ascbc, 2);
  builder.AddDataEdge(data3, 0, ascbc, 3);
  builder.AddDataEdge(data4, 0, ascbc, 4);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_matmul_elewise_brc_store");
  CreateMatmulCompareScalarGraph(sub_graph);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

class CubeFuseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::PlatformContext::GetInstance().Reset();
    ASSERT_EQ(
        att::soc_perf_test::SetSocPerfTestPlatform(att::soc_perf_test::kV35NpuArch, att::soc_perf_test::kSocEmpty),
        af::SUCCESS);
    auto stub_v2 = std::make_shared<ge::RuntimeStubV2>();
    ge::RuntimeStub::SetInstance(stub_v2);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::RuntimeStub::Reset();
  }
};

/* cube CodeGen UT测试用例 */
TEST_F(CubeFuseTest, CubeElewiseBrcTest) {
  bool gen_success = true;
  const std::map<std::string, std::string> shape_info;
  auto graph = LoadMatmulElewiseBrcFusedGraph();

  try {
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    codegen::Codegen codegen(codegen::CodegenOptions{});

    std::vector<::ascir::ScheduledResult> schedule_results;
    ::ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
    EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);
    codegen::CodegenResult ub_result;
    codegen::CodegenResult common_result;
    ::ascir::FusedScheduledResult ub_schedule_result = fused_schedule_result;
    ::ascir::FusedScheduledResult common_schedule_result = fused_schedule_result;
    if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result)) {
      // 过滤CVFusion的UBResult ub模板结果
      ascgen_utils::FilterCVFusionUBResult(ub_schedule_result);
      // 过滤CVFusion的CommonResult 兜底模板结果
      ascgen_utils::FilterCVFusionCommonResult(common_schedule_result);
    }
    EXPECT_EQ(codegen.Generate(shape_info, ub_schedule_result, ub_result), 0);
    EXPECT_EQ(codegen.Generate(shape_info, common_schedule_result, common_result), 0);

    // 校验ub_result.kernel中是否包含IncludeMatmulHeadFiles方法返回的所有头文件内容
    std::vector<std::string> expected_headers = {"#include \"arch35/mat_mul_v3_tiling_key_public.h\"",
                                                 "#include \"arch35/mat_mul_tiling_data.h\"",
                                                 "#include \"mat_mul_v3_common.h\"",
                                                 "#include \"arch35/mat_mul_asw_block.h\"",
                                                 "#include \"arch35/mat_mul_asw_kernel.h\"",
                                                 "#include \"arch35/mat_mul_stream_k_block.h\"",
                                                 "#include \"arch35/mat_mul_stream_k_kernel.h\"",
                                                 "#include \"arch35/mat_mul_v3_full_load_kernel_helper.h\"",
                                                 "#include \"arch35/mat_mul_full_load.h\"",
                                                 "#include \"arch35/mm_extension_interface/mm_copy_cube_out.h\"",
                                                 "#include \"arch35/mm_extension_interface/mm_custom_mm_policy.h\"",
                                                 "#include \"arch35/mat_mul_fixpipe_opti.h\"",
                                                 "#include \"arch35/block_scheduler_aswt.h\"",
                                                 "#include \"arch35/block_scheduler_streamk.h\"",
                                                 "#include \"arch35/mat_mul_streamk_basic_cmct.h\"",
                                                 "#include \"arch35/mat_mul_fixpipe_opti_basic_cmct.h\"",
                                                 "#include \"arch35/mat_mul_input_k_eq_zero_clear_output.h\""};

    for (const auto &header : expected_headers) {
      EXPECT_NE(ub_result.kernel.find(header), std::string::npos)
          << "Expected header not found in ub kernel: " << header;

      EXPECT_NE(common_result.kernel.find(header), std::string::npos)
          << "Expected header not found in common kernel: " << header;
    }
  } catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}

TEST_F(CubeFuseTest, CubeCompareScalarTest) {
  bool gen_success = true;
  const std::map<std::string, std::string> shape_info;
  auto graph = LoadMatmulCompareScalarFusedGraph();

  try {
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    codegen::Codegen codegen(codegen::CodegenOptions{});

    std::vector<::ascir::ScheduledResult> schedule_results;
    ::ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
    EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);
    codegen::CodegenResult ub_result;
    codegen::CodegenResult common_result;
    ::ascir::FusedScheduledResult ub_schedule_result = fused_schedule_result;
    ::ascir::FusedScheduledResult common_schedule_result = fused_schedule_result;
    if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result)) {
      // 过滤CVFusion的UBResult ub模板结果
      ascgen_utils::FilterCVFusionUBResult(ub_schedule_result);
      // 过滤CVFusion的CommonResult 兜底模板结果
      ascgen_utils::FilterCVFusionCommonResult(common_schedule_result);
    }
    EXPECT_EQ(codegen.Generate(shape_info, ub_schedule_result, ub_result), 0);
    EXPECT_EQ(codegen.Generate(shape_info, common_schedule_result, common_result), 0);
  } catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}
}  // namespace af
