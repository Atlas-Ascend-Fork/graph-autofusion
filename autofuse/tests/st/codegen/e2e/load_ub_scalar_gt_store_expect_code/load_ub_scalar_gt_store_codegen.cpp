/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <gtest/gtest.h>

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"
#include "ascir_ops_utils.h"
#include "codegen.h"
#include "e2e_common.h"

using namespace std;
using namespace ge;
using namespace af;
using namespace af::ops;
using namespace af::ascir_op;

namespace {
// x2为ub_scalar(全1 shape的load输出)的gt compare场景, 覆盖CompareApiCall非const scalar的广播发射分支
void LoadUbScalarGtStore_BeforeAutofuse(af::AscGraph &graph, ge::DataType in_data_type, ge::DataType out_data_type) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x1("x1");
  graph.AddNode(x1);
  x1.attr.sched.axis = {z0.id, z1.id, z2.id};
  x1.y.dtype = in_data_type;
  *(x1.y.axis) = {z0.id, z1.id, z2.id};

  Data x2("x2");
  graph.AddNode(x2);
  x2.attr.sched.axis = {z0.id, z1.id, z2.id};
  x2.y.dtype = in_data_type;
  *(x2.y.axis) = {z0.id, z1.id, z2.id};

  Load load1("load1");
  graph.AddNode(load1);
  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.y.dtype = in_data_type;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, One};

  Load load2("load2");
  graph.AddNode(load2);
  load2.x = x2.y;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  load2.y.dtype = in_data_type;
  *load2.y.axis = {z0.id, z1.id, z2.id};
  *load2.y.repeats = {One, One, One};
  *load2.y.strides = {One, One, One};

  af::ascir_op::Gt gt("gt");
  graph.AddNode(gt);
  gt.x1 = load1.y;
  gt.x2 = load2.y;
  gt.y.dtype = out_data_type;
  gt.attr.sched.axis = {z0.id, z1.id, z2.id};
  *gt.y.repeats = {s0, s1, s2};
  *gt.y.strides = {s1 * s2, s2, One};
  gt.attr.tmp_buffers = {{{af::Symbol(8192), -1}, MemAttr(), 0}};

  Store store("store");
  graph.AddNode(store);
  store.x = gt.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.y.dtype = out_data_type;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1 * s2, s2, One};

  Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id};
  y.y.dtype = out_data_type;
  *y.y.axis = {z0.id, z1.id, z2.id};
}

void LoadUbScalarGtStore_AfterInferOutput(af::AscGraph &graph, ge::DataType in_data_type, ge::DataType out_data_type) {
  auto x1 = graph.FindNode("x1");
  x1->attr.api.compute_type = ComputeType::kComputeInvalid;
  auto x2 = graph.FindNode("x2");
  x2->attr.api.compute_type = ComputeType::kComputeInvalid;

  auto load1 = graph.FindNode("load1");
  load1->outputs[0].attr.dtype = in_data_type;
  load1->attr.api.compute_type = ComputeType::kComputeLoad;

  auto load2 = graph.FindNode("load2");
  load2->outputs[0].attr.dtype = in_data_type;
  load2->attr.api.compute_type = ComputeType::kComputeLoad;

  auto gt = graph.FindNode("gt");
  gt->outputs[0].attr.dtype = out_data_type;
  gt->outputs[0].attr.axis = load1->outputs[0].attr.axis;
  gt->outputs[0].attr.repeats = load1->outputs[0].attr.repeats;
  gt->outputs[0].attr.strides = load1->outputs[0].attr.strides;
  gt->attr.api.compute_type = ComputeType::kComputeElewise;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.dtype = out_data_type;
  store->attr.api.compute_type = ComputeType::kComputeStore;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = ComputeType::kComputeInvalid;
}

void LoadUbScalarGtStore_AfterGetApiInfo(af::AscGraph &graph) {
  auto x1 = graph.FindNode("x1");
  x1->attr.api.type = ApiType::kAPITypeBuffer;
  x1->attr.api.unit = ComputeUnit::kUnitNone;

  auto x2 = graph.FindNode("x2");
  x2->attr.api.type = ApiType::kAPITypeBuffer;
  x2->attr.api.unit = ComputeUnit::kUnitNone;

  auto load1 = graph.FindNode("load1");
  load1->attr.api.type = ApiType::kAPITypeCompute;
  load1->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.type = ApiType::kAPITypeCompute;
  load2->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto gt = graph.FindNode("gt");
  gt->attr.api.type = ApiType::kAPITypeCompute;
  gt->attr.api.unit = ComputeUnit::kUnitVector;

  auto store = graph.FindNode("store");
  store->attr.api.type = ApiType::kAPITypeCompute;
  store->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto y = graph.FindNode("y");
  y->attr.api.type = ApiType::kAPITypeBuffer;
  y->attr.api.unit = ComputeUnit::kUnitNone;
}

void LoadUbScalarGtStore_AfterScheduler(af::AscGraph &graph) {
  auto all_axis = graph.GetAllAxis();
  auto z0 = all_axis[0]->id;
  auto z1 = all_axis[1]->id;
  auto z2 = all_axis[2]->id;

  auto [z1T, z1t] = graph.TileSplit(z1);
  std::vector<AxisId> axes{z0, z1T->id};
  auto block_axis = graph.MergeAxis(axes);
  auto [z0B, z0b] = graph.BlockSplit(block_axis->id);
  vector<AxisId> vectorized_axis{z1t->id, z2};
  vector<af::Expression> vectorized_strides{One, One};
  vector<af::Expression> load2_vectorized_strides{One, One};
  auto size = ge::GetSizeByDataType(ge::DT_FLOAT16);
  vectorized_strides[0] = af::sym::Align(graph.FindAxis(vectorized_axis[1])->size, 32 / size);

  all_axis = graph.GetAllAxis();
  auto m_axis = all_axis[block_axis->id];

  auto load1 = graph.FindNode("load1");
  graph.ApplySplit(load1, z1T->id, z1t->id);
  graph.ApplySchedAxisMerge(load1, block_axis->id, m_axis->from);
  graph.ApplySplit(load1, z0B->id, z0b->id);
  load1->attr.sched.loop_axis = z0b->id;
  load1->outputs[0].attr.vectorized_axis = vectorized_axis;
  load1->outputs[0].attr.vectorized_strides = vectorized_strides;

  auto load2 = graph.FindNode("load2");
  graph.ApplySplit(load2, z1T->id, z1t->id);
  graph.ApplySchedAxisMerge(load2, block_axis->id, m_axis->from);
  graph.ApplySplit(load2, z0B->id, z0b->id);
  load2->attr.sched.loop_axis = z0b->id;
  load2->outputs[0].attr.vectorized_axis = vectorized_axis;
  load2->outputs[0].attr.vectorized_strides = load2_vectorized_strides;

  auto gt = graph.FindNode("gt");
  graph.ApplySplit(gt, z1T->id, z1t->id);
  graph.ApplySchedAxisMerge(gt, block_axis->id, m_axis->from);
  graph.ApplySplit(gt, z0B->id, z0b->id);
  gt->attr.sched.loop_axis = z0b->id;
  gt->outputs[0].attr.vectorized_axis = vectorized_axis;
  gt->outputs[0].attr.vectorized_strides = vectorized_strides;

  auto store = graph.FindNode("store");
  graph.ApplySplit(store, z1T->id, z1t->id);
  graph.ApplySchedAxisMerge(store, block_axis->id, m_axis->from);
  graph.ApplySplit(store, z0B->id, z0b->id);
  store->attr.sched.loop_axis = z0b->id;
  store->outputs[0].attr.vectorized_axis = vectorized_axis;
  store->outputs[0].attr.vectorized_strides = vectorized_strides;
}

void LoadUbScalarGtStore_AfterQueBufAlloc(af::AscGraph &graph) {
  auto x1 = graph.FindNode("x1");
  x1->outputs[0].attr.mem.tensor_id = 0;
  x1->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x1->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x1->outputs[0].attr.mem.position = Position::kPositionGM;

  auto x2 = graph.FindNode("x2");
  x2->outputs[0].attr.mem.tensor_id = 1;
  x2->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x2->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x2->outputs[0].attr.mem.position = Position::kPositionGM;

  auto load1 = graph.FindNode("load1");
  load1->outputs[0].attr.mem.tensor_id = 2;
  load1->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  load1->outputs[0].attr.mem.position = Position::kPositionVecIn;
  load1->outputs[0].attr.buf.id = af::kIdNone;
  load1->outputs[0].attr.que.id = 0;
  load1->outputs[0].attr.mem.reuse_id = 0;
  load1->outputs[0].attr.que.depth = 2;
  load1->outputs[0].attr.que.buf_num = 2;
  load1->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  load1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto load2 = graph.FindNode("load2");
  load2->outputs[0].attr.mem.tensor_id = 3;
  load2->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  load2->outputs[0].attr.mem.position = Position::kPositionVecIn;
  load2->outputs[0].attr.buf.id = af::kIdNone;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.mem.reuse_id = 0;
  load2->outputs[0].attr.que.depth = 2;
  load2->outputs[0].attr.que.buf_num = 2;
  load2->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto gt = graph.FindNode("gt");
  gt->outputs[0].attr.mem.tensor_id = 4;
  gt->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  gt->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  gt->outputs[0].attr.mem.position = Position::kPositionVecOut;
  gt->outputs[0].attr.buf.id = af::kIdNone;
  gt->outputs[0].attr.que.id = 2;
  gt->outputs[0].attr.mem.reuse_id = 1;
  gt->outputs[0].attr.que.depth = 2;
  gt->outputs[0].attr.que.buf_num = 2;
  gt->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  gt->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.mem.tensor_id = 5;
  store->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = Position::kPositionGM;
}
}  // namespace

std::vector<std::string> splitString(const std::string &input, char delimiter) {
  std::vector<std::string> result;
  std::stringstream ss(input);
  std::string token;

  while (std::getline(ss, token, delimiter)) {
    result.push_back(token);
  }

  return result;
}

class LoadUbScalarGtStoreTest : public testing::Test {};

TEST_F(LoadUbScalarGtStoreTest, LoadUbScalarGtStoreCodegen) {
  bool gen_success = true;
  af::AscGraph test_graph("load_ub_scalar_gt_store");
  std::string tilig_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";
  LoadUbScalarGtStore_BeforeAutofuse(test_graph, ge::DT_FLOAT16, ge::DT_UINT8);
  LoadUbScalarGtStore_AfterInferOutput(test_graph, ge::DT_FLOAT16, ge::DT_UINT8);

  std::vector<af::AscGraph> test_impl_graphs = {af::AscGraph("load_ub_scalar_gt_store_general_0_nil_0_nil")};

  test_impl_graphs[0].CopyFrom(test_graph);
  LoadUbScalarGtStore_AfterGetApiInfo(test_impl_graphs[0]);
  LoadUbScalarGtStore_AfterScheduler(test_impl_graphs[0]);
  LoadUbScalarGtStore_AfterQueBufAlloc(test_impl_graphs[0]);

  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');
  std::string kernel_src_file_name = parts[0];
  std::string tiling_src_file_name = parts[1];
  std::string tiling_data_src_file_name = parts[2];

  try {
    auto codegen = codegen::Codegen(codegen::CodegenOptions{.tiling_lib_path = ATT_SO_NAME,
                                                            .tiling_lib_codegen_symbol = "CodegenTiling",
                                                            .using_att_calc_qbt_size = false});

    std::fstream kernel_file(kernel_src_file_name, std::ios::out);
    std::fstream tiling_file(tiling_src_file_name, std::ios::out);
    std::fstream tiling_data_file(tiling_data_src_file_name, std::ios::out);

    ::ascir::ScheduledResult schedule_result;
    std::vector<::ascir::ScheduledResult> schedule_results{schedule_result};
    ::ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.fused_graph_name = af::AscendString("load_ub_scalar_gt_store");
    fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
    InitScheduleResultsByImplGraphs(test_impl_graphs, fused_schedule_result);
    codegen::CodegenResult result;
    EXPECT_EQ(codegen.Generate(fused_schedule_result, result), 0);
    kernel_file << tilig_stub << RemoveSubDirInclude(result.kernel);
    tiling_file << result.tiling;
    tiling_data_file << result.tiling_data;
  } catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}
