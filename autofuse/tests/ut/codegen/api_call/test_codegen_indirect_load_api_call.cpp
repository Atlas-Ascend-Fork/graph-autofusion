/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "codegen_kernel.h"
#include "common_utils.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph/utils/graph_utils.h"
#include "indirect_load_utils.h"
#include "platform_context.h"
#include "runtime_stub.h"
#include "utils/api_call_factory.h"
#include "v35/ascir/generator/v2_ascir_codegen_impl.h"
#include "v35/codegen/reg_api_call/reg_indirect_load_api_call.h"

using namespace af::ops;
using namespace af::ascir_op;
using namespace codegen;
using namespace ascgen_utils::indirect_load;

namespace {

TEST(IndirectLoadCodegenImplTest, LoadsApiHeadersInDeclarationOrder) {
  const auto headers = af::ascir::IndirectLoadAscIrCodegenImplV2().LoadApiHeaderFiles(false);
  const std::vector<std::string> expected = {"datacopy_reg_base.h",           "indirect_load_simd_policy_reg_base.h",
                                             "indirect_load_simd_reg_base.h", "indirect_load_sk_reg_base.h",
                                             "indirect_load_simt_reg_base.h", "transpose_reg_base.h"};
  EXPECT_EQ(headers, expected);
}

struct ILTestGraph {
  af::AscGraph graph;
  af::Expression s0;
  af::Expression s1;
  af::Expression s2;
  af::Expression s3;
  af::Axis z0;
  af::Axis z1;
  af::Axis z2;
  af::Axis z3;
  // BuildSimdGraph 分裂出的向量化视图轴（z2z3Tb / z2z3t），供 padded 视图设置使用。
  af::AxisId padded_row_axis;
  af::AxisId padded_tail_axis;

  explicit ILTestGraph(const char *name)
      : graph(name),
        s0(graph.CreateSizeVar("s0")),
        s1(graph.CreateSizeVar("s1")),
        s2(graph.CreateSizeVar("s2")),
        s3(graph.CreateSizeVar("s3")),
        z0(graph.CreateAxis("z0", s0)),
        z1(graph.CreateAxis("z1", s1)),
        z2(graph.CreateAxis("z2", s2)),
        z3(graph.CreateAxis("z3", s3)) {}

  ILTestGraph(const char *name, int64_t size0, int64_t size1, int64_t size2, int64_t size3)
      : graph(name),
        s0(af::Symbol(size0)),
        s1(af::Symbol(size1)),
        s2(af::Symbol(size2)),
        s3(af::Symbol(size3)),
        z0(graph.CreateAxis("z0", s0)),
        z1(graph.CreateAxis("z1", s1)),
        z2(graph.CreateAxis("z2", s2)),
        z3(graph.CreateAxis("z3", s3)) {}
};

// ======================== SIMD graph ========================
// z0, z1: input axes  |  z2, z3: output/index axes
// IndirectLoad axis=1, rank=2

// 变位宽生产者（cast）经通用对齐 pad 后的视图形态：行轴与尾轴共同构成两维向量化
// 视图，行间步长被改写到对齐块宽度（此处用 2 倍逻辑行步长模拟），与逻辑紧凑 strides
// 不一致。
void SetPaddedVectorizedViews(const ILTestGraph &g) {
  const auto il_node = g.graph.FindNode("indirect_load");
  il_node->outputs[0].attr.vectorized_axis = {g.padded_row_axis, g.padded_tail_axis};
  il_node->outputs[0].attr.vectorized_strides = {g.s3 * af::Symbol(2), af::Symbol(3)};
  const auto idx_node = g.graph.FindNode("idx_load");
  idx_node->outputs[0].attr.vectorized_axis = {g.padded_row_axis, g.padded_tail_axis};
  idx_node->outputs[0].attr.vectorized_strides = {g.s3 * af::Symbol(2), af::Symbol(3)};
}

void BuildSimdGraph(ILTestGraph &g, af::DataType input_dtype = ge::DT_FLOAT16, bool padded_views = false) {
  const af::Expression One = af::sym::kSymbolOne;

  Data x_data("x", g.graph);
  x_data.ir_attr.SetIndex(0);
  x_data.y.dtype = input_dtype;
  x_data.attr.sched.axis = {g.z0.id, g.z1.id};
  *x_data.y.axis = {g.z0.id, g.z1.id};
  *x_data.y.repeats = {g.s0, g.s1};
  *x_data.y.strides = {g.s1, One};

  Data idx_data("idx", g.graph);
  idx_data.ir_attr.SetIndex(1);
  idx_data.y.dtype = ge::DT_INT32;
  idx_data.attr.sched.axis = {g.z2.id, g.z3.id};
  *idx_data.y.axis = {g.z2.id, g.z3.id};
  *idx_data.y.repeats = {g.s2, g.s3};
  *idx_data.y.strides = {g.s3, One};

  Load x_load("x_load");
  g.graph.AddNode(x_load);
  x_load.x = x_data.y;
  x_load.y.dtype = input_dtype;
  x_load.attr.sched.axis = {g.z0.id, g.z1.id};
  *x_load.y.axis = {g.z0.id, g.z1.id};
  *x_load.y.repeats = {g.s0, g.s1};
  *x_load.y.strides = {g.s1, One};

  Load idx_load("idx_load");
  g.graph.AddNode(idx_load);
  idx_load.x = idx_data.y;
  idx_load.y.dtype = ge::DT_INT32;
  idx_load.attr.sched.axis = {g.z2.id, g.z3.id};
  *idx_load.y.axis = {g.z2.id, g.z3.id};
  *idx_load.y.repeats = {g.s2, g.s3};
  *idx_load.y.strides = {g.s3, One};

  IndirectLoad il("indirect_load");
  g.graph.AddNode(il);
  il.x1 = x_load.y;
  il.x2 = idx_load.y;
  il.ir_attr.SetAxis(1);
  il.attr.sched.axis = {g.z2.id, g.z3.id};
  *il.y.axis = {g.z2.id, g.z3.id};
  *il.y.repeats = {g.s2, g.s3};
  *il.y.strides = {g.s3, One};

  Store store("store");
  g.graph.AddNode(store);
  store.x = il.y;
  store.y.dtype = input_dtype;
  store.attr.sched.axis = {g.z2.id, g.z3.id};
  *store.y.axis = {g.z2.id, g.z3.id};
  *store.y.repeats = {g.s2, g.s3};
  *store.y.strides = {g.s3, One};

  Output y_out("y");
  g.graph.AddNode(y_out);
  y_out.ir_attr.SetIndex(0);
  y_out.x = store.y;
  y_out.y.dtype = input_dtype;

  // ----- API attrs -----
  for (const char *name : {"x", "idx"}) {
    auto n = g.graph.FindNode(name);
    n->attr.api.compute_type = af::ComputeType::kComputeInvalid;
    n->attr.api.type = af::ApiType::kAPITypeBuffer;
    n->attr.api.unit = af::ComputeUnit::kUnitNone;
  }
  for (const char *name : {"x_load", "idx_load"}) {
    auto n = g.graph.FindNode(name);
    n->attr.api.compute_type = af::ComputeType::kComputeLoad;
    n->attr.api.type = af::ApiType::kAPITypeCompute;
    n->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  }
  auto il_node = g.graph.FindNode("indirect_load");
  il_node->attr.api.compute_type = af::ComputeType::kComputeLoad;
  il_node->attr.api.type = af::ApiType::kAPITypeCompute;
  il_node->attr.api.unit = af::ComputeUnit::kUnitVector;

  auto store_node = g.graph.FindNode("store");
  store_node->attr.api.compute_type = af::ComputeType::kComputeStore;
  store_node->attr.api.type = af::ApiType::kAPITypeCompute;
  store_node->attr.api.unit = af::ComputeUnit::kUnitMTE2;

  auto y_node = g.graph.FindNode("y");
  y_node->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y_node->attr.api.type = af::ApiType::kAPITypeBuffer;
  y_node->attr.api.unit = af::ComputeUnit::kUnitNone;

  // ----- Axis splitting -----
  auto axes = g.graph.GetAllAxis();
  auto z2_id = axes[2]->id;
  auto z3_id = axes[3]->id;

  auto z2z3 = g.graph.MergeAxis({z2_id, z3_id});
  auto [z2z3T, z2z3t] = g.graph.TileSplit(z2z3->id);
  auto [z2z3TB, z2z3Tb] = g.graph.BlockSplit(z2z3T->id);
  g.padded_row_axis = z2z3Tb->id;
  g.padded_tail_axis = z2z3t->id;

  for (auto n : g.graph.GetAllNodes()) {
    if (IsOps<Data>(n) || IsOps<Output>(n) || n->GetName() == "x_load") continue;
    g.graph.ApplyMerge(n, z2z3->id);
    g.graph.ApplySplit(n, z2z3T->id, z2z3t->id);
    g.graph.ApplySplit(n, z2z3TB->id, z2z3Tb->id);
    g.graph.ApplyReorder(n, {z2z3TB->id, z2z3Tb->id, z2z3t->id});
  }

  il_node->attr.sched.loop_axis = z2z3Tb->id;
  store_node->attr.sched.loop_axis = z2z3Tb->id;
  if (padded_views) {
    SetPaddedVectorizedViews(g);
  } else {
    il_node->outputs[0].attr.vectorized_axis = {z2z3t->id};
    il_node->outputs[0].attr.vectorized_strides = {One};
  }
  store_node->outputs[0].attr.vectorized_axis = {z2z3t->id};
  store_node->outputs[0].attr.vectorized_strides = {One};

  il_node->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};

  // ----- Memory allocation -----
  auto x = g.graph.FindNode("x");
  x->outputs[0].attr.mem.tensor_id = 0;
  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  x->outputs[0].attr.mem.position = af::Position::kPositionGM;
  x->outputs[0].attr.que.id = af::kIdNone;

  auto idx = g.graph.FindNode("idx");
  idx->outputs[0].attr.mem.tensor_id = 3;
  idx->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  idx->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  idx->outputs[0].attr.mem.position = af::Position::kPositionGM;
  idx->outputs[0].attr.que.id = af::kIdNone;

  auto x_load_node = g.graph.FindNode("x_load");
  x_load_node->outputs[0].attr.mem.tensor_id = 4;
  x_load_node->outputs[0].attr.mem.reuse_id = 4;
  x_load_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  x_load_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  x_load_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  x_load_node->outputs[0].attr.que.id = 1;
  x_load_node->outputs[0].attr.que.depth = 2;
  x_load_node->outputs[0].attr.que.buf_num = 2;

  auto idx_load_node = g.graph.FindNode("idx_load");
  idx_load_node->outputs[0].attr.mem.tensor_id = 5;
  idx_load_node->outputs[0].attr.mem.reuse_id = 5;
  idx_load_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  idx_load_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  idx_load_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  idx_load_node->outputs[0].attr.que.id = 2;
  idx_load_node->outputs[0].attr.que.depth = 2;
  idx_load_node->outputs[0].attr.que.buf_num = 2;

  il_node->outputs[0].attr.dtype = input_dtype;
  il_node->outputs[0].attr.mem.tensor_id = 1;
  il_node->outputs[0].attr.mem.reuse_id = 1;
  il_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  il_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  il_node->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  il_node->outputs[0].attr.que.id = 0;
  il_node->outputs[0].attr.que.depth = 2;
  il_node->outputs[0].attr.que.buf_num = 2;

  store_node->outputs[0].attr.mem.tensor_id = 2;
  store_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  store_node->outputs[0].attr.mem.position = af::Position::kPositionGM;
  store_node->outputs[0].attr.que.id = af::kIdNone;
}

// ======================== SIMT graph ========================
// Same output axes z2, z3 as SIMD but annotated as SIMT.
// IndirectLoad output chain ends at Store (for FindSimtOutputStore).

void BuildSimtGraph(ILTestGraph &g, bool add_second_output = false, bool add_index_store = false) {
  const af::Expression One = af::sym::kSymbolOne;

  Data x_data("x", g.graph);
  x_data.ir_attr.SetIndex(0);
  x_data.y.dtype = ge::DT_FLOAT16;
  x_data.attr.sched.axis = {g.z0.id, g.z1.id};
  *x_data.y.axis = {g.z0.id, g.z1.id};
  *x_data.y.repeats = {g.s0, g.s1};
  *x_data.y.strides = {g.s1, One};

  Data idx_data("idx", g.graph);
  idx_data.ir_attr.SetIndex(1);
  idx_data.y.dtype = ge::DT_INT32;
  idx_data.attr.sched.axis = {g.z2.id, g.z3.id};
  *idx_data.y.axis = {g.z2.id, g.z3.id};
  *idx_data.y.repeats = {g.s2, g.s3};
  *idx_data.y.strides = {g.s3, One};

  Load x_load("x_load");
  g.graph.AddNode(x_load);
  x_load.x = x_data.y;
  x_load.y.dtype = ge::DT_FLOAT16;
  x_load.attr.sched.axis = {g.z0.id, g.z1.id};
  *x_load.y.axis = {g.z0.id, g.z1.id};
  *x_load.y.repeats = {g.s0, g.s1};
  *x_load.y.strides = {g.s1, One};

  Load idx_load("idx_load");
  g.graph.AddNode(idx_load);
  idx_load.x = idx_data.y;
  idx_load.y.dtype = ge::DT_INT32;
  idx_load.attr.sched.axis = {g.z2.id, g.z3.id};
  *idx_load.y.axis = {g.z2.id, g.z3.id};
  *idx_load.y.repeats = {g.s2, g.s3};
  *idx_load.y.strides = {g.s3, One};

  IndirectLoad il("indirect_load");
  g.graph.AddNode(il);
  il.x1 = x_load.y;
  il.x2 = idx_load.y;
  il.ir_attr.SetAxis(1);
  il.attr.sched.axis = {g.z2.id, g.z3.id};
  *il.y.axis = {g.z2.id, g.z3.id};
  *il.y.repeats = {g.s2, g.s3};
  *il.y.strides = {g.s3, One};

  Store store("store");
  g.graph.AddNode(store);
  store.x = il.y;
  store.y.dtype = ge::DT_FLOAT16;
  store.attr.sched.axis = {g.z2.id, g.z3.id};
  *store.y.axis = {g.z2.id, g.z3.id};
  *store.y.repeats = {g.s2, g.s3};
  *store.y.strides = {g.s3, One};

  Output y_out("y");
  g.graph.AddNode(y_out);
  y_out.ir_attr.SetIndex(0);
  y_out.x = store.y;
  y_out.y.dtype = ge::DT_FLOAT16;

  Store second_store("second_store");
  Output second_y_out("second_y");
  if (add_second_output) {
    g.graph.AddNode(second_store);
    second_store.x = il.y;
    second_store.y.dtype = ge::DT_FLOAT16;
    second_store.attr.sched.axis = {g.z2.id, g.z3.id};
    *second_store.y.axis = {g.z2.id, g.z3.id};
    *second_store.y.repeats = {g.s2, g.s3};
    *second_store.y.strides = {g.s3, One};

    g.graph.AddNode(second_y_out);
    second_y_out.ir_attr.SetIndex(1);
    second_y_out.x = second_store.y;
    second_y_out.y.dtype = ge::DT_FLOAT16;
  }

  // [index 链外部 store] index 生产链尾部（idx_load 输出）除喂 IndirectLoad 外的
  // 区域外直连 store：调度侧守卫会将其纳入 SIMT 融合区域，lowering 收为空节点
  // 输出链，codegen 用 Index() 求值器结果注入链值。
  Store index_store("index_store");
  Output index_y_out("index_y");
  if (add_index_store) {
    g.graph.AddNode(index_store);
    index_store.x = idx_load.y;
    index_store.y.dtype = ge::DT_INT32;
    index_store.attr.sched.axis = {g.z2.id, g.z3.id};
    *index_store.y.axis = {g.z2.id, g.z3.id};
    *index_store.y.repeats = {g.s2, g.s3};
    *index_store.y.strides = {g.s3, One};

    g.graph.AddNode(index_y_out);
    index_y_out.ir_attr.SetIndex(2);
    index_y_out.x = index_store.y;
    index_y_out.y.dtype = ge::DT_INT32;
  }

  // ----- API attrs -----
  for (const char *name : {"x", "idx"}) {
    auto n = g.graph.FindNode(name);
    n->attr.api.compute_type = af::ComputeType::kComputeInvalid;
    n->attr.api.type = af::ApiType::kAPITypeBuffer;
    n->attr.api.unit = af::ComputeUnit::kUnitNone;
  }
  for (const char *name : {"x_load", "idx_load"}) {
    auto n = g.graph.FindNode(name);
    n->attr.api.compute_type = af::ComputeType::kComputeLoad;
    n->attr.api.type = af::ApiType::kAPITypeCompute;
    n->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  }
  auto il_node = g.graph.FindNode("indirect_load");
  il_node->attr.api.compute_type = af::ComputeType::kComputeLoad;
  il_node->attr.api.type = af::ApiType::kAPITypeCompute;
  il_node->attr.api.unit = af::ComputeUnit::kUnitVector;

  auto store_node = g.graph.FindNode("store");
  store_node->attr.api.compute_type = af::ComputeType::kComputeStore;
  store_node->attr.api.type = af::ApiType::kAPITypeCompute;
  store_node->attr.api.unit = af::ComputeUnit::kUnitMTE2;

  if (add_second_output) {
    auto second_store_node = g.graph.FindNode("second_store");
    second_store_node->attr.api.compute_type = af::ComputeType::kComputeStore;
    second_store_node->attr.api.type = af::ApiType::kAPITypeCompute;
    second_store_node->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  }

  if (add_index_store) {
    auto index_store_node = g.graph.FindNode("index_store");
    index_store_node->attr.api.compute_type = af::ComputeType::kComputeStore;
    index_store_node->attr.api.type = af::ApiType::kAPITypeCompute;
    index_store_node->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  }

  auto y_node = g.graph.FindNode("y");
  y_node->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y_node->attr.api.type = af::ApiType::kAPITypeBuffer;
  y_node->attr.api.unit = af::ComputeUnit::kUnitNone;

  // ----- Axis splitting: merge output axes → z2z3 -----
  auto axes = g.graph.GetAllAxis();
  auto z2_id = axes[2]->id;
  auto z3_id = axes[3]->id;

  auto z2z3 = g.graph.MergeAxis({z2_id, z3_id});
  auto [z2z3T, z2z3t] = g.graph.TileSplit(z2z3->id);
  auto [z2z3TB, z2z3Tb] = g.graph.BlockSplit(z2z3T->id);

  for (auto n : g.graph.GetAllNodes()) {
    if (IsOps<Data>(n) || IsOps<Output>(n) || n->GetName() == "x_load") continue;
    g.graph.ApplyMerge(n, z2z3->id);
    g.graph.ApplySplit(n, z2z3T->id, z2z3t->id);
    g.graph.ApplySplit(n, z2z3TB->id, z2z3Tb->id);
    g.graph.ApplyReorder(n, {z2z3TB->id, z2z3Tb->id, z2z3t->id});
  }

  il_node->attr.sched.loop_axis = z2z3Tb->id;
  il_node->outputs[0].attr.vectorized_axis = {z2z3t->id};
  il_node->outputs[0].attr.vectorized_strides = {One};

  store_node->attr.sched.loop_axis = z2z3Tb->id;
  store_node->outputs[0].attr.vectorized_axis = {z2z3t->id};
  store_node->outputs[0].attr.vectorized_strides = {One};

  if (add_second_output) {
    auto second_store_node = g.graph.FindNode("second_store");
    second_store_node->attr.sched.loop_axis = z2z3Tb->id;
    second_store_node->outputs[0].attr.vectorized_axis = {z2z3t->id};
    second_store_node->outputs[0].attr.vectorized_strides = {One};
  }

  if (add_index_store) {
    auto index_store_node = g.graph.FindNode("index_store");
    index_store_node->attr.sched.loop_axis = z2z3Tb->id;
    index_store_node->outputs[0].attr.vectorized_axis = {z2z3t->id};
    index_store_node->outputs[0].attr.vectorized_strides = {One};
  }

  // ----- Memory allocation -----
  auto x = g.graph.FindNode("x");
  x->outputs[0].attr.mem.tensor_id = 0;
  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  x->outputs[0].attr.mem.position = af::Position::kPositionGM;
  x->outputs[0].attr.que.id = af::kIdNone;

  auto idx = g.graph.FindNode("idx");
  idx->outputs[0].attr.mem.tensor_id = 3;
  idx->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  idx->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  idx->outputs[0].attr.mem.position = af::Position::kPositionGM;
  idx->outputs[0].attr.que.id = af::kIdNone;

  auto x_load_node = g.graph.FindNode("x_load");
  x_load_node->outputs[0].attr.mem.tensor_id = 4;
  x_load_node->outputs[0].attr.mem.reuse_id = 4;
  x_load_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  x_load_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  x_load_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  x_load_node->outputs[0].attr.que.id = 1;
  x_load_node->outputs[0].attr.que.depth = 2;
  x_load_node->outputs[0].attr.que.buf_num = 2;

  auto idx_load_node = g.graph.FindNode("idx_load");
  idx_load_node->outputs[0].attr.mem.tensor_id = 5;
  idx_load_node->outputs[0].attr.mem.reuse_id = 5;
  idx_load_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  idx_load_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  idx_load_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  idx_load_node->outputs[0].attr.que.id = 2;
  idx_load_node->outputs[0].attr.que.depth = 2;
  idx_load_node->outputs[0].attr.que.buf_num = 2;

  il_node->outputs[0].attr.dtype = ge::DT_FLOAT16;  // needed for SIMT dtype check vs Store
  il_node->outputs[0].attr.mem.tensor_id = 1;
  il_node->outputs[0].attr.mem.reuse_id = 1;
  il_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  il_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  il_node->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  il_node->outputs[0].attr.que.id = 0;
  il_node->outputs[0].attr.que.depth = 2;
  il_node->outputs[0].attr.que.buf_num = 2;

  store_node->outputs[0].attr.mem.tensor_id = 2;
  store_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  store_node->outputs[0].attr.mem.position = af::Position::kPositionGM;
  store_node->outputs[0].attr.que.id = af::kIdNone;

  if (add_second_output) {
    auto second_store_node = g.graph.FindNode("second_store");
    second_store_node->outputs[0].attr.mem.tensor_id = 10;
    second_store_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
    second_store_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
    second_store_node->outputs[0].attr.mem.position = af::Position::kPositionGM;
    second_store_node->outputs[0].attr.que.id = af::kIdNone;
  }

  if (add_index_store) {
    auto index_store_node = g.graph.FindNode("index_store");
    index_store_node->outputs[0].attr.mem.tensor_id = 11;
    index_store_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
    index_store_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
    index_store_node->outputs[0].attr.mem.position = af::Position::kPositionGM;
    index_store_node->outputs[0].attr.que.id = af::kIdNone;
  }
}

// ======================== SIMT post-reduce graph ========================
// IndirectLoad + addend -> inline Add -> fake Reduce (compute_type=kComputeReduce) -> Store

void BuildSimtPostReduceGraph(ILTestGraph &g, bool add_parallel_output = false) {
  const af::Expression One = af::sym::kSymbolOne;

  Data x_data("x", g.graph);
  x_data.ir_attr.SetIndex(0);
  x_data.y.dtype = ge::DT_FLOAT16;
  x_data.attr.sched.axis = {g.z0.id, g.z1.id};
  *x_data.y.axis = {g.z0.id, g.z1.id};
  *x_data.y.repeats = {g.s0, g.s1};
  *x_data.y.strides = {g.s1, One};

  Data idx_data("idx", g.graph);
  idx_data.ir_attr.SetIndex(1);
  idx_data.y.dtype = ge::DT_INT32;
  idx_data.attr.sched.axis = {g.z2.id, g.z3.id};
  *idx_data.y.axis = {g.z2.id, g.z3.id};
  *idx_data.y.repeats = {g.s2, g.s3};
  *idx_data.y.strides = {g.s3, One};

  Data addend_data("addend", g.graph);
  addend_data.ir_attr.SetIndex(2);
  addend_data.y.dtype = ge::DT_FLOAT16;
  addend_data.attr.sched.axis = {g.z2.id, g.z3.id};
  *addend_data.y.axis = {g.z2.id, g.z3.id};
  *addend_data.y.repeats = {g.s2, g.s3};
  *addend_data.y.strides = {g.s3, One};

  Load x_load("x_load");
  g.graph.AddNode(x_load);
  x_load.x = x_data.y;
  x_load.y.dtype = ge::DT_FLOAT16;
  x_load.attr.sched.axis = {g.z0.id, g.z1.id};
  *x_load.y.axis = {g.z0.id, g.z1.id};
  *x_load.y.repeats = {g.s0, g.s1};
  *x_load.y.strides = {g.s1, One};

  Load idx_load("idx_load");
  g.graph.AddNode(idx_load);
  idx_load.x = idx_data.y;
  idx_load.y.dtype = ge::DT_INT32;
  idx_load.attr.sched.axis = {g.z2.id, g.z3.id};
  *idx_load.y.axis = {g.z2.id, g.z3.id};
  *idx_load.y.repeats = {g.s2, g.s3};
  *idx_load.y.strides = {g.s3, One};

  Load addend_load("addend_load");
  g.graph.AddNode(addend_load);
  addend_load.x = addend_data.y;
  addend_load.y.dtype = ge::DT_FLOAT16;
  addend_load.attr.sched.axis = {g.z2.id, g.z3.id};
  *addend_load.y.axis = {g.z2.id, g.z3.id};
  *addend_load.y.repeats = {g.s2, g.s3};
  *addend_load.y.strides = {g.s3, One};

  IndirectLoad il("indirect_load");
  g.graph.AddNode(il);
  il.x1 = x_load.y;
  il.x2 = idx_load.y;
  il.ir_attr.SetAxis(1);
  il.attr.sched.axis = {g.z2.id, g.z3.id};
  *il.y.axis = {g.z2.id, g.z3.id};
  *il.y.repeats = {g.s2, g.s3};
  *il.y.strides = {g.s3, One};

  Add output_transform("output_transform");
  g.graph.AddNode(output_transform);
  output_transform.x1 = il.y;
  output_transform.x2 = addend_load.y;
  output_transform.attr.sched.axis = {g.z2.id, g.z3.id};
  *output_transform.y.axis = {g.z2.id, g.z3.id};
  *output_transform.y.repeats = {g.s2, g.s3};
  *output_transform.y.strides = {g.s3, One};

  // Fake Reduce: Abs with compute_type=kComputeReduce
  Abs reduce("reduce");
  g.graph.AddNode(reduce);
  reduce.x = output_transform.y;
  reduce.attr.sched.axis = {g.z2.id, g.z3.id};
  *reduce.y.axis = {g.z2.id, g.z3.id};
  *reduce.y.repeats = {g.s2, g.s3};
  *reduce.y.strides = {g.s3, One};

  Store store("store");
  g.graph.AddNode(store);
  store.x = reduce.y;
  store.y.dtype = ge::DT_FLOAT16;
  store.attr.sched.axis = {g.z2.id, g.z3.id};
  *store.y.axis = {g.z2.id, g.z3.id};
  *store.y.repeats = {g.s2, g.s3};
  *store.y.strides = {g.s3, One};

  Output y_out("y");
  g.graph.AddNode(y_out);
  y_out.ir_attr.SetIndex(0);
  y_out.x = store.y;
  y_out.y.dtype = ge::DT_FLOAT16;

  Store direct_store("direct_store");
  Output direct_y("direct_y");
  if (add_parallel_output) {
    g.graph.AddNode(direct_store);
    direct_store.x = il.y;
    direct_store.y.dtype = ge::DT_FLOAT16;
    direct_store.attr.sched.axis = {g.z2.id, g.z3.id};
    *direct_store.y.axis = {g.z2.id, g.z3.id};
    *direct_store.y.repeats = {g.s2, g.s3};
    *direct_store.y.strides = {g.s3, One};

    g.graph.AddNode(direct_y);
    direct_y.ir_attr.SetIndex(1);
    direct_y.x = direct_store.y;
    direct_y.y.dtype = ge::DT_FLOAT16;
  }

  // ----- API attrs -----
  for (const char *name : {"x", "idx", "addend"}) {
    auto n = g.graph.FindNode(name);
    n->attr.api.compute_type = af::ComputeType::kComputeInvalid;
    n->attr.api.type = af::ApiType::kAPITypeBuffer;
    n->attr.api.unit = af::ComputeUnit::kUnitNone;
  }
  for (const char *name : {"x_load", "idx_load", "addend_load"}) {
    auto n = g.graph.FindNode(name);
    n->attr.api.compute_type = af::ComputeType::kComputeLoad;
    n->attr.api.type = af::ApiType::kAPITypeCompute;
    n->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  }
  auto il_node = g.graph.FindNode("indirect_load");
  il_node->attr.api.compute_type = af::ComputeType::kComputeLoad;
  il_node->attr.api.type = af::ApiType::kAPITypeCompute;
  il_node->attr.api.unit = af::ComputeUnit::kUnitVector;

  auto output_transform_node = g.graph.FindNode("output_transform");
  output_transform_node->attr.api.compute_type = af::ComputeType::kComputeElewise;
  output_transform_node->attr.api.type = af::ApiType::kAPITypeCompute;
  output_transform_node->attr.api.unit = af::ComputeUnit::kUnitVector;

  auto reduce_node = g.graph.FindNode("reduce");
  reduce_node->attr.api.compute_type = af::ComputeType::kComputeReduce;
  reduce_node->attr.api.type = af::ApiType::kAPITypeCompute;
  reduce_node->attr.api.unit = af::ComputeUnit::kUnitVector;

  auto store_node = g.graph.FindNode("store");
  store_node->attr.api.compute_type = af::ComputeType::kComputeStore;
  store_node->attr.api.type = af::ApiType::kAPITypeCompute;
  store_node->attr.api.unit = af::ComputeUnit::kUnitMTE2;

  auto y_node = g.graph.FindNode("y");
  y_node->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y_node->attr.api.type = af::ApiType::kAPITypeBuffer;
  y_node->attr.api.unit = af::ComputeUnit::kUnitNone;

  if (add_parallel_output) {
    auto direct_store_node = g.graph.FindNode("direct_store");
    direct_store_node->attr.api.compute_type = af::ComputeType::kComputeStore;
    direct_store_node->attr.api.type = af::ApiType::kAPITypeCompute;
    direct_store_node->attr.api.unit = af::ComputeUnit::kUnitMTE2;
    auto direct_y_node = g.graph.FindNode("direct_y");
    direct_y_node->attr.api.compute_type = af::ComputeType::kComputeInvalid;
    direct_y_node->attr.api.type = af::ApiType::kAPITypeBuffer;
    direct_y_node->attr.api.unit = af::ComputeUnit::kUnitNone;
  }

  // ----- Axis splitting -----
  auto axes = g.graph.GetAllAxis();
  auto z2_id = axes[2]->id;
  auto z3_id = axes[3]->id;

  auto z2z3 = g.graph.MergeAxis({z2_id, z3_id});
  auto [z2z3T, z2z3t] = g.graph.TileSplit(z2z3->id);
  auto [z2z3TB, z2z3Tb] = g.graph.BlockSplit(z2z3T->id);

  for (auto n : g.graph.GetAllNodes()) {
    if (IsOps<Data>(n) || IsOps<Output>(n) || n->GetName() == "x_load") continue;
    g.graph.ApplyMerge(n, z2z3->id);
    g.graph.ApplySplit(n, z2z3T->id, z2z3t->id);
    g.graph.ApplySplit(n, z2z3TB->id, z2z3Tb->id);
    g.graph.ApplyReorder(n, {z2z3TB->id, z2z3Tb->id, z2z3t->id});
  }
  // Production captures the SIMT lowering metadata before the tiling axis transform, so the
  // GM side load keeps its logical (z2, z3) view there; restore that view after the transform
  // to keep the fixture rank-consistent with the logical view stamped by AnnotateSimtTemplate.
  auto addend_node = g.graph.FindNode("addend_load");
  addend_node->outputs[0].attr.axis = {z2_id, z3_id};
  addend_node->outputs[0].attr.repeats = {g.s2, g.s3};
  addend_node->outputs[0].attr.strides = {g.s3, One};

  il_node->attr.sched.loop_axis = z2z3Tb->id;
  il_node->outputs[0].attr.vectorized_axis = {z2z3t->id};
  il_node->outputs[0].attr.vectorized_strides = {One};

  output_transform_node->attr.sched.loop_axis = z2z3Tb->id;
  output_transform_node->outputs[0].attr.vectorized_axis = {z2z3t->id};
  output_transform_node->outputs[0].attr.vectorized_strides = {One};

  reduce_node->attr.sched.loop_axis = z2z3Tb->id;
  reduce_node->outputs[0].attr.vectorized_axis = {z2z3t->id};
  reduce_node->outputs[0].attr.vectorized_strides = {One};

  store_node->attr.sched.loop_axis = z2z3Tb->id;
  store_node->outputs[0].attr.vectorized_axis = {z2z3t->id};
  store_node->outputs[0].attr.vectorized_strides = {One};

  if (add_parallel_output) {
    auto direct_store_node = g.graph.FindNode("direct_store");
    direct_store_node->attr.sched.loop_axis = z2z3Tb->id;
    direct_store_node->outputs[0].attr.vectorized_axis = {z2z3t->id};
    direct_store_node->outputs[0].attr.vectorized_strides = {One};
  }

  // ----- Memory allocation -----
  auto x = g.graph.FindNode("x");
  x->outputs[0].attr.mem.tensor_id = 0;
  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  x->outputs[0].attr.mem.position = af::Position::kPositionGM;
  x->outputs[0].attr.que.id = af::kIdNone;

  auto idx = g.graph.FindNode("idx");
  idx->outputs[0].attr.mem.tensor_id = 3;
  idx->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  idx->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  idx->outputs[0].attr.mem.position = af::Position::kPositionGM;
  idx->outputs[0].attr.que.id = af::kIdNone;

  auto addend = g.graph.FindNode("addend");
  addend->outputs[0].attr.mem.tensor_id = 6;
  addend->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  addend->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  addend->outputs[0].attr.mem.position = af::Position::kPositionGM;
  addend->outputs[0].attr.que.id = af::kIdNone;

  auto x_load_node = g.graph.FindNode("x_load");
  x_load_node->outputs[0].attr.mem.tensor_id = 7;
  x_load_node->outputs[0].attr.mem.reuse_id = 7;
  x_load_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  x_load_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  x_load_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  x_load_node->outputs[0].attr.que.id = 3;
  x_load_node->outputs[0].attr.que.depth = 2;
  x_load_node->outputs[0].attr.que.buf_num = 2;

  auto idx_load_node = g.graph.FindNode("idx_load");
  idx_load_node->outputs[0].attr.mem.tensor_id = 8;
  idx_load_node->outputs[0].attr.mem.reuse_id = 8;
  idx_load_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  idx_load_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  idx_load_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  idx_load_node->outputs[0].attr.que.id = 4;
  idx_load_node->outputs[0].attr.que.depth = 2;
  idx_load_node->outputs[0].attr.que.buf_num = 2;

  auto addend_load_node = g.graph.FindNode("addend_load");
  addend_load_node->outputs[0].attr.mem.tensor_id = 9;
  addend_load_node->outputs[0].attr.mem.reuse_id = 9;
  addend_load_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  addend_load_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  addend_load_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  addend_load_node->outputs[0].attr.que.id = 5;
  addend_load_node->outputs[0].attr.que.depth = 2;
  addend_load_node->outputs[0].attr.que.buf_num = 2;

  il_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  il_node->outputs[0].attr.mem.tensor_id = 1;
  il_node->outputs[0].attr.mem.reuse_id = 1;
  il_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  il_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  il_node->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  il_node->outputs[0].attr.que.id = 0;
  il_node->outputs[0].attr.que.depth = 2;
  il_node->outputs[0].attr.que.buf_num = 2;

  output_transform_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  output_transform_node->outputs[0].attr.mem.tensor_id = 5;
  output_transform_node->outputs[0].attr.mem.reuse_id = 5;
  output_transform_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  output_transform_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  output_transform_node->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  output_transform_node->outputs[0].attr.que.id = 2;
  output_transform_node->outputs[0].attr.que.depth = 2;
  output_transform_node->outputs[0].attr.que.buf_num = 2;

  reduce_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  reduce_node->outputs[0].attr.mem.tensor_id = 4;
  reduce_node->outputs[0].attr.mem.reuse_id = 4;
  reduce_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  reduce_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  reduce_node->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  reduce_node->outputs[0].attr.que.id = 1;
  reduce_node->outputs[0].attr.que.depth = 2;
  reduce_node->outputs[0].attr.que.buf_num = 2;

  store_node->outputs[0].attr.mem.tensor_id = 2;
  store_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  store_node->outputs[0].attr.mem.position = af::Position::kPositionGM;
  store_node->outputs[0].attr.que.id = af::kIdNone;

  if (add_parallel_output) {
    auto direct_store_node = g.graph.FindNode("direct_store");
    direct_store_node->outputs[0].attr.mem.tensor_id = 10;
    direct_store_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
    direct_store_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
    direct_store_node->outputs[0].attr.mem.position = af::Position::kPositionGM;
    direct_store_node->outputs[0].attr.que.id = af::kIdNone;
  }
}

// ======================== Template annotation ========================

void AnnotateSimdTemplate(af::AscGraph &graph, Implementation implementation = Implementation::kDefault) {
  auto il = graph.FindNode("indirect_load");
  ASSERT_NE(il, nullptr);
  ASSERT_EQ(::ascir::SetTemplateId(il, ::ascir::TemplateId::kIndirectLoadSimd), af::SUCCESS);
  ASSERT_EQ(SetImplementation(il, implementation), af::SUCCESS);

  auto axes = graph.GetAllAxis();
  af::AxisId outer_id = af::kIdNone, inner_id = af::kIdNone, input_inner_id = af::kIdNone;
  for (const auto &axis : axes) {
    if (axis == nullptr) continue;
    if (axis->name == "z2z3TB") outer_id = axis->id;
    if (axis->name == "z2z3t") inner_id = axis->id;
    if (axis->name == "z2z3t") input_inner_id = axis->id;
  }
  ASSERT_NE(outer_id, af::kIdNone);
  ASSERT_NE(inner_id, af::kIdNone);

  TemplateAxes t{outer_id, inner_id, input_inner_id};
  ASSERT_EQ(SetTemplateAxes(il, t), af::SUCCESS);

  const af::Expression One = af::sym::kSymbolOne;
  TemplateLogicalView view;
  view.input.axis_ids = {axes[0]->id, axes[1]->id};
  view.input.sizes = {graph.FindAxis(axes[0]->id)->size, graph.FindAxis(axes[1]->id)->size};
  view.input.strides = {graph.FindAxis(axes[1]->id)->size, One};
  view.index.axis_ids = {axes[2]->id, axes[3]->id};
  view.index.sizes = {graph.FindAxis(axes[2]->id)->size, graph.FindAxis(axes[3]->id)->size};
  view.index.strides = {graph.FindAxis(axes[3]->id)->size, One};
  view.output.axis_ids = {axes[2]->id, axes[3]->id};
  view.output.sizes = {graph.FindAxis(axes[2]->id)->size, graph.FindAxis(axes[3]->id)->size};
  view.output.strides = {graph.FindAxis(axes[3]->id)->size, One};
  ASSERT_EQ(ClassifyIndirectLoadLayout(view.input, view.input), af::SUCCESS);
  ASSERT_EQ(ClassifyIndirectLoadLayout(view.index, view.index), af::SUCCESS);
  ASSERT_EQ(SetTemplateLogicalView(il, view), af::SUCCESS);
}

void AnnotateSimtTemplate(af::AscGraph &graph) {
  auto il = graph.FindNode("indirect_load");
  ASSERT_NE(il, nullptr);
  ASSERT_EQ(::ascir::SetTemplateId(il, ::ascir::TemplateId::kIndirectLoadSimt), af::SUCCESS);
  ASSERT_EQ(SetImplementation(il, Implementation::kDefault), af::SUCCESS);
  const auto input_load = graph.FindNode("x_load");
  const auto index_load = graph.FindNode("idx_load");
  const auto store = graph.FindNode("store");
  ASSERT_NE(input_load, nullptr);
  ASSERT_NE(index_load, nullptr);
  ASSERT_NE(store, nullptr);
  ASSERT_EQ(SetTemplateRole(input_load, TemplateRole::kSimtInputBoundary), af::SUCCESS);
  ASSERT_EQ(SetTemplateRole(index_load, TemplateRole::kSimtDirectGmBoundary), af::SUCCESS);
  ASSERT_EQ(SetTemplateRole(store, TemplateRole::kSimtDirectGmBoundary), af::SUCCESS);
  const auto second_store = graph.FindNode("second_store");
  if (second_store != nullptr) {
    ASSERT_EQ(SetTemplateRole(second_store, TemplateRole::kSimtDirectGmBoundary), af::SUCCESS);
  }
  const auto index_store = graph.FindNode("index_store");
  if (index_store != nullptr) {
    // 与调度侧 FoldSimtIndexChainExternalStores 的区域标注保持一致。
    ASSERT_EQ(SetTemplateRole(index_store, TemplateRole::kSimtDirectGmBoundary), af::SUCCESS);
  }
  const auto direct_store = graph.FindNode("direct_store");
  if (direct_store != nullptr) {
    ASSERT_EQ(SetTemplateRole(direct_store, TemplateRole::kSimtDirectGmBoundary), af::SUCCESS);
  }
  const auto addend_load = graph.FindNode("addend_load");
  if (addend_load != nullptr) {
    ASSERT_EQ(SetTemplateRole(addend_load, TemplateRole::kSimtDirectGmBoundary), af::SUCCESS);
  }

  // SIMT outer_axis derives from all output axes (z2, z3).
  // Use the merged axis z2z3 which derives from both.
  auto axes = graph.GetAllAxis();
  af::AxisId outer_id = af::kIdNone;
  for (const auto &axis : axes) {
    if (axis == nullptr) continue;
    if (axis->name == "z2z3") outer_id = axis->id;
  }
  ASSERT_NE(outer_id, af::kIdNone);

  TemplateAxes t{outer_id, af::kIdNone, af::kIdNone};
  ASSERT_EQ(SetTemplateAxes(il, t), af::SUCCESS);

  const af::Expression One = af::sym::kSymbolOne;
  TemplateLogicalView view;
  view.input.axis_ids = {axes[0]->id, axes[1]->id};
  view.input.sizes = {graph.FindAxis(axes[0]->id)->size, graph.FindAxis(axes[1]->id)->size};
  view.input.strides = {graph.FindAxis(axes[1]->id)->size, One};
  view.index.axis_ids = {axes[2]->id, axes[3]->id};
  view.index.sizes = {graph.FindAxis(axes[2]->id)->size, graph.FindAxis(axes[3]->id)->size};
  view.index.strides = {graph.FindAxis(axes[3]->id)->size, One};
  view.output.axis_ids = {axes[2]->id, axes[3]->id};
  view.output.sizes = {graph.FindAxis(axes[2]->id)->size, graph.FindAxis(axes[3]->id)->size};
  view.output.strides = {graph.FindAxis(axes[3]->id)->size, One};
  ASSERT_EQ(ClassifyIndirectLoadLayout(view.input, view.input), af::SUCCESS);
  ASSERT_EQ(ClassifyIndirectLoadLayout(view.index, view.index), af::SUCCESS);
  ASSERT_EQ(SetTemplateLogicalView(il, view), af::SUCCESS);
}

ApiCall *FindCallByNodeName(const Loop &loop, const std::string &name) {
  for (const auto &body : loop.bodys) {
    if (body.type == LoopType::CALL && body.call != nullptr && body.call->node->GetName() == name) {
      return body.call;
    }
    if (body.type == LoopType::LOOP && body.loop != nullptr) {
      ApiCall *call = FindCallByNodeName(*body.loop, name);
      if (call != nullptr) {
        return call;
      }
    }
  }
  return nullptr;
}

class SimtCodegenContext {
 public:
  SimtCodegenContext() {
    ge::PlatformContext::GetInstance().Reset();
    ge::PlatformContext::GetInstance().SetPlatform("3510");
    ge::RuntimeStub::SetInstance(std::make_shared<ge::RuntimeStubV2Common>());
  }

  ~SimtCodegenContext() {
    ge::RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
  }
};

class IndirectLoadConstructFromNodesTest : public testing::Test {
 protected:
  SimtCodegenContext context;
};

void SetupTiler(const af::AscGraph &graph, Tiler &tiler) {
  for (const auto &axis : graph.GetAllAxis()) {
    if (axis != nullptr) {
      tiler.AddAxis(*axis);
      tiler.AddSizeVar(af::SizeVar(axis->size));
    }
  }
}

Status ParseGraphAndGenerateLoop(ILTestGraph &g, const std::vector<const char *> &input_names, Kernel &kernel,
                                 std::string &generated, const std::vector<const char *> &output_names = {"y"}) {
  ascir::FusedScheduledResult result;
  for (const char *name : input_names) {
    const auto input = g.graph.FindNode(name);
    GE_ASSERT_NOTNULL(input, "Input node[%s] is missing.", name);
    result.input_nodes.emplace_back(input);
  }
  for (const char *name : output_names) {
    const auto output = g.graph.FindNode(name);
    GE_ASSERT_NOTNULL(output, "Output node[%s] is missing.", name);
    result.output_nodes.emplace_back(output);
  }
  GE_ASSERT_SUCCESS(Kernel::ParseGraph(g.graph, result, kernel));
  return kernel.root_loop.Generate(kernel.tiler, kernel.tpipe, generated);
}

}  // namespace

// ==================== Factory ====================

TEST(IndirectLoadApiCallTest, FactoryCreatesIndirectLoadRegApiCall) {
  auto *handler = ApiCallFactory::Instance().Create("IndirectLoadRegApiCall", "IndirectLoadRegApiCall");
  ASSERT_NE(handler, nullptr);
  auto *il_handler = dynamic_cast<IndirectLoadRegApiCall *>(handler);
  ASSERT_NE(il_handler, nullptr);
  EXPECT_EQ(il_handler->api_name_, "IndirectLoadRegApiCall");
  delete handler;
}

// ==================== SIMD Init + Generate ====================

void GenerateSimdCallFromGraph(ILTestGraph &g, af::DataType input_dtype, Implementation implementation,
                               std::string &result, ge::DataType output_dtype = ge::DT_UNDEFINED,
                               bool expect_success = true, bool padded_views = false, bool delayed_padding = false) {
  BuildSimdGraph(g, input_dtype, padded_views);
  AnnotateSimdTemplate(g.graph, implementation);
  auto il_node = g.graph.FindNode("indirect_load");
  ASSERT_NE(il_node, nullptr);
  if (output_dtype != ge::DT_UNDEFINED) {
    // Must be set before the output tensor is registered in the TPipe.
    il_node->outputs[0].attr.dtype = output_dtype;
  }
  if (delayed_padding) {
    // 模拟通用对齐发生在 lowering metadata 序列化之后：先用紧凑视图定格 dense
    // fallback，再改写向量化视图引入行间 padding，覆盖 codegen 期兜底降级分支。
    bool metadata_supported = false;
    ASSERT_EQ(FinalizeLoweringMetadata(il_node, metadata_supported), af::SUCCESS);
    ASSERT_TRUE(metadata_supported);
    SetPaddedVectorizedViews(g);
  }
  Tiler tiler;
  SetupTiler(g.graph, tiler);
  TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(g.graph);

  tpipe.AddTensor(il_node->inputs[0]);
  tpipe.AddTensor(il_node->inputs[1]);
  tpipe.AddTensor(*il_node->outputs()[0]);

  IndirectLoadRegApiCall call("IndirectLoadRegApiCall");
  ASSERT_EQ(call.Init(il_node), af::SUCCESS);

  // Set up ApiTensors for inputs (outputs already populated by Init)
  ApiTensor x1, x2;
  x1.id = il_node->inputs[0].attr.mem.tensor_id;
  x2.id = il_node->inputs[1].attr.mem.tensor_id;
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::vector<std::reference_wrapper<const Tensor>> input_refs, output_refs;
  for (auto *in : call.inputs) {
    auto *t = tpipe.GetTensor(in->id);
    ASSERT_NE(t, nullptr);
    input_refs.emplace_back(*t);
  }
  for (auto &out : call.outputs) {
    auto *t = tpipe.GetTensor(out.id);
    ASSERT_NE(t, nullptr);
    output_refs.emplace_back(*t);
  }

  std::vector<af::AxisId> current_axis;
  for (const auto &axis : g.graph.GetAllAxis()) {
    if (axis != nullptr) current_axis.emplace_back(axis->id);
  }

  auto status = call.Generate(tpipe, current_axis, input_refs, output_refs, result);
  if (expect_success) {
    ASSERT_EQ(status, af::SUCCESS);
  } else {
    EXPECT_NE(status, af::SUCCESS);
  }
}

void GenerateSimdCall(af::DataType input_dtype, std::string &result) {
  ILTestGraph g("simd_gen");
  GenerateSimdCallFromGraph(g, input_dtype, Implementation::kDefault, result);
}

TEST(IndirectLoadApiCallTest, GenerateSimdProducesIndirectLoadSimdCall) {
  std::string result;
  GenerateSimdCall(ge::DT_FLOAT16, result);
  EXPECT_NE(result.find("// IndirectLoad SIMD"), std::string::npos);
  // 符号 strides 可证连续的动态 shape 视图判 dense，走向量化 RegisterGather
  // facade（此前动态 shape 一律保守判 strided，需要临时 UB 缓冲逐元素组装）。
  EXPECT_NE(result.find("IndirectLoadSimd<half, int32_t, 2, 1>"), std::string::npos);
  EXPECT_EQ(result.find("IndirectLoadSimdStrided"), std::string::npos);
  EXPECT_EQ(result.find("tmp_buf_0"), std::string::npos);
  EXPECT_NE(result.find("indirect_load_simd_params{static_cast<uint32_t>("), std::string::npos);
  EXPECT_NE(result.find("), static_cast<uint32_t>("), std::string::npos);
}

TEST(IndirectLoadApiCallTest, GenerateSimdUint32InputProducesTypedCall) {
  std::string result;
  GenerateSimdCall(ge::DT_UINT32, result);
  EXPECT_NE(result.find("IndirectLoadSimd<uint32_t, int32_t, 2, 1>"), std::string::npos);
}

TEST(IndirectLoadApiCallTest, GenerateSimdDenseSelectsDenseApi) {
  ILTestGraph g("simd_dense", 2, 8, 2, 8);
  std::string result;
  GenerateSimdCallFromGraph(g, ge::DT_FLOAT16, Implementation::kDefault, result);

  EXPECT_NE(result.find("IndirectLoadSimd<half, int32_t, 2, 1>"), std::string::npos);
  EXPECT_EQ(result.find("IndirectLoadSimdStrided"), std::string::npos);
  EXPECT_NE(result.find("IndirectLoadSimdFallback>(0), false>"), std::string::npos);
  EXPECT_NE(result.find("IndirectLoadSimdParams<IndirectLoadSimdCase, 2> indirect_load_simd_params"),
            std::string::npos);
}

TEST(IndirectLoadApiCallTest, GenerateSimdDenseGatherSelectsGatherApi) {
  ILTestGraph g("simd_gather", 2, 8, 2, 8);
  std::string result;
  GenerateSimdCallFromGraph(g, ge::DT_FLOAT16, Implementation::kGatherApi, result);

  EXPECT_NE(result.find("IndirectLoadSimdGatherApi<half, int32_t, 2, 1>"), std::string::npos);
  EXPECT_EQ(result.find("IndirectLoadSimdStrided"), std::string::npos);
}

TEST(IndirectLoadApiCallTest, GenerateSimdPaddedVectorizedViewsDowngradeToStrided) {
  // 变位宽生产者（cast）的通用对齐会在行间引入物理 padding：行轴与尾轴两维向量化视图
  // 的行步长无法静态证明等于逻辑紧凑步长时，dense facade（RegGather）按紧凑布局读索引
  // 会越过真实数据（越界读 UB），必须降级 strided facade，并按物理步长传索引布局。
  ILTestGraph g("simd_padded_views");
  std::string result;
  GenerateSimdCallFromGraph(g, ge::DT_FLOAT16, Implementation::kDefault, result, ge::DT_UNDEFINED, true, true);
  EXPECT_NE(result.find("IndirectLoadSimdStrided"), std::string::npos);
  EXPECT_EQ(result.find("IndirectLoadSimdFallback>(0)"), std::string::npos);
  // 索引窗口步长必须替换为张量物理视图步长（此处逻辑紧凑为 {1, 1}，物理视图后代轴
  // 步长为 3）：strided facade 按真实布局读索引，否则行漂移读进 padding 区。
  EXPECT_NE(result.find("{3, 3}"), std::string::npos);
}

TEST(IndirectLoadApiCallTest, GenerateSimdDelayedPaddingDowngradesAtCodegen) {
  // lowering metadata 序列化时视图仍紧凑（定格 dense fallback），通用对齐在其后改写
  // 向量化视图引入 padding：codegen 期兜底必须降级 strided 并按物理步长寻址。
  ILTestGraph g("simd_delayed_padding");
  std::string result;
  GenerateSimdCallFromGraph(g, ge::DT_FLOAT16, Implementation::kDefault, result, ge::DT_UNDEFINED, true, false, true);

  EXPECT_NE(result.find("IndirectLoadSimdStrided"), std::string::npos);
  EXPECT_EQ(result.find("IndirectLoadSimdFallback>(0)"), std::string::npos);
  EXPECT_NE(result.find("{3, 3}"), std::string::npos);
}

TEST(IndirectLoadApiCallTest, FinalizeSimdMetadataPaddedViewsForcesStrided) {
  // lowering 期判定与 codegen 期兜底同向：输出或索引视图可能带行间 padding 时，
  // fallback 必须在元数据序列化阶段就定格为 strided。
  ILTestGraph g("simd_padded_metadata");
  BuildSimdGraph(g, ge::DT_FLOAT16, true);
  AnnotateSimdTemplate(g.graph);
  auto il_node = g.graph.FindNode("indirect_load");
  ASSERT_NE(il_node, nullptr);
  bool metadata_supported = false;
  ASSERT_EQ(FinalizeLoweringMetadata(il_node, metadata_supported), af::SUCCESS);
  ASSERT_TRUE(metadata_supported);
  IndirectLoadLoweringMetadata metadata;
  ASSERT_EQ(GetLoweringMetadata(il_node, metadata), af::SUCCESS);
  EXPECT_EQ(metadata.simd.fallback, SimdFallback::kStrided);
}

TEST(IndirectLoadApiCallTest, GenerateSimdRejectsInputOutputDtypeMismatch) {
  ILTestGraph g("simd_dtype_mismatch");
  std::string result;
  GenerateSimdCallFromGraph(g, ge::DT_FLOAT16, Implementation::kDefault, result, ge::DT_FLOAT, false);
}

// ==================== SIMT Init + GenerateFuncDefinition ====================

void GenerateSimtFuncDefinitionFromGraph(ILTestGraph &g, std::string &definition) {
  Tiler tiler;
  SetupTiler(g.graph, tiler);
  TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(g.graph);

  auto il_node = g.graph.FindNode("indirect_load");
  ASSERT_NE(il_node, nullptr);

  IndirectLoadRegApiCall call("IndirectLoadRegApiCall");
  ASSERT_EQ(call.Init(il_node), af::SUCCESS);

  ApiTensor x1, x2;
  x1.id = il_node->inputs[0].attr.mem.tensor_id;
  x2.id = il_node->inputs[1].attr.mem.tensor_id;
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  tpipe.AddTensor(il_node->inputs[0]);
  tpipe.AddTensor(il_node->inputs[1]);

  std::stringstream ss;
  ASSERT_EQ(call.GenerateFuncDefinition(tpipe, tiler, ss), af::SUCCESS);
  definition = ss.str();
}

void GenerateSimtFuncDefinition(ILTestGraph &g, std::string &definition, bool add_second_output = false,
                                bool add_index_store = false) {
  SimtCodegenContext context;
  BuildSimtGraph(g, add_second_output, add_index_store);
  AnnotateSimtTemplate(g.graph);
  GenerateSimtFuncDefinitionFromGraph(g, definition);
}

void GenerateSimtPostReduceFuncDefinitionFromGraph(ILTestGraph &g, std::string &definition) {
  SimtCodegenContext context;
  AnnotateSimtTemplate(g.graph);
  auto output_transform = g.graph.FindNode("output_transform");
  ASSERT_NE(output_transform, nullptr);
  ASSERT_EQ(SetTemplateRole(output_transform, TemplateRole::kSimtInlineTransform), af::SUCCESS);
  GenerateSimtFuncDefinitionFromGraph(g, definition);
}

void GenerateSimtPostReduceFuncDefinition(ILTestGraph &g, std::string &definition, bool add_parallel_output = false) {
  BuildSimtPostReduceGraph(g, add_parallel_output);
  GenerateSimtPostReduceFuncDefinitionFromGraph(g, definition);
}

TEST(IndirectLoadApiCallTest, GenerateFuncDefinitionSimtProducesBodyStruct) {
  ILTestGraph g("simt_gen");
  std::string def;
  GenerateSimtFuncDefinition(g, def);

  EXPECT_NE(def.find("struct IndirectLoadSimtContext_"), std::string::npos);
  EXPECT_NE(def.find("struct IndirectLoadSimtBody_"), std::string::npos);
  EXPECT_NE(def.find("using Context = IndirectLoadSimtContext_"), std::string::npos);
  EXPECT_NE(def.find("__simt_callee__ __aicore__ inline static"), std::string::npos);
  EXPECT_NE(def.find(" Index(uint64_t output_index, uint64_t index_offset, const Context &context)"),
            std::string::npos);
  EXPECT_NE(def.find(" Outputs("), std::string::npos);
}

TEST(IndirectLoadApiCallTest, GenerateFuncDefinitionGuardsContextAcrossVariants) {
  // 同名节点的多个 impl 会拼接到同一源文件；Context 必须独立于 Body 防重，
  // 避免 Body 的 32/64 位双实例各自带 guard 时遗漏共享 Context。
  const std::string context_guard = "AUTOFUSE_INDIRECT_LOAD_SIMT_CONTEXT_DEFINED_indirect_load";
  const std::string context_begin =
      "#ifndef " + context_guard + "\n#define " + context_guard + "\nstruct IndirectLoadSimtContext_indirect_load {\n";
  for (bool static_shape : {false, true}) {
    ILTestGraph first("simt_variant", 2, 8, 2, 8);
    ILTestGraph second("simt_nddma_variant", 2, 8, 2, 8);
    ILTestGraph dynamic_first("simt_dynamic_variant");
    ILTestGraph dynamic_second("simt_dynamic_nddma_variant");
    std::string first_def;
    std::string second_def;
    GenerateSimtFuncDefinition(static_shape ? first : dynamic_first, first_def);
    GenerateSimtFuncDefinition(static_shape ? second : dynamic_second, second_def);
    for (const auto &definition : {first_def, second_def}) {
      const auto begin = definition.find(context_begin);
      ASSERT_NE(begin, std::string::npos);
      const auto end = definition.find("};\n#endif\n", begin);
      ASSERT_NE(end, std::string::npos);
      // Context 的 guard 先闭合，之后才是独立的 Body guard。
      const auto body_guard = definition.find("#ifndef AUTOFUSE_INDIRECT_LOAD_SIMT_BODY_DEFINED_", end);
      ASSERT_NE(body_guard, std::string::npos);
      EXPECT_LT(end, body_guard);
    }
  }
}

TEST(IndirectLoadApiCallTest, GenerateFuncDefinitionSimtEmitsMultiOutputEvaluator) {
  ILTestGraph g("simt_multi_output");
  std::string def;
  GenerateSimtFuncDefinition(g, def, true);

  EXPECT_NE(def.find("struct OutputPack"), std::string::npos);
  EXPECT_NE(def.find("half output0"), std::string::npos);
  EXPECT_NE(def.find("half output1"), std::string::npos);
  EXPECT_NE(def.find("struct OutputTargets"), std::string::npos);
  EXPECT_NE(def.find("__gm__ half *output0"), std::string::npos);
  EXPECT_NE(def.find("__gm__ half *output1"), std::string::npos);
  EXPECT_NE(def.find("outputs.output0 = value"), std::string::npos);
  EXPECT_NE(def.find("outputs.output1 = value"), std::string::npos);
  EXPECT_NE(def.find("targets.output0[output_index] = outputs.output0"), std::string::npos);
  EXPECT_NE(def.find("targets.output1[output_index] = outputs.output1"), std::string::npos);
}

TEST(IndirectLoadApiCallTest, GenerateFuncDefinitionUsesUint32OffsetsForStaticShapes) {
  ILTestGraph g("simt_static_power_of_two", 2, 8, 2, 8);
  std::string def;
  GenerateSimtFuncDefinition(g, def);

  EXPECT_NE(def.find("uint32_t output_index"), std::string::npos);
}

// [Bug A] index 生产链尾部的区域外直连 store 收为空节点输出链，链值由 Index()
// 求值器结果注入；gather 值链仍走 "value" 种子。
TEST(IndirectLoadApiCallTest, GenerateSimtFuncDefinitionEmitsIndexChainStore) {
  ILTestGraph g("simt_index_store");
  std::string def;
  GenerateSimtFuncDefinition(g, def, false, true);

  EXPECT_NE(def.find("struct OutputPack"), std::string::npos);
  EXPECT_NE(def.find("half output0"), std::string::npos);
  EXPECT_NE(def.find("int32_t output1"), std::string::npos);
  EXPECT_NE(def.find("struct OutputTargets"), std::string::npos);
  EXPECT_NE(def.find("__gm__ half *output0"), std::string::npos);
  EXPECT_NE(def.find("__gm__ int32_t *output1"), std::string::npos);
  EXPECT_NE(def.find("outputs.output0 = value"), std::string::npos);
  EXPECT_NE(def.find("outputs.output1 = Index(output_index, index_offset, context);"), std::string::npos);
  EXPECT_NE(def.find("targets.output0[output_index] = outputs.output0"), std::string::npos);
  EXPECT_NE(def.find("targets.output1[output_index] = outputs.output1"), std::string::npos);
}

// [Bug B] index load 物理视图与输出逻辑视图 dense 等价时保持线性偏移快路径。
TEST(IndirectLoadApiCallTest, GenerateSimtIndexLoadKeepsLinearOffsetForDenseEquivalentView) {
  ILTestGraph g("simt_dense_index_offset", 2, 8, 2, 8);
  std::string def;
  GenerateSimtFuncDefinition(g, def);

  EXPECT_NE(def.find("context.gm_5[index_offset]"), std::string::npos);
  EXPECT_EQ(def.find("index_coord_0"), std::string::npos);
}

// [Bug B] index load 物理视图非 dense 等价（行 stride 放大 2 倍的列切片形态）时，
// 线性偏移会丢行 stride，必须走坐标折叠重建。
TEST(IndirectLoadApiCallTest, GenerateSimtIndexLoadFoldsNonDenseEquivalentView) {
  SimtCodegenContext context;
  ILTestGraph g("simt_strided_index_offset", 2, 8, 2, 8);
  BuildSimtGraph(g);
  const auto idx_load = g.graph.FindNode("idx_load");
  ASSERT_NE(idx_load, nullptr);
  idx_load->outputs[0].attr.repeats = {g.s2, g.s3};
  idx_load->outputs[0].attr.strides = {g.s3 * af::Symbol(2), af::Symbol(3)};
  AnnotateSimtTemplate(g.graph);
  std::string def;
  GenerateSimtFuncDefinitionFromGraph(g, def);

  EXPECT_NE(def.find("index_coord_0"), std::string::npos);
  EXPECT_NE(def.find("context.gm_5[index_coord_0"), std::string::npos);
  EXPECT_EQ(def.find("context.gm_5[index_offset]"), std::string::npos);
}

// [Bug B 门控] 逻辑 index 视图与 load 物理步长一致（kStrided policy 已把 index 物理
// 步长嵌入 index_offset）时，即使视图对输出非 dense 等价，也必须保持线性偏移快路径
// （再做坐标折叠会双重乘 stride）。
TEST(IndirectLoadApiCallTest, GenerateSimtIndexLoadKeepsLinearOffsetWhenPolicyEmbedsStrides) {
  SimtCodegenContext context;
  ILTestGraph g("simt_strided_index_policy_match", 2, 8, 2, 8);
  BuildSimtGraph(g);
  const af::Expression row_stride = g.s3 * af::Symbol(2);
  const auto idx_load = g.graph.FindNode("idx_load");
  ASSERT_NE(idx_load, nullptr);
  idx_load->outputs[0].attr.repeats = {g.s2, g.s3};
  idx_load->outputs[0].attr.strides = {row_stride, af::sym::kSymbolOne};
  AnnotateSimtTemplate(g.graph);
  const auto il = g.graph.FindNode("indirect_load");
  ASSERT_NE(il, nullptr);
  TemplateLogicalView view;
  ASSERT_EQ(GetTemplateLogicalView(il, view), af::SUCCESS);
  view.index.strides = {row_stride, af::sym::kSymbolOne};
  ASSERT_EQ(ClassifyIndirectLoadLayout(view.index, view.index), af::SUCCESS);
  ASSERT_EQ(SetTemplateLogicalView(il, view), af::SUCCESS);
  std::string def;
  GenerateSimtFuncDefinitionFromGraph(g, def);

  EXPECT_NE(def.find("context.gm_5[index_offset]"), std::string::npos);
  EXPECT_EQ(def.find("index_coord_0"), std::string::npos);
}

// [Bug B 端到端] 调度器把 strided index load 的视图 split 到模板轴空间后（rank 高于
// 逻辑输出、含退化轴与切分残轴，如 [128]/[39] 被改写为 [128/Tb, Tb, 1]/[39*Tb, 39, 39]），
// 坐标折叠路径必须先做符号化扁平化（去退化轴 + 相邻轴合并）再重建坐标。
TEST(IndirectLoadApiCallTest, GenerateSimtIndexLoadFoldsSplitStridedView) {
  SimtCodegenContext context;
  ILTestGraph g("simt_split_strided_index_offset", 2, 8, 2, 8);
  BuildSimtGraph(g);
  const auto idx_load = g.graph.FindNode("idx_load");
  ASSERT_NE(idx_load, nullptr);
  // 模拟调度器 split 改写后的 rank-4 strided 视图：有效轴 [s2, s3]（行 stride 16），
  // 尾部带 size==1 的切分残轴。扁平化应还原为 {s2, s3}/{16, 1} 后走坐标折叠。
  idx_load->outputs[0].attr.repeats = {g.s2, af::Symbol(4), af::Symbol(2), af::ops::One};
  idx_load->outputs[0].attr.strides = {g.s3 * af::Symbol(2), af::Symbol(2), af::sym::kSymbolOne, af::sym::kSymbolOne};
  AnnotateSimtTemplate(g.graph);
  std::string def;
  GenerateSimtFuncDefinitionFromGraph(g, def);

  EXPECT_NE(def.find("index_coord_0"), std::string::npos);
  EXPECT_NE(def.find("context.gm_5[index_coord_0"), std::string::npos);
  EXPECT_EQ(def.find("context.gm_5[index_offset]"), std::string::npos);
}

// [符号稠密尾段] 动态 shape 图的 index 源 Load 经调度 split 改写为 [1,1,s3]/[0,0,1]
// （前导零贡献 + 符号尾段，稠密维对齐输出尾维 z3）：取模经 rt_size_N 字段 + magic
// 乘移展开，且除法在宽类型上求值后截断回 offset_type（UintDiv 无混参，P1）。
// [R5 收紧] 断言 magic 与宽类型被积元出现在同一表达式（UintDiv 三实参同型的直接
// 证据）；[R1 结构性结论] Tier 尺寸相等使"符号尾段 ⟹ 输出符号维 ⟹ 档位恒 64"，
// 32 位档（静态 shape）下尾段必为常量、走字面量取模不出现 UintDiv（见
// ConstDenseTail 用例的 npos 断言）——"32 位 + 符号尾段混参"场景被结构性排除，
// wide_type 包裹为纵深防御（覆盖双实例等旁路）。
TEST(IndirectLoadApiCallTest, GenerateSimtIndexLoadSymbolicDenseTailUsesRuntimeModulo) {
  SimtCodegenContext context;
  ILTestGraph g("simt_symbolic_dense_tail");
  BuildSimtGraph(g);
  const auto idx_load = g.graph.FindNode("idx_load");
  ASSERT_NE(idx_load, nullptr);
  idx_load->outputs[0].attr.axis = {g.z2.id, g.z2.id, g.z3.id};
  idx_load->outputs[0].attr.repeats = {af::sym::kSymbolOne, af::sym::kSymbolOne, g.s3};
  idx_load->outputs[0].attr.strides = {af::Symbol(0), af::Symbol(0), af::sym::kSymbolOne};
  AnnotateSimtTemplate(g.graph);
  std::string def;
  GenerateSimtFuncDefinitionFromGraph(g, def);

  EXPECT_NE(def.find("Simt::UintDiv(static_cast<uint64_t>(output_index), context.rt_div_magic_0"), std::string::npos);
  EXPECT_NE(def.find("static_cast<uint64_t>(output_index)"), std::string::npos);
  EXPECT_EQ(def.find("index_coord_0"), std::string::npos);
}

// [常量尾段回归·兼 32 位档锁] 静态 shape 的 [1,1,8]/[0,0,1]（softmax load4 形态，
// 稠密维对齐输出尾维 z3、尺寸相等）：tier 校验放行，字面量取模路径行为不变；且
// 静态 shape → offset 档位为 uint32——UintDiv 全程不出现，即 32 位档无任何运行期
// 除法（P1 混参场景在该档位不可达的直接锁定）。
TEST(IndirectLoadApiCallTest, GenerateSimtIndexLoadConstDenseTailKeepsLiteralModulo) {
  SimtCodegenContext context;
  ILTestGraph g("simt_const_dense_tail", 2, 8, 2, 8);
  BuildSimtGraph(g);
  const auto idx_load = g.graph.FindNode("idx_load");
  ASSERT_NE(idx_load, nullptr);
  idx_load->outputs[0].attr.axis = {g.z2.id, g.z2.id, g.z3.id};
  idx_load->outputs[0].attr.repeats = {af::sym::kSymbolOne, af::sym::kSymbolOne, g.s3};
  idx_load->outputs[0].attr.strides = {af::Symbol(0), af::Symbol(0), af::sym::kSymbolOne};
  AnnotateSimtTemplate(g.graph);
  std::string def;
  GenerateSimtFuncDefinitionFromGraph(g, def);

  EXPECT_NE(def.find("% 8"), std::string::npos);
  EXPECT_EQ(def.find("UintDiv"), std::string::npos);
}

// [尾侧单例·轴对齐分解] [1,s3,1]/[0,1,5]（尾侧真实 size-1 维、stride!=0，物理存在
// 但坐标恒 0）：单例维不贡献地址且位置不限，稠密维 z3 经输出 stride 反解坐标——
// 不再触发 rank mismatch 断言（P3 精确分类的落地）。
TEST(IndirectLoadApiCallTest, GenerateSimtIndexLoadTrailingSingletonAxesAligned) {
  SimtCodegenContext context;
  ILTestGraph g("simt_trailing_singleton");
  BuildSimtGraph(g);
  const auto idx_load = g.graph.FindNode("idx_load");
  ASSERT_NE(idx_load, nullptr);
  idx_load->outputs[0].attr.axis = {g.z2.id, g.z3.id, g.z0.id};
  idx_load->outputs[0].attr.repeats = {af::sym::kSymbolOne, g.s3, af::sym::kSymbolOne};
  idx_load->outputs[0].attr.strides = {af::Symbol(0), af::sym::kSymbolOne, af::Symbol(5)};
  AnnotateSimtTemplate(g.graph);
  std::string def;
  GenerateSimtFuncDefinitionFromGraph(g, def);

  EXPECT_NE(def.find("index_coord_0"), std::string::npos);
  EXPECT_NE(def.find("context.gm_5[index_coord_0"), std::string::npos);
}

// [稠密维在前导轴·不被取模窃占] [1,s2,1]/[0,s3,0]（稠密维 z2=输出 dim0、stride 为行
// 宽 s3 非 1）：简单取模被结构性拒绝（unit-stride + 尾侧块 + tier 校验三重门，否则
// % s2 静默错址——P2），轴对齐分解按输出 stride 反解（output_index / s3 取 s2 坐标
// 后乘行宽）。
TEST(IndirectLoadApiCallTest, GenerateSimtIndexLoadLeadingDenseNotStolenByModulo) {
  SimtCodegenContext context;
  ILTestGraph g("simt_leading_dense");
  BuildSimtGraph(g);
  const auto idx_load = g.graph.FindNode("idx_load");
  ASSERT_NE(idx_load, nullptr);
  idx_load->outputs[0].attr.axis = {g.z0.id, g.z2.id, g.z3.id};
  idx_load->outputs[0].attr.repeats = {af::sym::kSymbolOne, g.s2, af::sym::kSymbolOne};
  idx_load->outputs[0].attr.strides = {af::Symbol(0), g.s3, af::Symbol(0)};
  AnnotateSimtTemplate(g.graph);
  std::string def;
  GenerateSimtFuncDefinitionFromGraph(g, def);

  EXPECT_NE(def.find("index_coord_0"), std::string::npos);
  EXPECT_NE(def.find("context.gm_5[index_coord_0"), std::string::npos);
  EXPECT_EQ(def.find("index_offset]"), std::string::npos);
}

// [双稠密维 + 尾侧退化维·dense 等价快路径] [s2,s3,1]/[s3,1,0]：尾维 size==1 且
// stride==0 为退化维，IsDenseEquivalentView 丢弃后 total s2*s3 == 输出 total——
// dense 等价成立，地址 = i*s3 + j = output_index，线性 index_offset 是正确快路径
// （与 GenerateSimtIndexLoadKeepsLinearOffsetForDenseEquivalentView 同机制），不进
// 轴对齐分解。
TEST(IndirectLoadApiCallTest, GenerateSimtIndexLoadTwoDenseDimsTrailingDegenerateKeepsLinearOffset) {
  SimtCodegenContext context;
  ILTestGraph g("simt_two_dense_trailing_degenerate");
  BuildSimtGraph(g);
  const auto idx_load = g.graph.FindNode("idx_load");
  ASSERT_NE(idx_load, nullptr);
  idx_load->outputs[0].attr.axis = {g.z2.id, g.z3.id, g.z0.id};
  idx_load->outputs[0].attr.repeats = {g.s2, g.s3, af::sym::kSymbolOne};
  idx_load->outputs[0].attr.strides = {g.s3, af::sym::kSymbolOne, af::Symbol(0)};
  AnnotateSimtTemplate(g.graph);
  std::string def;
  GenerateSimtFuncDefinitionFromGraph(g, def);

  EXPECT_NE(def.find("context.gm_5[index_offset]"), std::string::npos);
  EXPECT_EQ(def.find("index_coord_0"), std::string::npos);
}

// [rank 相等·前导零贡献] [1,s3]/[1,1]（rank 2==2，前导 size==1 维、无零 stride——
// 含零 stride 的 [0,1] 形态会被 ApplySimtIndexPhysicalStrides 嵌入物理步长走 kStrided
// 线性快路径，见 KeepsLinearOffsetWhenPolicyEmbedsStrides）：不进 rank 失配分支，
// 坐标循环原生跳过零贡献维并按输出 stride 反解——锁定该形态经坐标循环正确处理。
TEST(IndirectLoadApiCallTest, GenerateSimtIndexLoadRankEqualLeadingBroadcastUsesCoordinateLoop) {
  SimtCodegenContext context;
  ILTestGraph g("simt_rank_equal_leading_bcast");
  BuildSimtGraph(g);
  const auto idx_load = g.graph.FindNode("idx_load");
  ASSERT_NE(idx_load, nullptr);
  idx_load->outputs[0].attr.axis = {g.z2.id, g.z3.id};
  idx_load->outputs[0].attr.repeats = {af::sym::kSymbolOne, g.s3};
  idx_load->outputs[0].attr.strides = {af::sym::kSymbolOne, af::sym::kSymbolOne};
  AnnotateSimtTemplate(g.graph);
  std::string def;
  GenerateSimtFuncDefinitionFromGraph(g, def);

  EXPECT_NE(def.find("index_coord_0"), std::string::npos);
  EXPECT_NE(def.find("context.gm_5[index_coord_0"), std::string::npos);
  EXPECT_EQ(def.find("context.gm_5[index_offset]"), std::string::npos);
}

void InsertSimtIdentityBroadcast(ILTestGraph &g, const char *name, const char *producer_name, const char *consumer_name,
                                 int32_t input_index, ascir::TensorId tensor_id) {
  const auto producer = g.graph.FindNode(producer_name);
  const auto consumer = g.graph.FindNode(consumer_name);
  ASSERT_NE(producer, nullptr);
  ASSERT_NE(consumer, nullptr);
  Broadcast broadcast(name);
  g.graph.AddNode(broadcast);
  const auto node = g.graph.FindNode(name);
  ASSERT_NE(node, nullptr);
  node->attr.sched.axis = producer->attr.sched.axis;
  node->attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  node->outputs[0].attr.dtype = producer->outputs[0].attr.dtype;
  node->outputs[0].attr.axis = producer->outputs[0].attr.axis;
  node->outputs[0].attr.repeats = producer->outputs[0].attr.repeats;
  node->outputs[0].attr.strides = producer->outputs[0].attr.strides;
  node->outputs[0].attr.mem.tensor_id = tensor_id;
  ASSERT_EQ(SetTemplateRole(node, TemplateRole::kSimtInlineTransform), af::SUCCESS);
  ASSERT_EQ(af::GraphUtils::ReplaceEdgeSrc(producer->GetOutDataAnchor(0), consumer->GetInDataAnchor(input_index),
                                           node->GetOutDataAnchor(0)),
            af::SUCCESS);
  ASSERT_EQ(af::GraphUtils::AddEdge(producer->GetOutDataAnchor(0), node->GetInDataAnchor(0)), af::SUCCESS);
  ASSERT_EQ(af::AscGraphUtils::GetComputeGraph(g.graph)->TopologicalSorting(), af::SUCCESS);
}

TEST_F(IndirectLoadConstructFromNodesTest, OmitsIdentityScalarTemporaries) {
  ILTestGraph g("simt_identity_chain", 2, 8, 2, 8);
  BuildSimtGraph(g);
  ASSERT_NO_FATAL_FAILURE(InsertSimtIdentityBroadcast(g, "identity0", "indirect_load", "store", 0, 20));
  ASSERT_NO_FATAL_FAILURE(InsertSimtIdentityBroadcast(g, "identity1", "identity0", "store", 0, 21));
  AnnotateSimtTemplate(g.graph);
  std::string definition;
  GenerateSimtFuncDefinitionFromGraph(g, definition);
  EXPECT_EQ(definition.find("v_20 ="), std::string::npos);
  EXPECT_EQ(definition.find("v_21 ="), std::string::npos);
  EXPECT_NE(definition.find("outputs.output0 = value;"), std::string::npos);
}

TEST_F(IndirectLoadConstructFromNodesTest, KeepsGmLoadMaterializedThroughIdentityBroadcast) {
  ILTestGraph g("simt_identity_gm_load");
  BuildSimtPostReduceGraph(g);
  ASSERT_NO_FATAL_FAILURE(InsertSimtIdentityBroadcast(g, "load_identity", "addend_load", "output_transform", 1, 50));
  std::string definition;
  GenerateSimtPostReduceFuncDefinitionFromGraph(g, definition);
  const auto load = definition.find("context.gm_9[");
  ASSERT_NE(load, std::string::npos);
  EXPECT_EQ(definition.find("context.gm_9[", load + 1), std::string::npos);
}

TEST_F(IndirectLoadConstructFromNodesTest, InlinesCompileTimeScalarWithoutTemporary) {
  ILTestGraph g("simt_inline_scalar");
  BuildSimtPostReduceGraph(g);
  Scalar scalar("constant_bias", g.graph);
  scalar.ir_attr.SetValue("2.5");
  scalar.y.dtype = af::DT_FLOAT16;
  const auto scalar_node = g.graph.FindNode("constant_bias");
  const auto load = g.graph.FindNode("addend_load");
  const auto add = g.graph.FindNode("output_transform");
  ASSERT_NE(scalar_node, nullptr);
  ASSERT_NE(load, nullptr);
  ASSERT_NE(add, nullptr);
  scalar_node->outputs[0].attr.mem.tensor_id = 40;
  ASSERT_EQ(af::GraphUtils::ReplaceEdgeSrc(load->GetOutDataAnchor(0), add->GetInDataAnchor(1),
                                           scalar_node->GetOutDataAnchor(0)),
            af::SUCCESS);
  ASSERT_EQ(af::AscGraphUtils::GetComputeGraph(g.graph)->TopologicalSorting(), af::SUCCESS);
  std::string definition;
  GenerateSimtPostReduceFuncDefinitionFromGraph(g, definition);
  EXPECT_EQ(definition.find("v_40 ="), std::string::npos);
  EXPECT_NE(definition.find("static_cast<half>(2.5"), std::string::npos);
}

TEST(IndirectLoadApiCallTest, GenerateSimtOutputMapsBroadcastLoadToColumn) {
  ILTestGraph g("simt_post_broadcast_offset", 2, 8, 2, 8);
  BuildSimtPostReduceGraph(g);
  // Reuse the side-input fixture with a direct Store, matching the position-bias output chain.
  const auto reduce = g.graph.FindNode("reduce");
  ASSERT_NE(reduce, nullptr);
  ASSERT_EQ(af::GraphUtils::IsolateNodeOneIO(reduce), af::SUCCESS);
  ASSERT_EQ(af::GraphUtils::RemoveNodeWithoutRelink(af::AscGraphUtils::GetComputeGraph(g.graph), reduce), af::SUCCESS);
  const auto addend_load = g.graph.FindNode("addend_load");
  ASSERT_NE(addend_load, nullptr);
  addend_load->outputs[0].attr.repeats = {af::sym::kSymbolOne, g.s3};
  addend_load->outputs[0].attr.strides = {af::sym::kSymbolZero, af::sym::kSymbolOne};

  std::string definition;
  GenerateSimtPostReduceFuncDefinitionFromGraph(g, definition);
  EXPECT_NE(definition.find("context.gm_9[index_coord_0]"), std::string::npos);
  // [SIMT 分解共享] 广播侧载维与 gather 轴尺寸对齐时复用 Address 已算的轴坐标
  // （coord_axis == output_index % 8，数值等价），消除独立 magic mod。
  EXPECT_NE(definition.find("index_coord_0 = coord_axis;"), std::string::npos);
  EXPECT_EQ(definition.find("output_index % 8"), std::string::npos);
  EXPECT_EQ(definition.find("context.gm_9[output_index]"), std::string::npos);
  EXPECT_EQ(definition.find("context.gm_9[index_offset]"), std::string::npos);
}

// A post-Reduce SIMT region still addresses every GM side load with the full logical
// output_index, so a broadcast side-input view must keep its coordinate folding there.
TEST(IndirectLoadApiCallTest, GenerateSimtPostReduceMapsBroadcastLoadToColumn) {
  ILTestGraph g("simt_post_reduce_broadcast_offset", 2, 8, 2, 8);
  BuildSimtPostReduceGraph(g);
  const auto addend_load = g.graph.FindNode("addend_load");
  ASSERT_NE(addend_load, nullptr);
  addend_load->outputs[0].attr.repeats = {af::sym::kSymbolOne, g.s3};
  addend_load->outputs[0].attr.strides = {af::sym::kSymbolZero, af::sym::kSymbolOne};

  std::string definition;
  GenerateSimtPostReduceFuncDefinitionFromGraph(g, definition);
  EXPECT_NE(definition.find("context.gm_9[index_coord_0]"), std::string::npos);
  // [SIMT 分解共享] 广播侧载维与 gather 轴尺寸对齐时复用 Address 已算的轴坐标
  // （coord_axis == output_index % 8，数值等价），消除独立 magic mod。
  EXPECT_NE(definition.find("index_coord_0 = coord_axis;"), std::string::npos);
  EXPECT_EQ(definition.find("output_index % 8"), std::string::npos);
  EXPECT_EQ(definition.find("context.gm_9[output_index]"), std::string::npos);
  EXPECT_EQ(definition.find("context.gm_9[index_offset]"), std::string::npos);
}

void GenerateSimtBroadcastPairDefinition(int32_t second_extent, std::string &definition) {
  SimtCodegenContext context;
  ILTestGraph g("simt_broadcast_pair", 2, 8, 2, 8);
  BuildSimtPostReduceGraph(g);
  const auto reduce = g.graph.FindNode("reduce");
  ASSERT_NE(reduce, nullptr);
  ASSERT_EQ(af::GraphUtils::IsolateNodeOneIO(reduce), af::SUCCESS);
  ASSERT_EQ(af::GraphUtils::RemoveNodeWithoutRelink(af::AscGraphUtils::GetComputeGraph(g.graph), reduce), af::SUCCESS);
  const auto first_load = g.graph.FindNode("addend_load");
  ASSERT_NE(first_load, nullptr);
  first_load->outputs[0].attr.repeats = {af::ops::One, g.s3};
  first_load->outputs[0].attr.strides = {af::ops::Zero, af::ops::One};

  Data extra("extra", g.graph);
  extra.ir_attr.SetIndex(3);
  extra.y.dtype = af::DT_FLOAT16;
  g.graph.FindNode("extra")->outputs[0].attr.mem.tensor_id = 41;
  Load second_load("second_bias_load");
  g.graph.AddNode(second_load);
  second_load.x = extra.y;
  second_load.y.dtype = af::DT_FLOAT16;
  second_load.attr.sched.axis = {g.z2.id, g.z3.id};
  *second_load.y.axis = {g.z2.id, g.z3.id};
  *second_load.y.repeats = {af::ops::One, af::Symbol(second_extent)};
  *second_load.y.strides = {af::ops::Zero, af::ops::One};
  const auto second_node = g.graph.FindNode("second_bias_load");
  ASSERT_NE(second_node, nullptr);
  second_node->outputs[0].attr.mem.tensor_id = 51;
  ASSERT_EQ(SetTemplateRole(second_node, TemplateRole::kSimtDirectGmBoundary), af::SUCCESS);

  Add add("add_second_bias");
  g.graph.AddNode(add);
  const auto add_node = g.graph.FindNode("add_second_bias");
  const auto first_add = g.graph.FindNode("output_transform");
  const auto store = g.graph.FindNode("store");
  ASSERT_NE(add_node, nullptr);
  ASSERT_NE(first_add, nullptr);
  ASSERT_NE(store, nullptr);
  add.y.dtype = af::DT_FLOAT16;
  add.attr.sched.axis = first_add->attr.sched.axis;
  *add.y.axis = first_add->outputs[0].attr.axis;
  *add.y.repeats = first_add->outputs[0].attr.repeats;
  *add.y.strides = first_add->outputs[0].attr.strides;
  add_node->outputs[0].attr.mem.tensor_id = 52;
  ASSERT_EQ(SetTemplateRole(add_node, TemplateRole::kSimtInlineTransform), af::SUCCESS);
  ASSERT_EQ(af::GraphUtils::ReplaceEdgeSrc(first_add->GetOutDataAnchor(0), store->GetInDataAnchor(0),
                                           add_node->GetOutDataAnchor(0)),
            af::SUCCESS);
  ASSERT_EQ(af::GraphUtils::AddEdge(first_add->GetOutDataAnchor(0), add_node->GetInDataAnchor(0)), af::SUCCESS);
  ASSERT_EQ(af::GraphUtils::AddEdge(second_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(1)), af::SUCCESS);
  ASSERT_EQ(af::AscGraphUtils::GetComputeGraph(g.graph)->TopologicalSorting(), af::SUCCESS);
  GenerateSimtPostReduceFuncDefinitionFromGraph(g, definition);
}

TEST(IndirectLoadApiCallTest, ReusesBroadcastCoordinatesAcrossLoads) {
  std::string definition;
  GenerateSimtBroadcastPairDefinition(8, definition);
  EXPECT_NE(definition.find("context.gm_9[index_coord_0]"), std::string::npos);
  EXPECT_NE(definition.find("context.gm_51[index_coord_0]"), std::string::npos);
  // [SIMT 分解共享] 尺寸对齐的两个广播侧载复用同一 coord_axis 坐标变量，
  // 逐元素 magic mod 站点数为零（旧版为一次共享 mod）。
  EXPECT_NE(definition.find("index_coord_0 = coord_axis;"), std::string::npos);
  EXPECT_EQ(definition.find("output_index % 8"), std::string::npos);
  EXPECT_EQ(definition.find("index_coord_1"), std::string::npos);
}

TEST(IndirectLoadApiCallTest, KeepsDistinctPhysicalExtentsInBroadcastCoordinates) {
  std::string definition;
  GenerateSimtBroadcastPairDefinition(4, definition);
  // [SIMT 分解共享] 与 gather 轴对齐（extent 8）的侧载走 coord_axis；物理 extent
  // 不同（4）的侧载保留独立 mod 站点，两者不得混用坐标。
  EXPECT_NE(definition.find("index_coord_0 = coord_axis;"), std::string::npos);
  EXPECT_NE(definition.find("% static_cast<uint64_t>(4)"), std::string::npos);
  EXPECT_EQ(definition.find("output_index % 8"), std::string::npos);
  EXPECT_NE(definition.find("context.gm_9[index_coord_0]"), std::string::npos);
  EXPECT_NE(definition.find("context.gm_51[index_coord_1]"), std::string::npos);
}

TEST(IndirectLoadApiCallTest, GenerateSimtPostReduceKeepsParallelStoreChain) {
  ILTestGraph g("simt_post_parallel_output");
  std::string definition;
  GenerateSimtPostReduceFuncDefinition(g, definition, true);

  EXPECT_NE(definition.find("struct OutputPack"), std::string::npos);
  EXPECT_NE(definition.find("outputs.output0 ="), std::string::npos);
  EXPECT_NE(definition.find("outputs.output1 ="), std::string::npos);
  EXPECT_NE(definition.find("__gm__ half *output"), std::string::npos);
  EXPECT_NE(definition.find("__ubuf__ half *output"), std::string::npos);
  EXPECT_NE(definition.find("[output_index] = outputs.output"), std::string::npos);
  EXPECT_NE(definition.find("targets.output1[(local_index / ("), std::string::npos);
  EXPECT_NE(definition.find("KernelUtils::SizeAlign"), std::string::npos);
}

// ==================== SIMT post-reduce Init ====================

// ======================== SIMT index 闭包预计算 ========================

TEST(IndirectLoadApiCallTest, BuildSimtIndexPrepassPlanAcceptsArangeChain) {
  ILTestGraph g("prepass_arange", 4, 8, 2, 16);
  BuildSimtPostReduceGraph(g);

  // 用“尾轴 ramp Arange + 常量”加法链替换 GM 装载的 index 输入。
  Arange col("prepass_col");
  col.ir_attr.SetBase(af::Symbol(0));
  col.ir_attr.SetStep(af::Symbol(1));
  col.attr.sched.axis = {g.z2.id, g.z3.id};
  col.y.dtype = af::DT_INT64;
  *col.y.axis = {g.z2.id, g.z3.id};
  *col.y.repeats = {af::Symbol(1), g.s3};
  *col.y.strides = {af::sym::kSymbolZero, af::sym::kSymbolOne};
  ASSERT_NE(g.graph.AddNode(col), nullptr);

  Scalar offset("prepass_offset", g.graph);
  offset.ir_attr.SetValue("16");
  offset.y.dtype = af::DT_INT64;
  auto offset_node = g.graph.FindNode("prepass_offset");
  ASSERT_NE(offset_node, nullptr);

  Add idx_add("prepass_index_add");
  g.graph.AddNode(idx_add);
  idx_add.attr.sched.axis = {g.z2.id, g.z3.id};
  *idx_add.y.axis = {g.z2.id, g.z3.id};
  *idx_add.y.repeats = {g.s2, g.s3};
  *idx_add.y.strides = {g.s3, af::sym::kSymbolOne};
  idx_add.x1 = col.y;
  idx_add.x2 = offset.y;
  auto add_node = g.graph.FindNode("prepass_index_add");
  ASSERT_NE(add_node, nullptr);

  auto il = g.graph.FindNode("indirect_load");
  auto idx_load = g.graph.FindNode("idx_load");
  auto x_load = g.graph.FindNode("x_load");
  ASSERT_NE(il, nullptr);
  ASSERT_NE(idx_load, nullptr);
  ASSERT_NE(x_load, nullptr);
  // 索引表复用 value tensor 存储：值 dtype 至少 4B（fp16 图需提升为 fp32）。
  x_load->outputs[0].attr.dtype = af::DT_FLOAT;
  ASSERT_EQ(af::GraphUtils::ReplaceEdgeSrc(idx_load->GetOutDataAnchor(0), il->GetInDataAnchor(1),
                                           add_node->GetOutDataAnchor(0)),
            af::SUCCESS);
  ASSERT_EQ(af::AscGraphUtils::GetComputeGraph(g.graph)->TopologicalSorting(), af::SUCCESS);
  AnnotateSimtTemplate(g.graph);

  SimtIndexPrepassPlan plan;
  ASSERT_EQ(BuildSimtIndexPrepassPlan(*il, plan), af::SUCCESS);
  EXPECT_TRUE(plan.eligible);
  EXPECT_GE(plan.four_byte_slot_count, 1UL);
  EXPECT_LE(plan.four_byte_slot_count, 6UL);
  EXPECT_EQ(plan.mask_slot_count, 0UL);
  EXPECT_LT(plan.result_slot, plan.four_byte_slot_count);
}

TEST(IndirectLoadApiCallTest, BuildSimtIndexPrepassPlanRejectsLoadedIndex) {
  ILTestGraph g("prepass_load", 4, 8, 2, 16);
  BuildSimtPostReduceGraph(g);
  AnnotateSimtTemplate(g.graph);

  auto il = g.graph.FindNode("indirect_load");
  ASSERT_NE(il, nullptr);
  SimtIndexPrepassPlan plan;
  ASSERT_EQ(BuildSimtIndexPrepassPlan(*il, plan), af::SUCCESS);
  // index 直接来自 GM 装载（数据依赖闭包）：不可预计算，必须回退逐元素路径。
  EXPECT_FALSE(plan.eligible);
}

// [漂移防护] 调度期 stash 与 codegen 期重建各算一次 scratch 需求，两份图副本之间
// 图可能被改写：预留必须覆盖重建需求（等价或更小均可），出现缺口或不可判定时
// 一律拒绝，回退逐元素路径。
TEST(IndirectLoadApiCallTest, SimtIndexPrepassPlanFitsTmpDemandGuardsDrift) {
  SimtIndexPrepassPlan plan;
  plan.eligible = true;
  plan.four_byte_slot_count = 4UL;
  plan.mask_slot_count = 1UL;
  plan.inner_size = af::Symbol(2048);
  const SimtIndexPrepassTmpDemand reserved{true, CalcSimtIndexPrepassTmpBytes(plan)};
  // 一致（无漂移）：允许。
  EXPECT_TRUE(SimtIndexPrepassPlanFitsTmpDemand(plan, reserved));
  // 漂移使链变复杂（调度期少 stash 一个槽）：重建需求超出预留，必须拒绝。
  SimtIndexPrepassPlan drifted = plan;
  drifted.four_byte_slot_count = plan.four_byte_slot_count + 1UL;
  EXPECT_FALSE(SimtIndexPrepassPlanFitsTmpDemand(drifted, reserved));
  // 漂移使链变简单（预留覆盖重建）：安全，允许。
  SimtIndexPrepassPlan simplified = plan;
  simplified.four_byte_slot_count = plan.four_byte_slot_count - 1UL;
  EXPECT_TRUE(SimtIndexPrepassPlanFitsTmpDemand(simplified, reserved));
  // 调度期未 stash 需求（attr 缺失）：回退。
  EXPECT_FALSE(SimtIndexPrepassPlanFitsTmpDemand(plan, SimtIndexPrepassTmpDemand{}));
  // 重建 plan 本身不可行：回退。
  plan.eligible = false;
  EXPECT_FALSE(SimtIndexPrepassPlanFitsTmpDemand(plan, reserved));
}

TEST(IndirectLoadApiCallTest, InitSimtPostReduceSucceeds) {
  ILTestGraph g("simt_post");
  BuildSimtPostReduceGraph(g);
  AnnotateSimtTemplate(g.graph);

  auto il_node = g.graph.FindNode("indirect_load");
  ASSERT_NE(il_node, nullptr);
  auto output_transform = g.graph.FindNode("output_transform");
  ASSERT_NE(output_transform, nullptr);
  ASSERT_EQ(SetTemplateRole(output_transform, TemplateRole::kSimtInlineTransform), af::SUCCESS);

  IndirectLoadRegApiCall call("IndirectLoadRegApiCall");
  auto status = call.Init(il_node);
  EXPECT_EQ(status, af::SUCCESS);
}

TEST(IndirectLoadApiCallTest, InitSimtGroupMemberLocalTargetWithoutPostReduceSucceeds) {
  // 组员形态回归：无 post-Reduce + value 直写 UB 局部目标（local target 不伴随 post-Reduce）。
  // ResolveSimtMetadata 的旧双向等价断言（has_post_reduce_ == (local_target_count == 1)）会使
  // 组员（含 fused primary）在 ApiCall::Init 的 ParseAttr 阶段失败——Init 先于 Generate 的
  // 成员跳过执行。断言已与 schedule 侧 ValidateSimtOutputPlan 对齐为单向蕴含。
  ILTestGraph g("simt_group_member");
  BuildSimtPostReduceGraph(g);
  AnnotateSimtTemplate(g.graph);

  auto il_node = g.graph.FindNode("indirect_load");
  ASSERT_NE(il_node, nullptr);
  ASSERT_EQ(SetSimtGroupMember(il_node), af::SUCCESS);
  bool metadata_supported = false;
  ASSERT_EQ(FinalizeLoweringMetadata(il_node, metadata_supported), af::SUCCESS);
  ASSERT_TRUE(metadata_supported);
  IndirectLoadLoweringMetadata metadata;
  ASSERT_EQ(GetLoweringMetadata(il_node, metadata), af::SUCCESS);
  EXPECT_FALSE(metadata.simt.has_post_reduce);
  ASSERT_EQ(metadata.simt.output_chains.size(), 1UL);
  EXPECT_TRUE(metadata.simt.output_chains[0].local_target);

  IndirectLoadRegApiCall call("IndirectLoadRegApiCall");
  EXPECT_EQ(call.Init(il_node), af::SUCCESS);
}

// ======================== SIMT fused group graph（多 gather L2 寄存器归并）========================
// 两路 IndirectLoad（组员形态：value 写 UB 局部目标）→ merge Add → Store → Output。
// 镜像调度侧 PlanSimtFusedEmission 的 L2 标注产物：primary 携带 fused 标记与归并信息，
// 归并树标 kSimtInlineTransform，Store 标 kSimtDirectGmBoundary（evaluator 归并结果经
// OutputTargets 直写其 GM 目标，跳过独立发射）。

void BuildSimtFusedGroupGraph(ILTestGraph &g, bool shared_reduce_terminal = false) {
  const af::Expression One = af::sym::kSymbolOne;

  Data x0_data("x0", g.graph);
  x0_data.ir_attr.SetIndex(0);
  x0_data.y.dtype = ge::DT_FLOAT16;
  x0_data.attr.sched.axis = {g.z0.id, g.z1.id};
  *x0_data.y.axis = {g.z0.id, g.z1.id};
  *x0_data.y.repeats = {g.s0, g.s1};
  *x0_data.y.strides = {g.s1, One};

  Data x1_data("x1", g.graph);
  x1_data.ir_attr.SetIndex(1);
  x1_data.y.dtype = ge::DT_FLOAT16;
  x1_data.attr.sched.axis = {g.z0.id, g.z1.id};
  *x1_data.y.axis = {g.z0.id, g.z1.id};
  *x1_data.y.repeats = {g.s0, g.s1};
  *x1_data.y.strides = {g.s1, One};

  Data idx0_data("idx0", g.graph);
  idx0_data.ir_attr.SetIndex(2);
  idx0_data.y.dtype = ge::DT_INT32;
  idx0_data.attr.sched.axis = {g.z2.id, g.z3.id};
  *idx0_data.y.axis = {g.z2.id, g.z3.id};
  *idx0_data.y.repeats = {g.s2, g.s3};
  *idx0_data.y.strides = {g.s3, One};

  Data idx1_data("idx1", g.graph);
  idx1_data.ir_attr.SetIndex(3);
  idx1_data.y.dtype = ge::DT_INT32;
  idx1_data.attr.sched.axis = {g.z2.id, g.z3.id};
  *idx1_data.y.axis = {g.z2.id, g.z3.id};
  *idx1_data.y.repeats = {g.s2, g.s3};
  *idx1_data.y.strides = {g.s3, One};

  Load x_load0("x_load0");
  g.graph.AddNode(x_load0);
  x_load0.x = x0_data.y;
  x_load0.y.dtype = ge::DT_FLOAT16;
  x_load0.attr.sched.axis = {g.z0.id, g.z1.id};
  *x_load0.y.axis = {g.z0.id, g.z1.id};
  *x_load0.y.repeats = {g.s0, g.s1};
  *x_load0.y.strides = {g.s1, One};

  Load x_load1("x_load1");
  g.graph.AddNode(x_load1);
  x_load1.x = x1_data.y;
  x_load1.y.dtype = ge::DT_FLOAT16;
  x_load1.attr.sched.axis = {g.z0.id, g.z1.id};
  *x_load1.y.axis = {g.z0.id, g.z1.id};
  *x_load1.y.repeats = {g.s0, g.s1};
  *x_load1.y.strides = {g.s1, One};

  Load idx_load0("idx_load0");
  g.graph.AddNode(idx_load0);
  idx_load0.x = idx0_data.y;
  idx_load0.y.dtype = ge::DT_INT32;
  idx_load0.attr.sched.axis = {g.z2.id, g.z3.id};
  *idx_load0.y.axis = {g.z2.id, g.z3.id};
  *idx_load0.y.repeats = {g.s2, g.s3};
  *idx_load0.y.strides = {g.s3, One};

  Load idx_load1("idx_load1");
  g.graph.AddNode(idx_load1);
  idx_load1.x = idx1_data.y;
  idx_load1.y.dtype = ge::DT_INT32;
  idx_load1.attr.sched.axis = {g.z2.id, g.z3.id};
  *idx_load1.y.axis = {g.z2.id, g.z3.id};
  *idx_load1.y.repeats = {g.s2, g.s3};
  *idx_load1.y.strides = {g.s3, One};

  IndirectLoad il0("indirect_load0");
  g.graph.AddNode(il0);
  il0.x1 = x_load0.y;
  il0.x2 = idx_load0.y;
  il0.ir_attr.SetAxis(1);
  il0.attr.sched.axis = {g.z2.id, g.z3.id};
  *il0.y.axis = {g.z2.id, g.z3.id};
  *il0.y.repeats = {g.s2, g.s3};
  *il0.y.strides = {g.s3, One};

  IndirectLoad il1("indirect_load1");
  g.graph.AddNode(il1);
  il1.x1 = x_load1.y;
  il1.x2 = idx_load1.y;
  il1.ir_attr.SetAxis(1);
  il1.attr.sched.axis = {g.z2.id, g.z3.id};
  *il1.y.axis = {g.z2.id, g.z3.id};
  *il1.y.repeats = {g.s2, g.s3};
  *il1.y.strides = {g.s3, One};

  Add merge_add("merge_add");
  g.graph.AddNode(merge_add);
  merge_add.x1 = il0.y;
  merge_add.x2 = il1.y;
  merge_add.y.dtype = ge::DT_FLOAT16;
  merge_add.attr.sched.axis = {g.z2.id, g.z3.id};
  *merge_add.y.axis = {g.z2.id, g.z3.id};
  *merge_add.y.repeats = {g.s2, g.s3};
  *merge_add.y.strides = {g.s3, One};

  if (shared_reduce_terminal) {
    // 共享 Reduce 终端（形态 2）：fake Reduce（Abs + kComputeReduce，同
    // BuildSimtPostReduceGraph 先例）——post-Reduce 使 primary 行为复位、不可提升，
    // fused 调用经循环体路径发射。
    Abs reduce_sum("reduce_sum");
    g.graph.AddNode(reduce_sum);
    reduce_sum.x = merge_add.y;
    reduce_sum.y.dtype = ge::DT_FLOAT16;
    reduce_sum.attr.sched.axis = {g.z2.id, g.z3.id};
    *reduce_sum.y.axis = {g.z2.id, g.z3.id};
    *reduce_sum.y.repeats = {g.s2, g.s3};
    *reduce_sum.y.strides = {g.s3, One};

    Output y_out("y");
    g.graph.AddNode(y_out);
    y_out.ir_attr.SetIndex(0);
    y_out.x = reduce_sum.y;
    y_out.y.dtype = ge::DT_FLOAT16;
  } else {
    Store store("store");
    g.graph.AddNode(store);
    store.x = merge_add.y;
    store.y.dtype = ge::DT_FLOAT16;
    store.attr.sched.axis = {g.z2.id, g.z3.id};
    *store.y.axis = {g.z2.id, g.z3.id};
    *store.y.repeats = {g.s2, g.s3};
    *store.y.strides = {g.s3, One};

    Output y_out("y");
    g.graph.AddNode(y_out);
    y_out.ir_attr.SetIndex(0);
    y_out.x = store.y;
    y_out.y.dtype = ge::DT_FLOAT16;
  }

  // ----- API attrs -----
  for (const char *name : {"x0", "x1", "idx0", "idx1", "y"}) {
    auto n = g.graph.FindNode(name);
    n->attr.api.compute_type = af::ComputeType::kComputeInvalid;
    n->attr.api.type = af::ApiType::kAPITypeBuffer;
    n->attr.api.unit = af::ComputeUnit::kUnitNone;
  }
  for (const char *name : {"x_load0", "x_load1", "idx_load0", "idx_load1"}) {
    auto n = g.graph.FindNode(name);
    n->attr.api.compute_type = af::ComputeType::kComputeLoad;
    n->attr.api.type = af::ApiType::kAPITypeCompute;
    n->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  }
  for (const char *name : {"indirect_load0", "indirect_load1"}) {
    auto n = g.graph.FindNode(name);
    n->attr.api.compute_type = af::ComputeType::kComputeLoad;
    n->attr.api.type = af::ApiType::kAPITypeCompute;
    n->attr.api.unit = af::ComputeUnit::kUnitVector;
  }
  auto merge_add_node = g.graph.FindNode("merge_add");
  merge_add_node->attr.api.compute_type = af::ComputeType::kComputeElewise;
  merge_add_node->attr.api.type = af::ApiType::kAPITypeCompute;
  merge_add_node->attr.api.unit = af::ComputeUnit::kUnitVector;

  auto store_node = g.graph.FindNode("store");
  if (store_node != nullptr) {
    store_node->attr.api.compute_type = af::ComputeType::kComputeStore;
    store_node->attr.api.type = af::ApiType::kAPITypeCompute;
    store_node->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  } else {
    auto reduce_node = g.graph.FindNode("reduce_sum");
    reduce_node->attr.api.compute_type = af::ComputeType::kComputeReduce;
    reduce_node->attr.api.type = af::ApiType::kAPITypeCompute;
    reduce_node->attr.api.unit = af::ComputeUnit::kUnitVector;
  }

  // ----- Axis splitting: merge output axes → z2z3 -----
  auto axes = g.graph.GetAllAxis();
  auto z2_id = axes[2]->id;
  auto z3_id = axes[3]->id;

  auto z2z3 = g.graph.MergeAxis({z2_id, z3_id});
  auto [z2z3T, z2z3t] = g.graph.TileSplit(z2z3->id);
  auto [z2z3TB, z2z3Tb] = g.graph.BlockSplit(z2z3T->id);

  for (auto n : g.graph.GetAllNodes()) {
    if (IsOps<Data>(n) || IsOps<Output>(n) || n->GetName() == "x_load0" || n->GetName() == "x_load1") continue;
    g.graph.ApplyMerge(n, z2z3->id);
    g.graph.ApplySplit(n, z2z3T->id, z2z3t->id);
    g.graph.ApplySplit(n, z2z3TB->id, z2z3Tb->id);
    g.graph.ApplyReorder(n, {z2z3TB->id, z2z3Tb->id, z2z3t->id});
  }

  const char *terminal_name = shared_reduce_terminal ? "reduce_sum" : "store";
  for (const char *name : {"indirect_load0", "indirect_load1", "merge_add", terminal_name}) {
    auto n = g.graph.FindNode(name);
    n->attr.sched.loop_axis = z2z3Tb->id;
    n->outputs[0].attr.vectorized_axis = {z2z3t->id};
    n->outputs[0].attr.vectorized_strides = {One};
  }

  // ----- Memory allocation -----
  auto set_gm = [](const af::AscNodePtr &node, ascir::TensorId tensor_id) {
    node->outputs[0].attr.mem.tensor_id = tensor_id;
    node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
    node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
    node->outputs[0].attr.mem.position = af::Position::kPositionGM;
    node->outputs[0].attr.que.id = af::kIdNone;
  };
  auto set_ub_que = [](const af::AscNodePtr &node, ascir::TensorId tensor_id, ascir::TensorId reuse_id,
                       af::Position position, int64_t que_id) {
    node->outputs[0].attr.mem.tensor_id = tensor_id;
    node->outputs[0].attr.mem.reuse_id = reuse_id;
    node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
    node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
    node->outputs[0].attr.mem.position = position;
    node->outputs[0].attr.que.id = que_id;
    node->outputs[0].attr.que.depth = 2;
    node->outputs[0].attr.que.buf_num = 2;
  };
  set_gm(g.graph.FindNode("x0"), 0);
  set_gm(g.graph.FindNode("x1"), 6);
  set_gm(g.graph.FindNode("idx0"), 3);
  set_gm(g.graph.FindNode("idx1"), 7);
  if (shared_reduce_terminal) {
    set_ub_que(g.graph.FindNode("reduce_sum"), 12, 12, af::Position::kPositionVecOut, 7);
    g.graph.FindNode("reduce_sum")->outputs[0].attr.dtype = ge::DT_FLOAT16;
  } else {
    set_gm(g.graph.FindNode("store"), 2);
  }
  set_ub_que(g.graph.FindNode("x_load0"), 4, 4, af::Position::kPositionVecIn, 1);
  set_ub_que(g.graph.FindNode("idx_load0"), 5, 5, af::Position::kPositionVecIn, 2);
  set_ub_que(g.graph.FindNode("x_load1"), 9, 9, af::Position::kPositionVecIn, 4);
  set_ub_que(g.graph.FindNode("idx_load1"), 10, 10, af::Position::kPositionVecIn, 5);
  set_ub_que(g.graph.FindNode("indirect_load0"), 1, 1, af::Position::kPositionVecOut, 0);
  set_ub_que(g.graph.FindNode("indirect_load1"), 8, 8, af::Position::kPositionVecOut, 3);
  set_ub_que(g.graph.FindNode("merge_add"), 11, 11, af::Position::kPositionVecOut, 6);
  g.graph.FindNode("indirect_load0")->outputs[0].attr.dtype = ge::DT_FLOAT16;
  g.graph.FindNode("indirect_load1")->outputs[0].attr.dtype = ge::DT_FLOAT16;
}

void AnnotateSimtFusedGroupTemplate(ILTestGraph &g) {
  af::AxisId outer_id = af::kIdNone;
  for (const auto &axis : g.graph.GetAllAxis()) {
    if (axis != nullptr && axis->name == "z2z3") outer_id = axis->id;
  }
  ASSERT_NE(outer_id, af::kIdNone);
  const auto axes = g.graph.GetAllAxis();
  const af::Expression One = af::sym::kSymbolOne;
  TemplateLogicalView view;
  view.input.axis_ids = {axes[0]->id, axes[1]->id};
  view.input.sizes = {g.graph.FindAxis(axes[0]->id)->size, g.graph.FindAxis(axes[1]->id)->size};
  view.input.strides = {g.graph.FindAxis(axes[1]->id)->size, One};
  view.index.axis_ids = {axes[2]->id, axes[3]->id};
  view.index.sizes = {g.graph.FindAxis(axes[2]->id)->size, g.graph.FindAxis(axes[3]->id)->size};
  view.index.strides = {g.graph.FindAxis(axes[3]->id)->size, One};
  view.output.axis_ids = {axes[2]->id, axes[3]->id};
  view.output.sizes = {g.graph.FindAxis(axes[2]->id)->size, g.graph.FindAxis(axes[3]->id)->size};
  view.output.strides = {g.graph.FindAxis(axes[3]->id)->size, One};

  for (const char *name : {"indirect_load0", "indirect_load1"}) {
    auto il = g.graph.FindNode(name);
    ASSERT_NE(il, nullptr);
    ASSERT_EQ(::ascir::SetTemplateId(il, ::ascir::TemplateId::kIndirectLoadSimt), af::SUCCESS);
    ASSERT_EQ(SetImplementation(il, Implementation::kDefault), af::SUCCESS);
    TemplateAxes t{outer_id, af::kIdNone, af::kIdNone};
    ASSERT_EQ(SetTemplateAxes(il, t), af::SUCCESS);
    ASSERT_EQ(ClassifyIndirectLoadLayout(view.input, view.input), af::SUCCESS);
    ASSERT_EQ(ClassifyIndirectLoadLayout(view.index, view.index), af::SUCCESS);
    ASSERT_EQ(SetTemplateLogicalView(il, view), af::SUCCESS);
  }
  // L2 角色标注（镜像调度侧 PlanSimtFusedEmission/ApplyGraphPassGroup 的产物）。
  for (const char *name : {"x_load0", "x_load1"}) {
    ASSERT_EQ(SetTemplateRole(g.graph.FindNode(name), TemplateRole::kSimtInputBoundary), af::SUCCESS);
  }
  for (const char *name : {"idx_load0", "idx_load1"}) {
    ASSERT_EQ(SetTemplateRole(g.graph.FindNode(name), TemplateRole::kSimtDirectGmBoundary), af::SUCCESS);
  }
  ASSERT_EQ(SetTemplateRole(g.graph.FindNode("merge_add"), TemplateRole::kSimtInlineTransform), af::SUCCESS);
  ASSERT_EQ(SetTemplateRole(g.graph.FindNode("store"), TemplateRole::kSimtDirectGmBoundary), af::SUCCESS);
  // 组员标记 + 成员形态 lowering 元数据（value 写 UB 局部目标）。
  for (const char *name : {"indirect_load0", "indirect_load1"}) {
    auto il = g.graph.FindNode(name);
    ASSERT_EQ(SetSimtGroupMember(il), af::SUCCESS);
    bool metadata_supported = false;
    ASSERT_EQ(FinalizeLoweringMetadata(il, metadata_supported), af::SUCCESS);
    ASSERT_TRUE(metadata_supported);
  }
  // fused 发射标记 + L2 归并信息（primary 携带全组成员名、归并树与 Store 终端名）。
  auto primary = g.graph.FindNode("indirect_load0");
  ASSERT_EQ(SetSimtFusedPrimary(primary, {"indirect_load0", "indirect_load1"}), af::SUCCESS);
  SimtFusedMergeInfo merge_info;
  merge_info.register_merge = true;
  merge_info.merge_node_names = {"merge_add"};
  merge_info.merge_store_names = {"store"};
  ASSERT_EQ(SetSimtFusedMergeInfo(primary, merge_info), af::SUCCESS);
}

TEST_F(IndirectLoadConstructFromNodesTest, GeneratesSimtFusedRegisterMergeWithDirectGmStore) {
  ILTestGraph g("simt_fused_group");
  BuildSimtFusedGroupGraph(g);
  AnnotateSimtFusedGroupTemplate(g);

  auto il0 = g.graph.FindNode("indirect_load0");
  ASSERT_NE(il0, nullptr);
  Tiler tiler;
  SetupTiler(g.graph, tiler);
  TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(g.graph);

  IndirectLoadRegApiCall call("IndirectLoadRegApiCall");
  ASSERT_EQ(call.Init(il0), af::SUCCESS);
  ApiTensor x1, x2;
  x1.id = il0->inputs[0].attr.mem.tensor_id;
  x2.id = il0->inputs[1].attr.mem.tensor_id;
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  // fused 发射按 tensor id 取各成员 input 指针与 value/UB 目标，逐个注册。
  for (const char *name : {"x_load0", "idx_load0", "x_load1", "idx_load1", "indirect_load0", "indirect_load1"}) {
    const auto node = g.graph.FindNode(name);
    ASSERT_NE(node, nullptr);
    tpipe.AddTensor(*node->outputs()[0]);
  }

  ascir::AxisId outer_tb = af::kIdNone;
  for (const auto &axis : g.graph.GetAllAxis()) {
    if (axis != nullptr && axis->name == "z2z3Tb") outer_tb = axis->id;
  }
  ASSERT_NE(outer_tb, af::kIdNone);
  std::string generated;
  ASSERT_EQ(call.Generate(tpipe, {outer_tb}, generated), af::SUCCESS);

  // L2 融合发射头：单 evaluator（seq-V1）+ 寄存器归并。
  EXPECT_NE(generated.find("// IndirectLoad SIMT Fused (seq-V1: single evaluator, serial loads, 2 paths, register "
                           "merge)"),
            std::string::npos);
  // FusedCtx：N 路 input 指针（常量驻留参数）。
  EXPECT_NE(generated.find("struct FusedCtx"), std::string::npos);
  EXPECT_NE(generated.find("__gm__ half *in0;"), std::string::npos);
  EXPECT_NE(generated.find("__gm__ half *in1;"), std::string::npos);
  // Store 终端经 OutputTargets 直写 GM（merge_store_names → out0；Store 自身跳过独立发射）。
  EXPECT_NE(generated.find("struct OutputTargets { __gm__ half *out0; };"), std::string::npos);
  // [simt 栈预算] Policy（含 magic/shift）已在向量侧每 tile 构造一次存入 FusedCtx
  // （走 VF_CALL 参数区/UB 6KB）；热循环内 FusedEval 每元素只读
  // context.policyK.GetAddress(output_index)——乘移级，零除法零循环。
  EXPECT_NE(generated.find("AscendC::Internal::IndirectLoadSimtCaseSelector<FusedCase0>::Policy policy0;"),
            std::string::npos);
  EXPECT_NE(generated.find("fctx.policy0 = AscendC::Internal::IndirectLoadSimtCaseSelector<FusedCase0>::MakePolicy("),
            std::string::npos);
  EXPECT_NE(generated.find("context.policy0.GetAddress(output_index)"), std::string::npos);
  EXPECT_NE(generated.find("__simt_callee__ __aicore__ inline static OutputPack FusedEval(OffsetType output_index, "
                           "const FusedCtx& context)"),
            std::string::npos);
  // [simt 栈预算防回退] MakePolicy 只允许出现在 fctx 构造段（向量侧、每 tile 一次）；
  // FusedEval 函数体内不得出现 MakePolicy（每元素重算 magic/shift 是 354 的根因）。
  const auto fctx_pos = generated.find("fctx.policy0 = AscendC::Internal::IndirectLoadSimtCaseSelector");
  ASSERT_NE(fctx_pos, std::string::npos);
  const auto fused_eval_pos = generated.find("OutputPack FusedEval(OffsetType output_index, const FusedCtx& context)");
  ASSERT_NE(fused_eval_pos, std::string::npos);
  ASSERT_LT(fctx_pos, fused_eval_pos) << "MakePolicy must precede FusedEval (vector-side, per-tile)";
  const auto eval_body_end = generated.find("__simt_callee__ __aicore__ inline static void Store");
  ASSERT_NE(eval_body_end, std::string::npos);
  EXPECT_EQ(generated.substr(fused_eval_pos, eval_body_end - fused_eval_pos).find("MakePolicy"), std::string::npos)
      << "MakePolicy must NOT appear inside FusedEval body (per-element magic computation is the 354 root cause)";
  // 逐成员 index 链求值 + AddressPolicy 寻址 + GM load（关键寻址表达式）。
  EXPECT_NE(generated.find("value0 = context.in0[input_offset0];"), std::string::npos);
  EXPECT_NE(generated.find("value1 = context.in1[input_offset1];"), std::string::npos);
  // L2 寄存器归并：merge_add 以 value0/value1 为种子折叠，结果写 OutputPack 并直写 GM。
  EXPECT_NE(generated.find("out.out0 = "), std::string::npos);
  EXPECT_NE(generated.find("t.out0[output_index] = o.out0;"), std::string::npos);
  EXPECT_NE(generated.find("AscendC::IndirectLoadSimtFused<FusedBody>(ftgt, fctx"), std::string::npos);
}

// L3（UB 归并）标注：register_merge=false 时归并链作为扇出分支向量代码（无核内归并树
// 与 Store 终端名，PlanSimtFusedEmission 判非寄存器归并即提前返回），成员 value 写各自
// UB 局部目标。成员侧标注（index 链 GM 直读 + input 边界 + 组员标记 + lowering 元数据）
// 与 L2 一致（组员元数据只含本路 index 链，与归并档位无关）。
void AnnotateSimtFusedUbMergeGroupTemplate(ILTestGraph &g) {
  af::AxisId outer_id = af::kIdNone;
  for (const auto &axis : g.graph.GetAllAxis()) {
    if (axis != nullptr && axis->name == "z2z3") outer_id = axis->id;
  }
  ASSERT_NE(outer_id, af::kIdNone);
  const auto axes = g.graph.GetAllAxis();
  const af::Expression One = af::sym::kSymbolOne;
  TemplateLogicalView view;
  view.input.axis_ids = {axes[0]->id, axes[1]->id};
  view.input.sizes = {g.graph.FindAxis(axes[0]->id)->size, g.graph.FindAxis(axes[1]->id)->size};
  view.input.strides = {g.graph.FindAxis(axes[1]->id)->size, One};
  view.index.axis_ids = {axes[2]->id, axes[3]->id};
  view.index.sizes = {g.graph.FindAxis(axes[2]->id)->size, g.graph.FindAxis(axes[3]->id)->size};
  view.index.strides = {g.graph.FindAxis(axes[3]->id)->size, One};
  view.output.axis_ids = {axes[2]->id, axes[3]->id};
  view.output.sizes = {g.graph.FindAxis(axes[2]->id)->size, g.graph.FindAxis(axes[3]->id)->size};
  view.output.strides = {g.graph.FindAxis(axes[3]->id)->size, One};

  for (const char *name : {"indirect_load0", "indirect_load1"}) {
    auto il = g.graph.FindNode(name);
    ASSERT_NE(il, nullptr);
    ASSERT_EQ(::ascir::SetTemplateId(il, ::ascir::TemplateId::kIndirectLoadSimt), af::SUCCESS);
    ASSERT_EQ(SetImplementation(il, Implementation::kDefault), af::SUCCESS);
    TemplateAxes t{outer_id, af::kIdNone, af::kIdNone};
    ASSERT_EQ(SetTemplateAxes(il, t), af::SUCCESS);
    ASSERT_EQ(ClassifyIndirectLoadLayout(view.input, view.input), af::SUCCESS);
    ASSERT_EQ(ClassifyIndirectLoadLayout(view.index, view.index), af::SUCCESS);
    ASSERT_EQ(SetTemplateLogicalView(il, view), af::SUCCESS);
  }
  for (const char *name : {"x_load0", "x_load1"}) {
    ASSERT_EQ(SetTemplateRole(g.graph.FindNode(name), TemplateRole::kSimtInputBoundary), af::SUCCESS);
  }
  for (const char *name : {"idx_load0", "idx_load1"}) {
    ASSERT_EQ(SetTemplateRole(g.graph.FindNode(name), TemplateRole::kSimtDirectGmBoundary), af::SUCCESS);
  }
  for (const char *name : {"indirect_load0", "indirect_load1"}) {
    auto il = g.graph.FindNode(name);
    ASSERT_EQ(SetSimtGroupMember(il), af::SUCCESS);
    bool metadata_supported = false;
    ASSERT_EQ(FinalizeLoweringMetadata(il, metadata_supported), af::SUCCESS);
    ASSERT_TRUE(metadata_supported);
  }
  auto primary = g.graph.FindNode("indirect_load0");
  ASSERT_EQ(SetSimtFusedPrimary(primary, {"indirect_load0", "indirect_load1"}), af::SUCCESS);
  SimtFusedMergeInfo merge_info;
  merge_info.member_count = 2U;
  ASSERT_EQ(SetSimtFusedMergeInfo(primary, merge_info), af::SUCCESS);
}

class IndirectLoadSimtFusedUbMergeTest : public IndirectLoadConstructFromNodesTest {
 protected:
  // 生成 L3 融合发射并返回 Store 静态方法体区段（Store 签名至 Phase 4 的 ftgt 初始化之间，
  // 该区段内不得出现 tiling 数据指针引用——t 在静态方法作用域不可见）。
  std::string GenerateUbMergeStoreSection(ILTestGraph &g, std::string &generated) {
    AnnotateSimtFusedUbMergeGroupTemplate(g);
    auto il0 = g.graph.FindNode("indirect_load0");
    EXPECT_NE(il0, nullptr);
    Tiler tiler;
    SetupTiler(g.graph, tiler);
    TPipe tpipe("tpipe", tiler);
    tpipe.CollectQues(g.graph);

    IndirectLoadRegApiCall call("IndirectLoadRegApiCall");
    EXPECT_EQ(call.Init(il0), af::SUCCESS);
    ApiTensor x1, x2;
    x1.id = il0->inputs[0].attr.mem.tensor_id;
    x2.id = il0->inputs[1].attr.mem.tensor_id;
    call.inputs.push_back(&x1);
    call.inputs.push_back(&x2);
    for (const char *name : {"x_load0", "idx_load0", "x_load1", "idx_load1", "indirect_load0", "indirect_load1"}) {
      const auto node = g.graph.FindNode(name);
      EXPECT_NE(node, nullptr);
      tpipe.AddTensor(*node->outputs()[0]);
    }

    ascir::AxisId outer_tb = af::kIdNone;
    for (const auto &axis : g.graph.GetAllAxis()) {
      if (axis != nullptr && axis->name == "z2z3Tb") outer_tb = axis->id;
    }
    EXPECT_NE(outer_tb, af::kIdNone);
    EXPECT_EQ(call.Generate(tpipe, {outer_tb}, generated), af::SUCCESS);

    const auto store_pos = generated.find("__simt_callee__ __aicore__ inline static void Store");
    const auto ftgt_pos = generated.find("FusedBody::OutputTargets ftgt = {");
    EXPECT_NE(store_pos, std::string::npos);
    EXPECT_NE(ftgt_pos, std::string::npos);
    EXPECT_LT(store_pos, ftgt_pos);
    // 空区段守卫：任一未命中（或逆序）时返回空串——本 helper 非 void（ASSERT_* 的
    // return; 不可编译），调用方 ASSERT_FALSE(empty()) 干净失败；避免 substr 抛
    // out_of_range（store_pos 为 npos）或按溢出 count 截获错误区段（ftgt_pos 为 npos）。
    if (store_pos == std::string::npos || ftgt_pos == std::string::npos || store_pos >= ftgt_pos) {
      return {};
    }
    return generated.substr(store_pos, ftgt_pos - store_pos);
  }
};

TEST_F(IndirectLoadSimtFusedUbMergeTest, GeneratesUbMergeStoreRowWidthViaContextRuntimeSize) {
  // 动态 shape（s0~s3 符号尺寸）：Store 行宽（output inner size）为运行期符号尺寸
  // （t->sN，tiling 数据指针 t 在 FusedBody 静态方法作用域不可见），必须经 FusedCtx 的
  // rt_size_N 字段传入，否则设备侧编译报 t 未声明（与单 gather 动态 shape 同根因）。
  ILTestGraph g("simt_fused_ub_merge_dynamic");
  BuildSimtFusedGroupGraph(g);
  std::string generated;
  const std::string store_section = GenerateUbMergeStoreSection(g, generated);
  ASSERT_FALSE(store_section.empty());

  // L3 融合发射头：单 evaluator（seq-V1）+ UB 归并。
  EXPECT_NE(generated.find("// IndirectLoad SIMT Fused (seq-V1: single evaluator, serial loads, 2 paths, UB merge)"),
            std::string::npos);
  // L3 OutputTargets：成员 value 的 UB 局部目标；FusedEval 仅逐成员取值（无核内归并树）。
  EXPECT_NE(generated.find("struct OutputTargets { __ubuf__ half *val0; __ubuf__ half *val1; };"), std::string::npos);
  EXPECT_NE(generated.find("out.v0 = value0;"), std::string::npos);
  EXPECT_NE(generated.find("out.v1 = value1;"), std::string::npos);
  EXPECT_EQ(generated.find("out.out0 = "), std::string::npos);
  // FusedCtx 运行期尺寸字段 + magic/shift 配对（kernel 作用域 t 可见处初始化/预算）。
  EXPECT_NE(generated.find("uint64_t rt_size_"), std::string::npos);
  EXPECT_NE(generated.find("uint64_t rt_div_magic_"), std::string::npos);
  EXPECT_NE(generated.find("static_cast<uint64_t>(t->"), std::string::npos);
  EXPECT_NE(generated.find("IndirectLoadGetUintDivMagicAndShift"), std::string::npos);
  // Store 行宽经 FusedCtx 运行期尺寸寻址（行对齐 + 取余），不得渲染 tiling 数据指针引用。
  EXPECT_NE(store_section.find("context.rt_size_"), std::string::npos);
  EXPECT_NE(store_section.find("KernelUtils::SizeAlign("), std::string::npos);
  EXPECT_EQ(store_section.find("t->"), std::string::npos);
  EXPECT_NE(generated.find("AscendC::IndirectLoadSimtFused<FusedBody>(ftgt, fctx"), std::string::npos);
}

TEST_F(IndirectLoadSimtFusedUbMergeTest, KeepsUbMergeStoreRowWidthConstInlineForStaticShape) {
  // 静态 shape：Store 行宽为编译期常量直接内联（uint32/uint64 偏移仅差 static_cast 包装），
  // 注册表为空——不新增 FusedCtx 运行期尺寸字段，与既有静态路径产物一致。
  ILTestGraph g("simt_fused_ub_merge_static", 8L, 64L, 4L, 16L);
  BuildSimtFusedGroupGraph(g);
  std::string generated;
  const std::string store_section = GenerateUbMergeStoreSection(g, generated);
  ASSERT_FALSE(store_section.empty());

  EXPECT_NE(generated.find("// IndirectLoad SIMT Fused (seq-V1: single evaluator, serial loads, 2 paths, UB merge)"),
            std::string::npos);
  // 行宽编译期常量直接内联（精确形态）：偏移位宽 uint32 渲染 local_index / (16)；
  // uint64 经 PromoteSizeExpr 包装为 local_index / (static_cast<uint64_t>(16))——两形态之一。
  EXPECT_TRUE(store_section.find("local_index / (16)") != std::string::npos ||
              store_section.find("local_index / (static_cast<uint64_t>(16))") != std::string::npos)
      << "static UB-merge Store must inline the constant row width";
  EXPECT_NE(store_section.find("KernelUtils::SizeAlign("), std::string::npos);
  EXPECT_EQ(store_section.find("context.rt_size_"), std::string::npos);
  EXPECT_EQ(store_section.find("t->"), std::string::npos);
  EXPECT_EQ(generated.find("uint64_t rt_size_"), std::string::npos);
  EXPECT_EQ(generated.find("static_cast<uint64_t>(t->"), std::string::npos);
}

TEST_F(IndirectLoadConstructFromNodesTest, DeclaresSimtFusedUbMergeGroupAtHoistBeforeFusedTargets) {
  // L3（UB 归并）提升形态回归：fused 调用的 ftgt 逐成员引用 UB 局部目标（含 primary
  // 自身——kSimtOp 无 post-Reduce 时 hoist 守卫不声明，历史依赖祖先 cross-loop 归类）。
  // 全组 tensor 声明必须先于 ftgt 初始化，成员循环体不再独立发射（提前返回 + skip）。
  ILTestGraph g("simt_fused_ub_merge_hoist");
  BuildSimtFusedGroupGraph(g);
  AnnotateSimtFusedUbMergeGroupTemplate(g);
  for (const char *name : {"indirect_load0", "indirect_load1"}) {
    // kSimtOp 无 post-Reduce：FindIndirectLoadOpCall(require_direct_gm) 可发现 → 触发提升。
    ASSERT_EQ(SetTemplateRole(g.graph.FindNode(name), TemplateRole::kSimtOp), af::SUCCESS);
  }
  // 生产形态：组员 value UB 局部目标为共享 buffer 型（TensorAlloc 定义 local_N 变量），
  // 提升点 AllocOutputs 即声明发射点。
  auto set_ub_buf = [&g](const char *name, int64_t buf_id) {
    auto node = g.graph.FindNode(name);
    ASSERT_NE(node, nullptr);
    node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
    node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
    node->outputs[0].attr.mem.position = af::Position::kPositionVecCalc;
    node->outputs[0].attr.que.id = af::kIdNone;
    node->outputs[0].attr.buf.id = buf_id;
    node->outputs[0].attr.opt.merge_scope = af::kIdNone;
  };
  set_ub_buf("indirect_load0", 0);
  set_ub_buf("indirect_load1", 1);

  Kernel kernel("simt_fused_ub_merge_hoist");
  std::string generated;
  ASSERT_EQ(ParseGraphAndGenerateLoop(g, {"x0", "x1", "idx0", "idx1"}, kernel, generated), af::SUCCESS);

  // 提升形态：fused 调用出现，BlockInner 循环体被替换（提前返回）。
  EXPECT_NE(generated.find("// IndirectLoad SIMT Fused"), std::string::npos);
  EXPECT_EQ(generated.find("for (int z2z3Tb"), std::string::npos);
  // L3 ftgt 逐成员引用 UB 局部目标（primary local_1 + 成员 local_8）。
  const auto ftgt_pos = generated.find("FusedBody::OutputTargets ftgt = {");
  ASSERT_NE(ftgt_pos, std::string::npos);
  EXPECT_NE(generated.find("local_1.GetPhyAddr()"), std::string::npos);
  EXPECT_NE(generated.find("local_8.GetPhyAddr()"), std::string::npos);
  // 全组声明（primary 与成员）先于 ftgt 初始化——提升形态下 hoist 守卫不声明 primary，
  // 由组路径补齐（回归：成员/primary 输出 undeclared identifier）。
  const auto primary_decl_pos = generated.find("local_1 =");
  const auto member_decl_pos = generated.find("local_8 =");
  ASSERT_NE(primary_decl_pos, std::string::npos);
  ASSERT_NE(member_decl_pos, std::string::npos);
  EXPECT_LT(primary_decl_pos, ftgt_pos);
  EXPECT_LT(member_decl_pos, ftgt_pos);
  // 成员 ApiCall 置 skip_api_emit（防循环体重复声明）。
  ApiCall *member_call = FindCallByNodeName(kernel.root_loop, "indirect_load1");
  ASSERT_NE(member_call, nullptr);
  EXPECT_TRUE(member_call->skip_api_emit);
}

TEST_F(IndirectLoadConstructFromNodesTest, DeclaresSimtFusedUbMergeGroupInLoopBodyBeforeFusedTargets) {
  // 共享 Reduce 终端的 L3（形态 2）回归：post-Reduce 使 primary 行为复位、不可提升
  // （FindIndirectLoadOpCall 不可发现），fused 调用经循环体路径发射——拓扑序在
  // primary 之后的成员声明此前落在 fused 调用之后（undeclared identifier，末位成员
  // 依赖祖先 cross-loop 归类）。修复后 primary 发射点为其后的成员统一补声明。
  ILTestGraph g("simt_fused_ub_merge_loop_body");
  BuildSimtFusedGroupGraph(g, true);
  AnnotateSimtFusedUbMergeGroupTemplate(g);
  for (const char *name : {"indirect_load0", "indirect_load1"}) {
    ASSERT_EQ(SetTemplateRole(g.graph.FindNode(name), TemplateRole::kSimtOp), af::SUCCESS);
  }
  // 生产形态：组员 value UB 局部目标为共享 buffer 型（TensorAlloc 定义 local_N 变量）。
  auto set_ub_buf = [&g](const char *name, int64_t buf_id) {
    auto node = g.graph.FindNode(name);
    ASSERT_NE(node, nullptr);
    node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
    node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
    node->outputs[0].attr.mem.position = af::Position::kPositionVecCalc;
    node->outputs[0].attr.que.id = af::kIdNone;
    node->outputs[0].attr.buf.id = buf_id;
    node->outputs[0].attr.opt.merge_scope = af::kIdNone;
  };
  set_ub_buf("indirect_load0", 0);
  set_ub_buf("indirect_load1", 1);

  Kernel kernel("simt_fused_ub_merge_loop_body");
  std::string generated;
  ASSERT_EQ(ParseGraphAndGenerateLoop(g, {"x0", "x1", "idx0", "idx1"}, kernel, generated), af::SUCCESS);

  // 循环体路径：BlockInner for 循环仍在（未提升），fused 调用在循环体内发射。
  EXPECT_NE(generated.find("for (int z2z3Tb"), std::string::npos);
  EXPECT_NE(generated.find("// IndirectLoad SIMT Fused"), std::string::npos);
  const auto ftgt_pos = generated.find("FusedBody::OutputTargets ftgt = {");
  ASSERT_NE(ftgt_pos, std::string::npos);
  EXPECT_NE(generated.find("local_1.GetPhyAddr()"), std::string::npos);
  EXPECT_NE(generated.find("local_8.GetPhyAddr()"), std::string::npos);
  // primary（post-Reduce 行为复位，通用 AllocOutputs 声明）与成员（primary 发射点
  // 统一补声明——回归：末位成员 undeclared identifier）的声明均先于 ftgt。
  const auto primary_decl_pos = generated.find("local_1 =");
  const auto member_decl_pos = generated.find("local_8 =");
  ASSERT_NE(primary_decl_pos, std::string::npos);
  ASSERT_NE(member_decl_pos, std::string::npos);
  EXPECT_LT(primary_decl_pos, ftgt_pos);
  EXPECT_LT(member_decl_pos, ftgt_pos);
  // 位置在后的成员由 primary 发射点补声明并置 skip（防其循环体位置重复声明）。
  ApiCall *member_call = FindCallByNodeName(kernel.root_loop, "indirect_load1");
  ASSERT_NE(member_call, nullptr);
  EXPECT_TRUE(member_call->skip_api_emit);
}

// L4（V2 逐成员发射）标注：成员携带组员标记与 lowering 元数据，但不设 fused
// primary——各成员走标准单 gather 发射（codegen 完整复用单 gather 机制）。组员形态
// post-Reduce 置空、value 直写 UB 局部目标（输出链为合成 local-target 链）。
void AnnotateSimtGroupMemberTemplate(ILTestGraph &g) {
  af::AxisId outer_id = af::kIdNone;
  for (const auto &axis : g.graph.GetAllAxis()) {
    if (axis != nullptr && axis->name == "z2z3") outer_id = axis->id;
  }
  ASSERT_NE(outer_id, af::kIdNone);
  const auto axes = g.graph.GetAllAxis();
  const af::Expression One = af::sym::kSymbolOne;
  TemplateLogicalView view;
  view.input.axis_ids = {axes[0]->id, axes[1]->id};
  view.input.sizes = {g.graph.FindAxis(axes[0]->id)->size, g.graph.FindAxis(axes[1]->id)->size};
  view.input.strides = {g.graph.FindAxis(axes[1]->id)->size, One};
  view.index.axis_ids = {axes[2]->id, axes[3]->id};
  view.index.sizes = {g.graph.FindAxis(axes[2]->id)->size, g.graph.FindAxis(axes[3]->id)->size};
  view.index.strides = {g.graph.FindAxis(axes[3]->id)->size, One};
  view.output.axis_ids = {axes[2]->id, axes[3]->id};
  view.output.sizes = {g.graph.FindAxis(axes[2]->id)->size, g.graph.FindAxis(axes[3]->id)->size};
  view.output.strides = {g.graph.FindAxis(axes[3]->id)->size, One};

  for (const char *name : {"x_load0", "x_load1"}) {
    ASSERT_EQ(SetTemplateRole(g.graph.FindNode(name), TemplateRole::kSimtInputBoundary), af::SUCCESS);
  }
  for (const char *name : {"idx_load0", "idx_load1"}) {
    ASSERT_EQ(SetTemplateRole(g.graph.FindNode(name), TemplateRole::kSimtDirectGmBoundary), af::SUCCESS);
  }
  auto il = g.graph.FindNode("indirect_load0");
  ASSERT_NE(il, nullptr);
  ASSERT_EQ(::ascir::SetTemplateId(il, ::ascir::TemplateId::kIndirectLoadSimt), af::SUCCESS);
  ASSERT_EQ(SetImplementation(il, Implementation::kDefault), af::SUCCESS);
  TemplateAxes t{outer_id, af::kIdNone, af::kIdNone};
  ASSERT_EQ(SetTemplateAxes(il, t), af::SUCCESS);
  ASSERT_EQ(ClassifyIndirectLoadLayout(view.input, view.input), af::SUCCESS);
  ASSERT_EQ(ClassifyIndirectLoadLayout(view.index, view.index), af::SUCCESS);
  ASSERT_EQ(SetTemplateLogicalView(il, view), af::SUCCESS);
  ASSERT_EQ(SetSimtGroupMember(il), af::SUCCESS);
  bool metadata_supported = false;
  ASSERT_EQ(FinalizeLoweringMetadata(il, metadata_supported), af::SUCCESS);
  ASSERT_TRUE(metadata_supported);
}

TEST_F(IndirectLoadConstructFromNodesTest, GeneratesGroupMemberStoreRowWidthViaContextRuntimeSize) {
  // L4 逐成员发射 + 动态 shape（s0~s3 符号尺寸）：组员 Store（local-target 行对齐
  // 寻址）的行宽在静态方法体内消费，动态 shape 下是运行期符号尺寸（t->sN），必须经
  // Context 的 rt_size_N 字段传入——组员形态 post-Reduce 置空，不走 post-Reduce 主
  // 路径的 output_inner_size 专用字段，否则设备侧编译报 t 未声明。
  // Body/Context 定义经 GenerateFuncDefinition 发射（resolver 的 Generate 只发射
  // kernel 作用域的 Context 初始化与调用段）。
  ILTestGraph g("simt_group_member_dynamic");
  BuildSimtFusedGroupGraph(g);
  AnnotateSimtGroupMemberTemplate(g);

  auto il0 = g.graph.FindNode("indirect_load0");
  ASSERT_NE(il0, nullptr);
  Tiler tiler;
  SetupTiler(g.graph, tiler);
  TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(g.graph);

  IndirectLoadRegApiCall call("IndirectLoadRegApiCall");
  ASSERT_EQ(call.Init(il0), af::SUCCESS);
  ApiTensor x1, x2;
  x1.id = il0->inputs[0].attr.mem.tensor_id;
  x2.id = il0->inputs[1].attr.mem.tensor_id;
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  for (const char *name : {"x_load0", "idx_load0", "indirect_load0"}) {
    const auto node = g.graph.FindNode(name);
    ASSERT_NE(node, nullptr);
    tpipe.AddTensor(*node->outputs()[0]);
  }

  std::stringstream definition_stream;
  ASSERT_EQ(call.GenerateFuncDefinition(tpipe, tiler, definition_stream), af::SUCCESS);
  const std::string definition = definition_stream.str();
  ascir::AxisId outer_tb = af::kIdNone;
  for (const auto &axis : g.graph.GetAllAxis()) {
    if (axis != nullptr && axis->name == "z2z3Tb") outer_tb = axis->id;
  }
  ASSERT_NE(outer_tb, af::kIdNone);
  std::string invocation;
  ASSERT_EQ(call.Generate(tpipe, {outer_tb}, invocation), af::SUCCESS);

  // 定义段：成员 Store 行宽经 Context 运行期尺寸寻址——除法操作数直接是 t-> 是
  // 成员形态的历史 bug 形态，不得出现。
  EXPECT_EQ(definition.find("local_index / (t->"), std::string::npos);
  EXPECT_NE(definition.find("local_index / (static_cast<"), std::string::npos);
  // 定义段：Context 携带 rt_size_N 字段（空跑收集含 Store 发射）。
  EXPECT_NE(definition.find("uint64_t rt_size_"), std::string::npos);
  // 组员形态不走 post-Reduce 主路径的专用字段。
  EXPECT_EQ(definition.find("output_inner_size"), std::string::npos);
  // 调用段：Context 初始化在 kernel 作用域（t 可见）取 tiler 表达式。
  EXPECT_NE(invocation.find("static_cast<uint64_t>(t->"), std::string::npos);
}

TEST_F(IndirectLoadConstructFromNodesTest, ConnectsRedirectedSimtOutputToReduce) {
  ILTestGraph g("simt_post_construct");
  BuildSimtPostReduceGraph(g);
  AnnotateSimtTemplate(g.graph);

  auto x = g.graph.FindNode("x");
  auto idx = g.graph.FindNode("idx");
  auto addend = g.graph.FindNode("addend");
  auto y = g.graph.FindNode("y");
  auto il = g.graph.FindNode("indirect_load");
  auto output_transform = g.graph.FindNode("output_transform");
  ASSERT_NE(x, nullptr);
  ASSERT_NE(idx, nullptr);
  ASSERT_NE(addend, nullptr);
  ASSERT_NE(y, nullptr);
  ASSERT_NE(il, nullptr);
  ASSERT_NE(output_transform, nullptr);
  ASSERT_EQ(SetTemplateRole(il, TemplateRole::kSimtOp), af::SUCCESS);
  ASSERT_EQ(SetTemplateRole(output_transform, TemplateRole::kSimtInlineTransform), af::SUCCESS);

  ascir::FusedScheduledResult result;
  result.input_nodes = {x, idx, addend};
  result.output_nodes = {y};
  Kernel kernel("simt_post_construct");
  ASSERT_EQ(Kernel::ParseGraph(g.graph, result, kernel), af::SUCCESS);

  ApiCall *il_call = FindCallByNodeName(kernel.root_loop, "indirect_load");
  ApiCall *transform_call = FindCallByNodeName(kernel.root_loop, "output_transform");
  ApiCall *reduce_call = FindCallByNodeName(kernel.root_loop, "reduce");
  ASSERT_NE(il_call, nullptr);
  ASSERT_NE(transform_call, nullptr);
  ASSERT_NE(reduce_call, nullptr);
  ASSERT_EQ(il_call->outputs.size(), 1UL);
  ASSERT_EQ(reduce_call->inputs.size(), 1UL);
  const ascir::TensorId redirected_id = output_transform->outputs[0].attr.mem.tensor_id;
  const Tensor *redirected_tensor = kernel.tpipe.GetTensor(redirected_id);
  ASSERT_NE(redirected_tensor, nullptr);
  EXPECT_TRUE(transform_call->skip_api_emit);
  EXPECT_EQ(il_call->outputs[0].id, redirected_id);
  EXPECT_EQ(il_call->outputs[0].reuse_id, redirected_tensor->reuse_id);
  EXPECT_EQ(reduce_call->inputs[0], &il_call->outputs[0]);
}

TEST_F(IndirectLoadConstructFromNodesTest, ConnectsRedirectedSimtPostReduceWithParallelStore) {
  ILTestGraph g("simt_post_parallel_construct");
  BuildSimtPostReduceGraph(g, true);
  AnnotateSimtTemplate(g.graph);

  auto il = g.graph.FindNode("indirect_load");
  auto output_transform = g.graph.FindNode("output_transform");
  ASSERT_NE(il, nullptr);
  ASSERT_NE(output_transform, nullptr);
  ASSERT_EQ(SetTemplateRole(il, TemplateRole::kSimtOp), af::SUCCESS);
  ASSERT_EQ(SetTemplateRole(output_transform, TemplateRole::kSimtInlineTransform), af::SUCCESS);

  ascir::FusedScheduledResult result;
  result.input_nodes = {g.graph.FindNode("x"), g.graph.FindNode("idx"), g.graph.FindNode("addend")};
  result.output_nodes = {g.graph.FindNode("y"), g.graph.FindNode("direct_y")};
  Kernel kernel("simt_post_parallel_construct");
  ASSERT_EQ(Kernel::ParseGraph(g.graph, result, kernel), af::SUCCESS);

  ApiCall *il_call = FindCallByNodeName(kernel.root_loop, "indirect_load");
  ApiCall *transform_call = FindCallByNodeName(kernel.root_loop, "output_transform");
  ApiCall *reduce_call = FindCallByNodeName(kernel.root_loop, "reduce");
  ApiCall *direct_store_call = FindCallByNodeName(kernel.root_loop, "direct_store");
  ASSERT_NE(il_call, nullptr);
  ASSERT_NE(transform_call, nullptr);
  ASSERT_NE(reduce_call, nullptr);
  ASSERT_NE(direct_store_call, nullptr);
  ASSERT_EQ(il_call->outputs.size(), 1UL);
  ASSERT_EQ(reduce_call->inputs.size(), 1UL);
  EXPECT_TRUE(transform_call->skip_api_emit);
  EXPECT_TRUE(direct_store_call->skip_api_emit);
  EXPECT_EQ(il_call->outputs[0].id, output_transform->outputs[0].attr.mem.tensor_id);
  EXPECT_EQ(reduce_call->inputs[0], &il_call->outputs[0]);

  std::string generated;
  ASSERT_EQ(kernel.root_loop.Generate(kernel.tiler, kernel.tpipe, generated), af::SUCCESS);
  EXPECT_NE(generated.find("AscendC::IndirectLoadSimt<"), std::string::npos);
  EXPECT_NE(generated.find("global_10.GetPhyAddr()"), std::string::npos);
}

TEST_F(IndirectLoadConstructFromNodesTest, FinalizesSimtMetadataWithRolelessNddmaSideInput) {
  // 回归场景：复合 Norm 普通调度侧输入经 NddmaTemplate::GenNddmaNode 就地转 Nddma
  // （SetType + 吸收 Broadcast 视图）后不携带 SIMT 角色。生产图中 Reduce 直接消费
  // IL，该 Nddma 只出现在含 Reduce 的被丢弃链中，SIMT 元数据后向 walk 必须把
  // Nddma 视为 GM 搬运边界，而不是穿过它走到 Data 图输入端子，触发 "has no
  // input" 断言导致整个 Codegen parse graph failed。
  ILTestGraph g("simt_post_nddma_side");
  BuildSimtPostReduceGraph(g);
  AnnotateSimtTemplate(g.graph);

  auto il = g.graph.FindNode("indirect_load");
  auto output_transform = g.graph.FindNode("output_transform");
  auto reduce = g.graph.FindNode("reduce");
  auto store = g.graph.FindNode("store");
  ASSERT_NE(il, nullptr);
  ASSERT_NE(output_transform, nullptr);
  ASSERT_NE(reduce, nullptr);
  ASSERT_NE(store, nullptr);
  ASSERT_EQ(SetTemplateRole(il, TemplateRole::kSimtOp), af::SUCCESS);
  ASSERT_EQ(SetTemplateRole(output_transform, TemplateRole::kSimtInlineTransform), af::SUCCESS);

  // 重绑为生产拓扑：Reduce 直接消费 IL（post-Reduce 输入生产者即 IL 本身），侧输入
  // 在 post-Reduce 区域汇入（output_transform: reduce + addend），store 消费该汇入结果。
  auto reduce_in = reduce->GetInDataAnchor(0UL);
  auto transform_in = output_transform->GetInDataAnchor(0UL);
  auto store_in = store->GetInDataAnchor(0UL);
  reduce_in->UnlinkAll();
  transform_in->UnlinkAll();
  store_in->UnlinkAll();
  ASSERT_EQ(reduce_in->LinkFrom(il->GetOutDataAnchor(0UL)), af::GRAPH_SUCCESS);
  ASSERT_EQ(transform_in->LinkFrom(reduce->GetOutDataAnchor(0UL)), af::GRAPH_SUCCESS);
  ASSERT_EQ(store_in->LinkFrom(output_transform->GetOutDataAnchor(0UL)), af::GRAPH_SUCCESS);

  // 与生产 NddmaTemplate::GenNddmaNode 的就地转换一致：单引用侧输入 Load 转 Nddma
  // 后不携带 SIMT 角色（普通调度区域节点统一不带角色，避免 no API call found）。
  // AnnotateSimtTemplate 会给侧输入 Load 标 kSimtDirectGmBoundary，需显式还原为
  // 无角色状态，否则 walk 在角色判定处截断，覆盖不到 Nddma 类型边界本身。
  auto side_node = g.graph.FindNode("addend_load");
  ASSERT_NE(side_node, nullptr);
  side_node->GetOpDesc()->SetType("Nddma");
  side_node->attr.type = "Nddma";
  ASSERT_EQ(SetTemplateRole(side_node, TemplateRole::kNone), af::SUCCESS);
  ASSERT_TRUE(af::ops::IsOps<af::ascir_op::Nddma>(side_node));

  bool is_supported = false;
  ASSERT_EQ(ascgen_utils::indirect_load::FinalizeLoweringMetadata(il, is_supported), af::SUCCESS);
  EXPECT_TRUE(is_supported);
}

TEST_F(IndirectLoadConstructFromNodesTest, GeneratesSimdLoopWithUbLifecycleAndNoLoadStoreSync) {
  ILTestGraph g("simd_tile_inner");
  BuildSimdGraph(g);
  AnnotateSimdTemplate(g.graph);

  Kernel kernel("simd_tile_inner");
  std::string generated;
  ASSERT_EQ(ParseGraphAndGenerateLoop(g, {"x", "idx"}, kernel, generated), af::SUCCESS);
  const size_t actual_size = generated.find("z2z3t_actual_size = ");
  const size_t indirect_load = generated.find("// IndirectLoad SIMD");
  ASSERT_NE(actual_size, std::string::npos);
  ASSERT_NE(indirect_load, std::string::npos);
  EXPECT_LT(actual_size, indirect_load);
  EXPECT_NE(generated.find("AllocTensor"), std::string::npos);
  EXPECT_NE(generated.find("EnQue"), std::string::npos);
  EXPECT_NE(generated.find("DeQue"), std::string::npos);
  EXPECT_EQ(generated.find("MTE2_MTE3"), std::string::npos);
}

TEST_F(IndirectLoadConstructFromNodesTest, GeneratesTileInnerOffsetFromTileOuterNotBlockOffset) {
  // IndirectLoad 固定 tile 的轴结构：merged(outer) → TileSplit[T × t] → BlockSplit 作用在 T 上。
  // TileInner 循环展开 merged 时必须由 TileOuter 与 TileInner 组合（outer = T * t_size + t），
  // 复用 BlockInner 的 block_dim_offset 会丢失 outerTb 维度，导致多行 block 内 data 行地址不推进。
  ILTestGraph g("tile_inner_offset");
  auto z2z3 = g.graph.MergeAxis({g.z2.id, g.z3.id});
  auto [z2z3T, z2z3t] = g.graph.TileSplit(z2z3->id);
  auto [z2z3TB, z2z3Tb] = g.graph.BlockSplit(z2z3T->id);
  (void)z2z3TB;

  Tiler tiler;
  SetupTiler(g.graph, tiler);

  const auto tile_inner_code = tiler.CalcFromAxis(z2z3t->id);
  EXPECT_NE(tile_inner_code.find("z2z3 = z2z3T *"), std::string::npos);
  EXPECT_EQ(tile_inner_code.find("z2z3 = block_dim_offset"), std::string::npos);

  // BlockInner 的来源轴（z2z3T）仍应使用 block_dim_offset 组合。
  const auto block_inner_code = tiler.CalcFromAxis(z2z3Tb->id);
  EXPECT_NE(block_inner_code.find("z2z3T = block_dim_offset + z2z3Tb"), std::string::npos);
}

TEST_F(IndirectLoadConstructFromNodesTest, GeneratesSimtDirectGmCallOutsideBlockInnerLoop) {
  ILTestGraph g("simt_block_inner");
  BuildSimtGraph(g);
  AnnotateSimtTemplate(g.graph);
  auto il = g.graph.FindNode("indirect_load");
  ASSERT_NE(il, nullptr);
  ASSERT_EQ(SetTemplateRole(il, TemplateRole::kSimtOp), af::SUCCESS);

  Kernel kernel("simt_block_inner");
  std::string generated;
  ASSERT_EQ(ParseGraphAndGenerateLoop(g, {"x", "idx"}, kernel, generated), af::SUCCESS);
  const size_t block_offset = generated.find("block_dim_offset");
  const size_t indirect_load = generated.find("// IndirectLoad SIMT");
  ASSERT_NE(block_offset, std::string::npos);
  ASSERT_NE(indirect_load, std::string::npos);
  EXPECT_LT(block_offset, indirect_load);
  EXPECT_EQ(generated.find("for (int z2z3Tb"), std::string::npos);
  EXPECT_EQ(generated.find("AllocTensor"), std::string::npos);
  EXPECT_EQ(generated.find("EnQue"), std::string::npos);
  EXPECT_EQ(generated.find("DeQue"), std::string::npos);
}

TEST_F(IndirectLoadConstructFromNodesTest, GeneratesSimtMultiOutputCall) {
  ILTestGraph g("simt_multi_output_call");
  BuildSimtGraph(g, true);
  AnnotateSimtTemplate(g.graph);
  auto il = g.graph.FindNode("indirect_load");
  ASSERT_NE(il, nullptr);
  ASSERT_EQ(SetTemplateRole(il, TemplateRole::kSimtOp), af::SUCCESS);

  Kernel kernel("simt_multi_output_call");
  std::string generated;
  ASSERT_EQ(ParseGraphAndGenerateLoop(g, {"x", "idx"}, kernel, generated, {"y", "second_y"}), af::SUCCESS);
  EXPECT_NE(generated.find("AscendC::IndirectLoadSimt<"), std::string::npos);
  EXPECT_NE(generated.find("OutputTargets"), std::string::npos);
  EXPECT_NE(generated.find("global_2.GetPhyAddr()"), std::string::npos);
  EXPECT_NE(generated.find("global_10.GetPhyAddr()"), std::string::npos);
}

TEST_F(IndirectLoadConstructFromNodesTest, GeneratesStaticPowerOfTwoPolicyWithUint32Offsets) {
  ILTestGraph g("simt_static_power_of_two", 2, 8, 2, 8);
  BuildSimtGraph(g);
  AnnotateSimtTemplate(g.graph);
  auto il = g.graph.FindNode("indirect_load");
  ASSERT_NE(il, nullptr);
  ASSERT_EQ(SetTemplateRole(il, TemplateRole::kSimtOp), af::SUCCESS);

  Kernel kernel("simt_static_power_of_two");
  std::string generated;
  ASSERT_EQ(ParseGraphAndGenerateLoop(g, {"x", "idx"}, kernel, generated), af::SUCCESS);
  EXPECT_NE(generated.find("IndirectLoadSimtCase>(0), uint32_t, 2, 1, 1ULL, 8ULL, 1ULL, 8ULL"), std::string::npos);
  EXPECT_NE(generated.find("IndirectLoadSimtParams<IndirectLoadSimtCase> indirect_load_simt_params{}"),
            std::string::npos);
}

TEST_F(IndirectLoadConstructFromNodesTest, GeneratesStaticInnerPolicyForLastAxis) {
  ILTestGraph g("simt_static_inner", 2, 7, 2, 7);
  BuildSimtGraph(g);
  AnnotateSimtTemplate(g.graph);
  auto il = g.graph.FindNode("indirect_load");
  ASSERT_NE(il, nullptr);
  ASSERT_EQ(SetTemplateRole(il, TemplateRole::kSimtOp), af::SUCCESS);

  Kernel kernel("simt_static_inner");
  std::string generated;
  ASSERT_EQ(ParseGraphAndGenerateLoop(g, {"x", "idx"}, kernel, generated), af::SUCCESS);
  EXPECT_NE(generated.find("IndirectLoadSimtCase>(1), uint32_t, 2, 1, 1ULL, 7ULL, 1ULL, 7ULL"), std::string::npos);
  EXPECT_NE(generated.find("indirect_load_simt_params{static_cast<uint32_t>(7)}"), std::string::npos);
}

TEST_F(IndirectLoadConstructFromNodesTest, GeneratesSimtPostReduceCallWithUbOutput) {
  ILTestGraph g("simt_post_reduce");
  BuildSimtPostReduceGraph(g);
  AnnotateSimtTemplate(g.graph);
  auto il = g.graph.FindNode("indirect_load");
  auto output_transform = g.graph.FindNode("output_transform");
  ASSERT_NE(il, nullptr);
  ASSERT_NE(output_transform, nullptr);
  ASSERT_EQ(SetTemplateRole(il, TemplateRole::kSimtOp), af::SUCCESS);
  ASSERT_EQ(SetTemplateRole(output_transform, TemplateRole::kSimtInlineTransform), af::SUCCESS);

  Kernel kernel("simt_post_reduce");
  std::string generated;
  ASSERT_EQ(ParseGraphAndGenerateLoop(g, {"x", "idx", "addend"}, kernel, generated), af::SUCCESS);
  EXPECT_NE(generated.find("// IndirectLoad SIMT"), std::string::npos);
  EXPECT_NE(generated.find("AllocTensor"), std::string::npos);
  EXPECT_NE(generated.find("static_cast<uint32_t>"), std::string::npos);
  EXPECT_EQ(generated.find("y_ptr"), std::string::npos);
}

// ==================== fused member input tensor resolution ====================

void BuildFusedMemberInputGraph(ILTestGraph &g) {
  const af::Expression One = af::sym::kSymbolOne;
  Data x_data("x", g.graph);
  x_data.ir_attr.SetIndex(0);
  x_data.y.dtype = ge::DT_FLOAT16;
  x_data.attr.sched.axis = {g.z0.id, g.z1.id};
  *x_data.y.axis = {g.z0.id, g.z1.id};
  *x_data.y.repeats = {g.s0, g.s1};
  *x_data.y.strides = {g.s1, One};

  Load x_load("x_load");
  g.graph.AddNode(x_load);
  x_load.x = x_data.y;
  x_load.y.dtype = ge::DT_FLOAT16;
  x_load.attr.sched.axis = {g.z0.id, g.z1.id};
  *x_load.y.axis = {g.z0.id, g.z1.id};
  *x_load.y.repeats = {g.s0, g.s1};
  *x_load.y.strides = {g.s1, One};

  Data idx_data("idx", g.graph);
  idx_data.ir_attr.SetIndex(1);
  idx_data.y.dtype = ge::DT_INT32;
  idx_data.attr.sched.axis = {g.z2.id, g.z3.id};
  *idx_data.y.axis = {g.z2.id, g.z3.id};
  *idx_data.y.repeats = {g.s2, g.s3};
  *idx_data.y.strides = {g.s3, One};

  Load idx_load("idx_load");
  g.graph.AddNode(idx_load);
  idx_load.x = idx_data.y;
  idx_load.y.dtype = ge::DT_INT32;
  idx_load.attr.sched.axis = {g.z2.id, g.z3.id};
  *idx_load.y.axis = {g.z2.id, g.z3.id};
  *idx_load.y.repeats = {g.s2, g.s3};
  *idx_load.y.strides = {g.s3, One};

  IndirectLoad il("indirect_load");
  g.graph.AddNode(il);
  il.x1 = x_load.y;
  il.x2 = idx_load.y;
  il.ir_attr.SetAxis(1);
  il.attr.sched.axis = {g.z2.id, g.z3.id};
  *il.y.axis = {g.z2.id, g.z3.id};
  *il.y.repeats = {g.s2, g.s3};
  *il.y.strides = {g.s3, One};

  // direct-GM 输入边界（kSimtInputBoundary 语义）：skips_api_emit、不注册 tpipe、
  // 不生成 api call——fused 路径按视图 id 直查会失败，须经回溯解析。
  ASSERT_EQ(SetTemplateRole(g.graph.FindNode("x_load"), TemplateRole::kSimtInputBoundary), af::SUCCESS);
  ASSERT_EQ(SetTemplateRole(g.graph.FindNode("indirect_load"), TemplateRole::kSimtOp), af::SUCCESS);
  // 显式分配输出 tensor id（未调度图的默认 id 不区分，无法构造"边界视图已分配、
  // GM 源已注册"的调度后状态）；resolver 首次查找读直接生产者（x_load）的输出 id。
  g.graph.FindNode("x")->outputs[0].attr.mem.tensor_id = 0;
  g.graph.FindNode("x_load")->outputs[0].attr.mem.tensor_id = 4;
  g.graph.FindNode("idx")->outputs[0].attr.mem.tensor_id = 3;
  g.graph.FindNode("idx_load")->outputs[0].attr.mem.tensor_id = 5;
  g.graph.FindNode("indirect_load")->outputs[0].attr.mem.tensor_id = 1;
}

TEST(IndirectLoadApiCallTest, ResolveFusedMemberInputTensorWalksSkippedBoundaryToData) {
  ILTestGraph g("fused_member_input_walk");
  BuildFusedMemberInputGraph(g);
  const auto member = g.graph.FindNode("indirect_load");
  const auto data_node = g.graph.FindNode("x");
  const auto load_node = g.graph.FindNode("x_load");
  ASSERT_NE(member, nullptr);
  ASSERT_NE(data_node, nullptr);
  ASSERT_NE(load_node, nullptr);

  Tiler tiler;
  SetupTiler(g.graph, tiler);
  TPipe tpipe("tpipe", tiler);
  // 仅注册 GM 源（Data 输出张量）——模拟 input_nodes 注册路径；边界 Load 不注册。
  tpipe.AddTensor(*data_node->outputs()[0]);

  const auto *resolved = codegen::ResolveFusedMemberInputTensor(tpipe, member);
  ASSERT_NE(resolved, nullptr);
  EXPECT_EQ(resolved->id, data_node->outputs()[0]->attr.mem.tensor_id);
}

TEST(IndirectLoadApiCallTest, ResolveFusedMemberInputTensorPrefersRegisteredViewTensor) {
  ILTestGraph g("fused_member_input_direct");
  BuildFusedMemberInputGraph(g);
  const auto member = g.graph.FindNode("indirect_load");
  const auto data_node = g.graph.FindNode("x");
  const auto load_node = g.graph.FindNode("x_load");
  ASSERT_NE(member, nullptr);
  ASSERT_NE(data_node, nullptr);
  ASSERT_NE(load_node, nullptr);

  Tiler tiler;
  SetupTiler(g.graph, tiler);
  TPipe tpipe("tpipe", tiler);
  // 成员 value 输入视图张量已注册（无边界或边界已注册形态）：直接命中，不回溯。
  tpipe.AddTensor(*load_node->outputs()[0]);
  tpipe.AddTensor(*data_node->outputs()[0]);

  const auto *resolved = codegen::ResolveFusedMemberInputTensor(tpipe, member);
  ASSERT_NE(resolved, nullptr);
  EXPECT_EQ(resolved->id, load_node->outputs()[0]->attr.mem.tensor_id);
}

TEST(IndirectLoadApiCallTest, ResolveFusedMemberInputTensorFailsLoudlyOnMultiInputValueNode) {
  // 多输入 skips_api_emit 节点（Select/Mul 等）出现在 value 链上：值不再是单一 GM
  // 表，fused 单 GM 值表指针模型不支持——必须返回 nullptr 显式失败，禁止沿 input 0
  // 猜测分支把无关 Data 当值表（静默错码）。
  ILTestGraph g("fused_member_input_multi");
  const af::Expression One = af::sym::kSymbolOne;
  Data x_data("x", g.graph);
  x_data.ir_attr.SetIndex(0);
  x_data.y.dtype = ge::DT_FLOAT16;
  x_data.attr.sched.axis = {g.z0.id, g.z1.id};
  *x_data.y.axis = {g.z0.id, g.z1.id};
  *x_data.y.repeats = {g.s0, g.s1};
  *x_data.y.strides = {g.s1, One};

  Load x_load("x_load");
  g.graph.AddNode(x_load);
  x_load.x = x_data.y;
  x_load.y.dtype = ge::DT_FLOAT16;
  x_load.attr.sched.axis = {g.z0.id, g.z1.id};
  *x_load.y.axis = {g.z0.id, g.z1.id};
  *x_load.y.repeats = {g.s0, g.s1};
  *x_load.y.strides = {g.s1, One};

  Data idx_data("idx", g.graph);
  idx_data.ir_attr.SetIndex(1);
  idx_data.y.dtype = ge::DT_INT32;
  idx_data.attr.sched.axis = {g.z2.id, g.z3.id};
  *idx_data.y.axis = {g.z2.id, g.z3.id};
  *idx_data.y.repeats = {g.s2, g.s3};
  *idx_data.y.strides = {g.s3, One};

  Load idx_load("idx_load");
  g.graph.AddNode(idx_load);
  idx_load.x = idx_data.y;
  idx_load.y.dtype = ge::DT_INT32;
  idx_load.attr.sched.axis = {g.z2.id, g.z3.id};
  *idx_load.y.axis = {g.z2.id, g.z3.id};
  *idx_load.y.repeats = {g.s2, g.s3};
  *idx_load.y.strides = {g.s3, One};

  // value 链：IL.x1 ← Select(cond, x_load, x_load)——三输入 skips_api_emit 节点。
  Select value_select("value_select");
  g.graph.AddNode(value_select);
  value_select.x1 = x_load.y;
  value_select.x2 = x_load.y;
  value_select.x3 = x_load.y;
  value_select.y.dtype = ge::DT_FLOAT16;
  value_select.attr.sched.axis = {g.z2.id, g.z3.id};
  *value_select.y.axis = {g.z2.id, g.z3.id};
  *value_select.y.repeats = {g.s2, g.s3};
  *value_select.y.strides = {g.s3, One};

  IndirectLoad il("indirect_load");
  g.graph.AddNode(il);
  il.x1 = value_select.y;
  il.x2 = idx_load.y;
  il.ir_attr.SetAxis(1);
  il.attr.sched.axis = {g.z2.id, g.z3.id};
  *il.y.axis = {g.z2.id, g.z3.id};
  *il.y.repeats = {g.s2, g.s3};
  *il.y.strides = {g.s3, One};

  ASSERT_EQ(SetTemplateRole(g.graph.FindNode("x_load"), TemplateRole::kSimtInputBoundary), af::SUCCESS);
  ASSERT_EQ(SetTemplateRole(g.graph.FindNode("value_select"), TemplateRole::kSimtInlineTransform), af::SUCCESS);
  ASSERT_EQ(SetTemplateRole(g.graph.FindNode("indirect_load"), TemplateRole::kSimtOp), af::SUCCESS);
  // 显式分配输出 tensor id：成员 value 输入视图读直接生产者（value_select）的输出
  // id=7（未注册）；GM 源（x）id=0 已注册——验证多输入节点挡住回溯、不得取到 x。
  g.graph.FindNode("x")->outputs[0].attr.mem.tensor_id = 0;
  g.graph.FindNode("x_load")->outputs[0].attr.mem.tensor_id = 4;
  g.graph.FindNode("value_select")->outputs[0].attr.mem.tensor_id = 7;
  g.graph.FindNode("idx")->outputs[0].attr.mem.tensor_id = 3;
  g.graph.FindNode("idx_load")->outputs[0].attr.mem.tensor_id = 5;

  Tiler tiler;
  SetupTiler(g.graph, tiler);
  TPipe tpipe("tpipe", tiler);
  // GM 源已注册，但 value 链上的多输入 Select 挡住回溯——不得猜测分支取 Data。
  tpipe.AddTensor(*g.graph.FindNode("x")->outputs()[0]);

  const auto *resolved = codegen::ResolveFusedMemberInputTensor(tpipe, g.graph.FindNode("indirect_load"));
  EXPECT_EQ(resolved, nullptr);
}
