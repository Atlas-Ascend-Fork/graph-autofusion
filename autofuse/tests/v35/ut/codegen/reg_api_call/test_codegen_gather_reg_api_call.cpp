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
#include "codegen_kernel.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "common_utils.h"
#include "utils/api_call_factory.h"
#include "../reg_api_call/reg_gather_api_call.h"
#include "graph_optimizer/graph_fusion/graph_pass.h"

using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;
using namespace codegen;

template <af::DataType T>
void CreateGraph(af::AscGraph &graph, int axis = 0) {
  af::Expression One = af::Symbol(1);
  af::Expression param_last_exp = af::Symbol(4000);
  af::Expression indices_first_exp = af::Symbol(100);
  af::Expression indices_second_exp = af::Symbol(100);

  auto z0 = graph.CreateAxis("z0", param_last_exp);
  auto z1 = graph.CreateAxis("z1", indices_first_exp);
  auto z2 = graph.CreateAxis("z2", indices_second_exp);

  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Data x2("x2", graph);
  af::ascir_op::Gather gather("gather");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");

  x1.attr.sched.axis = {z0.id};
  x1.y.dtype = T;
  *x1.y.axis = {z0.id};
  *x1.y.repeats = {param_last_exp};
  *x1.y.strides = {One};

  x2.attr.sched.axis = {z1.id, z2.id};
  x2.y.dtype = af::DT_INT64;
  *x2.y.axis = {z1.id, z2.id};
  *x2.y.repeats = {indices_first_exp, indices_second_exp};
  *x2.y.strides = {indices_second_exp, One};

  gather.x1 = x1.y;
  gather.x2 = x2.y;
  gather.attr.sched.axis = {z1.id, z2.id};
  gather.y.dtype = T;
  *gather.y.axis = {z1.id, z2.id};
  *gather.y.repeats = {indices_first_exp, indices_second_exp};
  *gather.y.strides = {indices_second_exp, One};
  gather.ir_attr.SetAxis(axis);
  gather.ir_attr.SetNegative_index_support(false);

  store.x = gather.y;
  store.attr.sched.axis = {z1.id, z2.id};
  store.y.dtype = T;
  *store.y.axis = {z1.id, z2.id};
  *store.y.repeats = {indices_first_exp, indices_second_exp};
  *store.y.strides = {indices_second_exp, One};

  y.x = store.y;
  y.attr.sched.axis = {z1.id, z2.id};
  y.y.dtype = T;
  *y.y.axis = {z1.id, z2.id};
  *y.y.repeats = {indices_first_exp, indices_second_exp};
  *y.y.strides = {indices_second_exp, One};
}

template <af::DataType T>
void CreateManyAxisGraph(af::AscGraph &graph, std::vector<af::Axis> &axes, std::vector<af::Expression> &exps) {
  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Data x2("x2", graph);
  af::ascir_op::Gather gather("gather");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");

  x1.attr.sched.axis = {axes[0].id, axes[1].id};
  x1.y.dtype = T;
  *x1.y.axis = {axes[0].id, axes[1].id};
  *x1.y.repeats = {exps[0], exps[1]};
  *x1.y.strides = {exps[1], exps[4]};

  x2.attr.sched.axis = {axes[2].id, axes[3].id};
  x2.y.dtype = af::DT_INT64;
  *x2.y.axis = {axes[2].id, axes[3].id};
  *x2.y.repeats = {exps[2], exps[3]};
  *x2.y.strides = {exps[3], exps[4]};

  gather.x1 = x1.y;
  gather.x2 = x2.y;
  gather.attr.sched.axis = {axes[2].id, axes[3].id};
  gather.y.dtype = T;
  *gather.y.axis = {axes[2].id, axes[3].id};
  *gather.y.repeats = {exps[2], exps[3]};
  *gather.y.strides = {exps[3], exps[4]};
  gather.ir_attr.SetAxis(1);
  gather.ir_attr.SetNegative_index_support(false);

  store.x = gather.y;
  store.attr.sched.axis = {axes[2].id, axes[3].id};
  store.y.dtype = T;
  *store.y.axis = {axes[2].id, axes[3].id};
  *store.y.repeats = {exps[2], exps[3]};
  *store.y.strides = {exps[3], exps[4]};

  y.x = store.y;
  y.attr.sched.axis = {axes[2].id, axes[3].id};
  y.y.dtype = T;
  *y.y.axis = {axes[2].id, axes[3].id};
  *y.y.repeats = {exps[2], exps[3]};
  *y.y.strides = {exps[3], exps[4]};
}

template <af::DataType T>
std::vector<af::AxisId> CreateGraphAttrAxisIsNotLastAxisAndOneVecAxis(af::AscGraph &graph, codegen::Tiler &tiler,
                                                                      int gather_axis = 0) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");
  auto s5 = graph.CreateSizeVar("s5");
  auto s6 = graph.CreateSizeVar("s6");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);
  auto z5 = graph.CreateAxis("z5", s5);
  auto z6 = graph.CreateAxis("z6", s6);

  Data x1_("x1");
  graph.AddNode(x1_);
  x1_.y.dtype = T;
  x1_.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *x1_.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *x1_.y.repeats = {s0, s1, s2, s3, s4};
  *x1_.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  Data x2_("x2");
  graph.AddNode(x2_);
  x2_.y.dtype = T;
  x2_.attr.sched.axis = {z5.id, z6.id};
  *x2_.y.axis = {z5.id, z6.id};
  *x2_.y.repeats = {s5, s6};
  *x2_.y.strides = {s6, One};

  Gather gather_("gather");
  graph.AddNode(gather_);
  gather_.x1 = x1_.y;
  gather_.x2 = x2_.y;
  gather_.ir_attr.SetAxis(gather_axis);
  gather_.ir_attr.SetNegative_index_support(false);
  gather_.attr.sched.axis = {z0.id, z1.id, z5.id, z6.id, z3.id, z4.id};
  *gather_.y.axis = {z0.id, z1.id, z5.id, z6.id, z3.id, z4.id};
  *gather_.y.repeats = {s0, s1, s5, s6, s3, s4};
  *gather_.y.strides = {s1 * s5 * s6 * s3 * s4, s5 * s6 * s3 * s4, s6 * s3 * s4, s3 * s4, s4, One};

  Store store_("store");
  graph.AddNode(store_);
  store_.x = gather_.y;
  store_.attr.sched.axis = {z0.id, z1.id, z5.id, z6.id, z3.id, z4.id};
  *store_.y.axis = {z0.id, z1.id, z5.id, z6.id, z3.id, z4.id};
  *store_.y.repeats = {s0, s1, s5, s6, s3, s4};
  *store_.y.strides = {s1 * s5 * s6 * s3 * s4, s5 * s6 * s3 * s4, s6 * s3 * s4, s3 * s4, s4, One};

  Output y_("y");
  graph.AddNode(y_);
  y_.x = store_.y;
  y_.y.dtype = T;

  auto x1 = graph.FindNode("x1");
  x1->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x1->attr.api.type = af::ApiType::kAPITypeBuffer;
  x1->attr.api.unit = af::ComputeUnit::kUnitNone;

  auto x2 = graph.FindNode("x2");
  x2->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x2->attr.api.type = af::ApiType::kAPITypeBuffer;
  x2->attr.api.unit = af::ComputeUnit::kUnitNone;

  auto gather = graph.FindNode("gather");
  gather->outputs[0].attr.dtype = T;
  gather->attr.api.compute_type = af::ComputeType::kComputeGather;
  gather->attr.api.type = af::ApiType::kAPITypeCompute;
  gather->attr.api.unit = af::ComputeUnit::kUnitMTE2;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = af::ComputeType::kComputeStore;
  store->outputs[0].attr.dtype = T;
  store->attr.api.type = af::ApiType::kAPITypeCompute;
  store->attr.api.unit = af::ComputeUnit::kUnitMTE2;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y->attr.api.type = af::ApiType::kAPITypeBuffer;
  y->attr.api.unit = af::ComputeUnit::kUnitNone;

  // Scheduler
  auto axes = graph.GetAllAxis();
  auto z0_id = axes[0]->id;
  auto z1_id = axes[1]->id;
  auto z2_id = axes[2]->id;
  auto z3_id = axes[3]->id;
  auto z4_id = axes[4]->id;
  auto z5_id = axes[5]->id;
  auto z6_id = axes[6]->id;

  auto z0z1z5z6 = graph.MergeAxis({z0_id, z1_id, z5_id, z6_id});
  auto [z0z1z5z6B, z0z1z5z6b] = graph.BlockSplit(z0z1z5z6->id);

  auto z3z4 = graph.MergeAxis({z3_id, z4_id});
  auto [z3z4T, z3z4t] = graph.TileSplit(z3z4->id);

  for (auto node : graph.GetAllNodes()) {
    if (IsOps<Data>(node) || IsOps<Output>(node)) {
      continue;
    }
    graph.ApplyMerge(node, z0z1z5z6->id);
    graph.ApplySplit(node, z0z1z5z6B->id, z0z1z5z6b->id);

    graph.ApplyMerge(node, z3z4->id);
    graph.ApplySplit(node, z3z4T->id, z3z4t->id);

    graph.ApplyReorder(node, {z0z1z5z6B->id, z0z1z5z6b->id, z3z4T->id, z3z4t->id});
  }

  for (auto axis : graph.GetAllAxis()) {
    tiler.AddAxis(*axis);
  }

  auto size = af::GetSizeByDataType(T);
  vector<af::Expression> vectorized_strides{One};

  // Vectorized/Loop axis
  gather->attr.sched.loop_axis = z3z4T->id;
  gather->outputs[0].attr.vectorized_axis = {z3z4t->id};
  gather->outputs[0].attr.vectorized_strides = vectorized_strides;

  store->attr.sched.loop_axis = z3z4T->id;
  store->outputs[0].attr.vectorized_axis = {z3z4t->id};
  store->outputs[0].attr.vectorized_strides = vectorized_strides;

  // Que/Buf alloc
  x1->outputs[0].attr.mem.tensor_id = 0;
  x1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x1->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  x1->outputs[0].attr.mem.position = af::Position::kPositionGM;
  x1->outputs[0].attr.buf.id = af::kIdNone;
  x1->outputs[0].attr.que.id = af::kIdNone;
  x1->outputs[0].attr.mem.reuse_id = af::kIdNone;
  x1->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  x1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  x2->outputs[0].attr.mem.tensor_id = 1;
  x2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x2->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  x2->outputs[0].attr.mem.position = af::Position::kPositionGM;
  x2->outputs[0].attr.buf.id = af::kIdNone;
  x2->outputs[0].attr.que.id = af::kIdNone;
  x2->outputs[0].attr.mem.reuse_id = af::kIdNone;
  x2->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  x2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  gather->outputs[0].attr.mem.tensor_id = 2;
  gather->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  gather->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  gather->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  gather->outputs[0].attr.buf.id = af::kIdNone;
  gather->outputs[0].attr.que.id = 0;
  gather->outputs[0].attr.mem.reuse_id = 0;
  gather->outputs[0].attr.que.depth = 2;
  gather->outputs[0].attr.que.buf_num = 2;
  gather->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  gather->outputs[0].attr.opt.merge_scope = af::kIdNone;

  store->outputs[0].attr.mem.tensor_id = 4;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = af::Position::kPositionGM;
  store->outputs[0].attr.buf.id = af::kIdNone;
  store->outputs[0].attr.que.id = af::kIdNone;
  store->outputs[0].attr.mem.reuse_id = af::kIdNone;
  store->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  std::vector<af::AxisId> current_axis = {z0z1z5z6B->id, z0z1z5z6b->id, z3z4T->id};
  return current_axis;
}

template <af::DataType T>
std::vector<af::AxisId> CreateGraphAttrAxisIsNotLastAxisAndTwoVecAxis(af::AscGraph &graph, codegen::Tiler &tiler,
                                                                      int gather_axis = 0) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");
  auto s5 = graph.CreateSizeVar("s5");
  auto s6 = graph.CreateSizeVar("s6");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);
  auto z5 = graph.CreateAxis("z5", s5);
  auto z6 = graph.CreateAxis("z6", s6);

  Data x1_("x1");
  graph.AddNode(x1_);
  x1_.y.dtype = T;
  x1_.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *x1_.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *x1_.y.repeats = {s0, s1, s2, s3, s4};
  *x1_.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  Data x2_("x2");
  graph.AddNode(x2_);
  x2_.y.dtype = T;
  x2_.attr.sched.axis = {z5.id, z6.id};
  *x2_.y.axis = {z5.id, z6.id};
  *x2_.y.repeats = {s5, s6};
  *x2_.y.strides = {s6, One};

  Gather gather_("gather");
  graph.AddNode(gather_);
  gather_.x1 = x1_.y;
  gather_.x2 = x2_.y;
  gather_.ir_attr.SetAxis(gather_axis);
  gather_.ir_attr.SetNegative_index_support(false);
  gather_.attr.sched.axis = {z0.id, z1.id, z5.id, z6.id, z3.id, z4.id};
  *gather_.y.axis = {z0.id, z1.id, z5.id, z6.id, z3.id, z4.id};
  *gather_.y.repeats = {s0, s1, s5, s6, s3, s4};
  *gather_.y.strides = {s1 * s5 * s6 * s3 * s4, s5 * s6 * s3 * s4, s6 * s3 * s4, s3 * s4, s4, One};

  Store store_("store");
  graph.AddNode(store_);
  store_.x = gather_.y;
  store_.attr.sched.axis = {z0.id, z1.id, z5.id, z6.id, z3.id, z4.id};
  *store_.y.axis = {z0.id, z1.id, z5.id, z6.id, z3.id, z4.id};
  *store_.y.repeats = {s0, s1, s5, s6, s3, s4};
  *store_.y.strides = {s1 * s5 * s6 * s3 * s4, s5 * s6 * s3 * s4, s6 * s3 * s4, s3 * s4, s4, One};

  Output y_("y");
  graph.AddNode(y_);
  y_.x = store_.y;
  y_.y.dtype = T;

  auto x1 = graph.FindNode("x1");
  x1->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x1->attr.api.type = af::ApiType::kAPITypeBuffer;
  x1->attr.api.unit = af::ComputeUnit::kUnitNone;

  auto x2 = graph.FindNode("x2");
  x2->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x2->attr.api.type = af::ApiType::kAPITypeBuffer;
  x2->attr.api.unit = af::ComputeUnit::kUnitNone;

  auto gather = graph.FindNode("gather");
  gather->outputs[0].attr.dtype = T;
  gather->attr.api.compute_type = af::ComputeType::kComputeGather;
  gather->attr.api.type = af::ApiType::kAPITypeCompute;
  gather->attr.api.unit = af::ComputeUnit::kUnitMTE2;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = af::ComputeType::kComputeStore;
  store->outputs[0].attr.dtype = T;
  store->attr.api.type = af::ApiType::kAPITypeCompute;
  store->attr.api.unit = af::ComputeUnit::kUnitMTE2;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y->attr.api.type = af::ApiType::kAPITypeBuffer;
  y->attr.api.unit = af::ComputeUnit::kUnitNone;

  // Scheduler
  auto axes = graph.GetAllAxis();
  auto z0_id = axes[0]->id;
  auto z1_id = axes[1]->id;
  auto z2_id = axes[2]->id;
  auto z3_id = axes[3]->id;
  auto z4_id = axes[4]->id;
  auto z5_id = axes[5]->id;
  auto z6_id = axes[6]->id;

  auto z0z1z5z6 = graph.MergeAxis({z0_id, z1_id, z5_id, z6_id});
  auto [z0z1z5z6T, z0z1z5z6t] = graph.TileSplit(z0z1z5z6->id);
  auto [z0z1z5z6TB, z0z1z5z6Tb] = graph.BlockSplit(z0z1z5z6T->id);

  auto z3z4 = graph.MergeAxis({z3_id, z4_id});
  auto [z3z4T, z3z4t] = graph.TileSplit(z3z4->id);

  for (auto node : graph.GetAllNodes()) {
    if (IsOps<Data>(node) || IsOps<Output>(node)) {
      continue;
    }
    graph.ApplyMerge(node, z0z1z5z6->id);
    graph.ApplySplit(node, z0z1z5z6T->id, z0z1z5z6t->id);
    graph.ApplySplit(node, z0z1z5z6TB->id, z0z1z5z6Tb->id);

    graph.ApplyMerge(node, z3z4->id);
    graph.ApplySplit(node, z3z4T->id, z3z4t->id);

    graph.ApplyReorder(node, {z0z1z5z6TB->id, z0z1z5z6Tb->id, z3z4T->id, z0z1z5z6t->id, z3z4t->id});
  }

  for (auto axis : graph.GetAllAxis()) {
    tiler.AddAxis(*axis);
  }

  auto size = af::GetSizeByDataType(T);
  vector<af::Expression> vectorized_strides{One, One};
  vectorized_strides[0] = af::sym::Align(graph.FindAxis(z3z4t->id)->size, 32 / size);

  // Vectorized/Loop axis
  gather->attr.sched.loop_axis = z3z4T->id;
  gather->outputs[0].attr.vectorized_axis = {z0z1z5z6t->id, z3z4t->id};
  gather->outputs[0].attr.vectorized_strides = vectorized_strides;

  store->attr.sched.loop_axis = z3z4T->id;
  store->outputs[0].attr.vectorized_axis = {z0z1z5z6t->id, z3z4t->id};
  store->outputs[0].attr.vectorized_strides = vectorized_strides;

  // Que/Buf alloc
  x1->outputs[0].attr.mem.tensor_id = 0;
  x1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x1->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  x1->outputs[0].attr.mem.position = af::Position::kPositionGM;
  x1->outputs[0].attr.buf.id = af::kIdNone;
  x1->outputs[0].attr.que.id = af::kIdNone;
  x1->outputs[0].attr.mem.reuse_id = af::kIdNone;
  x1->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  x1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  x2->outputs[0].attr.mem.tensor_id = 1;
  x2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x2->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  x2->outputs[0].attr.mem.position = af::Position::kPositionGM;
  x2->outputs[0].attr.buf.id = af::kIdNone;
  x2->outputs[0].attr.que.id = af::kIdNone;
  x2->outputs[0].attr.mem.reuse_id = af::kIdNone;
  x2->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  x2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  gather->outputs[0].attr.mem.tensor_id = 2;
  gather->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  gather->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  gather->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  gather->outputs[0].attr.buf.id = af::kIdNone;
  gather->outputs[0].attr.que.id = 0;
  gather->outputs[0].attr.mem.reuse_id = 0;
  gather->outputs[0].attr.que.depth = 2;
  gather->outputs[0].attr.que.buf_num = 2;
  gather->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  gather->outputs[0].attr.opt.merge_scope = af::kIdNone;

  store->outputs[0].attr.mem.tensor_id = 4;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = af::Position::kPositionGM;
  store->outputs[0].attr.buf.id = af::kIdNone;
  store->outputs[0].attr.que.id = af::kIdNone;
  store->outputs[0].attr.mem.reuse_id = af::kIdNone;
  store->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  std::vector<af::AxisId> current_axis = {z0z1z5z6TB->id, z0z1z5z6Tb->id, z3z4T->id};
  return current_axis;
}

template <af::DataType T>
std::vector<af::AxisId> CreateGraphAttrAxisIsLastAxisAndParamHasMoreThanOneAxis(af::AscGraph &graph,
                                                                                codegen::Tiler &tiler,
                                                                                int gather_axis = 0) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");
  auto s5 = graph.CreateSizeVar("s5");
  auto s6 = graph.CreateSizeVar("s6");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);
  auto z5 = graph.CreateAxis("z5", s5);
  auto z6 = graph.CreateAxis("z6", s6);

  Data x1_("x1");
  graph.AddNode(x1_);
  x1_.y.dtype = T;
  x1_.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *x1_.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *x1_.y.repeats = {s0, s1, s2, s3, s4};
  *x1_.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  Data x2_("x2");
  graph.AddNode(x2_);
  x2_.y.dtype = T;
  x2_.attr.sched.axis = {z5.id, z6.id};
  *x2_.y.axis = {z5.id, z6.id};
  *x2_.y.repeats = {s5, s6};
  *x2_.y.strides = {s6, One};

  Gather gather_("gather");
  graph.AddNode(gather_);
  gather_.x1 = x1_.y;
  gather_.x2 = x2_.y;
  gather_.ir_attr.SetAxis(gather_axis);
  gather_.ir_attr.SetNegative_index_support(false);
  gather_.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z5.id, z6.id};
  *gather_.y.axis = {z0.id, z1.id, z2.id, z3.id, z5.id, z6.id};
  *gather_.y.repeats = {s0, s1, s2, s3, s5, s6};
  *gather_.y.strides = {s1 * s2 * s3 * s5 * s6, s2 * s3 * s5 * s6, s3 * s5 * s6, s5 * s6, s6, One};

  Store store_("store");
  graph.AddNode(store_);
  store_.x = gather_.y;
  store_.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z5.id, z6.id};
  *store_.y.axis = {z0.id, z1.id, z2.id, z3.id, z5.id, z6.id};
  *store_.y.repeats = {s0, s1, s2, s3, s5, s6};
  *store_.y.strides = {s1 * s2 * s3 * s5 * s6, s2 * s3 * s5 * s6, s3 * s5 * s6, s5 * s6, s6, One};

  Output y_("y");
  graph.AddNode(y_);
  y_.x = store_.y;
  y_.y.dtype = T;

  auto x1 = graph.FindNode("x1");
  x1->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x1->attr.api.type = af::ApiType::kAPITypeBuffer;
  x1->attr.api.unit = af::ComputeUnit::kUnitNone;

  auto x2 = graph.FindNode("x2");
  x2->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x2->attr.api.type = af::ApiType::kAPITypeBuffer;
  x2->attr.api.unit = af::ComputeUnit::kUnitNone;

  auto gather = graph.FindNode("gather");
  gather->outputs[0].attr.dtype = T;
  gather->attr.api.compute_type = af::ComputeType::kComputeGather;
  gather->attr.api.type = af::ApiType::kAPITypeCompute;
  gather->attr.api.unit = af::ComputeUnit::kUnitMTE2;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = af::ComputeType::kComputeStore;
  store->outputs[0].attr.dtype = T;
  store->attr.api.type = af::ApiType::kAPITypeCompute;
  store->attr.api.unit = af::ComputeUnit::kUnitMTE2;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y->attr.api.type = af::ApiType::kAPITypeBuffer;
  y->attr.api.unit = af::ComputeUnit::kUnitNone;

  // Scheduler
  auto axes = graph.GetAllAxis();
  auto z0_id = axes[0]->id;
  auto z1_id = axes[1]->id;
  auto z2_id = axes[2]->id;
  auto z3_id = axes[3]->id;
  auto z4_id = axes[4]->id;
  auto z5_id = axes[5]->id;
  auto z6_id = axes[6]->id;

  auto z0z1z2z3 = graph.MergeAxis({z0_id, z1_id, z2_id, z3_id});
  auto [z0z1z2z3B, z0z1z2z3b] = graph.BlockSplit(z0z1z2z3->id);

  auto z5z6 = graph.MergeAxis({z5_id, z6_id});
  auto [z5z6T, z5z6t] = graph.TileSplit(z5z6->id);

  for (auto node : graph.GetAllNodes()) {
    if (IsOps<Data>(node) || IsOps<Output>(node)) {
      continue;
    }
    graph.ApplyMerge(node, z0z1z2z3->id);
    graph.ApplySplit(node, z0z1z2z3B->id, z0z1z2z3b->id);

    graph.ApplyMerge(node, z5z6->id);
    graph.ApplySplit(node, z5z6T->id, z5z6t->id);

    graph.ApplyReorder(node, {z0z1z2z3B->id, z0z1z2z3b->id, z5z6T->id, z5z6t->id});
  }

  for (auto axis : graph.GetAllAxis()) {
    tiler.AddAxis(*axis);
  }

  auto size = af::GetSizeByDataType(T);
  vector<af::Expression> vectorized_strides{One};

  // Vectorized/Loop axis
  gather->attr.sched.loop_axis = z5z6T->id;
  gather->outputs[0].attr.vectorized_axis = {z5z6t->id};
  gather->outputs[0].attr.vectorized_strides = vectorized_strides;

  store->attr.sched.loop_axis = z5z6T->id;
  store->outputs[0].attr.vectorized_axis = {z5z6t->id};
  store->outputs[0].attr.vectorized_strides = vectorized_strides;

  // Que/Buf alloc
  x1->outputs[0].attr.mem.tensor_id = 0;
  x1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x1->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  x1->outputs[0].attr.mem.position = af::Position::kPositionGM;
  x1->outputs[0].attr.buf.id = af::kIdNone;
  x1->outputs[0].attr.que.id = af::kIdNone;
  x1->outputs[0].attr.mem.reuse_id = af::kIdNone;
  x1->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  x1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  x2->outputs[0].attr.mem.tensor_id = 1;
  x2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x2->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  x2->outputs[0].attr.mem.position = af::Position::kPositionGM;
  x2->outputs[0].attr.buf.id = af::kIdNone;
  x2->outputs[0].attr.que.id = af::kIdNone;
  x2->outputs[0].attr.mem.reuse_id = af::kIdNone;
  x2->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  x2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  gather->outputs[0].attr.mem.tensor_id = 2;
  gather->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  gather->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  gather->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  gather->outputs[0].attr.buf.id = af::kIdNone;
  gather->outputs[0].attr.que.id = 0;
  gather->outputs[0].attr.mem.reuse_id = 0;
  gather->outputs[0].attr.que.depth = 2;
  gather->outputs[0].attr.que.buf_num = 2;
  gather->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  gather->outputs[0].attr.opt.merge_scope = af::kIdNone;

  store->outputs[0].attr.mem.tensor_id = 4;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = af::Position::kPositionGM;
  store->outputs[0].attr.buf.id = af::kIdNone;
  store->outputs[0].attr.que.id = af::kIdNone;
  store->outputs[0].attr.mem.reuse_id = af::kIdNone;
  store->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  std::vector<af::AxisId> current_axis = {z0z1z2z3B->id, z0z1z2z3b->id, z5z6T->id};
  return current_axis;
}

std::vector<af::AxisId> CreateGraphGatherWithComputeType_Mid_Axis(
    af::AscGraph &graph, codegen::Tiler &tiler, af::ComputeType compute_type = af::ComputeType::kComputeGather) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  Data x1_("x1");
  graph.AddNode(x1_);
  x1_.y.dtype = af::DT_FLOAT;
  x1_.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x1_.y.axis = {z0.id, z1.id, z2.id};
  *x1_.y.repeats = {s0, s1, s2};
  *x1_.y.strides = {s1 * s2, s2, One};

  Data x2_("x2");
  graph.AddNode(x2_);
  x2_.y.dtype = af::DT_FLOAT;
  x2_.attr.sched.axis = {z3.id, z4.id};
  *x2_.y.axis = {z3.id, z4.id};
  *x2_.y.repeats = {s3, s4};
  *x2_.y.strides = {s4, One};

  Gather gather_("gather");
  graph.AddNode(gather_);
  gather_.x1 = x1_.y;
  gather_.x2 = x2_.y;
  gather_.ir_attr.SetAxis(1);
  gather_.ir_attr.SetNegative_index_support(false);
  gather_.attr.sched.axis = {z0.id, z3.id, z4.id, z2.id};
  *gather_.y.axis = {z0.id, z3.id, z4.id, z2.id};
  *gather_.y.repeats = {s0, s3, s4, s2};
  *gather_.y.strides = {s3 * s4 * s2, s4 * s2, s2, One};

  Store store_("store");
  graph.AddNode(store_);
  store_.x = gather_.y;
  store_.attr.sched.axis = {z0.id, z3.id, z4.id, z2.id};
  *store_.y.axis = {z0.id, z3.id, z4.id, z2.id};
  *store_.y.repeats = {s0, s3, s4, s2};
  *store_.y.strides = {s3 * s4 * s2, s4 * s2, s2, One};

  Output y_("y");
  graph.AddNode(y_);
  y_.x = store_.y;
  y_.y.dtype = af::DT_FLOAT;

  auto x1 = graph.FindNode("x1");
  x1->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x1->attr.api.type = af::ApiType::kAPITypeBuffer;
  x1->attr.api.unit = af::ComputeUnit::kUnitNone;

  auto x2 = graph.FindNode("x2");
  x2->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x2->attr.api.type = af::ApiType::kAPITypeBuffer;
  x2->attr.api.unit = af::ComputeUnit::kUnitNone;

  auto gather = graph.FindNode("gather");
  gather->outputs[0].attr.dtype = af::DT_FLOAT;
  gather->attr.api.compute_type = compute_type;
  gather->attr.api.type = af::ApiType::kAPITypeCompute;
  gather->attr.api.unit = af::ComputeUnit::kUnitMTE2;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = af::ComputeType::kComputeStore;
  store->outputs[0].attr.dtype = af::DT_FLOAT;
  store->attr.api.type = af::ApiType::kAPITypeCompute;
  store->attr.api.unit = af::ComputeUnit::kUnitMTE2;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y->attr.api.type = af::ApiType::kAPITypeBuffer;
  y->attr.api.unit = af::ComputeUnit::kUnitNone;

  // Scheduler
  auto axes = graph.GetAllAxis();
  auto z0_id = axes[0]->id;
  auto z1_id = axes[1]->id;
  auto z2_id = axes[2]->id;
  auto z3_id = axes[3]->id;
  auto z4_id = axes[4]->id;

  auto z0z3z4z2 = graph.MergeAxis({z0_id, z3_id, z4_id, z2_id});
  auto [z0z3z4z2T, z0z3z4z2t] = graph.TileSplit(z0z3z4z2->id);

  auto [z0z3z4z2TB, z0z3z4z2Tb] = graph.BlockSplit(z0z3z4z2T->id);

  for (auto node : graph.GetAllNodes()) {
    if (IsOps<Data>(node) || IsOps<Output>(node)) {
      continue;
    }
    graph.ApplyMerge(node, z0z3z4z2->id);
    graph.ApplySplit(node, z0z3z4z2T->id, z0z3z4z2t->id);
    graph.ApplySplit(node, z0z3z4z2TB->id, z0z3z4z2Tb->id);

    graph.ApplyReorder(node, {z0z3z4z2T->id, z0z3z4z2t->id, z0z3z4z2TB->id, z0z3z4z2Tb->id});
  }

  for (auto axis : graph.GetAllAxis()) {
    tiler.AddAxis(*axis);
  }

  auto size = af::GetSizeByDataType(af::DT_FLOAT);
  vector<af::Expression> vectorized_strides{One};

  // Vectorized/Loop axis
  gather->attr.sched.loop_axis = z0z3z4z2Tb->id;
  gather->outputs[0].attr.vectorized_axis = {z0z3z4z2t->id};
  gather->outputs[0].attr.vectorized_strides = vectorized_strides;

  store->attr.sched.loop_axis = z0z3z4z2Tb->id;
  store->outputs[0].attr.vectorized_axis = {z0z3z4z2t->id};
  store->outputs[0].attr.vectorized_strides = vectorized_strides;

  // Que/Buf alloc
  x1->outputs[0].attr.mem.tensor_id = 0;
  x1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x1->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  x1->outputs[0].attr.mem.position = af::Position::kPositionGM;
  x1->outputs[0].attr.buf.id = af::kIdNone;
  x1->outputs[0].attr.que.id = af::kIdNone;
  x1->outputs[0].attr.mem.reuse_id = af::kIdNone;
  x1->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  x1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  x2->outputs[0].attr.mem.tensor_id = 1;
  x2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x2->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  x2->outputs[0].attr.mem.position = af::Position::kPositionGM;
  x2->outputs[0].attr.buf.id = af::kIdNone;
  x2->outputs[0].attr.que.id = af::kIdNone;
  x2->outputs[0].attr.mem.reuse_id = af::kIdNone;
  x2->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  x2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  gather->outputs[0].attr.mem.tensor_id = 2;
  gather->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  gather->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  gather->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  gather->outputs[0].attr.buf.id = af::kIdNone;
  gather->outputs[0].attr.que.id = 0;
  gather->outputs[0].attr.mem.reuse_id = 0;
  gather->outputs[0].attr.que.depth = 2;
  gather->outputs[0].attr.que.buf_num = 2;
  gather->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  gather->outputs[0].attr.opt.merge_scope = af::kIdNone;

  store->outputs[0].attr.mem.tensor_id = 4;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = af::Position::kPositionGM;
  store->outputs[0].attr.buf.id = af::kIdNone;
  store->outputs[0].attr.que.id = af::kIdNone;
  store->outputs[0].attr.mem.reuse_id = af::kIdNone;
  store->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  std::vector<af::AxisId> current_axis = {z0z3z4z2TB->id, z0z3z4z2Tb->id};
  return current_axis;
}

std::vector<af::AxisId> CreateGraphGatherWithComputeType_First_Axis(
    af::AscGraph &graph, codegen::Tiler &tiler, af::ComputeType compute_type = af::ComputeType::kComputeGather) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x1_("x1");
  graph.AddNode(x1_);
  x1_.y.dtype = af::DT_FLOAT;
  x1_.attr.sched.axis = {z0.id, z1.id};
  *x1_.y.axis = {z0.id, z1.id};
  *x1_.y.repeats = {s0, s1};
  *x1_.y.strides = {s1, One};

  Data x2_("x2");
  graph.AddNode(x2_);
  x2_.y.dtype = af::DT_FLOAT;
  x2_.attr.sched.axis = {z2.id};
  *x2_.y.axis = {z2.id};
  *x2_.y.repeats = {s2};
  *x2_.y.strides = {One};

  Gather gather_("gather");
  graph.AddNode(gather_);
  gather_.x1 = x1_.y;
  gather_.x2 = x2_.y;
  gather_.ir_attr.SetAxis(0);
  gather_.ir_attr.SetNegative_index_support(false);
  gather_.attr.sched.axis = {z2.id, z1.id};
  *gather_.y.axis = {z2.id, z1.id};
  *gather_.y.repeats = {s2, s1};
  *gather_.y.strides = {s1, One};

  Store store_("store");
  graph.AddNode(store_);
  store_.x = gather_.y;
  store_.attr.sched.axis = {z2.id, z1.id};
  *store_.y.axis = {z2.id, z1.id};
  *store_.y.repeats = {s2, s1};
  *store_.y.strides = {s1, One};

  Output y_("y");
  graph.AddNode(y_);
  y_.x = store_.y;
  y_.y.dtype = af::DT_FLOAT;

  auto x1 = graph.FindNode("x1");
  x1->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x1->attr.api.type = af::ApiType::kAPITypeBuffer;
  x1->attr.api.unit = af::ComputeUnit::kUnitNone;

  auto x2 = graph.FindNode("x2");
  x2->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x2->attr.api.type = af::ApiType::kAPITypeBuffer;
  x2->attr.api.unit = af::ComputeUnit::kUnitNone;

  auto gather = graph.FindNode("gather");
  gather->outputs[0].attr.dtype = af::DT_FLOAT;
  gather->attr.api.compute_type = compute_type;
  gather->attr.api.type = af::ApiType::kAPITypeCompute;
  gather->attr.api.unit = af::ComputeUnit::kUnitMTE2;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = af::ComputeType::kComputeStore;
  store->outputs[0].attr.dtype = af::DT_FLOAT;
  store->attr.api.type = af::ApiType::kAPITypeCompute;
  store->attr.api.unit = af::ComputeUnit::kUnitMTE2;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y->attr.api.type = af::ApiType::kAPITypeBuffer;
  y->attr.api.unit = af::ComputeUnit::kUnitNone;

  // Scheduler
  auto axes = graph.GetAllAxis();
  auto z0_id = axes[0]->id;
  auto z1_id = axes[1]->id;
  auto z2_id = axes[2]->id;

  auto z2z1 = graph.MergeAxis({z2_id, z1_id});
  auto [z2z1T, z2z1t] = graph.TileSplit(z2z1->id);

  auto [z2z1TB, z2z1Tb] = graph.BlockSplit(z2z1T->id);

  for (auto node : graph.GetAllNodes()) {
    if (IsOps<Data>(node) || IsOps<Output>(node)) {
      continue;
    }
    graph.ApplyMerge(node, z2z1->id);
    graph.ApplySplit(node, z2z1T->id, z2z1t->id);
    graph.ApplySplit(node, z2z1TB->id, z2z1Tb->id);

    graph.ApplyReorder(node, {z2z1T->id, z2z1t->id, z2z1TB->id, z2z1Tb->id});
  }

  for (auto axis : graph.GetAllAxis()) {
    tiler.AddAxis(*axis);
  }

  auto size = af::GetSizeByDataType(af::DT_FLOAT);
  vector<af::Expression> vectorized_strides{One};

  // Vectorized/Loop axis
  gather->attr.sched.loop_axis = z2z1Tb->id;
  gather->outputs[0].attr.vectorized_axis = {z2z1t->id};
  gather->outputs[0].attr.vectorized_strides = vectorized_strides;

  store->attr.sched.loop_axis = z2z1Tb->id;
  store->outputs[0].attr.vectorized_axis = {z2z1t->id};
  store->outputs[0].attr.vectorized_strides = vectorized_strides;

  // Que/Buf alloc
  x1->outputs[0].attr.mem.tensor_id = 0;
  x1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x1->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  x1->outputs[0].attr.mem.position = af::Position::kPositionGM;
  x1->outputs[0].attr.buf.id = af::kIdNone;
  x1->outputs[0].attr.que.id = af::kIdNone;
  x1->outputs[0].attr.mem.reuse_id = af::kIdNone;
  x1->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  x1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  x2->outputs[0].attr.mem.tensor_id = 1;
  x2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x2->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  x2->outputs[0].attr.mem.position = af::Position::kPositionGM;
  x2->outputs[0].attr.buf.id = af::kIdNone;
  x2->outputs[0].attr.que.id = af::kIdNone;
  x2->outputs[0].attr.mem.reuse_id = af::kIdNone;
  x2->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  x2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  gather->outputs[0].attr.mem.tensor_id = 2;
  gather->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  gather->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  gather->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  gather->outputs[0].attr.buf.id = af::kIdNone;
  gather->outputs[0].attr.que.id = 0;
  gather->outputs[0].attr.mem.reuse_id = 0;
  gather->outputs[0].attr.que.depth = 2;
  gather->outputs[0].attr.que.buf_num = 2;
  gather->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  gather->outputs[0].attr.opt.merge_scope = af::kIdNone;

  store->outputs[0].attr.mem.tensor_id = 4;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = af::Position::kPositionGM;
  store->outputs[0].attr.buf.id = af::kIdNone;
  store->outputs[0].attr.que.id = af::kIdNone;
  store->outputs[0].attr.mem.reuse_id = af::kIdNone;
  store->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  std::vector<af::AxisId> current_axis = {z2z1TB->id, z2z1Tb->id};
  return current_axis;
}

std::vector<af::AxisId> CreateGraphGatherWithComputeType_Single_Axis(
    af::AscGraph &graph, codegen::Tiler &tiler, af::ComputeType compute_type = af::ComputeType::kComputeGather) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x1_("x1");
  graph.AddNode(x1_);
  x1_.y.dtype = af::DT_FLOAT;
  x1_.attr.sched.axis = {z0.id};
  *x1_.y.axis = {z0.id};
  *x1_.y.repeats = {s0};
  *x1_.y.strides = {One};

  Data x2_("x2");
  graph.AddNode(x2_);
  x2_.y.dtype = af::DT_FLOAT;
  x2_.attr.sched.axis = {z1.id};
  *x2_.y.axis = {z1.id};
  *x2_.y.repeats = {s1};
  *x2_.y.strides = {One};

  Gather gather_("gather");
  graph.AddNode(gather_);
  gather_.x1 = x1_.y;
  gather_.x2 = x2_.y;
  gather_.ir_attr.SetAxis(0);
  gather_.ir_attr.SetNegative_index_support(false);
  gather_.attr.sched.axis = {z1.id};
  *gather_.y.axis = {z1.id};
  *gather_.y.repeats = {s1};
  *gather_.y.strides = {One};

  Store store_("store");
  graph.AddNode(store_);
  store_.x = gather_.y;
  store_.attr.sched.axis = {z1.id};
  *store_.y.axis = {z1.id};
  *store_.y.repeats = {s1};
  *store_.y.strides = {One};

  Output y_("y");
  graph.AddNode(y_);
  y_.x = store_.y;
  y_.y.dtype = af::DT_FLOAT;

  auto x1 = graph.FindNode("x1");
  x1->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x1->attr.api.type = af::ApiType::kAPITypeBuffer;
  x1->attr.api.unit = af::ComputeUnit::kUnitNone;

  auto x2 = graph.FindNode("x2");
  x2->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x2->attr.api.type = af::ApiType::kAPITypeBuffer;
  x2->attr.api.unit = af::ComputeUnit::kUnitNone;

  auto gather = graph.FindNode("gather");
  gather->outputs[0].attr.dtype = af::DT_FLOAT;
  gather->attr.api.compute_type = compute_type;
  gather->attr.api.type = af::ApiType::kAPITypeCompute;
  gather->attr.api.unit = af::ComputeUnit::kUnitMTE2;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = af::ComputeType::kComputeStore;
  store->outputs[0].attr.dtype = af::DT_FLOAT;
  store->attr.api.type = af::ApiType::kAPITypeCompute;
  store->attr.api.unit = af::ComputeUnit::kUnitMTE2;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y->attr.api.type = af::ApiType::kAPITypeBuffer;
  y->attr.api.unit = af::ComputeUnit::kUnitNone;

  // Scheduler
  auto axes = graph.GetAllAxis();

  auto [z1T, z1t] = graph.TileSplit(z1.id);

  auto [z1TB, z1Tb] = graph.BlockSplit(z1T->id);

  for (auto node : graph.GetAllNodes()) {
    if (IsOps<Data>(node) || IsOps<Output>(node)) {
      continue;
    }
    graph.ApplySplit(node, z1T->id, z1t->id);
    graph.ApplySplit(node, z1TB->id, z1Tb->id);

    graph.ApplyReorder(node, {z1T->id, z1t->id, z1TB->id, z1Tb->id});
  }

  for (auto axis : graph.GetAllAxis()) {
    tiler.AddAxis(*axis);
  }

  auto size = af::GetSizeByDataType(af::DT_FLOAT);
  vector<af::Expression> vectorized_strides{One};

  // Vectorized/Loop axis
  gather->attr.sched.loop_axis = z1Tb->id;
  gather->outputs[0].attr.vectorized_axis = {z1t->id};
  gather->outputs[0].attr.vectorized_strides = vectorized_strides;

  store->attr.sched.loop_axis = z1Tb->id;
  store->outputs[0].attr.vectorized_axis = {z1t->id};
  store->outputs[0].attr.vectorized_strides = vectorized_strides;

  // Que/Buf alloc
  x1->outputs[0].attr.mem.tensor_id = 0;
  x1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x1->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  x1->outputs[0].attr.mem.position = af::Position::kPositionGM;
  x1->outputs[0].attr.buf.id = af::kIdNone;
  x1->outputs[0].attr.que.id = af::kIdNone;
  x1->outputs[0].attr.mem.reuse_id = af::kIdNone;
  x1->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  x1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  x2->outputs[0].attr.mem.tensor_id = 1;
  x2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x2->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  x2->outputs[0].attr.mem.position = af::Position::kPositionGM;
  x2->outputs[0].attr.buf.id = af::kIdNone;
  x2->outputs[0].attr.que.id = af::kIdNone;
  x2->outputs[0].attr.mem.reuse_id = af::kIdNone;
  x2->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  x2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  gather->outputs[0].attr.mem.tensor_id = 2;
  gather->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  gather->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  gather->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  gather->outputs[0].attr.buf.id = af::kIdNone;
  gather->outputs[0].attr.que.id = 0;
  gather->outputs[0].attr.mem.reuse_id = 0;
  gather->outputs[0].attr.que.depth = 2;
  gather->outputs[0].attr.que.buf_num = 2;
  gather->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  gather->outputs[0].attr.opt.merge_scope = af::kIdNone;

  store->outputs[0].attr.mem.tensor_id = 4;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = af::Position::kPositionGM;
  store->outputs[0].attr.buf.id = af::kIdNone;
  store->outputs[0].attr.que.id = af::kIdNone;
  store->outputs[0].attr.mem.reuse_id = af::kIdNone;
  store->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  std::vector<af::AxisId> current_axis = {z1TB->id, z1Tb->id};
  return current_axis;
}

std::vector<af::AxisId> CreateGraphGatherWithComputeType_Tail_Axis(
    af::AscGraph &graph, codegen::Tiler &tiler, af::ComputeType compute_type = af::ComputeType::kComputeGather) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x1_("x1");
  graph.AddNode(x1_);
  x1_.y.dtype = af::DT_FLOAT;
  x1_.attr.sched.axis = {z0.id, z1.id};
  *x1_.y.axis = {z0.id, z1.id};
  *x1_.y.repeats = {s0, s1};
  *x1_.y.strides = {s1, One};

  Data x2_("x2");
  graph.AddNode(x2_);
  x2_.y.dtype = af::DT_FLOAT;
  x2_.attr.sched.axis = {z2.id};
  *x2_.y.axis = {z2.id};
  *x2_.y.repeats = {s2};
  *x2_.y.strides = {One};

  Gather gather_("gather");
  graph.AddNode(gather_);
  gather_.x1 = x1_.y;
  gather_.x2 = x2_.y;
  gather_.ir_attr.SetAxis(1);
  gather_.ir_attr.SetNegative_index_support(false);
  gather_.attr.sched.axis = {z0.id, z2.id};
  *gather_.y.axis = {z0.id, z2.id};
  *gather_.y.repeats = {s0, s2};
  *gather_.y.strides = {s2, One};

  Store store_("store");
  graph.AddNode(store_);
  store_.x = gather_.y;
  store_.attr.sched.axis = {z0.id, z2.id};
  *store_.y.axis = {z0.id, z2.id};
  *store_.y.repeats = {s0, s2};
  *store_.y.strides = {s2, One};

  Output y_("y");
  graph.AddNode(y_);
  y_.x = store_.y;
  y_.y.dtype = af::DT_FLOAT;

  auto x1 = graph.FindNode("x1");
  x1->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x1->attr.api.type = af::ApiType::kAPITypeBuffer;
  x1->attr.api.unit = af::ComputeUnit::kUnitNone;

  auto x2 = graph.FindNode("x2");
  x2->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x2->attr.api.type = af::ApiType::kAPITypeBuffer;
  x2->attr.api.unit = af::ComputeUnit::kUnitNone;

  auto gather = graph.FindNode("gather");
  gather->outputs[0].attr.dtype = af::DT_FLOAT;
  gather->attr.api.compute_type = compute_type;
  gather->attr.api.type = af::ApiType::kAPITypeCompute;
  gather->attr.api.unit = af::ComputeUnit::kUnitMTE2;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = af::ComputeType::kComputeStore;
  store->outputs[0].attr.dtype = af::DT_FLOAT;
  store->attr.api.type = af::ApiType::kAPITypeCompute;
  store->attr.api.unit = af::ComputeUnit::kUnitMTE2;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y->attr.api.type = af::ApiType::kAPITypeBuffer;
  y->attr.api.unit = af::ComputeUnit::kUnitNone;

  // Scheduler
  auto axes = graph.GetAllAxis();
  auto z0_id = axes[0]->id;
  auto z1_id = axes[1]->id;
  auto z2_id = axes[2]->id;

  auto z0z2 = graph.MergeAxis({z0_id, z2_id});
  auto [z0z2T, z0z2t] = graph.TileSplit(z0z2->id);

  auto [z0z2TB, z0z2Tb] = graph.BlockSplit(z0z2T->id);

  for (auto node : graph.GetAllNodes()) {
    if (IsOps<Data>(node) || IsOps<Output>(node)) {
      continue;
    }
    graph.ApplyMerge(node, z0z2->id);
    graph.ApplySplit(node, z0z2T->id, z0z2t->id);
    graph.ApplySplit(node, z0z2TB->id, z0z2Tb->id);

    graph.ApplyReorder(node, {z0z2T->id, z0z2t->id, z0z2TB->id, z0z2Tb->id});
  }

  for (auto axis : graph.GetAllAxis()) {
    tiler.AddAxis(*axis);
  }

  auto size = af::GetSizeByDataType(af::DT_FLOAT);
  vector<af::Expression> vectorized_strides{One};

  // Vectorized/Loop axis
  gather->attr.sched.loop_axis = z0z2Tb->id;
  gather->outputs[0].attr.vectorized_axis = {z0z2t->id};
  gather->outputs[0].attr.vectorized_strides = vectorized_strides;

  store->attr.sched.loop_axis = z0z2Tb->id;
  store->outputs[0].attr.vectorized_axis = {z0z2t->id};
  store->outputs[0].attr.vectorized_strides = vectorized_strides;

  // Que/Buf alloc
  x1->outputs[0].attr.mem.tensor_id = 0;
  x1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x1->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  x1->outputs[0].attr.mem.position = af::Position::kPositionGM;
  x1->outputs[0].attr.buf.id = af::kIdNone;
  x1->outputs[0].attr.que.id = af::kIdNone;
  x1->outputs[0].attr.mem.reuse_id = af::kIdNone;
  x1->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  x1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  x2->outputs[0].attr.mem.tensor_id = 1;
  x2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x2->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  x2->outputs[0].attr.mem.position = af::Position::kPositionGM;
  x2->outputs[0].attr.buf.id = af::kIdNone;
  x2->outputs[0].attr.que.id = af::kIdNone;
  x2->outputs[0].attr.mem.reuse_id = af::kIdNone;
  x2->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  x2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  gather->outputs[0].attr.mem.tensor_id = 2;
  gather->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  gather->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  gather->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  gather->outputs[0].attr.buf.id = af::kIdNone;
  gather->outputs[0].attr.que.id = 0;
  gather->outputs[0].attr.mem.reuse_id = 0;
  gather->outputs[0].attr.que.depth = 2;
  gather->outputs[0].attr.que.buf_num = 2;
  gather->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  gather->outputs[0].attr.opt.merge_scope = af::kIdNone;

  store->outputs[0].attr.mem.tensor_id = 4;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = af::Position::kPositionGM;
  store->outputs[0].attr.buf.id = af::kIdNone;
  store->outputs[0].attr.que.id = af::kIdNone;
  store->outputs[0].attr.mem.reuse_id = af::kIdNone;
  store->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  std::vector<af::AxisId> current_axis = {z0z2TB->id, z0z2Tb->id};
  return current_axis;
}

std::vector<af::AxisId> CreateGraphGatherWithComputeType_Failed(
    af::AscGraph &graph, codegen::Tiler &tiler, af::ComputeType compute_type = af::ComputeType::kComputeGather) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x1_("x1");
  graph.AddNode(x1_);
  x1_.y.dtype = af::DT_FLOAT;
  x1_.attr.sched.axis = {z0.id, z1.id};
  *x1_.y.axis = {z0.id, z1.id};
  *x1_.y.repeats = {s0, s1};
  *x1_.y.strides = {s1, One};

  Data x2_("x2");
  graph.AddNode(x2_);
  x2_.y.dtype = af::DT_FLOAT;
  x2_.attr.sched.axis = {z2.id};
  *x2_.y.axis = {z2.id};
  *x2_.y.repeats = {s2};
  *x2_.y.strides = {One};

  Gather gather_("gather");
  graph.AddNode(gather_);
  gather_.x1 = x1_.y;
  gather_.x2 = x2_.y;
  gather_.ir_attr.SetAxis(-1);
  gather_.ir_attr.SetNegative_index_support(false);
  gather_.attr.sched.axis = {z0.id, z2.id};
  *gather_.y.axis = {z0.id, z2.id};
  *gather_.y.repeats = {s0, s2};
  *gather_.y.strides = {s2, One};

  Store store_("store");
  graph.AddNode(store_);
  store_.x = gather_.y;
  store_.attr.sched.axis = {z0.id, z2.id};
  *store_.y.axis = {z0.id, z2.id};
  *store_.y.repeats = {s0, s2};
  *store_.y.strides = {s2, One};

  Output y_("y");
  graph.AddNode(y_);
  y_.x = store_.y;
  y_.y.dtype = af::DT_FLOAT;

  auto x1 = graph.FindNode("x1");
  x1->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x1->attr.api.type = af::ApiType::kAPITypeBuffer;
  x1->attr.api.unit = af::ComputeUnit::kUnitNone;

  auto x2 = graph.FindNode("x2");
  x2->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x2->attr.api.type = af::ApiType::kAPITypeBuffer;
  x2->attr.api.unit = af::ComputeUnit::kUnitNone;

  auto gather = graph.FindNode("gather");
  gather->outputs[0].attr.dtype = af::DT_FLOAT;
  gather->attr.api.compute_type = compute_type;
  gather->attr.api.type = af::ApiType::kAPITypeCompute;
  gather->attr.api.unit = af::ComputeUnit::kUnitMTE2;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = af::ComputeType::kComputeStore;
  store->outputs[0].attr.dtype = af::DT_FLOAT;
  store->attr.api.type = af::ApiType::kAPITypeCompute;
  store->attr.api.unit = af::ComputeUnit::kUnitMTE2;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y->attr.api.type = af::ApiType::kAPITypeBuffer;
  y->attr.api.unit = af::ComputeUnit::kUnitNone;

  // Scheduler
  auto axes = graph.GetAllAxis();
  auto z0_id = axes[0]->id;
  auto z1_id = axes[1]->id;
  auto z2_id = axes[2]->id;

  auto z0z2 = graph.MergeAxis({z0_id, z2_id});
  auto [z0z2T, z0z2t] = graph.TileSplit(z0z2->id);

  auto [z0z2TB, z0z2Tb] = graph.BlockSplit(z0z2T->id);

  for (auto node : graph.GetAllNodes()) {
    if (IsOps<Data>(node) || IsOps<Output>(node)) {
      continue;
    }
    graph.ApplyMerge(node, z0z2->id);
    graph.ApplySplit(node, z0z2T->id, z0z2t->id);
    graph.ApplySplit(node, z0z2TB->id, z0z2Tb->id);

    graph.ApplyReorder(node, {z0z2T->id, z0z2t->id, z0z2TB->id, z0z2Tb->id});
  }

  for (auto axis : graph.GetAllAxis()) {
    tiler.AddAxis(*axis);
  }

  auto size = af::GetSizeByDataType(af::DT_FLOAT);
  vector<af::Expression> vectorized_strides{One};

  // Vectorized/Loop axis
  gather->attr.sched.loop_axis = z0z2Tb->id;
  gather->outputs[0].attr.vectorized_axis = {z0z2t->id};
  gather->outputs[0].attr.vectorized_strides = vectorized_strides;

  store->attr.sched.loop_axis = z0z2Tb->id;
  store->outputs[0].attr.vectorized_axis = {z0z2t->id};
  store->outputs[0].attr.vectorized_strides = vectorized_strides;

  // Que/Buf alloc
  x1->outputs[0].attr.mem.tensor_id = 0;
  x1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x1->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  x1->outputs[0].attr.mem.position = af::Position::kPositionGM;
  x1->outputs[0].attr.buf.id = af::kIdNone;
  x1->outputs[0].attr.que.id = af::kIdNone;
  x1->outputs[0].attr.mem.reuse_id = af::kIdNone;
  x1->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  x1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  x2->outputs[0].attr.mem.tensor_id = 1;
  x2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x2->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  x2->outputs[0].attr.mem.position = af::Position::kPositionGM;
  x2->outputs[0].attr.buf.id = af::kIdNone;
  x2->outputs[0].attr.que.id = af::kIdNone;
  x2->outputs[0].attr.mem.reuse_id = af::kIdNone;
  x2->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  x2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  gather->outputs[0].attr.mem.tensor_id = 2;
  gather->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  gather->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  gather->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  gather->outputs[0].attr.buf.id = af::kIdNone;
  gather->outputs[0].attr.que.id = 0;
  gather->outputs[0].attr.mem.reuse_id = 0;
  gather->outputs[0].attr.que.depth = 2;
  gather->outputs[0].attr.que.buf_num = 2;
  gather->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  gather->outputs[0].attr.opt.merge_scope = af::kIdNone;

  store->outputs[0].attr.mem.tensor_id = 4;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = af::Position::kPositionGM;
  store->outputs[0].attr.buf.id = af::kIdNone;
  store->outputs[0].attr.que.id = af::kIdNone;
  store->outputs[0].attr.mem.reuse_id = af::kIdNone;
  store->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  std::vector<af::AxisId> current_axis = {z0z2TB->id, z0z2Tb->id};
  return current_axis;
}

template <af::DataType T>
std::vector<af::AxisId> CreateGraphAttrAxisIsLastAxisAndParamHasOnlyOneAxis(af::AscGraph &graph, codegen::Tiler &tiler,
                                                                            int gather_axis = 0) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x1_("x1");
  graph.AddNode(x1_);
  x1_.y.dtype = T;
  x1_.attr.sched.axis = {z0.id};
  *x1_.y.axis = {z0.id};
  *x1_.y.repeats = {s0};
  *x1_.y.strides = {One};

  Data x2_("x2");
  graph.AddNode(x2_);
  x2_.y.dtype = T;
  x2_.attr.sched.axis = {z1.id};
  *x2_.y.axis = {z1.id};
  *x2_.y.repeats = {s1};
  *x2_.y.strides = {One};

  Gather gather_("gather");
  graph.AddNode(gather_);
  gather_.x1 = x1_.y;
  gather_.x2 = x2_.y;
  gather_.ir_attr.SetAxis(gather_axis);
  gather_.ir_attr.SetNegative_index_support(false);
  gather_.attr.sched.axis = {z1.id};
  *gather_.y.axis = {z1.id};
  *gather_.y.repeats = {s1};
  *gather_.y.strides = {One};

  Store store_("store");
  graph.AddNode(store_);
  store_.x = gather_.y;
  store_.attr.sched.axis = {z1.id};
  *store_.y.axis = {z1.id};
  *store_.y.repeats = {s1};
  *store_.y.strides = {One};

  Output y_("y");
  graph.AddNode(y_);
  y_.x = store_.y;
  y_.y.dtype = T;

  auto x1 = graph.FindNode("x1");
  x1->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x1->attr.api.type = af::ApiType::kAPITypeBuffer;
  x1->attr.api.unit = af::ComputeUnit::kUnitNone;

  auto x2 = graph.FindNode("x2");
  x2->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x2->attr.api.type = af::ApiType::kAPITypeBuffer;
  x2->attr.api.unit = af::ComputeUnit::kUnitNone;

  auto gather = graph.FindNode("gather");
  gather->outputs[0].attr.dtype = T;
  gather->attr.api.compute_type = af::ComputeType::kComputeGather;
  gather->attr.api.type = af::ApiType::kAPITypeCompute;
  gather->attr.api.unit = af::ComputeUnit::kUnitMTE2;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = af::ComputeType::kComputeStore;
  store->outputs[0].attr.dtype = T;
  store->attr.api.type = af::ApiType::kAPITypeCompute;
  store->attr.api.unit = af::ComputeUnit::kUnitMTE2;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y->attr.api.type = af::ApiType::kAPITypeBuffer;
  y->attr.api.unit = af::ComputeUnit::kUnitNone;

  // Scheduler
  auto axes = graph.GetAllAxis();
  auto z0_id = axes[0]->id;
  auto z1_id = axes[1]->id;

  auto [z1T, z1t] = graph.TileSplit(z1.id);
  auto [z1TB, z1Tb] = graph.BlockSplit(z1T->id);

  for (auto node : graph.GetAllNodes()) {
    if (IsOps<Data>(node) || IsOps<Output>(node)) {
      continue;
    }
    graph.ApplySplit(node, z1T->id, z1t->id);
    graph.ApplySplit(node, z1TB->id, z1Tb->id);

    graph.ApplyReorder(node, {z1TB->id, z1Tb->id, z1t->id});
  }

  for (auto axis : graph.GetAllAxis()) {
    tiler.AddAxis(*axis);
  }

  auto size = af::GetSizeByDataType(T);
  vector<af::Expression> vectorized_strides{One};

  // Vectorized/Loop axis
  gather->attr.sched.loop_axis = z1Tb->id;
  gather->outputs[0].attr.vectorized_axis = {z1Tb->id};
  gather->outputs[0].attr.vectorized_strides = vectorized_strides;

  store->attr.sched.loop_axis = z1Tb->id;
  store->outputs[0].attr.vectorized_axis = {z1Tb->id};
  store->outputs[0].attr.vectorized_strides = vectorized_strides;

  // Que/Buf alloc
  x1->outputs[0].attr.mem.tensor_id = 0;
  x1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x1->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  x1->outputs[0].attr.mem.position = af::Position::kPositionGM;
  x1->outputs[0].attr.buf.id = af::kIdNone;
  x1->outputs[0].attr.que.id = af::kIdNone;
  x1->outputs[0].attr.mem.reuse_id = af::kIdNone;
  x1->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  x1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  x2->outputs[0].attr.mem.tensor_id = 1;
  x2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x2->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  x2->outputs[0].attr.mem.position = af::Position::kPositionGM;
  x2->outputs[0].attr.buf.id = af::kIdNone;
  x2->outputs[0].attr.que.id = af::kIdNone;
  x2->outputs[0].attr.mem.reuse_id = af::kIdNone;
  x2->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  x2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  gather->outputs[0].attr.mem.tensor_id = 2;
  gather->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  gather->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  gather->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  gather->outputs[0].attr.buf.id = af::kIdNone;
  gather->outputs[0].attr.que.id = 0;
  gather->outputs[0].attr.mem.reuse_id = 0;
  gather->outputs[0].attr.que.depth = 2;
  gather->outputs[0].attr.que.buf_num = 2;
  gather->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  gather->outputs[0].attr.opt.merge_scope = af::kIdNone;

  store->outputs[0].attr.mem.tensor_id = 4;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = af::Position::kPositionGM;
  store->outputs[0].attr.buf.id = af::kIdNone;
  store->outputs[0].attr.que.id = af::kIdNone;
  store->outputs[0].attr.mem.reuse_id = af::kIdNone;
  store->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  std::vector<af::AxisId> current_axis = {z1TB->id, z1Tb->id};
  return current_axis;
}

TEST(CodegenKernel, LoadGatherRegApiCall_ShouldReturnSuccess_WhenParamHasOnlyOneAxis) {
  std::string binaryname = "GatherExtend";

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test");

  auto current_axis = CreateGraphAttrAxisIsLastAxisAndParamHasOnlyOneAxis<af::DT_FLOAT>(graph, tiler, 0);
  auto node = graph.FindNode("gather");
  node->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  tpipe.AddTensor(node->inputs[0]);
  tpipe.AddTensor(node->inputs[1]);
  tpipe.AddTensor(node->outputs[0]);
  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = node->inputs[0].attr.mem.tensor_id;
  x2.id = node->inputs[1].attr.mem.tensor_id;

  codegen::GatherRegApiCall call("GatherExtend");
  EXPECT_EQ(call.Init(node), 0);

  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  auto status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(status, af::SUCCESS);
}

TEST(CodegenKernel, LoadGatherRegApiCall_WhenAttrAxisIsNotLastAxisAndOneVecAxis) {
  std::string binaryname = "GatherExtend";

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test");

  auto current_axis = CreateGraphAttrAxisIsNotLastAxisAndOneVecAxis<af::DT_FLOAT>(graph, tiler, 2);
  auto node = graph.FindNode("gather");
  node->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  tpipe.AddTensor(node->inputs[0]);
  tpipe.AddTensor(node->inputs[1]);
  tpipe.AddTensor(node->outputs[0]);
  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = node->inputs[0].attr.mem.tensor_id;
  x2.id = node->inputs[1].attr.mem.tensor_id;

  codegen::GatherRegApiCall call("GatherExtend");
  EXPECT_EQ(call.Init(node), 0);

  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  auto status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(status, af::SUCCESS);
}

TEST(CodegenKernel, LoadGatherRegApiCall_WhenAttrAxisIsNotLastAxisAndTwoVecAxis) {
  std::string binaryname = "GatherExtend";

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test");

  auto current_axis = CreateGraphAttrAxisIsNotLastAxisAndTwoVecAxis<af::DT_FLOAT>(graph, tiler, 2);
  auto node = graph.FindNode("gather");
  node->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  tpipe.AddTensor(node->inputs[0]);
  tpipe.AddTensor(node->inputs[1]);
  tpipe.AddTensor(node->outputs[0]);
  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = node->inputs[0].attr.mem.tensor_id;
  x2.id = node->inputs[1].attr.mem.tensor_id;

  codegen::GatherRegApiCall call("GatherExtend");
  EXPECT_EQ(call.Init(node), 0);

  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  auto status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(status, af::SUCCESS);
}

TEST(CodegenKernel, LoadGatherRegApiCall_WhenAxisIsBiggerThanParamSize) {
  std::string binaryname = "GatherExtend";

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test");

  auto current_axis = CreateGraphAttrAxisIsNotLastAxisAndTwoVecAxis<af::DT_FLOAT>(graph, tiler, 6);
  auto node = graph.FindNode("gather");
  node->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  tpipe.AddTensor(node->inputs[0]);
  tpipe.AddTensor(node->inputs[1]);
  tpipe.AddTensor(node->outputs[0]);
  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = node->inputs[0].attr.mem.tensor_id;
  x2.id = node->inputs[1].attr.mem.tensor_id;

  codegen::GatherRegApiCall call("GatherExtend");
  EXPECT_EQ(call.Init(node), 0);

  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  auto status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(status, af::FAILED);
}

TEST(CodegenKernel, LoadGatherRegApiCall_WhenAixsIsLastAxisAndParamHasMoreThanOneAxis) {
  std::string binaryname = "GatherExtend";

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test");

  auto current_axis = CreateGraphAttrAxisIsLastAxisAndParamHasMoreThanOneAxis<af::DT_FLOAT>(graph, tiler, 4);
  auto node = graph.FindNode("gather");
  node->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  tpipe.AddTensor(node->inputs[0]);
  tpipe.AddTensor(node->inputs[1]);
  tpipe.AddTensor(node->outputs[0]);
  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = node->inputs[0].attr.mem.tensor_id;
  x2.id = node->inputs[1].attr.mem.tensor_id;

  codegen::GatherRegApiCall call("GatherExtend");
  EXPECT_EQ(call.Init(node), 0);

  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  auto status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(status, af::SUCCESS);
}

TEST(CodegenKernel, LoadGatherRegApiCall_WhenGatherComputeTypeIsGather) {
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  af::AscGraph graph("test");
  auto current_axis = CreateGraphGatherWithComputeType_Mid_Axis(graph, tiler);
  auto node = graph.FindNode("gather");
  node->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  tpipe.AddTensor(node->inputs[0]);
  tpipe.AddTensor(node->inputs[1]);
  tpipe.AddTensor(node->outputs[0]);
  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = node->inputs[0].attr.mem.tensor_id;
  x2.id = node->inputs[1].attr.mem.tensor_id;
  codegen::GatherRegApiCall call("GatherExtend");
  EXPECT_EQ(call.Init(node), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  std::string result;
  auto status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(status, af::SUCCESS);
}

TEST(CodegenKernel, LoadGatherRegApiCall_WhenGatherComputeTypeIsLoad_Mid_Axis) {
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  af::AscGraph graph("test");
  auto current_axis = CreateGraphGatherWithComputeType_Mid_Axis(graph, tiler, af::ComputeType::kComputeLoad);
  auto node = graph.FindNode("gather");
  node->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  tpipe.AddTensor(node->inputs[0]);
  tpipe.AddTensor(node->inputs[1]);
  tpipe.AddTensor(node->outputs[0]);
  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = node->inputs[0].attr.mem.tensor_id;
  x2.id = node->inputs[1].attr.mem.tensor_id;

  codegen::GatherRegApiCall call("GatherExtend");
  EXPECT_EQ(call.Init(node), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  auto status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(status, af::SUCCESS);
}

TEST(CodegenKernel, LoadGatherRegApiCall_WhenGatherComputeTypeIsLoad_First_Axis) {
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  af::AscGraph graph("test");
  auto current_axis = CreateGraphGatherWithComputeType_First_Axis(graph, tiler, af::ComputeType::kComputeLoad);
  auto node = graph.FindNode("gather");
  node->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  tpipe.AddTensor(node->inputs[0]);
  tpipe.AddTensor(node->inputs[1]);
  tpipe.AddTensor(node->outputs[0]);
  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = node->inputs[0].attr.mem.tensor_id;
  x2.id = node->inputs[1].attr.mem.tensor_id;

  codegen::GatherRegApiCall call("GatherExtend");
  EXPECT_EQ(call.Init(node), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  auto status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(status, af::SUCCESS);
}

TEST(CodegenKernel, LoadGatherRegApiCall_WhenGatherComputeTypeIsLoad_Single_Axis) {
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  af::AscGraph graph("test");
  auto current_axis = CreateGraphGatherWithComputeType_Single_Axis(graph, tiler, af::ComputeType::kComputeLoad);
  auto node = graph.FindNode("gather");
  node->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  tpipe.AddTensor(node->inputs[0]);
  tpipe.AddTensor(node->inputs[1]);
  tpipe.AddTensor(node->outputs[0]);
  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = node->inputs[0].attr.mem.tensor_id;
  x2.id = node->inputs[1].attr.mem.tensor_id;

  codegen::GatherRegApiCall call("GatherExtend");
  EXPECT_EQ(call.Init(node), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  auto status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(status, af::SUCCESS);
}

TEST(CodegenKernel, LoadGatherRegApiCall_WhenGatherComputeTypeIsLoad_Tail_Axis) {
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  af::AscGraph graph("test");
  auto current_axis = CreateGraphGatherWithComputeType_Tail_Axis(graph, tiler, af::ComputeType::kComputeLoad);
  auto node = graph.FindNode("gather");
  node->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  tpipe.AddTensor(node->inputs[0]);
  tpipe.AddTensor(node->inputs[1]);
  tpipe.AddTensor(node->outputs[0]);
  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = node->inputs[0].attr.mem.tensor_id;
  x2.id = node->inputs[1].attr.mem.tensor_id;

  codegen::GatherRegApiCall call("GatherExtend");
  EXPECT_EQ(call.Init(node), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  auto status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(status, af::SUCCESS);
}

TEST(CodegenKernel, LoadGatherRegApiCall_WhenGatherComputeTypeIsLoad_Failed) {
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  af::AscGraph graph("test");
  auto current_axis = CreateGraphGatherWithComputeType_Failed(graph, tiler, af::ComputeType::kComputeLoad);
  auto node = graph.FindNode("gather");
  tpipe.AddTensor(node->inputs[0]);
  tpipe.AddTensor(node->inputs[1]);
  tpipe.AddTensor(node->outputs[0]);
  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = node->inputs[0].attr.mem.tensor_id;
  x2.id = node->inputs[1].attr.mem.tensor_id;

  codegen::GatherRegApiCall call("GatherExtend");
  EXPECT_EQ(call.Init(node), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  auto status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(status, af::FAILED);
}

TEST(CodegenKernel, LoadGatherRegApiCall_WhenGatherComputeTypeIsInValid_Faild) {
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  af::AscGraph graph("test");
  auto current_axis = CreateGraphGatherWithComputeType_Failed(graph, tiler, af::ComputeType::kComputeInvalid);
  auto node = graph.FindNode("gather");
  tpipe.AddTensor(node->inputs[0]);
  tpipe.AddTensor(node->inputs[1]);
  tpipe.AddTensor(node->outputs[0]);
  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = node->inputs[0].attr.mem.tensor_id;
  x2.id = node->inputs[1].attr.mem.tensor_id;

  codegen::GatherRegApiCall call("GatherExtend");
  EXPECT_EQ(call.Init(node), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  auto status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(status, af::FAILED);
}

TEST(CodegenKernel, LoadGatherRegApiCall_WhenValueDtypeIsInt64) {
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  af::AscGraph graph("test");
  auto current_axis = CreateGraphAttrAxisIsLastAxisAndParamHasOnlyOneAxis<af::DT_INT64>(graph, tiler, 0);
  auto node = graph.FindNode("gather");
  node->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  node->attr.api.compute_type = af::ComputeType::kComputeLoad;
  tpipe.AddTensor(node->inputs[0]);
  tpipe.AddTensor(node->inputs[1]);
  tpipe.AddTensor(node->outputs[0]);
  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = node->inputs[0].attr.mem.tensor_id;
  x2.id = node->inputs[1].attr.mem.tensor_id;

  codegen::GatherRegApiCall call("GatherExtend");
  EXPECT_EQ(call.Init(node), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  auto status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(status, af::SUCCESS);
  EXPECT_NE(result.find("GatherExtend<int64_t, int64_t"), std::string::npos);
  const auto api_param = CodegenApiParam::GetNodeApiParam(node);
  ASSERT_NE(api_param, nullptr);
  EXPECT_EQ(api_param->semantic_feature, "Adaptive");
}
