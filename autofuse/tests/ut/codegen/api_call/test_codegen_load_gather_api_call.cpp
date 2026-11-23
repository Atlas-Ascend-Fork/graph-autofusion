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
#include "gather/gather_api_call.h"

using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;
using namespace codegen;

template <ge::DataType T>
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
  x2.y.dtype = ge::DT_INT64;
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

template <ge::DataType T>
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
  x2.y.dtype = ge::DT_INT64;
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

template <ge::DataType T>
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

  auto size = ge::GetSizeByDataType(T);
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

template <ge::DataType T>
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

  auto size = ge::GetSizeByDataType(T);
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

template <ge::DataType T>
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

  auto size = ge::GetSizeByDataType(T);
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

template <ge::DataType T>
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

  auto size = ge::GetSizeByDataType(T);
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

TEST(CodegenKernel, LoadGatherApiCall_ShouldReturnSuccess_WhenInputsAreNormal) {
  std::string binaryname = "GatherExtend";
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  af::AscGraph graph("test");
  CreateGraph<ge::DT_FLOAT>(graph);
  auto node = graph.FindNode("gather");
  node->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  tpipe.AddTensor(node->inputs[0]);
  tpipe.AddTensor(node->inputs[1]);
  tpipe.AddTensor(node->outputs[0]);
  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = node->inputs[0].attr.mem.tensor_id;
  x2.id = node->inputs[1].attr.mem.tensor_id;

  codegen::GatherApiCall call("GatherExtend");
  EXPECT_EQ(call.Init(node), 0);

  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result,
            std::string{"GatherExtend(global_-1, global_-1, global_-1[0], 4000, global_-1_actual_size, tmp_buf_0);\n"});
}

TEST(CodegenKernel, LoadGatherApiCall_ShouldReturnSuccess_WhenParamHasOnlyOneAxis) {
  std::string binaryname = "GatherExtend";

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test");

  auto current_axis = CreateGraphAttrAxisIsLastAxisAndParamHasOnlyOneAxis<ge::DT_FLOAT>(graph, tiler, 0);
  auto node = graph.FindNode("gather");
  node->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  tpipe.AddTensor(node->inputs[0]);
  tpipe.AddTensor(node->inputs[1]);
  tpipe.AddTensor(node->outputs[0]);
  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = node->inputs[0].attr.mem.tensor_id;
  x2.id = node->inputs[1].attr.mem.tensor_id;

  codegen::GatherApiCall call("GatherExtend");
  EXPECT_EQ(call.Init(node), 0);

  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  auto status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(status, af::SUCCESS);
  EXPECT_EQ(result,
            std::string{"GatherExtend(local_2, global_0, global_1[(int64_t)z1TB * (int64_t)(z1Tb_size * z1t_size) + "
                        "(int64_t)z1Tb * (int64_t)z1t_size], (s0)/(1), local_2_actual_size, tmp_buf_0);\n"});
}

TEST(CodegenKernel, LoadGatherApiCall_WhenAttrAxisIsNotLastAxisAndOneVecAxis) {
  std::string binaryname = "GatherExtend";

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test");

  auto current_axis = CreateGraphAttrAxisIsNotLastAxisAndOneVecAxis<ge::DT_FLOAT>(graph, tiler, 2);
  auto node = graph.FindNode("gather");
  tpipe.AddTensor(node->inputs[0]);
  tpipe.AddTensor(node->inputs[1]);
  tpipe.AddTensor(node->outputs[0]);
  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = node->inputs[0].attr.mem.tensor_id;
  x2.id = node->inputs[1].attr.mem.tensor_id;

  codegen::GatherApiCall call("GatherExtend");
  EXPECT_EQ(call.Init(node), 0);

  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  auto status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(status, af::SUCCESS);
}

TEST(CodegenKernel, LoadGatherApiCall_WhenAttrAxisIsNotLastAxisAndTwoVecAxis) {
  std::string binaryname = "GatherExtend";

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test");

  auto current_axis = CreateGraphAttrAxisIsNotLastAxisAndTwoVecAxis<ge::DT_FLOAT>(graph, tiler, 2);
  auto node = graph.FindNode("gather");
  tpipe.AddTensor(node->inputs[0]);
  tpipe.AddTensor(node->inputs[1]);
  tpipe.AddTensor(node->outputs[0]);
  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = node->inputs[0].attr.mem.tensor_id;
  x2.id = node->inputs[1].attr.mem.tensor_id;

  codegen::GatherApiCall call("GatherExtend");
  EXPECT_EQ(call.Init(node), 0);

  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  auto status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(status, af::SUCCESS);
}

TEST(CodegenKernel, LoadGatherApiCall_WhenAxisIsBiggerThanParamSize) {
  std::string binaryname = "GatherExtend";

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test");

  auto current_axis = CreateGraphAttrAxisIsNotLastAxisAndTwoVecAxis<ge::DT_FLOAT>(graph, tiler, 6);
  auto node = graph.FindNode("gather");
  tpipe.AddTensor(node->inputs[0]);
  tpipe.AddTensor(node->inputs[1]);
  tpipe.AddTensor(node->outputs[0]);
  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = node->inputs[0].attr.mem.tensor_id;
  x2.id = node->inputs[1].attr.mem.tensor_id;

  codegen::GatherApiCall call("GatherExtend");
  EXPECT_EQ(call.Init(node), 0);

  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  auto status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(status, af::FAILED);
}

TEST(CodegenKernel, LoadGatherApiCall_WhenAixsIsLastAxisAndParamHasMoreThanOneAxis) {
  std::string binaryname = "GatherExtend";

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test");

  auto current_axis = CreateGraphAttrAxisIsLastAxisAndParamHasMoreThanOneAxis<ge::DT_FLOAT>(graph, tiler, 4);
  auto node = graph.FindNode("gather");
  node->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  tpipe.AddTensor(node->inputs[0]);
  tpipe.AddTensor(node->inputs[1]);
  tpipe.AddTensor(node->outputs[0]);
  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = node->inputs[0].attr.mem.tensor_id;
  x2.id = node->inputs[1].attr.mem.tensor_id;

  codegen::GatherApiCall call("GatherExtend");
  EXPECT_EQ(call.Init(node), 0);

  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  auto status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(status, af::SUCCESS);
  EXPECT_EQ(result, std::string{"GatherExtend(local_2, global_0[z0z1z2z3 * (s4)/(1)], global_1[(int64_t)z5z6T * "
                                "(int64_t)z5z6t_size], (s4)/(1), local_2_actual_size, tmp_buf_0);\n"});
}

TEST(CodegenKernel, LoadGatherApiCall_WhenLastAxisOutputIsPadded) {
  std::string binaryname = "GatherExtend";
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  af::AscGraph graph("test");
  auto current_axis = CreateGraphAttrAxisIsLastAxisAndParamHasMoreThanOneAxis<ge::DT_FLOAT>(graph, tiler, 4);
  auto node = graph.FindNode("gather");
  node->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  af::AxisId block_inner_id = af::kIdNone;
  af::AxisId tile_inner_id = af::kIdNone;
  for (const auto &axis : graph.GetAllAxis()) {
    if (axis->name == "z0z1z2z3b") {
      block_inner_id = axis->id;
    }
    if (axis->name == "z5z6t") {
      tile_inner_id = axis->id;
    }
  }
  ASSERT_NE(block_inner_id, af::kIdNone);
  ASSERT_NE(tile_inner_id, af::kIdNone);
  const auto size = ge::GetSizeByDataType(ge::DT_FLOAT);
  node->outputs[0].attr.vectorized_axis = {block_inner_id, tile_inner_id};
  node->outputs[0].attr.vectorized_strides = {af::sym::Align(graph.FindAxis(tile_inner_id)->size, 32 / size), One};
  std::vector<af::AxisId> padded_axis;
  for (const auto axis_id : current_axis) {
    if (axis_id != block_inner_id) {
      padded_axis.push_back(axis_id);
    }
  }
  tpipe.AddTensor(node->inputs[0]);
  tpipe.AddTensor(node->inputs[1]);
  tpipe.AddTensor(node->outputs[0]);
  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = node->inputs[0].attr.mem.tensor_id;
  x2.id = node->inputs[1].attr.mem.tensor_id;
  codegen::GatherApiCall call("GatherExtend");
  EXPECT_EQ(call.Init(node), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  std::string result;
  auto status = call.Generate(tpipe, padded_axis, result);
  EXPECT_EQ(status, af::SUCCESS);
  EXPECT_EQ(result, std::string{"GatherExtend(local_2, global_0[0 * (s4)/(1)], global_1[(int64_t)z5z6T * "
                                "(int64_t)z5z6t_size], (s4)/(1), (z0z1z2z3b_size * z5z6t_size), tmp_buf_0, "
                                "z5z6t_size, ((8 * Ceiling((Rational(1 , 8) * z5z6t_size))))/(1));\n"});
}

// 与上一用例同布局，但 current_axis 保留 block_inner（真实 split 场景）：
// first_merge_axis 取块内轴未切分前缀，param 偏移携带 z0z1z2z3，padded 参数不变。
TEST(CodegenKernel, LoadGatherApiCall_WhenLastAxisOutputIsPaddedWithBlockInnerAxis) {
  std::string binaryname = "GatherExtend";
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  af::AscGraph graph("test");
  auto current_axis = CreateGraphAttrAxisIsLastAxisAndParamHasMoreThanOneAxis<ge::DT_FLOAT>(graph, tiler, 4);
  auto node = graph.FindNode("gather");
  node->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  af::AxisId block_inner_id = af::kIdNone;
  af::AxisId tile_inner_id = af::kIdNone;
  for (const auto &axis : graph.GetAllAxis()) {
    if (axis->name == "z0z1z2z3b") {
      block_inner_id = axis->id;
    }
    if (axis->name == "z5z6t") {
      tile_inner_id = axis->id;
    }
  }
  ASSERT_NE(block_inner_id, af::kIdNone);
  ASSERT_NE(tile_inner_id, af::kIdNone);
  const auto size = ge::GetSizeByDataType(ge::DT_FLOAT);
  node->outputs[0].attr.vectorized_axis = {block_inner_id, tile_inner_id};
  node->outputs[0].attr.vectorized_strides = {af::sym::Align(graph.FindAxis(tile_inner_id)->size, 32 / size), One};
  tpipe.AddTensor(node->inputs[0]);
  tpipe.AddTensor(node->inputs[1]);
  tpipe.AddTensor(node->outputs[0]);
  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = node->inputs[0].attr.mem.tensor_id;
  x2.id = node->inputs[1].attr.mem.tensor_id;
  codegen::GatherApiCall call("GatherExtend");
  EXPECT_EQ(call.Init(node), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  std::string result;
  auto status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(status, af::SUCCESS);
  EXPECT_EQ(result, std::string{"GatherExtend(local_2, global_0[z0z1z2z3 * (s4)/(1)], global_1[(int64_t)z5z6T * "
                                "(int64_t)z5z6t_size], (s4)/(1), (z0z1z2z3b_size * z5z6t_size), tmp_buf_0, "
                                "z5z6t_size, ((8 * Ceiling((Rational(1 , 8) * z5z6t_size))))/(1));\n"});
}
