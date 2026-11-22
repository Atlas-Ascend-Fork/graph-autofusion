/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "e2e_rargmax.h"

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "common_utils.h"

using namespace ge;
using namespace af;
using namespace af::ops;
using namespace af::ascir_op;

void LoadRargmaxStore_BeforeAutofuse(af::AscGraph &graph, ge::DataType data_type) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x("x");
  graph.AddNode(x);
  x.y.dtype = data_type;

  Load load("load");
  graph.AddNode(load);
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id};
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, s1};
  *load.y.strides = {s1, One};

  af::ascir_op::Argmax rargmax("rargmax");
  graph.AddNode(rargmax);
  rargmax.x = load.y;
  rargmax.attr.sched.axis = {z0.id, z1.id};
  *rargmax.y.axis = {z0.id, z1.id};
  *rargmax.y.repeats = {s0, One};
  *rargmax.y.strides = {One, Zero};
  // Argmax requires tmp buffer for indices
  rargmax.attr.tmp_buffers = {{{af::Symbol(8192), -1}, MemAttr(), 0}, {{af::Symbol(8192), 0}, MemAttr(), 1}};

  Store store("store");
  graph.AddNode(store);
  store.x = rargmax.y;
  store.attr.sched.axis = {z0.id, z1.id};
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, One};
  *store.y.strides = {One, Zero};

  Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  // Argmax outputs indices (int64)
  y.y.dtype = ge::DT_INT64;
}

void LoadRargmaxStore_AfterAutofuse(af::AscGraph &graph, ge::DataType data_type) {
  auto x = graph.FindNode("x");
  x->attr.api.compute_type = ComputeType::kComputeInvalid;
  x->attr.api.type = ApiType::kAPITypeBuffer;
  x->attr.api.unit = ComputeUnit::kUnitNone;

  auto load = graph.FindNode("load");
  load->outputs[0].attr.dtype = data_type;

  load->attr.api.compute_type = ComputeType::kComputeLoad;
  load->attr.api.type = ApiType::kAPITypeCompute;
  load->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto rargmax = graph.FindNode("rargmax");
  rargmax->attr.api.compute_type = ComputeType::kComputeReduce;
  // Argmax outputs int64 indices
  rargmax->outputs[0].attr.dtype = ge::DT_INT64;

  rargmax->attr.api.type = ApiType::kAPITypeCompute;
  rargmax->attr.api.unit = ComputeUnit::kUnitVector;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = ComputeType::kComputeStore;
  // Store outputs int64 indices
  store->outputs[0].attr.dtype = ge::DT_INT64;

  store->attr.api.type = ApiType::kAPITypeCompute;
  store->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = ComputeType::kComputeInvalid;
  y->attr.api.type = ApiType::kAPITypeBuffer;
  y->attr.api.unit = ComputeUnit::kUnitNone;

  // Scheduler
  auto z0 = load->attr.sched.axis[0];
  auto z1 = load->attr.sched.axis[1];
  vector<af::Expression> vectorized_strides{One, One};

  auto [z0T, z0t] = graph.TileSplit(z0);
  auto [z0TB, z0Tb] = graph.BlockSplit(z0T->id);
  auto [z1T, z1t] = graph.TileSplit(z1);

  for (auto node : graph.GetAllNodes()) {
    if (IsOps<Data>(node) || IsOps<Output>(node)) {
      continue;
    }

    graph.ApplySplit(node, z0T->id, z0t->id);
    graph.ApplySplit(node, z0TB->id, z0Tb->id);
    graph.ApplySplit(node, z1T->id, z1t->id);
    graph.ApplyReorder(node, {z0TB->id, z0Tb->id, z1T->id, z0t->id, z1t->id});
  }

  // Vectorized/Loop axis
  load->attr.sched.loop_axis = z1T->id;
  load->outputs[0].attr.vectorized_axis = {z0t->id, z1t->id};
  auto size = ge::GetSizeByDataType(data_type);
  vectorized_strides[0] = af::sym::Align(graph.FindAxis(z1t->id)->size, 32 / size);
  load->outputs[0].attr.vectorized_strides = vectorized_strides;

  rargmax->attr.sched.loop_axis = z1T->id;
  rargmax->outputs[0].attr.vectorized_axis = {z0t->id, z1t->id};
  rargmax->outputs[0].attr.vectorized_strides = {One, Zero};

  store->attr.sched.loop_axis = z0Tb->id;
  store->outputs[0].attr.vectorized_axis = {z0t->id, z1t->id};
  store->outputs[0].attr.vectorized_strides = {One, Zero};

  // Que/Buf alloc
  x->outputs[0].attr.mem.tensor_id = 0;
  x->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x->outputs[0].attr.mem.position = Position::kPositionGM;
  x->outputs[0].attr.buf.id = af::kIdNone;
  x->outputs[0].attr.que.id = af::kIdNone;
  x->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  x->outputs[0].attr.opt.merge_scope = af::kIdNone;

  load->outputs[0].attr.mem.tensor_id = 1;
  load->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  load->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  load->outputs[0].attr.mem.position = Position::kPositionVecIn;
  load->outputs[0].attr.buf.id = af::kIdNone;
  load->outputs[0].attr.que.id = 0;
  load->outputs[0].attr.mem.reuse_id = 0;
  load->outputs[0].attr.que.depth = 2;
  load->outputs[0].attr.que.buf_num = 2;
  load->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  rargmax->outputs[0].attr.mem.tensor_id = 2;
  rargmax->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  rargmax->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  rargmax->outputs[0].attr.mem.position = Position::kPositionVecOut;
  rargmax->outputs[0].attr.buf.id = af::kIdNone;
  rargmax->outputs[0].attr.que.id = 1;
  rargmax->outputs[0].attr.mem.reuse_id = 1;
  rargmax->outputs[0].attr.que.depth = 2;
  rargmax->outputs[0].attr.que.buf_num = 2;
  rargmax->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  rargmax->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto impl = ascgen_utils::GetAscIrCodegenImpl(rargmax->GetType());
  std::vector<std::unique_ptr<af::TmpBufDesc>> buffers = impl->CalcTmpBufSize(*rargmax);
  for (auto &buf_desc : buffers) {
    if (buf_desc != nullptr) {
      af::TmpBuffer temp_buffer;
      temp_buffer.buf_desc = std::move(*buf_desc);
      rargmax->attr.tmp_buffers.emplace_back(temp_buffer);
    }
  }

  store->outputs[0].attr.mem.tensor_id = 3;
  store->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = Position::kPositionGM;
  store->outputs[0].attr.buf.id = af::kIdNone;
  store->outputs[0].attr.que.id = af::kIdNone;
  store->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;
}
